"""Soft Actor-Critic, and the replay buffer the preference queries are drawn from.

Milestones 1 to 6 asked whether the reward model ranks held-out preference pairs
correctly. The paper never reports that number. What it reports is policy success
rate, so this is the file that finally makes our numbers comparable to theirs.

Deliberately plain: a Gaussian actor with a tanh squash, twin Q critics with
target networks, and a learned temperature. Nothing here is novel and nothing
here should be clever, because when a Milestone 7 run fails we need to be able to
tell "the reward model is wrong" apart from "the RL is wrong". The ground-truth
reward arm exists for exactly that reason: if SAC on the true reward does not
solve Window Close, the problem is in this file, not in the preference model.

Hyperparameters are the paper's Table 3: lr 3e-4, discount 0.99, init temp 0.1,
EMA tau 0.995, batch 512, 3x256 actor and critic.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import ACT_DIM, OBS_DIM

# Paper Table 3, "Artificial Feedback" column.
LR = 3e-4
DISCOUNT = 0.99
INIT_TEMP = 0.1
EMA_TAU = 0.995
BATCH_SIZE = 512
HIDDEN = (256, 256, 256)
TARGET_UPDATE_FREQ = 2
BETAS = (0.9, 0.999)

LOG_STD_MIN, LOG_STD_MAX = -10.0, 2.0


def mlp(in_dim: int, out_dim: int, hidden=HIDDEN) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            pre = mu
        else:
            pre = mu + std * torch.randn_like(mu)
        action = torch.tanh(pre)
        if not with_logprob:
            return action, None
        # Change of variables for the tanh squash. The log(1 - tanh^2) form is
        # numerically poor at saturation; use the standard stable rewrite.
        logp = (-0.5 * ((pre - mu) / std) ** 2 - log_std
                - 0.5 * np.log(2 * np.pi)).sum(dim=-1)
        logp = logp - (2 * (np.log(2) - pre - F.softplus(-2 * pre))).sum(dim=-1)
        return action, logp


class Critic(nn.Module):
    """Twin Q networks; the min of the two is what the actor is scored against."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=HIDDEN):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    """Flat transition store. Also the pool preference queries are sampled from.

    That dual role is the point: the paper's queries come from the policy's own
    replay buffer, so early queries compare two mediocre behaviors rather than an
    expert against a random flail. Our offline datasets could not reproduce that,
    and it is the regime where a pretrained prior should matter most.
    """

    def __init__(self, capacity: int, obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.true_reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        # Episode boundaries, so a query segment never straddles a reset.
        self.episode_id = np.zeros(capacity, dtype=np.int64)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, next_obs, true_reward, done, episode_id):
        i = self.idx
        self.obs[i] = obs
        self.action[i] = action
        self.next_obs[i] = next_obs
        self.true_reward[i] = true_reward
        self.done[i] = done
        self.episode_id[i] = episode_id
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int, rng: np.random.RandomState):
        n = len(self)
        i = rng.randint(0, n, size=batch_size)
        t = lambda a, d=torch.float32: torch.as_tensor(a[i], dtype=d, device=self.device)  # noqa: E731
        return t(self.obs), t(self.action), t(self.next_obs), t(self.true_reward), t(self.done)

    def valid_segment_starts(self, segment_size: int) -> np.ndarray:
        """Starts whose whole segment lies inside one episode and inside the buffer.

        Mirrors `prefs.valid_segment_starts` but over live buffer indices. When
        the buffer has wrapped, indices near the write head interleave old and
        new data, and the episode-id check rejects those automatically.
        """
        n = len(self)
        if n < segment_size:
            return np.empty(0, dtype=np.int64)
        starts = np.arange(0, n - segment_size + 1, dtype=np.int64)
        ep = self.episode_id[:n]
        same = ep[starts] == ep[starts + segment_size - 1]
        return starts[same]

    def segments(self, starts: np.ndarray, segment_size: int):
        """(N, T, obs_dim) and (N, T, act_dim) tensors for the given starts."""
        idx = starts[:, None] + np.arange(segment_size)[None, :]
        obs = torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(self.action[idx], dtype=torch.float32, device=self.device)
        return obs, act

    def segment_true_return(self, starts: np.ndarray, segment_size: int) -> np.ndarray:
        """Ground-truth return per segment. The oracle's answer, never a target."""
        idx = starts[:, None] + np.arange(segment_size)[None, :]
        return self.true_reward[idx].sum(axis=1)


class SAC:
    """Soft Actor-Critic with a learned temperature."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, device="cpu", lr=LR,
                 discount=DISCOUNT, init_temp=INIT_TEMP, tau=EMA_TAU,
                 target_update_freq=TARGET_UPDATE_FREQ, hidden=HIDDEN, seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.discount = discount
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.act_dim = act_dim

        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(np.log(init_temp), dtype=torch.float32,
                                      device=device, requires_grad=True)
        # The usual heuristic: one nat of entropy per action dimension.
        self.target_entropy = -float(act_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=BETAS)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=BETAS)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, betas=BETAS)
        self.step_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch, reward: torch.Tensor):
        """One actor-critic update. `reward` is passed in, not read from the batch.

        That signature is the whole integration point with the reward model: the
        caller decides whether these transitions are labeled by the ground truth
        or relabeled by the learned reward, and nothing in SAC knows which.
        """
        obs, action, next_obs, _, done = batch

        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            target_v = torch.min(tq1, tq2) - self.alpha.detach() * next_logp
            target_q = reward + (1.0 - done) * self.discount * target_v

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        new_action, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        actor_loss = (self.alpha.detach() * logp - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = (self.alpha * (-logp - self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(self.tau).add_(p.data, alpha=1.0 - self.tau)

        return dict(critic_loss=float(critic_loss.item()),
                    actor_loss=float(actor_loss.item()),
                    alpha=float(self.alpha.item()))
