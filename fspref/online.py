"""The online feedback loop: ask, label, re-adapt.

This is the half of the paper that Milestones 1 to 6 could not test. Offline, we
handed the reward model a fixed pile of preferences drawn from scripted experts
and asked how well it ranked held-out pairs. Online, the queries come from the
policy's own replay buffer, so early on they compare two mediocre behaviors, and
the reward model's job is to be useful *now*, to a policy that is still bad.

Three details from the paper that are easy to get wrong, all of them load-bearing:

1.  **The budget is a schedule, not a number.** Window Close is 200 total
    feedback given 8 at a time every 5000 steps (Table 4). The model therefore
    adapts on 8 pairs, then 16, then 24, and so on. The interesting regime is the
    first few sessions, which is exactly where our low-budget sweep found the
    meta-init ahead and where Milestone 6's 25-label floor never looked.

2.  **Reset before re-adapting.** "We crucially reset the reward model for
    adaptation" (Section 3). Every session the weights go back to the
    meta-initialization and adapt on *all* feedback collected so far, rather than
    continuing to fine-tune. Continuing to fine-tune is the Init baseline, and
    the difference between the two is the paper's central algorithmic claim.

3.  **Disagreement sampling.** After the first session, candidate pairs are
    oversampled by 10x and the ones the ensemble most disagrees about are kept
    (Table 3). The first session has no trained ensemble to disagree, so it is
    uniform.

The "human" here is `oracle_label`, which compares ground-truth segment returns.
The numeric rewards are never a regression target; they only decide which of two
segments wins, which is the same information a person clicking a button gives.
"""
from __future__ import annotations

import numpy as np
import torch

from .adapt import TARGET_SUPPORT_ACC, train_on_support
from .prefs import SEGMENT_SIZE
from .reward_model import RewardEnsemble, disagreement

# Paper Table 3 / Table 4.
DISAGREEMENT_MULTIPLIER = 10
REWARD_BATCH = 256


class FeedbackSchedule:
    """Constant schedule: `per_session` queries every `frequency` steps, up to `total`."""

    def __init__(self, total: int, per_session: int, frequency: int):
        self.total = total
        self.per_session = per_session
        self.frequency = frequency
        self.collected = 0

    def due(self, step: int) -> bool:
        return (step % self.frequency == 0 and step > 0
                and self.collected < self.total)

    def next_batch_size(self) -> int:
        return min(self.per_session, self.total - self.collected)


def oracle_label(buffer, a_starts, b_starts, segment_size=SEGMENT_SIZE,
                 tie_margin: float = 0.0):
    """The 'fake human': which segment has the higher ground-truth return.

    Returns (labels, keep_mask). Ties within `tie_margin` are dropped rather than
    answered arbitrarily -- the Milestone 4 lesson, and also what the paper's
    human users do when they press skip.
    """
    ra = buffer.segment_true_return(a_starts, segment_size)
    rb = buffer.segment_true_return(b_starts, segment_size)
    keep = np.abs(ra - rb) > tie_margin
    return (ra > rb).astype(np.float32), keep


def select_queries(buffer, net, n_queries: int, rng, segment_size=SEGMENT_SIZE,
                   use_disagreement: bool = True, device="cpu"):
    """Pick `n_queries` segment pairs from the replay buffer.

    Uniform when `use_disagreement` is False or no model is available yet;
    otherwise oversample by DISAGREEMENT_MULTIPLIER and keep the pairs whose
    ensemble members disagree most about who wins.
    """
    starts = buffer.valid_segment_starts(segment_size)
    if len(starts) < 2:
        return None, None
    n_candidates = n_queries * (DISAGREEMENT_MULTIPLIER if use_disagreement and net else 1)
    a = rng.choice(starts, size=n_candidates, replace=True)
    b = rng.choice(starts, size=n_candidates, replace=True)
    # A segment compared against itself carries no information.
    distinct = a != b
    a, b = a[distinct], b[distinct]
    if len(a) == 0:
        return None, None
    if not (use_disagreement and net is not None) or len(a) <= n_queries:
        return a[:n_queries], b[:n_queries]

    obs_a, act_a = buffer.segments(a, segment_size)
    obs_b, act_b = buffer.segments(b, segment_size)
    with torch.no_grad():
        logits = net(obs_a, act_a, obs_b, act_b)
        score = disagreement(logits).cpu().numpy()
    top = np.argsort(-score)[:n_queries]
    return a[top], b[top]


class PreferenceDataset:
    """Every query answered so far, as index pairs into the replay buffer.

    Storing indices rather than materialized segments matters: the reward model
    is re-adapted on the full history every session, so the same pairs are
    re-gathered many times, and the buffer already holds the observations.

    This is only valid while the replay buffer has NOT wrapped. Once it does, a
    stored index names a transition that has since been overwritten, and every
    earlier label silently re-attaches to an unrelated segment -- no exception,
    no visible symptom, just a quietly corrupted preference history. The caller
    is responsible for sizing the buffer to at least the step budget;
    `scripts/m7_sac.py` refuses to start otherwise.
    """

    def __init__(self, device="cpu", segment_size=SEGMENT_SIZE):
        self.a: list[int] = []
        self.b: list[int] = []
        self.y: list[float] = []
        self.device = device
        self.segment_size = segment_size

    def __len__(self):
        return len(self.y)

    def add(self, a_starts, b_starts, labels):
        self.a += list(map(int, a_starts))
        self.b += list(map(int, b_starts))
        self.y += list(map(float, labels))

    def batch(self, buffer, idx=None):
        """(obs_a, act_a, obs_b, act_b, y) for the given rows, or all of them."""
        a = np.asarray(self.a, dtype=np.int64)
        b = np.asarray(self.b, dtype=np.int64)
        y = np.asarray(self.y, dtype=np.float32)
        if idx is not None:
            a, b, y = a[idx], b[idx], y[idx]
        obs_a, act_a = buffer.segments(a, self.segment_size)
        obs_b, act_b = buffer.segments(b, self.segment_size)
        return (obs_a, act_a, obs_b, act_b,
                torch.as_tensor(y, dtype=torch.float32, device=self.device))


def readapt(mode: str, maml, prev_net, dataset, buffer, ensemble_size=3,
            device="cpu", seed=0, lr=3e-4, max_maml_steps=40,
            target_acc=TARGET_SUPPORT_ACC):
    """Rebuild the reward model on all feedback so far. Returns (net, info).

    Where the weights start is the only thing that separates these arms, and it
    is the whole experiment:

        few_shot    reset to the meta-init, adapt with learned inner rates,
                    fall back to Adam if 40 steps do not reach 95%
        init        pretrained weights ONCE, then keep fine-tuning them
        init_reset  reset to the meta-init every session, then plain Adam
        pebble      fresh random weights every session, plain Adam

    `init` is the paper's baseline and is deliberately denied the reset: "Instead
    of re-adapting the reward model each time new feedback is collected, we
    initialize the reward model with the pretrained weights, and then perform
    standard updates with the Adam optimizer as in PEBBLE" (Section 4.1). It
    exists to isolate the paper's central algorithmic claim -- that resetting and
    re-adapting each session is what makes the prior pay off, because the reward
    implied by the accumulated preferences drifts as the policy improves. So it
    continues from `prev_net` on every session after the first.

    An earlier version of this function reset `init` every session, which handed
    it exactly the mechanism it is supposed to lack and made it a strictly
    stronger baseline than the paper's. `init_reset` preserves that behaviour
    under an honest name, because it does answer a real question: with the reset
    held fixed for both arms, do the learned per-parameter inner rates buy
    anything over plain Adam?
    """
    support = dataset.batch(buffer)
    if mode == "pebble":
        torch.manual_seed(seed)
        net = RewardEnsemble(ensemble_size=ensemble_size).to(device)
        net, steps, acc = train_on_support(net, support, lr=lr, target_acc=target_acc)
        return net, dict(steps=steps, support_acc=acc, fallback=False)

    if maml is None:
        raise ValueError(f"mode '{mode}' needs a meta-initialization checkpoint")

    if mode in ("init", "init_reset"):
        if mode == "init_reset" or prev_net is None:
            # init_reset always restarts; init restarts only on the first session,
            # because it has no pretrained-but-fine-tuned weights to continue from.
            net = maml.as_reward_ensemble(maml.init_params())
        else:
            net = prev_net          # the paper's Init: keep fine-tuning
        net, steps, acc = train_on_support(net, support, lr=lr, target_acc=target_acc)
        return net, dict(steps=steps, support_acc=acc, fallback=False,
                         continued=(mode == "init" and prev_net is not None))

    if mode == "few_shot":
        params = maml.adapt(support, steps=max_maml_steps, target_acc=target_acc)
        net = maml.as_reward_ensemble(params)
        from .reward_model import preference_accuracy
        with torch.no_grad():
            _, acc = preference_accuracy(net(*support[:4]), support[4])
        if acc >= target_acc:
            return net, dict(steps=0, support_acc=acc, fallback=False)
        net, steps, acc = train_on_support(net, support, lr=lr, target_acc=target_acc)
        return net, dict(steps=steps, support_acc=acc, fallback=True)

    raise ValueError(f"unknown mode '{mode}'")


@torch.no_grad()
def relabel(net: RewardEnsemble, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Learned reward for a batch of transitions: the ensemble mean of r(s, a)."""
    return net.step_reward(obs, action).mean(dim=0)
