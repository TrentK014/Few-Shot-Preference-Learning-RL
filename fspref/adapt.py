"""Test-time adaptation arms, and the diagnostic that says whether it worked.

Milestone 5 left one thing open. Its meta-initialization scored 0.9401 on
held-out tasks with zero gradient steps, and adapting moved it by 0.0004. So
"MAML beats from scratch" there was really "pretrained features beat no
features". Separating those two claims needs four arms, not two:

    scratch      fresh weights, Adam on K labels                the floor
    meta_init    the checkpoint, zero labels                    pure pretraining
    init         the checkpoint, then ordinary Adam on K labels  the paper's "Init"
    maml         reset to checkpoint, learned-rate SGD, then Adam fallback

Every trained arm goes through the same `train_on_support` loop with the same
stopping rule, so no arm is favoured by construction. That matters more than it
sounds: giving MAML a different step budget or convergence criterion than Init
would manufacture exactly the result we are trying to test for.
"""
from __future__ import annotations

import numpy as np
import torch

from .reward_model import RewardEnsemble, preference_accuracy, preference_loss

# Reference FewShotPEBBLE: adapt to 0.95 support accuracy or 40 MAML steps, then
# fall back to Adam (capped at 1000 epochs there).
TARGET_SUPPORT_ACC = 0.95
MAX_MAML_STEPS = 40
MAX_ADAM_STEPS = 1000


def train_on_support(net: RewardEnsemble, support, lr: float = 3e-4,
                     target_acc: float = TARGET_SUPPORT_ACC,
                     max_steps: int = MAX_ADAM_STEPS):
    """Adam on the support pairs until support accuracy hits target_acc.

    Shared by the scratch and Init arms, and by MAML's fallback phase. Returns
    (net, steps_taken, final support accuracy).
    """
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    obs_a, act_a, obs_b, act_b, y = support
    acc = 0.0
    for step in range(max_steps):
        logits = net(obs_a, act_a, obs_b, act_b)
        with torch.no_grad():
            _, acc = preference_accuracy(logits, y)
        if acc >= target_acc:
            return net, step, acc
        loss = preference_loss(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    return net, max_steps, acc


def scratch_arm(support, ensemble_size: int = 3, seed: int = 0, device="cpu", **kw):
    torch.manual_seed(seed)
    net = RewardEnsemble(ensemble_size=ensemble_size).to(device)
    return train_on_support(net, support, **kw)


def init_arm(maml, support, **kw):
    """Pretrained weights, then plain Adam. The paper's Init baseline."""
    net = maml.as_reward_ensemble(maml.init_params())
    return train_on_support(net, support, **kw)


def maml_arm(maml, support, max_maml_steps: int = MAX_MAML_STEPS, **kw):
    """Reset to the meta-init, adapt with learned per-parameter rates, then fall back.

    Mirrors the reference: the SGD phase runs to 0.95 support accuracy or 40
    steps, and only if it fails to converge does Adam take over.
    """
    params = maml.adapt(support, steps=max_maml_steps, target_acc=TARGET_SUPPORT_ACC)
    net = maml.as_reward_ensemble(params)
    with torch.no_grad():
        _, acc = preference_accuracy(net(*support[:4]), support[4])
    if acc >= TARGET_SUPPORT_ACC:
        return net, 0, acc          # converged inside the MAML phase
    net, steps, acc = train_on_support(net, support, **kw)
    return net, steps, acc          # steps > 0 means the Adam fallback was needed


@torch.no_grad()
def reward_velocity_correlation(net: RewardEnsemble, var, starts, device="cpu",
                                obj_x: int = 4, prev_obj_x: int = 22) -> float:
    """Correlation between predicted per-step reward and sash x-velocity.

    The mechanistic test of whether adaptation did the one thing Window Close
    requires. Window Open rewards moving the sash in +x and Window Close rewards
    the reverse, so a model carrying Window Open's belief should score positive
    here and a correctly adapted one negative. A sign flip is direct evidence the
    model was argued out of the wrong belief rather than merely nudged.

    obs[18:36] holds the previous frame, so obs[4] - obs[22] is one-step
    x-velocity of the object.
    """
    obs, act = var.gather(starts)
    r = net.step_reward(obs, act).mean(dim=0)            # average over ensemble -> (N, T)
    vel = obs[..., obj_x] - obs[..., prev_obj_x]         # (N, T)
    r = r.reshape(-1).cpu().numpy()
    vel = vel.reshape(-1).cpu().numpy()
    if vel.std() < 1e-12 or r.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(vel, r)[0, 1])
