"""Preference-based reward model: r_psi(s, a) -> scalar, trained from A-vs-B labels.

Architecture follows the reference implementation
(jhejna/few-shot-preference-rl, research/networks/mlp.py::MetaRewardMLPEnsemble):
raw concatenation of observation and action, no normalization anywhere, three
hidden layers of 256 with leaky ReLU, tanh output, and an ensemble of 3.

A segment's score is the sum of per-step rewards over its 25 steps. Preferences
use the Bradley-Terry model, which reduces to binary cross-entropy on the
difference of segment returns:

    P[A > B] = exp(R_A) / (exp(R_A) + exp(R_B)) = sigmoid(R_A - R_B)

so the logit is simply R_A - R_B and the target is the preference label.

The ensemble is batched into single tensors of shape (E, in, out) so all members
run in one `baddbmm`. Every layer is a plain nn.Module parameter, so MAML in
Milestone 5 can drive this with torch.func.functional_call without a rewrite.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HIDDEN = (256, 256, 256)
DEFAULT_ENSEMBLE = 3


class EnsembleLinear(nn.Module):
    """`ensemble_size` independent linear layers applied in one batched matmul."""

    def __init__(self, in_features: int, out_features: int, ensemble_size: int = DEFAULT_ENSEMBLE):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))
        # Per-member init identical in distribution to nn.Linear's default.
        bound = 1.0 / np.sqrt(in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (E, B, in) -> (E, B, out)
        return torch.baddbmm(self.bias, x, self.weight)


class RewardEnsemble(nn.Module):
    """Ensemble of per-timestep reward functions r(s, a) in [-1, 1]."""

    def __init__(self, obs_dim: int = 39, act_dim: int = 4, hidden=DEFAULT_HIDDEN,
                 ensemble_size: int = DEFAULT_ENSEMBLE):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.obs_dim, self.act_dim = obs_dim, act_dim
        dims = [obs_dim + act_dim, *hidden]
        self.layers = nn.ModuleList(
            [EnsembleLinear(dims[i], dims[i + 1], ensemble_size) for i in range(len(dims) - 1)])
        self.head = EnsembleLinear(dims[-1], 1, ensemble_size)

    def step_reward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-timestep reward. obs (..., obs_dim), action (..., act_dim) -> (E, ...)."""
        x = torch.cat((obs, action), dim=-1)
        flat = x.reshape(-1, x.shape[-1])
        h = flat.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for layer in self.layers:
            h = F.leaky_relu(layer(h))
        out = torch.tanh(self.head(h))  # (E, N, 1)
        return out.reshape(self.ensemble_size, *x.shape[:-1])

    def segment_return(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Sum of predicted reward over a segment. obs (B, T, obs_dim) -> (E, B)."""
        return self.step_reward(obs, action).sum(dim=-1)

    def forward(self, obs_a, act_a, obs_b, act_b) -> torch.Tensor:
        """Preference logits for A over B, shape (E, B). sigmoid(logit) = P[A > B]."""
        return self.segment_return(obs_a, act_a) - self.segment_return(obs_b, act_b)


def preference_loss(logits: torch.Tensor, label: torch.Tensor):
    """Bradley-Terry loss: BCE on (R_A - R_B). label 1.0 means A is preferred.

    Reduction matches the reference exactly (research/algs/pebble.py): mean over
    the batch, then SUM over ensemble members. Members are independent parameter
    blocks so summing does not rescale any one member's gradient, but averaging
    over E instead would divide every gradient by E and silently make lr 3e-4
    mean something different.
    """
    target = label.unsqueeze(0).expand_as(logits)
    per_element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return per_element.mean(dim=-1).sum(dim=0)


@torch.no_grad()
def preference_accuracy(logits: torch.Tensor, label: torch.Tensor):
    """(per-member mean accuracy, ensemble-mean accuracy)."""
    pred = (logits > 0).float()
    target = label.unsqueeze(0).expand_as(logits)
    per_member = (pred == target).float().mean().item()
    ens = ((logits.mean(dim=0) > 0).float() == label).float().mean().item()
    return per_member, ens


@torch.no_grad()
def disagreement(logits: torch.Tensor) -> torch.Tensor:
    """std over ensemble members of P[A > B]; the reference's query-selection score."""
    return torch.sigmoid(logits).std(dim=0)
