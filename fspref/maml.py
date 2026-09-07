"""MAML over the preference reward model.

Ordinary training produces weights that answer well on the tasks they saw. MAML
produces a *starting point*: weights from which a couple of gradient steps on a
handful of preferences land somewhere good, for a task never seen before.

One meta-step, for each of several tasks:

    1. take `num_support` preference pairs from the task
    2. run `num_inner_steps` gradient steps from the current start (the inner loop)
    3. score the adapted weights on `num_query` *different* pairs from that task
    4. push that error back into the ORIGINAL start, not the adapted weights

Step 4 differentiates through step 2, which is the second-order part and the
usual place MAML implementations break. `torch.func.functional_call` drives the
existing RewardEnsemble from an explicit parameter dict so nothing about the
model had to be rewritten.

The inner learning rates are themselves learned, one tensor per parameter tensor
and elementwise within it (the Meta-SGD formulation the reference uses via
`learn_inner_lr`). That lets the model decide which of its own weights are the
task-specific knob and which are shared machinery: weights encoding what contact
looks like learn to barely move during adaptation, weights encoding which
direction of object motion is good learn to move a lot. Nothing supervises that
split; it falls out of the objective.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.func import functional_call

from .reward_model import RewardEnsemble, preference_accuracy, preference_loss

# Reference values, configs/metaworld/maml.yaml.
DEFAULT_INNER_LR = 1e-3
DEFAULT_OUTER_LR = 1e-4
DEFAULT_INNER_STEPS = 2
DEFAULT_SUPPORT = 32
DEFAULT_QUERY = 32
DEFAULT_TASK_BATCH = 4
# The reference adapts for at most 40 steps at test time before falling back to
# Adam. Steps are cheap; human labels are not.
MAX_ADAPT_STEPS = 40


class MAMLReward(nn.Module):
    """A reward-model initialization plus the learned inner learning rates."""

    def __init__(self, obs_dim: int = 39, act_dim: int = 4, ensemble_size: int = 3,
                 inner_lr: float = DEFAULT_INNER_LR, inner_steps: int = DEFAULT_INNER_STEPS):
        super().__init__()
        self.net = RewardEnsemble(obs_dim=obs_dim, act_dim=act_dim, ensemble_size=ensemble_size)
        self.inner_steps = inner_steps
        # One learned learning-rate tensor per parameter tensor, elementwise.
        self.inner_lrs = nn.ParameterDict({
            self._key(name): nn.Parameter(torch.full_like(p, inner_lr))
            for name, p in self.net.named_parameters()})

    @staticmethod
    def _key(name: str) -> str:
        return name.replace(".", "__")  # ParameterDict forbids dots in keys

    def init_params(self):
        """The meta-learned starting point, as a name -> tensor dict."""
        return {name: p for name, p in self.net.named_parameters()}

    def forward_with(self, params, batch):
        obs_a, act_a, obs_b, act_b, _ = batch
        return functional_call(self.net, params, (obs_a, act_a, obs_b, act_b))

    def inner_loop(self, support_batch, params=None, steps: int | None = None,
                   create_graph: bool = True):
        """Adapt from `params` on the support set. Returns the adapted parameters.

        create_graph=True keeps the second-order path needed during meta-training.
        At adaptation time it is unnecessary and wasteful, so callers pass False.
        """
        params = dict(self.init_params() if params is None else params)
        steps = self.inner_steps if steps is None else steps
        y = support_batch[4]
        for _ in range(steps):
            loss = preference_loss(self.forward_with(params, support_batch), y)
            names = list(params)
            grads = torch.autograd.grad(loss, [params[n] for n in names],
                                        create_graph=create_graph)
            params = {n: params[n] - self.inner_lrs[self._key(n)] * g
                      for n, g in zip(names, grads)}
        return params

    def outer_step(self, tasks):
        """Meta-loss over a batch of tasks, each a (support_batch, query_batch) pair.

        Returns (loss, mean query accuracy before adaptation, mean after).
        """
        total = 0.0
        pre_acc, post_acc = [], []
        for support, query in tasks:
            with torch.no_grad():
                _, a0 = preference_accuracy(
                    self.forward_with(self.init_params(), query), query[4])
            adapted = self.inner_loop(support, create_graph=True)
            logits = self.forward_with(adapted, query)
            total = total + preference_loss(logits, query[4])
            with torch.no_grad():
                _, a1 = preference_accuracy(logits, query[4])
            pre_acc.append(a0); post_acc.append(a1)
        n = max(len(tasks), 1)
        return total / n, sum(pre_acc) / n, sum(post_acc) / n

    @torch.enable_grad()
    def adapt(self, support_batch, steps: int = MAX_ADAPT_STEPS, target_acc: float = 0.95):
        """Test-time adaptation: plain gradient steps from the meta-init.

        Stops early once support accuracy reaches target_acc, matching the
        reference's FewShotPEBBLE, which adapts to 0.95 or 40 steps.
        No second-order graph is needed here.
        """
        params = {n: p.detach().clone().requires_grad_(True)
                  for n, p in self.init_params().items()}
        y = support_batch[4]
        for _ in range(steps):
            logits = self.forward_with(params, support_batch)
            with torch.no_grad():
                _, acc = preference_accuracy(logits, y)
            if acc >= target_acc:
                break
            loss = preference_loss(logits, y)
            names = list(params)
            grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=False)
            params = {n: (params[n] - self.inner_lrs[self._key(n)].detach() * g
                          ).detach().requires_grad_(True)
                      for n, g in zip(names, grads)}
        return {n: p.detach() for n, p in params.items()}

    @torch.no_grad()
    def evaluate(self, params, batch):
        logits = self.forward_with(params, batch)
        return preference_accuracy(logits, batch[4])

    def as_reward_ensemble(self, params=None) -> RewardEnsemble:
        """Materialize adapted parameters into a plain module, for SAC later."""
        net = copy.deepcopy(self.net)
        if params is not None:
            net.load_state_dict({k: v.detach() for k, v in params.items()})
        return net
