"""Behavior generation for preference datasets.

Implements the reference implementation's dual-environment "policy transplant"
(jhejna/few-shot-preference-rl, scripts/metaworld/collect_policy_dataset.py):

    two envs are stepped with the SAME action stream; the scripted policy sees
    src_env's observation, but every recorded transition comes from dest_env.

Because the two envs start from different object placements they diverge
immediately, so the actions are plausible but wrong in dest_env. Only the choice
of src_env distinguishes the three policy sources, which is why they collapse
into one code path here:

    expert  -> src is the same variation as dest      (correct behavior)
    within  -> src is another variation, same family  (plausible, misdirected)
    cross   -> src is another family, its own policy  (unrelated but valid actions)
    random  -> no src env at all                      (quality floor)

dest_env is always goal-masked (obs[36:39] == 0) so the reward model cannot read
the target off the observation. src_env is goal-visible so goal-reading policies
such as push still work.
"""
from __future__ import annotations

import numpy as np

from .envs import (MAX_PATH_LENGTH, PRIOR_TASKS, get_scripted_policy, make_env,
                   reset_env, set_task_variation, step_env)

# Reference mixture (README invocation): 15 expert / 25 within / 10 cross / 2 random.
DEFAULT_MIXTURE = {"expert": 15, "within": 25, "cross": 10, "random": 2}
DEFAULT_EPSILON = 0.1  # gaussian action noise, reference default
# Steps to keep recording after success. The reference stops immediately, which
# works only if the task's reward is shaped. MetaWorld v3 drawer-close is
# effectively binary: 0.45 at reset, then exactly 0.0 for the whole approach,
# then 10.0 once the drawer shuts. Stopping at the success step therefore throws
# away every bit of signal it has, and 74% of its preference pairs came out
# separated by less than 1e-6. One extra segment length lets at least one whole
# segment observe the post-success state. Set 0 to reproduce the reference.
DEFAULT_SUCCESS_TAIL = 25
SOURCES = ["expert", "within", "cross", "random"]
SOURCE_ID = {s: i for i, s in enumerate(SOURCES)}


def collect_episode(src_env, dest_env, policy, rng, epsilon=DEFAULT_EPSILON,
                    max_steps=MAX_PATH_LENGTH, stop_on_success=True,
                    success_tail=DEFAULT_SUCCESS_TAIL):
    """Step src_env and dest_env with one shared action stream; record dest_env.

    The policy is driven by src_env's observation. Noise is added before either
    env steps, so the two stay on the same action sequence.
    """
    obs = reset_env(dest_env)
    src_obs = reset_env(src_env)
    O, A, R, S = [], [], [], []
    remaining = None
    for _ in range(max_steps):
        act = np.asarray(policy.get_action(src_obs), dtype=np.float32)
        if epsilon > 0:
            act = act + epsilon * rng.randn(*act.shape)
        # Scripted policies do not clip their own output; the env would clip
        # internally, so clip here to keep the recorded action truthful.
        act = np.clip(act, -1.0, 1.0).astype(np.float32)
        O.append(obs); A.append(act)
        obs, rew, _, _, info = step_env(dest_env, act)
        src_obs, _, _, _, src_info = step_env(src_env, act)
        R.append(rew); S.append(float(info.get("success", 0.0)))
        if stop_on_success and (info.get("success", 0.0) or src_info.get("success", 0.0)):
            if remaining is None:
                remaining = success_tail
        if remaining is not None:
            if remaining <= 0:
                break
            remaining -= 1
    return _pack(O, A, R, S)


def collect_random_episode(dest_env, rng, max_steps=MAX_PATH_LENGTH, stop_on_success=True,
                           success_tail=DEFAULT_SUCCESS_TAIL):
    """Uniform random actions in dest_env. No source env."""
    obs = reset_env(dest_env)
    low, high = dest_env.action_space.low, dest_env.action_space.high
    O, A, R, S = [], [], [], []
    remaining = None
    for _ in range(max_steps):
        act = rng.uniform(low, high).astype(np.float32)
        O.append(obs); A.append(act)
        obs, rew, _, _, info = step_env(dest_env, act)
        R.append(rew); S.append(float(info.get("success", 0.0)))
        if stop_on_success and info.get("success", 0.0):
            if remaining is None:
                remaining = success_tail
        if remaining is not None:
            if remaining <= 0:
                break
            remaining -= 1
    return _pack(O, A, R, S)


def _pack(O, A, R, S):
    return dict(obs=np.asarray(O, dtype=np.float32), action=np.asarray(A, dtype=np.float32),
                reward=np.asarray(R, dtype=np.float32), success=np.asarray(S, dtype=np.float32))


class _SourceEnvPool:
    """Lazily built, reused source envs. Building a MuJoCo env is far slower than resetting one."""

    def __init__(self, family: str, mt1_seed: int, cross_families):
        self.family = family
        self.mt1_seed = mt1_seed
        self.cross_families = list(cross_families)
        # Source envs are goal-VISIBLE so goal-reading policies (push) work.
        self._same, _ = make_env(family, variation=0, mt1_seed=mt1_seed,
                                 partially_observable=False)
        self._same_policy = get_scripted_policy(family)
        self._cross = {}

    def same_family(self, variation: int):
        set_task_variation(self._same, variation)
        return self._same, self._same_policy

    def other_family(self, other: str, variation: int):
        if other not in self._cross:
            env, _ = make_env(other, variation=0, mt1_seed=self.mt1_seed,
                              partially_observable=False)
            self._cross[other] = (env, get_scripted_policy(other))
        env, policy = self._cross[other]
        set_task_variation(env, variation)
        return env, policy


def collect_variation(family: str, variation: int, seed: int, mt1_seed: int = 0,
                      mixture=None, epsilon=DEFAULT_EPSILON, max_steps=MAX_PATH_LENGTH,
                      stop_on_success=True, cross_families=None, pool=None,
                      n_variations: int = 50, success_tail=DEFAULT_SUCCESS_TAIL):
    """Generate the full episode mixture for one (family, variation).

    Returns flat arrays plus an episode index, so segments can be sampled without
    materializing copies. `seed` makes the result reproducible.
    """
    mixture = dict(DEFAULT_MIXTURE if mixture is None else mixture)
    if cross_families is None:
        # Only prior tasks may act as behavior sources. Window Close must never
        # appear here, or the held-out task leaks into pretraining.
        cross_families = [t for t in PRIOR_TASKS if t != family]
    assert "window-close" not in cross_families, "held-out task used as a behavior source"

    rng = np.random.RandomState(seed)
    dest, _ = make_env(family, variation=variation, mt1_seed=mt1_seed,
                       partially_observable=True)  # goal masked in recorded data
    pool = pool if pool is not None else _SourceEnvPool(family, mt1_seed, cross_families)

    episodes, sources = [], []
    for _ in range(mixture.get("expert", 0)):
        src, pol = pool.same_family(variation)
        episodes.append(collect_episode(src, dest, pol, rng, epsilon, max_steps,
                                        stop_on_success, success_tail))
        sources.append("expert")
    for _ in range(mixture.get("within", 0)):
        other = int(rng.randint(n_variations))
        while other == variation and n_variations > 1:
            other = int(rng.randint(n_variations))
        src, pol = pool.same_family(other)
        episodes.append(collect_episode(src, dest, pol, rng, epsilon, max_steps,
                                        stop_on_success, success_tail))
        sources.append("within")
    for _ in range(mixture.get("cross", 0)):
        fam = cross_families[int(rng.randint(len(cross_families)))]
        src, pol = pool.other_family(fam, int(rng.randint(n_variations)))
        episodes.append(collect_episode(src, dest, pol, rng, epsilon, max_steps,
                                        stop_on_success, success_tail))
        sources.append("cross")
    for _ in range(mixture.get("random", 0)):
        episodes.append(collect_random_episode(dest, rng, max_steps, stop_on_success,
                                               success_tail))
        sources.append("random")

    return _flatten(episodes, sources), pool


def _flatten(episodes, sources):
    lengths = np.array([len(e["reward"]) for e in episodes], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    out = {k: np.concatenate([e[k] for e in episodes]) for k in ("obs", "action", "reward", "success")}
    out["episode_start"] = starts
    out["episode_length"] = lengths
    out["source"] = np.array([SOURCE_ID[s] for s in sources], dtype=np.int64)
    # Per-step episode id, so segment sampling can reject boundary-crossing windows.
    out["episode_id"] = np.repeat(np.arange(len(episodes), dtype=np.int64), lengths)
    return out
