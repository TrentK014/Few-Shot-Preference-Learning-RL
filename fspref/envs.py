"""MetaWorld environment + scripted-policy helpers.

All MetaWorld quirks live here so later milestones never touch them:

- task ids are versioned (`window-open-v3`), resolved via `MT1.ENV_NAMES`
- goal observability is a flag pickled inside `Task.data`, not an env attribute
- `env.observation_space` is built before `set_task` and keeps the
  partially-observable bounds (goal dims pinned to 0); `env.sawyer_observation_space`
  is the correct one and the one `step()` clips against
- scripted policies do not clip their own output, so we clip before recording
"""
from __future__ import annotations

import pickle
import re

import numpy as np

import metaworld
import metaworld.policies as mw_policies
from metaworld.types import Task

MAX_PATH_LENGTH = 500  # MetaWorld default episode length
OBS_DIM = 39
ACT_DIM = 4
GOAL_SLICE = slice(36, 39)  # obs[36:39] is the goal, zeroed when partially observable

# Meta-training families: the ML10 training set, which is exactly the ten tasks
# in the Figure 2 header of the paper. Milestones 4-6 used only the first three,
# and the Milestone 6 result traced back to that: with three families the
# meta-learner memorizes three solutions instead of learning a transferable
# prior. See notes/06b-paper-crosscheck.md.
#
# Window Close is the held-out task and must never be used as a behavior source,
# or it leaks into pretraining. Note that Window Open IS a prior task, exactly as
# in the paper; the held-out task being the reverse of a prior one is the
# paper's setup too, not an accident of ours.
PRIOR_TASKS = [
    "reach", "push", "pick-place", "door-open", "drawer-close",
    "button-press-topdown", "peg-insert-side", "window-open", "sweep", "basketball",
]
# The three families Milestones 4-6 actually used, kept so the 3-vs-10 comparison
# can be rerun without editing code.
PRIOR_TASKS_V1 = ["window-open", "push", "drawer-close"]
HELD_OUT_TASK = "window-close"
TASKS = PRIOR_TASKS + [HELD_OUT_TASK]

_MT1_CACHE: dict[tuple[str, int], "metaworld.MT1"] = {}


def _versioned_name(task: str) -> str:
    """Map 'window-open' -> the task id this MetaWorld install knows."""
    if re.search(r"-v\d$", task):
        return task
    names = list(getattr(metaworld.MT1, "ENV_NAMES", []))
    for v in ("v3", "v2"):
        if f"{task}-{v}" in names:
            return f"{task}-{v}"
    raise ValueError(f"Unknown MetaWorld task '{task}' (known: {len(names)} names)")


def get_mt1(task: str, mt1_seed: int = 0):
    """Cached MT1 benchmark. The seed is pinned so variation i is always the same goal."""
    name = _versioned_name(task)
    key = (name, mt1_seed)
    if key not in _MT1_CACHE:
        _MT1_CACHE[key] = metaworld.MT1(name, seed=mt1_seed)
    return _MT1_CACHE[key], name


def task_with_observability(task: Task, partially_observable: bool) -> Task:
    """Return a copy of `task` with the goal-observability flag flipped.

    Safe pickle use: `task.data` is produced by MetaWorld itself inside this
    process (metaworld/__init__.py pickles a dict of rand_vec/env_cls/flags) and
    is never read from an external or user-supplied file. We round-trip it only
    to flip one boolean the public API does not expose.

    MT1 builds tasks with partially_observable=False (goal visible in obs[36:39]).
    Recorded data must have the goal masked, so the reward model cannot read the
    target off the observation; scripted policies driving the *source* env still
    need it visible. Mirrors what the reference implementation does.
    """
    data = pickle.loads(task.data)
    data["partially_observable"] = partially_observable
    return Task(env_name=task.env_name, data=pickle.dumps(data))


def make_env(task: str, variation: int = 0, mt1_seed: int = 0,
             partially_observable: bool = True, render: bool = False):
    """Create a single-task MetaWorld env pinned to one goal variation.

    partially_observable=True (default) zeroes obs[36:39]; that is what we record.
    Returns (env, task_name).
    """
    mt1, name = get_mt1(task, mt1_seed)
    env = mt1.train_classes[name](render_mode="rgb_array") if render else mt1.train_classes[name]()
    env._fspref_tasks = mt1.train_tasks
    env._fspref_po = partially_observable
    set_task_variation(env, variation)
    if hasattr(env, "max_path_length"):
        env.max_path_length = MAX_PATH_LENGTH
    return env, name


def set_task_variation(env, idx: int) -> int:
    """Switch env to goal variation idx (mod 50), preserving its observability setting.

    Call before reset(); set_task does not reset, and the target is derived from
    the frozen rand_vec inside reset_model().
    """
    tasks = env._fspref_tasks
    idx = idx % len(tasks)
    env.set_task(task_with_observability(tasks[idx], env._fspref_po))
    return idx


def n_variations(task: str, mt1_seed: int = 0) -> int:
    mt1, _ = get_mt1(task, mt1_seed)
    return len(mt1.train_tasks)


def get_scripted_policy(task: str):
    """MetaWorld's scripted expert for a task name like 'window-open'."""
    name = _versioned_name(task)
    policy_map = getattr(mw_policies, "ENV_POLICY_MAP", None)
    if policy_map and name in policy_map:
        return policy_map[name]()
    # Fallback for installs without ENV_POLICY_MAP.
    camel = "".join(p.capitalize() for p in re.sub(r"-v\d$", "", name).split("-"))
    for suffix in ("V3Policy", "V2Policy", "Policy"):
        cls = getattr(mw_policies, f"Sawyer{camel}{suffix}", None)
        if cls is not None:
            return cls()
    raise ValueError(f"No scripted policy found for '{task}'")


def reset_env(env, seed: int | None = None):
    """gymnasium-style reset -> obs."""
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def step_env(env, action):
    """Normalize step output to (obs, reward, terminated, truncated, info)."""
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, reward, done, info = out  # legacy gym 4-tuple
    return obs, reward, done, False, info


def rollout(env, policy=None, noise_std: float = 0.0, max_steps: int = MAX_PATH_LENGTH,
            seed: int | None = None, render: bool = False, rng=None):
    """Run one single-env episode. policy=None -> uniform random actions.

    Kept for Milestone 1 style diagnostics; data generation uses fspref.collect,
    which needs two envs stepped in lockstep.
    """
    rng = rng if rng is not None else np.random.RandomState(seed)
    obs = reset_env(env, seed)
    low, high = env.action_space.low, env.action_space.high
    O, A, R, NO, D, S, F = [], [], [], [], [], [], []
    for _ in range(max_steps):
        if policy is None:
            act = rng.uniform(low, high)
        else:
            act = np.asarray(policy.get_action(obs), dtype=np.float32)
            if noise_std > 0:
                act = act + rng.normal(0.0, noise_std, size=act.shape)
        act = np.clip(act, low, high).astype(np.float32)
        if render:
            F.append(env.render())
        next_obs, rew, term, trunc, info = step_env(env, act)
        O.append(obs); A.append(act); R.append(rew); NO.append(next_obs)
        D.append(term or trunc); S.append(float(info.get("success", 0.0)))
        obs = next_obs
        if term or trunc:
            break
    ep = dict(obs=np.array(O, dtype=np.float32), action=np.array(A, dtype=np.float32),
              reward=np.array(R, dtype=np.float32), next_obs=np.array(NO, dtype=np.float32),
              done=np.array(D, dtype=bool), success=np.array(S, dtype=np.float32))
    ep["return"] = float(ep["reward"].sum())
    ep["success_any"] = bool(ep["success"].max() > 0)
    ep["success_final"] = bool(ep["success"][-1] > 0)
    if render:
        ep["frames"] = np.array(F)
    return ep
