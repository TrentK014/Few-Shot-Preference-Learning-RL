"""MetaWorld environment + scripted-policy helpers.

All MetaWorld version quirks (v2 vs v3 task names, policy class names, gym vs
gymnasium step signatures) live here so later milestones never touch them.
"""
from __future__ import annotations

import re

import numpy as np

import metaworld
import metaworld.policies as mw_policies

MAX_PATH_LENGTH = 500  # MetaWorld default episode length

TASKS = ["window-open", "window-close", "push", "drawer-close"]


def _versioned_name(task: str) -> str:
    """Map 'window-open' -> the task id this MetaWorld install knows ('window-open-v3' or '-v2')."""
    if re.search(r"-v\d$", task):
        return task
    if hasattr(metaworld, "MT1"):
        # MT1 exposes the full set of registered env names.
        try:
            names = list(metaworld.MT1.ENV_NAMES) if hasattr(metaworld.MT1, "ENV_NAMES") else None
        except Exception:
            names = None
        if names:
            for v in ("v3", "v2"):
                if f"{task}-{v}" in names:
                    return f"{task}-{v}"
    # Fallback: probe MT1 for each version.
    for v in ("v3", "v2"):
        try:
            metaworld.MT1(f"{task}-{v}", seed=0)
            return f"{task}-{v}"
        except Exception:
            continue
    raise ValueError(f"Unknown MetaWorld task '{task}'")


def make_env(task: str, seed: int = 0, task_index: int | None = None, render: bool = False):
    """Create a single-task MetaWorld env with a fixed goal variation.

    task_index selects one of the 50 parametric variations (None -> random with `seed`).
    Returns (env, task_name).
    """
    name = _versioned_name(task)
    mt1 = metaworld.MT1(name, seed=seed)
    # v3 envs only render if render_mode is given at construction.
    env = mt1.train_classes[name](render_mode="rgb_array") if render else mt1.train_classes[name]()
    rng = np.random.RandomState(seed)
    tasks = mt1.train_tasks
    idx = rng.randint(len(tasks)) if task_index is None else task_index
    env.set_task(tasks[idx])
    env._fspref_tasks = tasks  # keep all 50 goal variations for set_task_variation()
    # v3 envs expose max_path_length; v2 too. Make sure it's the standard 500.
    if hasattr(env, "max_path_length"):
        env.max_path_length = MAX_PATH_LENGTH
    return env, name


def set_task_variation(env, idx: int):
    """Switch env to parametric goal variation idx (mod 50). Call before reset()."""
    tasks = env._fspref_tasks
    env.set_task(tasks[idx % len(tasks)])
    return idx % len(tasks)


def get_scripted_policy(task: str):
    """Return MetaWorld's scripted expert policy instance for a task name like 'window-open'."""
    base = re.sub(r"-v\d$", "", task)
    camel = "".join(p.capitalize() for p in base.split("-"))
    for suffix in ("V3Policy", "V2Policy", "Policy"):
        cls = getattr(mw_policies, f"Sawyer{camel}{suffix}", None)
        if cls is not None:
            return cls()
    raise ValueError(f"No scripted policy found for '{task}' in metaworld.policies")


def reset_env(env, seed: int | None = None):
    """gymnasium-style reset -> obs (handles envs that return (obs, info) or obs)."""
    out = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def step_env(env, action):
    """Normalize step output to (obs, reward, terminated, truncated, info)."""
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, reward, done, info = out  # legacy gym 4-tuple
    return obs, reward, done, False, info


def rollout(env, policy=None, noise_std: float = 0.0, max_steps: int = MAX_PATH_LENGTH,
            seed: int | None = None, render: bool = False, rng=None):
    """Run one episode. policy=None -> uniform random actions.

    Returns a dict of arrays: obs, action, reward, next_obs, done, success, plus
    scalars 'return', 'success_any', and optionally 'frames'.
    """
    rng = rng or np.random.RandomState(seed)
    obs = reset_env(env, seed)
    low, high = env.action_space.low, env.action_space.high
    O, A, R, NO, D, S, F = [], [], [], [], [], [], []
    for t in range(max_steps):
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
