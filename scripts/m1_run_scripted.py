"""Milestone 1: run a MetaWorld scripted policy, record transitions, check success.

    python scripts/m1_run_scripted.py --task window-open --episodes 5 --out data/window-open_expert.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.envs import get_scripted_policy, make_env, rollout, set_task_variation  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="window-open")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian action noise on the expert")
    p.add_argument("--random", action="store_true", help="uniform random actions instead of the expert")
    p.add_argument("--out", default=None, help=".npz path for saved transitions")
    p.add_argument("--render", action="store_true", help="save a GIF of episode 0 next to --out")
    p.add_argument("--vary-goal", action="store_true", help="use goal variation i for episode i")
    args = p.parse_args()

    env, name = make_env(args.task, seed=args.seed, render=args.render)
    policy = None if args.random else get_scripted_policy(name)
    print(f"task={name} obs_space={env.observation_space.shape} act_space={env.action_space.shape}")
    print(f"policy={'random' if policy is None else type(policy).__name__} noise_std={args.noise_std}")

    episodes = []
    t0 = time.time()
    for ep_i in range(args.episodes):
        if args.vary_goal:
            set_task_variation(env, ep_i)
        ep = rollout(env, policy, noise_std=args.noise_std, seed=args.seed + ep_i,
                     render=(args.render and ep_i == 0))
        episodes.append(ep)
        r = ep["reward"]
        print(f"ep {ep_i}: steps={len(r):3d} return={ep['return']:8.2f} "
              f"reward[min/mean/max]={r.min():.2f}/{r.mean():.2f}/{r.max():.2f} "
              f"first_success_t={int(np.argmax(ep['success'])) if ep['success_any'] else -1:4d} "
              f"success_any={ep['success_any']} success_final={ep['success_final']}")
    print(f"success_any rate = {np.mean([e['success_any'] for e in episodes]):.2f} "
          f"success_final rate = {np.mean([e['success_final'] for e in episodes]):.2f} "
          f"({time.time() - t0:.1f}s)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        keys = ["obs", "action", "reward", "next_obs", "done", "success"]
        flat = {k: np.concatenate([e[k] for e in episodes]) for k in keys}
        flat["episode_id"] = np.concatenate([np.full(len(e["reward"]), i) for i, e in enumerate(episodes)])
        np.savez_compressed(args.out, task=name, **flat)
        print(f"saved {args.out}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in flat.items()))
        if args.render and "frames" in episodes[0]:
            import imageio
            gif = os.path.splitext(args.out)[0] + "_ep0.gif"
            imageio.mimsave(gif, episodes[0]["frames"][::4], duration=0.05)
            print(f"saved {gif} ({len(episodes[0]['frames'])} frames)")


if __name__ == "__main__":
    main()
