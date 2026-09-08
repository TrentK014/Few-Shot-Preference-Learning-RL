"""Milestone 2: generate diverse trajectories and labeled preference pairs.

One .npz per (family, goal variation). Milestone 4 is this same command run for
push and drawer-close.

    python scripts/m2_generate_prefs.py --task window-open --variations 25 \
        --out-root $FSPREF_ROOT/data/prefs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.collect import (DEFAULT_EPSILON, DEFAULT_MIXTURE, DEFAULT_SUCCESS_TAIL,  # noqa: E402
                             SOURCES, collect_variation)
from fspref.envs import HELD_OUT_TASK, n_variations  # noqa: E402
from fspref.prefs import (N_PAIRS, SEGMENT_SIZE, build_pairs, save_dataset,  # noqa: E402
                          valid_segment_starts, variation_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="window-open", help="task family to generate")
    p.add_argument("--variations", type=int, default=25,
                   help="goal variations per family (reference uses tasks-per-env=25)")
    p.add_argument("--start-variation", type=int, default=0)
    p.add_argument("--out-root", default="data/prefs")
    p.add_argument("--seed", type=int, default=0, help="base seed; variation i uses seed+i")
    p.add_argument("--mt1-seed", type=int, default=0, help="pinned so variation i is a stable goal")
    p.add_argument("--pairs", type=int, default=N_PAIRS)
    p.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON, help="gaussian action noise")
    p.add_argument("--discount", type=float, default=1.0,
                   help="1.0 = paper's plain sum; 0.99 matches the reference code")
    p.add_argument("--tie-margin", type=float, default=0.0)
    p.add_argument("--success-tail", type=int, default=DEFAULT_SUCCESS_TAIL,
                   help="steps recorded after success; 0 reproduces the reference. "
                        "Needed because drawer-close reward is binary")
    p.add_argument("--no-stop-on-success", action="store_true",
                   help="run every episode to truncation instead of ending on success")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--allow-held-out", action="store_true",
                   help=f"permit --task {HELD_OUT_TASK}. Milestone 6 only, and never into the "
                        "pretraining root")
    p.add_argument("--pretrain-root", default=None,
                   help="path the held-out task may never be written into (defaults to a sibling "
                        "'prefs' directory of --out-root)")
    args = p.parse_args()

    if args.task == HELD_OUT_TASK and not args.allow_held_out:
        raise SystemExit(f"refusing to generate '{HELD_OUT_TASK}': it is the held-out task "
                         "and must not enter MAML pretraining. Milestone 6 passes --allow-held-out "
                         "and writes to a separate root.")
    if args.task == HELD_OUT_TASK:
        # Even with the flag, never let the held-out task land where meta-training looks.
        out = os.path.abspath(args.out_root)
        forbidden = os.path.abspath(args.pretrain_root) if args.pretrain_root else \
            os.path.join(os.path.dirname(out), "prefs")
        if out == forbidden or out.startswith(forbidden + os.sep):
            raise SystemExit(f"refusing to write '{HELD_OUT_TASK}' into the pretraining root "
                             f"{forbidden}: meta-training globs that directory, so this would leak "
                             "the held-out task into the initialization")
        print(f"NOTE: generating the HELD-OUT task '{HELD_OUT_TASK}' into {out} "
              f"(pretraining root {forbidden} left untouched)")

    n_var_total = n_variations(args.task, args.mt1_seed)
    stop_on_success = not args.no_stop_on_success
    print(f"task={args.task} variations={args.start_variation}..{args.start_variation + args.variations - 1} "
          f"of {n_var_total} | mixture={DEFAULT_MIXTURE} eps={args.epsilon} "
          f"stop_on_success={stop_on_success} success_tail={args.success_tail}")
    print(f"pairs/variation={args.pairs} segment={args.segment_size} discount={args.discount} "
          f"tie_margin={args.tie_margin}")

    pool = None
    t0 = time.time()
    for v in range(args.start_variation, args.start_variation + args.variations):
        data, pool = collect_variation(
            args.task, v, seed=args.seed + v, mt1_seed=args.mt1_seed,
            epsilon=args.epsilon, max_steps=args.max_steps,
            stop_on_success=stop_on_success, pool=pool, n_variations=n_var_total,
            success_tail=args.success_tail)
        pairs = build_pairs(data, n_pairs=args.pairs, segment_size=args.segment_size,
                            discount=args.discount, tie_margin=args.tie_margin,
                            seed=args.seed + v)
        meta = dict(task=args.task, variation=v, seed=args.seed + v, mt1_seed=args.mt1_seed,
                    segment_size=args.segment_size, discount=args.discount,
                    tie_margin=args.tie_margin, epsilon=args.epsilon,
                    stop_on_success=stop_on_success, success_tail=args.success_tail,
                    sources=",".join(SOURCES))
        path = variation_path(args.out_root, args.task, v)
        save_dataset(path, data, pairs, meta)

        n_seg = len(valid_segment_starts(data["episode_id"], args.segment_size))
        ep_ret = np.array([data["reward"][s:s + l].sum()
                           for s, l in zip(data["episode_start"], data["episode_length"])])
        succ = np.array([data["success"][s:s + l].max()
                         for s, l in zip(data["episode_start"], data["episode_length"])])
        print(f"var{v:02d}: {len(data['episode_start'])} eps, {len(data['reward'])} steps, "
              f"{n_seg} segments, {len(pairs['label'])} pairs | "
              f"ep_return mean={ep_ret.mean():7.1f} min={ep_ret.min():6.1f} max={ep_ret.max():7.1f} | "
              f"success={succ.mean():.2f} | label_mean={pairs['label'].mean():.3f} | "
              f"{os.path.relpath(path, args.out_root)}")

    print(f"done in {time.time() - t0:.1f}s -> {os.path.join(args.out_root, args.task)}")


if __name__ == "__main__":
    main()
