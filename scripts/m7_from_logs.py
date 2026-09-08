"""Reconstruct Milestone 7 shard results from SLURM .out logs.

Insurance, not the primary path. A shard writes its JSON only when it finishes
(or, since the incremental-write change, after each evaluation) -- but every
shard prints one line per evaluation to its .out file the whole way through:

    [init seed0] step  120000  success 0.00  feedback  192  (9258s)

So if a shard is killed by the wall clock, its curve is still fully recoverable
from the log. That matters most for the Init arm, which runs plain Adam to
convergence every feedback session where MAML converges inside its 40
learned-rate steps, and is therefore several times slower in wall-clock terms
and the arm most likely to run out of time.

Writes the same shard schema scripts/m7_combine.py already reads, marked
partial, so a recovered shard merges alongside real ones.

    python scripts/m7_from_logs.py --logs 'fspref-m7arr-11795010_*.out' \
        --out $FSPREF_ROOT/runs/m7_3family
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

# "    [init seed0] step  120000  success 0.00  feedback  192  (9258s)"
LINE = re.compile(
    r"\[(?P<arm>\w+)\s+seed(?P<seed>\d+)\]\s+step\s+(?P<step>\d+)\s+"
    r"success\s+(?P<succ>[\d.]+)\s+feedback\s+(?P<fb>\d+)")


def parse(path: str):
    arm = seed = None
    curve = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            arm = m.group("arm")
            seed = int(m.group("seed"))
            curve.append([int(m.group("step")), float(m.group("succ")),
                          int(m.group("fb"))])
    return arm, seed, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="glob for the .out files")
    ap.add_argument("--out", required=True, help="directory to write shard json into")
    ap.add_argument("--final-window", type=int, default=3)
    ap.add_argument("--max-feedback", type=int, default=200)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing shard file; by default a real "
                         "(completed) shard result is never clobbered")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.expanduser(args.logs)))
    if not paths:
        raise SystemExit(f"no logs matched {args.logs}")
    os.makedirs(args.out, exist_ok=True)

    written, skipped = 0, 0
    for p in paths:
        arm, seed, curve = parse(p)
        if not curve:
            print(f"  {os.path.basename(p)}: no evaluation lines, skipped")
            continue
        dest = os.path.join(args.out, f"m7_results_{arm}_s{seed}.json")
        if os.path.exists(dest) and not args.overwrite:
            with open(dest) as f:
                existing = json.load(f)
            done = not existing["results"][arm][0].get("partial", False)
            if done:
                print(f"  {arm} s{seed}: completed shard already present, left alone")
                skipped += 1
                continue
        run = dict(arm=arm, seed=seed, curve=curve, feedback=[], readapt=[],
                   total_feedback=curve[-1][2],
                   final_success=float(np.mean([c[1] for c in curve[-args.final_window:]])),
                   best_success=max(c[1] for c in curve),
                   partial=True, recovered_from_log=os.path.basename(p))
        with open(dest, "w") as f:
            json.dump({"args": {"max_feedback": args.max_feedback},
                       "results": {arm: [run]}}, f)
        print(f"  {arm} s{seed}: {len(curve)} evals, last step {curve[-1][0]}, "
              f"final {run['final_success']:.3f} -> {os.path.basename(dest)}")
        written += 1

    print(f"\nwrote {written}, skipped {skipped} already-complete")


if __name__ == "__main__":
    main()
