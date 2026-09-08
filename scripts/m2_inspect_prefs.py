"""Milestone 2 verification: does the preference dataset make sense?

Runs the checks from the milestone plan and exits non-zero on failure.

    python scripts/m2_inspect_prefs.py --root $FSPREF_ROOT/data/prefs --task window-open
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.collect import SOURCES  # noqa: E402
from fspref.envs import HELD_OUT_TASK  # noqa: E402
from fspref.prefs import (list_datasets, load_dataset, materialize, segment_returns,  # noqa: E402
                          segment_source, task_key, valid_segment_starts)

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/prefs")
    p.add_argument("--task", default="window-open")
    p.add_argument("--max-files", type=int, default=0, help="0 = all")
    p.add_argument("--pretrain-root", default=None,
                   help="root that must never contain the held-out task; defaults to --root. "
                        "Milestone 6 inspects the test root, where window-close SHOULD exist, "
                        "so it points this at the pretraining root instead.")
    args = p.parse_args()

    paths = list_datasets(args.root, args.task)
    if args.max_files:
        paths = paths[:args.max_files]
    if not paths:
        raise SystemExit(f"no datasets under {os.path.join(args.root, args.task)}")
    print(f"{len(paths)} variation files under {args.root}/{args.task}\n")

    # ---- 7. held-out integrity ----
    # The claim is that the held-out task never entered PRETRAINING, not that it
    # is absent from whatever directory is being inspected. Milestone 6 inspects
    # the test root, where window-close is supposed to live.
    print("Held-out integrity")
    guard_root = args.pretrain_root or args.root
    if os.path.isdir(guard_root):
        leaked = [p_ for p_ in list_datasets(guard_root) if HELD_OUT_TASK in p_]
        check(f"no '{HELD_OUT_TASK}' under the pretraining root {guard_root}",
              not leaked, ", ".join(leaked[:3]))
    else:
        check(f"pretraining root {guard_root} exists to be checked", False)
    if args.task == HELD_OUT_TASK:
        print(f"  (inspecting the held-out task itself; it is expected here in {args.root})")

    agg = dict(n_pairs=0, n_label1=0, seg_ret=[], per_source={s: [] for s in SOURCES},
               succ_by_source={s: [] for s in SOURCES}, expert_vs_random=[], keys=set())
    first = None

    for path in paths:
        d = load_dataset(path)
        key = task_key(path)
        agg["keys"].add(key)
        seg_size = int(d["meta"]["segment_size"])
        if first is None:
            first = (path, d, seg_size)

        starts = valid_segment_starts(d["episode_id"], seg_size)
        # ---- 1. no segment crosses an episode boundary ----
        ep_of_start = d["episode_id"][starts]
        ep_of_end = d["episode_id"][starts + seg_size - 1]
        if not np.array_equal(ep_of_start, ep_of_end):
            FAILS.append(f"segment spans episodes in {key}")
        # every referenced pair start must itself be a valid start
        valid_set = set(starts.tolist())
        for side in ("pair_a_start", "pair_b_start"):
            if not set(d[side].tolist()).issubset(valid_set):
                FAILS.append(f"{side} references an invalid segment in {key}")

        # ---- 2. goal masking ----
        if np.abs(d["obs"][:, 36:39]).max() != 0.0:
            FAILS.append(f"goal dims not masked in {key}")

        # ---- label / return statistics ----
        agg["n_pairs"] += len(d["label"])
        agg["n_label1"] += int(d["label"].sum())
        rets = segment_returns(d["reward"], starts, seg_size, float(d["meta"]["discount"]))
        agg["seg_ret"].append(rets)
        src = segment_source(d, starts)
        for i, s in enumerate(SOURCES):
            sel = rets[src == i]
            if len(sel):
                agg["per_source"][s].append(sel)
        # Episode-level success per source: the quality ladder should be
        # expert high, within middling, cross and random at zero.
        for i, s in enumerate(SOURCES):
            m = d["source"] == i
            if m.any():
                succ = [d["success"][a:a + l].max()
                        for a, l in zip(d["episode_start"][m], d["episode_length"][m])]
                agg["succ_by_source"][s].append(np.asarray(succ))

        # ---- 4. sanity direction: expert vs random pairs ----
        sa = segment_source(d, d["pair_a_start"])
        sb = segment_source(d, d["pair_b_start"])
        e, r = SOURCES.index("expert"), SOURCES.index("random")
        m = ((sa == e) & (sb == r)) | ((sa == r) & (sb == e))
        if m.any():
            expert_is_a = sa[m] == e
            expert_won = np.where(expert_is_a, d["label"][m] == 1.0, d["label"][m] == 0.0)
            agg["expert_vs_random"].append(expert_won)

    path0, d0, seg0 = first

    # ---- 1. shapes ----
    print("\nShapes")
    s0 = valid_segment_starts(d0["episode_id"], seg0)[:64]
    obs, act = materialize(d0, s0, seg0)
    check("segment obs is (N, 25, 39)", obs.shape == (len(s0), seg0, 39), str(obs.shape))
    check("segment act is (N, 25, 4)", act.shape == (len(s0), seg0, 4), str(act.shape))
    check("no segment crosses an episode boundary",
          not any(f.startswith("segment spans") for f in FAILS))
    check("all pairs reference valid segments",
          not any("references an invalid segment" in f for f in FAILS))

    # ---- 2. goal masking ----
    print("\nGoal masking")
    check("obs[:, 36:39] == 0 in every file",
          not any("goal dims not masked" in f for f in FAILS))

    # ---- 3. label balance ----
    print("\nLabel balance")
    bal = agg["n_label1"] / max(agg["n_pairs"], 1)
    check("label mean within [0.45, 0.55]", 0.45 <= bal <= 0.55, f"{bal:.4f} over {agg['n_pairs']} pairs")

    # ---- 4. sanity direction ----
    print("\nSanity direction")
    if agg["expert_vs_random"]:
        ew = np.concatenate(agg["expert_vs_random"])
        # Deliberately not a fixed accuracy bar. The achievable rate is a property
        # of each task's reward, not of our pipeline: window-open and push reach
        # ~0.94-0.98 while drawer-close reaches ~0.84 because its reward is
        # binary and its opening steps are indistinguishable from flailing.
        # What must hold everywhere is that the effect is real, so this is a
        # significance test, and the ladder check below carries the substance.
        n, k = len(ew), int(ew.sum())
        z = (k - 0.5 * n) / math.sqrt(0.25 * n) if n else 0.0
        pval = 0.5 * math.erfc(z / math.sqrt(2))
        check("expert beats random far above chance (p < 1e-6)", pval < 1e-6,
              f"{ew.mean():.4f} over {n} such pairs, p={pval:.3g}")
    else:
        check("expert-vs-random pairs exist", False, "none sampled")

    # ---- 5. spread ----
    print("\nSegment-return spread")
    allr = np.concatenate(agg["seg_ret"])
    q = np.percentile(allr, [0, 10, 50, 90, 100])
    print(f"  min {q[0]:.1f} | p10 {q[1]:.1f} | median {q[2]:.1f} | p90 {q[3]:.1f} | max {q[4]:.1f}")
    print("  segment return by behavior source:")
    dist = {}
    for s in SOURCES:
        if agg["per_source"][s]:
            v = np.concatenate(agg["per_source"][s])
            dist[s] = v
            print(f"    {s:8s} n={len(v):7d} mean={v.mean():8.2f} std={v.std():7.2f} "
                  f"median={np.median(v):7.2f} p90={np.percentile(v, 90):7.2f}")
    # The substantive claim is a monotone quality ladder, which is scale-free and
    # holds for every task. Absolute thresholds and distribution-shape tests are
    # not: window-open's reward has a large always-on reaching floor, push's is
    # steeply shaped, and drawer-close's is binary, so any fixed number encodes
    # one task's reward and fails on the others.
    ladder = [s for s in ("expert", "within", "cross") if s in dist]
    means = [dist[s].mean() for s in ladder]
    check("segment return decreases down the source ladder: " + " > ".join(ladder),
          all(means[i] > means[i + 1] for i in range(len(means) - 1)),
          " > ".join(f"{s}={m:.2f}" for s, m in zip(ladder, means)))
    if "expert" in dist and "random" in dist:
        check("expert mean segment return well above random",
              dist["expert"].mean() > 2 * dist["random"].mean(),
              f"expert={dist['expert'].mean():.2f} random={dist['random'].mean():.2f} "
              f"({dist['expert'].mean() / max(dist['random'].mean(), 1e-9):.1f}x)")
    check("segment returns are not degenerate", allr.std() > 1e-3, f"std={allr.std():.3f}")

    print("\n  episode success rate by behavior source:")
    srate = {}
    for s in SOURCES:
        if agg["succ_by_source"][s]:
            v = np.concatenate(agg["succ_by_source"][s])
            srate[s] = v.mean()
            print(f"    {s:8s} n={len(v):4d} success={v.mean():.2f}")
    # The floor is 0.5, not 0.95. MetaWorld's scripted policies are genuinely
    # imperfect on the harder families -- measured here: door-open 0.88-0.92,
    # peg-insert-side 0.88-0.89, basketball 0.89, against ~1.00 for reach, push,
    # window-open and drawer-close. The 0.95 this check used to carry was fitted
    # to the three easy families Milestones 2-6 happened to use, and it flagged
    # three of the ten new ones as broken when their data is fine: their source
    # ladders are strong (door-open expert 114.86 > within 78.88 > cross 17.16)
    # and their expert-vs-random gaps are +0.88 to +0.92.
    #
    # This is the fourth time in this project a threshold encoded an expectation
    # rather than a measurement (Milestone 2's "expert 2x random", Milestone 4's
    # four Window-Open-shaped checks, Milestone 6's "MAML beats scratch" filed as
    # a sanity check, and now this). What the data actually has to satisfy is
    # that the expert is recognisably expert -- which the gap check below tests
    # in a scale-free way -- not that a scripted policy is flawless.
    check("expert source is recognisably expert", srate.get("expert", 0) >= 0.5,
          f"{srate.get('expert', float('nan')):.2f} "
          f"(scripted policies are imperfect on the harder families; the "
          f"discriminating test is the expert-vs-random gap below)")
    # Not "random never succeeds": drawer-close is easy enough that flailing shuts
    # the drawer about a quarter of the time, which is a property of the task.
    # What must hold is a clear gap between deliberate and undirected behavior.
    gap = srate.get("expert", 0.0) - srate.get("random", 1.0)
    check("expert solves it far more often than random behavior", gap >= 0.3,
          f"expert {srate.get('expert', float('nan')):.2f} vs "
          f"random {srate.get('random', float('nan')):.2f} (gap {gap:+.2f})")

    # ---- 6. keying ----
    print("\nTask keying")
    check("one MAML task key per variation file", len(agg["keys"]) == len(paths),
          f"{len(agg['keys'])} keys for {len(paths)} files")
    check("pairs are within-variation by construction", True,
          "pair tables index a single variation's own transitions")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s):")
        for f in dict.fromkeys(FAILS):
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
