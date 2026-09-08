"""Milestone 6: adapt the meta-initialization to held-out Window Close.

The experiment everything else was scaffolding for. Four arms on identical
support pairs, four label budgets, several goal variations and support draws.

Verification is split in two on purpose. Sanity checks fail the job. Hypotheses
are the experiment's *findings*: each prints CONFIRMED or REFUTED with its
numbers, and a refuted hypothesis is a result to write up, not a threshold to
loosen. Milestones 2 and 4 both shipped thresholds that were wrong because they
encoded an expectation rather than a measurement.

    python scripts/m6_window_close.py --test-root $FSPREF_ROOT/data/test_prefs \
        --checkpoint $FSPREF_ROOT/runs/m5/maml_init.pt --out $FSPREF_ROOT/runs/m6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.adapt import (init_arm, maml_arm, reward_velocity_correlation,  # noqa: E402
                          scratch_arm)
from fspref.baselines import calibration_report  # noqa: E402
from fspref.data import VariationData  # noqa: E402
from fspref.envs import HELD_OUT_TASK  # noqa: E402
from fspref.maml import MAMLReward  # noqa: E402
from fspref.prefs import SEGMENT_SIZE, list_datasets  # noqa: E402
from fspref.reward_model import preference_accuracy  # noqa: E402

ARMS = ["scratch", "init", "maml"]
FAILS: list[str] = []
FINDINGS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def finding(name, confirmed, detail=""):
    print(f"  [{'CONFIRMED' if confirmed else 'REFUTED  '}] {name}" + (f": {detail}" if detail else ""))
    FINDINGS.append((name, confirmed, detail))
    return confirmed


def paired_stats(a, b):
    """Paired comparison of two arms over runs that shared the same support pairs.

    Runs are appended in the same (variation, draw) order for every arm, so row j
    of `a` and row j of `b` saw identical labels and differ only in where the
    weights started. That pairing removes the between-variation variance that
    swamps a naive two-sample comparison at these sample sizes.

    Returns (mean difference, sd of the difference, t statistic, two-sided p,
    number of wins, n). p comes from the normal approximation to the paired t,
    which is adequate at n = variations x draws and avoids a scipy dependency.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), 1.0, 0, n
    md, sd = float(d.mean()), float(d.std(ddof=1))
    t = md / (sd / math.sqrt(n)) if sd > 0 else 0.0
    pval = math.erfc(abs(t) / math.sqrt(2))
    return md, sd, t, pval, int((d > 0).sum()), n


def binom_p(correct: int, n: int) -> float:
    if n == 0:
        return 1.0
    z = (correct - 0.5 * n) / math.sqrt(0.25 * n)
    return 0.5 * math.erfc(z / math.sqrt(2))


@torch.no_grad()
def acc_of(net, batch):
    _, ens = preference_accuracy(net(*batch[:4]), batch[4])
    return ens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", default="data/test_prefs")
    p.add_argument("--pretrain-root", default="data/prefs")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="runs/m6")
    p.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    p.add_argument("--variations", type=int, default=5)
    p.add_argument("--draws", type=int, default=3, help="independent support draws per variation")
    p.add_argument("--eval-pairs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hard-quantile", type=float, default=0.25)
    p.add_argument("--support-sources", nargs="+", default=None,
                   choices=["expert", "within", "cross", "random"],
                   help="restrict support pairs to these behavior sources. The paper's "
                        "online queries come from a still-bad policy's replay buffer, so "
                        "both segments are mediocre; our offline mixture contains scripted "
                        "experts. Pass 'within cross random' to approximate the online "
                        "regime, which is where a prior should matter most.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # ---------------------------------------------------------------- isolation
    print("Data isolation")
    pre_paths = list_datasets(args.pretrain_root) if os.path.isdir(args.pretrain_root) else []
    leaked = [q for q in pre_paths if HELD_OUT_TASK in q]
    check(f"no '{HELD_OUT_TASK}' under the pretraining root", not leaked, ", ".join(leaked[:3]))
    # weights_only=True: the checkpoint holds only tensors, dicts, lists and strings,
    # so there is no reason to allow arbitrary unpickling here.
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_paths = ck.get("train_paths", [])
    ck_leak = [q for q in train_paths if HELD_OUT_TASK in q]
    check("no meta-training task was Window Close", not ck_leak,
          f"{len(train_paths)} meta-train tasks, {len(ck_leak)} leaked")

    test_paths = list_datasets(args.test_root, HELD_OUT_TASK)[:args.variations]
    if not test_paths:
        raise SystemExit(f"no Window Close data under {args.test_root}")
    print(f"\n{len(test_paths)} Window Close variations | budgets {args.budgets} | "
          f"{args.draws} support draws each | device={args.device}")

    torch.manual_seed(args.seed)
    maml = MAMLReward(ensemble_size=ck["args"]["ensemble"],
                      inner_lr=ck["args"]["inner_lr"],
                      inner_steps=ck["args"]["inner_steps"]).to(args.device)
    maml.load_state_dict(ck["state_dict"])
    print(f"loaded meta-init from {args.checkpoint}")

    tasks = [VariationData(q, args.device, SEGMENT_SIZE, seed=args.seed) for q in test_paths]

    # ------------------------------------------------------------- the experiment
    results = {a: {k: [] for k in args.budgets} for a in ARMS}
    meta_init_accs, cal_overall, cal_hard = [], [], []
    maml_hard, init_hard, scratch_hard = ({k: [] for k in args.budgets} for _ in range(3))
    sup_acc = {a: {k: [] for k in args.budgets} for a in ARMS}
    corr_before, corr_after = [], []
    fallbacks = {k: 0 for k in args.budgets}
    t0 = time.time()

    for ti, var in enumerate(tasks):
        ea, eb, ey = var.episode_holdout_pairs(n_pairs=args.eval_pairs,
                                               seed=args.seed + 7, which="test")
        eval_batch = var.batch(ea, eb, ey)
        cal = calibration_report(var.obs_np, ea, eb, ey, var.segment_size, args.hard_quantile)
        cal_overall.append(cal["hand_dist_acc"]); cal_hard.append(cal["hard_heuristic_acc"])
        hard_batch = var.batch(ea[cal["hard_mask"]], eb[cal["hard_mask"]], ey[cal["hard_mask"]])

        base_net = maml.as_reward_ensemble(maml.init_params())
        meta_init_accs.append(acc_of(base_net, eval_batch))
        eval_starts = var.test_starts[:400]
        corr_before.append(reward_velocity_correlation(base_net, var, eval_starts, args.device))

        for k in args.budgets:
            for d in range(args.draws):
                rng = np.random.RandomState(args.seed + 1000 * ti + 31 * d + k)
                support = var.sample_batch(k, rng, split="train",
                                           sources=args.support_sources)
                for arm in ARMS:
                    if arm == "scratch":
                        net, _, sa = scratch_arm(support, ensemble_size=maml.net.ensemble_size,
                                                 seed=args.seed + ti * 10 + d, device=args.device,
                                                 lr=args.lr)
                    elif arm == "init":
                        net, _, sa = init_arm(maml, support, lr=args.lr)
                    else:
                        net, steps, sa = maml_arm(maml, support, lr=args.lr)
                        fallbacks[k] += steps > 0
                    sup_acc[arm][k].append(sa)
                    results[arm][k].append(acc_of(net, eval_batch))
                    hd = acc_of(net, hard_batch) if len(ey[cal["hard_mask"]]) else float("nan")
                    {"maml": maml_hard, "init": init_hard, "scratch": scratch_hard}[arm][k].append(hd)
                    if arm == "maml" and k == max(args.budgets) and d == 0:
                        corr_after.append(reward_velocity_correlation(net, var, eval_starts,
                                                                     args.device))
        print(f"  {var.key}: meta-init {meta_init_accs[-1]:.4f} | "
              f"heuristic {cal['hand_dist_acc']:.4f} | done ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------- report
    def m(x):
        return float(np.mean(x)), float(np.std(x))

    print(f"\n{'=' * 78}\nRESULTS  (mean +/- sd over {len(tasks)} variations x {args.draws} draws)\n")
    mi, mis = m(meta_init_accs)
    print(f"  meta-init, unadapted (0 labels): {mi:.4f} +/- {mis:.4f}")
    print(f"  hand-distance heuristic        : {np.mean(cal_overall):.4f}  "
          f"(hard subset {np.mean(cal_hard):.4f})")
    print(f"\n  {'labels':>7s} {'scratch':>16s} {'Init':>16s} {'MAML':>16s} {'MAML-scratch':>13s} "
          f"{'MAML-Init':>10s}")
    summary = {}
    for k in args.budgets:
        s, ss = m(results["scratch"][k]); i, isd = m(results["init"][k]); z, zs = m(results["maml"][k])
        summary[k] = dict(scratch=s, scratch_sd=ss, init=i, init_sd=isd, maml=z, maml_sd=zs,
                          maml_hard=float(np.nanmean(maml_hard[k])),
                          init_hard=float(np.nanmean(init_hard[k])),
                          scratch_hard=float(np.nanmean(scratch_hard[k])),
                          n=len(results["maml"][k]), adam_fallbacks=int(fallbacks[k]),
                          **{f"{a}_support": float(np.mean(sup_acc[a][k])) for a in ARMS})
        print(f"  {k:7d} {s:8.4f}+/-{ss:.3f} {i:8.4f}+/-{isd:.3f} {z:8.4f}+/-{zs:.3f} "
              f"{z - s:+13.4f} {z - i:+10.4f}")
    # ---------------------------------------------------------- the hard subset
    # Promoted to a primary metric in Milestone 6b. On the full distribution the
    # hand-distance shortcut scores ~0.88 and every arm lands within a few points
    # of it, so there is almost no headroom for a prior to show. The hard subset
    # is the quarter of pairs where that shortcut is least informative, and it is
    # where the 3-family prior was found to be actively worse than scratch --
    # below chance at the smallest budgets. That is the transfer question.
    print(f"\n{'=' * 78}\nHARD SUBSET  (shortcut uninformative; the primary transfer metric)")
    print(f"  heuristic on this subset: {np.mean(cal_hard):.4f}  "
          f"(vs {np.mean(cal_overall):.4f} on the full distribution)\n")
    print(f"  {'labels':>7s} {'scratch':>16s} {'Init':>16s} {'MAML':>16s} "
          f"{'MAML-scratch':>26s}")
    for k in args.budgets:
        sh, shs = m(scratch_hard[k]); ih, ihs = m(init_hard[k]); zh, zhs = m(maml_hard[k])
        md, sd_, t, pv, wins, n = paired_stats(maml_hard[k], scratch_hard[k])
        mdi, _, _, pvi, winsi, _ = paired_stats(maml_hard[k], init_hard[k])
        summary[k].update(scratch_hard_sd=shs, init_hard_sd=ihs, maml_hard_sd=zhs,
                          hard_maml_minus_scratch=md, hard_p=pv, hard_wins=wins, hard_n=n,
                          hard_maml_minus_init=mdi, hard_p_init=pvi, hard_wins_init=winsi)
        print(f"  {k:7d} {sh:8.4f}+/-{shs:.3f} {ih:8.4f}+/-{ihs:.3f} {zh:8.4f}+/-{zhs:.3f} "
              f"{md:+9.4f}  p={pv:.3f}  {wins}/{n} wins")
    print("\n  (paired over the same (variation, draw) support sets, so the comparison\n"
          "   removes between-variation variance; p is a two-sided paired t.)")
    print("\n  support-set accuracy reached (all arms stop at 0.95; a gap between support\n"
          "  and held-out accuracy is overfitting to the few labels):")
    for k in args.budgets:
        print(f"  {k:7d} " + "  ".join(
            f"{a} {np.mean(sup_acc[a][k]):.3f}->{summary[k][a]:.3f}" for a in ARMS))
    cb = float(np.nanmean(corr_before)); ca = float(np.nanmean(corr_after))
    print(f"\n  reward vs sash x-velocity correlation: before {cb:+.4f} -> after {ca:+.4f}")
    print(f"  MAML runs needing the Adam fallback: "
          f"{ {k: summary[k]['adam_fallbacks'] for k in args.budgets} }")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "few_shot_window_close.json"), "w") as f:
        json.dump(dict(summary=summary, meta_init=mi, meta_init_sd=mis,
                       heuristic=float(np.mean(cal_overall)),
                       heuristic_hard=float(np.mean(cal_hard)),
                       corr_before=cb, corr_after=ca,
                       n_variations=len(tasks), draws=args.draws), f, indent=2)

    # ------------------------------------------------------------------- checks
    print(f"\n{'=' * 78}\nSANITY (these fail the job)\n")
    allv = [v for a in ARMS for k in args.budgets for v in results[a][k]]
    check("all arms produced finite accuracies", all(np.isfinite(allv)),
          f"{sum(not np.isfinite(v) for v in allv)} non-finite of {len(allv)}")
    kmax = max(args.budgets)
    n_tot = int(summary[kmax]["n"] * args.eval_pairs)
    pv = binom_p(int(summary[kmax]["scratch"] * n_tot), n_tot)
    check(f"scratch at {kmax} labels is above chance (harness works)", pv < 1e-6,
          f"{summary[kmax]['scratch']:.4f}, p={pv:.3g}")
    # NOTE: "MAML beats scratch" deliberately does NOT live here. The plan filed it
    # as a sanity check on the grounds that Milestone 5 already showed it, so a
    # failure would mean a bug. That reasoning was wrong, and it is the same error
    # Milestones 2 and 4 made: encoding an expectation as a correctness test.
    # Milestone 5's held-out tasks were new goals of practiced families, where the
    # meta-init was already right. Window Close reverses the direction, so the init
    # is actively wrong, and "wrong beliefs are a worse starting point than none"
    # is a legitimate possible finding rather than a defect. It is H0 below.

    print(f"\n{'=' * 78}\nHYPOTHESES (findings, not pass/fail)\n")
    gaps0 = [summary[k]["maml"] - summary[k]["scratch"] for k in args.budgets]
    finding("H0 MAML beats from-scratch at every budget (the project's main question)",
            all(g > 0 for g in gaps0),
            " ".join(f"{k}:{g:+.3f}" for k, g in zip(args.budgets, gaps0)))
    # H1 is scored on BOTH distributions. With the 3-family prior it was refuted on
    # the full distribution (0.7136) and confirmed on the hard subset, where the
    # pretrained arms fell below chance. The prediction was right; the metric hid
    # it, because the hand-distance shortcut carried the full-distribution number.
    finding("H1 unadapted meta-init is at or below chance on Window Close "
            "(full distribution)", mi <= 0.50, f"{mi:.4f} +/- {mis:.4f}")
    kmin = min(args.budgets)
    hard_init0 = float(np.nanmean(init_hard[kmin]))
    finding(f"H1b pretrained arms are at or below chance on the HARD subset at "
            f"{kmin} labels", hard_init0 <= 0.50,
            f"Init {hard_init0:.4f}, MAML {float(np.nanmean(maml_hard[kmin])):.4f}, "
            f"scratch {float(np.nanmean(scratch_hard[kmin])):.4f}")

    # The transfer question, on the metric where it actually lives.
    hard_gaps = [summary[k]["hard_maml_minus_scratch"] for k in args.budgets]
    finding("H7 MAML beats scratch ON THE HARD SUBSET (the claim the 3-family prior "
            "failed most clearly)", all(g > 0 for g in hard_gaps),
            " ".join(f"{k}:{g:+.3f}(p={summary[k]['hard_p']:.2f})"
                     for k, g in zip(args.budgets, hard_gaps)))
    d_small = summary[min(args.budgets)]["maml"] - summary[min(args.budgets)]["init"]
    d_large = summary[kmax]["maml"] - summary[kmax]["init"]
    finding(f"H2 MAML beats Init at {min(args.budgets)} labels", d_small > 0, f"{d_small:+.4f}")
    finding("H2b the MAML-over-Init gap shrinks as labels grow", d_small > d_large,
            f"{d_small:+.4f} at {min(args.budgets)} vs {d_large:+.4f} at {kmax}")
    finding("H3 reward/velocity correlation flips sign after adaptation",
            (cb > 0) and (ca < 0), f"{cb:+.4f} -> {ca:+.4f}")
    finding(f"H4 MAML at {kmax} labels beats the hand-distance heuristic",
            summary[kmax]["maml"] > np.mean(cal_overall),
            f"{summary[kmax]['maml']:.4f} vs {np.mean(cal_overall):.4f}")
    gaps = [summary[k]["maml"] - summary[k]["scratch"] for k in args.budgets]
    finding("H5 MAML's advantage over scratch is largest at the smallest budget",
            gaps[0] == max(gaps), " ".join(f"{k}:{g:+.3f}" for k, g in zip(args.budgets, gaps)))

    print()
    if FAILS:
        print(f"SANITY FAILED ({len(FAILS)}):")
        for f in dict.fromkeys(FAILS):
            print(f"  - {f}")
        raise SystemExit(1)
    n_conf = sum(c for _, c, _ in FINDINGS)
    print(f"SANITY PASSED. {n_conf}/{len(FINDINGS)} hypotheses confirmed.")


if __name__ == "__main__":
    main()
