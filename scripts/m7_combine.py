"""Merge Milestone 7 array-job shards into one report.

`carc/m7_array.sbatch` runs each (arm, seed) as its own task so the twelve runs
go in parallel, and each shard writes `m7_results_<arm>_s<seed>.json`. This
recombines them and prints the same report and verification the single-process
run prints, so the two paths produce identical output.

    python scripts/m7_combine.py --out $FSPREF_ROOT/runs/m7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FAILS: list[str] = []
FINDINGS: list[tuple[str, bool, str]] = []
PREFERENCE_ARMS = ("few_shot", "init", "init_reset", "pebble")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def finding(name, confirmed, detail=""):
    print(f"  [{'CONFIRMED' if confirmed else 'REFUTED  '}] {name}"
          f"{(': ' + detail) if detail else ''}")
    FINDINGS.append((name, confirmed, detail))
    return confirmed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/m7")
    p.add_argument("--expect-seeds", type=int, default=3)
    p.add_argument("--oracle-floor", type=float, default=0.5)
    p.add_argument("--success-threshold", type=float, default=1.0,
                   help="success rate that counts as solving the task, for the "
                        "query-efficiency comparison")
    args = p.parse_args()

    shards = sorted(glob.glob(os.path.join(args.out, "m7_results_*.json")))
    if not shards:
        raise SystemExit(f"no shards under {args.out}")

    results: dict[str, list[dict]] = {}
    max_feedback = None
    for path in shards:
        with open(path) as f:
            blob = json.load(f)
        max_feedback = blob["args"].get("max_feedback", max_feedback)
        for arm, runs in blob["results"].items():
            results.setdefault(arm, []).extend(runs)

    arms = [a for a in ("sac_oracle", "few_shot", "init", "init_reset", "pebble")
            if a in results]
    print(f"merged {len(shards)} shards | arms {arms} | "
          f"{ {a: len(results[a]) for a in arms} } runs")

    print("\n" + "=" * 78 + "\nRESULTS  (success rate, mean +/- sd over seeds)\n")
    summary = {}
    print(f"  {'arm':<12s} {'final':>16s} {'best':>16s} {'feedback':>10s} {'seeds':>6s}")
    for arm in arms:
        fin = np.array([r["final_success"] for r in results[arm]])
        best = np.array([r["best_success"] for r in results[arm]])
        fb = int(np.mean([r["total_feedback"] for r in results[arm]]))
        summary[arm] = dict(final_mean=float(fin.mean()), final_sd=float(fin.std()),
                            best_mean=float(best.mean()), best_sd=float(best.std()),
                            feedback=fb, n_seeds=len(fin))
        print(f"  {arm:<12s} {fin.mean():8.3f}+/-{fin.std():<6.3f} "
              f"{best.mean():8.3f}+/-{best.std():<6.3f} {fb:10d} {len(fin):6d}")

    # ------------------------------------------------------- query efficiency
    # The discriminating metric. Several arms saturate at 1.00 final success on
    # Window Close, so "did it solve the task" separates nothing; the paper's
    # claim is about how much feedback it takes to get there. For each seed, the
    # first evaluation reaching `--success-threshold`, and the feedback spent by
    # that point. Seeds that never reach it are reported separately rather than
    # folded in as a large number, which would silently reward failure.
    print(f"\n  query efficiency: first evaluation reaching "
          f"{args.success_threshold:.0%} success\n")
    print(f"  {'arm':<12s} {'steps':>18s} {'queries':>18s} {'solved':>8s}")
    for arm in arms:
        steps_to, q_to = [], []
        for r in results[arm]:
            hit = next((c for c in r["curve"] if c[1] >= args.success_threshold), None)
            if hit:
                steps_to.append(hit[0]); q_to.append(hit[2])
        n_solved = len(steps_to)
        summary[arm]["n_solved"] = n_solved
        if n_solved:
            summary[arm]["steps_to_success"] = float(np.mean(steps_to))
            summary[arm]["queries_to_success"] = float(np.mean(q_to))
            solved = f"{n_solved}/{len(results[arm])}"
            print(f"  {arm:<12s} {np.mean(steps_to):10.0f}+/-{np.std(steps_to):<7.0f}"
                  f"{np.mean(q_to):11.1f}+/-{np.std(q_to):<7.1f}{solved:>8s}")
        else:
            solved = f"0/{len(results[arm])}"
            print(f"  {arm:<12s} {'never':>18s}{'never':>19s}{solved:>8s}")

    # Success-rate curves, so the query-efficiency claim can be read off directly.
    print("\n  success rate vs environment steps (mean over seeds)\n")
    steps = sorted({c[0] for arm in arms for r in results[arm] for c in r["curve"]})
    show = steps[:: max(1, len(steps) // 12)]
    print("  " + "step".ljust(12) + "".join(f"{a:>12s}" for a in arms))
    for st in show:
        row = f"  {st:<12d}"
        for arm in arms:
            vals = [c[1] for r in results[arm] for c in r["curve"] if c[0] == st]
            row += f"{np.mean(vals):12.3f}" if vals else f"{'-':>12s}"
        print(row)

    with open(os.path.join(args.out, "m7_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78 + "\nSANITY (these fail the job)\n")
    for arm in arms:
        check(f"{arm} has all {args.expect_seeds} seeds",
              summary[arm]["n_seeds"] >= args.expect_seeds,
              f"{summary[arm]['n_seeds']} present")
    if "sac_oracle" in summary:
        check("SAC on ground-truth reward solves the task",
              summary["sac_oracle"]["best_mean"] >= args.oracle_floor,
              f"best success {summary['sac_oracle']['best_mean']:.3f} >= {args.oracle_floor} "
              f"(if this fails the RL is broken, not the reward model)")
    if max_feedback:
        for arm in arms:
            if arm in PREFERENCE_ARMS:
                check(f"{arm} stayed within the query budget",
                      summary[arm]["feedback"] <= max_feedback,
                      f"{summary[arm]['feedback']} <= {max_feedback}")

    print("\n" + "=" * 78 + "\nHYPOTHESES (findings, not pass/fail)\n")
    if "few_shot" in summary and "pebble" in summary:
        d = summary["few_shot"]["final_mean"] - summary["pebble"]["final_mean"]
        finding("H10 few-shot beats PEBBLE on final success at the same query budget",
                d > 0,
                f"{summary['few_shot']['final_mean']:.3f} vs "
                f"{summary['pebble']['final_mean']:.3f} ({d:+.3f}) at {max_feedback} queries")
        # The sharper form, and the one the paper actually argues: same
        # performance, fewer queries. Only meaningful if both arms solve it.
        fq = summary["few_shot"].get("queries_to_success")
        pq = summary["pebble"].get("queries_to_success")
        if fq is not None and pq is not None:
            finding("H10b few-shot needs FEWER queries than PEBBLE to solve the task "
                    "(the paper's central claim)", fq < pq,
                    f"{fq:.1f} vs {pq:.1f} queries "
                    f"({summary['few_shot']['n_solved']}/{summary['pebble']['n_solved']} "
                    f"seeds solved)")
        else:
            finding("H10b few-shot needs FEWER queries than PEBBLE to solve the task",
                    fq is not None and pq is None,
                    f"few_shot solved {summary['few_shot'].get('n_solved', 0)} seeds, "
                    f"pebble solved {summary['pebble'].get('n_solved', 0)}")
    if "few_shot" in summary and "init" in summary:
        d = summary["few_shot"]["final_mean"] - summary["init"]["final_mean"]
        finding("H11 re-adaptation beats plain fine-tuning (few_shot > Init)", d > 0,
                f"{summary['few_shot']['final_mean']:.3f} vs "
                f"{summary['init']['final_mean']:.3f} ({d:+.3f})")
    if "few_shot" in summary and "sac_oracle" in summary:
        ratio = (summary["few_shot"]["final_mean"]
                 / max(summary["sac_oracle"]["final_mean"], 1e-9))
        finding("H12 few-shot approaches the ground-truth-reward ceiling", ratio >= 0.8,
                f"{summary['few_shot']['final_mean']:.3f} vs oracle "
                f"{summary['sac_oracle']['final_mean']:.3f} ({100 * ratio:.0f}% of ceiling)")

    print()
    if FAILS:
        print(f"SANITY FAILED ({len(FAILS)}):")
        for f_ in dict.fromkeys(FAILS):
            print(f"  - {f_}")
        raise SystemExit(1)
    n_conf = sum(c for _, c, _ in FINDINGS)
    print(f"SANITY PASSED. {n_conf}/{len(FINDINGS)} hypotheses confirmed.")


if __name__ == "__main__":
    main()
