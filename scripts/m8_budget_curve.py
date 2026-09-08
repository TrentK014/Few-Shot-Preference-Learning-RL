"""Milestone 8: success rate against total query budget.

Milestone 7 showed the method works at 200 queries and that few-shot needed the
fewest of them. The interesting question is the other direction: how far can the
budget be cut before each method breaks? At 200 every method eventually solves
Window Close, so the arms bunch up and the comparison says little. The low end is
where a reward-function prior cannot be substituted for by asking more questions.

Reads one run directory per budget (the Milestone 8 sweep plus Milestone 7's
200-query point) and prints the curve, plus where each arm falls off.

    python scripts/m8_budget_curve.py \
        --budget 25  $FSPREF_ROOT/runs/m8_b25 \
        --budget 50  $FSPREF_ROOT/runs/m8_b50 \
        --budget 100 $FSPREF_ROOT/runs/m8_b100 \
        --budget 200 $FSPREF_ROOT/runs/m7
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ARMS = ["few_shot", "init", "pebble"]
FINDINGS: list[tuple[str, bool, str]] = []


def finding(name, confirmed, detail=""):
    print(f"  [{'CONFIRMED' if confirmed else 'REFUTED  '}] {name}"
          f"{(': ' + detail) if detail else ''}")
    FINDINGS.append((name, confirmed, detail))


def load_dir(path: str):
    """arm -> list of runs, from a directory of shard jsons."""
    out: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(os.path.join(path, "m7_results_*.json"))):
        with open(p) as f:
            blob = json.load(f)
        for arm, runs in blob["results"].items():
            out.setdefault(arm, []).extend(runs)
    return out


class BudgetAction(argparse.Action):
    def __call__(self, parser, ns, values, option_string=None):
        items = getattr(ns, self.dest) or []
        items.append((int(values[0]), values[1]))
        setattr(ns, self.dest, items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", nargs=2, action=BudgetAction, dest="budgets",
                    metavar=("N", "DIR"), required=True,
                    help="repeatable: total query budget and its run directory")
    ap.add_argument("--oracle", type=float, default=1.0,
                    help="ground-truth-reward ceiling, from Milestone 7")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="success rate counted as solving the task")
    args = ap.parse_args()

    budgets = sorted(args.budgets)
    data = {n: load_dir(d) for n, d in budgets}

    # ------------------------------------------------------------ final success
    print(f"{'=' * 74}\nFINAL SUCCESS RATE vs TOTAL QUERY BUDGET")
    print(f"  ground-truth-reward ceiling: {args.oracle:.3f}\n")
    print(f"  {'budget':>7s} " + "".join(f"{a:>20s}" for a in ARMS))
    final = {a: {} for a in ARMS}
    for n, _ in budgets:
        row = f"  {n:7d} "
        for a in ARMS:
            runs = data[n].get(a, [])
            if runs:
                v = np.array([r["final_success"] for r in runs])
                final[a][n] = (float(v.mean()), float(v.std()), len(v))
                row += f"{v.mean():11.3f}+/-{v.std():<6.3f}"
            else:
                row += f"{'-':>20s}"
        print(row)

    # --------------------------------------------------------- solved fraction
    print(f"\n{'=' * 74}\nSEEDS REACHING {args.threshold:.0%} SUCCESS\n")
    print(f"  {'budget':>7s} " + "".join(f"{a:>20s}" for a in ARMS))
    solved = {a: {} for a in ARMS}
    for n, _ in budgets:
        row = f"  {n:7d} "
        for a in ARMS:
            runs = data[n].get(a, [])
            if runs:
                k = sum(any(c[1] >= args.threshold for c in r["curve"]) for r in runs)
                solved[a][n] = (k, len(runs))
                row += f"{f'{k}/{len(runs)}':>20s}"
            else:
                row += f"{'-':>20s}"
        print(row)

    # -------------------------------------------------------------- the margin
    print(f"\n{'=' * 74}\nFEW-SHOT MINUS PEBBLE, FINAL SUCCESS\n")
    print(f"  {'budget':>7s} {'few_shot':>12s} {'pebble':>12s} {'margin':>10s}")
    margins = {}
    for n, _ in budgets:
        if n in final["few_shot"] and n in final["pebble"]:
            f_, p_ = final["few_shot"][n][0], final["pebble"][n][0]
            margins[n] = f_ - p_
            print(f"  {n:7d} {f_:12.3f} {p_:12.3f} {f_ - p_:+10.3f}")

    # --------------------------------------------------------------- findings
    print(f"\n{'=' * 74}\nFINDINGS\n")
    ns = [n for n, _ in budgets if n in margins]
    if len(ns) >= 2:
        lo, hi = ns[0], ns[-1]
        finding(f"H13 few-shot's advantage over PEBBLE widens as the budget shrinks",
                margins[lo] > margins[hi],
                f"{margins[lo]:+.3f} at {lo} queries vs {margins[hi]:+.3f} at {hi}")
    lo = ns[0] if ns else None
    if lo is not None and lo in solved["pebble"]:
        k, tot = solved["pebble"][lo]
        finding(f"H14 PEBBLE fails at the smallest budget ({lo} queries)", k == 0,
                f"{k}/{tot} seeds reached {args.threshold:.0%}")
    if lo is not None and lo in final["few_shot"]:
        v = final["few_shot"][lo][0]
        finding(f"H15 few-shot still reaches the ceiling at {lo} queries",
                v >= args.oracle - 1e-9,
                f"{v:.3f} vs ceiling {args.oracle:.3f}")

    print()
    n_conf = sum(c for _, c, _ in FINDINGS)
    print(f"{n_conf}/{len(FINDINGS)} findings confirmed.")
    print("\nA refuted finding is a result. If the margin does NOT widen as the budget\n"
          "shrinks, the prior is not buying query efficiency on this task, and the\n"
          "Milestone 7 ordering was about stability rather than sample efficiency.")


if __name__ == "__main__":
    main()
