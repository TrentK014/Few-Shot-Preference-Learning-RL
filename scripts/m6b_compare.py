"""The 3-family versus 10-family comparison: did more prior tasks fix the transfer?

Milestone 6 held out Window Close from a 3-family prior and found no advantage
for MAML over training from scratch. The diagnosis
(`notes/06b-diagnosis.md`) attributed that to three causes, of which prior
breadth was the one that could be tested directly by rebuilding the prior set as
ML10's ten training families. This merges the Milestone 6 runs with the
Milestone 6b run and prints the contrast on both metrics that matter.

The full distribution is reported because it is what Milestone 6 reported. The
hard subset is reported because it is where the question actually lives: the
hand-distance shortcut scores ~0.88 on the full distribution and every arm lands
near it, whereas on the hard subset the 3-family prior was *below chance* and
worse than scratch.

    python scripts/m6b_compare.py \
        --three $FSPREF_ROOT/runs/m6_low/few_shot_window_close.json \
                $FSPREF_ROOT/runs/m6/few_shot_window_close.json \
        --ten   $FSPREF_ROOT/runs/m6b_wc/few_shot_window_close.json
"""
from __future__ import annotations

import argparse
import json

FINDINGS: list[tuple[str, bool, str]] = []


def finding(name, confirmed, detail=""):
    print(f"  [{'CONFIRMED' if confirmed else 'REFUTED  '}] {name}"
          f"{(': ' + detail) if detail else ''}")
    FINDINGS.append((name, confirmed, detail))
    return confirmed


def load(paths):
    """Merge several result files into one budget -> summary map.

    Milestone 6's budgets were split across two runs (25/50/100/200 and the
    4/8/16/25 follow-up), so the 3-family side needs both. Where they overlap at
    25 the later file wins, which is the low-budget run with more draws.
    """
    merged, meta = {}, {}
    for p in paths:
        with open(p) as f:
            blob = json.load(f)
        meta = blob  # last one wins for the scalar fields
        for k, v in blob["summary"].items():
            merged[int(k)] = v
    return merged, meta


def fmt(v, w=8, p=4):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--three", nargs="+", required=True,
                    help="Milestone 6 result json(s), 3-family prior")
    ap.add_argument("--ten", nargs="+", required=True,
                    help="Milestone 6b result json(s), 10-family prior")
    ap.add_argument("--label-three", default="3-family")
    ap.add_argument("--label-ten", default="10-family")
    args = ap.parse_args()

    three, m3 = load(args.three)
    ten, m10 = load(args.ten)
    budgets = sorted(set(three) & set(ten))
    if not budgets:
        raise SystemExit(f"no shared budgets: {sorted(three)} vs {sorted(ten)}")

    print(f"{'=' * 78}\nUNADAPTED META-INIT ON WINDOW CLOSE\n")
    print(f"  {args.label_three:<12s} {m3.get('meta_init', float('nan')):.4f} "
          f"+/- {m3.get('meta_init_sd', float('nan')):.4f}")
    print(f"  {args.label_ten:<12s} {m10.get('meta_init', float('nan')):.4f} "
          f"+/- {m10.get('meta_init_sd', float('nan')):.4f}")
    print(f"  heuristic    {m10.get('heuristic', float('nan')):.4f} "
          f"(hard subset {m10.get('heuristic_hard', float('nan')):.4f})")

    # ---------------------------------------------------------- full distribution
    print(f"\n{'=' * 78}\nMAML - SCRATCH, FULL DISTRIBUTION\n")
    print(f"  {'labels':>7s} {args.label_three:>12s} {args.label_ten:>12s} "
          f"{'change':>10s}")
    for k in budgets:
        d3 = three[k]["maml"] - three[k]["scratch"]
        d10 = ten[k]["maml"] - ten[k]["scratch"]
        print(f"  {k:7d} {d3:+12.4f} {d10:+12.4f} {d10 - d3:+10.4f}")

    # ----------------------------------------------------------------- hard subset
    print(f"\n{'=' * 78}\nMAML - SCRATCH, HARD SUBSET  (the transfer question)\n")
    print(f"  {'labels':>7s} {args.label_three:>12s} {args.label_ten:>12s} "
          f"{'change':>10s} {'p (10-fam)':>11s}")
    for k in budgets:
        d3 = three[k].get("hard_maml_minus_scratch",
                          three[k].get("maml_hard", float("nan"))
                          - three[k].get("scratch_hard", float("nan")))
        d10 = ten[k].get("hard_maml_minus_scratch",
                         ten[k].get("maml_hard", float("nan"))
                         - ten[k].get("scratch_hard", float("nan")))
        pv = ten[k].get("hard_p", float("nan"))
        print(f"  {k:7d} {d3:+12.4f} {d10:+12.4f} {d10 - d3:+10.4f} {pv:11.3f}")

    print(f"\n  absolute hard-subset accuracy (scratch / MAML):")
    print(f"  {'labels':>7s} {args.label_three:>20s} {args.label_ten:>20s}")
    for k in budgets:
        print(f"  {k:7d}   {three[k].get('scratch_hard', float('nan')):.4f} / "
              f"{three[k].get('maml_hard', float('nan')):.4f}        "
              f"{ten[k].get('scratch_hard', float('nan')):.4f} / "
              f"{ten[k].get('maml_hard', float('nan')):.4f}")

    # ------------------------------------------------------------------- findings
    print(f"\n{'=' * 78}\nFINDINGS\n")
    small = [k for k in budgets if k <= 16] or budgets[:1]

    g3 = [three[k]["maml"] - three[k]["scratch"] for k in small]
    g10 = [ten[k]["maml"] - ten[k]["scratch"] for k in small]
    finding("H6 the 10-family prior beats the 3-family prior at the small budgets",
            sum(g10) / len(g10) > sum(g3) / len(g3),
            f"mean MAML-scratch over {small}: {sum(g3) / len(g3):+.4f} -> "
            f"{sum(g10) / len(g10):+.4f}")

    h10 = [ten[k].get("hard_maml_minus_scratch", float("nan")) for k in budgets]
    finding("H7 MAML beats scratch on the hard subset under the 10-family prior",
            all(v > 0 for v in h10 if v == v),
            " ".join(f"{k}:{v:+.3f}" for k, v in zip(budgets, h10)))

    kmin = min(budgets)
    hard_init3 = three[kmin].get("init_hard", float("nan"))
    hard_init10 = ten[kmin].get("init_hard", float("nan"))
    finding("H1b the pretrained arms are no longer below chance on the hard subset",
            hard_init10 > 0.5,
            f"Init at {kmin} labels: {hard_init3:.4f} ({args.label_three}) -> "
            f"{hard_init10:.4f} ({args.label_ten}); chance is 0.50")

    print()
    n = sum(c for _, c, _ in FINDINGS)
    print(f"{n}/{len(FINDINGS)} findings confirmed.")
    print("\nA refuted finding here is a result, not a bug. If prior breadth does not\n"
          "close the hard-subset gap, the remaining explanation is that Window Open's\n"
          "reversed direction dominates regardless of how many other families are\n"
          "present -- which is testable on the other ML10 test tasks.")


if __name__ == "__main__":
    main()
