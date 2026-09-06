"""Milestone 3: train a preference reward model with plain Adam. No MAML, no SAC.

Proves the Bradley-Terry loss, the architecture and the data pipeline work, so a
Milestone 5 failure is attributable to MAML alone.

    python scripts/m3_train_reward.py --root $FSPREF_ROOT/data/prefs --task window-open \
        --mode per-variation --variations 5
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.baselines import calibration_report  # noqa: E402
from fspref.data import PreferenceData, VariationData  # noqa: E402
from fspref.prefs import SEGMENT_SIZE, list_datasets  # noqa: E402
from fspref.reward_model import (RewardEnsemble, preference_accuracy,  # noqa: E402
                                 preference_loss)

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def binomial_p(correct: int, n: int) -> float:
    """One-sided p-value against a fair coin, normal approximation (n is large here)."""
    if n == 0:
        return 1.0
    z = (correct - 0.5 * n) / math.sqrt(0.25 * n)
    return 0.5 * math.erfc(z / math.sqrt(2))


@torch.no_grad()
def untrained_accuracy(var: VariationData, a, b, y, args, n_seeds: int = 8):
    """Mean accuracy of randomly initialized models over several seeds.

    A single untrained draw is NOT at chance: a random smooth function of state
    correlates with the true reward by luck, and one seed measured 0.73. Only the
    mean over seeds is meaningful as a no-label-leakage control.
    """
    accs = []
    for s in range(n_seeds):
        torch.manual_seed(10_000 + s)
        m = RewardEnsemble(ensemble_size=args.ensemble).to(args.device)
        _, ens, _ = evaluate(m, var, a, b, y)
        accs.append(ens)
    return float(np.mean(accs)), float(np.std(accs))


@torch.no_grad()
def evaluate(model, var: VariationData, a, b, y, mask=None):
    """(per-member accuracy, ensemble accuracy, n) on one set of pairs."""
    if mask is not None:
        a, b, y = a[mask], b[mask], y[mask]
    if len(y) == 0:
        return float("nan"), float("nan"), 0
    obs_a, act_a, obs_b, act_b, yt = var.batch(a, b, y)
    logits = model(obs_a, act_a, obs_b, act_b)
    per, ens = preference_accuracy(logits, yt)
    return per, ens, len(y)


def validation_accuracy(model, data: PreferenceData, args):
    """Mean accuracy on fresh pairs from the VALIDATION episodes of each variation.

    Disjoint from both the training pairs and the reported test episodes.
    """
    accs, ns = [], []
    for var in data.variations:
        a, b, y = var.episode_holdout_pairs(n_pairs=args.val_pairs, seed=args.seed + 3,
                                            which="val")
        _, ens, n = evaluate(model, var, a, b, y)
        accs.append(ens * n); ns.append(n)
    return float(sum(accs) / max(sum(ns), 1))


def train_model(data: PreferenceData, args, seed: int = 0, log_prefix: str = ""):
    """Train with early stopping on held-out validation episodes.

    The reference stops when TRAINING accuracy reaches 0.95 and has no validation
    split at all. That rule is calibrated for online PEBBLE, where the buffer
    holds 8 to 200 pairs and 0.95 train accuracy means near-memorization. Offline
    with ~4,600 pairs it fires at epoch 8 while the model is still underfit:
    measured held-out-episode accuracy 0.907 at that point versus 0.948 after 120
    epochs, and 0.672 versus 0.806 on the hard subset. Pass --stop-rule reference
    to reproduce its behavior exactly.
    """
    torch.manual_seed(seed)
    model = RewardEnsemble(ensemble_size=args.ensemble).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    best = (-1.0, -1, None)  # (val acc, epoch, state)
    since_best = 0
    t0 = time.time()
    for epoch in range(args.max_epochs):
        accs, losses = [], []
        for obs_a, act_a, obs_b, act_b, y in data.iter_batches(
                "train", batch_size=args.batch_size, seed=seed * 1000 + epoch):
            logits = model(obs_a, act_a, obs_b, act_b)
            loss = preference_loss(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            per, _ = preference_accuracy(logits.detach(), y)
            accs.append(per); losses.append(loss.item())
        mean_acc, mean_loss = float(np.mean(accs)), float(np.mean(losses))

        if args.stop_rule == "reference":
            history.append((mean_loss, mean_acc, float("nan")))
            if epoch < 3 or (epoch + 1) % args.log_every == 0:
                print(f"    {log_prefix}epoch {epoch + 1:4d}  loss {mean_loss:.4f}  "
                      f"train_acc {mean_acc:.4f}")
            if mean_acc >= args.target_acc:
                print(f"    {log_prefix}stopped at epoch {epoch + 1} "
                      f"(train_acc {mean_acc:.4f} >= {args.target_acc}, reference rule) "
                      f"in {time.time() - t0:.1f}s")
                break
            continue

        val_acc = validation_accuracy(model, data, args)
        history.append((mean_loss, mean_acc, val_acc))
        if val_acc > best[0] + args.min_delta:
            best = (val_acc, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            since_best = 0
        else:
            since_best += 1
        if epoch < 3 or (epoch + 1) % args.log_every == 0:
            print(f"    {log_prefix}epoch {epoch + 1:4d}  loss {mean_loss:.4f}  "
                  f"train_acc {mean_acc:.4f}  val_acc {val_acc:.4f}")
        if since_best >= args.patience:
            print(f"    {log_prefix}early stop at epoch {epoch + 1}; best val_acc "
                  f"{best[0]:.4f} at epoch {best[1] + 1} ({time.time() - t0:.1f}s)")
            break
    else:
        print(f"    {log_prefix}hit max_epochs={args.max_epochs} "
              f"(train_acc {history[-1][1]:.4f}) in {time.time() - t0:.1f}s")
    if args.stop_rule != "reference" and best[2] is not None:
        model.load_state_dict(best[2])
    return model, history


def report_variation(model, var: VariationData, args, untrained=None):
    """Evaluate one variation on all splits and against the measured heuristic bars."""
    out = {}
    for split in ("train", "val", "test"):
        a, b, y = var.stored_pairs(split)
        _, ens, n = evaluate(model, var, a, b, y)
        out[f"pairs_{split}"] = (ens, n)

    # The honest split: fresh pairs from episodes the model never trained on.
    a, b, y = var.episode_holdout_pairs(n_pairs=args.eval_pairs, seed=args.seed + 7,
                                        which="test")
    per, ens, n = evaluate(model, var, a, b, y)
    cal = calibration_report(var.obs_np, a, b, y, var.segment_size, args.hard_quantile)
    _, ens_hard, n_hard = evaluate(model, var, a, b, y, mask=cal["hard_mask"])
    out.update(episode_holdout=(ens, n), episode_holdout_per_member=per,
               episode_hard=(ens_hard, n_hard), cal=cal)
    if untrained:
        u_mean, u_std = untrained_accuracy(var, a, b, y, args)
        out["untrained"] = u_mean
        out["untrained_std"] = u_std
    return out


def print_variation_report(key, r):
    cal = r["cal"]
    print(f"\n  {key}")
    print(f"    stored pairs   train {r['pairs_train'][0]:.4f}  val {r['pairs_val'][0]:.4f}  "
          f"test {r['pairs_test'][0]:.4f}  (n_test={r['pairs_test'][1]})")
    print(f"    held-out episodes (honest split, n={r['episode_holdout'][1]}):")
    print(f"      model    {r['episode_holdout'][0]:.4f}   vs heuristic "
          f"{cal['hand_dist_acc']:.4f} (best feature '{cal['best_feature']}' "
          f"{cal['best_feature_acc']:.4f})")
    print(f"      hard subset (n={cal['n_hard']}): model "
          f"{r['episode_hard'][0]:.4f}   vs heuristic {cal['hard_heuristic_acc']:.4f}")
    if "untrained" in r:
        print(f"      untrained control {r['untrained']:.4f} "
              f"(mean over seeds, sd {r['untrained_std']:.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/prefs")
    p.add_argument("--task", default="window-open")
    p.add_argument("--mode", choices=["per-variation", "pooled"], default="per-variation")
    p.add_argument("--variations", type=int, default=5, help="how many variations to use")
    p.add_argument("--held-out-variations", type=int, default=5,
                   help="pooled mode: variations reserved for generalization testing")
    p.add_argument("--lr", type=float, default=3e-4, help="reference pebble.yaml value")
    p.add_argument("--batch-size", type=int, default=64,
                   help="reference effective value; its config says 256 but that field is dead code")
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--stop-rule", choices=["val", "reference"], default="val",
                   help="'val' early-stops on held-out validation episodes (default); "
                        "'reference' reproduces PEBBLE's train-accuracy rule")
    p.add_argument("--target-acc", type=float, default=0.95,
                   help="reference stopping rule threshold, used only with --stop-rule reference")
    p.add_argument("--patience", type=int, default=15, help="epochs without val improvement")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--val-pairs", type=int, default=1000)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-pairs", type=int, default=2000)
    p.add_argument("--hard-quantile", type=float, default=0.25)
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = list_datasets(args.root, args.task)
    if not paths:
        raise SystemExit(f"no datasets under {os.path.join(args.root, args.task)}")
    print(f"mode={args.mode} task={args.task} device={args.device} "
          f"lr={args.lr} batch={args.batch_size} ensemble={args.ensemble} "
          f"stop_rule={args.stop_rule}"
          + (f" target_acc={args.target_acc}" if args.stop_rule == "reference"
             else f" patience={args.patience} max_epochs={args.max_epochs}"))

    results = []
    if args.mode == "per-variation":
        use = paths[:args.variations]
        print(f"training one model per variation, {len(use)} variations\n")
        for path in use:
            data = PreferenceData([path], args.device, SEGMENT_SIZE, args.holdout_frac, args.seed)
            var = data.variations[0]
            print(f"  {var.key}: {len(var.pair_splits['train'])} train pairs, "
                  f"{len(var.val_episodes)} val + {len(var.test_episodes)} test episodes held out")
            model, _ = train_model(data, args, seed=args.seed, log_prefix=f"{var.key} ")
            r = report_variation(model, var, args, untrained=True)
            r["key"] = var.key
            results.append(r)
            print_variation_report(var.key, r)
    else:
        n_train = max(1, args.variations - args.held_out_variations)
        train_paths, held_paths = paths[:n_train], paths[n_train:args.variations]
        print(f"pooled model over {len(train_paths)} variations, "
              f"{len(held_paths)} variations held out entirely\n")
        data = PreferenceData(train_paths, args.device, SEGMENT_SIZE, args.holdout_frac, args.seed)
        model, _ = train_model(data, args, seed=args.seed)
        for var in data.variations:
            r = report_variation(model, var, args, untrained=True)
            r["key"] = var.key
            results.append(r)
            print_variation_report(var.key, r)
        if held_paths:
            print("\n  --- variations held out of training entirely ---")
            held = PreferenceData(held_paths, args.device, SEGMENT_SIZE, args.holdout_frac, args.seed)
            for var in held.variations:
                r = report_variation(model, var, args, untrained=True)
                r["key"] = var.key + " (unseen variation)"
                r["unseen"] = True
                results.append(r)
                print_variation_report(r["key"], r)

    # ------------------------------------------------------------------ checks
    print("\n" + "=" * 72 + "\nVERIFICATION\n")
    seen = [r for r in results if not r.get("unseen")]

    print("Untrained control")
    u = np.array([r["untrained"] for r in seen if "untrained" in r])
    check("untrained models average to chance (0.40-0.60)", 0.40 <= u.mean() <= 0.60,
          f"mean {u.mean():.4f} over {len(u)} variations x 8 seeds")

    print("\nBeats the shortcut (held-out-episode split)")
    m = np.array([r["episode_holdout"][0] for r in seen])
    h = np.array([r["cal"]["hand_dist_acc"] for r in seen])
    bf = np.array([r["cal"]["best_feature_acc"] for r in seen])
    print(f"  model {m.mean():.4f} | hand-distance heuristic {h.mean():.4f} | "
          f"best single feature {bf.mean():.4f}")
    check("model beats the hand-distance heuristic", m.mean() > h.mean(),
          f"{m.mean():.4f} vs {h.mean():.4f}")
    check("model beats the best single feature", m.mean() > bf.mean(),
          f"{m.mean():.4f} vs {bf.mean():.4f}")

    print("\nHard subset (shortcut uninformative)")
    mh = np.array([r["episode_hard"][0] for r in seen])
    hh = np.array([r["cal"]["hard_heuristic_acc"] for r in seen])
    print(f"  model {mh.mean():.4f} | heuristic {hh.mean():.4f}")
    check("model beats the heuristic on the hard subset", mh.mean() > hh.mean(),
          f"{mh.mean():.4f} vs {hh.mean():.4f}")

    print("\nSignificance")
    n_tot = sum(r["episode_holdout"][1] for r in seen)
    corr = int(sum(r["episode_holdout"][0] * r["episode_holdout"][1] for r in seen))
    pv = binomial_p(corr, n_tot)
    check("above chance at p < 1e-6", pv < 1e-6, f"p={pv:.3g} on {n_tot} pairs")

    print("\nEnsemble")
    pm = np.array([r["episode_holdout_per_member"] for r in seen])
    # Averaging member logits does NOT mathematically guarantee accuracy at or
    # above the mean member's, so this is a sanity bound, not a correctness
    # property. A measured run came in 0.0003 below, which is two pairs in 6000.
    check("ensemble accuracy within 1 point of mean single-member accuracy",
          m.mean() >= pm.mean() - 0.01,
          f"ensemble {m.mean():.4f} vs per-member {pm.mean():.4f} "
          f"({m.mean() - pm.mean():+.4f})")

    print("\nOverfitting")
    tr = np.array([r["pairs_train"][0] for r in seen])
    va = np.array([r["pairs_val"][0] for r in seen])
    print(f"  stored-pair train {tr.mean():.4f} | val {va.mean():.4f} | gap {tr.mean() - va.mean():+.4f}")
    print(f"  held-out-episode accuracy {m.mean():.4f} is the number that matters; the "
          f"stored-pair gap understates\n  generalization difficulty because those pairs reuse "
          f"segments seen in training.")

    unseen = [r for r in results if r.get("unseen")]
    if unseen:
        print("\nUnseen variations (pooled mode)")
        mu = np.array([r["episode_holdout"][0] for r in unseen])
        hu = np.array([r["cal"]["hand_dist_acc"] for r in unseen])
        print(f"  model {mu.mean():.4f} | heuristic {hu.mean():.4f} over {len(unseen)} variations")
        check("generalizes to unseen variations better than the shortcut", mu.mean() > hu.mean(),
              f"{mu.mean():.4f} vs {hu.mean():.4f}")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s):")
        for f in dict.fromkeys(FAILS):
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
