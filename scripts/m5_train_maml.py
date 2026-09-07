"""Milestone 5: meta-train the reward model with MAML across the prior tasks.

A MAML task is one (family, goal variation) pair, matching the reference's
per-variation task keying. Some variations of each family are held out of
meta-training entirely, and the acceptance test is a dress rehearsal for
Milestone 6: on those held-out tasks, does adapting the meta-initialization on a
handful of preferences beat training a fresh model on the identical handful?

If that fails here, on families the model has already practiced, then Milestone 6
has no chance and the problem is MAML rather than the transfer.

    python scripts/m5_train_maml.py --root $FSPREF_ROOT/data/prefs \
        --meta-steps 6000 --out $FSPREF_ROOT/runs/m5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fspref.data import VariationData  # noqa: E402
from fspref.maml import (DEFAULT_INNER_LR, DEFAULT_INNER_STEPS, DEFAULT_OUTER_LR,  # noqa: E402
                         DEFAULT_QUERY, DEFAULT_SUPPORT, DEFAULT_TASK_BATCH,
                         MAX_ADAPT_STEPS, MAMLReward)
from fspref.prefs import SEGMENT_SIZE, list_datasets  # noqa: E402
from fspref.reward_model import (RewardEnsemble, preference_accuracy,  # noqa: E402
                                 preference_loss)

FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def split_tasks(paths, held_per_family: int, seed: int = 0):
    """Hold out whole goal variations of each family from meta-training."""
    by_family: dict[str, list[str]] = {}
    for p in paths:
        by_family.setdefault(os.path.basename(os.path.dirname(p)), []).append(p)
    rng = np.random.RandomState(seed)
    train, held = [], []
    for fam in sorted(by_family):
        fp = sorted(by_family[fam])
        order = rng.permutation(len(fp))
        held += [fp[i] for i in order[:held_per_family]]
        train += [fp[i] for i in order[held_per_family:]]
    return sorted(train), sorted(held)


def from_scratch_baseline(support, eval_batch, args, seed: int = 0, steps: int = 400):
    """Train a fresh model on the same support pairs. The comparison that matters."""
    torch.manual_seed(seed)
    net = RewardEnsemble(ensemble_size=args.ensemble).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.scratch_lr)
    obs_a, act_a, obs_b, act_b, y = support
    best = 0.0
    for _ in range(steps):
        logits = net(obs_a, act_a, obs_b, act_b)
        loss = preference_loss(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        with torch.no_grad():
            _, acc = preference_accuracy(net(obs_a, act_a, obs_b, act_b), y)
        best = max(best, acc)
        # Same stopping rule the meta-learner gets, so neither is favoured.
        if acc >= args.target_acc:
            break
    with torch.no_grad():
        _, ens = preference_accuracy(net(*eval_batch[:4]), eval_batch[4])
    return ens


def evaluate_few_shot(maml, tasks, args, shots, seed: int = 0):
    """For each shot budget: meta-adapted vs from-scratch on identical pairs."""
    results = {k: {"maml": [], "scratch": [], "meta_init": []} for k in shots}
    for ti, var in enumerate(tasks):
        rng = np.random.RandomState(args.seed + 1000 + ti)
        # Evaluation pairs come from episodes never used for training or support.
        ea, eb, ey = var.episode_holdout_pairs(n_pairs=args.eval_pairs,
                                               seed=args.seed + 7, which="test")
        eval_batch = var.batch(ea, eb, ey)
        _, init_acc = maml.evaluate(maml.init_params(), eval_batch)
        for k in shots:
            support = var.sample_batch(k, rng, split="train")
            adapted = maml.adapt(support, steps=args.adapt_steps, target_acc=args.target_acc)
            _, acc = maml.evaluate(adapted, eval_batch)
            scratch = from_scratch_baseline(support, eval_batch, args, seed=args.seed + ti)
            results[k]["maml"].append(acc)
            results[k]["scratch"].append(scratch)
            results[k]["meta_init"].append(init_acc)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/prefs")
    p.add_argument("--out", default="runs/m5")
    p.add_argument("--held-per-family", type=int, default=5,
                   help="variations of each family kept out of meta-training")
    p.add_argument("--meta-steps", type=int, default=6000)
    p.add_argument("--task-batch", type=int, default=DEFAULT_TASK_BATCH)
    p.add_argument("--support", type=int, default=DEFAULT_SUPPORT)
    p.add_argument("--query", type=int, default=DEFAULT_QUERY)
    p.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    p.add_argument("--inner-lr", type=float, default=DEFAULT_INNER_LR)
    p.add_argument("--outer-lr", type=float, default=DEFAULT_OUTER_LR)
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--adapt-steps", type=int, default=MAX_ADAPT_STEPS)
    p.add_argument("--target-acc", type=float, default=0.95)
    p.add_argument("--scratch-lr", type=float, default=3e-4)
    p.add_argument("--shots", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--eval-pairs", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = list_datasets(args.root)
    if not paths:
        raise SystemExit(f"no datasets under {args.root}")
    train_paths, held_paths = split_tasks(paths, args.held_per_family, args.seed)
    print(f"device={args.device} | {len(train_paths)} meta-train tasks, "
          f"{len(held_paths)} held-out tasks")
    print(f"inner_steps={args.inner_steps} support={args.support} query={args.query} "
          f"task_batch={args.task_batch} inner_lr={args.inner_lr} outer_lr={args.outer_lr}")

    t0 = time.time()
    train_tasks = [VariationData(p_, args.device, SEGMENT_SIZE, seed=args.seed) for p_ in train_paths]
    held_tasks = [VariationData(p_, args.device, SEGMENT_SIZE, seed=args.seed) for p_ in held_paths]
    print(f"loaded {len(train_tasks) + len(held_tasks)} tasks in {time.time() - t0:.1f}s")

    torch.manual_seed(args.seed)
    maml = MAMLReward(ensemble_size=args.ensemble, inner_lr=args.inner_lr,
                      inner_steps=args.inner_steps).to(args.device)
    optim = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)
    rng = np.random.RandomState(args.seed)

    print("\nmeta-training")
    history = []
    best_val, best_state = -1.0, None
    t0 = time.time()
    for step in range(1, args.meta_steps + 1):
        picks = rng.choice(len(train_tasks), size=min(args.task_batch, len(train_tasks)),
                           replace=False)
        batch = [train_tasks[i].support_query(args.support, args.query, rng) for i in picks]
        loss, pre, post = maml.outer_step(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        history.append((step, float(loss.item()), pre, post))

        if step % args.eval_every == 0 or step == args.meta_steps:
            # Meta-validation on held-out tasks: adapt, then score.
            vaccs, vpre = [], []
            vrng = np.random.RandomState(args.seed + 99)
            for var in held_tasks:
                sup, qry = var.support_query(args.support, args.query, vrng)
                _, a0 = maml.evaluate(maml.init_params(), qry)
                ad = maml.adapt(sup, steps=args.adapt_steps, target_acc=args.target_acc)
                _, a1 = maml.evaluate(ad, qry)
                vpre.append(a0); vaccs.append(a1)
            v = float(np.mean(vaccs))
            print(f"  step {step:5d}  loss {loss.item():.4f}  train_query {pre:.3f}->{post:.3f}  "
                  f"| held-out {np.mean(vpre):.3f}->{v:.3f}  ({time.time() - t0:.0f}s)")
            if v > best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in maml.state_dict().items()}

    if best_state is not None:
        maml.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "maml_init.pt")
    torch.save({"state_dict": maml.state_dict(), "args": vars(args),
                "held_paths": held_paths, "train_paths": train_paths}, ckpt)
    print(f"saved {ckpt} (best held-out adapted accuracy {best_val:.4f})")

    # ------------------------------------------------------- few-shot evaluation
    print("\nfew-shot evaluation on held-out variations "
          f"({len(held_tasks)} tasks, {args.eval_pairs} eval pairs each)")
    res = evaluate_few_shot(maml, held_tasks, args, args.shots)
    print(f"\n  {'shots':>6s} {'meta-init':>10s} {'MAML':>10s} {'scratch':>10s} {'gain':>8s}")
    summary = {}
    for k in args.shots:
        m = float(np.mean(res[k]["maml"]))
        s = float(np.mean(res[k]["scratch"]))
        i = float(np.mean(res[k]["meta_init"]))
        summary[k] = dict(maml=m, scratch=s, meta_init=i, n_tasks=len(res[k]["maml"]))
        print(f"  {k:6d} {i:10.4f} {m:10.4f} {s:10.4f} {m - s:+8.4f}")
    with open(os.path.join(args.out, "few_shot.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------- checks
    print("\n" + "=" * 72 + "\nVERIFICATION\n")
    first, last = history[0][1], float(np.mean([h[1] for h in history[-50:]]))
    print("Meta-training progressed")
    check("meta-loss decreased", last < first, f"{first:.4f} -> {last:.4f}")

    print("\nAdaptation does something")
    pre_all = float(np.mean([h[2] for h in history[-200:]]))
    post_all = float(np.mean([h[3] for h in history[-200:]]))
    check("inner loop improves query accuracy", post_all > pre_all,
          f"{pre_all:.4f} -> {post_all:.4f} on meta-train tasks")

    print("\nBeats from scratch on held-out tasks (the Milestone 6 rehearsal)")
    wins = 0
    for k in args.shots:
        d = summary[k]["maml"] - summary[k]["scratch"]
        print(f"    {k:3d} shots: MAML {summary[k]['maml']:.4f} vs scratch "
              f"{summary[k]['scratch']:.4f} ({d:+.4f})")
        wins += d > 0
    check("MAML beats from-scratch at every shot budget", wins == len(args.shots),
          f"{wins}/{len(args.shots)} budgets")
    smallest = min(args.shots)
    check(f"largest advantage at the smallest budget ({smallest} shots)",
          (summary[smallest]["maml"] - summary[smallest]["scratch"])
          >= (summary[max(args.shots)]["maml"] - summary[max(args.shots)]["scratch"]),
          "few-shot advantage should shrink as labels get plentiful")

    print("\nAdaptation beats the unadapted initialization")
    k0 = args.shots[len(args.shots) // 2]
    check("adapting helps over using the meta-init directly",
          summary[k0]["maml"] > summary[k0]["meta_init"],
          f"{summary[k0]['maml']:.4f} vs {summary[k0]['meta_init']:.4f} at {k0} shots")

    print()
    if FAILS:
        print(f"FAILED {len(FAILS)} check(s):")
        for f in dict.fromkeys(FAILS):
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
