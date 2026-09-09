"""Milestone 7: SAC on held-out Window Close, driven by the learned reward model.

This is the first milestone whose numbers are directly comparable to the paper.
Milestones 1 to 6 measured preference accuracy on held-out pairs, which the paper
never reports; every figure in it is success rate or reward against environment
steps. Milestone 6's flat accuracy curve therefore did not contradict anything
the paper claims, and this script produces the number that could.

Four arms:

    sac_oracle   SAC on the ground-truth reward. The ceiling, and the control
                 that tells us a failure is in the RL rather than the reward.
    few_shot     the meta-init, re-adapted from scratch every feedback session
    init         the meta-init, plain Adam every session (the paper's Init)
    pebble       fresh weights every session, no prior data (the floor)

The feedback schedule is the paper's, Table 4: 200 total queries for Window
Close, 8 per session, a session every 5000 environment steps, first session
uniform and later ones disagreement-sampled.

    python scripts/m7_sac.py --checkpoint $FSPREF_ROOT/runs/m6b/maml_init.pt \
        --arms sac_oracle few_shot init pebble --seeds 3 --steps 500000
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
from fspref.envs import (ACT_DIM, HELD_OUT_TASK, MAX_PATH_LENGTH, OBS_DIM,  # noqa: E402
                         make_env, reset_env, step_env)
from fspref.maml import MAMLReward  # noqa: E402
from fspref.online import (FeedbackSchedule, PreferenceDataset, oracle_label,  # noqa: E402
                           readapt, relabel, select_queries)
from fspref.prefs import SEGMENT_SIZE  # noqa: E402
from fspref.sac import BATCH_SIZE, SAC, ReplayBuffer  # noqa: E402

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


@torch.no_grad()
def evaluate(agent, env, n_episodes: int, seed: int, max_steps: int) -> float:
    """Success rate as MetaWorld defines it, with the deterministic policy."""
    successes = 0
    for ep in range(n_episodes):
        obs = reset_env(env, seed=seed + ep)
        done_success = False
        for _ in range(max_steps):
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, info = step_env(env, action)
            if info.get("success", 0.0):
                done_success = True
                break
            if terminated or truncated:
                break
        successes += int(done_success)
    return successes / max(n_episodes, 1)


def usable_device(requested: str) -> str:
    """Verify the device actually runs a matmul before the run commits to it.

    CARC's gpu partition mixes generations, and the p100 (CC 6.0) and v100
    (CC 7.0) nodes have no kernels in the installed torch build. Without this
    check the job queues, starts, and only then dies inside the actor's first
    forward pass with CUBLAS_STATUS_ARCH_MISMATCH. Falling back to CPU is much
    better than losing the allocation: SAC on MetaWorld is small enough that CPU
    is slow but not hopeless.
    """
    if not requested.startswith("cuda"):
        return requested
    if not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu", flush=True)
        return "cpu"
    try:
        a = torch.zeros(8, 8, device=requested)
        _ = (a @ a).sum().item()
        print(f"device check: {torch.cuda.get_device_name(0)} OK", flush=True)
        return requested
    except Exception as e:
        print(f"WARNING: {torch.cuda.get_device_name(0)} cannot run matmuls with this "
              f"torch build ({type(e).__name__}); falling back to cpu.\n"
              f"  Request a supported GPU instead, e.g. --gres=gpu:a40:1 "
              f"(a40/a100/l40s work; p100/v100 do not).", flush=True)
        return "cpu"


def _write_partial(path, args, arm, seed, curve, feedback, readapt_log, collected,
                   final_window):
    """Rewrite the shard's result file with the run so far.

    Same schema as the final write, so scripts/m7_combine.py merges a partial
    shard without knowing the difference. A partial run just has a shorter curve.
    """
    run = dict(arm=arm, seed=seed, curve=curve, feedback=feedback,
               readapt=readapt_log, total_feedback=collected,
               final_success=float(np.mean([c[1] for c in curve[-final_window:]]))
               if curve else 0.0,
               best_success=max((c[1] for c in curve), default=0.0),
               partial=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"args": vars(args), "results": {arm: [run]}}, f)
    os.replace(tmp, path)   # atomic, so a reader never sees a half-written file


def run_arm(arm: str, args, maml, seed: int, partial_path: str | None = None):
    """Train one arm for one seed. Returns a dict of curves and final metrics.

    If `partial_path` is given, the run's curve is rewritten there after every
    evaluation. A 500k-step shard takes hours -- the Init arm longest, because it
    runs plain Adam to convergence every feedback session where MAML converges
    inside its 40 learned-rate steps -- and without this a shard that hits the
    SLURM wall clock loses every evaluation it ever made. Writing as we go means a
    timeout costs the tail of one curve instead of the whole run.
    """
    device = args.device
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # make_env pins the goal variation itself and returns (env, task_name).
    env, _ = make_env(args.task, variation=args.variation, mt1_seed=args.mt1_seed,
                      partially_observable=True)
    eval_env, _ = make_env(args.task, variation=args.variation, mt1_seed=args.mt1_seed,
                           partially_observable=True)

    agent = SAC(obs_dim=OBS_DIM, act_dim=ACT_DIM, device=device, seed=seed)
    buffer = ReplayBuffer(args.buffer_size, device=device)
    schedule = FeedbackSchedule(args.max_feedback, args.per_session, args.session_freq)
    dataset = PreferenceDataset(device=device, segment_size=args.segment_size)

    reward_net = None          # None until the first feedback session
    use_true_reward = arm == "sac_oracle"
    curve, feedback_curve, readapt_log = [], [], []

    obs = reset_env(env, seed=seed)
    episode_id, ep_step = 0, 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # ---------------------------------------------------------- act & store
        if step <= args.seed_steps:
            action = rng.uniform(-1.0, 1.0, size=ACT_DIM).astype(np.float32)
        else:
            action = agent.act(obs)
        next_obs, true_r, terminated, truncated, info = step_env(env, action)
        ep_step += 1
        timeout = ep_step >= args.max_episode_steps
        # A timeout is not a real terminal state; bootstrapping through it is
        # correct and treating it as terminal biases the value function low.
        done_flag = float(terminated and not timeout)
        buffer.add(obs, action, next_obs, true_r, done_flag, episode_id)
        obs = next_obs
        if terminated or truncated or timeout:
            obs = reset_env(env, seed=seed + episode_id + 1)
            episode_id += 1
            ep_step = 0

        # ------------------------------------------------------ ask for feedback
        if not use_true_reward and schedule.due(step):
            n = schedule.next_batch_size()
            a, b = select_queries(buffer, reward_net, n, rng,
                                  segment_size=args.segment_size,
                                  use_disagreement=(reward_net is not None
                                                    and not args.uniform_queries),
                                  device=device)
            if a is not None and len(a) > 0:
                labels, keep = oracle_label(buffer, a, b, args.segment_size,
                                            tie_margin=args.tie_margin)
                # Dropped ties still count against the budget, exactly as the
                # paper counts a human's "skip".
                schedule.collected += len(a)
                if keep.sum() > 0:
                    dataset.add(a[keep], b[keep], labels[keep])
                    reward_net, info_r = readapt(
                        arm, maml, reward_net, dataset, buffer,
                        ensemble_size=args.ensemble, device=device, seed=seed,
                        lr=args.reward_lr, max_maml_steps=args.max_maml_steps)
                    readapt_log.append(dict(step=step, n_labels=len(dataset), **info_r))
                feedback_curve.append((step, schedule.collected))

        # ------------------------------------------------------------- RL update
        if step > args.seed_steps and len(buffer) >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = buffer.sample(args.batch_size, rng)
                if use_true_reward:
                    reward = batch[3]
                elif reward_net is not None:
                    reward = relabel(reward_net, batch[0], batch[1])
                else:
                    # No feedback yet, so no reward signal exists. Skipping the
                    # update is honest; feeding zeros would train the critic on a
                    # reward function we never learned.
                    continue
                agent.update(batch, reward)

        # ----------------------------------------------------------- evaluation
        if step % args.eval_every == 0 or step == args.steps:
            sr = evaluate(agent, eval_env, args.eval_episodes, seed * 1000, args.max_episode_steps)
            curve.append((step, sr, schedule.collected))
            print(f"    [{arm} seed{seed}] step {step:7d}  success {sr:.2f}  "
                  f"feedback {schedule.collected:4d}  ({time.time() - t0:.0f}s)",
                  flush=True)
            if partial_path:
                _write_partial(partial_path, args, arm, seed, curve, feedback_curve,
                               readapt_log, schedule.collected, args.final_window)

    env.close(); eval_env.close()
    final = float(np.mean([c[1] for c in curve[-args.final_window:]])) if curve else 0.0
    return dict(arm=arm, seed=seed, curve=curve, feedback=feedback_curve,
                readapt=readapt_log, final_success=final,
                best_success=max((c[1] for c in curve), default=0.0),
                total_feedback=schedule.collected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="MAML meta-init; required for the few_shot and init arms")
    p.add_argument("--out", default="runs/m7")
    p.add_argument("--task", default=HELD_OUT_TASK)
    p.add_argument("--variation", type=int, default=0)
    p.add_argument("--mt1-seed", type=int, default=0)
    p.add_argument("--arms", nargs="+",
                   choices=["sac_oracle", "few_shot", "init", "init_reset", "pebble"],
                   default=["sac_oracle", "few_shot", "init", "pebble"],
                   help="init is the paper's baseline (pretrained weights once, then "
                        "fine-tune); init_reset resets to the meta-init every session, "
                        "which isolates learned inner rates vs plain Adam")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="first seed index; lets an array job run one (arm, seed) per task "
                        "and have the shards combine into one seed sweep")
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--seed-steps", type=int, default=1000,
                   help="uniform-random actions before the policy takes over")
    p.add_argument("--buffer-size", type=int, default=500000)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=MAX_PATH_LENGTH)
    # Paper Table 4, Window Close.
    p.add_argument("--max-feedback", type=int, default=200)
    p.add_argument("--per-session", type=int, default=8)
    p.add_argument("--session-freq", type=int, default=5000)
    p.add_argument("--segment-size", type=int, default=SEGMENT_SIZE)
    p.add_argument("--tie-margin", type=float, default=0.01)
    p.add_argument("--uniform-queries", action="store_true",
                   help="disable disagreement sampling (the paper's Figure 7 ablation)")
    p.add_argument("--reward-lr", type=float, default=3e-4)
    p.add_argument("--max-maml-steps", type=int, default=40)
    p.add_argument("--ensemble", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--shard-tag", action="store_true",
                   help="suffix output files with arm and seed offset, so array-job "
                        "shards do not overwrite each other")
    p.add_argument("--oracle-floor", type=float, default=0.5,
                   help="success rate the ground-truth-reward arm must reach for the "
                        "run to be trustworthy; lower it only for short smoke runs")
    p.add_argument("--final-window", type=int, default=3,
                   help="evaluations averaged into the reported final success rate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.device = usable_device(args.device)

    # PreferenceDataset stores replay-buffer INDICES, not materialized segments,
    # because the reward model is re-adapted on the entire label history every
    # session and the buffer already holds the observations. That is only sound
    # while the ring buffer has not wrapped: once it does, a stored index points
    # at a transition that has been overwritten, and every old label silently
    # re-attaches to an unrelated segment. Nothing would crash and the run would
    # look fine, so make it impossible rather than merely unlikely.
    if args.buffer_size < args.steps:
        raise SystemExit(
            f"--buffer-size ({args.buffer_size}) must be >= --steps ({args.steps}): "
            "preference labels are stored as buffer indices, so a wrapped buffer "
            "would silently corrupt the label history.")

    needs_ckpt = any(a in ("few_shot", "init", "init_reset") for a in args.arms)
    maml = None
    if needs_ckpt:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for the few_shot and init arms")
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        maml = MAMLReward(ensemble_size=ck["args"]["ensemble"],
                          inner_lr=ck["args"]["inner_lr"],
                          inner_steps=ck["args"]["inner_steps"]).to(args.device)
        maml.load_state_dict(ck["state_dict"])
        train_paths = ck.get("train_paths", [])
        print(f"loaded {args.checkpoint}: {len(train_paths)} meta-training tasks")
        # Held-out isolation is a hard project constraint, so re-assert it here
        # rather than trusting that the checkpoint was built correctly.
        leaked = [t for t in train_paths if HELD_OUT_TASK in str(t)]
        if leaked:
            raise SystemExit(f"checkpoint was meta-trained on {HELD_OUT_TASK}: {leaked[:3]}")

    print(f"device={args.device} | task={args.task} variation={args.variation}")
    print(f"schedule: {args.max_feedback} queries, {args.per_session} per session, "
          f"every {args.session_freq} steps, "
          f"{'uniform' if args.uniform_queries else 'disagreement'} selection")
    print(f"arms={args.arms} seeds={args.seeds} steps={args.steps}\n")

    os.makedirs(args.out, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for arm in args.arms:
        print(f"########## {arm} ##########")
        seeds = range(args.seed_offset, args.seed_offset + args.seeds)
        tag = f"_{'-'.join(args.arms)}_s{args.seed_offset}" if args.shard_tag else ""
        partial = (os.path.join(args.out, f"m7_results{tag}.json")
                   if args.shard_tag and args.seeds == 1 else None)
        results[arm] = [run_arm(arm, args, maml, seed, partial_path=partial)
                        for seed in seeds]
        with open(os.path.join(args.out, f"m7_results{tag}.json"), "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 78 + "\nRESULTS  (success rate, mean +/- sd over seeds)\n")
    summary = {}
    print(f"  {'arm':<12s} {'final':>16s} {'best':>16s} {'feedback':>10s}")
    for arm in args.arms:
        fin = np.array([r["final_success"] for r in results[arm]])
        best = np.array([r["best_success"] for r in results[arm]])
        fb = int(np.mean([r["total_feedback"] for r in results[arm]]))
        summary[arm] = dict(final_mean=float(fin.mean()), final_sd=float(fin.std()),
                            best_mean=float(best.mean()), best_sd=float(best.std()),
                            feedback=fb)
        print(f"  {arm:<12s} {fin.mean():8.3f}+/-{fin.std():<6.3f} "
              f"{best.mean():8.3f}+/-{best.std():<6.3f} {fb:10d}")

    with open(os.path.join(args.out, f"m7_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ checks
    print("\n" + "=" * 78 + "\nSANITY (these fail the job)\n")
    if "sac_oracle" in summary:
        check("SAC on ground-truth reward solves the task",
              summary["sac_oracle"]["best_mean"] >= args.oracle_floor,
              f"best success {summary['sac_oracle']['best_mean']:.3f} "
              f">= {args.oracle_floor} (if this fails the RL is broken, not the reward model)")
    for arm in args.arms:
        if arm in PREFERENCE_ARMS:
            got = summary[arm]["feedback"]
            check(f"{arm} stayed within the query budget", got <= args.max_feedback,
                  f"{got} <= {args.max_feedback}")
    all_curves = [c for arm in args.arms for r in results[arm] for c in r["curve"]]
    check("all evaluations produced finite success rates",
          all(np.isfinite(c[1]) for c in all_curves), f"{len(all_curves)} evaluations")

    print("\n" + "=" * 78 + "\nHYPOTHESES (findings, not pass/fail)\n")
    if "few_shot" in summary and "pebble" in summary:
        d = summary["few_shot"]["final_mean"] - summary["pebble"]["final_mean"]
        finding("H10 few-shot beats PEBBLE at the same query budget "
                "(the paper's central claim)", d > 0,
                f"{summary['few_shot']['final_mean']:.3f} vs "
                f"{summary['pebble']['final_mean']:.3f} ({d:+.3f}) at {args.max_feedback} queries")
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
        for f in dict.fromkeys(FAILS):
            print(f"  - {f}")
        raise SystemExit(1)
    n_conf = sum(c for _, c, _ in FINDINGS)
    print(f"SANITY PASSED. {n_conf}/{len(FINDINGS)} hypotheses confirmed.")


if __name__ == "__main__":
    main()
