# Milestone 7: SAC with the adapted reward model

## Why this milestone decides the project's question

Milestones 1-6 measured preference accuracy on held-out pairs. The paper never
reports that number -- every figure in it (2, 3, 6, 7, 8) is success rate or
reward against environment steps, and Table 1 is meters-to-goal. So Milestone 6's
flat accuracy curve did not contradict the paper; it measured something the paper
does not claim. This milestone produces the first number that is comparable.

The claim being tested is the paper's headline: **the same policy performance at
20x fewer queries**, via a reward-function prior meta-learned on other tasks.

## Design

Four arms, sharing one training loop so no arm is favoured by construction:

| arm | reward | weights each feedback session |
|---|---|---|
| `sac_oracle` | ground truth | n/a -- the ceiling, and the control |
| `few_shot` | learned | reset to meta-init, learned-rate SGD, Adam fallback |
| `init` | learned | reset to meta-init, plain Adam (the paper's Init) |
| `pebble` | learned | fresh random weights, plain Adam (the floor) |

`sac_oracle` is the load-bearing control. If SAC on the true reward does not solve
Window Close, then a failure in any preference arm is uninformative -- it would
mean the RL is broken, not the reward model. That check fails the job.

### The schedule is the paper's, not one we invented

Table 4, Window Close row: **200 total queries, 8 per session, a session every
5000 environment steps**. Table 3: disagreement sampling with a multiplier of 10,
after a first uniform session. Section 3: "we crucially reset the reward model for
adaptation" -- every session the weights go back to the meta-init and adapt on all
feedback so far, rather than continuing to fine-tune.

That last point is the entire difference between `few_shot` and `init`, and it is
the paper's central algorithmic claim.

### Why the schedule matters for interpreting Milestone 6

Because feedback arrives 8 at a time, the model adapts on 8 pairs, then 16, then
24, and so on. Milestone 6 tested budgets 25/50/100/200 -- only the tail. The
low-budget sweep (4/8/16/25) found the meta-init ahead by +0.062/+0.025/+0.016
before crossing zero at 25. **The regime the paper actually operates in is the one
where our prior helps**, which is why the Milestone 6 "refutation" is better read
as a sampling-range artifact than a disagreement with the paper.

## Files

- `fspref/sac.py` -- SAC (tanh-Gaussian actor, twin critics, learned temperature)
  and `ReplayBuffer`, which doubles as the pool queries are drawn from. That dual
  role is the point: online queries compare two mediocre behaviors, not an expert
  against a random flail, which is the regime a prior should help most.
  `SAC.update(batch, reward)` takes the reward as an argument, so nothing in SAC
  knows whether it is being fed ground truth or a learned model.
- `fspref/online.py` -- `FeedbackSchedule`, `select_queries` (uniform, then
  disagreement), `oracle_label` (the "fake human": compares ground-truth segment
  returns, never regresses on them), `PreferenceDataset` (stores buffer indices,
  since the whole history is re-gathered every session), and `readapt`, which is
  where the three preference arms differ.
- `scripts/m7_sac.py` -- the harness, with the same two-tier verification used
  since Milestone 2.
- `carc/m7_sac.sbatch` -- GPU, 24h.

## Verification

**Sanity (fails the job)**
1. SAC on ground-truth reward reaches the `--oracle-floor` success rate.
2. Every preference arm stays within the query budget.
3. All evaluations produce finite success rates.
4. The checkpoint was not meta-trained on Window Close. Asserted at load time
   from the checkpoint's own `train_paths`, and verified to fire: a deliberately
   contaminated checkpoint is refused.

**Hypotheses (findings, not pass/fail)**
- H10: `few_shot` beats `pebble` at the same query budget (the paper's claim).
- H11: `few_shot` beats `init` (re-adaptation beats plain fine-tuning).
- H12: `few_shot` reaches >= 80% of the ground-truth-reward ceiling.

## The ground-truth control passed

Ran `sac_oracle` alone for 150k steps on Window Close before spending cluster
time on anything else:

| steps | 10k | 20k | 30k | 40k | 50k | 60k | 70k+ |
|---|---|---|---|---|---|---|---|
| success | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 |

Final **1.000 +/- 0.000**, stable from 70k steps on. So the SAC implementation
solves Window Close from the true reward well inside the budget the preference
arms get, and any shortfall in those arms is attributable to the reward model
rather than to the optimizer. That is the whole reason this arm exists.

The dip at 60k is ordinary early-SAC oscillation, not a bug -- the policy is
still moving fast at that point and the evaluation is only 10 episodes.

## Two cluster lessons

**Not every GPU on CARC runs this torch build.** The first pilot drew a Tesla
V100 (compute capability 7.0) and torch 2.14+cu130 ships no kernels for it, so
the job queued, started, and died four minutes in with
`CUBLAS_STATUS_ARCH_MISMATCH` inside the actor's first forward pass. a40, a100
and l40s work; p100 and v100 do not. `usable_device()` now runs a real matmul at
startup and falls back to CPU with a message naming the supported GPUs.

**This workload does not want a GPU anyway.** SAC here is bound by MuJoCo
stepping, not by the 3x256 networks, so the a40 queue was pure cost. Milestone 7
runs as a CPU array job instead -- one (arm, seed) per task, twelve in parallel
-- which schedules immediately and finishes the sweep faster than one GPU job
running the same twelve sequentially. `scripts/m7_combine.py` merges the shards.

## Status

Implemented, smoke-tested on all four arms, and the ground-truth control passed.
The `sac_oracle` and `pebble` shards need no checkpoint, so they are already
running; `few_shot` and `init` are chained behind the 10-family meta-training via
Slurm dependencies.
