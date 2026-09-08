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

## Status

Implemented and smoke-tested locally end to end: all four arms run, feedback is
collected and capped at the budget, and the contamination guard refuses a leaky
checkpoint. Pending: the ground-truth SAC validation run, then the 10-family
prior (Milestone 6b) whose checkpoint this milestone consumes.
