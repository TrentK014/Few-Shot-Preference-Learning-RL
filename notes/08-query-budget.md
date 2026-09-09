# Milestone 8: how few queries can this work with?

Milestone 7 fixed the budget at the paper's 200 and asked whether the method
works. This varies the budget -- 25, 50, 100, 200 -- with the schedule shape
unchanged (8 queries per session, a session every 5000 steps), so a smaller
budget simply means feedback stops earlier and SAC trains on against a frozen
reward model. That is the paper's Figure 6 ablation.

27 runs (3 budgets x 3 arms x 3 seeds); the 200-query point is Milestone 7's.
`sac_oracle` is excluded because it never consumes feedback, so its 1.000 is the
ceiling at every budget.

## Results

Final success rate (mean over 3 seeds):

| budget | few_shot | init | pebble |
|---|---|---|---|
| 25 | 0.222 +/- 0.157 | 0.000 | 0.000 |
| 50 | 0.111 +/- 0.157 | 0.000 | 0.000 |
| 100 | 0.667 +/- 0.471 | 0.222 +/- 0.314 | 0.667 +/- 0.471 |
| 200 | **1.000 +/- 0.000** | **1.000 +/- 0.000** | 0.667 +/- 0.471 |

Seeds that *ever* reached 100%:

| budget | few_shot | init | pebble |
|---|---|---|---|
| 25 | **3/3** | **3/3** | **0/3** |
| 50 | 2/3 | 2/3 | **0/3** |
| 100 | 3/3 | 3/3 | 3/3 |
| 200 | 3/3 | 3/3 | 3/3 |

## The prediction was wrong, and the reason matters

On record before the run: few-shot's advantage over PEBBLE should *widen* as the
budget shrinks, since at 200 every method eventually solves the task.

- **H13 REFUTED** -- the margin does not widen: +0.222 at 25, +0.111 at 50,
  +0.000 at 100, +0.333 at 200.
- **H15 REFUTED** -- few-shot does not hold the ceiling at 25 queries (0.222).
- **H14 CONFIRMED** -- PEBBLE fails outright at the smallest budget, 0/3 seeds.

The two tables disagree in a way that explains all three. At budget 25, **all
three few-shot seeds reach 100% success** and the final score is still 0.222.
The policy finds the behavior and then loses it.

## PEBBLE does not merely lose the behavior -- it never finds it

The per-budget reports carry a `best` column the curve table above does not:

**Budget 25**

| arm | final | best ever | solved | first reached |
|---|---|---|---|---|
| few_shot | 0.222 | **1.000** | 3/3 | 123k steps |
| init | 0.000 | **1.000** | 3/3 | 57k steps |
| pebble | 0.000 | **0.000** | **0/3** | **never** |

PEBBLE's *best* is 0.000. Across 3 seeds and 500k steps at 25 queries -- and
again at 50 -- it never once reached success at any evaluation. This is not
"reached it and drifted off", which is what happens to the pretrained arms. It is
never finding the behavior at all.

So at low budgets the prior is not a marginal head start. It is the difference
between finding Window Close and not finding it.

Note also that `init` reaches success *faster* than `few_shot` here (57k vs 123k
steps) and then ends at 0.000 against few_shot's 0.222 -- the same
speed-versus-stability split Milestone 7 found, appearing again independently.

## What the budget actually buys

Success trajectories for few_shot seed0, sampled every 50k steps:

| budget | feedback stops | trajectory |
|---|---|---|
| 25 | step 10k (2% of training) | `0 0 0 0 0 0 0 0 1 0` |
| 50 | step 30k (6%) | `0 1 0 0 0 0 0 0 0 0` |
| 200 | step 120k (24%) | `0 0 1 1 1 1 1 1 1 1` |

After feedback stops, SAC keeps optimizing a **frozen** reward model for the rest
of the run -- 98% of training at budget 25, 76% at budget 200. The policy
overoptimizes the learned reward and drifts off the true task. A larger budget
means feedback continues deeper into training, so the reward model keeps being
corrected as the policy improves, and the policy stays anchored.

**The budget does not buy discovery. It buys stability.**

The prior buys discovery: at 25 and 50 queries, both pretrained arms reach 100%
on 2-3 of 3 seeds while PEBBLE reaches it on none. A reward-function prior tells
the agent where to go with almost no feedback; only sustained feedback keeps it
there.

## Connection to Milestone 7

Milestone 7 found that re-adaptation buys reliability rather than speed: Init was
*faster* to first success under the 3-family prior (85.0 vs 103.7 queries) but
ended at 0.778 against few-shot's 1.000. Same mechanism. As the policy improves,
the replay-buffer distribution shifts and the reward function implied by the
accumulated preferences moves with it. Anything that fails to track that drift --
too few queries, or stale fine-tuning instead of re-adaptation -- produces a
policy that solves the task transiently and then wanders off.

That reframes what the method is for. The hard part of preference-based RL here
is not learning a reward from few labels; a shortcut-heavy prior does that almost
immediately. The hard part is keeping the reward correct while the thing being
rewarded changes underneath it.

## Caveat

3 seeds per cell, one held-out task, and success rate measured over 10 episodes,
so single-cell differences are noisy -- the 100-query row in particular is not
cleanly ordered. The claims that survive that noise are the two extremes: PEBBLE
never solving Window Close at 25 or 50 queries (0/6 runs), and every arm holding
1.000 only at 200.
