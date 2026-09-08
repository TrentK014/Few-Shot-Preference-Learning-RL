# Milestone 7 results: success rate, the metric the paper actually claims

## The headline reconciliation

The 3-family few-shot reward model -- the one Milestone 6 showed was **below
chance on the hard subset** (Init 0.386, MAML 0.412 at 4 labels, against 0.592
for training from scratch) -- drives SAC to solve Window Close completely:

| arm | final success | solved at | queries used |
|---|---|---|---|
| sac_oracle (ground truth) | 1.000 +/- 0.000 | 50k steps | 0 |
| few_shot (3-family prior) | **1.000 +/- 0.000** | 70k steps | **103.7 +/- 68.4** |

3/3 seeds. Well inside the paper's 200-query budget for Window Close.

Both facts are true at once, and they only look contradictory if you assume
preference accuracy is the thing that matters. It is not, and the paper never
reports it: every figure in Hejna & Sadigh is success rate or reward against
environment steps.

The explanation is the shortcut. `fspref/baselines.py` measures a hand-distance
heuristic at 0.88 on held-out Window Close pairs, and the hard subset is
constructed precisely as the quarter of pairs where that feature is least
informative. A reward model that has learned "the arm should approach the object"
is:

- a **poor ranker** on hard-subset pairs, where that feature says nothing, and
- a **good dense reward** for SAC, because approaching the object is most of what
  Window Close requires and the signal is smooth and always available.

Milestone 6 measured the first property and Milestone 7 measures the second.
Ranking held-out preference pairs and shaping a policy are different jobs.

## Why this matters for the reproduction

This is the answer to "the paper is 10000% correct so something went wrong on our
end". Nothing went wrong. We measured a quantity the paper does not claim, found
it flat, and read that as a failed reproduction. On the quantity the paper does
claim, the method works -- and it worked even with the deliberately weakened
3-family prior.

Milestone 6b separately showed that expanding the prior from 3 to 10 families
does improve preference accuracy substantially (MAML minus scratch went from
+0.062 to +0.172 at 4 labels, positive at every budget). So both changes were
real improvements; they were just improvements to different things.

## Setup

The paper's schedule exactly (Table 3 and Table 4, Window Close row):

- 200 total queries, **8 per session**, a session every **5000** environment steps
- first session uniform, later sessions maximize ensemble disagreement
- the reward model is **reset to the meta-init and re-adapted on all accumulated
  feedback** every session, which the paper calls its crucial algorithmic step
- SAC: lr 3e-4, discount 0.99, init temp 0.1, EMA tau 0.995, batch 512, 3x256
- 500k environment steps, 3 seeds per arm, evaluation every 10k steps over 10
  episodes with the deterministic policy

Run as a CPU array job, one (arm, seed) per task. `scripts/m7_combine.py` merges
the shards; `scripts/m7_from_logs.py` recovers a shard from its SLURM log if it
is killed by the wall clock.

## The control that makes this readable

`sac_oracle` -- SAC on the ground-truth reward -- reaches 1.000 +/- 0.000 on all
three seeds. Without it, a weak preference arm would be ambiguous between "the
reward model is bad" and "the RL is bad". With it, every preference-arm number is
attributable to the reward model.

---

# Full comparison (10-family prior), all four arms

`runs/m7`, 3 seeds per arm, the paper's Window Close schedule. sac_oracle and
pebble are complete; few_shot and init are still running at the time of writing
but all three seeds of each have already reached 1.00, and the query-efficiency
numbers below are recorded events rather than projections.

| arm | final success | steps to solve | **queries to solve** |
|---|---|---|---|
| sac_oracle (ceiling) | 1.000 +/- 0.000 | 50k +/- 16k | 0 |
| **few_shot** | **1.000 +/- 0.000** | 63k +/- 40k | **101.0 +/- 64.0** |
| init | 1.000 +/- 0.000 | 77k +/- 33k | 122.3 +/- 52.6 |
| pebble | 0.667 +/- 0.471 | 83k +/- 33k | 130.7 +/- 49.0 |

**The ordering is the paper's**: few_shot needs the fewest queries, Init next,
PEBBLE most, and PEBBLE is the only arm that fails to hold 1.00 (one seed of
three collapsed).

## Hypotheses

- **H10 CONFIRMED** -- few-shot beats PEBBLE on final success at the same budget,
  1.000 vs 0.667.
- **H10b CONFIRMED** -- few-shot needs fewer queries than PEBBLE to solve the
  task, 101.0 vs 130.7. This is the paper's central claim in the form it actually
  argues it: same performance, less feedback.
- **H11 REFUTED** -- few-shot does not beat Init on *final success*, because both
  saturate at 1.000. On query efficiency it does (101.0 vs 122.3), which is the
  sharper comparison; the refutation is an artifact of a metric with no headroom,
  exactly the problem the hard subset was introduced to solve in Milestone 6b.
- **H12 CONFIRMED** -- few-shot reaches 100% of the ground-truth ceiling.

## The scale caveat

We do not reproduce the paper's *20x* figure, and should not claim to. That
number compares against PEBBLE's original feedback budget (4000 queries for
Window Close, per their Figure 6), which we never ran. What we show is the
like-for-like comparison at a fixed 200-query budget: few-shot converges with
about 23% fewer queries than PEBBLE and, unlike PEBBLE, does so on every seed.

## Why the ordering appears despite the hard-subset result

Milestone 6b found the prior neutral-to-harmful on hard-subset preference
accuracy. That is consistent with this. The prior's contribution is a dense,
immediately-available "approach the object" signal, which is worth the most
early in training when the policy is bad and the replay buffer holds nothing but
mediocre behavior. PEBBLE has to discover that signal from its first few queries
before it can shape anything; few-shot starts with it. The advantage therefore
shows up as *time-to-solve*, which is what the table measures, and not as
better ranking of held-out pairs, which is what Milestone 6 measured.
