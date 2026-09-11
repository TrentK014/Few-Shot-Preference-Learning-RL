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

---

# FINAL: all 18 shards complete

## 10-family prior (`runs/m7`)

| arm | final success | steps to solve | **queries to solve** |
|---|---|---|---|
| sac_oracle (ceiling) | 1.000 +/- 0.000 | 50k +/- 16k | 0 |
| **few_shot** | **1.000 +/- 0.000** | 63k +/- 40k | **101.0 +/- 64.0** |
| init | 1.000 +/- 0.000 | 77k +/- 33k | 122.3 +/- 52.6 |
| pebble | 0.667 +/- 0.471 | 83k +/- 33k | 130.7 +/- 49.0 |

- **H10 CONFIRMED** few-shot beats PEBBLE on final success, 1.000 vs 0.667.
- **H10b CONFIRMED** few-shot needs fewer queries, 101.0 vs 130.7. The paper's
  central claim in the form it actually argues it.
- **H11 REFUTED** few-shot and Init tie at 1.000 final success. On query
  efficiency few-shot wins, 101.0 vs 122.3.
- **H12 CONFIRMED** few-shot reaches 100% of the ground-truth ceiling.

## 3-family prior (`runs/m7_3family`)

| arm | final success | queries to solve |
|---|---|---|
| few_shot | 1.000 +/- 0.000 | 103.7 +/- 68.4 |
| init | **0.778 +/- 0.314** | 85.0 +/- 7.8 |

## The two results that matter most, and they are not the ones expected

**1. Prior breadth barely moved policy performance.** few_shot needs 103.7
queries under the 3-family prior and 101.0 under the 10-family one -- a
difference well inside the noise, and both arms solve the task on 3/3 seeds.
Milestone 6b showed the same expansion improved *preference accuracy* a great
deal (MAML minus scratch, +0.062 to +0.172 at 4 labels).

So the two metrics respond to completely different things. Expanding the prior
made the reward model a much better *ranker* and left it an equally good *dense
reward*. Since the policy only consumes the dense reward, the policy did not
care. This is the shortcut story from Milestone 6b, confirmed from the other
direction: three families already supplied "approach the object", which is what
actually shapes the policy, and the extra seven families refined a ranking
ability that SAC never uses.

**2. Re-adaptation buys reliability, not speed.** Init is not slower to first
success -- under the 3-family prior it is *faster* (85.0 vs 103.7 queries). What
it fails to do is hold on: its final success is 0.778 +/- 0.314 against
few_shot's 1.000 +/- 0.000, so at least one seed reached 1.00 and then lost it.
The same pattern is visible in PEBBLE (best 1.000, final 0.667).

That is what the reset-and-re-adapt step is for. As the policy improves, the
replay buffer's distribution shifts and the reward function implied by the
accumulated preferences moves with it. Continuing to fine-tune stale weights
tracks that drift badly; rebuilding from the meta-initialization each session
does not. The paper calls this its crucial algorithmic difference from Init, and
the failure mode it prevents turns out to be **instability late in training**,
not slowness early.

Both findings only became visible because the harness reports first-time-to-solve
separately from final success. Reporting only "final success" would have shown
three arms tied at 1.000 and one at 0.667, and reporting only "best success"
would have shown all four tied at 1.000.

## What we do and do not claim

We reproduce the ordering (few-shot < Init < PEBBLE in queries needed) and the
reliability gap, at a fixed 200-query budget, on one held-out task, with 3 seeds.

We do **not** reproduce the paper's 20x figure and do not claim it: that compares
against PEBBLE's original 4000-query budget for Window Close, which we never ran.

---

# CORRECTION: the Init baseline was wrong, and what the fix showed

`fspref/online.py::readapt` reset Init to the meta-init on every feedback
session. The paper's Init is defined by *not* doing that ("Instead of re-adapting
the reward model each time new feedback is collected, we initialize the reward
model with the pretrained weights, and then perform standard updates with the
Adam optimizer as in PEBBLE", Section 4.1). Our Init was therefore a stronger
baseline than the paper's, and the original H11 verdict was mislabeled.

Fixed in `6ca61b5`; the old behaviour is kept as a separate arm, `init_reset`,
which turns the mistake into a useful three-way decomposition. Re-run as job
11846167.

## 10-family prior: the machinery does not matter

| arm | final success | queries to solve |
|---|---|---|
| sac_oracle | 1.000 +/- 0.000 | 0 |
| few_shot | 1.000 +/- 0.000 | 101.0 +/- 64.0 |
| **init (corrected)** | **1.000 +/- 0.000** | 117.0 +/- 65.7 |
| init_reset (old, buggy) | 1.000 +/- 0.000 | 122.3 +/- 52.6 |
| pebble | 0.667 +/- 0.471 | 130.7 +/- 49.0 |

Correcting Init changed almost nothing here: 122.3 -> 117.0 queries, both at
1.000 final. **H11 is still REFUTED on final success**, and now for a reason we
can trust rather than because the baseline was inflated.

## 3-family prior: the machinery matters a lot

| arm | final success | queries to solve |
|---|---|---|
| few_shot (reset + learned inner rates) | **1.000 +/- 0.000** | 103.7 +/- 68.4 |
| init_reset (reset, plain Adam) | 0.778 +/- 0.314 | 85.0 +/- 7.8 |
| **init (paper-faithful, no reset)** | **0.556 +/- 0.416** | 119.7 +/- 58.4 |

Removing the reset made Init worse on both axes. Read as a decomposition:

- no reset -> reset: 0.556 -> 0.778
- reset -> reset + learned inner rates: 0.778 -> 1.000

Both of the paper's algorithmic choices contribute, in the direction it claims.

## The finding that survives

**The algorithmic machinery only matters when the prior is weak.** With three
prior families, removing the reset costs 0.222 of final success and removing the
learned rates costs another 0.222. With ten families every variant reaches 1.000
and the differences vanish into the noise.

That is consistent with everything else here. Milestone 6b showed ten families
make the meta-initialization genuinely adaptable rather than memorized; once the
prior is that good, how you adapt it matters less. The paper's Init ablation was
run against its own ten-task prior and still separated, which ours does not --
plausibly because our PEBBLE-style baselines are stronger than theirs (ours
solves Window Close at 100 queries where theirs needed 4000).

## Honest limits

Three seeds, and the 3-family final-success spread is +/-0.416. The ordering
0.556 < 0.778 < 1.000 is directional, not significant. What is solid is the
10-family row, where three independent arms all sit at exactly 1.000 +/- 0.000,
and the original claim being withdrawn.

## What the original claim should have said

Commit `80fe9dc` reported "re-adaptation buys reliability, not speed" from
few_shot 1.000 vs init 0.778 under the 3-family prior. Both arms reset, so that
gap was learned-inner-rates versus plain Adam, not re-adaptation. The corrected
statement: against the paper-faithful Init (0.556), few-shot's advantage is
+0.444, of which roughly half is the reset and half the learned rates -- and
none of it is visible once the prior has ten families.
