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
