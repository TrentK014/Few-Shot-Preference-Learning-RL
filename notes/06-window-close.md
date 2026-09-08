# Milestone 6: adapting to held-out Window Close

## Status
Main run complete on CARC (job 11789719). Low-budget follow-up running as job 11790403.

**The main hypothesis was refuted.** On Window Close, adapting the meta-initialization does not beat
training from scratch on the same labels. This is the honest result of the simplified reproduction, and it
is recorded here as a finding rather than smoothed away.

## Setup

Window Close data was generated into a separate root, `data/test_prefs`, with the generator refusing both
to produce the held-out task without an explicit flag and to write it anywhere the meta-training loader
globs. The run verified directly that none of the 60 meta-training tasks was Window Close.

Four arms on identical support pairs, all sharing one training loop and one stopping rule (support
accuracy 0.95) so no arm is favoured by construction:

| arm | what it is |
|---|---|
| scratch | fresh weights, Adam on K labels |
| meta-init | the checkpoint, zero labels |
| Init | the checkpoint, then ordinary Adam (the paper's "Init" baseline) |
| MAML | reset to checkpoint, learned-rate SGD to 0.95 or 40 steps, then Adam fallback |

Five goal variations x 3 support draws per budget, evaluated on 1000 fresh pairs from Window Close
episodes no arm trained on.

## Results

Unadapted meta-init: **0.7136 +/- 0.0613**. Hand-distance heuristic on the same pairs: **0.8800**.

| labels | scratch | Init | MAML | MAML - scratch | MAML - Init |
|---|---|---|---|---|---|
| 25 | 0.8326 +/- 0.058 | 0.7949 +/- 0.071 | 0.8179 +/- 0.051 | **-0.0147** | +0.0230 |
| 50 | 0.8609 +/- 0.043 | 0.8577 +/- 0.047 | 0.8610 +/- 0.051 | +0.0001 | +0.0033 |
| 100 | 0.8628 +/- 0.062 | 0.8561 +/- 0.066 | 0.8689 +/- 0.061 | +0.0061 | +0.0128 |
| 200 | 0.8865 +/- 0.053 | 0.8780 +/- 0.049 | 0.8874 +/- 0.039 | +0.0009 | +0.0094 |

Every MAML-versus-scratch difference is between -0.015 and +0.006 against standard deviations near 0.05.
They are ties.

## What was confirmed, and what was not

| hypothesis | outcome | numbers |
|---|---|---|
| H0 MAML beats scratch at every budget | **REFUTED** | -0.015, +0.000, +0.006, +0.001 |
| H1 unadapted meta-init at or below chance | **REFUTED** | 0.7136, well above 0.50 |
| H2 MAML beats Init at 25 labels | CONFIRMED | +0.0230 |
| H2b that gap shrinks with budget | CONFIRMED | +0.0230 at 25 -> +0.0094 at 200 |
| H3 reward/velocity correlation flips sign | REFUTED | -0.1034 -> -0.2056, already negative |
| H4 MAML at 200 beats the heuristic | CONFIRMED | 0.8874 vs 0.8800 |
| H5 advantage largest at smallest budget | REFUTED | it is *negative* at 25 |

## Reading the refutations

**H1 and H3 fail for the same reason, and it is informative.** The prediction was that the meta-init would
be confidently backwards on Window Close, since Window Open rewards sliding the sash in +x and Window
Close rewards the reverse. It is not. It scores 0.7136 unadapted, and its reward already correlates
*negatively* with sash x-velocity (-0.1034) before adaptation touches it.

The likely reason is that meta-training saw three families whose correct directions disagree, so the
initialization could not commit to any one of them. It appears to encode direction-agnostic structure,
contact and purposeful motion, and leaves direction to adaptation. That is arguably MAML working as
intended. It also means Window Close was never the adversarial test the plan assumed.

**H0 is the one that matters, and it failed cleanly.** The meta-initialization transfers something
(0.71 unadapted, against 0.50 chance), but not enough to beat starting from random weights. At 25 labels
it is actually slightly *worse* than scratch.

**What did work is narrower but real.** MAML beats Init at every budget, by the most at the smallest one.
Since both start from identical weights and differ only in how they adapt, the meta-learned adaptation
procedure is doing genuine work. The meta-learning produced a more adaptable initialization; that
initialization just is not better than random for this particular task.

## Why the headline may have failed here

Candidate explanations, none yet tested, listed so Milestone 7 and 8 can discriminate:

1. **The metric may be saturating.** A one-line hand-distance heuristic scores 0.88 on these pairs, and
   every arm lands between 0.83 and 0.89. There is very little room between a trivial baseline and the
   ceiling for pretraining to show an advantage. Preference accuracy may simply not discriminate here.
2. **Offline support pairs are unusually informative.** Our support sets are drawn from a mixture
   containing genuine experts and pure random flailing, so even 25 labels contain stark good-versus-bad
   contrasts. In the paper's online setting the policy is bad early and preferences compare mediocre
   behavior to mediocre behavior, which is much harder to learn from and where pretraining should matter
   more. Milestone 7 moves to that setting.
3. **Three prior families instead of ten.** The paper meta-trains across 10 families and 250 variations.
   With 3 families the initialization has far less shared structure to offer.
4. **25 labels may already be past the interesting regime.** Milestone 5 saw its largest gain at 8 labels.
   Job 11790403 sweeps 4, 8, 16 and 25 to check.

## Note on a check I had to reclassify

The plan filed "MAML beats scratch at every budget" as a sanity check that would fail the job, reasoning
that Milestone 5 had already established it. That was wrong, and it repeats the Milestone 2 and 4 mistake
of encoding an expectation as a correctness test. Milestone 5's held-out tasks were new goals of practiced
families; Window Close is a different situation entirely, and "pretraining does not help here" is a
legitimate finding rather than a bug. It was moved to H0 before the run, not after seeing the result.

Separately, the inspection script asserted the held-out task was absent from whatever root it inspected,
which is wrong when that root is deliberately the Window Close one. It now guards the pretraining root
specifically, which was always the intent.
