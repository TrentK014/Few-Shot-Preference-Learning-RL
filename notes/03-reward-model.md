# Milestone 3: preference reward model (no MAML, no SAC)

## Status
Complete and verified on CARC (job 11772542, debug partition, CPU only, 29m37s, COMPLETED).
Both training modes pass every check.

Trains `r_psi(s, a)` from A-vs-B labels with plain Adam. Proves the Bradley-Terry loss, the architecture
and the data pipeline work, so a Milestone 5 failure is attributable to MAML alone.

## Architecture and hyperparameters (from the reference)
Raw obs+action concat, 43 inputs, no normalization anywhere. Three hidden layers of 256, leaky ReLU,
tanh output, ensemble of 3 batched through one `baddbmm`. Segment score is the sum of 25 per-step rewards.
Bradley-Terry reduces to BCE on `R_A - R_B`, since `P[A>B] = sigmoid(R_A - R_B)`.

Adam lr 3e-4, no weight decay, no schedule. Batch 64 (the reference's config says 256 but
`reward_batch_size` is dead code; 64 is what actually runs). Loss reduction is mean over the batch then
**sum** over ensemble members, matching the reference; averaging over members instead would divide every
gradient by 3 and silently change what lr 3e-4 means.

All 3 ensemble members see the identical batch. There is no bootstrapping anywhere in the reference;
diversity comes only from independent initialization.

## The calibration problem, and why it dominated this milestone

MetaWorld's dense reward is dominated by its reaching term. A one-line heuristic preferring whichever
segment kept the hand closer to the object scores about 0.91 on held-out pairs. So a reward model
reporting "88% accuracy" would be **worse than trivial arithmetic** while sounding like a success. The
reference publishes no preference-accuracy numbers to compare against, so measured heuristic bars are the
only honest calibration available.

Two metrics are therefore reported against bars recomputed in-script for whatever data is passed:
- **Overall** accuracy on held-out episodes, versus the best single hand-picked feature.
- **Hard subset**: the 25% of pairs whose segments have the most similar hand-object distance, where the
  shortcut is uninformative. This is where task understanding actually shows up.

## Three bugs this milestone surfaced

**1. The reference's stopping rule badly underfits offline.** PEBBLE trains until *training* accuracy
reaches 0.95 and has no validation split at all. That is calibrated for online use, where the buffer holds
8 to 200 pairs and 0.95 train accuracy means near-memorization. Offline with ~2,500 training pairs it
fires at epoch 8 while the model is still improving:

| stopping point | held-out accuracy | hard subset |
|---|---|---|
| train acc 0.95 (epoch 8, reference rule) | 0.907 | 0.672 |
| train acc 0.99 (epoch 91) | 0.933 | 0.754 |
| 120 epochs | 0.948 | 0.806 |

Fixed by early stopping on held-out validation *episodes*, a signal the reference does not have.
`--stop-rule reference` still reproduces its behavior exactly.

**2. Stratifying the holdout but not the val/test split of it.** Episodes were stratified by behavior
source when choosing which to hold out, but then split into val and test at random. That gave
val = 2 expert / 2 within / 1 cross and test = 1 expert / 3 within / 1 cross / 1 random. Expert-vs-random
pairs are easy and within-vs-within pairs are hard, so validation read 0.966 while test read 0.880 on what
should be the same distribution, and early stopping optimized the wrong thing. Fixed by dealing each
source's episodes alternately into val and test, with a running parity so odd counts do not always favor
one side.

**3. A single untrained model is not at chance.** The untrained control first measured 0.73. A randomly
initialized smooth function of state correlates with the true reward by luck; over 8 seeds the mean is
0.48 with a standard deviation of 0.14. Only the multi-seed mean is a meaningful no-leakage control.

## Why held-out *pairs* are not the honest metric
The 6,000 stored pairs of a variation touch 96% of its ~3,900 distinct segments, so a held-out pair almost
always consists of two segments seen inside other training pairs. That measures memorization of segment
scores. Three splits are reported; the held-out-episode one is the headline.

## CARC results (job 11772542, 10 variations)

Per-variation models, one per goal variation:

| metric | model | heuristic |
|---|---|---|
| held-out episodes, overall | 0.9574 | 0.8915 |
| held-out episodes, hard subset | 0.9022 | 0.7132 |
| untrained control (10 variations x 8 seeds) | 0.4919 | - |
| stored-pair train / val | 0.9772 / 0.9553 | - |

Pooled model over 7 variations, 3 variations held out of training entirely:

| metric | model | heuristic |
|---|---|---|
| held-out episodes, overall | 0.9774 | 0.8991 |
| held-out episodes, hard subset | 0.9400 | 0.7320 |
| **unseen variations** | **0.9805** | 0.8738 |

The pooled model beats the per-variation models on every metric, and generalizes to variations it never
trained on at 0.9805. That confirms the earlier measurement that "move the window in +x" is a
variation-independent rule within Window Open: pooling helps rather than causing interference. The hard
transfer is Window Open versus Window Close, which are direction-reversed, and that is Milestone 6.

Locally, three variations gave 0.9540 overall and 0.8927 hard against heuristic bars of 0.9083 and 0.7360.
Per-variation results are noisy: the model beat the heuristic on two of three and came 0.006 under on the
third, so the aggregate is what counts. Re-running with the same seed reproduces the stop epoch and every
accuracy exactly.

## Note on the ensemble check
Averaging member logits does not mathematically guarantee accuracy at or above the mean member's. One run
came in 0.0003 below, which is two pairs in 6,000. The check is a sanity bound with a 1-point tolerance,
not a correctness property.
