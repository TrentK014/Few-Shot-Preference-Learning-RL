# Milestone 5: MAML over the reward model

## Status
Complete. Meta-trained on CARC (job 11786711, 1h37m, 10,000 meta-steps, all checks passed).
Checkpoint at `/scratch1/kobielus/fspref/runs/m5/maml_init.pt`.

## What it does

Ordinary training produces weights that answer well on the tasks they saw. MAML produces a *starting
point*: weights from which two gradient steps on a handful of preferences land somewhere good on a task
never seen before.

One meta-step, per task in the batch:

1. take 32 support pairs from the task
2. run 2 gradient steps from the current start (the inner loop)
3. score the adapted weights on 32 *different* pairs from that task
4. push that error back into the **original** start, not the adapted weights

Step 4 differentiates through step 2. That second-order path is where MAML implementations usually break,
so it is checked explicitly: a smoke test confirms the backward pass reaches all 8 parameter tensors and
all 8 learned learning-rate tensors.

## Why a starting point can be "good at learning"

All four tasks are the same arm moving an object. Judging any of them needs the same perception: where the
hand is relative to the object, whether it is in contact, whether the object is moving purposefully. What
differs is small. Sliding the sash right is good for Window Open and bad for Window Close.

So a judge is really a large shared part plus a small task-specific part. Trained from scratch on 25
comparisons, a model has to discover both, and it will never learn what contact is from 25 examples. The
meta-initialization supplies the shared part, leaving the handful of labels to pin down only the rest.

The objective is not the average of the task solutions, which would be mediocre at all of them and no
closer to any. It is a point from which a short walk reaches any of them.

## The learned inner learning rates

Inner learning rates are trained alongside the weights, one tensor per parameter tensor and elementwise
within it (Meta-SGD, matching the reference's `learn_inner_lr`). This lets the model decide which of its
own weights are the task-specific knob and which are shared machinery. Nothing supervises that split; it
falls out of the objective.

## Settings (reference `configs/metaworld/maml.yaml`)

| setting | value |
|---|---|
| inner steps | 2 |
| inner learning rate | 1e-3 initial, then learned per parameter |
| support / query | 32 / 32, disjoint |
| tasks per meta-batch | 4 |
| outer optimizer | Adam, 1e-4 |
| test-time adaptation | up to 40 steps, stop at 0.95 support accuracy |

Support and query must be disjoint. Overlap would let the inner loop memorize the very pairs the outer
loss scores, and the objective would stop measuring adaptation at all.

## Task split

5 goal variations of each family are held out of meta-training entirely, so 60 meta-train tasks and 15
held-out. Evaluation pairs for a held-out task come from episodes that were never used for training or as
support, reusing the Milestone 3 episode-level split.

## Acceptance test

This is a dress rehearsal for Milestone 6. On the held-out tasks, adapt the meta-initialization on K
support pairs and compare against training a fresh model on the **identical** pairs, with the same
stopping rule so neither side is favored. If MAML does not win here, on families it has already practiced,
then Milestone 6 has no chance and the fault is MAML rather than the transfer.

Checks: meta-loss decreases; the inner loop improves query accuracy; MAML beats from-scratch at every
budget; the advantage is largest at the smallest budget, since extra labels should erode it; and adapting
beats using the unadapted initialization directly.

## Results (60 meta-train tasks, 15 held out, 1000 evaluation pairs per task)

Meta-training converged by roughly step 5000; held-out adapted accuracy plateaued near 0.96 and the best
checkpoint scored 0.9646. The per-step meta-loss is noisy because each step sees only 4 of 60 tasks.

| shots | meta-init, unadapted | MAML adapted | from scratch | gain over scratch |
|---|---|---|---|---|
| 8 | 0.9401 | 0.9251 | 0.6826 | **+0.2425** |
| 16 | 0.9401 | 0.9343 | 0.7187 | +0.2156 |
| 32 | 0.9401 | 0.9405 | 0.7771 | +0.1634 |
| 64 | 0.9401 | 0.9443 | 0.8220 | +0.1223 |
| 128 | 0.9401 | 0.9438 | 0.8407 | +0.1031 |

The advantage over training from scratch is large and behaves exactly as a few-shot method should: biggest
where labels are scarcest, eroding steadily as they become plentiful.

## The caveat that matters more than the headline

**Almost none of that gain comes from adaptation.** Read the first two columns rather than the last one.
The meta-initialization scores 0.9401 on these held-out tasks *without a single gradient step*. Adapting
on 32 pairs moves it to 0.9405, and on 8 pairs it actively hurts, dropping to 0.9251. The "adapting beats
the unadapted init" check passes by 0.0004, which is noise.

So what this milestone actually demonstrates is that **pretrained features beat no features**, not that
MAML's adaptability is doing work. That is a weaker claim than it first appears, and it is precisely the
distinction the paper's "Init" baseline exists to isolate: pretrained weights plus ordinary Adam, with no
re-adaptation.

The reason is visible in Milestone 3's measurement. Within Window Open, "move the sash in +x" is a
variation-independent rule, so a model that has practiced on 20 variations of a family already knows
everything it needs for the 21st. Held-out *variations* were never going to require adaptation.

**Window Close is the case that does.** It shares Window Open's object and features but reverses the
direction, so the unadapted initialization should be actively wrong there rather than merely unadapted.
That is the test in Milestone 6, and it is where these numbers will either mean something or not.

Milestone 6 must therefore report three arms, not two: from scratch, the unadapted meta-init, and the
adapted meta-init. Reporting only the first and third would credit adaptation with a gain that these
results suggest belongs to pretraining.
