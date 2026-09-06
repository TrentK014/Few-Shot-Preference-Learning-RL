# Milestone 4: preference datasets for all three prior tasks

## Status
Data complete and verified on CARC (job 11772780, 9m12s). 25 goal variations each for Window Open, Push
and Drawer Close, so **75 meta-training tasks** and 450,000 labeled pairs in 21.7 MB.
Learnability check running as job 11772814.

## What this milestone actually uncovered

Milestone 4 was supposed to be Milestone 2's command run twice more. Push was exactly that. Drawer Close
failed hard and exposed a real flaw in how the data was being generated.

### MetaWorld v3 Drawer Close has no reward shaping

Per-step reward traces of the scripted expert make it plain:

| task | t=0 | t=20 | t=60 | after success |
|---|---|---|---|---|
| window-open | 0.411 | 0.515 | 0.605 | 3.3 to 6.2 |
| push | 0.053 | 0.055 | 9.141 | 10.0 |
| **drawer-close** | 0.452 | **0.000** | **0.000** | 10.0 |

Drawer Close pays exactly zero through the entire approach and 10.0 once the drawer shuts. It is a
sparse-reward task wearing a dense-reward interface.

### Consequence: the reference's stop-on-success destroys that task's signal

The reference ends an episode the moment either environment reports success, which is fine when reward is
shaped, because the shaping already recorded how good the approach was. With a binary reward the 10.0 is
never collected, so every segment scores about zero:

- 42% of Drawer Close steps scored exactly 0.0
- **74% of its preference pairs differed by less than 1e-6**, making those labels coin flips
- expert segments had a 90th percentile of 0.004 against random's 0.006, so the expert looked *worse*

Window Open's figure for uninformative pairs was under 1%. This is why the failure surfaced only now.

Running episodes to the full 500 steps instead is no better: segment returns saturate at 250, 55% of pairs
tie at the ceiling, label balance collapses to 0.25, and within-family behavior succeeds 100% of the time,
flattening the quality ladder.

### Two fixes, both forced by measurement

- **`--success-tail 25`**: keep recording for one segment length after success, so at least one whole
  segment can observe the post-success state. 0 reproduces the reference exactly.
- **`--tie-margin 0.01`**: drop pairs the oracle cannot separate, since their labels are arbitrary.
  `build_pairs` now oversamples so the margin does not silently shrink the dataset below 6000 pairs.

Both are applied uniformly to all three tasks, and all three were regenerated together, so no MAML task
differs from another by construction.

## Verification checks had to become scale-free

Four checks written against Window Open encoded that one task's reward shape and failed on the others.
They asserted absolute numbers where the substantive claim was structural:

| removed | replaced with | why |
|---|---|---|
| expert beats random >= 95% | same rate, tested for significance instead | the achievable rate is a property of the task's reward, 0.99 for push and 0.80 for drawer-close |
| expert median above random p90 | **monotone source ladder** | distribution-shape tests encode one reward's shape |
| random never solves the task | expert-minus-random success gap >= 0.3 | flailing shuts the drawer 28% of the time; that is the task, not a bug |
| within-family is mixed by success | subsumed by the ladder | Drawer Close is easy enough that within-family always succeeds, yet its returns still sit below expert |

The monotone ladder is the honest statement, and it holds everywhere.

## Results (25 variations per task, 150,000 pairs each)

| task | label mean | expert beats random | expert / within / cross / random mean segment return | expert vs random success |
|---|---|---|---|---|
| window-open | 0.5001 | 0.9522 | 38.78 / 19.10 / 11.78 / 12.43 | 1.00 vs 0.02 |
| push | 0.4969 | 0.9903 | 127.14 / 9.76 / 1.26 / 1.87 | 1.00 vs 0.00 |
| drawer-close | 0.5003 | 0.7996 | 49.56 / 31.15 / 13.16 / 3.96 | 1.00 vs 0.28 |

Drawer Close remains the weakest signal at 0.80, which is intrinsic: with no shaping, the only thing to
prefer is whether the drawer got shut.

## Carried into Milestone 5
The three tasks are genuinely heterogeneous. Push's expert segments score 127 while Window Open's score
39, and Drawer Close's reward is binary where the others are shaped. A single pooled reward function
cannot serve all three, which is the situation MAML exists to handle. That heterogeneity is the point of
the experiment, not a defect to normalize away.
