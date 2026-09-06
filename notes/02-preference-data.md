# Milestone 2: diverse trajectories + automatic preference pairs

## Status
Complete and verified on CARC (job 11772518, debug partition, CPU only, 3m45s, COMPLETED).
25 Window Open variations, 150,000 labeled pairs, 5.8 MB total at
`/scratch1/kobielus/fspref/data/prefs/window-open/var{00..24}.npz`.

## What the reference implementation actually does

Two findings from `github.com/jhejna/few-shot-preference-rl` overturn the obvious reading of the paper.

**1. "Actions from parametric variations" is a dual-environment policy transplant**, not action replay
and not goal spoofing. In `scripts/metaworld/collect_policy_dataset.py::collect_episode`, two envs are
stepped with the same action stream: the scripted policy sees `src_env`'s observation, every recorded
transition comes from `dest_env`. They diverge immediately because their object placements differ, so the
actions are plausible but wrong in `dest_env`. Only the choice of `src_env` separates the three policy
sources, so they collapse into one code path (`fspref/collect.py::collect_episode`).

**2. The reward model never sees the goal.** The reference records from an ML10 env with
`partially_observable=True`, so `obs[36:39]` is zeroed in every observation the reward model ever sees.
This is load-bearing: with the goal visible, one function "reward = -distance(object, goal)" would fit
Window Open and Window Close alike and the transfer result would be vacuous. We use MT1, which sets the
flag False, so `envs.task_with_observability` flips it back by round-tripping `Task.data`.
Source envs stay goal-visible so goal-reading policies (push) still work.

## Mixture per variation (the reference's 52 episodes, verbatim)
15 expert / 25 within-family / 10 cross-family / 2 random, Gaussian action noise epsilon=0.1 on every
policy action. Episodes stop on success in either env, which is why they run ~85 steps not 500.
Cross-family sources are drawn only from {window-open, push, drawer-close}; Window Close never appears.
`m2_generate_prefs.py` refuses `--task window-close` outright.

## Results (25 variations, 150k pairs)
Behavior sources form the intended quality ladder:

| source | episodes | success | mean segment return | median |
|---|---|---|---|---|
| expert | 375 | 1.00 | 17.83 | 15.38 |
| within | 625 | 0.10 | 14.23 | 13.76 |
| cross | 250 | 0.00 | 11.04 | 10.83 |
| random | 50 | 0.00 | 11.07 | 10.87 |

Label balance 0.4982. Expert beats random in 97.1% of expert-vs-random pairs.
Regenerating a variation with the same seed reproduces byte-identical arrays.

## Calibration note: why 97% and not 100%
MetaWorld's reward carries a large always-on reaching term, so any 25-step segment scores about 10
regardless of behavior. Expert segments bottom out near 12.1 and random segments top out near 12.5, so the
distributions overlap slightly at the start of expert episodes, before the arm has moved. Two initial
verification thresholds were wrong for this reason and were replaced with principled ones: a ratio test
("expert 2x random") is unreachable by construction and became a separation test (expert median above
random p90), and the 99% expert-vs-random bar became 95%.

## Deviations from the paper, deliberate
- **Labels use a plain sum**, matching the paper text and this project's notes. The reference code applies
  gamma=0.99 over the 25 steps. `--discount 0.99` reproduces the reference exactly.
- **25 variations per family**, matching the reference's `tasks-per-env=25`, giving 75 meta-training tasks
  across our three families.
- Our on-disk format is one npz per (family, variation) with a pair table of indices, not the reference's
  flat replay-buffer directories. Not byte-compatible, same content.

## Gotchas recorded for later milestones
- `env.observation_space` is built before `set_task` and keeps partially-observable bounds, with the goal
  dims pinned to zero. `env.sawyer_observation_space` is the correct one. Any normalization written
  against the former would silently zero the goal.
- Window Open, Window Close and Drawer Close policies read only `obs[0:3]` and `obs[4:7]` and derive the
  target from a hardcoded offset, so goal spoofing does nothing to them. Push does read `obs[36:39]`.
- Scripted policies do not clip their own output; we clip before recording.
- Reward-model settings from the reference, for Milestone 3: raw obs+action concat (43 inputs), no
  normalization anywhere, 3x256 leaky ReLU, tanh output, ensemble of 3, summed over the segment,
  `BCEWithLogitsLoss` on `r2 - r1`. MAML uses 2 inner steps with learned per-parameter inner rates.
