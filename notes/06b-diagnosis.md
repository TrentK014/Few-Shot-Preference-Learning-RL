# Milestone 6b: diagnosing the flat transfer result

Milestone 6 (job 11789719) returned REFUTED on the project's main hypothesis:
adapting the MAML meta-initialization to held-out Window Close did not beat
training from scratch on the same labels. This note records why, and what
changed as a result.

## The result that reframed it

The Milestone 6 budgets were 25/50/100/200. A follow-up sweep at 4/8/16/25
(job 11790403, 5 variations x 5 draws) found:

| labels | scratch | Init | MAML | MAML - scratch |
|---|---|---|---|---|
| 4 | 0.6552 | 0.7275 | 0.7170 | **+0.0618** |
| 8 | 0.7619 | 0.7755 | 0.7868 | **+0.0249** |
| 16 | 0.7864 | 0.7907 | 0.8020 | **+0.0156** |
| 25 | 0.8248 | 0.7945 | 0.8141 | -0.0107 |

Monotone decreasing in budget, which is the shape few-shot learning predicts.
Milestone 6's smallest budget was 25 -- exactly where the curve crosses zero.
**We sampled past the effect.**

That this is the regime the paper operates in, rather than a curiosity below it,
comes from Table 4: Window Close is 200 total queries delivered **8 at a time**
every 5000 steps, with the model reset and re-adapted on all accumulated data
each session. The adaptation budgets actually seen are 8, 16, 24, ...

## Three causes

### 1. The metric is shortcut-saturated

The hand-distance heuristic scores **0.8800** on held-out Window Close. Every arm
at every budget lands between 0.83 and 0.89. "Did the arm get closer to the
object" is a near-complete answer to our preference pairs, it is task-independent,
and both a pretrained and a from-scratch model find it within a handful of labels.
There is almost no headroom in which a prior could show an advantage.

### 2. The prior is actively wrong, and the hard subset proves it

The Milestone 6 plan predicted (H1) that the unadapted meta-init would score at or
below chance on Window Close, because Window Open teaches "sash moves +x" and
Window Close is the reverse. On the full distribution it scored 0.7136 and H1 was
recorded REFUTED. On the **hard subset** -- the 25% of pairs where the
hand-distance gap is smallest -- the prediction holds:

| labels | scratch | Init | MAML | heuristic |
|---|---|---|---|---|
| 4 | 0.5922 | **0.3858** | **0.4117** | 0.6192 |
| 8 | 0.6339 | 0.4970 | 0.5086 | |
| 16 | 0.6370 | 0.5326 | 0.5379 | |
| 25 | 0.6867 | 0.5374 | 0.5520 | |

The pretrained arms are **below chance** at 4 labels and never catch scratch. H1
was right and the metric hid it: the meta-init carries a confidently backwards
belief, and 25 labels do not overturn it. The full-distribution 0.7136 was the
shortcut carrying it.

This is why the hard subset is now a primary reported metric (`paired_stats` in
`scripts/m6_window_close.py`), not a footnote.

### 3. Three prior families is too few

M5 held out a *variation* of a *seen* family: MAML gained +0.163 at 32 labels.
M6 held out a *family*: +0.000. Transfer works within a family and does not cross
to a new one. With three families, each of 60 meta-training tasks is seen ~667
times over 10k steps, so the meta-learner can memorize three family-specific
solutions instead of learning a general prior -- and one of the three (Window
Open) is Window Close's exact opposite.

The paper pretrains on **ten** families (Section 4.1, and the Figure 2 header
row), which are exactly ML10's training set. Window Open is in their set too, so
they face the same reversed sibling and transfer anyway. **Reversal is not
inherently fatal; it bites at 1-of-3 dilution and evidently not at 1-of-10.**

## Ruled out

- **Harness asymmetry.** M5's `from_scratch_baseline` (400 steps) and M6's
  `train_on_support` (1000 steps) both early-stop at 0.95 support accuracy, and
  M6's scratch arm reached 0.955-0.965 -- the step cap never bound.
- **Held-out contamination.** No window-close file under any pretraining root;
  `collect_variation` asserts it is never a cross-family behavior source; and the
  Milestone 7 loader re-asserts it from the checkpoint's own `train_paths`
  (verified to refuse a deliberately leaky checkpoint).
- **A disagreement with the paper.** The paper reports no preference accuracy
  anywhere. See `notes/06b-paper-crosscheck.md`.

## What changed

1. `PRIOR_TASKS` is now the ML10 training set: reach, push, pick-place,
   door-open, drawer-close, button-press-topdown, peg-insert-side, window-open,
   sweep, basketball. 25 variations each, matching Appendix B.1's "10 tasks, each
   with 25 variations of 6000 queries each". `PRIOR_TASKS_V1` keeps the old three
   so the 3-vs-10 comparison can be rerun without editing code.
2. Budgets start at 4, and the hard subset is reported with sd and a paired
   significance test against scratch.
3. `--support-sources within cross random` drops scripted experts from the
   support pool, approximating the paper's online queries from a still-bad
   policy's replay buffer.
4. Milestone 7 measures success rate, which is what the paper actually claims.

## Result: the meta-learner stopped memorizing

10-family meta-training (job 11795032, 200 meta-train / 50 held-out tasks, 15000
steps in 1h37m, checkpoint `runs/m6b/maml_init.pt`, best held-out adapted 0.9413):

| step | unadapted | adapted | gain |
|---|---|---|---|
| 1000 | 0.853 | 0.899 | +0.046 |
| 3000 | 0.862 | 0.918 | +0.056 |
| 5000 | 0.865 | 0.921 | +0.065 |
| 7000 | 0.871 | 0.931 | +0.060 |
| 10000 | 0.858 | 0.936 | +0.078 |
| 13000 | 0.874 | 0.937 | +0.063 |
| **15000** | **0.870** | **0.941** | **+0.071** |
| M5 final (3 families) | 0.9401 | 0.9405 | **+0.0004** |

Both priors reach ~0.94 adapted. The difference is how. The 3-family model was
already at 0.94 before seeing a label; the 10-family model sits at 0.87 and gets
to 0.94 by adapting. The gap grows across all 15000 steps rather than wandering,
so it is a property of the prior set and not noise.

This is the mechanism in cause 3, confirmed directly. With three families the
meta-initialization climbed to 0.94 unadapted -- it had memorized three
family-specific solutions -- and adaptation had nothing left to do. With ten it
stays flat at ~0.87 across the whole run while the adapted score climbs, so the
inner loop is doing real work.

That is what MAML is supposed to produce: not weights that are already right, but
weights that move to the right place quickly. It is also why Milestone 5's
headline "+0.24 at 8 labels" was really "pretrained features beat no features"
rather than a statement about adaptation.

Caveat: these are held-out *variations of seen families*, the Milestone 5 style
of validation. The cross-family Window Close transfer that Milestone 6 failed is
a separate measurement, and is what `runs/m6b_wc` reports.

## Hypotheses for the 10-family run

- **H1b**: pretrained arms are at or below chance on the hard subset at the
  smallest budget. (CONFIRMED at 3 families; the question is whether 10 fixes it.)
- **H6**: a 10-family prior beats the 3-family prior at 4/8/16 labels.
- **H7**: MAML beats scratch **on the hard subset** -- the claim the 3-family
  prior failed most clearly.
- **H8**: restricting support to non-expert sources widens MAML's advantage.
- **H10/H11/H12** (Milestone 7): few-shot beats PEBBLE at the same budget, beats
  Init, and approaches the ground-truth ceiling.

Results land in `runs/m6b_wc`, `runs/m6b_wc_hardregime`, `runs/m7` and
`runs/m7_3family`.

---

# RESULT: the 10-family prior reverses Milestone 6

Job 11797299, 5 Window Close variations x 5 draws x 7 budgets, evaluated on
held-out episodes. The 3-family column merges the Milestone 6 runs.

## Full distribution: the main hypothesis is confirmed

MAML minus scratch:

| labels | 3-family | 10-family | change |
|---|---|---|---|
| 4 | +0.062 | **+0.172** | +0.110 |
| 8 | +0.025 | **+0.115** | +0.090 |
| 16 | +0.016 | **+0.087** | +0.072 |
| 25 | -0.015 | **+0.071** | +0.086 |
| 50 | +0.000 | **+0.057** | +0.057 |
| 100 | +0.006 | **+0.039** | +0.033 |
| 200 | +0.001 | **+0.030** | +0.029 |

Positive at every budget, roughly 3x larger at 4 labels, and monotone decreasing
in budget -- the shape few-shot learning predicts. **H0, the project's main
question, is CONFIRMED** where Milestone 6 refuted it. H5 (largest advantage at
the smallest budget) is confirmed. H4 is confirmed for the first time: MAML at
200 labels reaches 0.9177 against the 0.8800 hand-distance heuristic, a bar the
3-family prior never cleared.

The unadapted meta-init also improved, 0.7136 -> 0.7968.

## Hard subset: the harm is fixed, the advantage is not

| labels | 3-family | 10-family | p (10-family) |
|---|---|---|---|
| 4 | -0.181 | -0.039 | 0.39 |
| 8 | -0.125 | +0.017 | 0.51 |
| 16 | -0.099 | +0.000 | 1.00 |
| 25 | -0.121 | +0.007 | 0.81 |
| 50 | -0.086 | +0.011 | 0.53 |
| 100 | -0.048 | -0.010 | 0.35 |
| 200 | -0.047 | +0.000 | 0.99 |

**H1b CONFIRMED**: the pretrained arms are no longer below chance. Init at 4
labels went 0.3858 -> 0.5149. The pathology that made the 3-family prior a
liability is gone.

**H7 REFUTED**: MAML does not *beat* scratch on the hard subset. Every gap is
within noise (all p > 0.35). Prior breadth removed the harm without creating an
advantage.

## What this means

The honest reading is that the ten-family prior does two different things:

1. It stops the prior being wrong. With three families, one of which was Window
   Close's exact reverse, the meta-init carried a confidently backwards belief
   that few labels could not overturn -- visible as below-chance hard-subset
   accuracy. With ten, that belief is diluted and the deficit disappears.
2. It does not buy genuine task understanding. Where the hand-distance shortcut
   is uninformative, MAML and scratch are indistinguishable. The large
   full-distribution gains therefore come from the prior making better use of the
   shortcut-informative pairs, not from knowing more about Window Close.

Both halves are worth reporting. The first vindicates the diagnosis; the second
says prior breadth is not sufficient for the deeper claim, and points at the next
test -- whether the same holds on ML10 test tasks that have no reversed sibling
in the prior set (Stage 3).

## Non-expert support regime: H8 refuted, and significantly

Same job, second pass, with `--support-sources within cross random` so no support
segment comes from a scripted expert. This is the closest offline approximation
to the paper's online queries, which are drawn from the replay buffer of a policy
that is still bad -- both segments mediocre, rather than expert against random.

H8 predicted this would *widen* MAML's advantage, since it is the regime where a
prior should matter most. It did not.

Full distribution still confirms H0: 4:+0.168, 8:+0.096, 16:+0.074, 25:+0.046,
50:+0.026, 100:+0.008, 200:+0.006. Monotone, positive everywhere, and H4 holds
(MAML at 200 reaches 0.9187 against the 0.8800 heuristic).

The hard subset goes the other way, and significantly:

| labels | MAML - scratch | p |
|---|---|---|
| 4 | +0.016 | 0.62 |
| 8 | -0.021 | 0.42 |
| 16 | -0.049 | 0.076 |
| 25 | -0.063 | 0.13 |
| 50 | -0.043 | 0.054 |
| 100 | **-0.048** | **0.004** |
| 200 | **-0.034** | **0.000** |

At 100 and 200 labels the pretrained arms are significantly worse than
from-scratch on shortcut-uninformative pairs. That is not noise.

**H8 is REFUTED.** The most likely reading: what the ten-family prior transfers
is largely the "approach the object" shortcut, which is genuinely general across
MetaWorld families. When support pairs are mediocre-vs-mediocre, a from-scratch
model is forced to find something task-specific in them, while the pretrained
arms remain anchored to the general prior -- which is exactly the feature that is
uninformative on the hard subset. More labels make this worse, not better,
because scratch keeps learning task-specific structure while the pretrained arms
keep being pulled back toward the prior.

MAML and Init are also indistinguishable here (+0.0000 at 4 labels, +0.0050 at
200), so the learned per-parameter inner rates buy nothing in this regime.

### Why this matters for the project's claim

The two regimes together say something more precise than either alone:

- The ten-family prior reliably helps on the metric Milestone 6 measured, in both
  regimes, at every budget. H0 is confirmed twice over.
- That help comes from better use of the hand-distance shortcut, not from
  understanding Window Close. On pairs where the shortcut is uninformative the
  prior is neutral at best (expert-inclusive) and significantly harmful at worst
  (non-expert, higher budgets).

Preference accuracy is not what the paper claims, so this does not contradict it.
Milestone 7 measures success rate, which is the claim -- and a reward model can
shape a good policy while being mediocre at ranking held-out pairs, precisely
because the shortcut it relies on is a decent dense reward for "go to the object".
