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
