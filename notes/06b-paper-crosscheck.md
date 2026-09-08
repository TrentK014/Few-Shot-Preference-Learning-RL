# Paper cross-check (arXiv 2212.03363v1, full text + appendix)

Read the actual PDF (23 pages, pypdf extraction) rather than the arXiv HTML, to check
Milestone 6's setup against the paper before spending cluster time on the 6b plan.

## Matches the paper exactly

Verified line by line against Appendix B.1 (p.17-18) and Appendix C (p.20-21).

| item | paper | ours |
|---|---|---|
| behavior mixture | 15 expert, 25 same-family parametric, 10 cross-family, 2 uniform random | `DEFAULT_MIXTURE` identical |
| action noise | 0 mean, 0.1 std Gaussian | `DEFAULT_EPSILON = 0.1` |
| queries per variation | 6000, sampled uniformly at random | `N_PAIRS = 6000` |
| segment size | 25 (MetaWorld) | `SEGMENT_SIZE = 25` |
| goal observability | "goal unobserved" mode | `partially_observable=True`, `obs[36:39]` zeroed |
| reward arch | 3x256 dense, tanh output, ensemble 3 | same |
| MAML | outer 1e-4, inner 1e-3, support 32, query 32, task batch 4, learned inner LR | `fspref/maml.py` defaults, all identical |
| adaptation | train to 95% accuracy, max 40 MAML steps, then Adam fallback | `fspref/adapt.py`, identical |
| reset before re-adapt | "we crucially reset the reward model for adaptation" | `maml_arm` resets to the checkpoint |
| Init baseline | pretrained weights + standard Adam | `init_arm` |
| prior task set | Reach, Push, Sweep, Door Open, Insert, Window Open, Drawer Close, Basketball, Pick Place, Button Top (Fig. 2 header) | == the ML10 train list in the 6b plan |
| Window Close | held-out test task, 200 total queries | same |

Window Open **is** in the paper's pretraining set, so the "reversed sibling" situation is
theirs too, and it does not stop them from transferring to Window Close. That is evidence
against reversal being inherently fatal, and for it being fatal only at 1-of-3 dilution.

Cosmetic difference: the paper's Table 2 says ReLU, we use leaky ReLU. Not worth churning.

## Four real differences

### 1. The paper reports no preference accuracy at all
Figures 2, 3, 6, 7, 8 and Table 1 are success rate, reward, or meters-to-goal versus
environment steps or total feedback. There is no preference-accuracy number anywhere in
the 23 pages. So Milestone 6's flat accuracy curve does not contradict anything the paper
claims. Milestone 7 produces the first number that is comparable to theirs.

### 2. The 200-query budget is 25 sessions of 8, not one adaptation of 200
Table 4 (p.21): Window Close is Max Feedback **200**, Feedback Per Session **8**, Session
Frequency **5000** steps. The model is reset to the pretrained weights and re-adapted on
all accumulated data every session, so the adaptation budgets actually seen over a run are
8, 16, 24, ..., 200 -- and the early ones are the ones that shape the policy while it is
still bad.

Milestone 6 tested 25/50/100/200, i.e. only the late sessions. The follow-up low-budget
sweep tested 4/8/16/25 and found MAML ahead of scratch by +0.062, +0.025, +0.016 before
crossing zero at 25. **The low-budget sweep is measuring the operative regime; Milestone 6
was measuring the tail.** This reframes the "refutation" substantially.

### 3. Ten prior task families, not three
"Our reward models are pre-trained using only 10 prior tasks and evaluate query-efficiency
on six previously unseen tasks" (p.6). Confirmed by the Figure 2 header row. This is the
one structural difference between our setup and theirs, and it is what Stage 1 fixes.

### 4. Queries are online and disagreement-selected
Queries come from the SAC replay buffer of the policy being trained, and after the first
session are chosen to maximize ensemble disagreement std(P[s1 > s2]) with a Disagreement
Sample Multiplier of 10 (Table 3). Our support pairs are drawn uniformly from an offline
mixture that contains scripted experts. `disagreement()` already exists in
`fspref/reward_model.py` and is currently unused. Relevant to Stage 2 and Milestone 7.

## Correction to the 6b plan

The plan proposed dropping to 15 variations per family to hold memory down, estimating
~48G for 250 tasks. Measured from an actual dataset file
(`data/local_test/prefs/window-open/var00.npz`): 5116 steps x (39+4) float32 = **0.88 MB**
per variation, so 250 variations is about **0.44 GB** resident even counting the numpy and
torch copies separately. The estimate was wrong by roughly two orders of magnitude.

**Use the paper's 25 variations per family: 10 x 25 = 250 tasks.** 16G is ample.
