
TLDR: Developed a neural network reward model and reinforcement learning pipeline in a robotics simulator, matching
baseline policy performance with 4× fewer preference comparisons than prior methods

Trained on 10 prior tasks:
1. reach
2. push
3. pick-place
4. door-open
5. drawer-close
6. button-press-topdown
7. peg-insert-side
8. window-open
9. sweep
10. basketball

Then used few-shot preference learning to train the robot to close a window, atching
baseline policy performance with 4× fewer preference comparisons than prior methods (PEBBLE)


# Few-Shot Preference Learning for Human-in-the-Loop RL (simplified reproduction)

Incremental reproduction of Hejna & Sadigh, CoRL 2022 (arXiv 2212.03363) on MetaWorld.
Prior tasks: Window Open, Push, Drawer Close. Held-out task: Window Close.

## Layout
- `fspref/envs.py` - MetaWorld env + scripted-policy helpers (all version quirks live here)
- `scripts/m1_run_scripted.py` - Milestone 1: roll out a scripted expert, save transitions
- `carc/env.sh`, `carc/setup_carc.sh`, `carc/*.sbatch` - cluster environment and jobs


