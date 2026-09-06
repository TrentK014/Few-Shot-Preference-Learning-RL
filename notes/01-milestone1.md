# Milestone 1: MetaWorld + scripted Window Open

## Status
- Code verified locally (macOS, Python 3.11 venv, metaworld 3.1.1 / mujoco 3.3.0) on 2026-09-06.
- Pending: same run on CARC via `carc/setup_carc.sh` + `carc/m1_smoke.sbatch`.

## Local results (`scripts/m1_run_scripted.py --task window-open`)
| policy                | steps | return  | first success step | success |
|-----------------------|-------|---------|--------------------|---------|
| scripted expert       | 500   | 1451    | 78                 | 3/3     |
| expert + N(0,0.5) noise | 500 | 1437-1566 | 79-84            | 2/2     |
| uniform random        | 500   | ~235    | never              | 0/2     |
| expert, goal var 0-2  | 500   | 1662-2493 | 76-91            | 3/3     |

- obs (39,), action (4,), episode always runs to the 500-step truncation (no terminal state in MetaWorld).
- Per-step reward is in [0.4, ~9.3]; it rises from ~0.5 to ~3-5 once the window is open and stays there,
  so summed reward over a segment is a sensible proxy for "how good", as the paper's synthetic labeler assumes.
- `success` in info flips to 1 and stays 1 after the sash passes the threshold.
- With one fixed goal variation the expert is fully deterministic (identical episodes). Diversity for
  Milestone 2 must come from `--vary-goal` (50 parametric variations), action noise, and random/other-task policies.
- Rendering works (`--render` -> 480x480 GIF); frames show the sash sliding from centre to right.

## Gotchas
- gymnasium's mujoco module imports `packaging`, which pip did not pull in; it is now in requirements.txt.
- The scripted policy warns "Constant(s) may be too high" - harmless, actions are clipped to [-1, 1].
- MetaWorld 3.x requires Python >= 3.10; task ids are `*-v3`, policies `Sawyer*V3Policy`.

## CARC results (job 11772155, debug partition, CPU only, 1m05s, COMPLETED)
Identical to local, as expected from the same seeds: expert 5/5 success, first success step 78,
return 1451.44; random control 0/2, return ~235. Env there: metaworld 3.1.1, mujoco 3.3.0,
gymnasium 1.3.0, numpy 1.26.4, torch 2.14.0+cu130, Python 3.11 at `/scratch1/$USER/fspref/env`.
`/project2/biyik_1165` exists (not `/project`) and is the place for durable data later.

## Rendering on CARC
- Simulation needs no GL. `carc/env.sh` deliberately leaves MUJOCO_GL unset, because mujoco imports
  its GL backend eagerly and a bad backend breaks plain `import mujoco` (this is what failed first).
- Rendering is opt-in via `fspref_render_setup` and requires a **GPU node** (EGL from the NVIDIA driver).
  Verified on an a40 node: 500-frame 480x480 GIF of Window Open, sash slides open. Use `carc/m1_render.sbatch`.
- OSMesa is not available: CPU nodes have no libOSMesa, and conda-forge no longer ships an `osmesa` package
  (modern `mesalib` dropped it). So CPU-node rendering is not an option; use a GPU node when frames are needed.
