# Few-Shot Preference Learning for Human-in-the-Loop RL (simplified reproduction)

Incremental reproduction of Hejna & Sadigh, CoRL 2022 (arXiv 2212.03363) on MetaWorld.
Prior tasks: Window Open, Push, Drawer Close. Held-out task: Window Close.

Everything runs on USC CARC Discovery (see `carc/`). Milestone log lives in `notes/`.

## Layout
- `fspref/envs.py` - MetaWorld env + scripted-policy helpers (all version quirks live here)
- `scripts/m1_run_scripted.py` - Milestone 1: roll out a scripted expert, save transitions
- `carc/env.sh`, `carc/setup_carc.sh`, `carc/*.sbatch` - cluster environment and jobs

## CARC quick start
**USC VPN is required.** The CARC hostnames (`discovery.usc.edu`, `discovery1.hpc.usc.edu`, ...)
have no public DNS records, so off-VPN `ssh discovery` fails with "Could not resolve hostname".
Reconnect the VPN, then log in again.

```bash
ssh discovery                                   # once, VPN + Duo; warms the ControlMaster socket
rsync -av --exclude .git --exclude data . discovery:~/fspref-carc/
ssh discovery 'bash ~/fspref-carc/carc/setup_carc.sh'   # login node, one time
ssh discovery 'cd ~/fspref-carc && sbatch carc/m1_smoke.sbatch'
```
