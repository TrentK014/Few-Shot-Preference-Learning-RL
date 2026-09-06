#!/usr/bin/env bash
#SBATCH --job-name=fspref-m1-render
#SBATCH --account=biyik_1165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=%x-%j.out

# Rendering needs a GPU node: EGL comes from the NVIDIA driver. CPU nodes have
# neither EGL nor OSMesa, so simulation-only jobs must NOT call fspref_render_setup.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
fspref_render_setup

python scripts/m1_run_scripted.py --task window-open --episodes 1 --render \
    --out "$FSPREF_ROOT/data/render_test.npz"
echo "RENDER_OK ($MUJOCO_GL)"
