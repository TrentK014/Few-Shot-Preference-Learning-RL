#!/usr/bin/env bash
#SBATCH --job-name=fspref-m1
#SBATCH --account=biyik_1165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

# Milestone 1 smoke test: scripted Window Open expert, 5 episodes, save transitions.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4

python scripts/m1_run_scripted.py --task window-open --episodes 5 \
    --out "$FSPREF_ROOT/data/window-open_expert.npz"
echo "=== random-action control ==="
python scripts/m1_run_scripted.py --task window-open --episodes 2 --random
echo "M1_OK"
