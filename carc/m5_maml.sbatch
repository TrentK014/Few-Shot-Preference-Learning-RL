#!/usr/bin/env bash
#SBATCH --job-name=fspref-m5
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out

# Milestone 5: MAML meta-training over the 75 prior tasks.
# 5 variations of each family are held out of meta-training entirely; the
# acceptance test adapts on those and compares against training from scratch on
# the identical support pairs.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python scripts/m5_train_maml.py \
    --root "$FSPREF_ROOT/data/prefs" \
    --out "$FSPREF_ROOT/runs/m5" \
    --held-per-family 5 \
    --meta-steps "${META_STEPS:-10000}" \
    --eval-every 1000 \
    --shots 8 16 32 64 128 \
    --eval-pairs 1000
STATUS=$?
echo "M5_DONE status=$STATUS"
