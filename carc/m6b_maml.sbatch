#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6b-maml
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# Milestone 6b: MAML meta-training over the 250 ten-family prior tasks.
# Same script and hyperparameters as Milestone 5 -- only the data root changes,
# which is the point: the prior set is the single variable under test.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python scripts/m5_train_maml.py \
    --root "$FSPREF_ROOT/data/prefs10" \
    --out "$FSPREF_ROOT/runs/m6b" \
    --held-per-family 5 \
    --meta-steps "${META_STEPS:-15000}" \
    --eval-every 1000 \
    --shots 8 16 32 64 128 \
    --eval-pairs 1000
echo "M6B_MAML_DONE status=$?"
