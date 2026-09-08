#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6b-maml
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# Milestone 6b: MAML meta-training over the 250 ten-family prior tasks.
# Same script and hyperparameters as Milestone 5 -- only the data root changes,
# which is the point: the prior set is the single variable under test.
#
# 4 CPUs / 12G, not 8 / 32G. Measured from a real dataset file, a variation is
# ~0.9MB of obs+action, so 250 of them are well under 2GB resident even counting
# the numpy and torch copies separately. The first submission asked for 32G,
# which on a full main partition (0 idle nodes) left it stuck behind smaller
# jobs for no reason -- MAML is the critical path for everything downstream.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4
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
