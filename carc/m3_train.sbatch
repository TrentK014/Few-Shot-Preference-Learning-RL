#!/usr/bin/env bash
#SBATCH --job-name=fspref-m3
#SBATCH --account=biyik_1165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out

# Milestone 3: preference reward model with plain Adam. No MAML, no SAC.
# CPU is enough: the model is ~427k parameters.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
# Unbuffered, or slurm shows nothing until the job ends.
export PYTHONUNBUFFERED=1

ROOT="$FSPREF_ROOT/data/prefs"
TASK="${TASK:-window-open}"
VARIATIONS="${VARIATIONS:-10}"

echo "############ per-variation ############"
python scripts/m3_train_reward.py --root "$ROOT" --task "$TASK" \
    --mode per-variation --variations "$VARIATIONS" \
    --max-epochs 150 --patience 30 --log-every 50

echo
echo "############ pooled ############"
python scripts/m3_train_reward.py --root "$ROOT" --task "$TASK" \
    --mode pooled --variations "$VARIATIONS" --held-out-variations 3 \
    --max-epochs 150 --patience 30 --log-every 50

echo "M3_OK"
