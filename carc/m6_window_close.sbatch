#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

# Milestone 6: the experiment. Generate the held-out Window Close data into a
# SEPARATE root, verify it, then run four arms x four label budgets.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PRETRAIN_ROOT="$FSPREF_ROOT/data/prefs"
TEST_ROOT="$FSPREF_ROOT/data/test_prefs"

echo "############ generate held-out Window Close ############"
# --allow-held-out is required, and the generator additionally refuses to write
# into the pretraining root that meta-training globs.
python scripts/m2_generate_prefs.py --task window-close --variations 25 \
    --out-root "$TEST_ROOT" --pretrain-root "$PRETRAIN_ROOT" --allow-held-out \
    --tie-margin 0.01 --success-tail 25

echo
echo "############ verify it ############"
python scripts/m2_inspect_prefs.py --root "$TEST_ROOT" --task window-close \
    --pretrain-root "$PRETRAIN_ROOT" || echo "INSPECT_FAILED"

echo
echo "############ four-arm experiment ############"
python scripts/m6_window_close.py \
    --test-root "$TEST_ROOT" --pretrain-root "$PRETRAIN_ROOT" \
    --checkpoint "$FSPREF_ROOT/runs/m5/maml_init.pt" \
    --out "$FSPREF_ROOT/runs/m6" \
    --variations 5 --draws 3 --budgets 25 50 100 200 --eval-pairs 1000
echo "M6_DONE status=$?"
