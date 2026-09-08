#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6low
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

# Milestone 6 follow-up. The main run found MAML tied with from-scratch at 25 to
# 200 labels. Two very different explanations fit that:
#   (a) the meta-init genuinely does not transfer to a direction-reversed task
#   (b) 25 labels was already past the point where the advantage lives
# Milestone 5 saw its largest gain at 8 labels, so this sweeps below 25.
# More draws because variance, not effect size, is the risk down here.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python scripts/m6_window_close.py \
    --test-root "$FSPREF_ROOT/data/test_prefs" \
    --pretrain-root "$FSPREF_ROOT/data/prefs" \
    --checkpoint "$FSPREF_ROOT/runs/m5/maml_init.pt" \
    --out "$FSPREF_ROOT/runs/m6_low" \
    --variations 5 --draws 5 --budgets 4 8 16 25 --eval-pairs 1000
echo "M6LOW_DONE status=$?"
