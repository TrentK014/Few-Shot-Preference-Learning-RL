#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6b-wc
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

# Milestone 6b: the Window Close comparison again, under the 10-family prior.
#
# Two changes from the Milestone 6 run, both forced by what that run showed:
#
#   budgets now start at 4. M6 tested 25/50/100/200 and saw nothing, but the
#   paper's schedule is 8 queries per session (Table 4), so the model actually
#   adapts on 8, 16, 24, ... The low-budget sweep found MAML ahead by
#   +0.062/+0.025/+0.016 at 4/8/16 before crossing zero at 25 -- M6 sampled
#   past the effect entirely.
#
#   the hard subset is now a primary reported metric. On the full distribution
#   the hand-distance shortcut scores 0.88 and every arm lands near it; the hard
#   subset is where the 3-family prior was actively worse than scratch.
#
# The Window Close data itself is unchanged and is NOT regenerated -- it lives
# under data/test_prefs and has never been part of any pretraining root.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-$FSPREF_ROOT/runs/m6b/maml_init.pt}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-$FSPREF_ROOT/data/prefs10}"
TEST_ROOT="$FSPREF_ROOT/data/test_prefs"
OUT="${OUT:-$FSPREF_ROOT/runs/m6b_wc}"

echo "########## expert-inclusive support (matches Milestone 6) ##########"
python scripts/m6_window_close.py \
    --test-root "$TEST_ROOT" --pretrain-root "$PRETRAIN_ROOT" \
    --checkpoint "$CKPT" --out "$OUT" \
    --variations 5 --draws 5 --budgets 4 8 16 25 50 100 200 --eval-pairs 1000
echo "M6B_WC_DONE status=$?"

echo
echo "########## non-expert support (approximates the paper's online queries) ##########"
# The paper's queries come from a still-bad policy's replay buffer, so both
# segments are mediocre. Dropping expert episodes from the support pool is the
# closest offline approximation, and it is where a prior should matter most.
python scripts/m6_window_close.py \
    --test-root "$TEST_ROOT" --pretrain-root "$PRETRAIN_ROOT" \
    --checkpoint "$CKPT" --out "${OUT}_hardregime" \
    --variations 5 --draws 5 --budgets 4 8 16 25 50 100 200 --eval-pairs 1000 \
    --support-sources within cross random
echo "M6B_WC_HARDREGIME_DONE status=$?"
