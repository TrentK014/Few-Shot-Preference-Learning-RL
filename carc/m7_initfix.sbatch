#!/usr/bin/env bash
#SBATCH --job-name=fspref-initfix
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.out

# Re-run the Init arm after fixing it to match the paper.
#
# The old `init` reset to the meta-init on every feedback session, which is
# exactly the mechanism the paper's Init is defined to LACK ("Instead of
# re-adapting the reward model each time new feedback is collected, we
# initialize the reward model with the pretrained weights, and then perform
# standard updates with the Adam optimizer as in PEBBLE", Section 4.1). That
# made our Init a strictly stronger baseline than theirs and meant the
# few_shot-vs-init comparison isolated learned inner rates rather than
# re-adaptation.
#
# Only Init is re-run; few_shot, pebble and sac_oracle never touch that branch.
#
# 15 tasks:
#   0-2   10-family, budget 200  -> runs/m7_initfix          (corrects Milestone 7)
#   3-5    3-family, budget 200  -> runs/m7_3family_initfix  (the 0.778 number)
#   6-14  10-family, budgets 25/50/100 -> runs/m8_b*_initfix (corrects Milestone 8)
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

T=$SLURM_ARRAY_TASK_ID
CKPT10="$FSPREF_ROOT/runs/m6b/maml_init.pt"
CKPT3="$FSPREF_ROOT/runs/m5/maml_init.pt"

if   [ $T -le 2 ]; then CKPT=$CKPT10; BUDGET=200; SEED=$T;         OUT="$FSPREF_ROOT/runs/m7_initfix"
elif [ $T -le 5 ]; then CKPT=$CKPT3;  BUDGET=200; SEED=$((T-3));   OUT="$FSPREF_ROOT/runs/m7_3family_initfix"
else
    K=$((T-6)); SEED=$((K%3)); BI=$((K/3))
    BUDGETS=(25 50 100); BUDGET=${BUDGETS[$BI]}
    CKPT=$CKPT10; OUT="$FSPREF_ROOT/runs/m8_b${BUDGET}_initfix"
fi

echo "task $T -> init seed=$SEED budget=$BUDGET ckpt=$(basename $(dirname $CKPT)) out=$(basename $OUT)"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }

python scripts/m7_sac.py --checkpoint "$CKPT" \
    --out "$OUT" --shard-tag \
    --arms init --seeds 1 --seed-offset "$SEED" \
    --steps "${STEPS:-500000}" \
    --max-feedback "$BUDGET" --per-session 8 --session-freq 5000 \
    --eval-every 10000 --eval-episodes 10 --device cpu
echo "INITFIX_DONE task=$T seed=$SEED budget=$BUDGET status=$?"
