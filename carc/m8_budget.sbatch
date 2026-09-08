#!/usr/bin/env bash
#SBATCH --job-name=fspref-m8
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-26
#SBATCH --output=%x-%A_%a.out

# Milestone 8: how few queries can few-shot preference learning get away with?
#
# Milestone 7 answered "does it work at 200 queries" -- yes, and few-shot needed
# the fewest of them. This sweeps the budget itself: 25, 50 and 100 total
# queries, same schedule shape (8 per session, a session every 5000 steps), so
# feedback simply stops earlier and SAC trains on to convergence with a frozen
# reward model. That is what the paper does in its Figure 6 ablation.
#
# The 200-query point already exists in runs/m7 and is not re-run.
#
# sac_oracle is deliberately NOT in this sweep. It never consumes feedback, so
# its result is identical at every budget; re-running it three more times would
# be nine wasted 500k-step jobs. The Milestone 7 oracle number (1.000 +/- 0.000)
# is the ceiling for every budget here.
#
# 3 budgets x 3 arms x 3 seeds = 27 tasks.
#   budget = BUDGETS[task / 9]
#   arm    = ARMS[(task % 9) / 3]
#   seed   = task % 3
#
# Low budgets are also *cheaper* for the Init arm, which runs Adam to
# convergence once per feedback session: 25 queries is 4 sessions where 200 is
# 25, so the arm that was the wall-clock bottleneck in Milestone 7 is not one
# here.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-$FSPREF_ROOT/runs/m6b/maml_init.pt}"
STEPS="${STEPS:-500000}"

BUDGETS=(25 50 100)
ARMS=(few_shot init pebble)

BUDGET=${BUDGETS[$((SLURM_ARRAY_TASK_ID / 9))]}
ARM=${ARMS[$(( (SLURM_ARRAY_TASK_ID % 9) / 3 ))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))
OUT="$FSPREF_ROOT/runs/m8_b${BUDGET}"

echo "shard $SLURM_ARRAY_TASK_ID -> budget=$BUDGET arm=$ARM seed=$SEED steps=$STEPS"

CKPT_ARG=""
if [ "$ARM" = "few_shot" ] || [ "$ARM" = "init" ]; then
    if [ ! -f "$CKPT" ]; then
        echo "FATAL: $ARM needs a checkpoint and none exists at $CKPT"
        exit 1
    fi
    CKPT_ARG="--checkpoint $CKPT"
fi

python scripts/m7_sac.py $CKPT_ARG \
    --out "$OUT" --shard-tag \
    --arms "$ARM" --seeds 1 --seed-offset "$SEED" \
    --steps "$STEPS" \
    --max-feedback "$BUDGET" --per-session 8 --session-freq 5000 \
    --eval-every 10000 --eval-episodes 10 \
    --device cpu
echo "M8_SHARD_DONE budget=$BUDGET arm=$ARM seed=$SEED status=$?"
