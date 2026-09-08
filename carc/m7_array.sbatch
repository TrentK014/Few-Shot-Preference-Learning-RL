#!/usr/bin/env bash
#SBATCH --job-name=fspref-m7arr
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=%x-%A_%a.out

# Milestone 7 as a CPU array job: one (arm, seed) per task, 4 arms x 3 seeds.
#
# Why CPU rather than the gpu partition: SAC here is bound by MuJoCo stepping,
# not by the 3x256 networks, so a GPU buys little -- and the a40 queue was the
# real bottleneck. Twelve small CPU tasks schedule immediately and run in
# parallel, which finishes the whole sweep faster than one GPU job running the
# same twelve sequentially.
#
# Each shard writes m7_results_<arm>_s<seed>.json; scripts/m7_combine.py merges
# them into the single report.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-$FSPREF_ROOT/runs/m6b/maml_init.pt}"
OUT="${OUT:-$FSPREF_ROOT/runs/m7}"
STEPS="${STEPS:-500000}"

ARMS=(sac_oracle few_shot init pebble)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

echo "shard $SLURM_ARRAY_TASK_ID -> arm=$ARM seed=$SEED steps=$STEPS"

# Only the two pretrained arms need the checkpoint; sac_oracle and pebble do not,
# so those shards can run before meta-training has finished.
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
    --max-feedback 200 --per-session 8 --session-freq 5000 \
    --eval-every 10000 --eval-episodes 10 \
    --device cpu --oracle-floor 0.5
echo "M7_SHARD_DONE arm=$ARM seed=$SEED status=$?"
