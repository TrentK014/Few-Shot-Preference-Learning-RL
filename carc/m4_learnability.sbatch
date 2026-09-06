#!/usr/bin/env bash
#SBATCH --job-name=fspref-m4lrn
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

# Milestone 4 acceptance: every prior task must be learnable on its own before
# MAML pools them. Runs the Milestone 3 reward model per task.
# NOTE: no `-e`. m3_train_reward.py exits non-zero when a check fails, and with
# `set -e` the first failing task aborted the job so later tasks never ran.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

ROOT="$FSPREF_ROOT/data/prefs"
STATUS=0
for TASK in window-open push drawer-close; do
    echo "############ $TASK ############"
    python scripts/m3_train_reward.py --root "$ROOT" --task "$TASK" \
        --mode per-variation --variations 4 --max-epochs 150 --patience 30 --log-every 50 \
        || { echo "TASK_FAILED $TASK"; STATUS=1; }
    echo
done

# Push is the one task whose goal is sampled independently of the object rather
# than being a fixed offset from it, and the goal is masked from the observation.
# So its reward model has to infer the target from preferences alone. Check
# whether the marginal result is underfitting rather than that difficulty.
echo "############ push, longer budget ############"
python scripts/m3_train_reward.py --root "$ROOT" --task push \
    --mode per-variation --variations 4 --max-epochs 400 --patience 60 --log-every 100 \
    || echo "TASK_FAILED push-long"

echo "############ push, pooled ############"
python scripts/m3_train_reward.py --root "$ROOT" --task push \
    --mode pooled --variations 10 --held-out-variations 3 \
    --max-epochs 300 --patience 40 --log-every 100 || echo "TASK_FAILED push-pooled"

echo "M4_LEARNABILITY_DONE status=$STATUS"
