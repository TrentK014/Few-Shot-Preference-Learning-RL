#!/usr/bin/env bash
#SBATCH --job-name=fspref-m7pilot
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out

# Milestone 7 pilot on the EXISTING 3-family Milestone 5 checkpoint.
#
# Runs while the 10-family data generates. Two jobs at once:
#   1. proves the whole M7 pipeline works on the cluster GPU end to end
#   2. gives the 3-family success-rate number, which is the baseline the
#      10-family run has to beat -- the 3-vs-10 comparison the diagnosis calls for
#
# 200k steps rather than 500k, 2 seeds rather than 3: enough to see whether the
# curves separate, cheap enough to not block the real run.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

python scripts/m7_sac.py \
    --checkpoint "$FSPREF_ROOT/runs/m5/maml_init.pt" \
    --out "$FSPREF_ROOT/runs/m7_pilot_3family" \
    --arms sac_oracle few_shot init pebble \
    --seeds 2 --steps 200000 \
    --max-feedback 200 --per-session 8 --session-freq 5000 \
    --eval-every 10000 --eval-episodes 10 \
    --oracle-floor 0.5
echo "M7_PILOT_DONE status=$?"
