#!/usr/bin/env bash
#SBATCH --job-name=fspref-m7
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

# Milestone 7: SAC on held-out Window Close with the learned reward model.
#
# The first milestone whose numbers are comparable to the paper's. Everything
# before this measured preference accuracy on held-out pairs, which the paper
# never reports; Figures 2/3/6/7/8 are all success rate versus environment steps.
#
# Feedback schedule is the paper's Table 4 for Window Close: 200 total queries,
# 8 per session, a session every 5000 steps, disagreement-selected after the
# first (uniform) session.
#
# ARMS defaults to all four. The sac_oracle arm is the control: if it does not
# solve the task, the RL is broken and no preference-arm result means anything.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-$FSPREF_ROOT/runs/m6b/maml_init.pt}"
ARMS="${ARMS:-sac_oracle few_shot init pebble}"
STEPS="${STEPS:-500000}"
SEEDS="${SEEDS:-3}"

if [ ! -f "$CKPT" ]; then
    echo "FATAL: no checkpoint at $CKPT"
    echo "Run carc/m6b_generate_all.sbatch then carc/m6b_maml.sbatch first."
    exit 1
fi

python scripts/m7_sac.py \
    --checkpoint "$CKPT" \
    --out "$FSPREF_ROOT/runs/m7" \
    --arms $ARMS \
    --seeds "$SEEDS" \
    --steps "$STEPS" \
    --max-feedback 200 --per-session 8 --session-freq 5000 \
    --eval-every 10000 --eval-episodes 10
echo "M7_DONE status=$?"
