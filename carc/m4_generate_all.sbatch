#!/usr/bin/env bash
#SBATCH --job-name=fspref-m4
#SBATCH --account=biyik_1165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:45:00
#SBATCH --output=%x-%j.out

# Milestone 4: preference datasets for all three prior tasks.
# Same generator as Milestone 2. Window Close stays held out and the generator
# refuses it outright.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

ROOT="$FSPREF_ROOT/data/prefs"
VARIATIONS="${VARIATIONS:-25}"
# All three prior tasks are regenerated together so they share identical
# settings. Two changes since Milestone 2, both forced by measurement:
#   --success-tail 25: drawer-close reward is binary (0 through the approach,
#     10 once shut), so stopping at the success step discarded all of its signal.
#   --tie-margin 0.01: a pair the oracle cannot separate carries no information
#     and its label is arbitrary. Uninformative pairs were 74% for drawer-close
#     and under 1% for window-open; build_pairs oversamples to hold 6000 pairs.
TIE_MARGIN="${TIE_MARGIN:-0.01}"
SUCCESS_TAIL="${SUCCESS_TAIL:-25}"

for TASK in window-open push drawer-close; do
    echo "############ $TASK ############"
    python scripts/m2_generate_prefs.py --task "$TASK" --variations "$VARIATIONS" \
        --out-root "$ROOT" --tie-margin "$TIE_MARGIN" --success-tail "$SUCCESS_TAIL"
    echo
    python scripts/m2_inspect_prefs.py --root "$ROOT" --task "$TASK"
    echo
done
echo "M4_OK"
