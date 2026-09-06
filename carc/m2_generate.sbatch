#!/usr/bin/env bash
#SBATCH --job-name=fspref-m2
#SBATCH --account=biyik_1165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=%x-%j.out

# Milestone 2: generate Window Open preference data and verify it.
# CPU only; MetaWorld/MuJoCo simulation needs no GL context.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=4

TASK="${TASK:-window-open}"
VARIATIONS="${VARIATIONS:-25}"
ROOT="$FSPREF_ROOT/data/prefs"

python scripts/m2_generate_prefs.py --task "$TASK" --variations "$VARIATIONS" --out-root "$ROOT"
echo
python scripts/m2_inspect_prefs.py --root "$ROOT" --task "$TASK"
echo "M2_OK"
