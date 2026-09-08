#!/usr/bin/env bash
#SBATCH --job-name=fspref-m6b-gen
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=%x-%j.out

# Milestone 6b: preference data for all TEN ML10 training families.
#
# Milestone 4 used three families and Milestone 6 showed that was the problem:
# with three, the meta-learner memorizes three solutions rather than learning a
# transferable prior. The paper pre-trains on ten (Appendix B.1: "10 tasks, each
# with 25 variations of 6000 queries each"), so this matches it exactly.
#
# Writes to a NEW root so the 3-family data survives for the 3-vs-10 comparison.
# No `set -e`: the Milestone 4 lesson is that one failing family must not abort
# the other nine. Status is collected per family and reported at the end.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source carc/env.sh
fspref_activate
export OPENBLAS_NUM_THREADS=8
export PYTHONUNBUFFERED=1

ROOT="$FSPREF_ROOT/data/prefs10"
VARIATIONS="${VARIATIONS:-25}"
TIE_MARGIN="${TIE_MARGIN:-0.01}"
SUCCESS_TAIL="${SUCCESS_TAIL:-25}"

TASKS="reach push pick-place door-open drawer-close \
button-press-topdown peg-insert-side window-open sweep basketball"

declare -A GEN_STATUS
declare -A INSPECT_STATUS

for TASK in $TASKS; do
    echo "############################ $TASK ############################"
    python scripts/m2_generate_prefs.py --task "$TASK" --variations "$VARIATIONS" \
        --out-root "$ROOT" --tie-margin "$TIE_MARGIN" --success-tail "$SUCCESS_TAIL"
    GEN_STATUS[$TASK]=$?
    echo
    python scripts/m2_inspect_prefs.py --root "$ROOT" --task "$TASK"
    INSPECT_STATUS[$TASK]=$?
    echo
done

echo "=============================================================="
echo "SUMMARY (generate / inspect exit status, 0 is good)"
FAILED=0
for TASK in $TASKS; do
    G=${GEN_STATUS[$TASK]:-999}
    I=${INSPECT_STATUS[$TASK]:-999}
    printf "  %-24s gen=%-3s inspect=%-3s\n" "$TASK" "$G" "$I"
    [ "$G" -ne 0 ] && FAILED=$((FAILED+1))
done
echo
echo "datasets written: $(find "$ROOT" -name '*.npz' | wc -l) (expect $((10 * VARIATIONS)))"
# Held-out isolation is a hard project constraint, so assert it here too rather
# than trusting the generator's own refusal.
LEAK=$(find "$ROOT" -name '*window-close*' -o -path '*window-close*' -name '*.npz' | wc -l)
echo "window-close files under the pretraining root: $LEAK (MUST be 0)"
[ "$LEAK" -ne 0 ] && FAILED=$((FAILED+1))
echo "M6B_GEN_DONE failed=$FAILED"
