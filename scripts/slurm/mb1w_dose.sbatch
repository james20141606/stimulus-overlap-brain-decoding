#!/bin/bash
#SBATCH --job-name=bci-mb1w
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-19%1
#SBATCH --requeue

# MB-1w: DOSE-RESPONSE. Estimate the leakage curve directly instead of inferring it from endpoints.
#
# Every magnitude result so far compares two endpoints -- evaluation story fully in pretraining versus
# fully absent -- and each endpoint design carries a confound (substituting a story changes
# composition; dropping one changes volume). Triangulating them brackets the effect but does not
# measure its shape. Varying the DOSE does: keep a fraction of the other subjects' exposure to the
# evaluation story and backfill the segment count, so pretraining volume, subject composition and
# optimisation are constant and only the amount of exposure moves.
#
# Doses 0.00 / 0.25 / 0.50 / 0.75 / 1.00 on four subjects. Task 0 runs dose=1.0 first and prints the
# segment count that the remaining doses backfill to; DOSE_TARGET below is that number, filled in
# after the first task reports it (0 = no backfill, which the analysis will detect and flag).
#
# A monotone curve is the claim the endpoint designs cannot make: it shows the effect scales with
# exposure rather than appearing only when exposure is removed entirely.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
# DOSE_TARGET is the dose=1.0 segment count; every lower dose is resampled up to it so that
# pretraining VOLUME does not co-vary with dose. The first run left this at 0, which silently disabled
# the backfill: segments fell 55753 -> 51136 (8.3%) from dose 1.0 to 0.0, so the curve confounded
# exposure with data volume -- the exact confound the design exists to remove. Worse, the log line
# still announced that the count "is restored", because that text was written for the intended
# behaviour rather than the actual one. 55753 is the observed dose=1.0 count.
DOSE_TARGET=${DOSE_TARGET:-55753}
mkdir -p "$OUT"

SUBS=(01 02 05 07)
DOSES=(1.00 0.75 0.50 0.25 0.00)
I=$SLURM_ARRAY_TASK_ID
S=${SUBS[$((I / 5))]}
D=${DOSES[$((I % 5))]}
# TAG must follow the DESIGN, not be constant: mb1x means volume-pinned (DOSE_TARGET=55753).
# Writing a DOSE_TARGET=0 run under the mb1x prefix would poison the pinned namespace -- the exact
# mb1w/mb1x confusion that once retracted a true claim (BANNED list, memory 2026-08-10).
if [ "${DOSE_TARGET}" = "0" ]; then TAG="mb1w_dose${D}_s${S}"; else TAG="mb1x_dose${D}_s${S}"; fi
echo "[INFO] host=$(hostname) subject=$S dose=$D target=$DOSE_TARGET tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 0 --eval_story 0 --dose "$D" --dose_target "$DOSE_TARGET" \
    --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
