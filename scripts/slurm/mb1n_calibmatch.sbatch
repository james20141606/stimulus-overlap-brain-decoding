#!/bin/bash
#SBATCH --job-name=bci-mb1n
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1n: CALIBRATION-DIVERSITY CONTROL for the headline comparison.
#
# In the current headline the two arms are matched on pretraining set, pool structure and calibration
# SIZE, and the evaluation story is absent from calibration in both. But the number of distinct
# calibration stories differs: SEEN (eval story 0) calibrates on {1,2}, UNSEEN (eval story 3)
# calibrates on {0,1,2}. Same k segments, different stimulus diversity. It is a second-order
# asymmetry, and "second-order" is the judgement that has been wrong repeatedly in this project, so it
# gets measured rather than argued.
#
# Here BOTH arms are pinned to calibration stories {1,2}, so the one-bit contrast really is one bit.
#   * result matches the unpinned headline -> calibration diversity is irrelevant, objection closed
#   * result differs                       -> the pinned version becomes the headline
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(01 02 05 07 08)
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then
  S=${SUBS[$SLURM_ARRAY_TASK_ID]}; EVAL=0; TAG="mb1n_seen_s${S}"
else
  S=${SUBS[$((SLURM_ARRAY_TASK_ID-5))]}; EVAL=3; TAG="mb1n_unseen_s${S}"
fi
echo "[INFO] host=$(hostname) subject=$S eval_story=$EVAL tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story 3 --eval_story "$EVAL" --calib_stories "1,2" --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
