#!/bin/bash
#SBATCH --job-name=bci-mb1s
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-19%1
#SBATCH --requeue

# MB-1s: extend from TWO withheld stimuli to ALL FOUR. Mandatory robustness, not optional.
#
# Every current claim rests on stories 0 and 3. With two stimuli the effective stimulus sample is
# n=2 while all the uncertainty reported is subject-level -- the objection a statistics reviewer makes
# first, and the reason the paper currently says "the two stories tested" rather than making a
# stimulus-general claim. Stories 1 and 2 are already in the corpus; not running them is the cheapest
# unforced weakness in the paper.
#
# For each of stories 1 and 2 this runs the matched pair the magnitude claim needs:
#   SEEN   pretrain excludes a DIFFERENT story, so the eval story IS in pretraining
#   UNSEEN pretrain excludes the eval story itself
# Both evaluate the same story, both pretrain on three stories, so the pair is volume-matched and only
# the presence of the evaluation stimulus differs -- the same design as mb1k vs mb1j0.
#
# Tasks 0-9  : story 1 (SEEN then UNSEEN, 5 subjects each)
# Tasks 10-19: story 2 (SEEN then UNSEEN, 5 subjects each)
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

SUBS=(01 02 05 07 08)
I=$SLURM_ARRAY_TASK_ID
if   [ "$I" -lt 5 ];  then EV=1; HOLD=3; ARM=seen;   S=${SUBS[$I]}
elif [ "$I" -lt 10 ]; then EV=1; HOLD=1; ARM=unseen; S=${SUBS[$((I-5))]}
elif [ "$I" -lt 15 ]; then EV=2; HOLD=3; ARM=seen;   S=${SUBS[$((I-10))]}
else                       EV=2; HOLD=2; ARM=unseen; S=${SUBS[$((I-15))]}
fi
TAG="mb1s_story${EV}_${ARM}_s${S}"
echo "[INFO] host=$(hostname) subject=$S eval_story=$EV holdout=$HOLD arm=$ARM tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --holdout_story "$HOLD" --eval_story "$EV" --guard_scope stimulus \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
