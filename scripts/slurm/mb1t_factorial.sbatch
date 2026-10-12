#!/bin/bash
#SBATCH --job-name=bci-mb1t
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1t: the 2x2 FACTORIAL that separates guard scope from calibration overlap.
#
# The paper currently attributes the collapse of the pre-registered +8.26 advantage to "the guard-scope
# and calibration-overlap change". Those are TWO changes bundled into one comparison: mb1f uses a
# recording-scoped guard AND a recording-level calibration/evaluation split, while mb1k uses a
# stimulus-scoped guard AND a story-level split. Attributing the effect to either one is unsupported.
# This is the fifth two-variable comparison built in this project; the pattern is the reason for the
# standing rule to enumerate every variable a change touches before running it.
#
#   guard \ split      recording-level        story-level
#   recording          mb1f      HAVE          cell C    THIS JOB
#   stimulus           cell D    THIS JOB      mb1k/mb1g HAVE
#
# Both new cells keep the full four-story pretraining corpus, so pretraining exposure is constant
# across the whole factorial and only the two evaluation-side factors move.
#   tasks 0-4 : cell C  recording guard + story-level split
#   tasks 5-9 : cell D  stimulus guard  + recording-level split
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
if [ "$I" -lt 5 ]; then
  S=${SUBS[$I]};       GUARD=recording; SPLIT=story;     TAG="mb1t_cellC_s${S}"
else
  S=${SUBS[$((I-5))]}; GUARD=stimulus;  SPLIT=recording; TAG="mb1t_cellD_s${S}"
fi
echo "[INFO] host=$(hostname) subject=$S guard=$GUARD split=$SPLIT tag=$TAG date=$(date -Is)"
[ -f "$OUT/result_${TAG}.json" ] && { echo "[skip] $TAG exists"; exit 0; }

$PY "$CODE/train_mb.py" --out_dir "$OUT" \
    --data_dir "$R/meg_papers/data/meg_masc_full" \
    --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
    --test_subject "$S" --variants conv,dynamic_big --layer 6 \
    --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
    --pre_epochs 80 --adapt_epochs 120 \
    --guard_scope "$GUARD" --split_by "$SPLIT" \
    --tag "$TAG"
rc=$?
echo "[$(date -Is)] $TAG rc=${rc}"
exit $rc
