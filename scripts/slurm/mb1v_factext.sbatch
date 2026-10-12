#!/bin/bash
#SBATCH --job-name=bci-mb1v
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%A_%a.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%A_%a.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --array=0-9%1
#SBATCH --requeue

# MB-1v: extend the 2x2 factorial from n=5 to n=10.
#
# The factorial is now the paper's HEADLINE -- it is the result that says tightening the adjacency
# guard is not a fix and at one budget makes things worse. It currently rests on five subjects, which
# is fine for a supporting appendix and not fine for a lead. The two cells that need subjects are the
# off-diagonal ones this job created:
#     cell C  recording guard + story-level split
#     cell D  stimulus guard  + recording-level split
# The diagonal cells (mb1f, mb1k) already have ten.
#
# tasks 0-4: cell C on the five missing subjects; tasks 5-9: cell D on the same.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"

NEW=(03 09 10 11 13)
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -lt 5 ]; then
  S=${NEW[$I]};       GUARD=recording; SPLIT=story;     TAG="mb1t_cellC_s${S}"
else
  S=${NEW[$((I-5))]}; GUARD=stimulus;  SPLIT=recording; TAG="mb1t_cellD_s${S}"
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
