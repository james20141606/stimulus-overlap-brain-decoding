#!/bin/bash
#SBATCH --job-name=bci-story
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m1
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Held-out-STORY (cross-stimulus) control for §4.2. Hold out one story (task 0..3),
# train on the OTHER stories (all subjects), eval PER SUBJECT on the held-out story.
# EPOCHS / STORIES overridable for a smoke test (e.g. --export=ALL,EPOCHS=2,STORIES=0).
EPOCHS=${EPOCHS:-80}
STORIES=${STORIES:-"0 1 2 3"}
TAGSFX=${TAGSFX:-""}
echo "[INFO] EPOCHS=$EPOCHS STORIES=$STORIES"

for K in $STORIES; do
  OUTF="$OUT/result_story_holdout_task${K}${TAGSFX}.json"
  if [ -f "$OUTF" ]; then
    echo "[$(date -Is)] [skip] story $K already done"; continue
  fi
  echo "[$(date -Is)] === held-out story task=$K (epochs=$EPOCHS) ==="
  $PY "$CODE/train_story_holdout.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --holdout_task "$K" --layer 6 --seg_s 3.0 --min_words 3 --decim 2 --epochs "$EPOCHS" \
      --tag "story_holdout_task${K}${TAGSFX}"
done

echo "[$(date -Is)] DONE"
