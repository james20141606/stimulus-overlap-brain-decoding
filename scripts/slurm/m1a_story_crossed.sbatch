#!/bin/bash
#SBATCH --job-name=bci-crossed
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=08:00:00
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

# CROSSED control (ARIS R4's one required experiment): hold out subject AND story
# simultaneously. Train on (other subjects x other stories); eval the held-out subject on
# the held-out story (target NEVER in training, subject NEVER in training). Parallels §4.2
# LOSO but with the story also held out. Stories 2,3 chosen for the largest per-subject pools.
EPOCHS=${EPOCHS:-80}
STORIES=${STORIES:-"2 3"}
SUBJECTS=${SUBJECTS:-"01 02 03 05 07 08 09 10 11 13"}
TAGSFX=${TAGSFX:-""}
echo "[INFO] EPOCHS=$EPOCHS STORIES=$STORIES SUBJECTS=$SUBJECTS TAGSFX=$TAGSFX"

for K in $STORIES; do
  for S in $SUBJECTS; do
    OUTF="$OUT/result_story_crossed_task${K}_subj${S}${TAGSFX}.json"
    if [ -f "$OUTF" ]; then
      echo "[$(date -Is)] [skip] story $K subj $S done"; continue
    fi
    echo "[$(date -Is)] === CROSSED: hold out story=$K subj=$S (epochs=$EPOCHS) ==="
    $PY "$CODE/train_story_holdout.py" --out_dir "$OUT" \
        --data_dir "$R/meg_papers/data/meg_masc_full" \
        --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
        --holdout_task "$K" --holdout_subject "$S" --layer 6 --seg_s 3.0 --min_words 3 \
        --decim 2 --epochs "$EPOCHS" --tag "story_crossed_task${K}_subj${S}${TAGSFX}"
  done
done

echo "[$(date -Is)] DONE"
