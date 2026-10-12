#!/bin/bash
#SBATCH --job-name=bci-m1a-nuisstory
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/meg_masc_full
M0=$SCRATCH/project/results/m0
OUT=$SCRATCH/project/results/m1
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Nuisance/low-level control in the CONFOUND-CONTROLLED setting (held-out story + crossed),
# matching train_story_holdout.py. Decisive test: timing-only encoder on a story whose targets
# were never in training. Expect ~chance if the brain effect (held-out-story 2.79x, crossed
# 2.49x) is genuinely neural.
EPOCHS=${EPOCHS:-80}
TASKS=${TASKS:-"0 1 2 3"}
SUBJECTS=${SUBJECTS:-"01 02 03 05 07 08 09 10 11 13"}
TAGSFX=${TAGSFX:-""}
MODE=${MODE:-"both"}    # story | crossed | both
ADJ=${ADJ:-10.0}
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --layer 6 --seg_s 3.0 --min_words 3 --adj_s $ADJ --epochs $EPOCHS --batch 256"
echo "[INFO] EPOCHS=$EPOCHS TASKS=$TASKS MODE=$MODE ADJ=$ADJ"

if [ "$MODE" = "story" ] || [ "$MODE" = "both" ]; then
  for T in $TASKS; do
    OUTF="$OUT/result_nuis_story${TAGSFX}_task${T}.json"
    if [ -f "$OUTF" ]; then echo "[skip] story task$T"; continue; fi
    echo "[$(date -Is)] === nuisance held-out-story task=$T ==="
    $PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task "$T" \
        --tag "nuis_story${TAGSFX}_task${T}"
  done
fi

if [ "$MODE" = "crossed" ] || [ "$MODE" = "both" ]; then
  for T in $TASKS; do
    for S in $SUBJECTS; do
      OUTF="$OUT/result_nuis_crossed${TAGSFX}_task${T}_subj${S}.json"
      if [ -f "$OUTF" ]; then echo "[skip] crossed task$T subj$S"; continue; fi
      echo "[$(date -Is)] === nuisance crossed task=$T subj=$S ==="
      $PY "$CODE/train_nuisance_story.py" $COMMON --holdout_task "$T" --holdout_subject "$S" \
          --tag "nuis_crossed${TAGSFX}_task${T}_subj${S}"
    done
  done
fi
echo "[$(date -Is)] DONE"
