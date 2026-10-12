#!/bin/bash
#SBATCH --job-name=bci-m0-masc
#SBATCH --output=$SCRATCH/project/logs/m0/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m0/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=03:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/meg_masc_full
OUT=$SCRATCH/project/results/m0
mkdir -p "$OUT" $SCRATCH/project/logs/m0
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# --- STEP 1: contextual GPT-2 targets for MEG-MASC (skip if cached) ---
if [ ! -f "$OUT/masc_gpt2_contextual.npy" ]; then
  echo "[$(date -Is)] STEP 1: extract contextual GPT-2 for MEG-MASC"
  $PY "$CODE/extract_contextual.py" \
      --data_dir "$DATA" --out_dir "$OUT" --model gpt2 \
      --layers 4 6 8 10 12 --tag masc_gpt2
else
  echo "[$(date -Is)] STEP 1: contextual cache exists, skipping"
fi

# --- STEP 2: cross-subject LOSO retrieval (the M0 gate), layer 12 ---
for SUBJ in 01 02 03; do
  echo "[$(date -Is)] STEP 2: LOSO held-out subject=$SUBJ (GPT-2 L12)"
  $PY "$CODE/train_infonce.py" \
      --data_dir "$DATA" \
      --ctx "$OUT/masc_gpt2_contextual.npy" \
      --ctx_meta "$OUT/masc_gpt2_contextual_meta.json" \
      --out_dir "$OUT" --layer 12 \
      --split_mode subject --test_subject "$SUBJ" \
      --t0 50 --t1 250 --decim 2 \
      --max_train 90000 --epochs 40 --batch 512 \
      --tag "masc_loso_s${SUBJ}_L12"
done

echo "[$(date -Is)] DONE"
