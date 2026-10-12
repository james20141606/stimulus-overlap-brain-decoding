#!/bin/bash
#SBATCH --job-name=bci-m0-seg
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
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# 3-s SEGMENT cross-subject LOSO (fair metric). L6 across 3 subjects + L12 check.
run() {  # $1=subject  $2=layer
  echo "[$(date -Is)] SEG LOSO subject=$1 layer=$2"
  $PY "$CODE/segment_retrieval.py" \
      --data_dir "$DATA" \
      --ctx "$OUT/masc_gpt2_contextual.npy" \
      --ctx_meta "$OUT/masc_gpt2_contextual_meta.json" \
      --out_dir "$OUT" --layer "$2" \
      --split_mode subject --test_subject "$1" \
      --seg_s 3.0 --min_words 3 --t0 50 --t1 250 --decim 2 \
      --epochs 60 --batch 256 --tag "masc_seg_s${1}_L${2}"
}

run 01 6
run 01 12
run 02 6
run 03 6

echo "[$(date -Is)] DONE"
