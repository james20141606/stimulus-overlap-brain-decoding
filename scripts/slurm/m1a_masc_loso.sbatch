#!/bin/bash
#SBATCH --job-name=bci-m1a-loso
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=03:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/meg_masc_full
OUT=$SCRATCH/project/results/m1
mkdir -p "$OUT" $SCRATCH/project/logs/m1
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Full 10-fold LOSO, 3-s segment, GPT-2 L6, honest (adjacency-guarded) metric.
for SUBJ in 01 02 03 05 07 08 09 10 11 13; do
  $PY "$CODE/train_m1a_seg.py" \
      --data_dir "$DATA" \
      --ctx "$SCRATCH/project/results/m0/masc_gpt2_contextual.npy" \
      --ctx_meta "$SCRATCH/project/results/m0/masc_gpt2_contextual_meta.json" \
      --out_dir "$OUT" --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 \
      --test_subject "$SUBJ" --epochs 80 --batch 256 \
      --tag "m1a_seg_s${SUBJ}_L6"
done
echo "[$(date -Is)] DONE"
