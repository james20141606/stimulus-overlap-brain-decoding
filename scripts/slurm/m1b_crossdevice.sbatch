#!/bin/bash
#SBATCH --job-name=bci-m1b-xdev
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=04:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition

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

MASC="meg_masc:$R/meg_papers/data/meg_masc_full:$M0/masc_gpt2_contextual.npy:$M0/masc_gpt2_contextual_meta.json:subject=01"
LIBRI="libribrain:$R/meg_papers/data/libribrain_full:$R/meg_papers/results/p3_decode/gpt2_contextual.npy:$R/meg_papers/results/p3_decode/gpt2_contextual_meta.json:session=12/Sherlock3"

echo "[$(date -Is)] Config A: MASC-only (baseline)"
$PY "$CODE/train_m1b.py" --out_dir "$OUT" --layer 6 --epochs 80 \
    --cohort "$MASC" --tag "m1b_masc_only"

echo "[$(date -Is)] Config B: LibriBrain-only (baseline)"
$PY "$CODE/train_m1b.py" --out_dir "$OUT" --layer 6 --epochs 80 \
    --cohort "$LIBRI" --tag "m1b_libri_only"

echo "[$(date -Is)] Config C: JOINT (cross-device pooling)"
$PY "$CODE/train_m1b.py" --out_dir "$OUT" --layer 6 --epochs 80 \
    --cohort "$MASC" --cohort "$LIBRI" --tag "m1b_joint"

echo "[$(date -Is)] DONE"
