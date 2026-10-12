#!/bin/bash
#SBATCH --job-name=bci-mb1c
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# MB-1c: attribution (flow vs same-size no-flow) + robustness across 3 held subjects.
for S in 01 02 03; do
  echo "[$(date -Is)] === held subject $S ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big,dynamic_big_noflow --layer 6 \
      --klist 500 --pre_epochs 80 --adapt_epochs 120 \
      --tag "mb1c_masc_s${S}"
done

echo "[$(date -Is)] DONE"
