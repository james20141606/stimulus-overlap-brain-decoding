#!/bin/bash
#SBATCH --job-name=bci-m3c-fewshot
#SBATCH --output=$SCRATCH/project/logs/m3/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m3/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=08:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition,gpuB_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m3
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

MASC="meg_masc:$R/meg_papers/data/meg_masc_full:$M0/masc_gpt2_contextual.npy:$M0/masc_gpt2_contextual_meta.json:subject=01"
LIBRI="libribrain:$R/meg_papers/data/libribrain_full:$R/meg_papers/results/p3_decode/gpt2_contextual.npy:$R/meg_papers/results/p3_decode/gpt2_contextual_meta.json:session=12/Sherlock3"

# Few-shot cross-device CALIBRATION curve: pretrain trunk on source, adapt lightweight
# per-device front-end+BN on k target segments (FM-ADAPT) vs train-from-scratch on same k.
echo "[$(date -Is)] m3c dirA: pretrain Neuromag-306 (LibriBrain) -> few-shot adapt KIT-208 (MASC)"
$PY "$CODE/train_m3c.py" --out_dir "$OUT" --layer 6 --cohort "$MASC" --cohort "$LIBRI" \
    --source_name libribrain --target_name meg_masc \
    --pre_epochs 80 --adapt_epochs 200 --klist 100,200,500,1000,2000 \
    --tag "m3c_libri2masc"

echo "[$(date -Is)] m3c dirB: pretrain KIT-208 (MASC) -> few-shot adapt Neuromag-306 (LibriBrain)"
$PY "$CODE/train_m3c.py" --out_dir "$OUT" --layer 6 --cohort "$MASC" --cohort "$LIBRI" \
    --source_name meg_masc --target_name libribrain \
    --pre_epochs 80 --adapt_epochs 200 --klist 100,200,500,1000,2000 \
    --tag "m3c_masc2libri"

echo "[$(date -Is)] DONE"
