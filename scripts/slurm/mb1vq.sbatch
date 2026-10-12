#!/bin/bash
#SBATCH --job-name=bci-mb1vq
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=10:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

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

# n=10 VQ row for Table 7: same full-res substrate as MB-1e, variants conv / dynamic_big (flow)
# / dynamic_big_vq (flow + VQ). Closes the VQ claim at n=10 (was only 1-3 subjects before).
PRE=${PRE:-80}; ADAPT=${ADAPT:-120}
SUBJECTS=${SUBJECTS:-"01 02 03 05 07 08 09 10 11 13"}
TAGSFX=${TAGSFX:-""}
echo "[INFO] PRE=$PRE ADAPT=$ADAPT SUBJECTS=$SUBJECTS"
for S in $SUBJECTS; do
  if [ -f "$OUT/result_mb1vq${TAGSFX}_s${S}.json" ]; then echo "[skip] $S"; continue; fi
  echo "[$(date -Is)] === MB-1 VQ subject $S ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big,dynamic_big_vq --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 500 --pre_epochs "$PRE" --adapt_epochs "$ADAPT" \
      --tag "mb1vq${TAGSFX}_s${S}"
done
echo "[$(date -Is)] DONE"
