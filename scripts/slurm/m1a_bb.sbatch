#!/bin/bash
#SBATCH --job-name=bci-m1a-bb
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=04:00:00
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

# M1a LOSO re-run with ENHANCED eval: block-bootstrap CI (10-seg blocks) + 100-draw shuffled
# null. New tags (_bb) preserve the verified originals for comparison.
EPOCHS=${EPOCHS:-80}
SUBJECTS=${SUBJECTS:-"01 02 03 05 07 08 09 10 11 13"}
TAGSFX=${TAGSFX:-""}
echo "[INFO] EPOCHS=$EPOCHS SUBJECTS=$SUBJECTS"
for S in $SUBJECTS; do
  OUTF="$OUT/result_m1a_seg_bb${TAGSFX}_s${S}_L6.json"
  if [ -f "$OUTF" ]; then echo "[skip] $S"; continue; fi
  echo "[$(date -Is)] === M1a-bb subject $S (epochs=$EPOCHS) ==="
  $PY "$CODE/train_m1a_seg.py" --data_dir "$DATA" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --out_dir "$OUT" --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 \
      --test_subject "$S" --epochs "$EPOCHS" --batch 256 --tag "m1a_seg_bb${TAGSFX}_s${S}_L6"
done
echo "[$(date -Is)] DONE"
