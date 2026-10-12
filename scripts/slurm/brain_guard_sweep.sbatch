#!/bin/bash
#SBATCH --job-name=bci-brain-sweep
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=03:00:00
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

# BRAIN held-out-story task 0, one training, evaluated at guards {10,30,60,120,240} via
# --adj_multi. Compared against nuis_guard_sweep: at the guard where the timing-only baseline
# collapses to ~chance, does the brain retrieval stay above chance? (= language signal beyond
# slow temporal/position structure).
OUTF="$OUT/result_brain_swp_task0.json"
if [ -f "$OUTF" ]; then echo "[skip] exists"; else
  echo "[$(date -Is)] === brain held-out-story task0 multi-guard ==="
  $PY "$CODE/train_story_holdout.py" \
      --data_dir "$DATA" --ctx "$M0/masc_gpt2_contextual.npy" \
      --ctx_meta "$M0/masc_gpt2_contextual_meta.json" --out_dir "$OUT" \
      --layer 6 --seg_s 3.0 --min_words 3 --adj_s 10.0 --adj_multi "10,30,60,120,240" \
      --holdout_task 0 --epochs 80 --batch 256 --tag "brain_swp_task0"
fi
echo "[$(date -Is)] DONE"
