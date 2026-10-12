#!/bin/bash
#SBATCH --job-name=bci-mb1f
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# MB-1f: the CALIBRATION SWEEP that MANIFOLD_BRIDGE.md's headline figure requires and that no run
# has produced yet. MB-1e settled the zero-calibration comparison (dynamic/flow > conv, 10/10) but
# used --klist 500, a single point. The program's stated deliverable is "x chance vs k, curves for
# D and S side by side" -- i.e. k in {100,200,500,1000,2000}. Without the sweep there is no curve,
# and the claim "dynamic wins at SMALL calibration budget" (the whole manifold-token payoff) is
# untested: k=500 alone cannot show a crossing.
#
# Identical to mb1e_solidify.sbatch except --klist. Same subjects, same full-res substrate
# (decim=1, seg_s=1.0, min_words=1), same variants, so the k=500 column should reproduce MB-1e
# within seed noise -- a built-in consistency check.
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

for S in 01 02 03 05 07 08 09 10 11 13; do
  if [ -f "$OUT/result_mb1f_ksweep_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] subject $S already done"; continue
  fi
  echo "[$(date -Is)] === held subject $S (k-sweep) ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big,dynamic_big_noflow --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,200,500,1000,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --tag "mb1f_ksweep_s${S}"
done

echo "[$(date -Is)] DONE"
