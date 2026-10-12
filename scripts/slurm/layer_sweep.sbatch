#!/bin/bash
#SBATCH --job-name=bci-layer-sweep
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1 --time=12:00:00
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
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

# TARGET BREADTH: the paper tests exactly one target (GPT-2 layer 6), so a reviewer can ask whether
# the confound is a quirk of that layer. The ctx tensor already holds layers 4/6/8/10/12, so this
# costs one training per layer per arm and buys a real breadth axis. Both arms, held-out story 0.
LAYERS=${LAYERS:-"4 8 10 12"}
EPOCHS=${EPOCHS:-80}
T=${T:-0}
COMMON="--data_dir $DATA --ctx $M0/masc_gpt2_contextual.npy --ctx_meta $M0/masc_gpt2_contextual_meta.json --out_dir $OUT --seg_s 3.0 --min_words 3 --adj_s 10.0 --epochs $EPOCHS --batch 256 --holdout_task $T"
echo "[INFO] LAYERS=$LAYERS T=$T EPOCHS=$EPOCHS"

for L in $LAYERS; do
  OUTF="$OUT/result_nuis_L${L}_task${T}.json"
  if [ ! -f "$OUTF" ]; then
    echo "[$(date -Is)] === TIMING layer=$L task=$T ==="
    $PY "$CODE/train_nuisance_story.py" $COMMON --layer "$L" --tag "nuis_L${L}_task${T}"
  else echo "[skip] timing L$L"; fi
  OUTF="$OUT/result_brain_L${L}_task${T}.json"
  if [ ! -f "$OUTF" ]; then
    echo "[$(date -Is)] === BRAIN layer=$L task=$T ==="
    $PY "$CODE/train_story_holdout.py" $COMMON --layer "$L" --decim 2 --tag "brain_L${L}_task${T}"
  else echo "[skip] brain L$L"; fi
done
echo "[$(date -Is)] DONE"
