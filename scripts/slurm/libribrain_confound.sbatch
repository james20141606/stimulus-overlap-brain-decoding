#!/bin/bash
#SBATCH --job-name=bci-lb-confound
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=192G --gres=gpu:1 --time=12:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
DATA=$SCRATCH/project/meg_papers/data/libribrain_full
M0=$SCRATCH/project/results/m0
OUT=$SCRATCH/project/results/lb
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# SECOND DATASET + the regime our own paper named as its escape hatch.
# LibriBrain = ONE subject x 36 sessions of Sherlock audiobooks => the high-SNR WITHIN-SUBJECT
# setting where main.tex says "higher-SNR decoders may well clear the timing baseline". Testing it
# is therefore a test of our own limitation clause, and both outcomes are informative:
#   timing still wins  => the confound is not a low-SNR/cross-subject artifact (much broader claim)
#   brain finally wins => our scope statement is vindicated AND we get a positive result showing
#                         where decoding is real, which is a strictly better paper.
# Protocol is the MASC one, unchanged: hold out a whole story (Sherlock3), train on the others,
# evaluate per SESSION (--eval_col session, since there is only one subject) => 12 paired folds.
# LibriBrain has no speech_rate column, so the timing model here uses only features computable from
# word onsets/durations/text; --drop_srate makes MASC match that set for an apples-to-apples read.
HOLD=${HOLD:-Sherlock3}
LAYER=${LAYER:-6}
EPOCHS=${EPOCHS:-80}
ADJ=${ADJ:-10.0}
COMMON="--data_dir $DATA --ctx $M0/libribrain_gpt2_contextual.npy --ctx_meta $M0/libribrain_gpt2_contextual_meta.json --out_dir $OUT --layer $LAYER --seg_s 3.0 --min_words 3 --adj_s $ADJ --epochs $EPOCHS --batch 256 --eval_col session --holdout_task $HOLD"
echo "[INFO] HOLD=$HOLD LAYER=$LAYER EPOCHS=$EPOCHS ADJ=$ADJ"

# (1) timing-only
OUTF="$OUT/result_lb_nuis_${HOLD}_L${LAYER}.json"
if [ ! -f "$OUTF" ]; then
  echo "[$(date -Is)] === LibriBrain TIMING-only, hold $HOLD ==="
  $PY "$CODE/train_nuisance_story.py" $COMMON --adj_multi 10,30,60,120,240 \
      --tag "lb_nuis_${HOLD}_L${LAYER}"
else echo "[skip] timing"; fi

# (2) brain
OUTF="$OUT/result_lb_brain_${HOLD}_L${LAYER}.json"
if [ ! -f "$OUTF" ]; then
  echo "[$(date -Is)] === LibriBrain BRAIN, hold $HOLD ==="
  $PY "$CODE/train_story_holdout.py" $COMMON --decim 2 --adj_multi 10,30,60,120,240 \
      --tag "lb_brain_${HOLD}_L${LAYER}"
else echo "[skip] brain"; fi
echo "[$(date -Is)] DONE"
