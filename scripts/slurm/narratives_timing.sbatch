#!/bin/bash
#SBATCH --job-name=bci-narr-timing
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1 --time=06:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

set -euo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
META=$SCRATCH/project/fmri/narratives_meta
OUT=$SCRATCH/project/results/narr
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

# The fMRI REFERENCE BASELINE, computed with NO BOLD DATA.
# The timing arm never touches the brain -- it maps per-segment word count/duration/length to the
# frozen GPT-2 target -- so the number a no-brain model reaches in the Narratives protocol is
# computable from the stimulus alignments alone. That number is the bar any contrastive-retrieval
# decoding result on this dataset has to clear. Same script, same metric as the MEG arms.
#
# Held-out unit is the STORY GROUP, not the story: milkyway{original,synonyms,vodka} etc. are
# variants of one text, so holding out a story while training on its sibling would leak.

# (1) GPT-2 contextual targets for the 42k aligned words.
# Run it from an ISOLATED dir: python puts the script's own directory on sys.path[0], and
# src/tokenizers.py (our MB-1 encoders) then shadows HuggingFace's `tokenizers` package, so
# transformers dies with "cannot import name 'Tokenizer' from 'tokenizers'". extract_contextual.py
# imports nothing from src/, so copying it out is the minimal fix and touches no project file.
if [ ! -f "$META/narr_contextual.npy" ]; then
  echo "[$(date -Is)] === extract GPT-2 contextual ==="
  ISO=$(mktemp -d)
  cp "$CODE/extract_contextual.py" "$ISO/"
  (cd "$ISO" && $PY "$ISO/extract_contextual.py" --data_dir "$META" --out_dir "$META" \
      --model gpt2 --layers 4 6 8 10 12 --tag narr)
  rm -rf "$ISO"
fi

# (2) timing-only baseline, holding out one story GROUP at a time
SEGS=${SEGS:-"3.0"}
# NB: do NOT name this GROUPS -- that is a bash builtin array (the caller's unix group IDs) and
# bash silently ignores assignments to it, so `GROUPS=${GROUPS:-...}` yields a numeric gid and the
# job "succeeds" while holding out nothing and evaluating zero folds.
STORYSET=${STORYSET:-"21styear bigbang black bronx forgot friends himym lucy merlin milkyway notthefall pieman prettymouth santa seinfeld shame shapes sherlock slumlord tunnel upintheair vinny"}
EPOCHS=${EPOCHS:-80}
for SEG in $SEGS; do
  for G in $STORYSET; do
    TAG="narr_nuis_${G}_seg${SEG}"
    if [ -f "$OUT/result_${TAG}.json" ]; then echo "[skip] $TAG"; continue; fi
    echo "[$(date -Is)] === timing-only: hold out group=$G seg=${SEG}s ==="
    $PY "$CODE/train_nuisance_story.py" \
        --data_dir "$META" --ctx "$META/narr_contextual.npy" \
        --ctx_meta "$META/narr_contextual_meta.json" --out_dir "$OUT" \
        --layer 6 --seg_s "$SEG" --min_words 3 --adj_s 10.0 --adj_multi 10,30,60,120,240 \
        --holdout_task "$G" --holdout_col story_group --eval_col task \
        --epochs "$EPOCHS" --batch 256 --tag "$TAG"
  done
done
echo "[$(date -Is)] DONE"
