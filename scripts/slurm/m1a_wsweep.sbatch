#!/bin/bash
#SBATCH --job-name=bci-wsweep
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=02:00:00
#SBATCH --account=CLUSTER_ACCOUNT
#SBATCH --partition=gpuA_partition

# Segment-window (w) sensitivity sweep -- cc R3's top non-reproduction item, and the check the
# paper's own scope caveat says it "has not run". Everything is identical to the main held-out-story
# runs EXCEPT --seg_s; word count scales with w, so a finer window puts fewer words per segment and
# should weaken the single-feature (word-count-alone) confound specifically. This bounds where it
# fades. Both arms per w: brain (train_story_holdout.py) and word-count-alone
# (train_nuisance_story.py --keep_feats n_words). w=3.0 already exists (result_*_task{K}.json);
# reuse those in the collection script. Each run is cheap (~80s brain, ~30s word-count per task).
set -uo pipefail                       # NOT -e: one failed w must not kill the rest of the sweep
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/m1
M0=$R/bci_foundation/results/m0
DATA=$R/meg_papers/data/meg_masc_full
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

EPOCHS=${EPOCHS:-80}
TASKS=${TASKS:-"0 1 2 3"}
WSET=${WSET:-"1.0 1.5 2.0 4.0 5.0"}    # 3.0 already on disk; collection script reuses it
echo "[INFO] EPOCHS=$EPOCHS TASKS=$TASKS WSET=$WSET"

for W in $WSET; do
  WT=$(echo "$W" | sed 's/\.//')       # 1.5 -> 15, 2.0 -> 20 : dot-free tag suffix
  for K in $TASKS; do
    # --- brain arm ---
    BTAG="story_holdout_task${K}_w${WT}"
    if [ -f "$OUT/result_${BTAG}.json" ]; then echo "[skip] brain w=$W task=$K"; else
      echo "[$(date -Is)] === BRAIN w=$W task=$K ==="
      $PY "$CODE/train_story_holdout.py" --out_dir "$OUT" --data_dir "$DATA" \
          --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
          --holdout_task "$K" --layer 6 --seg_s "$W" --min_words 3 --decim 2 --epochs "$EPOCHS" \
          --tag "$BTAG" || echo "[warn] BRAIN failed w=$W task=$K"
    fi
    # --- word-count-alone arm ---
    NTAG="nuis_keepR_n_words_task${K}_w${WT}"
    if [ -f "$OUT/result_${NTAG}.json" ]; then echo "[skip] wordcount w=$W task=$K"; else
      echo "[$(date -Is)] === WORDCOUNT w=$W task=$K ==="
      $PY "$CODE/train_nuisance_story.py" --out_dir "$OUT" --data_dir "$DATA" \
          --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
          --holdout_task "$K" --layer 6 --seg_s "$W" --min_words 3 --epochs "$EPOCHS" \
          --keep_feats n_words --tag "$NTAG" || echo "[warn] WORDCOUNT failed w=$W task=$K"
    fi
  done
done
echo "[$(date -Is)] DONE"
