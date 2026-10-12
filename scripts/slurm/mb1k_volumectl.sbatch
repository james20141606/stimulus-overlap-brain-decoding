#!/bin/bash
#SBATCH --job-name=bci-mb1k
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# MB-1k: THE PRETRAINING-VOLUME CONTROL. Without it the headline is confounded.
#
# MB-1h removes story 3 from pretraining and evaluates on story 3, and the reported transfer collapses
# 68-86%. But removing a story also removes ~25% of the pretraining data, so "the eval stimulus was
# never seen" and "there was less pretraining data" move together. A reviewer will say so immediately,
# and they would be right -- this is the third time in two days I have built a two-variable experiment
# (see F2's seg_blocks, and MB-1h's guard+holdout).
#
# This arm holds pretraining VOLUME fixed and flips only the stimulus-overlap bit:
#     pretrain excludes story 3   (IDENTICAL to MB-1h -- same 3 stories, same data volume)
#     evaluate on story 0         (which pretraining DID see, through the other 9 subjects; comment corrected 2026-09-25, was '26')
#     calibrate on stories 1,2    (story 3 never used, so the comparison is clean)
#
# Read against MB-1h on the same subjects:
#   MB-1h  pretrain{0,1,2}  eval story 3 (UNSEEN)  -> ~3.4-4.6x
#   MB-1k  pretrain{0,1,2}  eval story 0 (SEEN)    -> ?
#   * MB-1k stays HIGH (~20-30x) -> the collapse is the stimulus overlap. Headline confirmed, and the
#     pretraining-volume objection is dead.
#   * MB-1k also LOW -> the collapse is mostly lost pretraining data, the shared-stimulus story is
#     wrong, and the headline must be withdrawn.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

for S in 01 02 05 07 08; do
  if [ -f "$OUT/result_mb1k_volctl_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] s${S}"; continue
  fi
  echo "[$(date -Is)] === s${S}: pretrain excludes story 3, EVAL ON story 0 (seen) ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --holdout_story 3 --eval_story 0 --guard_scope stimulus \
      --tag "mb1k_volctl_s${S}"
  rc=$?
  echo "[$(date -Is)] s${S} rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] s${S} exited ${rc}"
done
echo "[$(date -Is)] DONE"
