#!/bin/bash
#SBATCH --job-name=bci-mb1g
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# MB-1g: the k-sweep under a CLEAN protocol. This is what turns paper 3's positive claim from
# "true of paired differences only" into something whose absolute numbers can also be reported.
#
# MB-1f established H1 (dynamic tokens win at small calibration budgets: +8.26 at k=100, CI excluding
# zero, pre-registered). But the paper has to disclose two confounds that inflate the ABSOLUTE scores:
#   1. the guard was RECORDING-scoped. MEG-MASC records each story twice, so every masked neighbour
#      left an unmasked twin at the same stimulus time in the other session, with an identical target.
#   2. calibration and evaluation were split by RECORDING, so 9/10 held-out subjects had a story on
#      BOTH sides -- part of what the budget k buys is stimulus familiarity, not subject adaptation.
# Both cancel in the paired between-arm contrast, which is why H1 stands. Neither cancels in the levels.
#
# This run sets --guard_scope stimulus (masks the twin) and --split_by story (calibration and
# evaluation never share a narrative). Helpers unit-tested before launch: twins share a stimulus key,
# and the story sets come out disjoint on the real 4-stories-x-2-sessions layout.
#
# READ IT THIS WAY: absolute x chance is NOT comparable to MB-1f -- the pool and the guard both change.
# The question is whether the PAIRED dynamic-conv advantage survives, and whether it keeps the low-shot
# shape. If it does, paper 3's claim no longer needs a confound disclosure attached to it.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

for S in 01 02 03 05 07 08 09 10 11 13; do
  if [ -f "$OUT/result_mb1g_clean_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] subject $S already done"; continue
  fi
  echo "[$(date -Is)] === held subject $S (clean protocol) ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big,dynamic_big_noflow --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,200,500,1000,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --guard_scope stimulus --split_by story \
      --tag "mb1g_clean_s${S}"
  rc=$?
  echo "[$(date -Is)] subject $S rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] subject $S exited ${rc}"
done
echo "[$(date -Is)] DONE"
