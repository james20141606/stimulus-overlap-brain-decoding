#!/bin/bash
#SBATCH --job-name=bci-mb1i
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# MB-1i: A TUNED FROM-SCRATCH BASELINE. This neutralises the most likely rejection reason.
#
# Four independent 2025-26 audits -- arXiv:2507.01196, 2605.26910, 2601.17883, 2605.14698 -- conclude
# that TUNED SUPERVISED BASELINES MATCH OR BEAT brain foundation models. Our from-scratch arm is not
# tuned: it inherits one learning rate (3e-4) and one weight decay (1e-2) from the fine-tuning config
# and never leaves chance (1.0-1.7x). Reporting a +14 to +26 x-chance "pretraining gain" against that
# is a strawman, and a reviewer will say so before they say anything about novelty.
#
# --scratch_grid sweeps lr x wd and picks the config by VALIDATION top-10 only, then evaluates that
# config once on the held-out set. 100-2000 samples with a large encoder is a regime where weight decay
# should matter enormously, and 1e-2 is almost certainly far too weak -- hence 1e-1 and 1.0 in the grid.
#
# Run in the CLEAN protocol (story 3 out of pretraining, stimulus-scoped guard), because that is the
# protocol the paper will use and it is the harder test: the pretraining advantage there is ~10x
# smaller, so a tuned baseline has a real chance of closing it.
#
# GRID EXTENDED 2026-08-01: the first run's winning lr was 3e-4 for 11 of 12 cells -- the LOWEST value
# in the grid, i.e. the optimum was ON THE BOUNDARY and may lie below it. A reviewer would say the grid
# failed to bracket the optimum, and they would be right. Added 1e-4 and 3e-5. (Winning wd was 0.1 or
# 1.0 in 9 of 12 cells against a 0.01 default, confirming the untuned arm was under-regularised.)
#
# KILL GATE: if tuned scratch matches or beats the pretrained model at any budget, the cross-subject
# transfer claim is dead and the paper becomes purely methodological. That is a result worth having
# early rather than after submission.
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
  if [ -f "$OUT/result_mb1i_tuned_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] subject $S done"; continue
  fi
  echo "[$(date -Is)] === subject $S, TUNED scratch grid, clean protocol ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --holdout_story 3 --guard_scope stimulus \
      --scratch_grid "3e-5,1e-4,3e-4,1e-3,3e-3:1e-2,1e-1,1.0" \
      --tag "mb1i_tuned_s${S}"
  rc=$?
  echo "[$(date -Is)] subject $S rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] subject $S exited ${rc}"
done
echo "[$(date -Is)] DONE"
