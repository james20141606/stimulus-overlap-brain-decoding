#!/bin/bash
#SBATCH --job-name=bci-mb1h
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# MB-1h: THE EXPERIMENT THAT DECIDES WHAT PAPER 3'S TRANSFER EFFECT ACTUALLY IS.
#
# MB-1f showed a huge pretraining effect: from-scratch encoders never leave chance (1.0-1.7x at every
# budget, 30/30 arm-subject cells) while pretrained ones reach 23-30x. I was about to promote that as
# paper 3's strongest positive claim. But `tr_mask = subj != test_subject` means pretraining sees ALL
# FOUR STORIES from the other 9 subjects (our MEG-MASC copy has 10; comment corrected 2026-09-25, was '26') -- so the EVALUATION STIMULI have already been seen, through
# other brains, before the held-out subject is touched.
#
# That admits an alternative explanation which an earlier analysis found to be real and large: the
# model matches a stimulus-locked template learned across subjects, i.e. it identifies WHICH TIME POINT
# OF WHICH STORY a segment came from, without decoding language. On fMRI that mechanism drove a
# brain-free baseline to top-10 accuracy 1.000, and it collapsed to 0.40 the moment segments were held
# out. MB-1g does NOT fix this: it only makes calib/eval disjoint WITHIN the held-out subject.
#
# Here story 3 is removed from pretraining AND from calibration, and used as the evaluation set. No
# brain in the training set has ever heard it. So:
#   * if the pretrained-vs-scratch gap SURVIVES -> the transfer is genuinely about neural language
#     representation, and paper 3 has a strong, clean positive claim.
#   * if it COLLAPSES -> the 23-30x was largely cross-subject time-point identification, and all three
#     papers are the same finding in three modalities, which is a bigger and more coherent paper.
# Either outcome is publishable and I am not predicting which.
#
# Runs 5 subjects (enough to see a large effect; the full 10 follow if it is close).
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
  if [ -f "$OUT/result_mb1h_stimho_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] subject $S done"; continue
  fi
  echo "[$(date -Is)] === held subject $S, story 3 held out of PRETRAINING ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --holdout_story 3 --guard_scope stimulus \
      --tag "mb1h_stimho_s${S}"
  rc=$?
  echo "[$(date -Is)] subject $S rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] subject $S exited ${rc}"
done
echo "[$(date -Is)] DONE"
