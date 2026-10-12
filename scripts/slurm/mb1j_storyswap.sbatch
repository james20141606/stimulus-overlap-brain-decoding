#!/bin/bash
#SBATCH --job-name=bci-mb1j
#SBATCH --output=$SCRATCH/project/logs/mb/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/mb/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G --gres=gpu:1 --time=24:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition
#SBATCH --requeue

# MB-1j: the control that can OVERTURN the headline, plus the decomposition arm.
#
# MB-1h (story 3 held out of pretraining, n=5) showed the pre-registered H1 not merely failing but
# REVERSING: dynamic - conv went from +8.26 SIG (confounded) to -0.40 [-0.57,-0.22] at k=500 and
# -0.57 [-1.15,-0.00] at k=2000. That is the paper's strongest claim -- the confound inverts the
# architectural conclusion, not just the levels.
#
# But MB-1h held out ONE story. The reversal could be a property of story 3 rather than of the
# protocol. This job holds out STORY 0 instead, same 5 subjects, so the comparison is paired and the
# only thing that changed is which stimulus was withheld.
#   * reversal REPLICATES on story 0 -> the claim is about the protocol. Headline stands.
#   * reversal VANISHES -> it was story-specific, the headline must be withdrawn, and MB-1h becomes a
#     single-stimulus observation. I would rather find that here than in review.
#
# Also adds dynamic_big_noflow, absent from MB-1h, so the flow-vs-architecture decomposition can be
# redone in the clean protocol -- currently it exists only in the confounded regime.
#
# PART 2 extends story 3 to the remaining 5 subjects, raising the headline from n=5 to n=10.
set -uo pipefail
export PYTHONUNBUFFERED=1
PY=$SCRATCH/envs/analysis-env/bin/python
CODE=$SCRATCH/project/src
R=$SCRATCH/project
OUT=$R/bci_foundation/results/mb
M0=$R/bci_foundation/results/m0
mkdir -p "$OUT"
echo "[INFO] host=$(hostname) date=$(date -Is) job=${SLURM_JOB_ID:-NA}"

run_one () {   # $1 = held-out story, $2 = subject, $3 = tag prefix
  local ST=$1 S=$2 PFX=$3
  if [ -f "$OUT/result_${PFX}_s${S}.json" ]; then
    echo "[$(date -Is)] [skip] ${PFX} s${S}"; return 0
  fi
  echo "[$(date -Is)] === ${PFX} subject $S, story $ST out of pretraining ==="
  $PY "$CODE/train_mb.py" --out_dir "$OUT" \
      --data_dir "$R/meg_papers/data/meg_masc_full" \
      --ctx "$M0/masc_gpt2_contextual.npy" --ctx_meta "$M0/masc_gpt2_contextual_meta.json" \
      --test_subject "$S" --variants conv,dynamic_big,dynamic_big_noflow --layer 6 \
      --decim 1 --seg_s 1.0 --min_words 1 --klist 100,500,2000 \
      --pre_epochs 80 --adapt_epochs 120 \
      --holdout_story "$ST" --guard_scope stimulus \
      --tag "${PFX}_s${S}"
  local rc=$?
  echo "[$(date -Is)] ${PFX} s${S} rc=${rc}"
  [ "$rc" -ne 0 ] && echo "[FAIL] ${PFX} s${S} exited ${rc}"
  return 0
}

# PART 1 (CRITICAL): story 0 held out, the same 5 subjects as MB-1h -> is the reversal story-specific?
for S in 01 02 05 07 08; do run_one 0 "$S" mb1j_story0; done
# PART 2: extend the story-3 condition to the remaining subjects -> headline n=5 -> n=10
for S in 03 09 10 11 13; do run_one 3 "$S" mb1j_story3ext; done
echo "[$(date -Is)] DONE"
