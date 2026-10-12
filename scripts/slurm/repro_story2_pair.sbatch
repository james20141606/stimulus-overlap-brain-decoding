#!/bin/bash
#SBATCH --job-name=bci-story2
#SBATCH --output=$SCRATCH/project/logs/m1/mixed/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/mixed/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# DESIGN 4 (experiment scout, 2026-08-19; launched 2026-08-30): a SECOND EVALUATION NARRATIVE for
# the external exposure contrast. Kills the single-narrative scope of every S5 number ("the
# corrected number is for that assignment") -- the last unlaunched on-corpus design.
#
#     Cprime2  seed 4601  eval=easy_money, train = lw1 + The_Black_Willow (whole), natural volume
#     D2full   seed 4602  same + easy_money's block-hash TRAIN blocks at FULL dose
#                         (BM_TRAIN_FRAC_EVALSTORY=1.0), others thinned to match volume
#
# Valid stays cable_spool_fort in both arms (story-pure); test = easy_money's block-hash TEST
# blocks under the untouched original assignment -- byte-identical across the two arms.
#
# CALIBRATION (CPU dry runs, logs/m1/mixed/dryrun_C2nat/D2nat/D2full*.log):
#   Cprime2 natural: 163,189 | 53,933 | 24,843.  D2 natural: 218,773 (EM train blocks = 55,584).
#   D2full: naive 0.6594 -> 160,840 (-1.44%); calibrated 0.6741 -> 162,548 (-0.39%). ACCEPTED --
#   err-low is the conservative direction (an under-fed D2 can only understate the leak).
#
# POOL: test retains ~12.4k queries -> the 7,140 cap BINDS; within-pair the test set is
# byte-identical, pool membership run-dependent as disclosed (fitted-scaler mechanism);
# BM_WER_POOL_SEED=0 from birth. Absolute top-10 is NOT comparable to the Black-Willow pair
# (different narrative, different natural volume 163k vs 118k); the admissible cross-narrative
# statement is the SIGN and relative size of the within-pair leak.
#
# PRE-REGISTERED READING (fixed before launch): leak2 = D2full - Cprime2.
#   Same sign as the Black-Willow full-dose leak (+3.00) -> S5's "for that assignment" gains a
#     second-narrative replication clause; quote leak2 next to +3.00, sign first, size second.
#   Zero or opposite sign -> the paper prints "the external pricing did not replicate on a second
#     narrative", consistent with the grid's stimulus-dependence; no demotion of the BW numbers,
#     which stand on their own eight runs.
#   Escalate to 3 seeds per arm only if |leak2| < ~1.2 (arm A's seed spread).
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT BM_STORY_SPLIT_SEED
export BM_WER_MAX_SEGMENTS=7140
export BM_WER_POOL_SEED=0
export BM_MIXED_EVAL_STORY=easy_money
D2_FRAC=0.6741   # dry-run round 2: 162,548 train (-0.39% vs 163,189)

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/story2_pair_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/mixed
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_MIXED_EVAL_STORY bm/events.py || { echo "[FAIL] story-choice patch not applied"; exit 1; }
grep -q BM_TRAIN_FRAC_EVALSTORY bm/events.py || { echo "[FAIL] evalstory patch not applied"; exit 1; }
grep -q BM_WER_POOL_SEED bm/wer.py || { echo "[FAIL] pool-seed patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} D2_FRAC=$D2_FRAC date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/story2_pair_chunk_count
MAXCHUNKS=30
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_story2_pair.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local name=$1 S=$2 mode=$3 evfrac=$4 frac=$5
  [ -f "$LOGD/${name}_${S}.DONE" ] && { echo "[skip] ${name} ${S} finished"; return 0; }
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${name}: only ${left_min} min left. Leaving to successor."; return 0
  fi
  echo "[$(date -Is)] === ${name} seed=${S} mode=${mode} (${left_min:-?} min left) ==="
  (
    export BM_MIXED_SPLIT="$mode"
    if [ "$frac" != "-" ]; then
      export BM_TRAIN_FRAC="$frac" BM_TRAIN_FRAC_SEED=0 BM_TRAIN_FRAC_EVALSTORY="$evfrac"
    else
      unset BM_TRAIN_FRAC BM_TRAIN_FRAC_SEED BM_TRAIN_FRAC_EVALSTORY
    fi
    exec "$ENV/bin/dora" run -d seed="$S"
  ) >>"$LOGD/${name}_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/${name}_${S}.DONE"
  echo "[$(date -Is)] ${name} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${name}_${S}.log"
  return 0
}

run_one Cprime2 4601 Cprime - -
run_one D2full  4602 D 1.0 "$D2_FRAC"

DONE=$(ls "$LOGD"/Cprime2_4601.DONE "$LOGD"/D2full_4602.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/2"
[ "$DONE" -eq 2 ] && touch "$SENT" && echo "[chain] ALL DONE"
