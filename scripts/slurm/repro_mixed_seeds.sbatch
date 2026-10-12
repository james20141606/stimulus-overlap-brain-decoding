#!/bin/bash
#SBATCH --job-name=bci-mixseeds
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# THE ONE-BIT CONTRAST: does having OTHER blocks of the evaluation narrative in training inflate
# the score on that narrative's held-out blocks, at matched volume, matched eval stimulus, matched
# candidate pool?
#
# WHY. Arms A/B/C bound the volume contribution but no existing pair isolates narrative overlap
# alone: B vs C differs in overlap AND in what is evaluated. These two arms differ in exactly one
# bit. Both evaluate ONLY on The_Black_Willow's block-hash TEST blocks (original assignment, seed
# untouched -- byte-identical to the blocks the block-split run scores on for that story), and both
# hold valid = cable_spool_fort (whole story):
#     Cprime  train = lw1 + easy_money (whole stories only)            seed 4401
#     D       train = lw1 + easy_money + Black_Willow's TRAIN blocks,  seed 4402
#             thinned to Cprime's volume via BM_TRAIN_FRAC
# Black_Willow's block-hash VALID blocks are dropped in BOTH arms (patches/bm_mixed_split.py), so
# valid stays story-pure and the eval set is identical. If D beats Cprime, other blocks of the same
# narrative in training carry information about its held-out blocks; if they tie, the story-split
# drop was volume/composition, not narrative leakage.
#
# VOLUME MATCH (measured 2026-08-10, CPU dry runs in logs/m1/mixed/dryrun_*.log):
#     Cprime natural train | valid | test = 118310 | 53933 | 32340   (== arm C's 118310 train)
#     D      natural train | valid | test = 202360 | 53933 | 32340   (BW train blocks = 84050)
#     naive 118310/202360 = 0.5847        -> achieved 122278 train (+3.4%, per-recording round())
#     calibrated 0.5847*118310/122278 = 0.5657 -> achieved 119502 train (+1.0% vs target; arm B's
#     precedent was +1.8%: 120389 vs 118310). Hard-coded below. Erring low, not high, was the
#     choice available: overshooting D's volume would inflate the D-Cprime gap spuriously.
#
# POOL. BM_WER_MAX_SEGMENTS=7140 caps both arms identically. With test = 32340 examples the natural
# retained count may sit below the cap; that is fine BY CONSTRUCTION because both arms share the
# exact same test set, hence the same pool. The cap only guards against drift. Record the
# "negatives selected" line with the result; absolute numbers are NOT comparable to arms A/B/C
# (different pool), only D vs Cprime is the admissible comparison.
#
# DORA GOTCHA. Env vars do not enter the dora signature; the SEED is what separates the arms'
# checkpoints (4403/4405=Cprime, 4404/4406=D; all distinct from every previous run).
# Never swap the seed-mode pairing below or a resumed chunk would mix arms. Verified 2026-08-10:
# sigs 8baf1b03 (4401) and 38605352 (4402), neither present in outputs/xps before launch.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT                   # mixed split replaces story split; the patch asserts this
unset BM_STORY_SPLIT_SEED
unset BM_TRAIN_FRAC                    # set PER ARM in run_one; D only
unset BM_TRAIN_FRAC_SEED
export BM_WER_MAX_SEGMENTS=7140        # the pool cap, pinned identically in both arms
D_FRAC=0.5657                          # 0.5847*118310/122278, see header; achieved 119502 examples

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/mixed_seeds_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/mixed
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} D_FRAC=$D_FRAC date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

# Chunked chain, same structure as repro_volmatch: the ~2h reclamation under group GPU contention is
# real (see repro_storysplit.sbatch's header), so queue the successor before doing any work.
CHUNKS=$CACHE/mixed_seeds_chunk_count
MAXCHUNKS=40
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted; inspect rather than raising MAXCHUNKS."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_mixed_seeds.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local arm=$1 S=$2 mode=$3 frac=${4:-}
  [ -f "$LOGD/${arm}_${S}.DONE" ] && { echo "[skip] ${arm} ${S} finished"; return 0; }
  # Do not start an arm that cannot reach a checkpoint: dora commits AFTER the epoch's test stage, so
  # an epoch killed before commit leaves nothing and the next chunk restarts at epoch 1. See
  # repro_storysplit.sbatch, where this cost a full epoch on 2026-08-09.
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${arm}: only ${left_min} min left, need 45 to reach a checkpoint. Leaving to successor."
    return 0
  fi
  echo "[$(date -Is)] === ${arm} seed=${S} BM_MIXED_SPLIT=${mode} frac=${frac:-none} (${left_min:-?} min left) ==="
  (
    export BM_MIXED_SPLIT="$mode"
    if [ -n "$frac" ]; then export BM_TRAIN_FRAC="$frac" BM_TRAIN_FRAC_SEED=0; fi
    exec "$ENV/bin/dora" run -d seed="$S"
  ) >>"$LOGD/${arm}_${S}.log" 2>&1
  local rc=$?
  # rc==0 ONLY: dora logs a Test Summary every improving epoch, so a log-based DONE rule would mark
  # the arm finished after the first chunk and silently truncate training.
  [ "$rc" -eq 0 ] && touch "$LOGD/${arm}_${S}.DONE"
  echo "[$(date -Is)] ${arm} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${arm}_${S}.log"
  return 0
}

# SEED REPLICATION of the one-bit contrast, per the final external review: "the external isolation
# remains single-run" is the named gap between accept and strong accept. Seeds 4403-4406 give the
# D-C' paired difference at n=3 (with the original 4401/4402). Interleaved C'/D per seed so a partial
# chain still yields complete PAIRS rather than three of one arm.
run_one Cprime 4403 Cprime ""
run_one D      4404 D "$D_FRAC"
run_one Cprime 4405 Cprime ""
run_one D      4406 D "$D_FRAC"

DONE=$(ls "$LOGD"/Cprime_440[35].DONE "$LOGD"/D_440[46].DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/4"
[ "$DONE" -eq 4 ] && touch "$SENT" && echo "[chain] ALL DONE"
echo "[$(date -Is)] chunk complete"
