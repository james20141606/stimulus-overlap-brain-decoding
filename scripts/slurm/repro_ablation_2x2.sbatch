#!/bin/bash
#SBATCH --job-name=bci-ablat
#SBATCH --output=$SCRATCH/project/logs/m1/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=400G --gres=gpu:2 --time=01:45:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# DESIGN 5 (experiment scout, 2026-08-19): a PUBLISHED ingredient under the corrected split.
# Kills the Limits scope "the selection flip is an existence proof from one architecture pair of
# our own construction" -- the only reviewer rung no other proposal touches.
#
# THE 2x2: {SimpleConv as shipped, SimpleConv WITHOUT its subject-specific layer} x {block split,
# story split}. subject_layers is ON in the shipped config (bm/conf/model/clip_conv.yaml:14,
# VERIFIED 2026-08-19) -- i.e., a published ingredient of the reference model. The two WITH cells
# already exist at n=3 (arms A and C); this chain adds the three new cells:
#     noSL_block 4701   arm-A env + simpleconv.subject_layers=false
#     noSL_story 4702   arm-C env + simpleconv.subject_layers=false
#     noSL_volb  4703   arm-B env + simpleconv.subject_layers=false   (volume-matched control:
#                       a shrink under the story split could be volume, not stimulus; this cell
#                       gives the ordering its volume-matched comparison)
# The config override ENTERS the dora signature (unlike env vars), so no checkpoint collision is
# possible even at an old seed; fresh 47xx seeds are used anyway for log hygiene.
#
# PRE-REGISTERED CRITERION (fixed before submission): sign of (with - without) guarded top-10 at
# each cell's own best-validation checkpoint, within split, pool 7,140. Advantage present under
# the block split but < 1.2 pts (arm A's seed spread) or flipped under the story split -> the
# selection-flip claim generalizes to a published ingredient in the published pipeline; Limits
# bullet rewritten. Advantage survives the story split at similar size -> the paper must print "a
# published ingredient we tested did not flip". Escalate to 3 seeds per new cell only if either
# within-split gap < ~3.6 pts.
#
# POOLS: all new cells score under BM_WER_POOL_SEED=0. Block cells retain exactly 7,140 naturally
# (cap inert, pool identical by construction). Story cells: the cap binds, so the comparison
# against arm C's historical draws is size-matched with ~0.5-pt draw noise -- same disclosure as
# the one-bit arms pre-rescore; if the gap lands within noise, re-score arm C's checkpoints under
# the pinned draw via the repro_onebit_rescore.sbatch mechanism before reading the sign.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

export BM_WER_MAX_SEGMENTS=7140
export BM_WER_POOL_SEED=0

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
SENT=$CACHE/ablation_ALL_DONE
LOGD=$SCRATCH/project/logs/m1/ablat
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_WER_POOL_SEED bm/wer.py || { echo "[FAIL] pool-seed patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} date=$(date -Is)"

[ -f "$SENT" ] && { echo "[done] sentinel present"; exit 0; }

CHUNKS=$CACHE/ablation_chunk_count
MAXCHUNKS=40
N=$(cat "$CHUNKS" 2>/dev/null || echo 0); N=$((N + 1)); echo "$N" > "$CHUNKS"
echo "[chain] chunk ${N}/${MAXCHUNKS}"
if [ "$N" -ge "$MAXCHUNKS" ]; then
  echo "[chain] STOP: chunk budget exhausted."
else
  NEXT=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} \
         $SCRATCH/project/code/scripts/slurm/repro_ablation_2x2.sbatch \
         2>/dev/null || true)
  [ -n "$NEXT" ] && echo "[chain] successor queued: $NEXT" || echo "[chain] WARNING: no successor"
fi

run_one () {
  local name=$1 S=$2 envmode=$3
  [ -f "$LOGD/${name}_${S}.DONE" ] && { echo "[skip] ${name} ${S} finished"; return 0; }
  local left_min
  left_min=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | awk -F: '
    NF==3 {print $1*60 + $2} NF==2 {print $1} NF==1 {print 0}')
  if [ -n "$left_min" ] && [ "$left_min" -lt 45 ] 2>/dev/null; then
    echo "[skip] ${name}: only ${left_min} min left. Leaving to successor."; return 0
  fi
  echo "[$(date -Is)] === ${name} seed=${S} env=${envmode} (${left_min:-?} min left) ==="
  (
    unset BM_STORY_SPLIT BM_STORY_SPLIT_SEED BM_MIXED_SPLIT BM_TRAIN_FRAC BM_TRAIN_FRAC_SEED
    case "$envmode" in
      A) : ;;                                             # block split, full train
      B) export BM_TRAIN_FRAC=0.5824 BM_TRAIN_FRAC_SEED=0 ;;
      C) export BM_STORY_SPLIT=1 BM_STORY_SPLIT_SEED=9 ;;
    esac
    exec "$ENV/bin/dora" run -d seed="$S" simpleconv.subject_layers=false
  ) >>"$LOGD/${name}_${S}.log" 2>&1
  local rc=$?
  [ "$rc" -eq 0 ] && touch "$LOGD/${name}_${S}.DONE"
  echo "[$(date -Is)] ${name} ${S} dora_rc=${rc}"
  tail -3 "$LOGD/${name}_${S}.log"
  return 0
}

run_one noSL_block 4701 A
run_one noSL_story 4702 C
run_one noSL_volb  4703 B

DONE=$(ls "$LOGD"/noSL_*.DONE 2>/dev/null | wc -l)
echo "[chain] finished: ${DONE}/3"
[ "$DONE" -eq 3 ] && touch "$SENT" && echo "[chain] ALL DONE"
