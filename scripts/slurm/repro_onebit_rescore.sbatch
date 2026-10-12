#!/bin/bash
#SBATCH --job-name=bci-obrescore
#SBATCH --output=$SCRATCH/project/logs/m1/mixed/%x-%j.out
#SBATCH --error=$SCRATCH/project/logs/m1/mixed/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=200G --gres=gpu:2 --time=05:00:00
#SBATCH --account=CLUSTER_ACCOUNT --partition=gpuA_partition

# DESIGN 1 (experiment scout, 2026-08-19): pool-IDENTITY re-score of the six existing one-bit
# checkpoints. Deletes the paper's qualification (ii) -- "the pool is a size match, not an identity
# match" -- for ~6-10 GPU-h and NO new training.
#
# MECHANISM (the repro_top1_eval.sbatch precedent, six times over): env vars do not enter the dora
# signature, so seed NEW_SEED = OLD_SEED + 40000 gives each arm a fresh xp dir. We pre-seed that
# dir with a copy of the original best checkpoint patched so last_test_epoch=-1 (test stage WILL
# run) and best_loss=-1e30 (the one replayed epoch can never displace best_state). dora then runs
# one train+valid epoch, early-stops, and evaluates EXACTLY the stored best_state -- now under
# BM_WER_POOL_SEED=0, so all six arms share one pinned 7,140-query draw, verified by the
# "wer pool identity sha1" log line (patches/bm_wer_pool_seed.py).
#
# PRE-REGISTERED READING (fixed before any run): report only the BM_WER_POOL_SEED=0 draw, chosen
# blind. Log hashes match across arms -> the size-vs-identity sentence is deleted and D-C' is
# re-quoted on the common pool. Hashes differ -> retention is model-dependent, identity matching is
# impossible; record the constraint, keep the common-generator numbers as variance reduction only.
# Any D-C' pair <= 0 under the common pool -> the ~0.5-pt noise was load-bearing; complete
# separation demotes to "separation under per-arm draws" and the paper says so.
#
# PRE-FLIGHT (run once, interactively, before first submit):
#   1. BM_CHECKOUT=... python patches/bm_wer_pool_seed.py       (idempotent)
#   2. verify SIG discovery: "$ENV/bin/dora" info -d seed=44401 | head -3   -> confirm the parse
#      below extracts the 8-hex signature (VERIFIED: 4401 -> 8baf1b03). Per-arm .DONE
#      files make a resubmit continue where a killed chunk stopped.
set -uo pipefail
export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF

unset BM_STORY_SPLIT BM_STORY_SPLIT_SEED
export BM_WER_MAX_SEGMENTS=7140
export BM_WER_POOL_SEED=0            # the one blind-chosen draw; never shop this seed
D_FRAC=0.5657

BM=$SCRATCH/repro/brainmagick
ENV=$SCRATCH/envs/bm
CACHE=$SCRATCH/repro
LOGD=$SCRATCH/project/logs/m1/mixed
export TMPDIR=${SLURM_TMPDIR:-/tmp}/bm_$$
mkdir -p "$TMPDIR"
export XDG_CACHE_HOME=$CACHE/xdg_cache HF_HOME=$CACHE/hf \
       TRANSFORMERS_CACHE=$CACHE/hf TORCH_HOME=$CACHE/torch_home MNE_DATA=$CACHE/mne_data
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$MNE_DATA" "$LOGD"
cd "$BM" || { echo "[FAIL] no brainmagick dir"; exit 1; }
grep -q BM_WER_POOL_SEED bm/wer.py || { echo "[FAIL] pool-seed patch not applied"; exit 1; }
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-NA} date=$(date -Is)"

# arm  old_seed  mode  frac
ARMS="Cprime 4401 Cprime -
D 4402 D $D_FRAC
Cprime 4403 Cprime -
D 4404 D $D_FRAC
Cprime 4405 Cprime -
D 4406 D $D_FRAC"

sig_of () {  # seed -> dora signature. VERIFIED 2026-08-19: `dora info seed=4401` prints
  # "Found sheep Sheep(8baf1b03, ...)" -- matches the ELN's recorded 4401 signature exactly.
  "$ENV/bin/dora" info seed="$1" 2>&1 | grep -oE 'Sheep\([0-9a-f]{8}' | head -1 | cut -c7-14
}

echo "$ARMS" | while read -r arm S mode frac; do
  NS=$((S + 40000))
  DONE="$LOGD/${arm}_${S}_rescore.DONE"
  [ -f "$DONE" ] && { echo "[skip] ${arm} ${S} already re-scored"; continue; }
  (
    export BM_MIXED_SPLIT="$mode"
    if [ "$frac" != "-" ]; then export BM_TRAIN_FRAC="$frac" BM_TRAIN_FRAC_SEED=0; else unset BM_TRAIN_FRAC BM_TRAIN_FRAC_SEED; fi
    OLD_SIG=$(sig_of "$S"); NEW_SIG=$(sig_of "$NS")
    [ -n "$OLD_SIG" ] && [ -n "$NEW_SIG" ] || { echo "[FAIL] sig discovery failed (old=$OLD_SIG new=$NEW_SIG) for ${arm} ${S}"; exit 1; }
    OLD_CKPT="$BM/outputs/xps/$OLD_SIG/checkpoint.th"
    NEW_DIR="$BM/outputs/xps/$NEW_SIG"
    [ -f "$OLD_CKPT" ] || { echo "[FAIL] no checkpoint at $OLD_CKPT"; exit 1; }
    mkdir -p "$NEW_DIR"
    "$ENV/bin/python" - "$OLD_CKPT" "$NEW_DIR/checkpoint.th" <<'PYEOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
s = torch.load(src, map_location="cpu")
s["last_test_epoch"] = -1        # force the test stage
s["best_loss"] = -1e30           # the replayed epoch can never displace best_state
torch.save(s, dst)
print(f"[seeded] {dst}: best_epoch={s.get('best_epoch')} history={len(s.get('history', []))}")
PYEOF
    echo "[$(date -Is)] === rescore ${arm} old_seed=${S} -> new_seed=${NS} sig ${OLD_SIG}->${NEW_SIG} ==="
    "$ENV/bin/dora" run -d seed="$NS" >>"$LOGD/${arm}_${S}_rescore.log" 2>&1
    rc=$?
    echo "[$(date -Is)] ${arm} ${S} rescore rc=$rc"
    grep -E "Test Summary|pool identity|negatives selected" "$LOGD/${arm}_${S}_rescore.log" | tail -4
    [ "$rc" -eq 0 ] && grep -q "pool identity" "$LOGD/${arm}_${S}_rescore.log" && touch "$DONE"
  )
done
N=$(ls "$LOGD"/*_rescore.DONE 2>/dev/null | wc -l)
echo "[rescore] finished: ${N}/6"
[ "$N" -eq 6 ] && touch "$CACHE/onebit_rescore_ALL_DONE" && echo "[rescore] ALL DONE"
