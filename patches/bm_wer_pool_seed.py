#!/usr/bin/env python
"""Pin the WER pool DRAW, not just its size: BM_WER_POOL_SEED makes two arms whose natural retained
count exceeds BM_WER_MAX_SEGMENTS truncate to the IDENTICAL query set instead of two different
random 7,140-query draws.

WHY. The one-bit arms (D vs Cprime) share byte-identical test blocks, but their natural retained
count (~16k) exceeds the 7,140 cap, so each arm was scored on its own random draw -- a size match,
not an identity match (the paper discloses this as qualification (ii), with ~0.5-point per-arm
sampling noise). Pinning the shuffle removes that noise term and lets the disclosure be deleted.

MECHANISM. get_wer's loader is built shuffle=True with no explicit generator, so its permutation is
drawn from the global torch CPU RNG when iteration begins. This patch (applied ON TOP of
bm_wer_pool_match.py -- it anchors on that patch's injected text) saves the CPU RNG state, reseeds
deterministically, and restores the state after the pool is materialized, so a training-time call
leaves the run's randomness untouched. It also logs a permutation-invariant sha1 of the sorted
retained word_hashes, so pool IDENTITY across arms is verified in the logs rather than assumed --
if retention turns out to be model-dependent, the hashes will differ and say so.

Unset -> byte-identical current behaviour.

Apply:  BM_CHECKOUT=/path/to/brainmagick python patches/bm_wer_pool_seed.py
"""
import ast
import os as _os
import shutil

W = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/wer.py")
src = open(W).read()

if "BM_WER_POOL_SEED" in src:
    print("[skip] bm/wer.py already contains BM_WER_POOL_SEED")
    raise SystemExit(0)
assert "BM_WER_MAX_SEGMENTS" in src, "apply bm_wer_pool_match.py first (this patch anchors on it)"

OLD_INIT = """    _wer_cap = int(os.environ.get("BM_WER_MAX_SEGMENTS", "0")) or None
    _n_kept = 0"""
NEW_INIT = """    _wer_cap = int(os.environ.get("BM_WER_MAX_SEGMENTS", "0")) or None
    _n_kept = 0
    # BM_WER_POOL_SEED pins the shuffle that draws the retained subset (pool IDENTITY, not just
    # size). CPU RNG state is saved and restored so training-time calls stay untouched. See
    # patches/bm_wer_pool_seed.py.
    _pool_seed = os.environ.get("BM_WER_POOL_SEED")
    if _pool_seed is not None:
        _rng_state = torch.get_rng_state()
        torch.manual_seed(int(_pool_seed))"""
assert OLD_INIT in src, "init anchor not found -- was bm_wer_pool_match.py applied?"
src = src.replace(OLD_INIT, NEW_INIT, 1)

OLD_TAIL = """        estimates = estimates[:_wer_cap]
        outputs = outputs[:_wer_cap]
        word_hashes = word_hashes[:_wer_cap]"""
NEW_TAIL = """        estimates = estimates[:_wer_cap]
        outputs = outputs[:_wer_cap]
        word_hashes = word_hashes[:_wer_cap]
    if _pool_seed is not None:
        torch.set_rng_state(_rng_state)
    if _wer_cap is not None or _pool_seed is not None:
        # Permutation-invariant identity of the realized pool: cross-arm identity is verified in
        # the logs, never assumed.
        import hashlib as _hashlib
        _sorted = torch.sort(word_hashes.flatten()).values.cpu()
        logger.info("wer pool identity sha1=%s n=%d",
                    _hashlib.sha1(bytes(str(_sorted.tolist()), "utf-8")).hexdigest()[:12],
                    int(_sorted.numel()))"""
assert OLD_TAIL in src, "tail anchor not found -- was bm_wer_pool_match.py applied?"
src = src.replace(OLD_TAIL, NEW_TAIL, 1)

ast.parse(src)
shutil.copy(W, W + ".orig_poolseed")
open(W, "w").write(src)
print("[patched] bm/wer.py: BM_WER_POOL_SEED pins the pool draw (backup .orig_poolseed)")
