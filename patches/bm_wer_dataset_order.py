#!/usr/bin/env python
"""Pin the ORDER (and hence the wer_recordings subset) of the WER datasets list.

WHY. Production logs falsified the pool-identity assumption of bm_wer_pool_seed.py: with
BM_WER_POOL_SEED=0 the retained-pool sha1 was stable WITHIN a process but changed across process
restarts of the SAME arm (D_4407: 9367d90c1a9f twice in one chunk, f9f5e780fdb5 after resume).
The seeded shuffle inside get_wer is therefore not the only randomness: `datasets` arrives in a
process-dependent order, and `datasets[:wer_recordings]` then keeps a DIFFERENT 40-recording
subset per process. Seeding the shuffle over a different underlying list still yields a different
retained set.

THE FIX. When BM_WER_POOL_SEED is set, sort `datasets` by a stable per-recording key before the
truncation, so both the subset and the concatenation order are process-independent; the seeded
shuffle then makes the retained pool identical by construction (verified by the sha1 line).
Unset -> byte-identical current behaviour.

Apply (on top of bm_wer_pool_match.py + bm_wer_pool_seed.py):
  BM_CHECKOUT=/path/to/brainmagick python patches/bm_wer_dataset_order.py
"""
import ast
import os as _os
import shutil

W = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/wer.py")
src = open(W).read()

if "BM_WER_DATASET_ORDER" in src:
    print("[skip] bm/wer.py already order-pinned")
    raise SystemExit(0)
assert "BM_WER_POOL_SEED" in src, "apply bm_wer_pool_seed.py first"

OLD = """        if test_args.wer_recordings is not None:
            datasets = datasets[:test_args.wer_recordings]"""
NEW = """        # BM_WER_DATASET_ORDER marker: with BM_WER_POOL_SEED set, pin the dataset order (and
        # hence the wer_recordings subset) to a stable per-recording key -- the upstream list
        # order is process-dependent, which silently changed the retained pool across restarts.
        # See patches/bm_wer_dataset_order.py.
        if os.environ.get("BM_WER_POOL_SEED") is not None:
            datasets = sorted(datasets, key=lambda d: (str(d.recording.subject_uid),
                                                       str(getattr(d.recording, "recording_uid", ""))))
        if test_args.wer_recordings is not None:
            datasets = datasets[:test_args.wer_recordings]"""
assert OLD in src, "wer_recordings anchor not found"
src = src.replace(OLD, NEW, 1)

ast.parse(src)
shutil.copy(W, W + ".orig_dsorder")
open(W, "w").write(src)
print("[patched] bm/wer.py: dataset order pinned under BM_WER_POOL_SEED (backup .orig_dsorder)")
