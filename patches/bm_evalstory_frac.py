#!/usr/bin/env python
"""Per-story training fraction under BM_MIXED_SPLIT: BM_TRAIN_FRAC_EVALSTORY overrides the global
BM_TRAIN_FRAC for the evaluation narrative's recordings only.

WHY. The one-bit D arm thins its whole training set uniformly, so it carries only ~59% of the
evaluation narrative's train blocks -- a 0.59-dose exposure (disclosed as qualification (i)). The
external dose ladder (design 3) needs arms where the narrative's own blocks are kept at a CHOSEN
density while the other stories absorb the volume match:

    D_full  BM_TRAIN_FRAC_EVALSTORY=1.0   BM_TRAIN_FRAC=<calibrated ~0.29>   dose 1.0
    D_mid   BM_TRAIN_FRAC_EVALSTORY=0.30  BM_TRAIN_FRAC=<calibrated ~0.79>   dose 0.30

With C' (dose 0) and D (dose 0.59) already at n>=3, these two arms complete a four-point external
dose-response at matched volume. Unset -> byte-identical current behaviour.

MECHANICS. Edits the BM_TRAIN_FRAC stage injected by bm_train_subsample.py (as later edited by
bm_mixed_split.py edit #4): the per-recording fraction becomes story-dependent. Requires
BM_MIXED_SPLIT (the story identity and eval-story name are only defined there); raises otherwise.
The kept-count stays round(len*frac) per recording, seed-invariant; BM_TRAIN_FRAC_SEED semantics
unchanged.

Apply: BM_CHECKOUT=/path/to/brainmagick python patches/bm_evalstory_frac.py
"""
import ast
import os as _os
import shutil

E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(E).read()

if "BM_TRAIN_FRAC_EVALSTORY" in src:
    print("[skip] bm/events.py already contains BM_TRAIN_FRAC_EVALSTORY")
    raise SystemExit(0)
assert "BM_MIXED_SPLIT" in src, "apply bm_mixed_split.py first"

OLD = """    _tf = os.environ.get('BM_TRAIN_FRAC')
    if _tf and float(_tf) < 1.0:
        _frac = float(_tf)"""
NEW = """    _tf = os.environ.get('BM_TRAIN_FRAC')
    # BM_TRAIN_FRAC_EVALSTORY: per-story override for the evaluation narrative under
    # BM_MIXED_SPLIT (external dose ladder). See patches/bm_evalstory_frac.py.
    _tf_eval = os.environ.get('BM_TRAIN_FRAC_EVALSTORY')
    if _tf_eval and not _mixed:
        raise RuntimeError('BM_TRAIN_FRAC_EVALSTORY requires BM_MIXED_SPLIT')
    _frac_env = _tf
    if _tf_eval and _mixed and story_id == _MIXED_EVAL_STORY:
        _frac_env = _tf_eval
    if _frac_env and float(_frac_env) < 1.0:
        _frac = float(_frac_env)"""
assert OLD in src, "train-frac anchor not found"
src = src.replace(OLD, NEW, 1)

ast.parse(src)
shutil.copy(E, E + ".orig_evalfrac")
open(E, "w").write(src)
print("[patched] bm/events.py: BM_TRAIN_FRAC_EVALSTORY per-story fraction (backup .orig_evalfrac)")
