#!/usr/bin/env python
"""Replace independent story hashing with a deterministic story PARTITION.

The first version hashed each story independently, which is fine with many stimuli but degenerate with
four: every story can land in the same split. A dry run put all four of MEG-MASC's stories in train,
leaving test and valid empty (assign_blocks would then raise on min_n_blocks_per_split, so it fails
loudly rather than silently -- but it fails).

This version sorts the unique stories by a seeded hash and walks the ratio cdf, guaranteeing every
split receives at least one story. With four stories and the paper's ratios that yields one test story,
one valid story and two train stories.

The resulting split is 1/1/2 rather than the nominal 0.2/0.1/0.7, so the TRAINING SET SHRINKS relative
to the original run. That is unavoidable at four stimuli and must be disclosed: the story-level number
is not only stimulus-disjoint, it is also trained on less data. It is the same volume-versus-composition
tension handled elsewhere by running both directions.
"""
import shutil, sys

import os as _os
P = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
OLD = """    split = list()
    _story_split = os.environ.get('BM_STORY_SPLIT', '') == '1'
    _story_col = blocks['story_key'].tolist() if 'story_key' in blocks.columns else None
    if _story_split and not _story_col:
        raise RuntimeError('BM_STORY_SPLIT=1 but blocks carry no story_key; re-run the patch')
    for _i, block in enumerate(blocks.event.iter()):
        # Default: hash the block's own uid (its word sequence), so blocks from one story may split
        # apart. With BM_STORY_SPLIT=1: hash the story, so they cannot.
        uid = _story_col[_i] if _story_split else block.uid
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + seed)
        score = rng.random()
        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                split.append(idx)
                break"""
NEW = """    split = list()
    _story_split = os.environ.get('BM_STORY_SPLIT', '') == '1'
    _story_col = blocks['story_key'].tolist() if 'story_key' in blocks.columns else None
    if _story_split and not _story_col:
        raise RuntimeError('BM_STORY_SPLIT=1 but blocks carry no story_key; re-run the patch')

    _story_of_split = {}
    if _story_split:
        # Deterministic PARTITION of stories across splits. Independent per-story hashing is
        # degenerate at four stimuli -- all four can land in one split, emptying test and valid.
        # Sort stories by a seeded hash, then walk the cdf, guaranteeing each split gets at least one.
        _uniq = sorted(set(_story_col))
        _uniq.sort(key=lambda s: int(hashlib.sha256((s + str(seed)).encode()).hexdigest(), 16))
        _n = len(_uniq)
        _want = [max(1, int(round(r * _n))) for r in ratios]
        while sum(_want) > _n:                      # trim the largest split first
            _want[int(np.argmax(_want))] -= 1
        while sum(_want) < _n:
            _want[int(np.argmax(ratios))] += 1
        _pos = 0
        for _idx, _cnt in enumerate(_want):
            for _s in _uniq[_pos:_pos + _cnt]:
                _story_of_split[_s] = _idx
            _pos += _cnt

    for _i, block in enumerate(blocks.event.iter()):
        if _story_split:
            split.append(_story_of_split[_story_col[_i]])
            continue
        uid = block.uid
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + seed)
        score = rng.random()
        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                split.append(idx)
                break"""
src = open(P).read()
if NEW in src:
    print("[skip] already applied"); sys.exit(0)
if OLD not in src:
    sys.exit("[FAIL] anchor not found -- was bm_story_split.py applied first?")
shutil.copy(P, P + ".orig2")
open(P, "w").write(src.replace(OLD, NEW, 1))
print("[patched] deterministic story partition installed")
