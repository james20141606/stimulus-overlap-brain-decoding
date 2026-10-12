#!/usr/bin/env python
"""Resolve the story from the RECORDING, not from the block row.

Third correction to this patch, each caught by verifying before running rather than after:
  1. hashing each story independently -- degenerate at four stories, empties test and valid
  2. writing story_key at block-creation time -- lands behind a 23 GB cache that already exists
  3. reading story_uid off block rows -- the column exists but is NULL on block rows (0/38)

The cached events resolve it. Each cache folder is one subject-session-STORY
(`01_session0_story0`), and inside each events.csv the `story` column has exactly one unique value
(lw1 / cable_spool_fort / easy_money / The_Black_Willow). So the story is a property of the recording,
and every block in a recording belongs to it. `assign_blocks` is called once per recording, so the
story can be taken from whatever non-null value the events carry -- no cache rebuild, no time-overlap
computation, no new column.

Passing it in from dataset.py, where the full events frame is in scope.
"""
import shutil, sys, ast

import os as _os
E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
D = _os.path.join(_os.path.dirname(E), "dataset.py")   # same checkout as E

src = open(E).read()
OLD = """    _story_col = None
    for _c in ('story_uid', 'story', 'story_key'):
        if _c in blocks.columns and blocks[_c].notna().any():
            _story_col = blocks[_c].astype(str).tolist()
            break
    if _story_split and _story_col is None:
        raise RuntimeError(
            'BM_STORY_SPLIT=1 but blocks carry none of story_uid/story/story_key; '
            f'available columns: {list(blocks.columns)}')"""
NEW = """    # The story is a property of the RECORDING: each cached events.csv covers one
    # subject-session-story and its `story` column holds a single value, though that value is NULL on
    # the block rows themselves. The caller passes it in.
    _story_col = [story_id] * len(blocks) if story_id is not None else None
    if _story_split and _story_col is None:
        raise RuntimeError('BM_STORY_SPLIT=1 but assign_blocks got no story_id from the caller')"""
if NEW in src:
    print("  [skip] events.py already applied")
elif OLD not in src:
    sys.exit("  [FAIL] events.py anchor not found")
else:
    src = src.replace(OLD, NEW, 1)
    sig_old = "def assign_blocks(blocks: pd.DataFrame, ratios: tp.List[float], seed: int,\n                  remove_ratio: float = 0, min_n_blocks_per_split: int = 20) -> pd.DataFrame:"
    sig_new = "def assign_blocks(blocks: pd.DataFrame, ratios: tp.List[float], seed: int,\n                  remove_ratio: float = 0, min_n_blocks_per_split: int = 20,\n                  story_id: tp.Optional[str] = None) -> pd.DataFrame:"
    assert sig_old in src, "signature not found"
    src = src.replace(sig_old, sig_new, 1)
    shutil.copy(E, E + ".orig4"); open(E, "w").write(src); ast.parse(src)
    print("  [patched] events.py: story_id parameter")

d = open(D).read()
OLD_D = """        blocks = assign_blocks(
            blocks, [test_ratio, valid_ratio], remove_ratio=remove_ratio, seed=split_assign_seed,
            min_n_blocks_per_split=min_n_blocks_per_split)"""
NEW_D = """        # One recording = one story; the events' `story` column has a single non-null value even
        # though it is NULL on the block rows. Passed through so BM_STORY_SPLIT can keep a story
        # wholly within one split.
        _sid = None
        for _c in ('story', 'story_uid'):
            if _c in events.columns:
                _v = events[_c].dropna().unique()
                if len(_v):
                    _sid = str(_v[0])
                    break
        blocks = assign_blocks(
            blocks, [test_ratio, valid_ratio], remove_ratio=remove_ratio, seed=split_assign_seed,
            min_n_blocks_per_split=min_n_blocks_per_split, story_id=_sid)"""
if NEW_D in d:
    print("  [skip] dataset.py already applied")
elif OLD_D not in d:
    sys.exit("  [FAIL] dataset.py anchor not found")
else:
    shutil.copy(D, D + ".orig4"); open(D, "w").write(d.replace(OLD_D, NEW_D, 1)); ast.parse(open(D).read())
    print("  [patched] dataset.py: pass story_id")
print("  syntax OK")
