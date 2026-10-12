#!/usr/bin/env python
"""Derive story_key from the CACHED events instead of at block-creation time.

The previous patch added `story_key` inside `_create_blocks`. That runs before the events are written
to `cache/studies/<subject>/events.csv`, and those caches already exist from earlier runs -- 23 GB of
them -- so the column was absent and the run failed with the guard I had put there:

    RuntimeError: BM_STORY_SPLIT=1 but blocks carry no story_key; re-run the patch

The guard did its job: loud failure rather than a silent wrong split. Rebuilding the cache to add one
column is not worth it, and it turns out to be unnecessary -- the cached events already carry `story`
and `story_uid` columns. This patch drops the block-creation hook and reads the story directly in
`assign_blocks` from the blocks DataFrame, which inherits those columns from the events.
"""
import shutil, sys

import os as _os
P = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(P).read()

# 1. remove the _create_blocks hook: it writes a column into a cache that is already built
CREATE_HOOK_START = "    # STORY KEY for story-level splitting"
if CREATE_HOOK_START in src:
    i = src.index(CREATE_HOOK_START)
    j = src.index("blocks_df['story_key'] = _story_keys", i) + len("blocks_df['story_key'] = _story_keys\n")
    src = src[:i] + src[j:]
    print("  [removed] _create_blocks story_key hook")

# 2. read the story from whichever column the cached events actually provide
OLD = """    _story_col = blocks['story_key'].tolist() if 'story_key' in blocks.columns else None
    if _story_split and not _story_col:
        raise RuntimeError('BM_STORY_SPLIT=1 but blocks carry no story_key; re-run the patch')"""
NEW = """    # The cached events carry 'story_uid' (and 'story'); prefer those over anything we add, so the
    # existing 23 GB of caches remain valid. 'story_key' is kept as a fallback for freshly built caches.
    _story_col = None
    for _c in ('story_uid', 'story', 'story_key'):
        if _c in blocks.columns and blocks[_c].notna().any():
            _story_col = blocks[_c].astype(str).tolist()
            break
    if _story_split and _story_col is None:
        raise RuntimeError(
            'BM_STORY_SPLIT=1 but blocks carry none of story_uid/story/story_key; '
            f'available columns: {list(blocks.columns)}')"""
if NEW in src:
    print("  [skip] already applied")
elif OLD not in src:
    sys.exit("  [FAIL] anchor not found")
else:
    src = src.replace(OLD, NEW, 1)
    print("  [patched] story column resolved from cached events")

shutil.copy(P, P + ".orig3")
open(P, "w").write(src)
import ast; ast.parse(src)
print("  syntax OK")
