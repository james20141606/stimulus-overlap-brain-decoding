#!/usr/bin/env python
"""Apply a STORY-LEVEL split option to the reproduced brainmagick pipeline.

WHY

`assign_blocks` hashes each block's uid to pick its split, and for MEG-MASC `_get_block_uid` returns
the concatenation of the WORDS in that block. Blocks from the same story therefore hash independently
and land in different splits: the model trains on parts of a story and is evaluated on other parts of
the same story. Individual blocks are disjoint -- which is what "stimulus-disjoint splits" is usually
taken to mean -- but the STORY is shared.

That is the same overlap our own experiments isolate at calibration level, and it is the one that
turned out to reverse an architecture comparison. This patch makes it switchable so the published
pipeline can be run both ways and the numbers compared directly.

BEHAVIOUR
  default (BM_STORY_SPLIT unset) : byte-for-byte the original behaviour
  BM_STORY_SPLIT=1               : all blocks from one story share a split

Idempotent -- running it twice is a no-op. Writes .orig backups.
"""
import os
import re
import shutil
import sys

import os as _os
BM = (_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun"))

CREATE_OLD = """    blocks_df = pd.DataFrame(block_events)
    blocks_df['kind'] = 'block'"""
CREATE_NEW = """    blocks_df = pd.DataFrame(block_events)
    blocks_df['kind'] = 'block'
    # STORY KEY for story-level splitting (see patches/bm_story_split.py). The block uid is a word
    # sequence, so blocks from one story hash to different splits; this records which story each
    # block came from so they can be kept together. Purely additive -- nothing reads it unless
    # BM_STORY_SPLIT is set.
    _story_keys = []
    for block, stop in zip(blocks, block_stops):
        _m = (events.start >= block.start) & ((events.start + events.duration) < stop)
        _fp = [f for f in events[_m].filepath.unique() if isinstance(f, str)]
        _story_keys.append(_fp[0] if _fp else '')
    blocks_df['story_key'] = _story_keys"""

ASSIGN_OLD = """    split = list()
    for block in blocks.event.iter():
        uid = block.uid
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)"""
ASSIGN_NEW = """    split = list()
    _story_split = os.environ.get('BM_STORY_SPLIT', '') == '1'
    _story_col = blocks['story_key'].tolist() if 'story_key' in blocks.columns else None
    if _story_split and not _story_col:
        raise RuntimeError('BM_STORY_SPLIT=1 but blocks carry no story_key; re-run the patch')
    for _i, block in enumerate(blocks.event.iter()):
        # Default: hash the block's own uid (its word sequence), so blocks from one story may split
        # apart. With BM_STORY_SPLIT=1: hash the story, so they cannot.
        uid = _story_col[_i] if _story_split else block.uid
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)"""


def patch(path, pairs):
    src = open(path).read()
    if all(new.split("\\n")[0] in src for _, new in pairs if "STORY KEY" in new or "_story_split" in new):
        pass
    changed = False
    for old, new in pairs:
        if new in src:
            print(f"  [skip] already patched: {path}")
            continue
        if old not in src:
            sys.exit(f"  [FAIL] anchor not found in {path}:\n{old[:80]}")
        src = src.replace(old, new, 1)
        changed = True
    if changed:
        shutil.copy(path, path + ".orig")
        open(path, "w").write(src)
        print(f"  [patched] {path} (backup at {path}.orig)")
    return changed


ev = os.path.join(BM, "bm/events.py")
src = open(ev).read()
if "import os" not in src.split("\n\n")[0] and not re.search(r"^import os$", src, re.M):
    src = src.replace("import hashlib", "import hashlib\nimport os", 1)
    open(ev, "w").write(src)
    print("  [patched] added `import os` to events.py")

patch(ev, [(CREATE_OLD, CREATE_NEW), (ASSIGN_OLD, ASSIGN_NEW)])
print("done. BM_STORY_SPLIT unset => original behaviour; =1 => story-level splits.")
