#!/usr/bin/env python
"""Simplify: independent per-story hashing at a seed verified to partition four stories 1/1/2.

FOURTH correction. The "deterministic partition" was written as though assign_blocks saw the whole
corpus. It does not -- it is called ONCE PER RECORDING, and each cached recording is one story. So the
partition logic received a single story, computed want=[0,0,1], and sent every recording to train,
emptying valid and test globally.

Independent hashing is the approach that works inside a per-recording call, because it needs only the
story's own name. It was rejected earlier for being degenerate at four stories, which is true for an
arbitrary seed but is a property of the SEED, not of the method: sweeping 200 seeds, 23 give a
non-degenerate partition. seed=9 gives exactly 1 test / 1 valid / 2 train:

    The_Black_Willow -> test      cable_spool_fort -> valid
    easy_money       -> train     lw1              -> train

Fixing the seed is legitimate here and must be disclosed: with four stimuli the assignment of stories
to splits is a discrete choice, not a random draw, and any seed that yields a usable partition is
selecting which story is evaluated. The corrected number is therefore for THIS story assignment, and
the paper says so.
"""
import shutil, sys, ast

import os as _os
E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(E).read()

start = src.index("    _story_of_split = {}")
end = src.index("    for _i, block in enumerate(blocks.event.iter()):")
NEW = """    # Story-level splitting hashes the STORY, not the block, so every block of a story shares a
    # split. assign_blocks is called once per recording and each recording is one story, so the
    # assignment must be computable from the story name alone -- hence independent hashing rather than
    # a partition over the corpus, which is not visible here. See patches/bm_story_split5.py: the seed
    # is chosen so that four stories split 1 test / 1 valid / 2 train.
"""
src = src[:start] + NEW + src[end:]

OLD_LOOP = """    for _i, block in enumerate(blocks.event.iter()):
        if _story_split:
            split.append(_story_of_split[_story_col[_i]])
            continue
        uid = block.uid"""
NEW_LOOP = """    for _i, block in enumerate(blocks.event.iter()):
        uid = _story_col[_i] if _story_split else block.uid"""
assert OLD_LOOP in src, "loop anchor not found"
src = src.replace(OLD_LOOP, NEW_LOOP, 1)
src = src.replace("""        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + seed)""",
"""        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        # BM_STORY_SPLIT_SEED pins the story->split assignment; with four stories most seeds are
        # degenerate (all in one split) and this one is verified to give 1 test / 1 valid / 2 train.
        _seed = int(os.environ['BM_STORY_SPLIT_SEED']) if (
            _story_split and 'BM_STORY_SPLIT_SEED' in os.environ) else seed
        rng = random.Random(hashed + _seed)""", 1)
shutil.copy(E, E + ".orig5"); open(E, "w").write(src); ast.parse(src)
print("[patched] independent story hashing with a pinned seed")
