#!/usr/bin/env python
"""Make the mixed-split evaluation narrative selectable: BM_MIXED_EVAL_STORY (default
The_Black_Willow) and BM_MIXED_VALID_STORY (default cable_spool_fort).

WHY. Every S5 external number is "for that assignment" -- The_Black_Willow fixed as the evaluation
narrative by the disclosed split-seed choice. Design 4 replicates the exposure contrast on a SECOND
narrative (easy_money), the only affordable move toward stimulus generality inside this corpus.
The whole-story assignment map is built dynamically: valid story -> valid, the two non-eval
non-valid stories -> train, eval story -> the original block-hash path (byte-identical test blocks,
as before). Defaults reproduce the current behaviour byte-identically.

Apply (on top of bm_mixed_split.py):
  BM_CHECKOUT=/path/to/brainmagick python patches/bm_mixed_story_choice.py
"""
import ast
import os as _os
import shutil

E = _os.path.join((_os.environ.get("BM_CHECKOUT") or exit("set BM_CHECKOUT=/path/to/brainmagick (the reference-pipeline checkout) and rerun")), "bm/events.py")
src = open(E).read()

if "BM_MIXED_EVAL_STORY" in src:
    print("[skip] bm/events.py already story-selectable")
    raise SystemExit(0)
assert "BM_MIXED_SPLIT" in src, "apply bm_mixed_split.py first"

OLD = """        _MIXED_EVAL_STORY = 'The_Black_Willow'
        # Split indices: ratios = [test_ratio, valid_ratio] -> 0 = test, 1 = valid, last = train.
        _mixed_whole = {'lw1': len(cdf) - 1, 'easy_money': len(cdf) - 1, 'cable_spool_fort': 1}"""
NEW = """        # BM_MIXED_EVAL_STORY / BM_MIXED_VALID_STORY make the narrative assignment selectable
        # (design 4: a second evaluation narrative). Defaults reproduce the original arms
        # byte-identically. See patches/bm_mixed_story_choice.py.
        _MIXED_EVAL_STORY = os.environ.get('BM_MIXED_EVAL_STORY', 'The_Black_Willow')
        _MIXED_VALID_STORY = os.environ.get('BM_MIXED_VALID_STORY', 'cable_spool_fort')
        _ALL_STORIES = ('lw1', 'easy_money', 'cable_spool_fort', 'The_Black_Willow')
        if _MIXED_EVAL_STORY not in _ALL_STORIES or _MIXED_VALID_STORY not in _ALL_STORIES:
            raise RuntimeError('BM_MIXED_SPLIT: unknown story name')
        if _MIXED_EVAL_STORY == _MIXED_VALID_STORY:
            raise RuntimeError('BM_MIXED_SPLIT: eval and valid story must differ')
        # Split indices: ratios = [test_ratio, valid_ratio] -> 0 = test, 1 = valid, last = train.
        _mixed_whole = {s: (1 if s == _MIXED_VALID_STORY else len(cdf) - 1)
                        for s in _ALL_STORIES if s != _MIXED_EVAL_STORY}"""
assert OLD in src, "mixed-setup anchor not found"
src = src.replace(OLD, NEW, 1)

ast.parse(src)
shutil.copy(E, E + ".orig_storychoice")
open(E, "w").write(src)
print("[patched] bm/events.py: mixed eval/valid story selectable (backup .orig_storychoice)")
