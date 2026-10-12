import io
p = "bm/dataset.py"
s = open(p).read()
old = '''            if os.environ.get("BM_TIMING_ONLY"):
                # R2 timing-only arm: replace the brain input with a word-onset raster (a clock).
                # Same shape as meg; a 1 at each word onset across all channels -> pure lexical
                # timing (count/rate/position), no brain, no acoustics, no lexical identity.
                seg_start, _seg_stop = self._get_bounds_times(index)
                raster = torch.zeros_like(meg_torch)
                n_t = raster.shape[1]
                for _ev in events:
                    if type(_ev).__name__ == "Word":
                        _ti = int(self.sample_rate.to_ind(_ev.start - seg_start))
                        if 0 <= _ti < n_t:
                            raster[:, _ti] = 1.0
                meg_torch = raster'''
new = '''            if os.environ.get("BM_TIMING_ONLY"):
                # R2 timing-only arm: replace the brain input with a word-onset raster (a clock).
                seg_start, _seg_stop = self._get_bounds_times(index)
                raster = torch.zeros_like(meg_torch)
                n_t = raster.shape[1]
                _onsets = []
                for _ev in events:
                    if type(_ev).__name__ == "Word":
                        _ti = int(self.sample_rate.to_ind(_ev.start - seg_start))
                        if 0 <= _ti < n_t:
                            _onsets.append(_ti)
                if os.environ.get("BM_TIMING_COUNT"):
                    # word-count-only: a constant channel = word count (matches the paper headline)
                    raster[:, :] = float(len(_onsets))
                elif os.environ.get("BM_TIMING_SHUFFLE"):
                    # leak/null control: a random raster independent of this segment -> must -> chance
                    _k = int(torch.randint(0, 31, (1,)).item())
                    if _k > 0 and n_t > 0:
                        raster[:, torch.randint(0, n_t, (_k,))] = 1.0
                else:
                    for _ti in _onsets:
                        raster[:, _ti] = 1.0
                meg_torch = raster'''
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("PATCHED_OK")
