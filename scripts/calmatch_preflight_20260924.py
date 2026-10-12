#!/usr/bin/env python3
"""Pre-training gate for the calibration-diversity control (PREREG: notes/PREREG_calibmatch_20260924.md).

Runs once per held-out subject, inside the Slurm task, BEFORE any training. It exits non-zero -- and
the task then trains nothing -- if any of these fails:

  1. BACKEND. The 2026-09-22 common-pool smoke (JID 18254449) ran ~195x slower than the historical
     recipe because its runner set cudnn.deterministic=True and exported CUBLAS_WORKSPACE_CONFIG
     (microbenchmark JID 18321669). The historical train_mb.py sets neither. This gate asserts both
     are absent and times five synthetic conv training steps at the formal shape (B=256, C=n_ch,
     T=200). Historical H200: ~0.0094 s/step; regressed bundle: ~2.03 s/step. The gate fails above
     --max_step_s (default 0.2 s, 20x historical, 10x below the regression).

  2. DESIGN IDENTITY. Re-derives, with the same mask logic as the pinned train_mb.py (lines 262-294),
     the held-out subject's evaluation and calibration sets for three configurations:
        SEEN      --holdout_story H_seen --eval_story E            (published arm, reused)
        UNSEEN    --holdout_story E                                (published arm, reused)
        MATCHED   --holdout_story E --calib_stories C              (the new arm)
     and requires: evaluation masks identical in all three; MATCHED calibration mask identical to
     SEEN's; UNSEEN's calibration strictly larger. Because train_mb.py draws the calibration/
     validation permutation from default_rng(seed) over the calibration pool, identical pools imply
     identical validation segments and identical first-k calibration subsets; the hashes of those
     index sets are written so the claim is checkable, not assumed.

  3. INPUTS. SHA-256 of the metadata TSV, GPT-2 context array and its meta JSON; size, mtime and the
     SHA-256 of the first 256 MiB of epochs_data.npy (the full file is tens of GB); SHA-256 of every
     pinned source file, and of the live src/eval_metrics.py for the record (the live copy carries an
     uninitialised `_tie_hits` in guarded_retrieval_metrics since 2026-08-18 and is NOT used).

Writes <out>/preflight_<tag>.json. Never reads a result JSON or any model outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha256_file(p, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            if limit is not None and n + len(chunk) > limit:
                chunk = chunk[: limit - n]
            h.update(chunk)
            n += len(chunk)
            if limit is not None and n >= limit:
                break
    return h.hexdigest()


def sha_arr(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()


def story_of(rec):
    # identical to train_mb._story_of
    return np.array([str(r).split("|")[2] if "|" in str(r) else str(r) for r in rec])


def masks(st_hs, holdout, eval_story, calib):
    """Mirror of train_mb.py (pinned SHA 1a4953ec) lines 262-283, single-story or comma holdout."""
    hold = {s.strip() for s in str(holdout).split(",") if s.strip()}
    ev_story = str(eval_story) if eval_story else sorted(hold)[0]
    ev = st_hs == ev_story
    ca = (~ev) & (~np.isin(st_hs, list(hold)))
    if calib:
        keep = set(str(calib).split(","))
        ca = ca & np.isin(st_hs, list(keep))
    return ev, ca


def scratch_quota_gate(min_free_gb, min_free_files):
    """2026-09-24: $SCRATCH stood at 99.86% of its 5 TB quota. A run writes < 1 MB (result JSON,
    logs, receipts; no checkpoints), but a full quota would lose the result JSON after ~20 min of GPU
    time, so refuse to start below --min_free_gb (0.2 GB since 2026-09-24 15:11: the run writes < 15 MB in total, and at 99.94% the quota has ~3 GB left). Parsed from `myquota` (the university HPC); if it cannot be read
    the gate records 'unknown' and does not block (it cannot measure)."""
    import re as _re
    import subprocess as _sp
    out = {"source": "myquota", "min_free_gb": min_free_gb, "min_free_files": min_free_files}
    try:
        txt = _sp.run(["myquota"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:                                   # noqa: BLE001
        out["status"] = f"unknown ({e.__class__.__name__})"
        return out, []
    txt = _re.sub(r"\x1b\[[0-9;]*m", "", txt)
    line = next((l for l in txt.splitlines() if l.strip().startswith("/scratch")), None)
    m = line and _re.search(r"([\d.]+)(TB|GB)/([\d.]+)([MK]?)\s+([\d.]+)(TB|GB)\(([\d.]+)%\)/(\d+)\(([\d.]+)%\)", line)
    if not m:
        out["status"] = "unknown (unparsed)"; out["raw"] = line
        return out, []
    unit = {"TB": 1000.0, "GB": 1.0}
    alloc_gb = float(m.group(1)) * unit[m.group(2)]
    pct = float(m.group(7))
    files_alloc = float(m.group(3)) * {"M": 1e6, "K": 1e3, "": 1.0}[m.group(4)]
    files_used = int(m.group(8))
    free_gb = alloc_gb * (1 - pct / 100.0)
    free_files = files_alloc - files_used
    out.update({"raw": line.strip(), "alloc_gb": alloc_gb, "used_pct": pct,
                "free_gb_est": round(free_gb, 2), "free_files": int(free_files), "status": "measured"})
    probs = []
    if free_gb < min_free_gb:
        probs.append(f"/scratch quota: ~{free_gb:.1f} GB free < {min_free_gb} GB; refusing to start")
    if free_files < min_free_files:
        probs.append(f"/scratch quota: {int(free_files)} files free < {min_free_files}; refusing to start")
    return out, probs


def backend_gate(n_ch, max_step_s, device_required):
    import torch
    import torch.nn.functional as F
    out = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    problems = []
    if out["cudnn_deterministic"]:
        problems.append("cudnn.deterministic is True (historical recipe leaves it False)")
    if out["CUBLAS_WORKSPACE_CONFIG"] is not None:
        problems.append("CUBLAS_WORKSPACE_CONFIG is set (historical jobs did not set it)")
    if out["deterministic_algorithms"]:
        problems.append("torch.use_deterministic_algorithms is on")
    if device_required and not out["cuda_available"]:
        problems.append("CUDA not available")
    if out["cuda_available"]:
        sys.path.insert(0, str(HERE / "calmatch_20260924" / "src_pinned"))
        from tokenizers import build_encoder                      # pinned copy
        from train_infonce import LLMHead
        dev = "cuda"
        out["gpu_name"] = torch.cuda.get_device_name(0)
        g = torch.Generator().manual_seed(0)
        x = torch.randn(256, n_ch, 200, generator=g).to(dev)
        y = torch.randn(256, 768, generator=g).to(dev)
        enc = build_encoder("conv", n_ch).to(dev)
        llm = LLMHead(768, 768).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(llm.parameters()), lr=3e-4)
        times = []
        for i in range(7):
            torch.cuda.synchronize(); t = time.perf_counter()
            zm, aux = enc(x); zm = F.normalize(zm, dim=-1); zl = F.normalize(llm(y), dim=-1)
            logits = 14.3 * zm @ zl.T; lab = torch.arange(256, device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) + aux
            opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        steady = times[2:]
        out["conv_step_s_median"] = float(np.median(steady))
        out["conv_step_s_all"] = [round(v, 5) for v in times]
        out["max_step_s"] = max_step_s
        if out["conv_step_s_median"] > max_step_s:
            problems.append(f"conv step {out['conv_step_s_median']:.4f}s > {max_step_s}s: "
                            "backend regression present; do not train")
        del enc, llm, opt, x, y
        torch.cuda.empty_cache()
    out["problems"] = problems
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--ctx_meta", required=True)
    ap.add_argument("--test_subject", required=True)
    ap.add_argument("--eval_story", required=True)
    ap.add_argument("--seen_holdout", required=True, help="story the published SEEN arm withheld")
    ap.add_argument("--calib_stories", required=True, help="the SEEN arm's calibration stories")
    ap.add_argument("--klist", default="100,500,2000")
    ap.add_argument("--val_n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--t0", type=int, default=50)
    ap.add_argument("--t1", type=int, default=250)
    ap.add_argument("--decim", type=int, default=1)
    ap.add_argument("--seg_s", type=float, default=1.0)
    ap.add_argument("--min_words", type=int, default=1)
    ap.add_argument("--max_step_s", type=float, default=0.2)
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--min_free_gb", type=float, default=0.2)
    ap.add_argument("--min_free_files", type=int, default=1000)
    ap.add_argument("--live_src", default="", help="live src/ dir, hashed for the record only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(HERE / "calmatch_20260924" / "src_pinned"))
    from train_m1a_seg import build_segments_with_meta          # pinned copy

    t_start = time.time()
    rec = {"tag": a.tag, "test_subject": a.test_subject, "eval_story": a.eval_story,
           "seen_holdout": a.seen_holdout, "calib_stories": a.calib_stories, "seed": a.seed,
           "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "slurm_array": os.environ.get("SLURM_ARRAY_JOB_ID"),
           "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    problems = []

    # ---- 3. inputs and code
    data_dir = Path(a.data_dir)
    ep = data_dir / "epochs_data.npy"
    st = ep.stat()
    rec["inputs"] = {
        "trial_metadata.tsv": sha256_file(data_dir / "trial_metadata.tsv"),
        "ctx": {"path": a.ctx, "sha256": sha256_file(a.ctx)},
        "ctx_meta": {"path": a.ctx_meta, "sha256": sha256_file(a.ctx_meta)},
        "epochs_data.npy": {"size": st.st_size, "mtime": int(st.st_mtime),
                            "sha256_first_256MiB": sha256_file(ep, limit=256 << 20)},
    }
    pinned = HERE / "calmatch_20260924" / "src_pinned"
    rec["pinned_src_sha256"] = {p.name: sha256_file(p) for p in sorted(pinned.glob("*.py"))}
    expect = {}
    sums = pinned / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text().splitlines():
            h, n = line.split()
            expect[n] = h
        for n, h in expect.items():
            if rec["pinned_src_sha256"].get(n) != h:
                problems.append(f"pinned source {n} does not match SHA256SUMS")
    else:
        problems.append("SHA256SUMS missing from pinned bundle")
    rec["this_script_sha256"] = sha256_file(__file__)
    if a.live_src:
        rec["live_src_sha256_for_record"] = {
            n: sha256_file(Path(a.live_src) / n) for n in
            ("train_mb.py", "eval_metrics.py", "train_infonce.py", "train_m1a_seg.py", "tokenizers.py")
            if (Path(a.live_src) / n).exists()}

    # ---- 2. design identity (held-out subject only; no pretraining data are built)
    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t",
                       dtype={"subject": str, "session": str, "task": str, "run": str},
                       usecols=lambda c: c in
                       {"trial_index", "subject", "session", "task", "run", "onset_s"})
    cmeta = json.loads(Path(a.ctx_meta).read_text())
    li = cmeta["layers_kept"].index(a.layer)
    ctx = np.load(a.ctx, mmap_mode="r")
    epochs = np.load(ep, mmap_mode="r")
    rec["n_ch"] = int(epochs.shape[1])
    subj = meta["subject"].values.astype(str)
    te_mask = subj == str(a.test_subject)
    Xhs, Yhs, rec_hs, on_hs = build_segments_with_meta(meta, te_mask, epochs, ctx, li, a.t0, a.t1,
                                                       a.decim, a.seg_s, a.min_words)
    st_hs = story_of(rec_hs)
    rec["heldout_segments_per_story"] = {s: int((st_hs == s).sum()) for s in sorted(set(st_hs))}

    E = str(a.eval_story)
    cfgs = {"SEEN": (a.seen_holdout, E, ""),
            "UNSEEN": (E, "", ""),
            "MATCHED": (E, "", a.calib_stories)}
    ks_req = [int(x) for x in a.klist.split(",")]
    design = {}
    M = {}
    for name, (h, e, c) in cfgs.items():
        ev, ca = masks(st_hs, h, e, c)
        M[name] = (ev, ca)
        rng = np.random.default_rng(a.seed)
        idx_ca = np.flatnonzero(ca)
        p = rng.permutation(len(idx_ca))
        val_idx = idx_ca[p[:a.val_n]]
        cal_idx = idx_ca[p[a.val_n:]]
        ks = [k for k in ks_req if k <= len(cal_idx)]
        design[name] = {
            "holdout_story": h, "eval_story_arg": e, "calib_stories_arg": c,
            "calib_story_set": sorted(set(st_hs[ca])),
            "eval_n": int(ev.sum()), "calib_pool_incl_val": int(ca.sum()),
            "eval_sha": sha_arr(np.flatnonzero(ev)),
            "val_sha": sha_arr(val_idx),
            "calib_first_k_sha": {str(k): sha_arr(cal_idx[:k]) for k in ks},
            "calib_first_k_Xsha": {str(k): sha_arr(Xhs[cal_idx[:k]]) for k in ks},
            "k_feasible": ks,
        }
    rec["design"] = design
    d = design
    if not (d["SEEN"]["eval_sha"] == d["UNSEEN"]["eval_sha"] == d["MATCHED"]["eval_sha"]):
        problems.append("evaluation sets differ across SEEN/UNSEEN/MATCHED")
    if not np.array_equal(M["MATCHED"][1], M["SEEN"][1]):
        problems.append("MATCHED calibration mask != SEEN calibration mask")
    if d["MATCHED"]["val_sha"] != d["SEEN"]["val_sha"]:
        problems.append("MATCHED validation segments != SEEN validation segments")
    if d["MATCHED"]["calib_first_k_sha"] != d["SEEN"]["calib_first_k_sha"]:
        problems.append("MATCHED first-k calibration subsets != SEEN")
    u_ca, m_ca = M["UNSEEN"][1], M["MATCHED"][1]
    if not (np.all(u_ca[m_ca]) and u_ca.sum() > m_ca.sum()):
        problems.append("UNSEEN calibration pool is not a strict superset of MATCHED")
    if sorted(d["MATCHED"]["calib_story_set"]) != sorted(a.calib_stories.split(",")):
        problems.append(f"MATCHED calibrates on {d['MATCHED']['calib_story_set']}, "
                        f"expected {a.calib_stories}")
    if E in d["MATCHED"]["calib_story_set"] or E in d["UNSEEN"]["calib_story_set"]:
        problems.append("evaluation story appears in a calibration set")

    # ---- 0. scratch quota (checked last so the receipt still records the design)
    rec["scratch_quota"], qprobs = scratch_quota_gate(a.min_free_gb, a.min_free_files)
    problems += qprobs
    # myquota reports a periodic snapshot; also prove the output directory accepts a 16 MiB write now.
    probe = Path(a.out) / f".quota_probe_{a.tag}"
    try:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        with open(probe, "wb") as fh:
            fh.write(b"\0" * (16 << 20)); fh.flush(); os.fsync(fh.fileno())
        rec["scratch_quota"]["write_probe_16MiB"] = "ok"
    except OSError as e:
        rec["scratch_quota"]["write_probe_16MiB"] = f"FAILED: {e}"
        problems.append(f"cannot write 16 MiB to {a.out}: {e}")
    finally:
        try:
            probe.unlink()
        except OSError:
            pass

    # ---- 1. backend
    rec["backend"] = backend_gate(int(epochs.shape[1]), a.max_step_s, a.require_cuda)
    problems += rec["backend"]["problems"]

    rec["problems"] = problems
    rec["status"] = "PASS" if not problems else "FAIL"
    rec["elapsed_s"] = round(time.time() - t_start, 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"preflight_{a.tag}.json"
    f.write_text(json.dumps(rec, indent=2))
    print(f"[preflight] {a.tag}: {rec['status']} -> {f}")
    for p in problems:
        print(f"[preflight]   FAIL: {p}")
    sys.exit(0 if not problems else 3)


if __name__ == "__main__":
    main()
