#!/usr/bin/env python
"""M1a: single-cohort segment LOSO with HONEST metrics (adjacency guard + CIs).

Same proven segment decoder as M0, but:
  * evaluation uses guarded_retrieval_metrics (mask same-recording +/-adj_s
    neighbours; bootstrap CI) alongside the naive M0 metric for comparison;
  * reports the permutation null;
  * pools ALL other subjects (no cap) for each held-out subject.

Gate: does the M0 5-7x segment floor SURVIVE the adjacency guard with a clean
shuffled control and a CI that excludes 1x chance?
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_infonce import (BrainEncoder, LLMHead, load_meg_window,
                           embed_all, retrieval_metrics)
from eval_metrics import guarded_retrieval_metrics, multi_shuffle_null

REC_KEYS = ["subject", "session", "task", "run"]


def build_segments_with_meta(meta, sel_mask, epochs, ctx, li, t0, t1, decim,
                             seg_s, min_words):
    """Returns Xs (n,C,Tw), Ys (n,768), rec_ids (n,), onsets (n,)."""
    sub = meta[sel_mask]
    seg_members, seg_rec, seg_on = [], [], []
    for rkey, rec in sub.groupby([k for k in REC_KEYS if k in sub.columns], sort=False):
        onset = rec["onset_s"].values.astype(float)
        seg_id = np.floor(onset / seg_s).astype(int)
        tix = rec["trial_index"].values
        rid = "|".join(map(str, rkey)) if isinstance(rkey, tuple) else str(rkey)
        for s in np.unique(seg_id):
            m = seg_id == s
            members = tix[m]
            if len(members) >= min_words:
                seg_members.append(members)
                seg_rec.append(rid)
                seg_on.append(float(onset[m].mean()))
    if not seg_members:
        raise SystemExit("no segments built")
    all_idx = np.unique(np.concatenate(seg_members))
    Xw = load_meg_window(epochs, all_idx, t0, t1, decim)
    pos = {int(g): k for k, g in enumerate(all_idx)}
    Yw = ctx[all_idx, li, :].astype(np.float32)
    n = len(seg_members)
    Xs = np.empty((n, Xw.shape[1], Xw.shape[2]), np.float32)
    Ys = np.empty((n, Yw.shape[1]), np.float32)
    for k, members in enumerate(seg_members):
        rows = [pos[int(mm)] for mm in members]
        Xs[k] = Xw[rows].mean(0)
        Ys[k] = Yw[rows].mean(0)
    good = np.isfinite(Ys).all(1) & (np.linalg.norm(Ys, axis=1) > 0)
    return (Xs[good], Ys[good], np.array(seg_rec)[good], np.array(seg_on)[good])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--ctx_meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--t0", type=int, default=50)
    ap.add_argument("--t1", type=int, default=250)
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--seg_s", type=float, default=3.0)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--adj_s", type=float, default=10.0)
    ap.add_argument("--test_subject", required=True)
    ap.add_argument("--proj_dim", type=int, default=768)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--n_blocks", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="m1a_seg")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t",
                       dtype={"subject": str, "session": str, "task": str, "run": str},
                       usecols=lambda c: c in
                       {"trial_index", "subject", "session", "task", "run", "onset_s"})
    cmeta = json.loads(Path(args.ctx_meta).read_text())
    li = cmeta["layers_kept"].index(args.layer)
    ctx = np.load(args.ctx, mmap_mode="r")
    epochs = np.load(data_dir / "epochs_data.npy", mmap_mode="r")
    n_ch = epochs.shape[1]

    subj = meta["subject"].values.astype(str)
    test_mask = subj == str(args.test_subject)
    train_mask = subj != str(args.test_subject)
    print(f"[m1a] held-out subject={args.test_subject} n_ch={n_ch}", flush=True)

    Xte, Yte, rec_te, on_te = build_segments_with_meta(
        meta, test_mask, epochs, ctx, li, args.t0, args.t1, args.decim,
        args.seg_s, args.min_words)
    Xtr, Ytr, _, _ = build_segments_with_meta(
        meta, train_mask, epochs, ctx, li, args.t0, args.t1, args.decim,
        args.seg_s, args.min_words)
    print(f"[m1a] test_seg={len(Xte)} train_seg={len(Xtr)}", flush=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(Xtr)); vn = min(2000, len(Xtr) // 10)
    va, trn = perm[:vn], perm[vn:]
    Xva, Yva = Xtr[va], Ytr[va]; Xtr, Ytr = Xtr[trn], Ytr[trn]

    brain = BrainEncoder(n_ch, args.hidden, args.proj_dim, args.n_blocks).to(device)
    llm = LLMHead(Ytr.shape[1], args.proj_dim).to(device)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32,
                                             device=device))
    opt = torch.optim.AdamW(list(brain.parameters()) + list(llm.parameters())
                            + [logit_scale], lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Xtr); best_val = -1.0; best = None
    for ep in range(args.epochs):
        brain.train(); llm.train()
        pm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, args.batch):
            b = pm[i:i + args.batch]
            if len(b) < 16:
                continue
            xb = torch.from_numpy(Xtr[b]).to(device)
            yb = torch.from_numpy(Ytr[b]).to(device)
            zm = F.normalize(brain(xb), dim=-1); zl = F.normalize(llm(yb), dim=-1)
            sc = logit_scale.clamp(max=np.log(100)).exp()
            logits = sc * zm @ zl.T; lab = torch.arange(len(b), device=device)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        zm, zl = embed_all(brain, llm, Xva, Yva, device)
        vm = retrieval_metrics(zm, zl)
        if vm["top10"] > best_val:
            best_val = vm["top10"]
            best = ({k: v.cpu().clone() for k, v in brain.state_dict().items()},
                    {k: v.cpu().clone() for k, v in llm.state_dict().items()})
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:02d}] loss={tot/max(nb,1):.4f} val top10={vm['top10']:.4f} "
                  f"(x{vm['top10_over_chance']:.1f})", flush=True)
    if best:
        brain.load_state_dict(best[0]); llm.load_state_dict(best[1])

    zm, zl = embed_all(brain, llm, Xte, Yte, device)
    naive = retrieval_metrics(zm, zl)                        # M0-comparable
    guarded = guarded_retrieval_metrics(zm, zl, rec_te, on_te, adj_s=args.adj_s,
                                        seed=args.seed, block_boot=True)
    ctrl = retrieval_metrics(zm, zl[torch.randperm(zl.shape[0])])  # permutation null
    shuffled_multi = multi_shuffle_null(zm, zl, n_draws=100, seed=args.seed)
    result = {"tag": args.tag, "held_out": f"subject={args.test_subject}",
              "layer": args.layer, "seg_s": args.seg_s, "n_ch": int(n_ch),
              "n_test_seg": len(Xte), "n_train_seg": len(Xtr),
              "val_best_top10": best_val,
              "naive_test": naive, "guarded_test": guarded,
              "shuffled_control": ctrl, "shuffled_multi": shuffled_multi,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out_dir / f"result_{args.tag}.json").write_text(json.dumps(result, indent=2))
    g = guarded
    print(f"\n===== M1a SEG guarded ({args.test_subject}, N={g['n_test']}, "
          f"pool~{g['mean_pool_size']:.0f}) =====")
    print(f"  guarded top10={g['top10']*100:.2f}% CI{[round(x*100,2) for x in g['top10_ci95']]} "
          f"=> {g['top10_over_chance']:.1f}x | naive top10 x{naive['top10_over_chance']:.1f} | "
          f"shuffled x{ctrl['top10_over_chance']:.1f}")
    print(f"[save] {out_dir}/result_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
