#!/usr/bin/env python
"""Contrastive brain -> contextual-LLM retrieval decoder (device-agnostic).

Generalised from Paper 4's `p3_train_infonce.py`. Same proven core (1x1 spatial
mix over sensors + dilated temporal ResBlocks + symmetric InfoNCE + learnable
temperature + best-val checkpoint + shuffled-target control), with two additions
for BCI_foundation M0:

  * `--n_ch auto`  : infer sensor count from epochs (306 Neuromag / 208 KIT / ...)
  * `--split_mode` : {session, subject}
        session -> hold out one recording session   (WITHIN-subject ceiling)
        subject -> hold out one whole subject (LOSO) (CROSS-subject generalisation)

The cross-subject (LOSO) number is the M0 gate this project cares about: can a
decoder trained on other people retrieve perceived language from a NEW person's
brain above chance? Paper 4 only tested within-subject.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- data ------------------------------------------
def load_meg_window(epochs_mmap, idxs, t0, t1, decim, chunk=4000):
    idxs = np.asarray(idxs)
    order = np.argsort(idxs); sidx = idxs[order]
    sl = slice(t0, t1, decim)
    Twin = len(range(*sl.indices(epochs_mmap.shape[2])))
    out = np.empty((len(idxs), epochs_mmap.shape[1], Twin), dtype=np.float32)
    buf = np.empty_like(out)
    for c0 in range(0, len(sidx), chunk):
        cs = sidx[c0:c0 + chunk]
        buf[c0:c0 + len(cs)] = np.asarray(epochs_mmap[cs], dtype=np.float32)[:, :, sl]
    out[order] = buf
    return out


# ----------------------------- model -----------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x):
        return x + F.gelu(self.bn(self.conv(x)))


class BrainEncoder(nn.Module):
    def __init__(self, n_ch, hidden=320, out_dim=768, n_blocks=5):
        super().__init__()
        self.in_bn = nn.BatchNorm1d(n_ch)
        self.spatial = nn.Conv1d(n_ch, hidden, 1)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, 2 ** (i % 5)) for i in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        x = self.in_bn(x)
        x = F.gelu(self.spatial(x))
        for b in self.blocks:
            x = b(x)
        return self.head(x.mean(dim=-1))


class LLMHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
            nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, y):
        return self.net(y)


class IdentityHead(nn.Module):
    """Target head that learns NOTHING: retrieval is against the raw frozen GPT-2 vectors.

    The default LLMHead learns a projection of the target, so an arm can shape the target space to
    suit itself (a reviewer's concern: our timing result might be about that learned space, not
    about fixed-target retrieval). This head removes that freedom entirely -- the model retrieves the
    unmodified embedding -- so it is the apples-to-apples check that the confound is not an artifact
    of a learnable target. It requires the encoder's out_dim to equal the target dim (768)."""

    def __init__(self, in_dim=768, out_dim=768):
        super().__init__()
        assert in_dim == out_dim, "IdentityHead needs proj_dim == target dim (768)"

    def forward(self, y):
        return y


# ----------------------------- eval ------------------------------------------
@torch.no_grad()
def embed_all(brain, llm, X, Y, device, bs=1024):
    brain.eval(); llm.eval()
    zm, zl = [], []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(Y[i:i + bs]).to(device)
        zm.append(F.normalize(brain(xb), dim=-1).cpu())
        zl.append(F.normalize(llm(yb), dim=-1).cpu())
    return torch.cat(zm), torch.cat(zl)


def retrieval_metrics(zm, zl):
    N = zm.shape[0]
    sim = zm @ zl.T
    true_sim = sim.diag().unsqueeze(1)
    rank = (sim > true_sim).sum(dim=1).numpy() + 1
    return {
        "n_candidates": int(N),
        "top1": float(np.mean(rank <= 1)), "top5": float(np.mean(rank <= 5)),
        "top10": float(np.mean(rank <= 10)),
        "median_rank": float(np.median(rank)), "mean_rank": float(np.mean(rank)),
        "chance_top1": 1.0 / N, "chance_top5": 5.0 / N, "chance_top10": 10.0 / N,
        "top10_over_chance": float(np.mean(rank <= 10) / (10.0 / N)),
    }


# ----------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--ctx_meta", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--t0", type=int, default=50)
    ap.add_argument("--t1", type=int, default=250)
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--split_mode", choices=["session", "subject"], default="subject")
    ap.add_argument("--test_session", default="12")
    ap.add_argument("--test_task", default="")
    ap.add_argument("--test_subject", default="01")
    ap.add_argument("--max_train", type=int, default=90000)
    ap.add_argument("--proj_dim", type=int, default=768)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--n_blocks", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="masc_loso")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(data_dir / "trial_metadata.tsv", sep="\t",
                       dtype={"subject": str, "session": str, "task": str},
                       usecols=lambda c: c in
                       {"trial_index", "session", "task", "subject", "word_text"})
    cmeta = json.loads(Path(args.ctx_meta).read_text())
    layers_kept = cmeta["layers_kept"]
    if args.layer not in layers_kept:
        raise SystemExit(f"layer {args.layer} not in extracted {layers_kept}")
    li = layers_kept.index(args.layer)
    ctx = np.load(args.ctx, mmap_mode="r")
    print(f"[ctx] {ctx.shape} using layer {args.layer} (idx {li})", flush=True)

    # ---- splits ----
    if args.split_mode == "subject":
        subj = meta["subject"].values.astype(str)
        test_idx = np.where(subj == str(args.test_subject))[0]
        train_pool = np.where(subj != str(args.test_subject))[0]
        held = f"subject={args.test_subject}"
    else:
        sess = meta["session"].values.astype(str)
        is_test = (sess == str(args.test_session))
        if args.test_task:
            is_test &= (meta["task"].values.astype(str) == str(args.test_task))
        test_idx = np.where(is_test)[0]
        train_pool = np.where(sess != str(args.test_session))[0]
        held = f"session={args.test_session}/{args.test_task}"
    if len(test_idx) == 0:
        raise SystemExit(f"empty test set for {held}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_pool)
    if len(train_pool) > args.max_train:
        train_pool = train_pool[:args.max_train]
    val_n = min(5000, len(train_pool) // 20)
    val_idx = train_pool[:val_n]; train_idx = train_pool[val_n:]
    print(f"[split] mode={args.split_mode} held-out {held} | "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
          flush=True)

    # ---- load MEG ----
    epochs = np.load(data_dir / "epochs_data.npy", mmap_mode="r")
    n_ch = epochs.shape[1]
    print(f"[meg] {epochs.shape} n_ch={n_ch}; window [{args.t0}:{args.t1}:{args.decim}]",
          flush=True)
    Xtr = load_meg_window(epochs, train_idx, args.t0, args.t1, args.decim)
    Xva = load_meg_window(epochs, val_idx, args.t0, args.t1, args.decim)
    Xte = load_meg_window(epochs, test_idx, args.t0, args.t1, args.decim)

    def gather_ctx(idx):
        return np.ascontiguousarray(ctx[idx, li, :].astype(np.float32))
    Ytr, Yva, Yte = gather_ctx(train_idx), gather_ctx(val_idx), gather_ctx(test_idx)

    def clean(X, Y):
        good = np.isfinite(Y).all(1) & (np.linalg.norm(Y, axis=1) > 0)
        return X[good], Y[good]
    Xtr, Ytr = clean(Xtr, Ytr); Xva, Yva = clean(Xva, Yva); Xte, Yte = clean(Xte, Yte)
    print(f"[clean] train={len(Xtr)} val={len(Xva)} test={len(Xte)}", flush=True)

    # ---- model ----
    brain = BrainEncoder(n_ch, args.hidden, args.proj_dim, args.n_blocks).to(device)
    llm = LLMHead(Ytr.shape[1], args.proj_dim).to(device)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32,
                                             device=device))
    params = list(brain.parameters()) + list(llm.parameters()) + [logit_scale]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n = len(Xtr)

    best_val = -1.0; best_state = None
    for ep in range(args.epochs):
        brain.train(); llm.train()
        perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, args.batch):
            bidx = perm[i:i + args.batch]
            if len(bidx) < 16:
                continue
            xb = torch.from_numpy(Xtr[bidx]).to(device)
            yb = torch.from_numpy(Ytr[bidx]).to(device)
            zm = F.normalize(brain(xb), dim=-1)
            zl = F.normalize(llm(yb), dim=-1)
            scale = logit_scale.clamp(max=np.log(100)).exp()
            logits = scale * zm @ zl.T
            labels = torch.arange(len(bidx), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) +
                          F.cross_entropy(logits.T, labels))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        zm, zl = embed_all(brain, llm, Xva, Yva, device)
        vm = retrieval_metrics(zm, zl)
        if vm["top10"] > best_val:
            best_val = vm["top10"]
            best_state = ({k: v.detach().cpu().clone() for k, v in brain.state_dict().items()},
                          {k: v.detach().cpu().clone() for k, v in llm.state_dict().items()})
        if ep % 2 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:02d}] loss={tot/max(nb,1):.4f} val top1={vm['top1']:.4f} "
                  f"top10={vm['top10']:.4f} (x{vm['top10_over_chance']:.1f})", flush=True)

    if best_state is not None:
        brain.load_state_dict(best_state[0]); llm.load_state_dict(best_state[1])

    zm, zl = embed_all(brain, llm, Xte, Yte, device)
    test_m = retrieval_metrics(zm, zl)
    ctrl = retrieval_metrics(zm, zl[torch.randperm(zl.shape[0])])

    result = {
        "tag": args.tag, "target": "contextual", "layer": args.layer,
        "split_mode": args.split_mode, "held_out": held,
        "n_ch": int(n_ch), "window_samples": [args.t0, args.t1, args.decim],
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "epochs": args.epochs, "val_best_top10": best_val,
        "test": test_m, "shuffled_control": ctrl,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / f"result_{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(f"\n===== TEST ({held}, N={test_m['n_candidates']}) =====")
    print(f"  top1={test_m['top1']*100:.2f}% top5={test_m['top5']*100:.2f}% "
          f"top10={test_m['top10']*100:.2f}% => {test_m['top10_over_chance']:.1f}x chance")
    print(f"  median rank {test_m['median_rank']:.0f}/{test_m['n_candidates']} | "
          f"shuffled top10={ctrl['top10']*100:.3f}% (~chance)")
    print(f"[save] {out_dir}/result_{args.tag}.json", flush=True)


if __name__ == "__main__":
    main()
