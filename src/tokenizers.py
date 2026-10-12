"""MB-1 encoders: STATIC patch vs DYNAMIC (flow + VQ) token, matched interface.

Both take a segment window x (B, C, T) and return (z, aux_loss) with z in R^out_dim,
so either can drop into the existing InfoNCE retrieval head (train_infonce.LLMHead).
aux_loss is the VQ commitment loss for the dynamic encoder, 0 for the static one.

The contrast isolates the manifold-tokenization thesis: does representing HOW the
latent state moves (local flow Δz + discrete dynamical-regime codes) transfer better
in low-shot than a static amplitude-patch representation? The `use_flow` / `use_vq`
switches give the attribution ablations (velocity-removed, VQ-removed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticPatchEncoder(nn.Module):
    """Amplitude patches -> transformer -> attention pool. Representative of the
    patch-based EEG-FM family (LaBraM/EEGPT/CBraMod) the program critiques."""
    def __init__(self, n_ch, hidden=192, out_dim=768, patch=10, d=192,
                 n_layers=2, n_heads=4, max_patch=64):
        super().__init__()
        self.patch = patch
        self.in_bn = nn.BatchNorm1d(n_ch)
        self.spatial = nn.Conv1d(n_ch, hidden, 1)
        self.embed = nn.Linear(hidden * patch, d)
        self.pos = nn.Parameter(torch.randn(1, max_patch, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d,
                                           batch_first=True, activation="gelu",
                                           dropout=0.0, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.head = nn.Sequential(nn.Linear(d, out_dim), nn.GELU(),
                                  nn.Linear(out_dim, out_dim))

    def forward(self, x):
        B = x.shape[0]
        h = F.gelu(self.spatial(self.in_bn(x)))          # (B, hidden, T)
        T = h.shape[-1]; np = T // self.patch
        h = h[:, :, :np * self.patch]
        h = h.reshape(B, h.shape[1], np, self.patch)     # (B, hidden, nP, patch)
        h = h.permute(0, 2, 1, 3).reshape(B, np, -1)     # (B, nP, hidden*patch)
        tok = self.embed(h) + self.pos[:, :np]           # (B, nP, d)
        tok = self.tr(tok)
        q = self.query.expand(B, -1, -1)                 # (B, 1, d)
        attn = torch.softmax((q @ tok.transpose(1, 2)) / (tok.shape[-1] ** 0.5), -1)
        z = (attn @ tok).squeeze(1)                      # (B, d)
        return self.head(z), x.new_zeros(())


class VectorQuantizer(nn.Module):
    """VQ-VAE codebook with straight-through estimator + commitment loss."""
    def __init__(self, n_codes=256, dim=128, beta=0.25):
        super().__init__()
        self.emb = nn.Embedding(n_codes, dim)
        self.emb.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta

    def forward(self, z):                                # z: (B, T, dim)
        flat = z.reshape(-1, z.shape[-1])
        d = ((flat ** 2).sum(1, keepdim=True)
             - 2 * flat @ self.emb.weight.t()
             + (self.emb.weight ** 2).sum(1))
        idx = d.argmin(1)
        q = self.emb(idx).view_as(z)
        commit = self.beta * F.mse_loss(q.detach(), z) + F.mse_loss(q, z.detach())
        q = z + (q - z).detach()                         # straight-through
        return q, commit, idx.view(z.shape[0], z.shape[1])


class DynamicTokenEncoder(nn.Module):
    """Latent trajectory -> local flow [z, Δz] -> VQ codes -> pool. The program's
    proposed 'neural token' (v0 slice of T_k = {q, Δt, flow(≈A)})."""
    def __init__(self, n_ch, hidden=192, out_dim=768, d_lat=128, n_codes=256,
                 use_flow=True, use_vq=True, n_conv=3):
        super().__init__()
        self.use_flow, self.use_vq = use_flow, use_vq
        self.in_bn = nn.BatchNorm1d(n_ch)
        self.spatial = nn.Conv1d(n_ch, hidden, 1)
        convs = []
        ch = hidden
        for i in range(n_conv):                          # dilated temporal trajectory model
            convs += [nn.Conv1d(ch, d_lat, 3, padding=2 ** i, dilation=2 ** i),
                      nn.BatchNorm1d(d_lat), nn.GELU()]
            ch = d_lat
        self.traj = nn.Sequential(*convs)
        flow_dim = 2 * d_lat if use_flow else d_lat
        self.proj = nn.Conv1d(flow_dim, d_lat, 1)        # flow features -> code space
        self.vq = VectorQuantizer(n_codes, d_lat) if use_vq else None
        self.head = nn.Sequential(nn.Linear(d_lat, out_dim), nn.GELU(),
                                  nn.Linear(out_dim, out_dim))

    def forward(self, x):
        h = F.gelu(self.spatial(self.in_bn(x)))          # (B, hidden, T)
        z = self.traj(h)                                 # (B, d_lat, T)  latent trajectory
        if self.use_flow:
            dz = z[:, :, 1:] - z[:, :, :-1]              # local velocity
            dz = F.pad(dz, (1, 0))                       # align to T
            z = torch.cat([z, dz], dim=1)               # (B, 2*d_lat, T)
        f = self.proj(z).transpose(1, 2)                 # (B, T, d_lat)
        if self.use_vq:
            q, commit, _ = self.vq(f)
        else:
            q, commit = f, x.new_zeros(())
        z_out = q.mean(dim=1)                            # pool over trajectory
        return self.head(z_out), commit


class ConvEncoderWrap(nn.Module):
    """The proven M1a conv encoder (dilated temporal ResBlocks) as the strong
    reference baseline to beat. Implicit temporal dynamics, no explicit flow/VQ."""
    def __init__(self, n_ch, out_dim=768, hidden=320, n_blocks=5):
        super().__init__()
        from train_infonce import BrainEncoder
        self.enc = BrainEncoder(n_ch, hidden, out_dim, n_blocks)

    def forward(self, x):
        return self.enc(x), x.new_zeros(())


def build_encoder(variant, n_ch, out_dim=768, **kw):
    if variant == "conv":
        return ConvEncoderWrap(n_ch, out_dim=out_dim)
    if variant == "static":
        return StaticPatchEncoder(n_ch, out_dim=out_dim)
    if variant == "dynamic":
        return DynamicTokenEncoder(n_ch, out_dim=out_dim, use_flow=True, use_vq=True)
    if variant == "dynamic_novq":                        # ablation: continuous flow, no VQ
        return DynamicTokenEncoder(n_ch, out_dim=out_dim, use_flow=True, use_vq=False)
    if variant == "dynamic_noflow":                      # ablation: latent position only
        return DynamicTokenEncoder(n_ch, out_dim=out_dim, use_flow=False, use_vq=True)
    if variant == "dynamic_big":                         # param-matched to conv (capacity control)
        return DynamicTokenEncoder(n_ch, hidden=320, out_dim=out_dim, d_lat=320,
                                   use_flow=True, use_vq=False, n_conv=5)
    if variant == "dynamic_big_noflow":                  # attribution: same size, NO flow
        return DynamicTokenEncoder(n_ch, hidden=320, out_dim=out_dim, d_lat=320,
                                   use_flow=False, use_vq=False, n_conv=5)
    if variant == "dynamic_big_vq":                      # n=10 VQ row: same size, flow + VQ on
        return DynamicTokenEncoder(n_ch, hidden=320, out_dim=out_dim, d_lat=320,
                                   use_flow=True, use_vq=True, n_conv=5)
    raise ValueError(f"unknown variant {variant}")
