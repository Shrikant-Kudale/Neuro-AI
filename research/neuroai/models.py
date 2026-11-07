"""PyTorch models: the Transformer encoder and its MLP / CNN baselines."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from . import config as C


class EncoderBlock(nn.Module):
    """Pre-LN Transformer block with explicit Q/K/V projections (so attention can be exported)."""
    def __init__(self, d=128, heads=8, ffn=256, p=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d, eps=1e-5), nn.LayerNorm(d, eps=1e-5)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.f1, self.f2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.drop_attn, self.drop = nn.Dropout(p), nn.Dropout(p)

    def forward(self, x, need_attn=False):
        B, T, D = x.shape
        z = self.ln1(x)
        sp = lambda t: t.view(B, T, self.h, self.dk).transpose(1, 2)
        q, k, v = sp(self.q(z)), sp(self.k(z)), sp(self.v(z))
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk), dim=-1)    # (B,H,T,T)
        ctx = (self.drop_attn(att) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.o(ctx))
        x = x + self.drop(self.f2(Fn.gelu(self.f1(self.ln2(x)))))
        return x, (att if need_attn else None)


class NeuroTransformer(nn.Module):
    """Input (B, 15, in_dim) → embed(128) + learnable positions + [CLS] → 4 encoder blocks → MLP head."""
    def __init__(self, in_dim=C.TOKEN_DIM, seq=C.N_TOKENS, d=128, heads=8, layers=4, ffn=256, p=0.1,
                 n_classes=C.N_CLASSES):
        super().__init__()
        self.in_dim, self.seq, self.d = in_dim, seq, d
        self.embed = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, seq + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.ModuleList(EncoderBlock(d, heads, ffn, p) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x, return_attn=False):
        h = self.embed(x)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], 1) + self.pos
        atts = []
        for blk in self.blocks:
            h, a = blk(h, need_attn=return_attn)
            atts.append(a)
        logits = self.head(self.ln_f(h)[:, 0])
        return (logits, atts) if return_attn else logits


class MLPBaseline(nn.Module):
    """Flattened 573-d input (15×38 PAC + 3 behavioural) → 256 → 128 → 3, ReLU + BatchNorm."""
    def __init__(self, use_beh=True, n_classes=C.N_CLASSES):
        super().__init__()
        self.use_beh = use_beh
        d_in = C.N_TOKENS * C.PAC_DIM + (3 if use_beh else 0)
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        flat = x[:, :, :C.PAC_DIM].flatten(1)
        if self.use_beh:
            flat = torch.cat([flat, x[:, 0, C.PAC_DIM:]], 1)
        return self.net(flat)


class CNNBaseline(nn.Module):
    """Three Conv1d layers (32/64/128, k=5, max-pool) over the 15-epoch axis, GAP, 128→64→3 head."""
    def __init__(self, in_dim=C.TOKEN_DIM, n_classes=C.N_CLASSES):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv1d(in_dim, 32, 5, padding=2), nn.Conv1d(32, 64, 5, padding=2), nn.Conv1d(64, 128, 5, padding=2)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        h = x.transpose(1, 2)
        h = Fn.max_pool1d(Fn.relu(self.c1(h)), 2)
        h = Fn.max_pool1d(Fn.relu(self.c2(h)), 2)
        h = Fn.relu(self.c3(h)).mean(-1)
        return self.head(h)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
