"""Explainability computed from the *trained* model — no hand-written attention maps.

* attention : last-layer [CLS]→epoch attention (mean over 8 heads) and attention rollout (Abnar & Zuidema).
* occlusion : drop in true-class probability when a channel / feature group is replaced by its
              training mean (model-agnostic, works for every baseline too)."""
from __future__ import annotations
import numpy as np
import torch
from . import config as C


@torch.no_grad()
def attention_tensors(model, X: np.ndarray, bs=128):
    """→ last-layer attention (N,H,16,16) and rollout CLS row (N,16)."""
    model.eval(); last, roll = [], []
    for i in range(0, len(X), bs):
        _, atts = model(torch.from_numpy(X[i:i + bs]), return_attn=True)
        last.append(atts[-1].numpy())
        R = None
        for a in atts:
            A = a.mean(1)
            A = 0.5 * A + 0.5 * torch.eye(A.size(-1))
            A = A / A.sum(-1, keepdim=True)
            R = A if R is None else A @ R
        roll.append(R[:, 0].numpy())
    return np.concatenate(last), np.concatenate(roll)


def attention_summary(model, X, y, pred):
    last, roll = attention_tensors(model, X)
    cls_row = last[:, :, 0, 1:].mean(1)                         # (N,15) CLS → epochs, head-mean
    out = {}
    for name, labels in (("by_true", y), ("by_pred", pred)):
        out[name] = {C.CLASS_NAMES[c]: dict(
            cls_to_epoch=cls_row[labels == c].mean(0).round(5).tolist(),
            rollout=roll[labels == c][:, 1:].mean(0).round(5).tolist(),
            matrix=last[labels == c].mean((0, 1)).round(5).tolist()) for c in range(3) if (labels == c).any()}
    heads = last[:, :, 0, 1:].mean(0)                           # (H,15)
    out["per_head_cls"] = heads.round(5).tolist()
    # concentration: share of CLS attention on epochs 6–9 (1-indexed tokens 6..9, seconds 10–18) vs uniform 4/15
    out["share_tokens_6_9"] = {C.CLASS_NAMES[c]: float(cls_row[y == c][:, 5:9].sum(1).mean()) for c in range(3)}
    out["uniform_share_6_9"] = 4 / 15
    return out


def _prob_true(model, X, y):
    from .train import predict_proba
    p = predict_proba(model, X)
    return p[np.arange(len(y)), y], p


def occlusion_importance(model, X, y, fill: np.ndarray):
    """``fill`` (41,) training-mean vector. Importance = mean drop in true-class probability (per true class)."""
    base, _ = _prob_true(model, X, y)
    def drop(cols):
        Xo = X.copy(); Xo[:, :, cols] = fill[cols]
        p, _ = _prob_true(model, Xo, y)
        return base - p
    ch = np.zeros((3, C.N_CH)); grp = {}
    for c in range(C.N_CH):
        d = drop([c, C.N_CH + c])
        for k in range(3):
            ch[k, c] = d[y == k].mean()
    tg = drop(list(range(C.N_CH))); ab = drop(list(range(C.N_CH, C.PAC_DIM)))
    beh = np.array([[drop([C.PAC_DIM + j])[y == k].mean() for j in range(3)] for k in range(3)])
    allbeh = drop([C.PAC_DIM, C.PAC_DIM + 1, C.PAC_DIM + 2])
    grp = dict(theta_gamma=[float(tg[y == k].mean()) for k in range(3)],
               alpha_beta=[float(ab[y == k].mean()) for k in range(3)],
               behavioural=[float(allbeh[y == k].mean()) for k in range(3)])
    return dict(channel=ch.round(5).tolist(), behavioural=beh.round(5).tolist(), groups=grp,
                channel_overall=ch.mean(0).round(5).tolist())
