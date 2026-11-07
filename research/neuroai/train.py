"""Training loop — Adam(3e-4, wd 1e-5), cosine annealing to 1e-6, cross-entropy, batch 32,
early stopping (patience 10 on validation loss), best-weights restore."""
from __future__ import annotations
import copy, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, torch.get_num_threads()))


def seed_all(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)


def predict_proba(model, X: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.softmax(model(torch.from_numpy(X[i:i + bs])), -1).numpy())
    return np.concatenate(out)


def fit(model, Xtr, ytr, Xva, yva, epochs=80, lr=3e-4, wd=1e-5, bs=32, patience=10, seed=42, verbose=False):
    seed_all(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, log = 1e9, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); perm = torch.randperm(len(Xt), generator=g); tl = tc = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:      # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            out = model(Xt[idx]); loss = lossf(out, yt[idx]); loss.backward(); opt.step()
            tl += loss.item() * len(idx); tc += (out.argmax(1) == yt[idx]).sum().item()
        sched.step()
        model.eval()
        with torch.no_grad():
            ov = model(Xv); vl = lossf(ov, yv).item(); va = (ov.argmax(1) == yv).float().mean().item()
        log.append(dict(epoch=ep, train_loss=tl / len(Xt), train_acc=tc / len(Xt), val_loss=vl, val_acc=va,
                        lr=sched.get_last_lr()[0]))
        if verbose and ep % 5 == 0:
            print(f"      ep{ep:3d} tr {tl/len(Xt):.3f}/{tc/len(Xt):.3f} va {vl:.3f}/{va:.3f}", flush=True)
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return log
