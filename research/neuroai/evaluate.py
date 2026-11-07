"""Metrics with uncertainty: accuracy / macro-P/R/F1 / AUC-ROC, per-class table, ROC curves,
confusion matrix, expected calibration error and percentile-bootstrap confidence intervals."""
from __future__ import annotations
import numpy as np
from sklearn import metrics as M
from . import config as C


def roc_points(y_true_bin, score, n=60):
    fpr, tpr, _ = M.roc_curve(y_true_bin, score)
    g = np.linspace(0, 1, n)
    return dict(fpr=g.round(4).tolist(), tpr=np.interp(g, fpr, tpr).round(4).tolist())


def ece(proba, y, n_bins=10):
    conf, pred = proba.max(1), proba.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
            bins.append(dict(lo=float(lo), hi=float(hi), conf=float(conf[m].mean()), acc=float(acc[m].mean()), n=int(m.sum())))
    return float(e), bins


def macro_auc(y, proba):
    return float(M.roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def bootstrap_ci(y, proba, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    accs, aucs = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 3:
            continue
        accs.append((proba[i].argmax(1) == y[i]).mean()); aucs.append(macro_auc(y[i], proba[i]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(acc=q(accs), auc=q(aucs))


def full_report(y, proba, with_curves=True, with_ci=True):
    pred = proba.argmax(1)
    p, r, f, s = M.precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    rep = dict(
        accuracy=float((pred == y).mean()),
        precision=float(p.mean()), recall=float(r.mean()), f1=float(f.mean()),
        auc_macro=macro_auc(y, proba),
        auc_per_class=[float(M.roc_auc_score((y == c).astype(int), proba[:, c])) for c in range(3)],
        per_class={C.CLASS_NAMES[c]: dict(precision=float(p[c]), recall=float(r[c]), f1=float(f[c]), support=int(s[c])) for c in range(3)},
        confusion=M.confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        n_test=int(len(y)),
    )
    rep["ece"], rep["calibration_bins"] = ece(proba, y)
    if with_ci:
        rep["ci95"] = bootstrap_ci(y, proba)
    if with_curves:
        rep["roc"] = {C.CLASS_NAMES[c]: roc_points((y == c).astype(int), proba[:, c]) for c in range(3)}
    return rep
