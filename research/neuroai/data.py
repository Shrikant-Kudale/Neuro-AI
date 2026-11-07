"""Cohort construction, caching and stratified train/val/test splitting."""
from __future__ import annotations
import os
import numpy as np
from . import config as C, generator, features

CACHE = os.path.join(os.path.dirname(__file__), "..", "artifacts", "cache")


def build_cohort(profile: str, n_per_class: int = 300, seed: int = C.DEFAULT_SEED, cache: bool = True,
                 verbose: bool = True, tag: str = "", **gen_kw):
    """Generate ``3·n_per_class`` virtual subjects and extract features. Raw EEG is not cached (≈1.6 GB);
    the 570-d PAC vectors and behavioural scores are (they are all the model sees)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{profile}{tag}_{n_per_class}_{seed}.npz")
    if cache and os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    ss = np.random.SeedSequence(seed).spawn(3 * n_per_class)
    n = 3 * n_per_class
    pac = np.zeros((n, 570), np.float32); X = np.zeros((n, C.N_TOKENS, C.TOKEN_DIM), np.float32)
    beh = np.zeros((n, 3), np.float32); y = np.zeros(n, np.int64); flagged = np.zeros(n, np.int32)
    sev = np.zeros(n, np.float32); aff = np.zeros((n, C.N_CH), np.float32)
    first_eeg = None
    for i in range(n):
        cls = i // n_per_class
        s = generator.generate_subject(cls, np.random.default_rng(ss[i]), profile, **gen_kw)
        v, fl = features.pac_vector(s.eeg)
        pac[i], beh[i], y[i], flagged[i] = v, s.beh, cls, fl.sum()
        X[i] = features.to_tokens(v, s.beh)
        sev[i], aff[i] = s.meta["severity"], s.meta["affected"]
        if first_eeg is None:
            first_eeg = {}
        if i % n_per_class == 0:
            first_eeg[cls] = s.eeg
        if verbose and (i + 1) % 150 == 0:
            print(f"    [{profile}{tag}] subject {i + 1}/{n}", flush=True)
    out = dict(pac=pac, X=X, beh=beh, y=y, flagged=flagged, severity=sev, affected=aff,
               example_eeg=np.stack([first_eeg[c] for c in range(3)]))
    if cache:
        np.savez_compressed(path, **out)
    return out


def split_indices(y: np.ndarray, seed: int = C.DEFAULT_SEED, fr=(0.70, 0.15, 0.15)):
    """Stratified 70/15/15 split → (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_tr, n_va = int(round(len(idx) * fr[0])), int(round(len(idx) * fr[1]))
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va]); te += list(idx[n_tr + n_va:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))
