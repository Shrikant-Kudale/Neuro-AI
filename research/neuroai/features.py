"""Pre-processing, Tort Modulation Index and feature fusion ."""
from __future__ import annotations
import numpy as np
from . import config as C
from .dsp import design_sos, filtfilt_zero_phase, analytic

SOS = {b: design_sos(*C.BANDS[b]) for b in ("theta", "gamma", "alpha", "beta")}


def modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = C.N_BINS) -> np.ndarray:
    """Tort et al. (2010) MI. ``phase``/``amp``: (..., T) → (...)  ; MI = KL(P‖U)/log N = 1 − H(P)/log N."""
    lead = phase.shape[:-1]
    T = phase.shape[-1]
    b = np.floor((phase + np.pi) / (2 * np.pi / n_bins)).astype(np.int64).clip(0, n_bins - 1)
    flat_idx = (np.arange(int(np.prod(lead)))[:, None] * n_bins + b.reshape(-1, T)).ravel()
    tot = np.bincount(flat_idx, weights=amp.reshape(-1), minlength=int(np.prod(lead)) * n_bins)
    cnt = np.bincount(flat_idx, minlength=int(np.prod(lead)) * n_bins)
    mean_amp = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0.0).reshape(-1, n_bins)
    p = mean_amp / (mean_amp.sum(-1, keepdims=True) + 1e-12)
    h = -(p * np.log(p + 1e-12)).sum(-1)
    return ((np.log(n_bins) - h) / np.log(n_bins)).reshape(lead)


def epoch(x: np.ndarray) -> np.ndarray:
    """(..., 7680) → (..., 15, 512) non-overlapping 2 s epochs."""
    return x[..., :C.N_EPOCHS * C.EPOCH_LEN].reshape(*x.shape[:-1], C.N_EPOCHS, C.EPOCH_LEN)


def pac_matrix(eeg: np.ndarray, artifact_uv: float = C.ARTIFACT_UV, reject: bool = True):
    """eeg (19, 7680) → (tg (19,15), ab (19,15), flagged (19,15) bool)."""
    out = {}
    for name, (ph_b, am_b) in C.PAC_PAIRS.items():
        ph = np.angle(analytic(epoch(filtfilt_zero_phase(SOS[ph_b], eeg))))
        am = np.abs(analytic(epoch(filtfilt_zero_phase(SOS[am_b], eeg))))
        out[name] = modulation_index(ph, am)
    flagged = (np.abs(epoch(eeg)).max(-1) > artifact_uv) if reject else np.zeros((C.N_CH, C.N_EPOCHS), bool)
    if flagged.any():
        for name in out:
            out[name] = interpolate_flagged(out[name], flagged)
    return out["tg"].astype(np.float32), out["ab"].astype(np.float32), flagged


def interpolate_flagged(mi: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Replace flagged epochs by linear interpolation from the nearest clean neighbours."""
    mi = mi.copy()
    x = np.arange(mi.shape[1])
    for c in range(mi.shape[0]):
        bad = flagged[c]
        if bad.all() or not bad.any():
            continue
        mi[c, bad] = np.interp(x[bad], x[~bad], mi[c, ~bad])
    return mi


def pac_vector(eeg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """→ (570,) vector [tg(19×15) | ab(19×15)] and the artefact mask."""
    tg, ab, fl = pac_matrix(eeg)
    return np.concatenate([tg.ravel(), ab.ravel()]), fl


def to_tokens(pac_vec: np.ndarray, beh: np.ndarray) -> np.ndarray:
    """(570,) + (3,) → (15, 41): per-epoch [tg(19), ab(19), ETVS, GAI, SSCS]."""
    tg = pac_vec[:C.N_CH * C.N_EPOCHS].reshape(C.N_CH, C.N_EPOCHS)
    ab = pac_vec[C.N_CH * C.N_EPOCHS:].reshape(C.N_CH, C.N_EPOCHS)
    tok = np.concatenate([tg.T, ab.T], axis=1)                       # (15, 38)
    return np.concatenate([tok, np.tile(beh[None, :], (C.N_EPOCHS, 1))], axis=1).astype(np.float32)


def extract_cohort(eeg: np.ndarray, beh: np.ndarray, verbose: bool = False):
    n = eeg.shape[0]
    pac = np.zeros((n, 2 * C.N_CH * C.N_EPOCHS), np.float32)
    X = np.zeros((n, C.N_TOKENS, C.TOKEN_DIM), np.float32)
    flagged = np.zeros(n, np.int32)
    for i in range(n):
        v, fl = pac_vector(eeg[i])
        pac[i] = v
        X[i] = to_tokens(v, beh[i])
        flagged[i] = int(fl.sum())
        if verbose and (i + 1) % 150 == 0:
            print(f"    features {i + 1}/{n}", flush=True)
    return pac, X, flagged


class MinMax:
    """Per-feature (pooled over epochs) min-max scaler fitted on the training split only."""
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.lo = flat.min(0)
        self.scale = np.where(flat.max(0) - self.lo > 1e-12, flat.max(0) - self.lo, 1.0)
        return self
    def transform(self, X):
        return ((X - self.lo) / self.scale).astype(np.float32)
