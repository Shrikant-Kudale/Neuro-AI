"""Virtual-subject generator.

Each subject is a 19-channel, 30 s, 256 Hz resting-state recording built from physiologically
motivated components (1/f background + five band-limited oscillations) in which *theta→gamma* and
*alpha→beta* phase-amplitude coupling is imposed explicitly and disrupted according to the class.
A virtual subject also carries three behavioural biomarkers (ETVS, GAI, SSCS).

``spatial="random"`` places disruption in randomly chosen channels; ``spatial="focal"`` places a
lesion on the scalp and lets disruption decay with distance — used for out-of-distribution tests and
by the web "Virtual Patient Studio"."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import config as C
from .dsp import band_noise, analytic_padded

EPS = 1e-12


@dataclass
class Subject:
    eeg: np.ndarray           # (19, 7680) µV
    beh: np.ndarray           # (3,) ETVS, GAI, SSCS
    label: int
    meta: dict = field(default_factory=dict)


def _clip_beh(etvs, gai, sscs):
    return np.array([np.clip(etvs, 0.5, 4.0), np.clip(gai, 0.0, 0.5), np.clip(sscs, 0.0, 1.0)], np.float32)


def sample_behaviour(rng, cls: int, prof: dict, f: float, fatigue: float) -> np.ndarray:
    """Behavioural scores. In the simulation the class mean is interpolated by the latent severity
    ``f`` and every score shares a fatigue/age confounder (in units of its own class SD)."""
    out = []
    for k in ("etvs", "gai", "sscs"):
        m0, _ = prof["beh"][0][k]
        mc, sd = prof["beh"][cls][k]
        mean = m0 + (mc - m0) * f
        val = rng.normal(mean, sd * prof["beh_sd_mult"]) + fatigue * sd
        out.append(val)
    return _clip_beh(*out)


def _envelope(rng, cls: int, prof: dict) -> np.ndarray:
    """Disruption intensity per epoch in [0, 1]. Constant 1 when disruption is stationary."""
    if not prof["temporal"] or cls == 0:
        return np.ones(C.N_EPOCHS)
    base = 0.55 if cls == 1 else 0.85
    z = np.zeros(C.N_EPOCHS)
    z[0] = rng.normal()
    for t in range(1, C.N_EPOCHS):
        z[t] = 0.7 * z[t - 1] + np.sqrt(1 - 0.49) * rng.normal()
    return np.clip(base + 0.30 * z, 0.0, 1.0)


def _affected_weights(rng, cls: int, prof: dict, f: float, spatial: str, lesion: tuple | None):
    """Per-channel weight w_c ∈ [0,1] telling how strongly channel c is affected."""
    w = np.zeros(C.N_CH)
    if cls == 0:
        return w
    if spatial == "focal":
        if lesion is None:
            ang = rng.uniform(0, 2 * np.pi)
            rad = rng.uniform(0.2, 0.7)
            lesion = (rad * np.cos(ang), rad * np.sin(ang), 0.45 if cls == 2 else 0.30)
        cx, cy, sigma = lesion
        d2 = ((C.COORDS - np.array([cx, cy])) ** 2).sum(1)
        return np.exp(-d2 / (2 * sigma ** 2))
    n = prof["n_affected"][cls]
    if prof["latent_sd"] > 0:
        n = int(np.clip(round(n * (0.5 + 0.5 * f)), 1, C.N_CH))
    idx = rng.choice(C.N_CH, size=n, replace=False)
    w[idx] = 1.0
    return w


def generate_subject(cls: int, rng: np.random.Generator, profile: str | dict = "sim",
                     spatial: str | None = None, lesion: tuple | None = None,
                     noise_mult: float = 1.0, overrides: dict | None = None) -> Subject:
    prof = dict(C.PROFILES[profile] if isinstance(profile, str) else profile)
    if overrides:
        prof.update(overrides)
    spatial = spatial or prof["spatial"]
    N, NC = C.N_SAMPLES, C.N_CH

    # latent severity and confounders -----------------------------------------------------------
    f = 1.0
    if prof["latent_sd"] > 0 and cls > 0:
        f = float(np.clip(1.0 + prof["latent_sd"] * rng.normal(), 0.0, 1.8))
    fatigue = rng.normal(0, prof["fatigue_sd"]) if prof["fatigue_sd"] > 0 else 0.0
    env = _envelope(rng, cls, prof)                               # (15,)
    env_t = np.interp(np.arange(N) / C.EPOCH_LEN - 0.5, np.arange(C.N_EPOCHS), env)  # (N,)
    w = _affected_weights(rng, cls, prof, f, spatial, lesion)       # (19,)

    # coupling depths ---------------------------------------------------------------------------
    d_tg = max(0.02, rng.normal(*prof["tg_depth"]))
    d_ab = max(0.02, rng.normal(*prof["ab_depth"]))
    s_tg = rng.uniform(*prof["tg_scale"][cls])
    s_ab = rng.uniform(*prof["ab_scale"][cls])
    red = (1.0 - s_tg) * f                                         # fractional theta-gamma loss
    gain = (s_ab - 1.0) * f                                        # fractional alpha-beta gain
    chan_jit_tg = rng.normal(1.0, 0.10, NC).clip(0.5, 1.5)
    chan_jit_ab = rng.normal(1.0, 0.10, NC).clip(0.5, 1.5)
    G = C.COUPLING_GAIN
    m_tg = (G * d_tg * chan_jit_tg[:, None] * (1.0 - red * w[:, None] * env_t[None, :]).clip(0.0, None)).clip(0, C.MAX_DEPTH)
    m_ab = (G * d_ab * chan_jit_ab[:, None] * (1.0 + gain * w[:, None] * env_t[None, :])).clip(0, C.MAX_DEPTH)
    g_boost = 1.0 + (prof["gamma_boost"][cls] - 1.0) * f * w                # (19,)

    # oscillatory components --------------------------------------------------------------------
    A = C.COMPONENT_AMP
    def amp(name):
        mu, sd = A[name]
        return np.maximum(0.05, rng.normal(mu, sd, size=(NC, 1)))

    delta = band_noise(rng, *C.BANDS["delta"], (NC,)) * amp("delta")
    theta = band_noise(rng, *C.BANDS["theta"], (NC,))
    alpha = band_noise(rng, *C.BANDS["alpha"], (NC,))
    beta = band_noise(rng, *C.BANDS["beta"], (NC,))
    gamma = band_noise(rng, *C.BANDS["gamma"], (NC,))
    pink = band_noise(rng, 1.0, 100.0, (NC,), slope=1.0) * A["pink"]

    # phase of the slow rhythm is taken from the full-length synthetic signal (before crop effects)
    ph_th = np.angle(analytic_padded(theta))
    ph_al = np.angle(analytic_padded(alpha))
    theta *= amp("theta")
    alpha *= amp("alpha")
    gamma = gamma * amp("gamma") * g_boost[:, None] * (1.0 + m_tg * np.cos(ph_th))
    beta = beta * amp("beta") * (1.0 + m_ab * np.cos(ph_al))

    white = rng.normal(0.0, prof["white_sd"] * noise_mult, size=(NC, N))
    sig = delta + theta + alpha + beta + gamma + pink + white

    # scale to physiological µV, apply per-channel electrode gain ---------------------------------
    rms = rng.uniform(*C.RMS_UV)
    sig = sig / (sig.std(axis=1, keepdims=True) + EPS) * rms
    if prof["impedance_sd"] > 0:
        sig = sig * rng.normal(1.0, prof["impedance_sd"], (NC, 1)).clip(0.5, 1.6)

    # sporadic ocular/muscle artefacts ---------------------------------------------------------------
    n_art = 0
    if prof["artifact_rate"] > 0:
        n_art = rng.binomial(C.N_EPOCHS, prof["artifact_rate"])
        frontal = np.array([1, 1, .6, .5, .4, .5, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
        t = np.arange(N) / C.FS
        for _ in range(n_art):
            t0 = rng.uniform(0.3, C.DURATION - 0.3)
            pulse = rng.uniform(180, 320) * np.exp(-0.5 * ((t - t0) / 0.12) ** 2)
            sig = sig + frontal[:, None] * pulse[None, :]

    beh = sample_behaviour(rng, cls, prof, f, fatigue)
    return Subject(sig.astype(np.float32), beh, cls,
                   dict(severity=f, fatigue=float(fatigue), affected=w.tolist(), n_artifacts=int(n_art),
                        tg_depth=float(d_tg), env=env.tolist()))


def generate_cohort(n_per_class: int = 300, profile: str | dict = "sim", seed: int = C.DEFAULT_SEED,
                    **kw):
    """Cohort of ``3 * n_per_class`` subjects, class-balanced, each with its own seeded stream."""
    ss = np.random.SeedSequence(seed)
    kids = ss.spawn(3 * n_per_class)
    eeg = np.zeros((3 * n_per_class, C.N_CH, C.N_SAMPLES), np.float32)
    beh = np.zeros((3 * n_per_class, 3), np.float32)
    y = np.zeros(3 * n_per_class, np.int64)
    metas = []
    i = 0
    for cls in range(3):
        for _ in range(n_per_class):
            s = generate_subject(cls, np.random.default_rng(kids[i]), profile, **kw)
            eeg[i], beh[i], y[i] = s.eeg, s.beh, cls
            metas.append(s.meta)
            i += 1
    return eeg, beh, y, metas
