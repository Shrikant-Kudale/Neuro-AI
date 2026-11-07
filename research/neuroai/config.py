"""Single source of truth for every constant shared by the generator, the feature pipeline,
the models and the browser application (exported to ``manifest.json``)."""
from __future__ import annotations
import numpy as np

# ── Recording geometry ───────────────────────────────────────────────────────
FS = 256                      # Hz
DURATION = 30                 # s
N_SAMPLES = FS * DURATION     # 7680
NFFT_SYNTH = 8192             # power-of-two synthesis length (cropped to N_SAMPLES)
EPOCH_SEC = 2
EPOCH_LEN = FS * EPOCH_SEC    # 512
N_EPOCHS = DURATION // EPOCH_SEC   # 15
N_BINS = 18                   # Tort et al. phase bins (20° each)
ARTIFACT_UV = 150.0           # epoch rejection threshold (µV)

CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
N_CH = len(CHANNELS)
# 2-D scalp coordinates (nose up, head radius = 1) used for lesion geometry and topomaps
COORDS = np.array([
    [-0.28, 0.86], [0.28, 0.86], [-0.74, 0.50], [-0.40, 0.50], [0.00, 0.45], [0.40, 0.50],
    [0.74, 0.50], [-0.88, 0.00], [-0.45, 0.00], [0.00, 0.00], [0.45, 0.00], [0.88, 0.00],
    [-0.74, -0.50], [-0.40, -0.50], [0.00, -0.45], [0.40, -0.50], [0.74, -0.50],
    [-0.28, -0.86], [0.28, -0.86]])

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 80.0)}
PAC_PAIRS = {"tg": ("theta", "gamma"), "ab": ("alpha", "beta")}
BUTTER_ORDER = 4

CLASS_NAMES = ["Normal", "Early Disruption", "High Risk"]
N_CLASSES = 3
BEH_NAMES = ["ETVS", "GAI", "SSCS"]
BEH_LONG = ["Eye-Tracking Variance Score", "Gait Asymmetry Index", "Speech Syntax Complexity Score"]

# ── Token layout for the sequence model ──────────────────────────────────────
N_TOKENS = N_EPOCHS            # 15 temporal tokens
PAC_DIM = 2 * N_CH             # 38  (19 theta-gamma + 19 alpha-beta)
TOKEN_DIM = PAC_DIM + 3        # 41  (+ ETVS, GAI, SSCS broadcast to every token)

# ── Simulation profile ───────────────────────────────────────────────────────
# A single, documented simulation ("sim"). Class effects follow the literature direction (theta–gamma coupling
# falls, alpha–beta rises, gamma power rises) and add the nuisance structure real cohorts have: a continuous
# latent severity (classes overlap), disruption that varies across epochs, noisy behavioural scores that share a
# fatigue/age confounder, electrode-gain variation, ocular artefacts and sensor noise.
PROFILES = {
    "sim": dict(
        description="Variable-severity simulation: overlapping classes, time-varying disruption, confounded behaviour, noise.",
        # per-class behavioural reference values (mean, sd)
        beh={0: dict(etvs=(1.20, 0.18), gai=(0.04, 0.010), sscs=(0.22, 0.04)),
             1: dict(etvs=(1.65, 0.22), gai=(0.09, 0.020), sscs=(0.34, 0.05)),
             2: dict(etvs=(2.30, 0.31), gai=(0.17, 0.030), sscs=(0.47, 0.07))},
        tg_depth=(0.18, 0.04),            # healthy theta-gamma coupling parameter (mean, sd)
        ab_depth=(0.12, 0.03),            # healthy alpha-beta coupling parameter
        n_affected={0: 0, 1: 5, 2: 10},   # channels involved (scaled by severity)
        tg_scale={0: (1.0, 1.0), 1: (0.70, 0.70), 2: (0.30, 0.45)},   # x coupling in affected channels
        ab_scale={0: (1.0, 1.0), 1: (1.15, 1.15), 2: (1.35, 1.40)},
        gamma_boost={0: 1.0, 1: 1.0, 2: 1.30},
        latent_sd=0.35,        # subject-level severity jitter (classes overlap)
        temporal=True,         # disruption envelope varies across the 15 epochs
        beh_sd_mult=2.2,       # behavioural measures are noisy and non-specific
        fatigue_sd=0.9,        # shared confounder (sleep debt / age) moving all three scores together
        impedance_sd=0.15, artifact_rate=0.04, white_sd=0.12, spatial="random",
    ),
}

# Relative amplitudes of the oscillatory components (resting-state-like; Niedermeyer & Lopes da Silva)
COMPONENT_AMP = dict(delta=(1.30, 0.20), theta=(0.90, 0.15), alpha=(1.00, 0.20),
                     beta=(0.50, 0.10), gamma=(0.30, 0.06), pink=0.60)
COUPLING_GAIN = 3.5            # maps the coupling parameter (0.18) to generative modulation depth, keeping measured MI in a physiological range
MAX_DEPTH = 0.95
RMS_UV = (14.0, 28.0)          # overall scalp RMS (µV)
DEFAULT_SEED = 42
