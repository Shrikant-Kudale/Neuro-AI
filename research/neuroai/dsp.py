"""Signal-processing primitives. Every function here has a line-for-line TypeScript twin in
``src/lib/dsp.ts``; ``tests/test_parity_fixture.py`` writes the fixtures that prove they agree."""
from __future__ import annotations
import numpy as np
from scipy import signal
from . import config as C

PAD = 512  # odd-extension padding (samples) used by the zero-phase filter


def design_sos(lo: float, hi: float, fs: int = C.FS, order: int = C.BUTTER_ORDER) -> np.ndarray:
    """4th-order Butterworth band-pass as second-order sections, shape (order, 6)."""
    return signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")


def filtfilt_zero_phase(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward-backward SOS filter with odd-extension padding (zero phase, double the order).
    ``x``: (..., n). Zero initial conditions, as in the TypeScript port."""
    n = x.shape[-1]
    left = 2 * x[..., :1] - x[..., PAD:0:-1]
    right = 2 * x[..., -1:] - x[..., -2:-PAD - 2:-1]
    ext = np.concatenate([left, x, right], axis=-1)
    y = signal.sosfilt(sos, ext, axis=-1)
    y = signal.sosfilt(sos, y[..., ::-1], axis=-1)[..., ::-1]
    return y[..., PAD:PAD + n]


def analytic(x: np.ndarray) -> np.ndarray:
    """Analytic signal along the last axis through the FFT (identical to scipy.signal.hilbert)."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(X * h, axis=-1)


def analytic_padded(x: np.ndarray, nfft: int = C.NFFT_SYNTH) -> np.ndarray:
    """Analytic signal of a zero-padded copy (power-of-two FFT length, shared with the browser port), cropped back."""
    n = x.shape[-1]
    pad = np.zeros(x.shape[:-1] + (nfft,), dtype=np.float64)
    pad[..., :n] = x
    return analytic(pad)[..., :n]


def band_noise(rng: np.random.Generator, lo: float, hi: float, shape: tuple[int, ...],
               nfft: int = C.NFFT_SYNTH, fs: int = C.FS, slope: float = 0.0,
               n_out: int = C.N_SAMPLES) -> np.ndarray:
    """Band-limited Gaussian-phase noise with spectral amplitude ∝ f^(-slope), unit RMS over the crop.
    Spectral synthesis keeps frequency/phase wandering realistic (unlike a pure sinusoid)."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    amp = np.zeros_like(freqs)
    m = (freqs >= lo) & (freqs <= hi)
    amp[m] = np.maximum(freqs[m], 0.5) ** (-slope)
    phase = rng.uniform(0, 2 * np.pi, size=shape + (freqs.size,))
    spec = amp * np.exp(1j * phase)
    x = np.fft.irfft(spec, n=nfft, axis=-1)[..., :n_out]
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)
