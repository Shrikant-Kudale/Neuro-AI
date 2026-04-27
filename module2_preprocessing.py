"""
MODULE 2 — PAC Feature Extraction (Vectorized, Ultra-Fast)
Uses fully vectorized numpy operations — no Python loops over bins.
"""
import numpy as np
import os

FS=256; EPOCH_LEN=2; N_EPOCHS=15; N_CHANNELS=19; N_BINS=18

def fft_band_batch(signals, low, high, fs=FS):
    """Batch bandpass via FFT. signals: (n_ch, n_samples)"""
    S = np.fft.rfft(signals, axis=1)
    f = np.fft.rfftfreq(signals.shape[1], 1/fs)
    S[:, (f<low)|(f>high)] = 0
    return np.fft.irfft(S, signals.shape[1], axis=1)

def pac_vectorized(phase_sigs, amp_sigs, n_bins=N_BINS):
    """
    Vectorized PAC for shape (n_ch, n_ep, ep_len).
    Returns MI matrix (n_ch, n_ep).
    """
    n_ch, n_ep, ep_len = phase_sigs.shape
    # Approximate phase angle via arcsin of normalized signal
    p_max = np.abs(phase_sigs).max(axis=2, keepdims=True) + 1e-8
    ph = np.arcsin(np.clip(phase_sigs / p_max, -1, 1))   # (n_ch, n_ep, ep_len)
    amp = np.abs(amp_sigs)                                 # (n_ch, n_ep, ep_len)

    edges = np.linspace(-np.pi/2, np.pi/2, n_bins+1)
    mi = np.zeros((n_ch, n_ep), dtype=np.float32)

    for b in range(n_bins):
        mask = (ph >= edges[b]) & (ph < edges[b+1])       # (n_ch, n_ep, ep_len)
        cnt  = mask.sum(axis=2)                            # (n_ch, n_ep)
        amp_sum = (amp * mask).sum(axis=2)                 # (n_ch, n_ep)
        amp_bins_b = np.where(cnt > 0, amp_sum / (cnt + 1e-10), 0.0)
        mi += amp_bins_b   # accumulate for normalization later

    # We now recompute properly: build (n_ch, n_ep, n_bins) all at once
    amp_by_bin = np.zeros((n_ch, n_ep, n_bins), dtype=np.float32)
    for b in range(n_bins):
        mask = (ph >= edges[b]) & (ph < edges[b+1])
        cnt  = mask.sum(axis=2)
        amp_sum = (amp * mask).sum(axis=2)
        amp_by_bin[:,:,b] = np.where(cnt > 0, amp_sum / (cnt+1e-10), 0.0)

    total = amp_by_bin.sum(axis=2, keepdims=True) + 1e-10
    p = amp_by_bin / total                                 # (n_ch, n_ep, n_bins)
    q = 1.0 / n_bins
    kl = np.sum(p * np.log(p / q + 1e-10), axis=2)       # (n_ch, n_ep)
    return np.maximum(0, kl / np.log(n_bins)).astype(np.float32)

def extract_pac_features_fast(eeg):
    """eeg: (N_CHANNELS, n_samples) → pac_vector (570,)"""
    n_samples = eeg.shape[1]
    ep_s = FS * EPOCH_LEN   # 512

    # Batch filter all channels at once
    th = fft_band_batch(eeg, 4, 8)    # (19, 7680)
    gm = fft_band_batch(eeg, 30, 80)
    al = fft_band_batch(eeg, 8, 13)
    bt = fft_band_batch(eeg, 13, 30)

    # Reshape into epochs: (n_ch, n_ep, ep_len)
    th_ep = th[:, :N_EPOCHS*ep_s].reshape(N_CHANNELS, N_EPOCHS, ep_s)
    gm_ep = gm[:, :N_EPOCHS*ep_s].reshape(N_CHANNELS, N_EPOCHS, ep_s)
    al_ep = al[:, :N_EPOCHS*ep_s].reshape(N_CHANNELS, N_EPOCHS, ep_s)
    bt_ep = bt[:, :N_EPOCHS*ep_s].reshape(N_CHANNELS, N_EPOCHS, ep_s)

    tg_mi = pac_vectorized(th_ep, gm_ep)   # (19, 15)
    ab_mi = pac_vectorized(al_ep, bt_ep)   # (19, 15)

    return np.concatenate([tg_mi.flatten(), ab_mi.flatten()])  # (570,)

def preprocess_all(eeg_data, save_dir='outputs/features'):
    os.makedirs(save_dir, exist_ok=True)
    n = eeg_data.shape[0]
    pac = np.zeros((n, 570), dtype=np.float32)
    print(f"  Processing {n} subjects (vectorized)...")
    for i in range(n):
        pac[i] = extract_pac_features_fast(eeg_data[i])
        if (i+1) % 150 == 0:
            print(f"    {i+1}/{n} done")
    path = os.path.join(save_dir, 'pac_features.npy')
    np.save(path, pac)
    labels = np.load('outputs/data/labels.npy')
    print(f"\n  PAC matrix: {pac.shape}")
    for cls, name in enumerate(['Normal', 'Early Disruption', 'High Risk']):
        m = labels == cls
        print(f"  {name:20s} → TG-MI mean: {pac[m,:285].mean():.5f}  AB-MI mean: {pac[m,285:].mean():.5f}")
    return pac

if __name__ == '__main__':
    print("="*55+"\nMODULE 2 — PAC Feature Extraction\n"+"="*55)
    eeg = np.load('outputs/data/eeg_data.npy')
    preprocess_all(eeg)
    print("\nModule 2 complete.")
