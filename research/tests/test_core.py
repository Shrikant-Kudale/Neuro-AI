import numpy as np
from neuroai import config as C, features, generator, models, data
from neuroai.dsp import design_sos, filtfilt_zero_phase


def test_mi_recovers_imposed_coupling():
    fs, n = C.FS, C.EPOCH_LEN
    t = np.arange(n) / fs
    ph = 2 * np.pi * 6 * t
    mis = []
    for depth in (0.0, 0.5, 0.9):
        amp = 1 + depth * np.cos(ph)
        mis.append(float(features.modulation_index(np.angle(np.exp(1j * ph)), amp)))
    assert mis[0] < 1e-6 < mis[1] < mis[2]


def test_zero_phase_filter_keeps_inband_sine_without_delay():
    t = np.arange(C.N_SAMPLES) / C.FS
    x = np.sin(2 * np.pi * 6 * t)
    y = filtfilt_zero_phase(design_sos(4, 8), x)
    assert np.corrcoef(x[1000:-1000], y[1000:-1000])[0, 1] > 0.999


def test_generator_orders_coupling_by_class():
    m = []
    for cls in range(3):
        v = [features.pac_vector(generator.generate_subject(cls, np.random.default_rng(s), "sim").eeg)[0][:285].mean() for s in range(4)]
        m.append(np.mean(v))
    assert m[0] > m[2]


def test_split_is_stratified_and_disjoint():
    y = np.repeat([0, 1, 2], 300)
    tr, va, te = data.split_indices(y, 1)
    assert len(set(tr) & set(te)) == 0 and len(set(va) & set(te)) == 0
    assert np.bincount(y[te]).tolist() == [45, 45, 45]


def test_transformer_shapes():
    import torch
    m = models.NeuroTransformer()
    out, att = m(torch.zeros(2, 15, 41), return_attn=True)
    assert out.shape == (2, 3) and att[-1].shape == (2, 8, 16, 16)
    assert 500_000 < models.n_params(m) < 600_000
