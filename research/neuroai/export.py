"""Serialise a trained NeuroTransformer for the browser: one flat float32 file + a JSON manifest."""
from __future__ import annotations
import json, os
import numpy as np
import torch
from . import config as C
from .dsp import design_sos


def export_transformer(model, scaler, path_prefix: str, meta: dict):
    tensors, entries, offset = [], [], 0
    for name, p in model.state_dict().items():
        a = p.detach().cpu().numpy().astype("<f4")
        entries.append(dict(name=name, shape=list(a.shape), offset=offset))
        tensors.append(a.ravel()); offset += a.size
    np.concatenate(tensors).tofile(path_prefix + ".bin")
    manifest = dict(
        arch=dict(in_dim=model.in_dim, seq=model.seq, d=model.d, heads=model.blocks[0].h,
                  layers=len(model.blocks), ffn=model.blocks[0].f1.out_features, n_classes=3),
        tensors=entries, n_floats=int(offset),
        scaler=dict(lo=scaler.lo.round(8).tolist(), scale=scaler.scale.round(8).tolist()),
        **meta)
    with open(path_prefix + ".json", "w") as f:
        json.dump(manifest, f)
    return manifest


def dsp_manifest():
    """Everything the TypeScript DSP needs to reproduce the Python features bit-for-bit (to float tolerance)."""
    return dict(fs=C.FS, duration=C.DURATION, n_samples=C.N_SAMPLES, nfft_synth=C.NFFT_SYNTH,
                epoch_len=C.EPOCH_LEN, n_epochs=C.N_EPOCHS, n_bins=C.N_BINS, artifact_uv=C.ARTIFACT_UV,
                pad=512, channels=C.CHANNELS, coords=C.COORDS.round(3).tolist(), bands=C.BANDS,
                class_names=C.CLASS_NAMES, beh_names=C.BEH_NAMES, beh_long=C.BEH_LONG,
                sos={b: design_sos(*C.BANDS[b]).round(14).tolist() for b in ("theta", "gamma", "alpha", "beta")},
                component_amp=C.COMPONENT_AMP, rms_uv=list(C.RMS_UV), coupling_gain=C.COUPLING_GAIN,
                max_depth=C.MAX_DEPTH, profiles={k: {kk: vv for kk, vv in v.items()} for k, v in C.PROFILES.items()})
