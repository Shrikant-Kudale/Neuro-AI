"""Writes the cross-language parity fixtures consumed by tests/parity.test.ts.

    python make_fixtures.py        (after run_pipeline.py has exported the models)
"""
import json, os
import numpy as np, torch
from neuroai import config as C, generator, features, models, train

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "..", "tests", "fixtures")
os.makedirs(OUT, exist_ok=True)

s = generator.generate_subject(2, np.random.default_rng(123), "hard")
s.eeg.astype("<f4").tofile(os.path.join(OUT, "signal.f32"))
tg, ab, flagged = features.pac_matrix(s.eeg.astype(np.float32).astype(np.float64))
vec = np.concatenate([tg.ravel(), ab.ravel()])
raw = features.to_tokens(vec, s.beh)

fx = dict(tg=tg.tolist(), ab=ab.tolist(), flagged=flagged.tolist(), beh=s.beh.tolist(), raw=raw.ravel().tolist(), models={})
for prof in ("paper", "hard"):
    mf = json.load(open(os.path.join(ROOT, "..", "public", "model", f"transformer_{prof}.json")))
    m = models.NeuroTransformer(); m.load_state_dict(torch.load(os.path.join(ROOT, "artifacts", f"transformer_{prof}.pt"))); m.eval()
    lo, sc = np.array(mf["scaler"]["lo"]), np.array(mf["scaler"]["scale"])
    x = ((raw - lo) / sc).astype(np.float32)
    with torch.no_grad():
        logits, atts = m(torch.from_numpy(x[None]), return_attn=True)
    fx["models"][prof] = dict(logits=logits[0].tolist(), probs=torch.softmax(logits, -1)[0].tolist(),
                              cls_attn=atts[-1][0, :, 0, 1:].mean(0).tolist())
json.dump(fx, open(os.path.join(OUT, "parity.json"), "w"))
print("fixtures written", {k: np.round(v["probs"], 4).tolist() for k, v in fx["models"].items()})
