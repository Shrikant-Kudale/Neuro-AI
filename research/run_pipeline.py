"""End-to-end reproducible pipeline.

    python run_pipeline.py                 # both profiles, full benchmark  (~1 h on one CPU core)
    python run_pipeline.py --profiles hard --seeds 42
    python run_pipeline.py --quick         # tiny smoke run

Writes ../public/data/*.json and ../public/model/* consumed by the web application."""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from neuroai import config as C, data, features, models, train, evaluate, explain, export

ROOT = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(ROOT, "..", "public")
os.makedirs(os.path.join(PUB, "data"), exist_ok=True)
os.makedirs(os.path.join(PUB, "model"), exist_ok=True)


def flat(X):
    return np.concatenate([X[:, :, :C.PAC_DIM].reshape(len(X), -1), X[:, 0, C.PAC_DIM:]], 1)


def t0(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    return time.time()


def run_torch(make, Xtr, ytr, Xva, yva, seed, epochs, verbose=False):
    train.seed_all(seed)
    m = make()
    log = train.fit(m, Xtr, ytr, Xva, yva, epochs=epochs, seed=seed, verbose=verbose)
    return m, log


def sk_models(seed):
    return {"Logistic regression": LogisticRegression(max_iter=3000, C=1.0),
            "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
            "Gradient boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=seed)}


def profile_run(profile: str, seeds, epochs, n_per_class, noise_levels, n_rob):
    T = t0(f"=== profile: {profile} ===")
    D = data.build_cohort(profile, n_per_class)
    X, y = D["X"], D["y"]
    res = dict(profile=profile, description=C.PROFILES[profile]["description"], n_subjects=int(len(y)),
               flagged_epochs_mean=float(D["flagged"].mean()), models={}, ablation={}, robustness={})

    # ---- class-level descriptive statistics (what the generator actually produced) -----------------
    stats = {}
    for c in range(3):
        m = y == c
        pac = D["pac"][m]
        stats[C.CLASS_NAMES[c]] = dict(
            tg_mean=float(pac[:, :285].mean()), ab_mean=float(pac[:, 285:].mean()),
            tg_by_channel=pac[:, :285].reshape(-1, C.N_CH, C.N_EPOCHS).mean((0, 2)).round(5).tolist(),
            ab_by_channel=pac[:, 285:].reshape(-1, C.N_CH, C.N_EPOCHS).mean((0, 2)).round(5).tolist(),
            tg_by_epoch=pac[:, :285].reshape(-1, C.N_CH, C.N_EPOCHS).mean((0, 1)).round(5).tolist(),
            beh_mean=D["beh"][m].mean(0).round(4).tolist(), beh_sd=D["beh"][m].std(0).round(4).tolist(),
            beh_hist={C.BEH_NAMES[j]: np.histogram(D["beh"][:, j], bins=24, range=(D["beh"][:, j].min(), D["beh"][:, j].max()))[0].tolist() for j in range(3)} if c == 0 else None)
    res["class_stats"] = stats
    res["beh_hist_edges"] = {C.BEH_NAMES[j]: [float(D["beh"][:, j].min()), float(D["beh"][:, j].max())] for j in range(3)}
    res["beh_hist"] = {C.CLASS_NAMES[c]: {C.BEH_NAMES[j]: np.histogram(D["beh"][y == c, j], bins=24, range=tuple(res["beh_hist_edges"][C.BEH_NAMES[j]]))[0].tolist() for j in range(3)} for c in range(3)}

    # ---- main benchmark: several independent splits / initialisations -------------------------------
    per_seed = {}
    headline = {}
    for seed in seeds:
        tr, va, te = data.split_indices(y, seed)
        sc = features.MinMax().fit(X[tr])
        Xtr, Xva, Xte = (sc.transform(X[i]) for i in (tr, va, te))
        ytr, yva, yte = y[tr], y[va], y[te]
        if seed == seeds[0]:
            res["split"] = dict(train=len(tr), val=len(va), test=len(te), seed=seed)
        zoo = {"Transformer (ours)": lambda: models.NeuroTransformer(), "CNN baseline": lambda: models.CNNBaseline(),
               "MLP baseline": lambda: models.MLPBaseline()}
        for name, make in zoo.items():
            s = t0(f"  seed {seed} · training {name}")
            m, log = run_torch(make, Xtr, ytr, Xva, yva, seed, epochs, verbose=(name.startswith("Trans") and seed == seeds[0]))
            proba = train.predict_proba(m, Xte)
            rep = evaluate.full_report(yte, proba, with_curves=(seed == seeds[0]), with_ci=(seed == seeds[0]))
            rep.update(params=models.n_params(m), epochs_run=len(log), train_seconds=round(time.time() - s, 1))
            per_seed.setdefault(name, []).append(rep)
            print(f"      acc {rep['accuracy']:.3f}  macroF1 {rep['f1']:.3f}  AUC {rep['auc_macro']:.3f}  ({len(log)} ep, {time.time()-s:.0f}s)", flush=True)
            if seed == seeds[0]:
                headline[name] = dict(model=m, log=log, proba=proba)
        for name, clf in sk_models(seed).items():
            s = t0(f"  seed {seed} · fitting {name}")
            clf.fit(flat(Xtr), ytr)
            proba = clf.predict_proba(flat(Xte))
            rep = evaluate.full_report(yte, proba, with_curves=(seed == seeds[0]), with_ci=(seed == seeds[0]))
            rep.update(params=None, epochs_run=None, train_seconds=round(time.time() - s, 1))
            per_seed.setdefault(name, []).append(rep)
            print(f"      acc {rep['accuracy']:.3f}  macroF1 {rep['f1']:.3f}  AUC {rep['auc_macro']:.3f}", flush=True)

    for name, reps in per_seed.items():
        agg = {k: [r[k] for r in reps] for k in ("accuracy", "precision", "recall", "f1", "auc_macro", "ece")}
        res["models"][name] = dict(
            headline=reps[0],
            across_seeds={k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), values=[float(x) for x in v]) for k, v in agg.items()},
            params=reps[0].get("params"))

    # ---- keep the primary split for ablation / robustness / explainability ----------------------------
    seed = seeds[0]
    tr, va, te = data.split_indices(y, seed)
    sc = features.MinMax().fit(X[tr])
    Xtr, Xva, Xte = (sc.transform(X[i]) for i in (tr, va, te))
    ytr, yva, yte = y[tr], y[va], y[te]
    tf = headline["Transformer (ours)"]["model"]
    res["training_log"] = {n.split(" ")[0].lower(): headline[n]["log"] for n in headline}

    # ---- ablations (retrained Transformer on feature subsets) ---------------------------------------
    subsets = {"PAC + behavioural (full)": list(range(41)), "PAC only": list(range(38)),
               "Behavioural only": [38, 39, 40], "Theta–gamma PAC only": list(range(19)),
               "Alpha–beta PAC only": list(range(19, 38))}
    for name, cols in subsets.items():
        if name.startswith("PAC + "):
            proba = headline["Transformer (ours)"]["proba"]
        else:
            s = t0(f"  ablation · {name}")
            m, _ = run_torch(lambda: models.NeuroTransformer(in_dim=len(cols)), Xtr[:, :, cols], ytr, Xva[:, :, cols], yva, seed, epochs)
            proba = train.predict_proba(m, Xte[:, :, cols])
        r = evaluate.full_report(yte, proba, with_curves=False, with_ci=False)
        res["ablation"][name] = dict(accuracy=r["accuracy"], f1=r["f1"], auc_macro=r["auc_macro"],
                                    f1_early=r["per_class"]["Early Disruption"]["f1"], f1_high=r["per_class"]["High Risk"]["f1"],
                                    n_features=len(cols))
        print(f"      {name:28s} acc {r['accuracy']:.3f}", flush=True)

    # ---- robustness / generalisation (evaluation only, same trained model) ------------------------------
    def eval_cohort(tag, **kw):
        Dr = data.build_cohort(profile, n_rob, seed=1000 + len(tag), cache=False, verbose=False, tag="_" + tag, **kw)
        Xr = sc.transform(Dr["X"]); p = train.predict_proba(tf, Xr)
        r = evaluate.full_report(Dr["y"], p, with_curves=False, with_ci=False)
        return dict(accuracy=r["accuracy"], f1=r["f1"], auc_macro=r["auc_macro"], f1_early=r["per_class"]["Early Disruption"]["f1"])
    res["robustness"]["noise"] = []
    for nm in noise_levels:
        s = t0(f"  robustness · sensor-noise ×{nm}")
        r = eval_cohort(f"noise{nm}", noise_mult=nm); r["noise_mult"] = nm
        res["robustness"]["noise"].append(r); print(f"      acc {r['accuracy']:.3f}", flush=True)
    t0("  robustness · focal-lesion geometry (unseen at training)")
    res["robustness"]["focal_lesion"] = eval_cohort("focal", spatial="focal")
    rng = np.random.default_rng(0)
    Xsh = np.stack([x[rng.permutation(C.N_TOKENS)] for x in Xte])
    r = evaluate.full_report(yte, train.predict_proba(tf, Xsh), with_curves=False, with_ci=False)
    res["robustness"]["epoch_shuffled_test"] = dict(accuracy=r["accuracy"], f1=r["f1"], auc_macro=r["auc_macro"])
    res["robustness"]["baseline_accuracy"] = res["models"]["Transformer (ours)"]["headline"]["accuracy"]

    # ---- explainability from the trained model ------------------------------------------------------------
    t0("  explainability")
    pred = headline["Transformer (ours)"]["proba"].argmax(1)
    res["explain"] = dict(attention=explain.attention_summary(tf, Xte, yte, pred),
                          occlusion=explain.occlusion_importance(tf, Xte, yte, sc.transform(X[tr]).reshape(-1, 41).mean(0)))

    # ---- cohort export (PCA map, predictions, raw features for the web "Cohort" page) ------------------------
    full_scaled = sc.transform(X)
    pca = PCA(n_components=2, random_state=0).fit(flat(full_scaled[tr]))
    emb = pca.transform(flat(full_scaled))
    all_p = train.predict_proba(tf, full_scaled)
    split_tag = np.zeros(len(y), int); split_tag[va] = 1; split_tag[te] = 2
    cohort = [dict(id=int(i), y=int(y[i]), split=int(split_tag[i]), beh=D["beh"][i].round(4).tolist(),
                   pc=emb[i].round(4).tolist(), p=all_p[i].round(4).tolist(),
                   tg=D["pac"][i, :285].reshape(C.N_CH, C.N_EPOCHS).mean(1).round(5).tolist(),
                   ab=D["pac"][i, 285:].reshape(C.N_CH, C.N_EPOCHS).mean(1).round(5).tolist(),
                   sev=round(float(D["severity"][i]), 3)) for i in range(len(y))]
    json.dump(dict(profile=profile, pca_var=pca.explained_variance_ratio_.round(4).tolist(), subjects=cohort),
              open(os.path.join(PUB, "data", f"cohort_{profile}.json"), "w"), separators=(",", ":"))
    np.concatenate([D["pac"], D["beh"]], 1).astype("<f4").tofile(os.path.join(PUB, "data", f"cohort_{profile}.f32"))

    # ---- model export ---------------------------------------------------------------------------------------
    meta = dict(profile=profile, params=models.n_params(tf), test_accuracy=res["models"]["Transformer (ours)"]["headline"]["accuracy"],
                class_names=C.CLASS_NAMES, seed=seed,
                fill=(sc.transform(X[tr]).reshape(-1, 41).mean(0) * sc.scale + sc.lo).round(8).tolist())
    export.export_transformer(tf, sc, os.path.join(PUB, "model", f"transformer_{profile}"), meta)
    torch.save(tf.state_dict(), os.path.join(ROOT, "artifacts", f"transformer_{profile}.pt"))
    res["runtime_minutes"] = round((time.time() - T) / 60, 1)
    json.dump(res, open(os.path.join(PUB, "data", f"results_{profile}.json"), "w"), separators=(",", ":"))
    t0(f"=== {profile} done in {res['runtime_minutes']} min ===")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", default=["paper", "hard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.n_per_class = [42], 6, 40
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    out = {}
    for p in a.profiles:
        out[p] = profile_run(p, a.seeds, a.epochs, a.n_per_class, [0.5, 1, 2, 4, 8], 45 if not a.quick else 10)
    json.dump(dict(dsp=export.dsp_manifest(), generated=time.strftime("%Y-%m-%d %H:%M"),
                   profiles=sorted(f[8:-5] for f in os.listdir(os.path.join(PUB, "data")) if f.startswith("results_")),
                   seeds=a.seeds), open(os.path.join(PUB, "data", "manifest.json"), "w"))
    print("done")
