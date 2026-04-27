"""
MODULE 6 — Evaluation
======================
Loads best saved models and evaluates on test set.

Computes:
  - Accuracy, Precision, Recall, F1 (macro)
  - AUC-ROC (per class and macro)
  - Per-class breakdown
  - Ablation study (PAC+Behav vs PAC Only vs Behav Only)
  - Confusion matrix values

Prints results tables matching the research paper.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader, TensorDataset
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from module4_model import TransformerClassifier, MLPBaseline, CNNBaseline

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR  = 'outputs/models'
CLASS_NAMES = ['Normal', 'Early Disruption', 'High Risk']


# ── Load data ──────────────────────────────────────────────────────────────
def load_test():
    data = torch.load('outputs/features/test.pt', weights_only=False)
    return data['X'], data['y']


# ── Load trained model ─────────────────────────────────────────────────────
def load_model(model_name):
    if model_name == 'transformer':
        model = TransformerClassifier(input_dim=38, seq_len=15, embed_dim=128,
                                      num_heads=8, num_layers=4, ffn_dim=256,
                                      dropout=0.1, num_classes=3)
    elif model_name == 'mlp':
        model = MLPBaseline(input_dim=570, num_classes=3)
    else:
        model = CNNBaseline(input_channels=38, seq_len=15, num_classes=3)

    path = os.path.join(MODELS_DIR, f'{model_name}_best.pt')
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────
def get_predictions(model, X):
    """Run inference, return (preds, probs)."""
    model.eval()
    all_probs = []
    all_preds = []

    loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(DEVICE)
            logits  = model(X_batch)
            probs   = torch.softmax(logits, dim=1)
            preds   = logits.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


# ── Metrics computation ────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC-ROC for High Risk class (one-vs-rest)
    try:
        auc_hr = roc_auc_score(
            (y_true == 2).astype(int),
            y_prob[:, 2]
        )
    except Exception:
        auc_hr = float('nan')

    return acc, prec, rec, f1, auc_hr


# ── Ablation: PAC-only and behavioral-only ─────────────────────────────────
def ablation_study(X_test_raw, y_test):
    """
    Retrain/re-evaluate Transformer on:
      1. PAC features only (no behavioral)
      2. Behavioral only (no PAC)
    Uses a quick 40-epoch train on train split for speed.
    """
    from module3_feature_fusion import fit_minmax, apply_minmax

    print("\n  Running ablation study (quick training)...")

    train_data = torch.load('outputs/features/train.pt', weights_only=False)
    X_tr, y_tr = train_data['X'], train_data['y']
    X_te = torch.load('outputs/features/test.pt', weights_only=False)['X']
    y_te = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

    results = {}

    for mode in ['pac_only', 'behav_only']:
        if mode == 'pac_only':
            # Zero out the behavioral contribution (first 3 tokens, position 0)
            X_tr_m = X_tr.clone()
            X_te_m = X_te.clone()
            X_tr_m[:, :3, 0] = 0.0
            X_te_m[:, :3, 0] = 0.0
        else:
            # Keep only behavioral (zero out PAC, keep behavioral tokens)
            X_tr_m = torch.zeros_like(X_tr)
            X_te_m = torch.zeros_like(X_te)
            X_tr_m[:, :3, 0] = X_tr[:, :3, 0]
            X_te_m[:, :3, 0] = X_te[:, :3, 0]

        # Quick train
        model = TransformerClassifier(input_dim=38, seq_len=15, embed_dim=128,
                                      num_heads=8, num_layers=4, ffn_dim=256,
                                      dropout=0.1, num_classes=3).to(DEVICE)
        opt  = torch.optim.Adam(model.parameters(), lr=3e-4)
        loss_fn = nn.CrossEntropyLoss()
        loader  = DataLoader(TensorDataset(X_tr_m, y_tr), batch_size=32, shuffle=True)

        model.train()
        for ep in range(40):
            for Xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(Xb.to(DEVICE)), yb.to(DEVICE)).backward()
                opt.step()

        preds, probs = get_predictions(model, X_te_m)
        acc  = accuracy_score(y_te, preds)
        f1   = f1_score(y_te, preds, average=None, zero_division=0)
        results[mode] = {'acc': acc, 'f1_ed': f1[1], 'f1_hr': f1[2]}
        print(f"    {mode:15s} → Acc: {acc:.1%} | Early Dis F1: {f1[1]:.2f} | High Risk F1: {f1[2]:.2f}")

    return results


# ── Main evaluation ────────────────────────────────────────────────────────
def evaluate_all():
    X_test, y_test = load_test()
    y_np = y_test.numpy()

    print("\n  ── TABLE I: Classification Performance Comparison ──")
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7} {'AUC-ROC':>8}")
    print("  " + "-" * 68)

    all_results   = {}
    transformer_preds = None
    transformer_probs = None

    for name in ['transformer', 'mlp', 'cnn']:
        try:
            model = load_model(name)
            preds, probs = get_predictions(model, X_test)
            acc, prec, rec, f1, auc = compute_metrics(y_np, preds, probs)
            all_results[name] = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'auc': auc, 'preds': preds, 'probs': probs}

            label = {'transformer': 'Transformer (Ours)', 'mlp': 'MLP Baseline', 'cnn': 'CNN Baseline'}[name]
            print(f"  {label:<22} {acc:>8.1%} {prec:>10.3f} {rec:>7.3f} {f1:>7.3f} {auc:>8.2f}")

            if name == 'transformer':
                transformer_preds = preds
                transformer_probs = probs
        except FileNotFoundError:
            print(f"  {name}: model file not found — skipping")

    # Per-class breakdown for Transformer
    if transformer_preds is not None:
        print("\n  ── TABLE II: Per-Class Performance (Transformer) ──")
        print(f"  {'Class':<22} {'Precision':>9} {'Recall':>7} {'F1':>7} {'Support':>8}")
        print("  " + "-" * 58)
        f1_per   = f1_score(y_np, transformer_preds, average=None, zero_division=0)
        prec_per = precision_score(y_np, transformer_preds, average=None, zero_division=0)
        rec_per  = recall_score(y_np, transformer_preds, average=None, zero_division=0)
        for i, cls in enumerate(CLASS_NAMES):
            sup = (y_np == i).sum()
            print(f"  {cls:<22} {prec_per[i]:>9.2f} {rec_per[i]:>7.2f} {f1_per[i]:>7.2f} {sup:>8}")

        # Confusion matrix
        cm = confusion_matrix(y_np, transformer_preds)
        print("\n  ── Confusion Matrix (Transformer) ──")
        print(f"  {'':>20} {'Pred:Normal':>13} {'Pred:EarlyDis':>14} {'Pred:HighRisk':>14}")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"  {'True:' + cls:<20} {cm[i][0]:>13} {cm[i][1]:>14} {cm[i][2]:>14}")

    # Ablation study
    print("\n  ── TABLE III: Ablation Study ──")
    X_test_raw = np.load('outputs/features/test_raw.npy')

    if transformer_preds is not None:
        full_f1 = f1_score(y_np, transformer_preds, average=None, zero_division=0)
        full_acc = accuracy_score(y_np, transformer_preds)
        print(f"  {'Configuration':<28} {'Accuracy':>9} {'Early Dis F1':>13} {'High Risk F1':>13}")
        print("  " + "-" * 65)
        print(f"  {'PAC + Behavioral (Full)':<28} {full_acc:>8.1%} {full_f1[1]:>13.2f} {full_f1[2]:>13.2f}")

        abl = ablation_study(X_test_raw, y_test)
        for mode, label in [('pac_only', 'PAC Only'), ('behav_only', 'Behavioral Only')]:
            if mode in abl:
                r = abl[mode]
                print(f"  {label:<28} {r['acc']:>8.1%} {r['f1_ed']:>13.2f} {r['f1_hr']:>13.2f}")

    # Save results summary
    summary = []
    for name, r in all_results.items():
        summary.append({'model': name, 'accuracy': r['acc'], 'precision': r['prec'],
                        'recall': r['rec'], 'f1': r['f1'], 'auc_roc': r['auc']})
    pd.DataFrame(summary).to_csv(os.path.join(MODELS_DIR, 'results_summary.csv'), index=False)

    # Save transformer predictions for visualization
    if transformer_preds is not None:
        np.save(os.path.join(MODELS_DIR, 'transformer_preds.npy'), transformer_preds)
        np.save(os.path.join(MODELS_DIR, 'transformer_probs.npy'), transformer_probs)
        np.save(os.path.join(MODELS_DIR, 'test_labels.npy'), y_np)

    return all_results


if __name__ == '__main__':
    print("=" * 55)
    print("MODULE 6 — Evaluation")
    print("=" * 55)
    evaluate_all()
    print("\nModule 6 complete.")
