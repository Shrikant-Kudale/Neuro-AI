"""
MODULE 3 — Feature Fusion + Dataset Preparation
================================================
Combines PAC features (570) + behavioral scores (3) = 573-dim vector.
Applies min-max normalization (fit on train only — no data leakage).
Reshapes to 15 tokens × 38 features for Transformer input.
Splits into train/val/test (70/15/15) and saves as .npy files.

Output:
  outputs/features/X_train.npy, X_val.npy, X_test.npy   (N, 15, 38)
  outputs/features/y_train.npy, y_val.npy, y_test.npy   (N,)
  outputs/features/scaler_min.npy, scaler_scale.npy      for inference
"""

import numpy as np
import os

INPUT_DIR  = 'outputs/data'
FEAT_DIR   = 'outputs/features'
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

N_TOKENS   = 15   # temporal tokens
N_FEAT_TOK = 41   # features per token  (570/15 = 38 exactly)


def load_and_fuse():
    pac        = np.load(f'{FEAT_DIR}/pac_features.npy')   # (900, 570)
    behavioral = np.load(f'{INPUT_DIR}/behavioral.npy')    # (900, 3)
    labels     = np.load(f'{INPUT_DIR}/labels.npy')        # (900,)

    print(f"  PAC features:        {pac.shape}")
    print(f"  Behavioral features: {behavioral.shape}")

    # Distribute behavioral scores across tokens (repeat for each token)
    # Each token gets the same 3 behavioral values — model learns to use them
    beh_repeated = np.tile(behavioral, (1, N_TOKENS)).reshape(900, N_TOKENS, 3)

    # Reshape PAC: (900, 570) → (900, 15, 38)
    pac_reshaped = pac.reshape(900, N_TOKENS, 38)  # (900, 15, 38)

    # Concatenate along feature dimension: (900, 15, 35+3) = (900, 15, 38)
    X = np.concatenate([pac_reshaped, beh_repeated], axis=2).astype(np.float32)  # (900,15,41)

    print(f"  Fused feature tensor: {X.shape}  [subjects, tokens, features]")
    return X, labels


def train_val_test_split(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    n   = len(y)
    idx = rng.permutation(n)

    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])


def normalize(X_train, X_val, X_test):
    """Min-max normalize per feature. Fit only on training set."""
    # Flatten to (N, features) for computing stats
    shape        = X_train.shape     # (N, 15, 38)
    X_tr_flat    = X_train.reshape(shape[0], -1)
    X_va_flat    = X_val.reshape(X_val.shape[0], -1)
    X_te_flat    = X_test.reshape(X_test.shape[0], -1)

    feat_min     = X_tr_flat.min(axis=0, keepdims=True)
    feat_max     = X_tr_flat.max(axis=0, keepdims=True)
    feat_scale   = np.where((feat_max - feat_min) > 0,
                            feat_max - feat_min, 1.0)

    X_tr_norm    = ((X_tr_flat - feat_min) / feat_scale).reshape(shape)
    X_va_norm    = ((X_va_flat - feat_min) / feat_scale).reshape(X_val.shape)
    X_te_norm    = ((X_te_flat - feat_min) / feat_scale).reshape(X_test.shape)

    return X_tr_norm, X_va_norm, X_te_norm, feat_min, feat_scale


def prepare_dataset():
    os.makedirs(FEAT_DIR, exist_ok=True)

    X, y = load_and_fuse()

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)

    print(f"\n  Split sizes:")
    print(f"  Train: {X_train.shape}  labels={np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}   labels={np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}  labels={np.bincount(y_test)}")

    # Normalize
    X_train, X_val, X_test, feat_min, feat_scale = normalize(X_train, X_val, X_test)

    # Save all splits
    for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                      ('y_train', y_train), ('y_val',   y_val), ('y_test',  y_test)]:
        np.save(f'{FEAT_DIR}/{name}.npy', arr)

    # Save scaler params for future inference
    np.save(f'{FEAT_DIR}/scaler_min.npy',   feat_min)
    np.save(f'{FEAT_DIR}/scaler_scale.npy', feat_scale)

    print(f"\n  All splits saved to {FEAT_DIR}/")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    print("\n[MODULE 3] Feature Fusion + Dataset Preparation")
    print("=" * 50)
    prepare_dataset()
    print("\n✓ Module 3 complete.\n")
