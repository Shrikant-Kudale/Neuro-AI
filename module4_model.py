"""
MODULE 4 — Model Definitions (NumPy / Scikit-learn version)
============================================================
No PyTorch/TensorFlow required. Pure NumPy implementation of:

  1. TransformerClassifier  — self-attention encoder + MLP head
  2. MLPBaseline            — 3-layer fully connected network
  3. CNNBaseline            — 1D conv using sliding window approach

All models expose: fit(X, y), predict(X), predict_proba(X)
"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


# ── Utility: Softmax ────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── Utility: GELU activation ────────────────────────────────────────────────
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


# ── Utility: Layer Normalization ─────────────────────────────────────────────
def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1,  keepdims=True)
    return (x - mean) / (std + eps)


# ── Attention head ───────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V):
    """Q,K,V: (seq_len, d_k)"""
    d_k    = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)      # (seq, seq)
    attn   = softmax(scores)              # (seq, seq)
    out    = attn @ V                     # (seq, d_k)
    return out, attn


# ══════════════════════════════════════════════════════════════════════════════
# 1. Transformer Classifier
# ══════════════════════════════════════════════════════════════════════════════
class TransformerClassifier:
    """
    Lightweight Transformer encoder implemented in pure NumPy.
    Trains using sklearn MLPClassifier on the transformer-projected features.
    This gives us the correct architecture behaviour while being dependency-free.
    """

    def __init__(self, n_tokens=15, n_feats=41, embed_dim=64,
                 n_heads=4, n_classes=3, random_state=42):
        self.n_tokens     = n_tokens
        self.n_feats      = n_feats
        self.embed_dim    = embed_dim
        self.n_heads      = n_heads
        self.n_classes    = n_classes
        self.random_state = random_state
        self.d_k          = embed_dim // n_heads
        self._init_weights()

    def _init_weights(self):
        rng = np.random.default_rng(self.random_state)
        d, e = self.n_feats, self.embed_dim

        # Input projection weights
        self.W_proj = rng.standard_normal((d, e)) * 0.02

        # Positional encoding (sinusoidal)
        pos  = np.arange(self.n_tokens + 1)[:, None]
        dims = np.arange(0, e, 2)[None, :]
        pe   = np.zeros((self.n_tokens + 1, e))
        pe[:, 0::2] = np.sin(pos / 10000 ** (dims / e))
        pe[:, 1::2] = np.cos(pos / 10000 ** (dims / e))
        self.pos_enc = pe  # (T+1, embed_dim)

        # Multi-head attention weights per head
        self.W_Q = [rng.standard_normal((e, self.d_k)) * 0.02 for _ in range(self.n_heads)]
        self.W_K = [rng.standard_normal((e, self.d_k)) * 0.02 for _ in range(self.n_heads)]
        self.W_V = [rng.standard_normal((e, self.d_k)) * 0.02 for _ in range(self.n_heads)]
        self.W_O = rng.standard_normal((e, e)) * 0.02

        # Feed-forward weights
        self.W_ff1 = rng.standard_normal((e, 128)) * 0.02
        self.b_ff1 = np.zeros(128)
        self.W_ff2 = rng.standard_normal((128, e)) * 0.02
        self.b_ff2 = np.zeros(e)

        # CLS token
        self.cls_token = rng.standard_normal((1, e)) * 0.02

        # Store last attention weights for explainability
        self.last_attn = None

    def _encode_one(self, x):
        """
        Forward pass for one sample.
        x: (n_tokens, n_feats)
        returns: cls_output (embed_dim,)
        """
        # Project input
        h = x @ self.W_proj             # (T, embed_dim)

        # Prepend CLS token
        h = np.vstack([self.cls_token, h])   # (T+1, embed_dim)

        # Add positional encoding
        h = h + self.pos_enc            # (T+1, embed_dim)

        # Layer norm (pre-LN)
        h_norm = layer_norm(h)

        # Multi-head self-attention
        heads = []
        attn_maps = []
        for i in range(self.n_heads):
            Q = h_norm @ self.W_Q[i]   # (T+1, d_k)
            K = h_norm @ self.W_K[i]
            V = h_norm @ self.W_V[i]
            head_out, attn = scaled_dot_product_attention(Q, K, V)
            heads.append(head_out)
            attn_maps.append(attn)

        # Concatenate heads → project
        multi_head = np.concatenate(heads, axis=-1)  # (T+1, embed_dim)
        attn_out   = multi_head @ self.W_O           # (T+1, embed_dim)

        # Residual + norm
        h = layer_norm(h + attn_out)

        # Store average attention for explainability
        self.last_attn = np.mean(attn_maps, axis=0)

        # Feed-forward block
        ff = gelu(h @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        h  = layer_norm(h + ff)

        # Return CLS token embedding
        return h[0]    # (embed_dim,)

    def transform(self, X):
        """Encode all samples. X: (N, T, F) → (N, embed_dim)"""
        return np.array([self._encode_one(x) for x in X])

    def fit(self, X, y):
        print("    Encoding training samples through Transformer...")
        X_enc = self.transform(X)          # (N, embed_dim)
        print(f"    Encoded shape: {X_enc.shape}")

        self.clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            max_iter=300,
            random_state=self.random_state,
            early_stopping=False,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        self.clf.fit(X_enc, y)
        return self

    def predict(self, X):
        return self.clf.predict(self.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.transform(X))

    def get_attention_weights(self, X_sample):
        """Return attention matrix for a single sample."""
        self._encode_one(X_sample)
        return self.last_attn

    def count_params(self):
        total = (self.W_proj.size + self.pos_enc.size + self.cls_token.size +
                 self.W_O.size + self.W_ff1.size + self.b_ff1.size +
                 self.W_ff2.size + self.b_ff2.size)
        for i in range(self.n_heads):
            total += self.W_Q[i].size + self.W_K[i].size + self.W_V[i].size
        return total


# ══════════════════════════════════════════════════════════════════════════════
# 2. MLP Baseline
# ══════════════════════════════════════════════════════════════════════════════
class MLPBaseline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            max_iter=300,
            random_state=random_state,
            early_stopping=False,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )

    def _flatten(self, X):
        return X.reshape(X.shape[0], -1)

    def fit(self, X, y):
        self.clf.fit(self._flatten(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self._flatten(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._flatten(X))

    def count_params(self):
        return sum(w.size for w in self.clf.coefs_) + sum(b.size for b in self.clf.intercepts_)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CNN Baseline (sliding window conv via sklearn)
# ══════════════════════════════════════════════════════════════════════════════
class CNNBaseline:
    """
    Simulates 1D Conv using manual sliding window feature extraction,
    then feeds into an MLP classifier.
    """
    def __init__(self, kernel_size=3, n_filters=64, random_state=42):
        self.kernel_size  = kernel_size
        self.n_filters    = n_filters
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        # Learnable filters: (n_filters, kernel_size, n_feats)
        # Will be set after seeing n_feats in fit()
        self.rng = rng

    def _conv_extract(self, X):
        """
        X: (N, T, F)
        Applies sliding window of size kernel_size along T dimension.
        Returns max-pooled features: (N, n_filters)
        """
        N, T, F = X.shape
        k = self.kernel_size

        # Apply each filter across the time dimension
        out = np.zeros((N, self.n_filters))
        for fi in range(self.n_filters):
            filt = self.filters[fi]      # (k, F)
            responses = []
            for t in range(T - k + 1):
                window = X[:, t:t+k, :]  # (N, k, F)
                score  = (window * filt).sum(axis=(1, 2))  # (N,)
                responses.append(score)
            # Max pool across time
            out[:, fi] = np.stack(responses, axis=1).max(axis=1)

        return np.tanh(out)   # activation

    def fit(self, X, y):
        N, T, F = X.shape
        # Initialize filters now that we know F
        self.filters = self.rng.standard_normal((self.n_filters, self.kernel_size, F)) * 0.1

        X_conv = self._conv_extract(X)   # (N, n_filters)

        self.clf = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            max_iter=300,
            random_state=self.random_state,
            early_stopping=False,
            validation_fraction=0.15,
            n_iter_no_change=15,
            verbose=False,
        )
        self.clf.fit(X_conv, y)
        return self

    def predict(self, X):
        return self.clf.predict(self._conv_extract(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._conv_extract(X))

    def count_params(self):
        return (self.filters.size +
                sum(w.size for w in self.clf.coefs_) +
                sum(b.size for b in self.clf.intercepts_))


if __name__ == '__main__':
    print("\n[MODULE 4] Model Definitions")
    print("=" * 40)
    rng   = np.random.default_rng(0)
    dummy = rng.random((8, 15, 41)).astype(np.float32)
    y_d   = np.array([0,0,1,1,2,2,0,1])

    for name, model in [
        ('Transformer', TransformerClassifier()),
        ('MLP',         MLPBaseline()),
        ('CNN',         CNNBaseline()),
    ]:
        model.fit(dummy, y_d)
        preds = model.predict(dummy)
        print(f"  {name:12s} | preds shape: {preds.shape} | sample: {preds[:4]}")

    print("\n✓ Module 4 complete.\n")
