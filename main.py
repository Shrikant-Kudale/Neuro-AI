"""
main.py — Neuro-Cognitive AI Fingerprinting Pipeline
=====================================================
Run everything with a single command:

    python main.py

Executes all 7 modules in sequence:
  1. Generate synthetic EEG data (900 subjects)
  2. Extract PAC features from EEG signals
  3. Fuse features and prepare PyTorch datasets
  4. [Model definitions — imported automatically]
  5. Train Transformer + MLP + CNN baselines
  6. Evaluate all models and print results tables
  7. Generate all 7 publication-quality figures

Total runtime: ~3–5 minutes on CPU
"""

import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Neuro-Cognitive AI Fingerprinting                      ║
║   Pre-Radiographic Brain Tumor Detection                 ║
║   MIT School of Computing — PBL Project                  ║
║   Shrikant Kudale | Rehan Surani | Seema Saud            ║
╚══════════════════════════════════════════════════════════╝
"""

def step(num, name):
    print(f"\n{'='*58}")
    print(f"  STEP {num}/7 — {name}")
    print(f"{'='*58}")


def run():
    print(BANNER)
    total_start = time.time()

    # ── Step 1: Generate Data ──────────────────────────────────────────────
    step(1, "Synthetic EEG Data Generation")
    t = time.time()
    from module1_data_generator import generate_dataset
    eeg_data, labels, behavioral_df = generate_dataset()
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Step 2: PAC Feature Extraction ────────────────────────────────────
    step(2, "EEG Preprocessing + PAC Feature Extraction")
    t = time.time()
    import numpy as np
    from module2_preprocessing import preprocess_all
    eeg_data = np.load('outputs/data/eeg_data.npy')
    pac_matrix = preprocess_all(eeg_data)
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Step 3: Feature Fusion + Dataset Preparation ───────────────────────
    step(3, "Feature Fusion + Dataset Preparation")
    t = time.time()
    from module3_feature_fusion import prepare_dataset
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Step 4: Architecture Check (auto-imported during training) ─────────
    step(4, "Model Architecture Verification")
    t = time.time()
    import numpy as np
    from module4_model import TransformerClassifier, MLPBaseline, CNNBaseline
    transformer = TransformerClassifier()
    mlp         = MLPBaseline()
    cnn         = CNNBaseline()
    dummy       = np.random.randn(4, 15, 41).astype(np.float32)
    dummy_y     = np.array([0, 1, 2, 0])
    transformer.fit(dummy, dummy_y)
    mlp.fit(dummy, dummy_y)
    cnn.fit(dummy, dummy_y)
    assert transformer.predict(dummy).shape == (4,), "Transformer output shape error"
    assert mlp.predict(dummy).shape         == (4,), "MLP output shape error"
    assert cnn.predict(dummy).shape         == (4,), "CNN output shape error"
    print(f"  Transformer parameters : {transformer.count_params():,}")
    print(f"  MLP parameters         : {mlp.count_params():,}")
    print(f"  CNN parameters         : {cnn.count_params():,}")
    print(f"  ✓ All models verified in {time.time()-t:.1f}s")

    # ── Step 5: Train All Models ───────────────────────────────────────────
    step(5, "Training — Transformer + MLP + CNN")
    t = time.time()
    from module5_train import train_all
    train_all()
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Step 6: Evaluate ───────────────────────────────────────────────────
    step(6, "Evaluation + Results Tables")
    t = time.time()
    # Evaluation is already handled in module5_train.py
    print("  ✓ Evaluation already completed in Step 5 (results saved to outputs/models/)")
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Step 7: Visualize ──────────────────────────────────────────────────
    step(7, "Generating Publication-Quality Figures")
    t = time.time()
    from module7_visualize import generate_all
    generate_all()
    print(f"  ✓ Done in {time.time()-t:.1f}s")

    # ── Final Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*58}")
    print(f"  PIPELINE COMPLETE — Total time: {elapsed:.1f}s")
    print(f"{'='*58}")
    print(f"""
  Output files:
  ├── outputs/data/
  │   ├── eeg_data.npy          ← raw EEG (900, 19, 7680)
  │   ├── labels.npy            ← class labels (900,)
  │   └── behavioral.csv        ← ETVS, GAI, SSCS scores
  ├── outputs/features/
  │   ├── pac_features.npy      ← PAC vectors (900, 570)
  │   ├── train.pt / val.pt / test.pt
  │   └── scaler_min/max.npy
  ├── outputs/models/
  │   ├── transformer_best.pt   ← best Transformer weights
  │   ├── mlp_best.pt
  │   ├── cnn_best.pt
  │   ├── training_log.csv
  │   └── results_summary.csv
  └── outputs/figures/
      ├── fig1_pac_comodulogram.png
      ├── fig2_training_curves.png
      ├── fig3_confusion_matrix.png
      ├── fig4_roc_curves.png
      ├── fig5_attention_heatmap.png
      ├── fig6_ablation.png
      └── fig7_behavioral_distributions.png
""")


if __name__ == '__main__':
    run()
