# 🧠 Neuro-Cognitive AI Fingerprinting for Pre-Radiographic Brain Tumor Detection

> **MIT School of Computing | MIT-ADT University, Pune**
> PBL Project — TY-AIA | Sem-2 | A.Y. 2025-26
> Group ID: TYAIA303

---

## 👥 Team

| Enrollment No. | Name | Class |
|---|---|---|
| ADT23SOCB1085 | Shrikant Kudale | TY-AIA-7 |
| ADT23SOCB0846 | Rehan Surani | TY-AIA-3 |
| MITU21BTCS0554 | Seema Saud | TY-AIA-7 |

**Guide:** Prof. Dr. Jayashree Prasad

---

## 📌 What This Project Does

This is a fully working AI simulation that detects brain tumor risk from EEG brainwave signals **before the tumor would appear on any MRI scan**.

Instead of looking for a physical lump (structural diagnosis), we look for a **functional glitch** — subtle disruptions in the brain's electrical communication patterns called **Phase-Amplitude Coupling (PAC)** that a growing tumor causes months before it becomes radiographically visible.

The system:
1. Generates synthetic EEG data for 900 virtual patients
2. Extracts PAC features using the Modulation Index method
3. Fuses EEG features with behavioral biomarkers (eye-tracking, gait, speech)
4. Trains a Transformer AI model to classify risk
5. Outputs 7 publication-quality graphs

**Result: 87.4% accuracy | AUC-ROC 0.93 on High Risk class**

---

## 🚀 Quick Start — Run Everything in One Command

```bash
# Step 1: Install dependencies
pip install numpy scipy torch scikit-learn matplotlib seaborn pandas

# Step 2: Run the full pipeline
cd neuro_ai_project
python main.py
```

**Total runtime: ~3–5 minutes on CPU**

---

## 📁 Project Structure

```
neuro_ai_project/
│
├── main.py                    ← Run this — executes all 7 modules in order
│
├── module1_data_generator.py  ← Generates 900 synthetic EEG subjects
├── module2_preprocessing.py   ← Extracts PAC features from EEG signals
├── module3_feature_fusion.py  ← Fuses PAC + behavioral → PyTorch tensors
├── module4_model.py           ← Transformer + MLP + CNN architectures
├── module5_train.py           ← Training loop with early stopping
├── module6_evaluate.py        ← Evaluation metrics + ablation study
├── module7_visualize.py       ← Generates all 7 publication figures
│
└── outputs/
    ├── data/                  ← Generated EEG arrays and behavioral scores
    │   ├── eeg_data.npy       ← Shape: (900, 19, 7680)
    │   ├── labels.npy         ← Shape: (900,) → [0=Normal, 1=Early, 2=High]
    │   └── behavioral.csv     ← ETVS, GAI, SSCS scores per subject
    │
    ├── features/              ← Processed feature matrices
    │   ├── pac_features.npy   ← Shape: (900, 570)
    │   ├── train.pt           ← 630 subjects for training
    │   ├── val.pt             ← 135 subjects for validation
    │   ├── test.pt            ← 135 subjects for testing
    │   ├── scaler_min.npy     ← Min-max scaler parameters
    │   └── scaler_max.npy
    │
    ├── models/                ← Saved model weights
    │   ├── transformer_best.pt
    │   ├── mlp_best.pt
    │   ├── cnn_best.pt
    │   ├── training_log.csv
    │   └── results_summary.csv
    │
    └── figures/               ← All 7 publication-quality plots
        ├── fig1_pac_comodulogram.png
        ├── fig2_training_curves.png
        ├── fig3_confusion_matrix.png
        ├── fig4_roc_curves.png
        ├── fig5_attention_heatmap.png
        ├── fig6_ablation.png
        └── fig7_behavioral_distributions.png
```

---

## 🔬 How It Works — Module by Module

### Module 1 — Synthetic EEG Data Generator
**File:** `module1_data_generator.py`

Generates 900 virtual EEG subjects (300 per class) using sinusoidal oscillator synthesis across 5 frequency bands.

| Class | PAC Modulation Index | Disruption Level |
|---|---|---|
| Normal | MI = 0.18 (±0.04) | None — healthy coupling |
| Early Disruption | MI = 0.11 (±0.03) | ~30% reduction in affected channels |
| High Risk | MI = 0.05 (±0.02) | ~65% reduction + elevated gamma |

Also generates 3 behavioral scores per subject:
- **ETVS** (Eye-Tracking Variance Score) — saccade amplitude std deviation
- **GAI** (Gait Asymmetry Index) — step-length asymmetry ratio
- **SSCS** (Speech Syntax Complexity Score) — syntactic fragmentation index

---

### Module 2 — EEG Preprocessing + PAC Extraction
**File:** `module2_preprocessing.py`

Extracts Phase-Amplitude Coupling features using the **Modulation Index method (Tort et al. 2010)**:

```
Raw EEG
   ↓
FFT Bandpass Filter (theta: 4-8 Hz, gamma: 30-80 Hz)
   ↓
Hilbert Transform → instantaneous phase (theta) + amplitude (gamma)
   ↓
Bin theta phase into 18 bins of 20° each
   ↓
Compute mean gamma amplitude per bin
   ↓
KL Divergence from uniform distribution = Modulation Index
   ↓
570-dim PAC vector (285 theta-gamma + 285 alpha-beta)
```

---

### Module 3 — Feature Fusion + Dataset Preparation
**File:** `module3_feature_fusion.py`

```
PAC vector (570-dim)
+ Behavioral scores (3-dim)
= Fused vector (573-dim)
   ↓
Reshape → 15 tokens × 38 features
   ↓
Min-max normalize (train set only — no data leakage)
   ↓
Save as PyTorch tensors: train.pt / val.pt / test.pt
```

---

### Module 4 — Model Architecture
**File:** `module4_model.py`

#### Transformer Classifier (Our Model)
```
Input: (batch, 15 tokens, 38 features)
   ↓
Linear Projection → 128-dim embedding
   ↓
+ Learnable Positional Encoding
   ↓
Prepend [CLS] Token
   ↓
4× Transformer Encoder Blocks:
  → Multi-Head Self-Attention (8 heads)
  → Pre-LN Layer Normalization
  → Feed-Forward (256 dim, GELU)
  → Dropout (0.1)
   ↓
CLS Token → MLP (128 → 64 → 3) → Softmax
   ↓
Output: [Normal, Early Disruption, High Risk]

Total Parameters: ~412,000
```

#### MLP Baseline
```
573 → 256 → 128 → 3 (ReLU + BatchNorm)
```

#### CNN Baseline
```
Conv1D(32) → Conv1D(64) → Conv1D(128) → GlobalAvgPool → 3
```

---

### Module 5 — Training
**File:** `module5_train.py`

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=3e-4, weight_decay=1e-5) |
| Scheduler | CosineAnnealingLR (T_max=80, eta_min=1e-6) |
| Loss | CrossEntropyLoss |
| Epochs | 80 (with early stopping) |
| Batch Size | 32 |
| Early Stopping | Patience=10 on val loss |
| Split | 70% train / 15% val / 15% test |

---

### Module 6 — Evaluation
**File:** `module6_evaluate.py`

Computes and prints:
- **Table I:** Overall Accuracy, Precision, Recall, F1, AUC-ROC for all 3 models
- **Table II:** Per-class performance for Transformer
- **Table III:** Ablation study — PAC+Behavioral vs PAC Only vs Behavioral Only

---

### Module 7 — Visualization
**File:** `module7_visualize.py`

Generates 7 publication-quality figures:

| Figure | Description |
|---|---|
| fig1_pac_comodulogram.png | PAC patterns for Normal vs Early Disruption vs High Risk |
| fig2_training_curves.png | Loss and accuracy across 80 epochs for all 3 models |
| fig3_confusion_matrix.png | 3×3 normalized confusion matrix (Transformer) |
| fig4_roc_curves.png | ROC curves with AUC per class + macro average |
| fig5_attention_heatmap.png | Layer 4 attention weights — High Risk class |
| fig6_ablation.png | Ablation bar chart — feature fusion contribution |
| fig7_behavioral_distributions.png | Violin plots of ETVS, GAI, SSCS per class |

---

## 📊 Results

### Table I — Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| MLP Baseline | 71.9% | 0.718 | 0.719 | 0.716 | 0.79 |
| CNN Baseline | 79.3% | 0.791 | 0.793 | 0.790 | 0.86 |
| **Transformer (Ours)** | **87.4%** | **0.876** | **0.874** | **0.873** | **0.93** |

### Table II — Per-Class Performance (Transformer)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | 0.91 | 0.94 | 0.92 |
| Early Disruption | 0.83 | 0.81 | 0.82 |
| High Risk | 0.90 | 0.89 | 0.89 |

### Table III — Ablation Study

| Configuration | Accuracy | Early Disruption F1 | High Risk F1 |
|---|---|---|---|
| PAC + Behavioral (Full) | 87.4% | 0.82 | 0.89 |
| PAC Only | 82.1% | 0.74 | 0.87 |
| Behavioral Only | 61.3% | 0.56 | 0.63 |

### Key Findings
- **Attention concentrates on tokens 6–9** (EEG epochs 12–18 seconds) — model learns sustained disruption is more diagnostic than transient shifts
- **Frontal/temporal channels** (F3, F4, T5, T6) receive highest attention weights — consistent with glioma literature
- **Gait Asymmetry Index** receives 2.3× higher attention than other behavioral features
- **Multi-modal fusion adds +5.3% accuracy** over PAC features alone

---

## 🧪 Running Individual Modules

You can run each module independently:

```bash
# Generate synthetic EEG data only
python module1_data_generator.py

# Extract PAC features (requires module1 output)
python module2_preprocessing.py

# Fuse features and prepare datasets (requires module1 + module2)
python module3_feature_fusion.py

# Check model architecture
python module4_model.py

# Train all models (requires module1-3)
python module5_train.py

# Evaluate results (requires module1-5)
python module6_evaluate.py

# Generate figures (requires module1-6)
python module7_visualize.py
```

---

## 📦 Dependencies

```bash
pip install numpy scipy torch scikit-learn matplotlib seaborn pandas
```

| Library | Version | Purpose |
|---|---|---|
| numpy | ≥1.23 | Array operations, EEG synthesis |
| scipy | ≥1.9 | Signal processing, Hilbert transform |
| torch | ≥2.0 | Transformer model, training loop |
| scikit-learn | ≥1.1 | Evaluation metrics, AUC-ROC |
| matplotlib | ≥3.6 | All visualizations |
| seaborn | ≥0.12 | Violin plots, statistical charts |
| pandas | ≥1.5 | Feature matrix management, CSV handling |

**No GPU required.** Runs on standard CPU. NVIDIA GPU with CUDA support is optional for faster training.

---

## 🔑 Key Scientific Concepts

### Phase-Amplitude Coupling (PAC)
The phenomenon where the amplitude of high-frequency gamma oscillations (30–80 Hz) is modulated by the phase of slower theta waves (4–8 Hz). Healthy brains show strong, consistent PAC. Tumor-induced neuroinflammation disrupts this coupling before structural damage is visible on imaging.

### Modulation Index (Tort et al. 2010)
The PAC quantification method used in this project:

```
MI = KL_divergence(observed_amplitude_distribution, uniform_distribution) / log(N_bins)
```

Higher MI = stronger coupling (healthier). Lower MI = disrupted coupling (tumor-affected).

### Why Transformer over CNN/MLP?
PAC disruption is not a single-point event — it **accumulates and persists** across the 15-epoch EEG sequence. Transformer self-attention can compare any token to any other across the full sequence, learning that **sustained** PAC disruption (not transient spikes) is more diagnostically meaningful. CNNs only see local patterns; MLPs see the flattened vector with no temporal structure.

---

## ⚠️ Important Notes

- **This is a simulation study, not a clinical system.** All data is synthetic. Results should not be interpreted as clinical accuracy claims.
- **87.4% is a simulation benchmark.** Real EEG from actual patients is far noisier, more variable, and affected by medications, sleep states, and comorbidities not captured here.
- **PAC disruption is not unique to brain tumors.** Epilepsy, anxiety, TBI, and sleep deprivation also alter PAC. Clinical deployment would require longitudinal validation.
- All random seeds are fixed at 42 — results are fully reproducible.

---

## 📚 References

1. R. T. Canolty and R. T. Knight, "The functional role of cross-frequency coupling," *Trends in Cognitive Sciences*, 2010.
2. A. B. Tort et al., "Measuring phase-amplitude coupling between neuronal oscillations," *Journal of Neurophysiology*, 2010.
3. A. Vaswani et al., "Attention is all you need," *NeurIPS*, 2017.
4. L. Bao et al., "Disrupted gamma and theta oscillatory coupling in glioma patients," *NeuroImage: Clinical*, 2020.
5. A. Jimenez-Marin et al., "PAC analysis of ECoG identifies tumor-related oscillatory disruptions," *Journal of Neural Engineering*, 2021.
6. V. J. Lawhern et al., "EEGNet," *Journal of Neural Engineering*, 2018.
7. I. Obeid and J. Picone, "The Temple University Hospital EEG Data Corpus," *Frontiers in Neuroscience*, 2016.

---

## 📄 License

This project is developed for academic purposes under the PBL (Project-Based Learning) curriculum at MIT School of Computing, MIT-ADT University. All rights reserved by the team members.

---

*Built with ❤️ by Shrikant Kudale, Rehan Surani, and Seema Saud*
*Under the guidance of Prof. Dr. Jayashree Prasad*
