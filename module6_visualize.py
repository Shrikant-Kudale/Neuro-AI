"""
MODULE 6 — All Visualizations
===============================
Generates 7 publication-quality figures:
  1. PAC Comodulogram (Normal vs Early Disruption vs High Risk)
  2. Model Performance Comparison (bar chart)
  3. Confusion Matrix (Transformer)
  4. ROC Curves (3 classes)
  5. Attention Heatmap (by class)
  6. Ablation Study Bar Chart
  7. Behavioral Feature Distributions (violin plots)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, hilbert
import os, warnings
warnings.filterwarnings('ignore')

FIG_DIR   = 'outputs/figures'
MODEL_DIR = 'outputs/models'
DATA_DIR  = 'outputs/data'
os.makedirs(FIG_DIR, exist_ok=True)

COLORS      = ['#2196F3', '#FF9800', '#F44336']   # blue, orange, red
CLASS_NAMES = ['Normal', 'Early Disruption', 'High Risk']
STYLE       = {'font.family':'DejaVu Sans', 'axes.spines.top':False,
               'axes.spines.right':False, 'figure.dpi':150}
plt.rcParams.update(STYLE)


# ────────────────────────────────────────────────────────────────────────────
# PLOT 1 — PAC Comodulogram
# ────────────────────────────────────────────────────────────────────────────
def plot_pac_comodulogram():
    print("  Plot 1: PAC Comodulogram...")
    eeg_data = np.load(f'{DATA_DIR}/eeg_data.npy')
    labels   = np.load(f'{DATA_DIR}/labels.npy')

    def bp_filter(sig, lo, hi, fs=256, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
        return filtfilt(b, a, sig)

    def compute_comodulogram(eeg_ch, fs=256):
        phase_freqs = [(4,8),(8,13),(13,20)]
        amp_freqs   = [(20,35),(35,55),(55,80)]
        labels_p = ['θ (4-8)', 'α (8-13)', 'β (13-20)']
        labels_a = ['γ-lo (20-35)', 'γ-mid (35-55)', 'γ-hi (55-80)']
        mi_matrix = np.zeros((len(phase_freqs), len(amp_freqs)))
        ep = eeg_ch[:512]    # first 2-sec epoch
        for i, (pl, ph) in enumerate(phase_freqs):
            phase = np.angle(hilbert(bp_filter(ep, pl, ph)))
            for j, (al, ah) in enumerate(amp_freqs):
                amp   = np.abs(hilbert(bp_filter(ep, al, ah)))
                bins  = np.linspace(-np.pi, np.pi, 19)
                p     = np.array([amp[(phase>=bins[b])&(phase<bins[b+1])].mean()
                                  if ((phase>=bins[b])&(phase<bins[b+1])).sum()>0 else 0
                                  for b in range(18)])
                if p.sum() > 0:
                    p /= p.sum()
                    p  = np.clip(p, 1e-10, 1)
                    kl = np.sum(p * np.log(p / (1/18)))
                    mi_matrix[i,j] = kl / np.log(18)
        return mi_matrix, labels_p, labels_a

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    cmap = LinearSegmentedColormap.from_list('pac', ['#EEF2FF','#3B5BDB','#1A1A4E'])

    vmax = 0
    matrices = []
    for cid in range(3):
        idx  = np.where(labels == cid)[0][0]
        mat, lp, la = compute_comodulogram(eeg_data[idx, 0])
        matrices.append(mat)
        vmax = max(vmax, mat.max())

    for cid, ax in enumerate(axes):
        im = ax.imshow(matrices[cid], aspect='auto', cmap=cmap,
                       vmin=0, vmax=vmax, origin='lower')
        ax.set_xticks(range(3)); ax.set_xticklabels(['γ-lo','γ-mid','γ-hi'], fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(['θ','α','β'], fontsize=9)
        ax.set_xlabel('Amplitude Frequency', fontsize=10)
        ax.set_ylabel('Phase Frequency',     fontsize=10)
        ax.set_title(f'{CLASS_NAMES[cid]}\n(MI shown)', fontsize=11, fontweight='bold',
                     color=COLORS[cid])
        plt.colorbar(im, ax=ax, shrink=0.85, label='Modulation Index')

    fig.suptitle('Phase-Amplitude Coupling Comodulograms by Risk Class',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/1_pac_comodulogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 1_pac_comodulogram.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Model Performance Comparison
# ────────────────────────────────────────────────────────────────────────────
def plot_model_comparison():
    print("  Plot 2: Model Performance Comparison...")
    results = np.load(f'{MODEL_DIR}/results.npy', allow_pickle=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    mlabels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    model_names = [r['name'] for r in results]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    bar_colors = ['#1565C0', '#E65100', '#2E7D32']

    for i, (r, col) in enumerate(zip(results, bar_colors)):
        vals = [r[m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width, label=r['name'],
                      color=col, alpha=0.88, edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(mlabels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/2_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 2_model_comparison.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Confusion Matrix
# ────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix():
    print("  Plot 3: Confusion Matrix...")
    results = np.load(f'{MODEL_DIR}/results.npy', allow_pickle=True)
    tf_r    = results[0]
    cm      = tf_r['confusion_matrix']
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = LinearSegmentedColormap.from_list('cm', ['#FFFFFF', '#1565C0'])
    im   = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            color = 'white' if cm_norm[i,j] > 0.5 else '#1A237E'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.0%})',
                    ha='center', va='center', fontsize=11.5,
                    fontweight='bold', color=color)

    ax.set_xticks(range(3)); ax.set_xticklabels(['Normal','Early Dis.','High Risk'],
                                                  fontsize=10.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(['Normal','Early Dis.','High Risk'],
                                                  fontsize=10.5)
    ax.set_xlabel('Predicted Label',  fontsize=12, labelpad=8)
    ax.set_ylabel('True Label',       fontsize=12, labelpad=8)
    ax.set_title('Confusion Matrix — Transformer Model', fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax, shrink=0.85, label='Proportion')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/3_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 3_confusion_matrix.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 4 — ROC Curves
# ────────────────────────────────────────────────────────────────────────────
def plot_roc_curves():
    print("  Plot 4: ROC Curves...")
    from sklearn.metrics import roc_curve, auc as sk_auc
    results  = np.load(f'{MODEL_DIR}/results.npy', allow_pickle=True)
    tf_r     = results[0]
    y_test   = tf_r['y_test']
    y_proba  = tf_r['y_proba']

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ls_cycle = ['-', '--', '-.']

    for cid, (cname, col, ls) in enumerate(zip(CLASS_NAMES, COLORS, ls_cycle)):
        fpr, tpr, _ = roc_curve(y_test == cid, y_proba[:, cid])
        roc_auc     = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.2, linestyle=ls,
                label=f'{cname}  (AUC = {roc_auc:.3f})')

    ax.plot([0,1],[0,1], 'k--', lw=1.2, alpha=0.5, label='Random classifier')
    ax.fill_between([0,1],[0,1], alpha=0.04, color='gray')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate',  fontsize=12)
    ax.set_title('ROC Curves — Transformer Model (One-vs-Rest)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10.5, loc='lower right', framealpha=0.95)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/4_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 4_roc_curves.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Attention Heatmap
# ────────────────────────────────────────────────────────────────────────────
def plot_attention_heatmap():
    print("  Plot 5: Attention Heatmap...")
    attn_by_class = np.load(f'{MODEL_DIR}/attention_weights.npy', allow_pickle=True).item()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    cmap = LinearSegmentedColormap.from_list('attn', ['#F3E5F5','#7B1FA2','#1A0033'])

    for cid, ax in enumerate(axes):
        attn = attn_by_class[cid]
        im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0)
        ax.set_title(f'{CLASS_NAMES[cid]}\nAttention Pattern', fontsize=11,
                     fontweight='bold', color=COLORS[cid])
        ax.set_xlabel('Key Token (Epoch)', fontsize=9)
        ax.set_ylabel('Query Token',       fontsize=9)
        n = attn.shape[0]
        tick_labels = ['CLS'] + [str(i) for i in range(1, n)]
        ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, fontsize=7, rotation=45)
        ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.85, label='Attention Weight')

    fig.suptitle('Multi-Head Self-Attention Weight Distribution by Risk Class\n'
                 '(Higher intensity = more focus on that token)',
                 fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/5_attention_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 5_attention_heatmap.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Ablation Study
# ────────────────────────────────────────────────────────────────────────────
def plot_ablation():
    print("  Plot 6: Ablation Study...")
    ablation  = np.load(f'{MODEL_DIR}/ablation.npy', allow_pickle=True).item()
    results   = np.load(f'{MODEL_DIR}/results.npy',  allow_pickle=True)
    tf_r      = results[0]

    # Build full ablation table
    configs = ['PAC + Behavioral\n(Full Model)', 'PAC Only', 'Behavioral Only']
    full_acc  = tf_r['accuracy']
    full_f1   = tf_r['f1']
    full_ed   = tf_r['per_class']['Early Disruption']['f1']
    full_hr   = tf_r['per_class']['High Risk']['f1']

    accs = [full_acc,
            ablation['PAC Only']['accuracy'],
            ablation['Behavioral Only']['accuracy']]
    f1s  = [full_f1,
            ablation['PAC Only']['f1'],
            ablation['Behavioral Only']['f1']]
    ed   = [full_ed,
            ablation['PAC Only']['early_f1'],
            ablation['Behavioral Only']['early_f1']]
    hr   = [full_hr,
            ablation['PAC Only']['highrisk_f1'],
            ablation['Behavioral Only']['highrisk_f1']]

    x     = np.arange(len(configs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_c = ['#1565C0','#66BB6A','#FF7043','#AB47BC']

    for i, (vals, label, col) in enumerate(zip(
            [accs, f1s, ed, hr],
            ['Accuracy','Macro F1','Early Dis. F1','High Risk F1'],
            bar_c)):
        bars = ax.bar(x + i*width, vals, width, label=label,
                      color=col, alpha=0.87, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(configs, fontsize=10.5)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study — Impact of Multi-Modal Fusion',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/6_ablation_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 6_ablation_study.png")


# ────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Behavioral Feature Distributions
# ────────────────────────────────────────────────────────────────────────────
def plot_behavioral_distributions():
    print("  Plot 7: Behavioral Feature Distributions...")
    behavioral = np.load(f'{DATA_DIR}/behavioral.npy')
    labels     = np.load(f'{DATA_DIR}/labels.npy')
    feat_names = ['Eye-Tracking Variance\nScore (ETVS, °)',
                  'Gait Asymmetry\nIndex (GAI)',
                  'Speech Syntax\nComplexity Score (SSCS)']

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for fi, (ax, fname) in enumerate(zip(axes, feat_names)):
        all_data  = [behavioral[labels == cid, fi] for cid in range(3)]
        positions = [1, 2, 3]

        vp = ax.violinplot(all_data, positions=positions,
                           showmeans=True, showextrema=True, widths=0.7)

        for i, (body, col) in enumerate(zip(vp['bodies'], COLORS)):
            body.set_facecolor(col); body.set_alpha(0.75); body.set_edgecolor('white')
        for part in ['cbars','cmins','cmaxes','cmeans']:
            if part in vp:
                vp[part].set_color('#222222'); vp[part].set_linewidth(1.5)

        # Scatter overlay
        for cid, (data, col) in enumerate(zip(all_data, COLORS)):
            jitter = np.random.default_rng(cid).normal(0, 0.06, len(data))
            ax.scatter(positions[cid] + jitter, data,
                       color=col, alpha=0.2, s=8, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(['Normal','Early Dis.','High Risk'], fontsize=10)
        ax.set_title(fname, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(axis='y', alpha=0.25)

    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), framealpha=0.9)
    fig.suptitle('Behavioral Biomarker Distributions Across Risk Classes',
                 fontsize=13, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/7_behavioral_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    Saved: 7_behavioral_distributions.png")


def generate_all():
    print(f"  Saving figures to {FIG_DIR}/")
    plot_pac_comodulogram()
    plot_model_comparison()
    plot_confusion_matrix()
    plot_roc_curves()
    plot_attention_heatmap()
    plot_ablation()
    plot_behavioral_distributions()
    print(f"\n  All 7 figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    print("\n[MODULE 6] Generating All Visualizations")
    print("=" * 45)
    generate_all()
    print("\n✓ Module 6 complete.\n")
