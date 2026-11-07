import { Panel } from "../components/charts";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { useManifest } from "../lib/data";

function Architecture() {
  const box = (x: number, y: number, w: number, h: number, t: string, s?: string, fill = "var(--paper-2)") => (
    <g><rect x={x} y={y} width={w} height={h} rx="4" fill={fill} stroke="var(--navy)" strokeWidth="1.3" /><text x={x + w / 2} y={y + (s ? h / 2 - 2 : h / 2 + 5)} textAnchor="middle" fontSize="12.5" fontWeight="600" fill="var(--ink)">{t}</text>{s && <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize="10.5" fill="var(--ink-2)">{s}</text>}</g>
  );
  return (
    <svg viewBox="0 0 760 320" width="100%" role="img" aria-label="Transformer architecture">
      {box(14, 128, 120, 56, "Input tokens", "15 × 41")}{box(164, 128, 120, 56, "Linear embedding", "41 → 128")}{box(314, 118, 134, 76, "+ positions", "+ [CLS] → 16 × 128", "var(--signal-soft)")}
      <rect x="478" y="30" width="172" height="262" rx="6" fill="none" stroke="var(--signal)" strokeWidth="1.5" strokeDasharray="5 4" /><text x="564" y="22" textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--signal)">× 4 encoder blocks (pre-LayerNorm)</text>
      {box(493, 44, 142, 46, "LayerNorm")}{box(493, 100, 142, 58, "Multi-head attention", "8 heads · d_k = 16")}{box(493, 168, 142, 40, "LayerNorm")}{box(493, 218, 142, 60, "Feed-forward", "128→256→128 · GELU")}
      {box(668, 98, 84, 46, "LN, [CLS]")}{box(668, 160, 84, 66, "MLP head", "128→64→3")}
      <g stroke="var(--navy)" strokeWidth="1.3" fill="none"><path d="M134 156h30M284 156h30M448 156h30M650 156h12M710 144v16" /></g>
      <text x="710" y="250" textAnchor="middle" fontSize="11" fill="var(--ink-2)">softmax</text><text x="710" y="266" textAnchor="middle" fontSize="11.5" fontWeight="600" fill="var(--ink)">3 classes</text>
    </svg>
  );
}

export default function Method() {
  const { data: man } = useManifest();
  const P = (man?.dsp?.profiles as any)?.sim;
  return (
    <div className="page" style={{ maxWidth: 980 }}>
      <PageHead step="Step 7 of 7 · Method" title="Method and software validation">Definitions, equations, the simulation specification, how the software was checked, and the limits of what it shows.</PageHead>

      <section><h2>Phase-amplitude coupling</h2>
        <p style={{ marginTop: 8 }}>Theta waves (4–8 Hz) provide a slow reference in which gamma bursts (30–80 Hz) occur at preferred phases. This cross-frequency coupling supports working memory and attention and is altered around glioma, in epilepsy and in other conditions. For each channel and 2-second epoch the theta phase φ comes from the Hilbert transform of the band-passed signal and the gamma amplitude A from its envelope. The phase circle is divided into N = 18 bins; P(j) is the normalised mean amplitude in bin j. The modulation index (Tort et al., 2010) is the Kullback–Leibler divergence from a uniform distribution, scaled to [0, 1]:</p>
        <div className="eq">MI = D<sub>KL</sub>(P ‖ U) / log N = ( log N − H(P) ) / log N, &nbsp; P(j) = ⟨A⟩<sub>φ∈j</sub> / Σ<sub>k</sub> ⟨A⟩<sub>φ∈k</sub></div>
        <p style={{ marginTop: 12 }}>Signals are resampled to 256 Hz where necessary, band-passed with 4th-order Butterworth filters applied forward and backward (zero phase), cut into non-overlapping 2-second epochs, and epochs with a peak above the artefact limit (default ±150 µV) are excluded. Two pairs are measured: theta→gamma and alpha (8–13 Hz)→beta (13–30 Hz). On the Analyse page the observed MI of each channel is compared with circular-shift surrogates (amplitude series shifted by at least one second), giving a z-score and a one-sided p-value.</p></section>

      <section className="section"><h2>Classifier</h2>
        <Panel><Architecture /><p className="cap"><b>Figure 1.</b> Each epoch becomes a token of 41 values: 19 theta–gamma MI, 19 alpha–beta MI and the three behavioural scores (ETVS eye-tracking variance, GAI gait asymmetry, SSCS speech complexity), which are repeated on every token.</p></Panel>
        <p style={{ marginTop: 14 }}>Learnable positions and a [CLS] token are added, followed by four pre-LayerNorm encoder blocks (8 heads, 256-unit GELU feed-forward, dropout 0.1) and a 128→64→3 head, about 546,000 parameters. Training uses Adam (3·10⁻⁴, weight decay 10⁻⁵), cosine decay to 10⁻⁶, batch 32, up to 80 epochs with early stopping (patience 10). Baselines: 1-D CNN, MLP, logistic regression, random forest and gradient boosting, trained on the same splits.</p></section>

      <section className="section"><h2>Simulation specification</h2>
        <p>No public dataset pairs early-tumour EEG with behavioural scores, so subjects are synthesised. Each recording is a 1/f background plus delta, theta, alpha, beta and gamma components made by spectral synthesis, sensor noise, and explicit coupling: gamma amplitude is multiplied by 1 + m·cos φ<sub>θ</sub> and beta amplitude by the alpha phase. Disruption lowers m in the affected channels. A fixed gain of 3.5 relates the coupling parameter (0.18 in healthy subjects) to the generative depth m so that the <i>measured</i> MI lies in a physiological range; without it the estimator's noise floor would hide the effect.</p>
        {P && <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>
          <tr><td>Subjects</td><td>900 (300 per class)</td></tr>
          <tr><td>Channels affected, Early / High Risk</td><td>5 / 10 (scaled by individual severity)</td></tr>
          <tr><td>Theta–gamma coupling multiplier, Early / High Risk</td><td>0.70 / 0.30–0.45</td></tr>
          <tr><td>Alpha–beta multiplier, Early / High Risk</td><td>1.15 / 1.35–1.40</td></tr>
          <tr><td>Gamma power multiplier, High Risk</td><td>1.30</td></tr>
          <tr><td>Latent severity jitter (SD)</td><td>{P.latent_sd} (classes overlap)</td></tr>
          <tr><td>Disruption varies across epochs</td><td>{String(P.temporal)}</td></tr>
          <tr><td>Behavioural SD multiplier / shared confounder SD</td><td>{P.beh_sd_mult} / {P.fatigue_sd}</td></tr>
          <tr><td>Electrode gain variability (SD) / artefact rate per epoch</td><td>{P.impedance_sd} / {P.artifact_rate}</td></tr></tbody></table></div>}
        <p className="cap" style={{ marginTop: 10 }}>All values live in <code>research/neuroai/config.py</code> and are exported to the browser, which runs the same generator.</p></section>

      <section className="section"><h2>How the software was checked</h2>
        <ol className="steps">
          <li><b>Cross-language parity.</b> Filter coefficients are exported from SciPy. The Hilbert transform, modulation index, artefact handling and Transformer forward pass are implemented again in TypeScript and tested against Python fixtures (MI within 2·10⁻⁴; logits and attention within 2·10⁻³).</li>
          <li><b>Recording reader.</b> The EDF and CSV readers, the arbitrary-rate resampler and the surrogate test have unit tests, including a round trip through a synthetic EDF file and recovery of a known coupling.</li>
          <li><b>Computed, not drawn.</b> ROC curves, confusion matrices, training logs, attention and importance maps are produced by the trained models and serialised to JSON; the interface only renders them.</li>
          <li><b>Uncertainty.</b> Three independent splits and initialisations, bootstrap intervals, calibration error and shift tests (noise, unseen lesion geometry, shuffled epochs).</li>
        </ol></section>

      <section className="section"><h2>Limitations</h2>
        <ul>
          <li>All training and evaluation data are simulated under stated assumptions; results measure learnability, not clinical accuracy.</li>
          <li>Scalp EEG blurs the intracranial coupling effects described in glioma studies; whether the signal survives is an open empirical question.</li>
          <li>Coupling also changes in epilepsy, anxiety, sleep loss and with alcohol. Specificity for tumours is untested.</li>
          <li>Ground truth for a pre-radiographic claim requires following people until imaging, which is what the study planner is for. The Temple University Hospital EEG Corpus is a practical first check of the coupling pipeline on real clinical EEG.</li>
        </ul>
        <div className="callout bad"><p><b>Responsible use.</b> This software must not be used to screen, diagnose or reassure any person. A clinical tool would need prospective data, regulatory approval and clinical oversight.</p></div></section>

      <section className="section"><h2>Project and references</h2>
        <p>Developed by <b>Shrikant Kudale</b> under the guidance of <b>Prof. Dr. Jayashree Prasad</b>. Source code and the reproducible research pipeline are in the repository.</p>
        <ol style={{ fontSize: "0.9rem", color: "var(--ink-2)" }}>
          <li>Canolty RT, Knight RT. The functional role of cross-frequency coupling. Trends Cogn Sci 2010;14(11).</li>
          <li>Tort ABL, et al. Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. J Neurophysiol 2010;104(2).</li>
          <li>Bao L, et al. Disrupted gamma and theta oscillatory coupling in glioma patients. NeuroImage: Clinical 2020;28.</li>
          <li>Vaswani A, et al. Attention is all you need. NeurIPS 2017.</li>
          <li>Hanley JA, McNeil BJ. The meaning and use of the area under a ROC curve. Radiology 1982;143.</li>
          <li>Buderer NMF. Statistical methodology: sample size for sensitivity and specificity. Acad Emerg Med 1996;3(9).</li>
          <li>Obeid I, Picone J. The Temple University Hospital EEG Data Corpus. Front Neurosci 2016;10.</li>
          <li>Abnar S, Zuidema W. Quantifying attention flow in Transformers. ACL 2020.</li>
        </ol></section>
      <StepNav prev={["/planner", "Plan a validation study"]} next={["/", "Back to the overview"]} />
    </div>
  );
}
