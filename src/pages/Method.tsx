import { Panel } from "../components/charts";
import { useManifest } from "../lib/data";

function Architecture() {
  const box = (x: number, y: number, w: number, h: number, t: string, s?: string, fill = "var(--paper-2)") => (
    <g><rect x={x} y={y} width={w} height={h} rx="8" fill={fill} stroke="var(--ink)" strokeWidth="1.5" /><text x={x + w / 2} y={y + (s ? h / 2 - 2 : h / 2 + 5)} textAnchor="middle" fontSize="13" fontWeight="700" fill="var(--ink)">{t}</text>{s && <text x={x + w / 2} y={y + h / 2 + 15} textAnchor="middle" fontSize="11" fill="var(--ink-2)">{s}</text>}</g>
  );
  return (
    <svg viewBox="0 0 760 330" width="100%" role="img" aria-label="Transformer architecture diagram">
      {box(20, 130, 120, 58, "Input tokens", "15 × 41")}{box(170, 130, 120, 58, "Linear embed", "41 → 128")}{box(320, 120, 130, 78, "+ positions", "+ [CLS] → 16 × 128", "var(--signal-soft)")}
      <rect x="480" y="30" width="170" height="270" rx="12" fill="none" stroke="var(--signal)" strokeWidth="2" strokeDasharray="6 4" /><text x="565" y="22" textAnchor="middle" fontSize="12.5" fontWeight="700" fill="var(--signal)">× 4 encoder blocks (pre-LN)</text>
      {box(495, 46, 140, 50, "LayerNorm")}{box(495, 106, 140, 60, "Multi-head attention", "8 heads · d_k = 16")}{box(495, 176, 140, 44, "LayerNorm")}{box(495, 230, 140, 56, "Feed-forward", "128→256→128 · GELU")}
      {box(670, 100, 80, 50, "LN, [CLS]")}{box(670, 170, 80, 70, "MLP head", "128→64→3")}
      <g stroke="var(--ink)" strokeWidth="1.5" fill="none"><path d="M140 159h30M290 159h30M450 159h30M650 159h20M710 150v20" /></g>
      <text x="710" y="270" textAnchor="middle" fontSize="12" fill="var(--ink-2)">softmax →</text><text x="710" y="286" textAnchor="middle" fontSize="12" fontWeight="700" fill="var(--ink)">3 classes</text>
    </svg>
  );
}

export default function Method() {
  const { data: man } = useManifest();
  const P = man?.dsp?.profiles as any;
  return (
    <div className="page" style={{ maxWidth: 1000 }}>
      <h1 style={{ fontSize: "clamp(1.9rem,3.6vw,2.8rem)" }}>Method</h1>
      <p className="lede" style={{ marginTop: 10 }}>The pipeline follows the published paper. Where the paper is silent or inconsistent, this page says what was decided.</p>

      <section className="section"><h2>The idea</h2>
        <p>Theta waves (4–8 Hz) set a slow clock; gamma bursts (30–80 Hz) ride on particular phases of it. This coordination, <i>phase-amplitude coupling</i>, supports working memory and attention, and it weakens around glioma. A scalp electrode can pick that up while the mass is still too small for an MRI. The model asks a screening question: is this recording's coupling pattern, together with eye-movement, gait and speech scores, closer to normal, early disruption or high risk?</p></section>

      <section className="section"><h2>Measuring coupling</h2>
        <p>For each channel and 2-second epoch, theta phase φ comes from the Hilbert transform of the band-passed signal and gamma amplitude A from the envelope. The phase circle is cut into N = 18 bins and the mean amplitude in bin j is normalised to a distribution P. The Modulation Index (Tort et al., 2010) is the Kullback–Leibler divergence from uniform, scaled to [0, 1]:</p>
        <div className="eq">MI = D<sub>KL</sub>(P ‖ U) / log N = ( log N − H(P) ) / log N, &nbsp; P(j) = ⟨A⟩<sub>φ∈j</sub> / Σ<sub>k</sub> ⟨A⟩<sub>φ∈k</sub></div>
        <p style={{ marginTop: 12 }}>Filters are 4th-order Butterworth band-passes applied forward and backward (zero phase). Epochs whose raw amplitude exceeds ±150 µV are flagged and their MI replaced by linear interpolation from neighbouring epochs. Two pairs are measured, theta→gamma and alpha(8–13)→beta(13–30), giving 19 × 15 × 2 = 570 values per subject.</p></section>

      <section className="section"><h2>The Transformer</h2>
        <Panel><Architecture /></Panel>
        <p style={{ marginTop: 14 }}>Each epoch becomes a token of 41 numbers: 19 theta–gamma MI, 19 alpha–beta MI and the three behavioural scores. Learnable positions and a [CLS] token are added; four pre-LayerNorm encoder blocks with 8 attention heads (d<sub>k</sub> = 16), a 256-unit GELU feed-forward layer and dropout 0.1 follow. The [CLS] output goes through 128→64→3. Training uses Adam (3·10⁻⁴, weight decay 10⁻⁵), cosine decay to 10⁻⁶, batch 32, up to 80 epochs, early stopping with patience 10.</p>
        <div className="callout"><p><b>Two paper details resolved.</b> The text says a 573-value vector is reshaped into 15 tokens of 38, but 15 × 38 = 570. We keep the 570 coupling values as 15 × 38 and broadcast the three behavioural scores onto every token (15 × 41). The paper's "≈412,000 parameters" matches three blocks; with the stated four blocks the model has about 545,000, and that is what we train.</p></div></section>

      <section className="section"><h2>Virtual patients</h2>
        <p>No public dataset pairs early-tumour EEG with behavioural scores, so each subject is synthesised: a 1/f background plus delta, theta, alpha, beta and gamma components made by spectral synthesis, white sensor noise, and explicit coupling in which gamma amplitude is multiplied by 1 + m·cos φ<sub>θ</sub> (and beta by the alpha phase). Disruption lowers m in affected channels. The paper's coupling parameter (0.18) is multiplied by a documented gain of 3.5 so that the <i>measured</i> MI sits in a physiological range; at the literal 0.18 the estimator's noise floor would swallow the effect.</p>
        {P && <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Setting</th><th>Paper spec</th><th>Hard mode</th></tr></thead><tbody>
          {([["Latent severity jitter (SD)", "latent_sd"], ["Disruption varies across epochs", "temporal"], ["Behavioural SD multiplier", "beh_sd_mult"], ["Shared fatigue/age confounder (SD)", "fatigue_sd"], ["Electrode gain variability (SD)", "impedance_sd"], ["Artefact rate per epoch", "artifact_rate"], ["White noise level", "white_sd"]] as [string, string][]).map(([l, k]) => <tr key={k}><td>{l}</td><td>{String(P.paper[k])}</td><td>{String(P.hard[k])}</td></tr>)}
          <tr><td>Channels affected (Early / High Risk)</td><td>5 / 10</td><td>5 / 10, scaled by severity</td></tr>
          <tr><td>Theta–gamma depth multiplier (Early / High)</td><td>0.70 / 0.30–0.45</td><td>same, scaled by severity</td></tr></tbody></table></div>}
        <p style={{ marginTop: 12 }}>Hard mode exists because the paper's own parameters make the behavioural scores almost perfectly separable, which no real screening cohort would be. The choice of which channels, how many, and what the alpha–beta and high-risk extents are is documented in <code>research/neuroai/config.py</code>; the paper leaves several unspecified.</p></section>

      <section className="section"><h2>Making it real, and checking it</h2>
        <ul>
          <li><b>One implementation, two languages.</b> The generator, filters, Hilbert transform, Modulation Index and the Transformer forward pass exist in Python (training) and TypeScript (this app). The filter coefficients are exported from SciPy; the model weights are exported as float32.</li>
          <li><b>Parity tests.</b> <code>tests/parity.test.ts</code> loads fixtures produced by the Python code and requires the browser pipeline to reproduce the Modulation Index and the network's logits and attention within float tolerance.</li>
          <li><b>No hand-made figures.</b> Training curves, attention, ROC, confusion matrices and importance maps are computed from the trained models and written to JSON; the interface only draws them.</li>
          <li><b>Uncertainty.</b> Three independent splits and initialisations, bootstrap intervals on the headline split, expected calibration error, noise and distribution-shift tests.</li>
        </ul></section>

      <section className="section"><h2>Limitations</h2>
        <ul>
          <li>All data are simulated under the authors' assumptions; results measure learnability, not clinical accuracy.</li>
          <li>Scalp EEG blurs the intracranial coupling effects reported in the glioma literature; whether the signal survives is an open empirical question.</li>
          <li>Coupling changes also occur in epilepsy, anxiety, sleep loss and with alcohol, so specificity for tumours is untested.</li>
          <li>Ground truth would require following people for years before diagnosis. The Temple University Hospital EEG Corpus is the natural first real-data check.</li>
        </ul></section>
    </div>
  );
}
