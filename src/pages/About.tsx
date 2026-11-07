import { Panel } from "../components/charts";

const REFS = [
  "R. Stupp et al., Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma, N Engl J Med 352(10), 2005.",
  "R. T. Canolty and R. T. Knight, The functional role of cross-frequency coupling, Trends Cogn Sci 14(11), 2010.",
  "L. Bao et al., Disrupted gamma and theta oscillatory coupling in glioma patients, NeuroImage: Clinical 28, 2020.",
  "A. B. Tort et al., Measuring phase-amplitude coupling between neuronal oscillations of different frequencies, J Neurophysiol 104(2), 2010.",
  "A. Vaswani et al., Attention is all you need, NeurIPS 30, 2017.",
  "V. J. Lawhern et al., EEGNet: a compact convolutional neural network for EEG-based BCIs, J Neural Eng 15(5), 2018.",
  "I. Obeid and J. Picone, The Temple University Hospital EEG Data Corpus, Front Neurosci 10, 2016.",
  "S. Abnar and W. Zuidema, Quantifying attention flow in Transformers, ACL 2020.",
];

export default function About() {
  return (
    <div className="page" style={{ maxWidth: 940 }}>
      <h1 style={{ fontSize: "clamp(1.9rem,3.6vw,2.8rem)" }}>The paper and the team</h1>
      <Panel title="Neuro-Cognitive AI Fingerprinting for Pre-Radiographic Brain Tumor Detection Using Transformer-Based Phase-Amplitude Coupling Analysis">
        <p>Shrikant Kudale, Rehan Surani, Seema Saud and Jayashree Prasad. Department of Computer Science and Engineering, MIT School of Computing, MIT-ADT University, Pune. <i>International Journal of Advanced Research in Science, Communication and Technology</i> 6(17), April 2026, pp. 314–319. DOI <a href="https://doi.org/10.48175/IJARSCT-34249">10.48175/IJARSCT-34249</a>.</p>
        <p>Source code: <a href="https://github.com/Shrikant-Kudale/Neuro-AI">github.com/Shrikant-Kudale/Neuro-AI</a>.</p>
      </Panel>
      <section className="section"><h2>Responsible use</h2>
        <div className="callout bad"><p><b>This is a research prototype, not a medical device.</b> It must not be used to screen, diagnose or reassure anyone. A real tool would need prospective clinical data, specificity against epilepsy and other conditions, regulatory approval and clinical oversight.</p></div>
        <p>The interface never asks for personal data. Everything, including the model, runs on your device.</p></section>
      <section className="section"><h2>References</h2><ol>{REFS.map((r) => <li key={r} style={{ marginBottom: 6 }}>{r}</li>)}</ol></section>
    </div>
  );
}
