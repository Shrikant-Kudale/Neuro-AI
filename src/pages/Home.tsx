import { Link } from "react-router-dom";
import PacScope from "../components/PacScope";
import StepNav from "../components/StepNav";
import { useResults, pct, PROFILE } from "../lib/data";

export default function Home() {
  const { data: r } = useResults(PROFILE);
  const tf = r?.models?.["Transformer"]?.across_seeds;
  return (
    <div className="page">
      <section className="hero">
        <div>
          <div className="crumb">Step 1 of 7 · Overview</div>
          <h1>Measuring how brain rhythms coordinate, from EEG.</h1>
          <p className="lede">NeuroPAC is a research workbench for <b>phase-amplitude coupling (PAC)</b>: the way the phase of slow theta waves gates the strength of fast gamma activity. Coupling is altered in several neurological conditions, including glioma. The workbench lets you analyse EEG recordings, test methods on simulations with known ground truth, and plan the study that real validation requires.</p>
          <div className="cta"><Link className="btn" to="/analyse">Analyse a recording</Link><Link className="btn ghost" to="/lab">Explore with simulated patients</Link></div>
        </div>
        <PacScope />
      </section>

      <section className="section">
        <h2>How to use this workbench</h2>
        <p className="lede" style={{ marginBottom: 16 }}>The pages follow the order in which a research question is usually worked through. Each page ends with a link to the next step.</p>
        <div className="flow">
          {[["Step 2", "Simulate", "Build a virtual subject with a chosen amount and location of coupling loss. See the EEG, the coupling analysis and how a classifier responds, with an explanation.", "/lab"],
            ["Step 3", "Analyse a recording", "Load your own EEG (EDF or CSV). Get theta–gamma and alpha–beta coupling for every channel and epoch, tested against surrogates, with an exportable table.", "/analyse"],
            ["Step 4", "Cohort", "Browse the 900 simulated subjects used for training and testing, and re-run the model on any of them.", "/cohort"],
            ["Step 5", "Evidence", "Read the benchmark: six models, repeated splits, calibration, ablations, noise and unseen-lesion tests.", "/results"],
            ["Step 6", "Plan a study", "Turn an expected accuracy into the number of participants a prospective validation study needs.", "/planner"],
            ["Step 7", "Method", "Equations, architecture, simulation specification, validation of the software and limitations.", "/method"]].map(([n, h, t, to]) => (
            <Link key={h} to={to} className="card"><div className="num">{n}</div><h3>{h}</h3><p>{t}</p><span className="go">Open</span></Link>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>What it is for</h2>
        <div className="uses" style={{ marginTop: 14 }}>
          <div><h3>Analysing real EEG privately</h3><p>The coupling pipeline (Butterworth band-pass, Hilbert phase and envelope, Tort modulation index, surrogate testing) runs on your files inside the browser, with a CSV export for statistics elsewhere.</p></div>
          <div><h3>Testing methods against ground truth</h3><p>Where the true coupling is known, you can measure how noise, artefacts, lesion geometry and recording length change an estimator. That is not possible with patient data alone.</p></div>
          <div><h3>Planning the validation study</h3><p>No public dataset pairs early-tumour EEG with behavioural measures. The planner converts an expected AUC or sensitivity into the cases and controls a prospective cohort needs.</p></div>
          <div><h3>Teaching cross-frequency coupling</h3><p>Interactive figures show how coupling appears in the raw trace, in the phase-amplitude histogram and in the modulation index.</p></div>
        </div>
        <div className="callout warn" style={{ marginTop: 18 }}>
          <p><b>What it is not.</b> It is not a diagnostic tool. The simulated cohorts encode assumptions about how tumours alter coupling, and the classifier has only ever seen those simulations{tf ? ` (accuracy ${pct(tf.accuracy.mean)} on held-out simulated subjects)` : ""}. Coupling also changes in epilepsy, anxiety and sleep loss, so any real use needs prospective clinical data.</p>
        </div>
      </section>
      <StepNav next={["/lab", "Simulate a patient"]} />
    </div>
  );
}
