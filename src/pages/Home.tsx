import { Link } from "react-router-dom";
import PacScope from "../components/PacScope";
import { Panel } from "../components/charts";
import ProfileSwitch from "../components/ProfileSwitch";
import { useProfile } from "../lib/profile";
import { useResults, pct, CLASS_NAMES } from "../lib/data";

export default function Home() {
  const { profile } = useProfile();
  const { data: r } = useResults(profile);
  const tf = r?.models?.["Transformer (ours)"];
  const best = r ? Object.entries<any>(r.models).sort((a, b) => b[1].across_seeds.accuracy.mean - a[1].across_seeds.accuracy.mean)[0] : null;
  return (
    <div className="page">
      <section className="hero">
        <div>
          <h1>Hear the rhythm break before the scan shows the mass.</h1>
          <p className="lede">A growing brain tumour disturbs how fast and slow brain waves coordinate long before an MRI can see it. This lab simulates that disturbance, measures it as phase-amplitude coupling, and lets a Transformer explain what it sees.</p>
          <div className="cta">
            <Link className="btn" to="/lab">Open the patient lab</Link>
            <Link className="btn ghost" to="/method">How it works</Link>
          </div>
        </div>
        <PacScope />
      </section>

      <section className="section">
        <h2>What you can do here</h2>
        <p className="lede">Everything runs in your browser. The EEG generator, the coupling analysis and the trained network are all executed locally, so nothing is uploaded.</p>
        <div className="grid3" style={{ marginTop: 18 }}>
          <Panel title="Build a virtual patient"><p>Choose how much theta–gamma coupling is lost, where on the scalp, and how the eye-tracking, gait and speech scores look. The lab synthesises a 30-second, 19-channel recording.</p><Link to="/lab">Go to the patient lab</Link></Panel>
          <Panel title="See what the model relied on"><p>Attention over the 15 two-second epochs and channel-level occlusion maps come from the trained network itself, computed on the recording you just generated.</p><Link to="/lab">See the explanation</Link></Panel>
          <Panel title="Audit the evidence"><p>Browse the 900-subject cohort, compare the Transformer with five baselines, and read how ablations, noise and unseen lesion geometry change the result.</p><Link to="/results">Read the results</Link></Panel>
        </div>
      </section>

      <section className="section">
        <div style={{ display: "flex", alignItems: "end", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div><h2>Measured on the held-out test set</h2><p className="lede" style={{ marginBottom: 0 }}>Numbers are produced by the code in this repository, not typed in.</p></div>
          <ProfileSwitch />
        </div>
        <div className="grid4" style={{ marginTop: 18 }}>
          {tf ? <>
            <Panel><div className="stat"><b>{pct(tf.across_seeds.accuracy.mean)}</b><span>Transformer accuracy, mean of {tf.across_seeds.accuracy.values.length} independent splits (SD {pct(tf.across_seeds.accuracy.sd)})</span></div></Panel>
            <Panel><div className="stat"><b>{tf.headline.auc_macro.toFixed(3)}</b><span>Macro AUC-ROC, one-vs-rest (95% CI {tf.headline.ci95.auc[0].toFixed(2)}–{tf.headline.ci95.auc[1].toFixed(2)})</span></div></Panel>
            <Panel><div className="stat"><b>{(tf.params / 1000).toFixed(0)}k</b><span>trainable parameters, runs on a laptop CPU</span></div></Panel>
            <Panel><div className="stat"><b>{r.n_subjects}</b><span>virtual subjects, {CLASS_NAMES.length} balanced classes, 19 channels × 30 s</span></div></Panel>
          </> : <Panel><p>Loading results…</p></Panel>}
        </div>
        {r && best && (
          <div className={"callout " + (best[0] !== "Transformer (ours)" ? "warn" : "")}>
            <p><b>{profile === "paper" ? "Paper-spec simulation." : "Hard-mode simulation."}</b> {r.description} {best[0] !== "Transformer (ours)" ? `On this profile the best model by accuracy is ${best[0]} (${pct(best[1].across_seeds.accuracy.mean)}), not the Transformer. We report that as measured.` : "Best model by accuracy on this profile: the Transformer."} Both profiles are synthetic; see the <Link to="/results">Results</Link> page for the comparison with the published figures.</p>
          </div>
        )}
      </section>

      <section className="section">
        <h2>From electrode to risk estimate</h2>
        <div className="pipeline" style={{ marginTop: 16 }}>
          {[["1 · Simulate", "19-channel, 256 Hz EEG with theta→gamma and alpha→beta coupling, plus three behavioural scores."],
            ["2 · Clean", "4th-order Butterworth band-pass, 15 two-second epochs, ±150 µV artefact rejection."],
            ["3 · Measure", "Hilbert phase and amplitude, then Tort's Modulation Index on 18 phase bins, per channel and epoch."],
            ["4 · Fuse", "570 coupling values plus ETVS, GAI and SSCS become 15 tokens of 41 features."],
            ["5 · Classify", "A 4-block Transformer encoder with a [CLS] token outputs Normal, Early Disruption or High Risk."]].map(([h, t]) => (
            <div key={h}><h4>{h}</h4><p>{t}</p></div>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="callout warn">
          <p><b>What this is not.</b> No patient data was used. The recordings are generated from assumptions about how tumours alter coupling, so a good score shows that the idea is computationally learnable, not that it works on people. Real validation needs longitudinal clinical EEG, and PAC changes also occur in epilepsy, anxiety and sleep loss.</p>
        </div>
      </section>
    </div>
  );
}
