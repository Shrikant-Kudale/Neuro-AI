import { useMemo, useState } from "react";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { Panel, LineChart } from "../components/charts";
import { totalForAUC, halfWidth, totalForSensSpec, inflate } from "../lib/planner";

function Num({ label, value, set, min, max, step, suffix = "" }: { label: string; value: number; set: (v: number) => void; min: number; max: number; step: number; suffix?: string }) {
  return <label className="fld"><span>{label}<span className="val">{suffix === "%" ? (value * 100).toFixed(0) + "%" : value.toFixed(step < 0.01 ? 3 : 2)}</span></span><input type="range" min={min} max={max} step={step} value={value} onChange={(e) => set(+e.target.value)} /></label>;
}

export default function Planner() {
  const [auc, setAuc] = useState(0.8);
  const [prev, setPrev] = useState(0.2);
  const [hw, setHw] = useState(0.07);
  const [sens, setSens] = useState(0.8);
  const [spec, setSpec] = useState(0.8);
  const [phw, setPhw] = useState(0.1);
  const [loss, setLoss] = useState(0.15);

  const nAuc = useMemo(() => totalForAUC(auc, hw, prev), [auc, hw, prev]);
  const ss = useMemo(() => totalForSensSpec(sens, spec, phw, prev), [sens, spec, phw, prev]);
  const need = Math.max(nAuc, ss.total);
  const enrol = inflate(need, loss);
  const curveN = useMemo(() => Array.from({ length: 24 }, (_, i) => 60 + i * 40), []);
  const scen = [0.7, 0.75, 0.8, 0.85, 0.9].map((a) => ({ a, n: totalForAUC(a, hw, prev) }));

  return (
    <div className="page">
      <PageHead step="Step 6 of 7 · Plan a validation study" title="Plan a validation study">
        Simulation shows that a signature is learnable under assumptions. Showing it in people needs a prospective cohort. This planner converts an expected accuracy into the participants required, using standard diagnostic-accuracy formulas.
      </PageHead>
      <div className="grid2" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1.2fr)", alignItems: "start" }}>
        <Panel title="Study assumptions" sub="Case = participant who goes on to have an imaging-confirmed tumour within the follow-up window.">
          <div style={{ display: "grid", gap: 14 }}>
            <Num label="Expected AUC of the screening test" value={auc} set={setAuc} min={0.6} max={0.97} step={0.01} />
            <Num label="Target 95% CI half-width for the AUC" value={hw} set={setHw} min={0.02} max={0.15} step={0.005} />
            <Num label="Case prevalence in the cohort" value={prev} set={setPrev} min={0.05} max={0.5} step={0.01} suffix="%" />
            <Num label="Expected sensitivity" value={sens} set={setSens} min={0.5} max={0.99} step={0.01} suffix="%" />
            <Num label="Expected specificity" value={spec} set={setSpec} min={0.5} max={0.99} step={0.01} suffix="%" />
            <Num label="Target half-width, sensitivity / specificity" value={phw} set={setPhw} min={0.03} max={0.2} step={0.005} />
            <Num label="Expected loss (unusable EEG, drop-out)" value={loss} set={setLoss} min={0} max={0.5} step={0.01} suffix="%" />
          </div>
        </Panel>
        <div style={{ display: "grid", gap: 16 }}>
          <Panel title="Participants required">
            <div className="grid3">
              <div className="stat"><b>{nAuc}</b><span>for the AUC precision target</span></div>
              <div className="stat"><b>{ss.total}</b><span>for sensitivity and specificity ({ss.cases} cases, {ss.controls} controls minimum)</span></div>
              <div className="stat"><b>{enrol}</b><span>to enrol, allowing {(loss * 100).toFixed(0)}% loss</span></div>
            </div>
            <p className="cap"><b>Result.</b> Enrol about {enrol} participants ({Math.round(enrol * prev)} expected cases) to meet both targets. Because tumours are rare in a screening population, the number of cases, not the total, is usually the limit; case-enriched designs (for example patients with persistent headaches or seizures) reduce the burden.</p>
          </Panel>
          <Panel title="Precision of the AUC estimate" sub="Hanley–McNeil standard error, at the AUC and prevalence above.">
            <LineChart x={curveN} series={[{ name: "95% CI half-width", y: curveN.map((n) => halfWidth(auc, n, prev)), color: "var(--signal)" }]} yMin={0} xLabel="total participants analysed" yLabel="half-width" xFmt={(v) => String(v)} />
            <p className="cap"><b>Figure 7.</b> Precision improves with the square root of the sample size.</p>
          </Panel>
          <Panel title="Sensitivity to the expected AUC" sub={`Total participants for a ±${hw.toFixed(3)} interval at ${(prev * 100).toFixed(0)}% prevalence.`}>
            <table className="tbl"><thead><tr><th>Expected AUC</th><th>Total N</th><th>Expected cases</th></tr></thead><tbody>{scen.map((s) => <tr key={s.a} className={s.a === Math.round(auc * 20) / 20 ? "hl" : ""}><td>{s.a.toFixed(2)}</td><td>{s.n}</td><td>{Math.round(s.n * prev)}</td></tr>)}</tbody></table>
          </Panel>
        </div>
      </div>
      <details style={{ marginTop: 20 }}><summary>Formulas and references</summary>
        <p style={{ marginTop: 10 }}>AUC variance (Hanley &amp; McNeil, 1982) with Q₁ = A/(2−A), Q₂ = 2A²/(1+A): Var(A) = [A(1−A) + (n₊−1)(Q₁−A²) + (n₋−1)(Q₂−A²)] / (n₊n₋). The smallest N such that 1.96·SE(A) is within the target is reported. Sensitivity and specificity sample sizes follow Buderer (1996): n = z²·p(1−p)/d², divided by prevalence (cases) or by 1−prevalence (controls), whichever is larger. These are planning approximations; consult a statistician before protocol submission.</p>
        <p>Suggested design: prospective, consecutive recruitment from a neuro-oncology or headache clinic; resting EEG with a standard 19-channel montage and a fixed protocol; behavioural measures acquired blind to EEG; reference standard is MRI follow-up at a pre-specified interval; analyse the pre-registered model with locked parameters, report calibration and specificity against epilepsy, anxiety and sleep-deprivation controls.</p>
      </details>
      <StepNav prev={["/results", "Review the evidence"]} next={["/method", "Read the method"]} />
    </div>
  );
}
