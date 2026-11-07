import { useState } from "react";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { useManifest, useResults, PROFILE, CLASS_HEX, CLASS_NAMES, BEH_NAMES, pct, f3 } from "../lib/data";
import { Panel, BarChart, LineChart, Legend, ConfusionMatrix, Scalp, Heatmap, diverge, ColorBar } from "../components/charts";

const MODEL_COLORS: Record<string, string> = { Transformer: "#0f3b5c", "1-D CNN": "#2f7d4f", MLP: "#a8690f", "Logistic regression": "#6a7a8c", "Random forest": "#8a5a2b", "Gradient boosting": "#b3362f" };

export default function Results() {
  const { data: r, error } = useResults(PROFILE);
  const { data: man } = useManifest();
  const [rocCls, setRocCls] = useState(2);
  if (error) return <div className="page"><div className="callout bad"><p>Results not found ({error}). Run the pipeline in <code>research/</code> and redeploy.</p></div></div>;
  if (!r || !man) return <div className="page"><p>Loading…</p></div>;
  const cfg = man.dsp as any, names = Object.keys(r.models), tf = r.models["Transformer"].headline;
  const bestName = names.slice().sort((a, b) => r.models[b].across_seeds.accuracy.mean - r.models[a].across_seeds.accuracy.mean)[0];
  const att = r.explain.attention, occ = r.explain.occlusion, abl = r.ablation, rob = r.robustness;
  const log = r.training_log.transformer as any[];
  const ablNames = Object.keys(abl);
  const mx = Math.max(0.02, ...occ.channel_overall.map(Math.abs));
  const nSeeds = r.models[names[0]].across_seeds.accuracy.values.length;

  return (
    <div className="page">
      <PageHead step="Step 5 of 7 · Evidence" title="Evidence and validation">
        Every figure on this page was computed by <code>research/run_pipeline.py</code> from {r.n_subjects} simulated subjects ({r.split.train} train, {r.split.val} validation, {r.split.test} test). It documents what the methods can do on simulated data. It is not evidence of clinical performance.
      </PageHead>

      <div className="callout warn"><p><b>How to read this page.</b> The same simulation generates the training and test data, so high accuracy shows that the signature is learnable under the stated assumptions. The ablations and robustness tests below show which assumptions the result depends on.</p></div>

      <section className="section" style={{ marginTop: 28 }}>
        <h2>Model comparison</h2>
        <p className="lede">Mean ± SD over {nSeeds} independent stratified splits and initialisations. 95% bootstrap interval for the Transformer on the primary split: accuracy {pct(tf.ci95.acc[0])}–{pct(tf.ci95.acc[1])}, macro AUC {tf.ci95.auc[0].toFixed(2)}–{tf.ci95.auc[1].toFixed(2)}.</p>
        <Panel>
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>AUC-ROC</th><th>Calibration error</th><th>Parameters</th></tr></thead>
            <tbody>{names.map((n) => { const a = r.models[n].across_seeds; return (
              <tr key={n} className={n === "Transformer" ? "hl" : ""}><td>{n}{n === bestName && <span className="tag" style={{ marginLeft: 8 }}>highest accuracy</span>}</td>
                {(["accuracy", "f1", "auc_macro", "ece"] as const).map((k) => <td key={k}>{k === "accuracy" ? pct(a[k].mean) : f3(a[k].mean)} <span className="sd">±{k === "accuracy" ? pct(a[k].sd) : f3(a[k].sd)}</span></td>)}
                <td>{r.models[n].params ? (r.models[n].params / 1000).toFixed(0) + "k" : "–"}</td></tr>); })}</tbody>
          </table></div>
          <p className="cap"><b>Table 1.</b> Held-out test performance (macro-averaged over the three classes). ECE = expected calibration error (lower is better).</p>
        </Panel>
        {bestName !== "Transformer" && <div className="callout"><p>{bestName} scores at least as well as the Transformer here. When class structure sits in a few strong features, tree ensembles are hard to beat. The Transformer is retained because it exposes epoch-level attention and handles variable disruption over time.</p></div>}
      </section>

      <section className="section">
        <div className="grid2">
          <Panel title="Per-class performance (Transformer)" sub={`Held-out test set, n = ${tf.n_test}`}>
            <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody>
              {CLASS_NAMES.map((n, i) => { const c = tf.per_class[n]; return <tr key={n}><td>{n}</td><td>{c.precision.toFixed(2)}</td><td>{c.recall.toFixed(2)}</td><td>{c.f1.toFixed(2)}</td><td>{tf.auc_per_class[i].toFixed(3)}</td></tr>; })}</tbody></table></div>
            <ConfusionMatrix cm={tf.confusion} />
            <p className="cap"><b>Figure 1.</b> Confusion matrix; rows are the true class.</p>
          </Panel>
          <Panel title="ROC curves" sub="One class versus the other two.">
            <div className="chips" style={{ marginBottom: 8 }}>{CLASS_NAMES.map((n, i) => <button key={n} className="chip" aria-pressed={rocCls === i} onClick={() => setRocCls(i)}>{n}</button>)}</div>
            <LineChart x={tf.roc[CLASS_NAMES[rocCls]].fpr} series={names.filter((n) => r.models[n].headline.roc).map((n) => ({ name: n, y: r.models[n].headline.roc[CLASS_NAMES[rocCls]].tpr, color: MODEL_COLORS[n], w: n === "Transformer" ? 3 : 1.5 }))} diag yMax={1} xLabel="false-positive rate" yLabel="true-positive rate" xFmt={(v) => v.toFixed(1)} />
            <Legend items={names.map((n) => ({ label: n, color: MODEL_COLORS[n] }))} />
            <p className="cap"><b>Figure 2.</b> Receiver operating characteristic.</p>
          </Panel>
        </div>
      </section>

      <section className="section">
        <h2>Which signals carry the information?</h2>
        <div className="grid2" style={{ marginTop: 14 }}>
          <Panel title="Ablation: Transformer retrained on feature subsets" sub="Single primary split.">
            <BarChart labels={ablNames} series={[{ name: "accuracy", values: ablNames.map((n) => abl[n].accuracy), color: "var(--navy)" }]} max={1} fmt={(v) => (v * 100).toFixed(0) + "%"} height={250} />
            <p className="cap"><b>Figure 3.</b> Accuracy with each feature subset. {abl["Behavioural only"].accuracy >= abl["PAC only"].accuracy ? "Behavioural scores are more informative than coupling alone in this simulation, and coupling adds information mainly for the intermediate class." : "Coupling features are at least as informative as behavioural scores."}</p>
          </Panel>
          <Panel title="Early Disruption F1 by feature set" sub="The intermediate class is the hardest and the most relevant to screening.">
            <BarChart labels={ablNames} series={[{ name: "Early F1", values: ablNames.map((n) => abl[n].f1_early), color: "var(--early)" }]} max={1} height={250} />
            <p className="cap"><b>Figure 4.</b> F1 for the Early Disruption class.</p>
          </Panel>
        </div>
      </section>

      <section className="section">
        <h2>Robustness and generalisation</h2>
        <div className="grid3" style={{ marginTop: 14 }}>
          <Panel title="Sensor noise" sub="Same trained model; test cohorts regenerated with more white noise.">
            <LineChart x={rob.noise.map((n: any) => n.noise_mult)} series={[{ name: "accuracy", y: rob.noise.map((n: any) => n.accuracy), color: "var(--navy)" }, { name: "Early F1", y: rob.noise.map((n: any) => n.f1_early), color: "var(--early)" }]} yMin={0} yMax={1} logX points xLabel="noise multiplier" xFmt={(v) => "×" + v} />
            <Legend items={[{ label: "accuracy", color: "var(--navy)" }, { label: "Early F1", color: "var(--early)" }]} />
          </Panel>
          <Panel title="Unseen lesion geometry" sub="Trained on scattered channels; tested on focal scalp lesions."><div className="stat"><b>{pct(rob.focal_lesion.accuracy)}</b><span>accuracy (in-distribution {pct(rob.baseline_accuracy)}); Early F1 {rob.focal_lesion.f1_early.toFixed(2)}</span></div></Panel>
          <Panel title="Epoch order shuffled" sub="Tokens permuted at test time."><div className="stat"><b>{pct(rob.epoch_shuffled_test.accuracy)}</b><span>accuracy (unshuffled {pct(rob.baseline_accuracy)}). A small drop means the model uses little temporal order.</span></div></Panel>
        </div>
      </section>

      <section className="section">
        <h2>Explainability</h2>
        <div className="grid2" style={{ marginTop: 14 }}>
          <Panel title="Attention across the 15 epochs" sub="Mean [CLS] attention by true class, last encoder layer.">
            <LineChart x={Array.from({ length: 15 }, (_, i) => i + 1)} series={CLASS_NAMES.map((n, i) => ({ name: n, y: att.by_true[n].cls_to_epoch, color: CLASS_HEX[i] }))} yMin={0} xLabel="epoch (2 s each)" points />
            <Legend />
            <p className="cap"><b>Figure 5.</b> Attention is close to uniform ({pct(att.uniform_share_6_9, 0)} share expected for epochs 6–9; measured {CLASS_NAMES.map((n) => pct(att.share_tokens_6_9[n], 0)).join(" / ")}). The model does not single out particular epochs; it uses the recording as a whole.</p>
          </Panel>
          <Panel title="Channel importance (occlusion)" sub="Mean drop in true-class probability when a site's coupling is replaced by a training average.">
            <Scalp values={occ.channel_overall} coords={cfg.coords} names={cfg.channels} color={diverge} vmin={-mx} vmax={mx} size={260} unit=" Δp" /><ColorBar lo={-mx} hi={mx} color={(t) => diverge(2 * t - 1)} fmt={(v) => v.toFixed(3)} />
            <p className="cap"><b>Figure 6.</b> Feature-group importance for High Risk: theta–gamma {occ.groups.theta_gamma[2].toFixed(3)}, alpha–beta {occ.groups.alpha_beta[2].toFixed(3)}, behavioural {occ.groups.behavioural[2].toFixed(3)}. Per score: {BEH_NAMES.map((n, j) => `${n} ${occ.behavioural[2][j].toFixed(3)}`).join(", ")}.</p>
          </Panel>
        </div>
        <div className="grid2" style={{ marginTop: 16 }}>
          <Panel title="Attention matrix, High Risk (last layer)" sub="Rows attend to columns; index 0 is the [CLS] token."><Heatmap data={att.by_true["High Risk"].matrix} rows={Array.from({ length: 16 }, (_, i) => (i ? String(i) : "CLS"))} cols={Array.from({ length: 16 }, (_, i) => (i ? String(i) : "C"))} cell={20} vmin={0} /></Panel>
          <Panel title="Training curves" sub={`Transformer, ${log.length} epochs before early stopping.`}>
            <LineChart x={log.map((l: any) => l.epoch)} series={[{ name: "train loss", y: log.map((l: any) => l.train_loss), color: "var(--navy)" }, { name: "validation loss", y: log.map((l: any) => l.val_loss), color: "var(--high)" }]} yMin={0} xLabel="epoch" yLabel="cross-entropy" />
            <Legend items={[{ label: "train", color: "var(--navy)" }, { label: "validation", color: "var(--high)" }]} />
          </Panel>
        </div>
      </section>
      <StepNav prev={["/cohort", "Browse the cohort"]} next={["/planner", "Plan a validation study"]} />
    </div>
  );
}
