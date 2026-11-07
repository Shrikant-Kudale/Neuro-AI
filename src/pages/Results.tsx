import { useState } from "react";
import { Link } from "react-router-dom";
import { useProfile } from "../lib/profile";
import ProfileSwitch from "../components/ProfileSwitch";
import { useManifest, useResults, CLASS_HEX, CLASS_NAMES, BEH_NAMES, pct, f3 } from "../lib/data";
import { Panel, BarChart, LineChart, Legend, ConfusionMatrix, Scalp, Heatmap, diverge, ColorBar } from "../components/charts";

const PAPER = { "Transformer (ours)": [0.874, 0.873, 0.93], "CNN baseline": [0.793, 0.79, 0.86], "MLP baseline": [0.719, 0.716, 0.79] } as Record<string, number[]>;
const PAPER_ABL = { "PAC + behavioural (full)": 0.874, "PAC only": 0.821, "Behavioural only": 0.613 } as Record<string, number>;
const MODEL_COLORS: Record<string, string> = { "Transformer (ours)": "#3d3be0", "CNN baseline": "#0e8f7e", "MLP baseline": "#cf7f00", "Logistic regression": "#7b7696", "Random forest": "#8a5a2b", "Gradient boosting": "#c41e4f" };

export default function Results() {
  const { profile } = useProfile();
  const { data: r, error } = useResults(profile);
  const { data: man } = useManifest();
  const [rocCls, setRocCls] = useState(2);
  if (error) return <div className="page"><div className="callout bad"><p>Results not found ({error}). Run the pipeline in <code>research/</code>.</p></div></div>;
  if (!r || !man) return <div className="page"><p>Loading results…</p></div>;
  const cfg = man.dsp as any, names = Object.keys(r.models), tf = r.models["Transformer (ours)"].headline;
  const bestName = names.slice().sort((a, b) => r.models[b].across_seeds.accuracy.mean - r.models[a].across_seeds.accuracy.mean)[0];
  const att = r.explain.attention, occ = r.explain.occlusion, abl = r.ablation, rob = r.robustness;
  const log = r.training_log.transformer as any[];
  const ablNames = Object.keys(abl);
  const mx = Math.max(0.02, ...occ.channel_overall.map(Math.abs));

  return (
    <div className="page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "end", gap: 16, flexWrap: "wrap" }}>
        <div><h1 style={{ fontSize: "clamp(1.9rem,3.6vw,2.8rem)" }}>Results and audit</h1>
          <p className="lede" style={{ margin: "8px 0 0" }}>Everything below was produced by <code>research/run_pipeline.py</code> on {r.n_subjects} simulated subjects ({r.split.train}/{r.split.val}/{r.split.test} train/validation/test).</p></div>
        <ProfileSwitch />
      </div>

      <div className="callout warn" style={{ marginTop: 20 }}>
        <p><b>Read this before comparing with the paper.</b> The published table reports 87.4% for the Transformer and 61.3% for behavioural features alone. Re-implementing the paper's generator and running every model end to end gives the numbers on this page. {profile === "paper" ? `With the paper's own behavioural parameters the three scores are almost perfectly separable (behavioural-only ${pct(abl["Behavioural only"].accuracy)}), so the 61.3% ablation cannot arise from that generator.` : "Hard mode adds the class overlap and confounders the paper lists as missing, which brings behavioural-only performance down to a realistic level."} The generator settings and choices are documented on the <Link to="/method">Method</Link> page.</p>
      </div>

      <section className="section">
        <h2>Model comparison</h2>
        <p className="lede">Mean ± SD over {r.models[names[0]].across_seeds.accuracy.values.length} independent stratified splits and initialisations. The Transformer headline split also carries bootstrap 95% intervals (accuracy {pct(tf.ci95.acc[0])}–{pct(tf.ci95.acc[1])}).</p>
        <Panel>
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>AUC-ROC</th><th>ECE</th><th>Params</th><th>Paper: acc / F1 / AUC</th></tr></thead>
            <tbody>{names.map((n) => { const a = r.models[n].across_seeds; const p = PAPER[n]; return (
              <tr key={n} className={n === "Transformer (ours)" ? "hl" : ""}><td>{n}{n === bestName && <span className="tag" style={{ marginLeft: 8 }}>best</span>}</td>
                {(["accuracy", "f1", "auc_macro", "ece"] as const).map((k) => <td key={k}>{k === "accuracy" ? pct(a[k].mean) : f3(a[k].mean)} <span className="sd">±{k === "accuracy" ? pct(a[k].sd) : f3(a[k].sd)}</span></td>)}
                <td>{r.models[n].params ? (r.models[n].params / 1000).toFixed(0) + "k" : "—"}</td><td>{p ? `${pct(p[0])} / ${p[1].toFixed(3)} / ${p[2].toFixed(2)}` : "not reported"}</td></tr>); })}</tbody>
          </table></div>
        </Panel>
        {bestName !== "Transformer (ours)" && <div className="callout"><p>On this profile {bestName} scores at least as well as the Transformer. When class structure is mostly in a few strong features, tree ensembles are hard to beat; the Transformer's case rests on sequence structure, which only matters when disruption varies over time (see the epoch-shuffle test below).</p></div>}
      </section>

      <section className="section">
        <div className="grid2">
          <Panel title="Per-class performance (Transformer)" sub={`Held-out test set, n = ${tf.n_test}`}>
            <div className="tbl-wrap"><table className="tbl"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC</th></tr></thead><tbody>
              {CLASS_NAMES.map((n, i) => { const c = tf.per_class[n]; return <tr key={n}><td>{n}</td><td>{c.precision.toFixed(2)}</td><td>{c.recall.toFixed(2)}</td><td>{c.f1.toFixed(2)}</td><td>{tf.auc_per_class[i].toFixed(3)}</td></tr>; })}</tbody></table></div>
            <ConfusionMatrix cm={tf.confusion} />
          </Panel>
          <Panel title="ROC curves" sub="One-vs-rest, Transformer.">
            <div className="chips" style={{ marginBottom: 8 }}>{CLASS_NAMES.map((n, i) => <button key={n} className="chip" aria-pressed={rocCls === i} onClick={() => setRocCls(i)}>{n}</button>)}</div>
            <LineChart x={tf.roc[CLASS_NAMES[rocCls]].fpr} series={names.filter((n) => r.models[n].headline.roc).map((n) => ({ name: n, y: r.models[n].headline.roc[CLASS_NAMES[rocCls]].tpr, color: MODEL_COLORS[n], w: n === "Transformer (ours)" ? 3 : 1.6 }))} diag yMax={1} xLabel="false-positive rate" yLabel="true-positive rate" xFmt={(v) => v.toFixed(1)} />
            <Legend items={names.map((n) => ({ label: n, color: MODEL_COLORS[n] }))} />
          </Panel>
        </div>
      </section>

      <section className="section">
        <h2>Which signals carry the information?</h2>
        <div className="grid2" style={{ marginTop: 14 }}>
          <Panel title="Ablation: retrained Transformer on feature subsets" sub="Single split. Dashed markers show the paper's reported full / PAC-only / behavioural-only values.">
            <BarChart labels={ablNames} series={[{ name: "this repo", values: ablNames.map((n) => abl[n].accuracy), color: "var(--signal)" }, { name: "paper", values: ablNames.map((n) => PAPER_ABL[n] ?? 0), color: "var(--ink-3)" }]} max={1} fmt={(v) => (v * 100).toFixed(0) + "%"} height={250} />
            <Legend items={[{ label: "measured here", color: "var(--signal)" }, { label: "reported in paper", color: "var(--ink-3)" }]} />
          </Panel>
          <Panel title="Early Disruption F1 by feature set" sub="The hardest class and the clinically relevant one.">
            <BarChart labels={ablNames} series={[{ name: "Early F1", values: ablNames.map((n) => abl[n].f1_early), color: "var(--early)" }]} max={1} height={250} />
          </Panel>
        </div>
      </section>

      <section className="section">
        <h2>Robustness and generalisation</h2>
        <div className="grid3" style={{ marginTop: 14 }}>
          <Panel title="Sensor noise" sub="Same trained model, test cohorts regenerated with more white noise.">
            <LineChart x={rob.noise.map((n: any) => n.noise_mult)} series={[{ name: "accuracy", y: rob.noise.map((n: any) => n.accuracy), color: "var(--signal)" }, { name: "Early F1", y: rob.noise.map((n: any) => n.f1_early), color: "var(--early)" }]} yMin={0} yMax={1} logX points xLabel="noise multiplier" xFmt={(v) => "×" + v} />
            <Legend items={[{ label: "accuracy", color: "var(--signal)" }, { label: "Early F1", color: "var(--early)" }]} />
          </Panel>
          <Panel title="Unseen lesion geometry" sub="Trained on scattered channels, tested on focal scalp lesions."><div className="stat"><b>{pct(rob.focal_lesion.accuracy)}</b><span>accuracy (vs {pct(rob.baseline_accuracy)} in-distribution), Early F1 {rob.focal_lesion.f1_early.toFixed(2)}</span></div></Panel>
          <Panel title="Epoch order shuffled" sub="Tokens permuted at test time. A model using temporal order should degrade."><div className="stat"><b>{pct(rob.epoch_shuffled_test.accuracy)}</b><span>accuracy (vs {pct(rob.baseline_accuracy)} unshuffled)</span></div></Panel>
        </div>
      </section>

      <section className="section">
        <h2>Explainability from the trained network</h2>
        <div className="grid2" style={{ marginTop: 14 }}>
          <Panel title="Attention across the 15 epochs" sub={`Mean [CLS] attention by true class. Share on epochs 6–9: ${CLASS_NAMES.map((n) => `${n} ${pct(att.share_tokens_6_9[n], 0)}`).join(", ")} (uniform = ${pct(att.uniform_share_6_9, 0)}).`}>
            <LineChart x={Array.from({ length: 15 }, (_, i) => i + 1)} series={CLASS_NAMES.map((n, i) => ({ name: n, y: att.by_true[n].cls_to_epoch, color: CLASS_HEX[i] }))} yMin={0} xLabel="epoch (2 s each)" points />
            <Legend />
            <p style={{ fontSize: "0.88rem" }}>{profile === "paper" ? "The paper reports attention concentrating on epochs 6–9. In the paper-spec generator disruption is constant across the recording, so there is nothing temporal to attend to and attention stays close to uniform." : "In hard mode disruption fluctuates across epochs, so the model has a reason to weigh epochs unequally."}</p>
          </Panel>
          <Panel title="Channel importance (occlusion)" sub="Mean drop in true-class probability when a site's coupling is replaced by a training average, over all test subjects.">
            <Scalp values={occ.channel_overall} coords={cfg.coords} names={cfg.channels} color={diverge} vmin={-mx} vmax={mx} size={260} unit=" Δp" /><ColorBar lo={-mx} hi={mx} color={(t) => diverge(2 * t - 1)} fmt={(v) => v.toFixed(3)} />
            <p style={{ fontSize: "0.88rem", marginTop: 8 }}>Feature groups (importance for High Risk): theta–gamma {occ.groups.theta_gamma[2].toFixed(3)}, alpha–beta {occ.groups.alpha_beta[2].toFixed(3)}, behavioural {occ.groups.behavioural[2].toFixed(3)}. Behavioural importance per score (High Risk): {BEH_NAMES.map((n, j) => `${n} ${occ.behavioural[2][j].toFixed(3)}`).join(", ")}.</p>
          </Panel>
        </div>
        <div className="grid2" style={{ marginTop: 18 }}>
          <Panel title="Attention matrix, High Risk (last layer)" sub="Rows attend to columns; index 0 is the [CLS] token."><Heatmap data={att.by_true["High Risk"].matrix} rows={Array.from({ length: 16 }, (_, i) => (i ? String(i) : "CLS"))} cols={Array.from({ length: 16 }, (_, i) => (i ? String(i) : "C"))} cell={20} vmin={0} /></Panel>
          <Panel title="Training curves (real)" sub={`Transformer, ${log.length} epochs before early stopping; learning rate on cosine schedule.`}>
            <LineChart x={log.map((l: any) => l.epoch)} series={[{ name: "train loss", y: log.map((l: any) => l.train_loss), color: "var(--signal)" }, { name: "validation loss", y: log.map((l: any) => l.val_loss), color: "var(--high)" }]} yMin={0} xLabel="epoch" yLabel="cross-entropy" />
            <Legend items={[{ label: "train", color: "var(--signal)" }, { label: "validation", color: "var(--high)" }]} />
          </Panel>
        </div>
      </section>
    </div>
  );
}
