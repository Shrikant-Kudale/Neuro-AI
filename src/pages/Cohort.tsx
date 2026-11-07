import { useMemo, useState } from "react";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { useAsync, getJSON, getBuffer, useManifest, useModel, useResults, PROFILE, CLASS_HEX, CLASS_NAMES, BEH_NAMES, argmax } from "../lib/data";
import { Panel, Legend, Scalp, RiskBars, BarChart, useTooltip, ColorBar } from "../components/charts";
import { tokensFromVector } from "../lib/pac";

export default function Cohort() {
  const profile = PROFILE;
  const { data: man } = useManifest();
  const { data: coh, error } = useAsync<any>(() => getJSON(`/data/cohort_${profile}.json`), [profile]);
  const { data: res } = useResults(profile);
  const { data: model } = useModel(profile);
  const [sel, setSel] = useState<number | null>(null);
  const [show, setShow] = useState([true, true, true]);
  const [splitF, setSplitF] = useState<"all" | "test">("all");
  const [live, setLive] = useState<{ probs: number[]; cls: number[] } | null>(null);
  const tip = useTooltip();
  const cfg = man?.dsp as any;

  const pts = useMemo(() => {
    if (!coh) return null;
    const S = coh.subjects as any[];
    const xs = S.map((s) => s.pc[0]), ys = S.map((s) => s.pc[1]);
    return { S, x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }, [coh]);

  const choose = async (id: number) => {
    setSel(id); setLive(null);
    if (!model) return;
    const buf = await getBuffer(`/data/cohort_${profile}.f32`);
    const row = new Float32Array(buf, id * 573 * 4, 573);
    const { X } = tokensFromVector(row);
    const p = model.predictRaw(X);
    setLive({ probs: p.probs, cls: p.clsAttn });
  };

  if (error) return <div className="page"><div className="callout bad"><p>Cohort data not found ({error}). Run the research pipeline first.</p></div></div>;
  if (!pts || !cfg) return <div className="page"><p>Loading cohort…</p></div>;
  const W = 640, H = 460, P = 28;
  const px = (v: number) => P + ((v - pts.x0) / (pts.x1 - pts.x0)) * (W - 2 * P), py = (v: number) => H - P - ((v - pts.y0) / (pts.y1 - pts.y0)) * (H - 2 * P);
  const s = sel !== null ? pts.S[sel] : null;
  const acc = (() => { const t = pts.S.filter((x) => x.split === 2); return t.filter((x) => argmax(x.p) === x.y).length / t.length; })();

  return (
    <div className="page">
      <PageHead step="Step 4 of 7 · Cohort" title="The simulated cohort">
        900 virtual subjects (300 per class) were generated and split 70/15/15 into training, validation and test sets. Each point is one subject, projected to two dimensions by PCA of its 573 fused features. Select a subject to re-run the classifier on it in your browser.
      </PageHead>
      <div className="grid2" style={{ gridTemplateColumns: "minmax(0,1.5fr) minmax(0,1fr)", marginTop: 24, alignItems: "start" }}>
        <Panel title="Feature space (PCA)" sub={`PC1 ${(coh.pca_var[0] * 100).toFixed(0)}% and PC2 ${(coh.pca_var[1] * 100).toFixed(0)}% of variance. Outlined points are held-out test subjects. Test accuracy of the stored predictions: ${(acc * 100).toFixed(1)}%.`}>
          <div className="chart"><svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="PCA scatter of the cohort">
            <rect x="0" y="0" width={W} height={H} fill="none" />
            {pts.S.map((d) => (show[d.y] && (splitF === "all" || d.split === 2)) && (
              <circle key={d.id} cx={px(d.pc[0])} cy={py(d.pc[1])} r={sel === d.id ? 7 : 4.2} fill={CLASS_HEX[d.y]} fillOpacity={0.62} stroke={d.split === 2 ? "var(--ink)" : "none"} strokeWidth="1.2" style={{ cursor: "pointer" }}
                onClick={() => choose(d.id)} onMouseMove={(e) => tip.show(e, <>#{d.id} · {CLASS_NAMES[d.y]}<br />predicted {CLASS_NAMES[argmax(d.p)]}</>)} onMouseLeave={tip.hide} />
            ))}
          </svg></div>
          {tip.el}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
            <div className="chips">{CLASS_NAMES.map((n, i) => <button key={n} className="chip" aria-pressed={show[i]} onClick={() => setShow(show.map((v, k) => (k === i ? !v : v)))} style={show[i] ? { background: CLASS_HEX[i], borderColor: CLASS_HEX[i], color: "#fff" } : undefined}>{n}</button>)}</div>
            <div className="seg"><button aria-pressed={splitF === "all"} onClick={() => setSplitF("all")}>All 900</button><button aria-pressed={splitF === "test"} onClick={() => setSplitF("test")}>Test only</button></div>
          </div>
        </Panel>
        <Panel title={s ? `Subject #${s.id}` : "Select a subject"} sub={s ? `True class: ${CLASS_NAMES[s.y]} · split: ${["train", "validation", "test"][s.split]} · latent severity ${s.sev}` : "Click any point on the map."}>
          {s ? <>
            <p style={{ marginBottom: 6, fontSize: "0.9rem" }}><b>Stored prediction (Python, PyTorch)</b></p><RiskBars probs={s.p} compact />
            {live && <><p style={{ margin: "14px 0 6px", fontSize: "0.9rem" }}><b>Re-computed in your browser (TypeScript)</b></p><RiskBars probs={live.probs} compact />
              <p style={{ fontSize: "0.8rem", color: "var(--ink-3)", marginTop: 6 }}>Largest difference: {(Math.max(...live.probs.map((v, i) => Math.abs(v - s.p[i]))) * 100).toFixed(2)} percentage points.</p></>}
            <p style={{ margin: "14px 0 4px", fontSize: "0.9rem" }}><b>Behavioural scores</b>: {BEH_NAMES.map((n, j) => `${n} ${s.beh[j]}`).join(" · ")}</p>
            <Scalp size={200} values={s.tg} coords={cfg.coords} names={cfg.channels} unit=" MI" /><ColorBar lo={Math.min(...s.tg)} hi={Math.max(...s.tg)} fmt={(v) => v.toFixed(4)} />
            <p style={{ fontSize: "0.82rem", color: "var(--ink-3)" }}>Theta–gamma MI averaged over epochs.</p>
          </> : <p style={{ color: "var(--ink-3)" }}>The browser model reproduces the stored PyTorch probabilities to within rounding; this agreement is tested automatically.</p>}
        </Panel>
      </div>
      {res && <div className="grid2" style={{ marginTop: 18 }}>
        <Panel title="Behavioural scores by class" sub="Mean of each score relative to the Normal class.">
          <BarChart labels={BEH_NAMES as unknown as string[]} series={CLASS_NAMES.map((n, i) => ({ name: n, values: BEH_NAMES.map((_, j) => res.class_stats[n].beh_mean[j] / res.class_stats.Normal.beh_mean[j]), color: CLASS_HEX[i] }))} fmt={(v) => v.toFixed(1) + "×"} height={220} />
          <Legend />
          <p style={{ fontSize: "0.85rem" }}>Shown relative to the Normal mean. Within-class SD as a fraction of the mean: {BEH_NAMES.map((n, j) => `${n} ${(res.class_stats.Normal.beh_sd[j] / res.class_stats.Normal.beh_mean[j] * 100).toFixed(0)}%`).join(", ")} (Normal class).</p>
        </Panel>
        <Panel title="Theta–gamma coupling by site and class" sub="Mean Modulation Index per electrode across the whole cohort.">
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "space-around" }}>
            {CLASS_NAMES.map((n) => <div key={n} style={{ textAlign: "center" }}><Scalp size={150} values={res.class_stats[n].tg_by_channel} coords={cfg.coords} names={cfg.channels.map(() => "")} vmin={Math.min(...CLASS_NAMES.flatMap((m) => res.class_stats[m].tg_by_channel))} vmax={Math.max(...CLASS_NAMES.flatMap((m) => res.class_stats[m].tg_by_channel))} /><div style={{ fontSize: "0.85rem", fontWeight: 600 }}>{n}</div></div>)}
          </div>
        </Panel>
      </div>}
      <StepNav prev={["/analyse", "Analyse a recording"]} next={["/results", "Review the evidence"]} />
    </div>
  );
}
