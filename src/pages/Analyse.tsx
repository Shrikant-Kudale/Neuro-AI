import { useMemo, useState } from "react";
import { useManifest } from "../lib/data";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { Panel, Heatmap, Scalp, ColorBar, ramp } from "../components/charts";
import PhaseAmp from "../components/PhaseAmp";
import { parseCSV, parseEDF, analyseRecording, toCSV, RawRecording, RecordingAnalysis } from "../lib/recording";
import { RNG } from "../lib/rng";
import { sampleParams, synthesize } from "../lib/generator";

const UNITS: Record<string, number> = { "µV": 1, mV: 1e3, V: 1e6 };

export default function Analyse() {
  const { data: man } = useManifest();
  const cfg = man?.dsp as any;
  const [rec, setRec] = useState<RawRecording | null>(null);
  const [name, setName] = useState("");
  const [fsText, setFsText] = useState("");
  const [unit, setUnit] = useState("µV");
  const [inc, setInc] = useState<boolean[]>([]);
  const [start, setStart] = useState(0);
  const [dur, setDur] = useState(30);
  const [thr, setThr] = useState(150);
  const [nSur, setNSur] = useState(200);
  const [res, setRes] = useState<RecordingAnalysis | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [ch, setCh] = useState(0);
  const [ep, setEp] = useState(0);

  const load = (r: RawRecording, label: string) => { setRec(r); setName(label); setInc(r.names.map(() => true)); setRes(null); setErr(""); setCh(0); setEp(0); };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    try {
      setErr("");
      if (/\.edf$/i.test(f.name)) load(parseEDF(await f.arrayBuffer()), f.name);
      else load(parseCSV(await f.text(), fsText ? +fsText : undefined), f.name);
    } catch (e: any) { setErr(e.message); setRec(null); }
  };
  const example = (cls: number) => {
    if (!cfg) return;
    const p = sampleParams(cfg, "sim", cls, new RNG(100 + cls));
    const data = synthesize(cfg, p, new RNG(500 + cls));
    load({ names: cfg.channels, data, fs: cfg.channels.map(() => 256), unitScale: 1, source: "simulated" }, `Simulated example (${["normal", "early disruption", "high-risk"][cls]} pattern, 256 Hz)`);
  };
  const downloadExample = () => {
    if (!cfg) return;
    const p = sampleParams(cfg, "sim", 1, new RNG(101)), d = synthesize(cfg, p, new RNG(501));
    const rows = [cfg.channels.join(",")]; for (let i = 0; i < d[0].length; i++) rows.push(d.map((c) => c[i].toFixed(3)).join(","));
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" })); a.download = "example_eeg_256Hz_19ch.csv"; a.click();
  };

  const run = () => {
    if (!rec || !cfg) return;
    setBusy(true); setErr("");
    setTimeout(() => {
      try { const a = analyseRecording(rec, cfg, { startSec: start, maxSec: dur, include: inc, artifactUv: thr, nSurrogates: nSur, unit: UNITS[unit] }); setRes(a); setCh(0); setEp(0); }
      catch (e: any) { setErr(e.message); setRes(null); }
      setBusy(false);
    }, 30);
  };
  const save = () => {
    if (!res) return;
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([toCSV(res)], { type: "text/csv" })); a.download = "pac_results.csv"; a.click();
  };

  const topo = useMemo(() => {
    if (!res || !cfg) return null;
    const idx: number[] = []; res.matched.forEach((m, i) => { if (m) idx.push(i); });
    if (idx.length < 8) return null;
    const coords = idx.map((i) => cfg.coords[cfg.channels.indexOf(res.matched[i]!)]);
    return { idx, coords, names: idx.map((i) => res.matched[i]!) };
  }, [res, cfg]);

  if (!cfg) return <div className="page"><p>Loading…</p></div>;
  const E = res?.nEpochs ?? 15;
  const nRej = res ? res.flagged.flat().filter(Boolean).length : 0;
  const sig = (z: number, p: number) => (Number.isFinite(p) && p <= 0.05 ? "sig" : "");

  return (
    <div className="page">
      <PageHead step="Step 3 of 7 · Analyse a recording" title="Analyse a recording">
        Measure phase-amplitude coupling in your own EEG. The file never leaves your device. If you only want to try the tool, load an example recording first.
      </PageHead>

      <div className="grid2" style={{ alignItems: "start" }}>
        <Panel title="1. Load a recording" sub="EDF, or CSV/TSV with one column per channel and one row per sample (a header row of channel names and a time column are detected automatically).">
          <div className="drop">
            <input type="file" accept=".edf,.csv,.tsv,.txt" aria-label="EEG file" onChange={(e) => onFile(e.target.files?.[0])} />
            <p className="cap" style={{ marginTop: 8 }}>Sampling rate for CSV (Hz), if there is no time column: <input type="number" style={{ width: 90, display: "inline-block" }} value={fsText} placeholder="256" onChange={(e) => setFsText(e.target.value)} /></p>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
            <span style={{ fontSize: "0.86rem", alignSelf: "center", color: "var(--ink-2)" }}>No file? Use a simulated example:</span>
            {["Normal", "Early disruption", "High-risk"].map((n, i) => <button key={n} className="btn ghost sm" onClick={() => example(i)}>{n}</button>)}
            <button className="btn ghost sm" onClick={downloadExample}>Download example CSV</button>
          </div>
          {err && <div className="callout bad"><p>{err}</p></div>}
          {rec && <dl className="kv" style={{ marginTop: 14 }}><dt>Loaded</dt><dd>{name}</dd><dt>Channels</dt><dd>{rec.names.length}</dd><dt>Sampling rate</dt><dd>{[...new Set(rec.fs.map((f) => f.toFixed(1)))].join(", ")} Hz</dd><dt>Duration</dt><dd>{(rec.data[0].length / rec.fs[0]).toFixed(1)} s</dd></dl>}
        </Panel>

        <Panel title="2. Choose channels and window" sub="Analysis resamples to 256 Hz, band-passes (zero phase), measures the modulation index per 2-second epoch and tests each channel against circular-shift surrogates.">
          {!rec ? <p style={{ color: "var(--ink-3)" }}>Load a recording to continue.</p> : <>
            <div style={{ maxHeight: 130, overflow: "auto", border: "1px solid var(--line-2)", borderRadius: 4, padding: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {rec.names.map((n, i) => <label key={i} className="chip" style={{ display: "flex", gap: 5, alignItems: "center" }}><input type="checkbox" checked={inc[i]} onChange={(e) => setInc(inc.map((v, k) => (k === i ? e.target.checked : v)))} />{n}</label>)}
            </div>
            <div className="grid3" style={{ marginTop: 12 }}>
              <label className="fld">Amplitude unit<select value={unit} onChange={(e) => setUnit(e.target.value)}>{Object.keys(UNITS).map((u) => <option key={u}>{u}</option>)}</select></label>
              <label className="fld">Start (s)<input type="number" min={0} value={start} onChange={(e) => setStart(Math.max(0, +e.target.value))} /></label>
              <label className="fld">Length (s)<input type="number" min={4} max={300} value={dur} onChange={(e) => setDur(Math.min(300, Math.max(4, +e.target.value)))} /></label>
              <label className="fld">Artefact limit (µV)<input type="number" min={20} value={thr} onChange={(e) => setThr(+e.target.value)} /></label>
              <label className="fld">Surrogates<input type="number" min={0} max={1000} step={50} value={nSur} onChange={(e) => setNSur(Math.min(1000, Math.max(0, +e.target.value)))} /></label>
            </div>
            <div style={{ marginTop: 14 }}><button className="btn" onClick={run} disabled={busy}>{busy ? "Analysing…" : "Run analysis"}</button></div>
          </>}
        </Panel>
      </div>

      {res && <div style={{ display: "grid", gap: 16, marginTop: 20 }}>
        <Panel title="3. Results" sub={`${res.names.length} channels · ${res.nEpochs} epochs (${res.seconds} s) · ${nRej} channel-epochs above the artefact limit were excluded from channel summaries${res.resampled ? " · resampled to 256 Hz" : ""}`}>
          <div className="tbl-wrap"><table className="tbl">
            <thead><tr><th>Channel</th><th>10–20 site</th><th>θ→γ MI</th><th>z</th><th>p</th><th>α→β MI</th><th>z</th><th>p</th><th>Rejected</th></tr></thead>
            <tbody>{res.names.map((n, c) => (
              <tr key={c} onClick={() => setCh(c)} style={{ cursor: "pointer", background: c === ch ? "var(--signal-soft)" : undefined }}>
                <td>{n}</td><td>{res.matched[c] ?? "–"}</td>
                <td>{res.tg.meanMI[c].toFixed(4)}</td><td>{Number.isFinite(res.tg.z[c]) ? res.tg.z[c].toFixed(1) : "–"}</td><td style={{ fontWeight: sig(res.tg.z[c], res.tg.p[c]) ? 700 : 400 }}>{Number.isFinite(res.tg.p[c]) ? res.tg.p[c].toFixed(3) : "–"}</td>
                <td>{res.ab.meanMI[c].toFixed(4)}</td><td>{Number.isFinite(res.ab.z[c]) ? res.ab.z[c].toFixed(1) : "–"}</td><td style={{ fontWeight: sig(res.ab.z[c], res.ab.p[c]) ? 700 : 400 }}>{Number.isFinite(res.ab.p[c]) ? res.ab.p[c].toFixed(3) : "–"}</td>
                <td>{res.flagged[c].filter(Boolean).length}</td></tr>))}</tbody>
          </table></div>
          <p className="cap"><b>Table 1.</b> Modulation index (MI), z-score against {nSur} circular-shift surrogates and one-sided p-value per channel. Bold p ≤ 0.05, uncorrected for the {res.names.length * 2} tests. Click a row to inspect that channel.</p>
          <div style={{ marginTop: 10 }}><button className="btn ghost sm" onClick={save}>Download results (CSV)</button></div>
        </Panel>

        <div className="grid2">
          <Panel title="Theta → gamma MI by channel and epoch"><Heatmap data={res.tg.mi.map((r) => Array.from(r))} rows={res.names} cols={Array.from({ length: E }, (_, i) => String(i + 1))} cell={Math.max(14, Math.min(26, Math.floor(520 / E)))} colLabel="epoch (2 s)" />
            <p className="cap"><b>Figure 2.</b> Brighter cells indicate stronger coupling.</p></Panel>
          <Panel title="Alpha → beta MI by channel and epoch"><Heatmap data={res.ab.mi.map((r) => Array.from(r))} rows={res.names} cols={Array.from({ length: E }, (_, i) => String(i + 1))} cell={Math.max(14, Math.min(26, Math.floor(520 / E)))} colLabel="epoch (2 s)" />
            <p className="cap"><b>Figure 3.</b> Compensatory alpha–beta coupling is of interest alongside theta–gamma loss.</p></Panel>
        </div>

        <div className="grid2">
          <Panel title={`Phase-amplitude histogram · ${res.names[ch]}`} sub={`Epoch ${ep + 1}. A flat histogram means no coupling in that epoch.`}>
            <PhaseAmp eeg={res.signals} cfg={cfg} channel={ch} epoch={ep} pair="tg" />
            <label className="fld"><span>Epoch<span className="val">{ep + 1}</span></span><input type="range" min={1} max={E} value={ep + 1} onChange={(e) => setEp(+e.target.value - 1)} /></label>
            <p className="cap"><b>Figure 4.</b> Mean gamma amplitude in 18 theta-phase bins.</p>
          </Panel>
          {topo ? <Panel title="Scalp distribution of theta → gamma MI" sub={`${topo.idx.length} channels matched to standard 10–20 sites.`}>
            <Scalp size={250} values={topo.idx.map((i) => res.tg.meanMI[i])} coords={topo.coords} names={topo.names} unit=" MI" color={ramp} />
            <ColorBar lo={Math.min(...topo.idx.map((i) => res.tg.meanMI[i]))} hi={Math.max(...topo.idx.map((i) => res.tg.meanMI[i]))} fmt={(v) => v.toFixed(4)} />
            <p className="cap"><b>Figure 5.</b> Interpolated between electrodes for display only.</p></Panel>
          : <Panel title="Scalp distribution"><p style={{ color: "var(--ink-3)" }}>A scalp map needs at least 8 channels with standard 10–20 names (T7/T8/P7/P8 are mapped to T3/T4/T5/T6).</p></Panel>}
        </div>

        <div className="callout">
          <p><b>Interpreting these numbers.</b> MI is biased upwards in short, noisy epochs, so compare channels and conditions rather than reading one value in isolation; the surrogate z-score tells you whether coupling exceeds what amplitude and phase with the same spectra would give by chance. Volume conduction, non-sinusoidal waveforms, 1/f background and muscle artefact can all create apparent coupling. Reference-montage choice matters.</p>
        </div>
      </div>}
      <StepNav prev={["/lab", "Simulate a patient"]} next={["/cohort", "Browse the simulated cohort"]} />
    </div>
  );
}
