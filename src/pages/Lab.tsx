import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PageHead } from "../components/StepNav";
import StepNav from "../components/StepNav";
import { useManifest, useModel, useResults, CLASS_NAMES, CLASS_HEX, BEH_NAMES, PROFILE } from "../lib/data";
import { analyse, presetUI, rescore, chMean, mean, Analysis, StudioUI } from "../lib/studio";
import { Panel, RiskBars, Scalp, ColorBar, BarChart, Heatmap, LineChart, ramp, diverge, Legend } from "../components/charts";
import EegViewer from "../components/EegViewer";
import PhaseAmp from "../components/PhaseAmp";

const BEH_RANGE: [number, number, number][] = [[0.5, 3.5, 0.01], [0, 0.35, 0.005], [0, 0.8, 0.01]];

function Slider({ label, value, min, max, step, onChange, fmt = (v: number) => v.toFixed(2), disabled }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string; disabled?: boolean }) {
  return (
    <label className="fld"><span>{label}<span className="val">{fmt(value)}</span></span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(+e.target.value)} /></label>
  );
}

export default function Lab() {
  const profile = PROFILE;
  const { data: man } = useManifest();
  const { data: model, error: modelErr } = useModel(profile);
  const { data: res } = useResults(profile);
  const cfg = man?.dsp as any;
  const [ui, setUi] = useState<StudioUI | null>(null);
  const [a, setA] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [channel, setChannel] = useState(3);
  const [epoch, setEpoch] = useState(7);
  const [occ, setOcc] = useState<number[] | null>(null);
  const [sweep, setSweep] = useState<{ red: number[]; probs: number[][] } | null>(null);
  const [sweeping, setSweeping] = useState(false);
  const timer = useRef<number>();
  const lastEEGKey = useRef("");

  // initialise / reset preset when profile or model changes
  useEffect(() => { if (cfg) setUi(presetUI(cfg, profile, 2, 7)); }, [cfg, profile]);

  const eegKey = (u: StudioUI) => JSON.stringify([profile, u.cls, u.seed, u.red, u.gain, u.gboost, u.layout, u.nAff, u.lesion, u.temporal, u.noise]);

  const run = useCallback((u: StudioUI) => {
    if (!cfg || !model) return;
    setBusy(true);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      const key = eegKey(u);
      if (a && key === lastEEGKey.current) { setA(rescore(model, a, u.beh)); setBusy(false); return; }
      const out = analyse(cfg, model, profile, u);
      lastEEGKey.current = key;
      setA(out); setOcc(null); setBusy(false); setSweep(null);
    }, 300);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cfg, model, profile, a]);

  useEffect(() => { if (ui && cfg && model) run(ui); /* eslint-disable-next-line */ }, [ui, cfg, model]);

  // occlusion map computed after the main result so the UI stays responsive
  useEffect(() => {
    if (!a || !model || occ) return;
    const t = window.setTimeout(() => setOcc(model.channelOcclusion(a.raw, 19, model.manifest.fill, a.pred.pred)), 50);
    return () => window.clearTimeout(t);
  }, [a, model, occ]);

  const set = (patch: Partial<StudioUI>) => setUi((u) => (u ? { ...u, ...patch } : u));
  const applyPreset = (cls: number) => { if (cfg) setUi(presetUI(cfg, profile, cls, ui?.seed ?? 7)); };

  const runSweep = async () => {
    if (!ui || !cfg || !model) return;
    setSweeping(true);
    const reds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], probs: number[][] = [];
    for (const r of reds) {
      await new Promise((res2) => setTimeout(res2, 20));
      probs.push(analyse(cfg, model, profile, { ...ui, red: r }).pred.probs);
      setSweep({ red: reds.slice(0, probs.length), probs: [...probs] });
    }
    setSweeping(false);
  };

  const ref = res?.class_stats?.Normal;
  const tgCh = a ? chMean(a.pac.tg) : null, abCh = a ? chMean(a.pac.ab) : null;
  const lowCh = useMemo(() => (tgCh && ref ? cfg.channels.filter((_: string, i: number) => tgCh[i] < 0.72 * ref.tg_by_channel[i]) : []), [tgCh, ref, cfg]);

  const download = () => {
    if (!a || !ui) return;
    const blob = new Blob([JSON.stringify({ note: "Simulated subject — research prototype, not a medical device.", profile, studio: ui, behavioural: ui.beh, probabilities: Object.fromEntries(CLASS_NAMES.map((n, i) => [n, a.pred.probs[i]])), tg_mi: a.pac.tg.map((r) => Array.from(r)), ab_mi: a.pac.ab.map((r) => Array.from(r)), attention_cls: a.pred.clsAttn }, null, 2)], { type: "application/json" });
    const el = document.createElement("a"); el.href = URL.createObjectURL(blob); el.download = `neuropac-subject-${ui.seed}.json`; el.click();
  };

  if (modelErr) return <div className="page"><div className="callout bad"><p>Model files were not found. Run <code>python research/run_pipeline.py</code> and redeploy. ({modelErr})</p></div></div>;
  if (!cfg || !ui || !model) return <div className="page"><p>Loading the lab…</p></div>;
  const prof = cfg.profiles[profile];

  return (
    <div className="page">
      <PageHead step="Step 2 of 7 · Simulate" title="Simulation studio">
        Define a virtual subject in the left panel (A to D). The studio synthesises 30 s of 19-channel EEG, measures coupling, and shows how the classifier responds. Because the true disruption is set by you, this is a controlled way to see what the pipeline can and cannot detect.
      </PageHead>

      <div className="lab">
        <aside className="lab-side panel" aria-label="Patient controls">
          <div className="fgroup">
            <h4><i>A</i>Start from a template</h4>
            <div className="chips">{CLASS_NAMES.map((n, i) => <button key={n} className="chip" aria-pressed={ui.cls === i} onClick={() => applyPreset(i)}>{n}</button>)}</div>
            <div style={{ display: "flex", gap: 8 }}>
              <label className="fld" style={{ flex: 1 }}>Random seed<input type="number" value={ui.seed} min={0} max={999999} onChange={(e) => set({ seed: Math.max(0, Math.floor(+e.target.value || 0)) })} /></label>
              <button className="btn ghost sm" style={{ alignSelf: "end" }} onClick={() => set({ seed: Math.floor(Math.random() * 99999) })}>New patient</button>
            </div>
          </div>
          <div className="fgroup">
            <h4><i>B</i>Coupling disruption</h4>
            <Slider label="Theta–gamma loss" value={ui.red} min={0} max={0.9} step={0.01} onChange={(v) => set({ red: v })} fmt={(v) => (v * 100).toFixed(0) + "%"} />
            <Slider label="Alpha–beta elevation" value={ui.gain} min={0} max={0.6} step={0.01} onChange={(v) => set({ gain: v })} fmt={(v) => "+" + (v * 100).toFixed(0) + "%"} />
            <Slider label="Gamma hyperexcitability" value={ui.gboost} min={1} max={1.6} step={0.01} onChange={(v) => set({ gboost: v })} fmt={(v) => "×" + v.toFixed(2)} />
          </div>
          <div className="fgroup">
            <h4><i>C</i>Spatial extent and time course</h4>
            <div className="seg" role="group" aria-label="Layout"><button aria-pressed={ui.layout === "random"} onClick={() => set({ layout: "random" })}>Scattered</button><button aria-pressed={ui.layout === "focal"} onClick={() => set({ layout: "focal" })}>Focal lesion</button></div>
            {ui.layout === "random" ? <Slider label="Affected channels" value={ui.nAff} min={0} max={19} step={1} onChange={(v) => set({ nAff: v })} fmt={(v) => String(v)} /> : <>
              <p style={{ fontSize: "0.85rem", margin: 0, color: "var(--ink-2)" }}>Click the head to place the lesion centre.</p>
              <Scalp size={190} values={cfg.channels.map((_: string, i: number) => Math.exp(-(((cfg.coords[i][0] - ui.lesion.x) ** 2 + (cfg.coords[i][1] - ui.lesion.y) ** 2) / (2 * ui.lesion.sigma ** 2))))} coords={cfg.coords} names={cfg.channels} vmin={0} vmax={1} onPick={(x, y) => set({ lesion: { ...ui.lesion, x: Math.max(-0.95, Math.min(0.95, x)), y: Math.max(-0.95, Math.min(0.95, y)) } })} lesion={ui.lesion} />
              <Slider label="Lesion spread" value={ui.lesion.sigma} min={0.15} max={0.8} step={0.01} onChange={(v) => set({ lesion: { ...ui.lesion, sigma: v } })} />
            </>}
            <label className="fld">Time course<select value={ui.temporal} onChange={(e) => set({ temporal: e.target.value as any })}><option value="sustained">Sustained across the recording</option><option value="intermittent">Intermittent</option><option value="ramp">Builds up over time</option></select></label>
          </div>
          <div className="fgroup">
            <h4><i>D</i>Behavioural measures</h4>
            {BEH_NAMES.map((n, j) => <Slider key={n} label={cfg.beh_long[j]} value={ui.beh[j]} min={BEH_RANGE[j][0]} max={BEH_RANGE[j][1]} step={BEH_RANGE[j][2]} fmt={(v) => v.toFixed(j === 0 ? 2 : 3)} onChange={(v) => { const b = [...ui.beh] as [number, number, number]; b[j] = v; set({ beh: b }); }} />)}
          </div>
          <div className="fgroup">
            <h4><i>E</i>Acquisition quality</h4>
            <Slider label="Sensor noise" value={ui.noise} min={0.25} max={8} step={0.05} onChange={(v) => set({ noise: v })} fmt={(v) => "×" + v.toFixed(2)} />
            <p style={{ fontSize: "0.8rem", color: "var(--ink-3)", margin: 0 }}>The simulation also adds ocular artefacts and electrode gain variation (white noise baseline ×{prof.white_sd.toFixed(2)}).</p>
          </div>
        </aside>

        <div className={busy ? "busy" : ""} style={{ display: "grid", gap: 18, minWidth: 0 }}>
          {a && <>
            <Panel title="Classifier output" sub={`Transformer trained on the simulated cohort. Computed in ${a.ms.toFixed(0)} ms in your browser. Research output only.`}>
              <div className="verdict"><span className="big" style={{ color: CLASS_HEX[a.pred.pred] }}>{CLASS_NAMES[a.pred.pred]}</span><span className={"tag c" + a.pred.pred}>{(a.pred.probs[a.pred.pred] * 100).toFixed(0)}% confidence</span></div>
              <RiskBars probs={a.pred.probs} />
              <div className="callout" style={{ marginTop: 14 }}>
                <p><b>Reading this.</b> Mean theta–gamma MI is {mean(tgCh!).toFixed(4)} (simulated normals average {ref ? ref.tg_mean.toFixed(4) : "…"}) and mean alpha–beta MI is {mean(abCh!).toFixed(4)}. {lowCh.length ? `Coupling is markedly lower than normal at ${lowCh.join(", ")}.` : "No channel shows a marked loss of theta–gamma coupling."} {a.params.artifacts.length ? `${a.params.artifacts.length} ocular artefact(s) were injected; ${a.pac.flagged.flat().filter(Boolean).length} channel-epochs were rejected and interpolated.` : ""}</p>
              </div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }} className="no-print"><button className="btn sm" onClick={download}>Download subject JSON</button><button className="btn ghost sm" onClick={() => window.print()}>Print report</button></div>
            </Panel>

            <Panel title="Simulated EEG recording" sub="19 channels, 256 Hz, 30 s. The coupling view shows how gamma amplitude follows theta phase on the selected channel.">
              <EegViewer eeg={a.eeg} cfg={cfg} channel={channel} onChannel={setChannel} />
            </Panel>

            <div className="grid2">
              <Panel title="Phase-amplitude coupling, theta → gamma" sub={`${cfg.channels[channel]}, epoch ${epoch + 1}. Flat bars mean no coupling.`}>
                <PhaseAmp eeg={a.eeg} cfg={cfg} channel={channel} epoch={epoch} pair="tg" />
                <Slider label="Epoch" value={epoch + 1} min={1} max={15} step={1} onChange={(v) => setEpoch(v - 1)} fmt={(v) => `${v} (${(v - 1) * 2}–${v * 2} s)`} />
              </Panel>
              <Panel title="Phase-amplitude coupling, alpha → beta" sub="Compensatory coupling tends to rise when theta–gamma coupling falls.">
                <PhaseAmp eeg={a.eeg} cfg={cfg} channel={channel} epoch={epoch} pair="ab" />
              </Panel>
            </div>

            <div className="grid3">
              <Panel title="Theta–gamma MI by site"><Scalp values={tgCh!} coords={cfg.coords} names={cfg.channels} unit=" MI" highlight={channel} /><ColorBar lo={Math.min(...tgCh!)} hi={Math.max(...tgCh!)} fmt={(v) => v.toFixed(4)} /></Panel>
              <Panel title="Alpha–beta MI by site"><Scalp values={abCh!} coords={cfg.coords} names={cfg.channels} unit=" MI" highlight={channel} /><ColorBar lo={Math.min(...abCh!)} hi={Math.max(...abCh!)} fmt={(v) => v.toFixed(4)} /></Panel>
              <Panel title="What the model relied on" sub={`Drop in P(${CLASS_NAMES[a.pred.pred]}) when a site's coupling is replaced by a typical value.`}>
                {occ ? (() => { const m = Math.max(0.02, ...occ.map(Math.abs)); return <><Scalp values={occ} coords={cfg.coords} names={cfg.channels} color={diverge} vmin={-m} vmax={m} unit=" Δp" highlight={channel} /><ColorBar lo={-m} hi={m} color={(t) => diverge(2 * t - 1)} fmt={(v) => v.toFixed(2)} /></>; })() : <p>Computing…</p>}
              </Panel>
            </div>

            <div className="grid2">
              <Panel title="Coupling across the recording" sub="Theta–gamma MI for every channel (rows) and 2-second epoch (columns).">
                <Heatmap data={a.pac.tg.map((r) => Array.from(r))} rows={cfg.channels} cols={Array.from({ length: 15 }, (_, i) => String(i + 1))} cell={24} colLabel="epoch" />
              </Panel>
              <Panel title="Where the Transformer looked in time" sub="Attention from the [CLS] token to each epoch (last layer) and attention rollout through all four layers.">
                <BarChart labels={Array.from({ length: 15 }, (_, i) => String(i + 1))} series={[{ name: "last-layer attention", values: a.pred.clsAttn, color: "var(--signal)" }, { name: "rollout", values: a.pred.rollout, color: "var(--high)" }]} ref0={{ y: 1 / 15, label: "uniform" }} height={230} fmt={(v) => v.toFixed(2)} />
                <Legend items={[{ label: "last layer", color: "var(--signal)" }, { label: "rollout", color: "var(--high)" }]} />
              </Panel>
            </div>

            <Panel title="Sensitivity sweep" sub="Same subject, theta–gamma loss varied from 0 to 90%: how much disruption does it take to move the estimate?">
              <button className="btn sm no-print" onClick={runSweep} disabled={sweeping}>{sweeping ? "Sweeping…" : "Run sensitivity sweep (about 10 s)"}</button>
              {sweep && <div style={{ marginTop: 12 }}><LineChart x={sweep.red} series={CLASS_NAMES.map((n, i) => ({ name: n, y: sweep.probs.map((p) => p[i]), color: CLASS_HEX[i] }))} yMax={1} xLabel="theta–gamma loss" yLabel="probability" xFmt={(v) => (v * 100).toFixed(0) + "%"} points /><Legend /></div>}
            </Panel>
          </>}
        </div>
      </div>
      <StepNav prev={["/", "Overview"]} next={["/analyse", "Analyse a real recording"]} />
    </div>
  );
}
