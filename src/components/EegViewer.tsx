import { useEffect, useMemo, useRef, useState } from "react";
import { filtfiltZeroPhase, analytic } from "../lib/dsp";
import type { DspConfig } from "../lib/pac";

type Mode = "stack" | "bands";

/** Canvas EEG viewer: all 19 channels stacked, or one channel decomposed into theta, gamma envelope and the raw trace. */
export default function EegViewer({ eeg, cfg, channel, onChannel }: { eeg: Float64Array[]; cfg: DspConfig; channel: number; onChannel: (c: number) => void }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>("bands");
  const [start, setStart] = useState(4);
  const win = mode === "stack" ? 8 : 4;
  const fs = cfg.fs;

  const bands = useMemo(() => {
    const x = eeg[channel];
    const th = filtfiltZeroPhase(cfg.sos.theta, x), ga = filtfiltZeroPhase(cfg.sos.gamma, x);
    // gamma amplitude envelope over whole record via 512-sample blocks (power-of-two FFT)
    const env = new Float64Array(x.length);
    for (let e = 0; e < cfg.n_epochs; e++) { const [r, i] = analytic(ga.subarray(e * 512, (e + 1) * 512)); for (let k = 0; k < 512; k++) env[e * 512 + k] = Math.hypot(r[k], i[k]); }
    return { th, ga, env };
  }, [eeg, channel, cfg]);

  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!, dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = mode === "stack" ? 520 : 330;
    cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const col = (n: string, d: string) => css.getPropertyValue(n).trim() || d;
    const ink = col("--ink", "#1d1842"), sig = col("--signal", "#3d3be0"), high = col("--high", "#c41e4f"), grid = col("--grid-major", "#e2c6d1"), mute = col("--ink-3", "#7b7696");
    const L = 44, i0 = Math.floor(start * fs), n = Math.floor(win * fs), px = (i: number) => L + ((i - i0) / n) * (w - L - 6);
    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = mute; ctx.font = "11px Figtree, sans-serif";
    for (let s = Math.ceil(start); s <= start + win; s++) { const x = px(s * fs); ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - 16); ctx.stroke(); ctx.fillText(`${s}s`, x + 3, h - 4); }
    const line = (arr: ArrayLike<number>, y0: number, scale: number, color: string, lw: number) => {
      ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round";
      const step = Math.max(1, Math.floor(n / (w * 1.4)));
      for (let i = i0; i < i0 + n; i += step) { const x = px(i), y = y0 - arr[i] * scale; i === i0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke();
    };
    if (mode === "stack") {
      const rowH = (h - 20) / eeg.length;
      eeg.forEach((x, c) => {
        const y0 = 8 + rowH * (c + 0.5);
        if (c === channel) { ctx.fillStyle = "rgba(61,59,224,0.08)"; ctx.fillRect(L, y0 - rowH / 2, w - L - 6, rowH); }
        ctx.fillStyle = c === channel ? ink : mute; ctx.font = (c === channel ? "700 " : "500 ") + "11px Figtree, sans-serif";
        ctx.fillText(cfg.channels[c], 6, y0 + 4);
        line(x, y0, rowH / 110, c === channel ? sig : ink, c === channel ? 1.4 : 0.9);
      });
    } else {
      const rows = [["Raw scalp EEG", eeg[channel], 80, ink], ["Theta band 4–8 Hz", bands.th, 70, sig], ["Gamma 30–80 Hz and its amplitude envelope", bands.ga, 90, high]] as const;
      const rowH = (h - 24) / 3;
      rows.forEach(([lab, arr, sc, color], r) => {
        const y0 = 6 + rowH * (r + 0.5);
        ctx.fillStyle = mute; ctx.font = "600 11.5px Figtree, sans-serif"; ctx.fillText(lab, L + 4, y0 - rowH / 2 + 14);
        line(arr as any, y0 + 6, rowH / sc * (r === 0 ? 0.9 : 0.8), color, 1.3);
        if (r === 2) { line(bands.env, y0 + 6, rowH / sc * 0.8, ink, 2); }
      });
    }
  }, [eeg, mode, start, channel, bands, cfg, fs, win]);

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
        <div className="seg" role="group" aria-label="View"><button aria-pressed={mode === "bands"} onClick={() => setMode("bands")}>Coupling view</button><button aria-pressed={mode === "stack"} onClick={() => setMode("stack")}>All channels</button></div>
        <label className="fld" style={{ flexDirection: "row", alignItems: "center", display: "flex", gap: 8 }}>Channel
          <select value={channel} onChange={(e) => onChannel(+e.target.value)} style={{ width: 90 }}>{cfg.channels.map((c, i) => <option key={c} value={i}>{c}</option>)}</select></label>
        <label className="fld" style={{ flex: 1, minWidth: 160 }}><span>Window start<span className="val">{start.toFixed(1)} s</span></span>
          <input type="range" min={0} max={cfg.n_samples / fs - win} step={0.25} value={Math.min(start, cfg.n_samples / fs - win)} onChange={(e) => setStart(+e.target.value)} /></label>
      </div>
      <canvas ref={ref} style={{ width: "100%", borderRadius: 8, background: "var(--paper-2)", border: "1px solid var(--line)" }} role="img" aria-label="Simulated EEG traces" />
    </div>
  );
}
