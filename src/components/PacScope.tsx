import { useEffect, useMemo, useRef, useState } from "react";
import { modulationIndex } from "../lib/pac";

/** Figure 1: a theta rhythm gating gamma amplitude. The slider changes coupling strength; the MI is computed with the real estimator. */
export default function PacScope() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [coupling, setCoupling] = useState(0.85);
  const cRef = useRef(coupling);
  cRef.current = coupling;

  const mi = useMemo(() => {
    const fs = 256, n = 1024, ph = new Float64Array(n), am = new Float64Array(n);
    for (let i = 0; i < n; i++) { const p = 2 * Math.PI * 6 * (i / fs); ph[i] = Math.atan2(Math.sin(p), Math.cos(p)); am[i] = 1 + coupling * Math.cos(p); }
    return modulationIndex(ph, am, 18);
  }, [coupling]);

  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, live = true; const t0 = performance.now();
    const css = getComputedStyle(document.documentElement);
    const col = (n: string, d: string) => css.getPropertyValue(n).trim() || d;
    const draw = (now: number) => {
      if (!live) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1), w = cv.clientWidth, h = 270;
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      const ink = col("--ink", "#15222f"), sig = col("--signal", "#1b6ca8"), high = col("--high", "#b3362f"), grid = col("--grid-major", "#dce2ea"), mute = col("--ink-3", "#6a7a8c");
      const t = reduce ? 1.1 : (now - t0) / 1400, m = cRef.current, sec = 2.4;
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      for (let s = 0; s <= 12; s++) { const x = (s / 12) * w; ctx.beginPath(); ctx.moveTo(x, 8); ctx.lineTo(x, h - 8); ctx.stroke(); }
      const theta = (u: number) => Math.sin(2 * Math.PI * 6 * u);
      const gamma = (u: number) => Math.sin(2 * Math.PI * 46 * u + 1.7 * Math.sin(2 * Math.PI * 0.7 * u));
      const env = (u: number) => 1 + m * Math.cos(2 * Math.PI * 6 * u);
      const trace = (fn: (u: number) => number, y0: number, amp: number, color: string, lw: number) => {
        ctx.beginPath();
        for (let x = 0; x <= w; x++) { const u = t - sec + (x / w) * sec, y = y0 - amp * fn(u); x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.stroke();
      };
      trace(theta, 48, 20, sig, 1.8);
      trace((u) => env(u) * gamma(u) * 0.55, 108, 20, high, 1.1);
      trace((u) => 0.5 * env(u) - 0.5, 108, 20, ink, 1.6);
      trace((u) => 0.8 * theta(u) + 0.44 * env(u) * gamma(u), 205, 22, ink, 1.4);
      ctx.fillStyle = mute; ctx.font = "600 11.5px 'IBM Plex Sans', sans-serif";
      ctx.fillText("Theta 4–8 Hz", 8, 16); ctx.fillText("Gamma 30–80 Hz (red) and its envelope (black)", 8, 76); ctx.fillText("Resulting scalp signal", 8, 162);
      if (!reduce) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { live = false; cancelAnimationFrame(raf); };
  }, []);

  return (
    <figure className="scope" style={{ margin: 0 }}>
      <div className="scope-head"><span>Figure 1. Theta phase gates gamma amplitude</span><span>Modulation index {mi.toFixed(3)}</span></div>
      <canvas ref={ref} height={270} role="img" aria-label="Animated theta rhythm with gamma bursts whose amplitude follows theta phase" />
      <div className="scope-ctl">
        <label className="fld"><span>Coupling strength<span className="val">{coupling < 0.25 ? "disrupted" : coupling < 0.6 ? "reduced" : "intact"}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={coupling} onChange={(e) => setCoupling(+e.target.value)} aria-label="Coupling strength" /></label>
        <p className="cap" style={{ marginTop: 6 }}>Lowering coupling flattens the gamma envelope; the modulation index falls towards 0.</p>
      </div>
    </figure>
  );
}
