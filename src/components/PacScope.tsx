import { useEffect, useMemo, useRef, useState } from "react";
import { modulationIndex } from "../lib/pac";

/** Hero instrument: a theta rhythm carrying gamma bursts. Drag the coupling slider and watch the Modulation Index fall. */
export default function PacScope() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [coupling, setCoupling] = useState(0.85);
  const cRef = useRef(coupling);
  cRef.current = coupling;

  const mi = useMemo(() => {
    const fs = 256, n = 1024, ph = new Float64Array(n), am = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = 2 * Math.PI * 6 * (i / fs);
      ph[i] = Math.atan2(Math.sin(p), Math.cos(p));
      am[i] = 1 + coupling * Math.cos(p);   // gamma envelope imposed by theta phase
    }
    return modulationIndex(ph, am, 18);
  }, [coupling]);

  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, t0 = performance.now(), live = true;
    const css = getComputedStyle(document.documentElement);
    const draw = (now: number) => {
      if (!live) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1), w = cv.clientWidth, h = 280;
      if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const ink = css.getPropertyValue("--ink").trim() || "#1d1842", sig = css.getPropertyValue("--signal").trim() || "#3d3be0";
      const high = css.getPropertyValue("--high").trim() || "#c41e4f", grid = css.getPropertyValue("--grid-major").trim() || "#e2c6d1";
      const t = reduce ? 1.1 : (now - t0) / 1000, m = cRef.current, sec = 2.4;
      const mid = 150;
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      for (let s = 0; s <= sec * 2; s++) { const x = (s / (sec * 2)) * w; ctx.beginPath(); ctx.moveTo(x, 20); ctx.lineTo(x, h - 8); ctx.stroke(); }
      const theta = (u: number) => Math.sin(2 * Math.PI * 6 * u);
      const gamma = (u: number) => Math.sin(2 * Math.PI * 46 * u + 1.7 * Math.sin(2 * Math.PI * 0.7 * u));
      const env = (u: number) => 1 + m * Math.cos(2 * Math.PI * 6 * u);
      const trace = (fn: (u: number) => number, y0: number, amp: number, color: string, lw: number) => {
        ctx.beginPath();
        for (let x = 0; x <= w; x += 1) { const u = t - sec + (x / w) * sec; const y = y0 - amp * fn(u); x ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.stroke();
      };
      trace((u) => theta(u), 52, 22, sig, 2.2);                                    // slow rhythm
      trace((u) => env(u) * gamma(u) * 0.55, 112, 22, high, 1.3);                  // gamma carrying the modulation
      trace((u) => 0.5 * env(u) - 0.5, 112, 22, ink, 1.2);                          // gamma envelope (dotted feel via lighter weight)
      trace((u) => 0.8 * theta(u) + 0.55 * env(u) * gamma(u) * 0.8, 215, 22, ink, 1.8);  // what the scalp electrode records
      ctx.fillStyle = css.getPropertyValue("--ink-3").trim() || "#7b7696"; ctx.font = "600 12px Figtree, sans-serif";
      ctx.fillText("theta, 4–8 Hz", 10, 20); ctx.fillText("gamma, 30–80 Hz, scaled by theta phase", 10, 84); ctx.fillText("scalp signal (sum)", 10, 186);
      if (!reduce) raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => { live = false; cancelAnimationFrame(raf); };
  }, []);

  useEffect(() => { /* repaint once for reduced-motion users when coupling changes */ const cv = ref.current!; cv.dispatchEvent(new Event("repaint")); }, [coupling]);

  return (
    <div className="scope">
      <div className="scope-head"><span>One simulated channel</span><span>Modulation Index {mi.toFixed(3)}</span></div>
      <canvas ref={ref} height={280} aria-label="Animated theta rhythm with gamma bursts. Lower the slider to see coupling break down." role="img" />
      <div className="scope-ctl">
        <label className="fld"><span>Theta–gamma coupling<span className="val">{coupling < 0.25 ? "disrupted" : coupling < 0.6 ? "weakened" : "intact"}</span></span>
          <input type="range" min={0} max={1} step={0.01} value={coupling} onChange={(e) => setCoupling(+e.target.value)} aria-label="Coupling strength" /></label>
      </div>
    </div>
  );
}
