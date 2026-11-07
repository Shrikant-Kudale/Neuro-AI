import { useEffect, useRef, useState, ReactNode } from "react";
import { CLASS_COLORS, CLASS_HEX, CLASS_NAMES } from "../lib/data";

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Perceptual-ish sequential ramp (paper → indigo → crimson) returning an rgb() string. */
export function ramp(t: number): string {
  t = Math.min(1, Math.max(0, t));
  const stops = [[247, 242, 244], [196, 190, 240], [61, 59, 224], [196, 30, 79]];
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const c = stops[i].map((v, k) => Math.round(lerp(v, stops[i + 1][k], f)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
/** Diverging ramp: negative → teal, 0 → paper, positive → crimson. */
export function diverge(t: number): string {
  t = Math.min(1, Math.max(-1, t));
  const a = t < 0 ? [14, 143, 126] : [196, 30, 79], z = [247, 242, 244], k = Math.abs(t);
  return `rgb(${a.map((v, i) => Math.round(lerp(z[i], v, k))).join(",")})`;
}

export function Legend({ items }: { items?: { label: string; color: string }[] }) {
  const its = items ?? CLASS_NAMES.map((n, i) => ({ label: n, color: CLASS_COLORS[i] }));
  return <div className="legend">{its.map((i) => <span key={i.label}><i style={{ background: i.color }} />{i.label}</span>)}</div>;
}

export function Panel({ title, sub, children, className = "" }: { title?: string; sub?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={"panel " + className}>
      {title && <h3>{title}</h3>}
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}

/** Horizontal/vertical bar chart (SVG). */
export function BarChart({ labels, series, height = 220, max, fmt = (v: number) => v.toFixed(2), colors, ref0 }: {
  labels: string[]; series: { name: string; values: number[]; color?: string }[]; height?: number; max?: number; fmt?: (v: number) => string; colors?: string[]; ref0?: { y: number; label: string };
}) {
  const W = 560, H = height, L = 38, B = 46, T = 14, R = 8;
  const mx = max ?? Math.max(...series.flatMap((s) => s.values), 1e-9);
  const n = labels.length, gw = (W - L - R) / n, bw = Math.min(46, (gw * 0.8) / series.length);
  return (
    <div className="chart"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart" width="100%">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => { const y = T + (H - T - B) * (1 - t); return <g key={t}><line x1={L} x2={W - R} y1={y} y2={y} stroke="var(--grid-major)" strokeWidth="1" /><text x={L - 6} y={y + 4} fontSize="11" textAnchor="end" fill="var(--ink-3)">{fmt(mx * t)}</text></g>; })}
      {ref0 && <g><line x1={L} x2={W - R} y1={T + (H - T - B) * (1 - ref0.y / mx)} y2={T + (H - T - B) * (1 - ref0.y / mx)} stroke="var(--ink)" strokeDasharray="5 4" /><text x={W - R} y={T + (H - T - B) * (1 - ref0.y / mx) - 4} fontSize="11" textAnchor="end" fill="var(--ink)">{ref0.label}</text></g>}
      {labels.map((lab, i) => (
        <g key={lab}>
          {series.map((s, k) => {
            const h = (H - T - B) * Math.max(0, s.values[i]) / mx, x = L + i * gw + (gw - bw * series.length) / 2 + k * bw;
            return <rect key={k} x={x} y={H - B - h} width={bw - 2} height={h} rx="3" fill={colors?.[i] ?? s.color ?? "var(--signal)"}><title>{`${s.name}: ${fmt(s.values[i])}`}</title></rect>;
          })}
          <text x={L + i * gw + gw / 2} y={H - B + 16} fontSize="11.5" textAnchor="middle" fill="var(--ink-2)">{lab.length > 16 ? lab.slice(0, 15) + "…" : lab}</text>
        </g>
      ))}
    </svg></div>
  );
}

/** Line chart with multiple series (SVG). */
export function LineChart({ x, series, height = 240, xLabel, yLabel, yMin = 0, yMax, xFmt = (v: number) => String(v), points = false, diag = false, logX = false }: {
  x: number[]; series: { name: string; y: number[]; color: string; dash?: string; x?: number[]; w?: number }[]; height?: number; xLabel?: string; yLabel?: string; yMin?: number; yMax?: number; xFmt?: (v: number) => string; points?: boolean; diag?: boolean; logX?: boolean;
}) {
  const W = 560, H = height, L = 46, B = 40, T = 10, R = 12;
  const allY = series.flatMap((s) => s.y);
  const ymax = yMax ?? Math.max(...allY, 1e-9), ymin = yMin;
  const tx = (v: number) => (logX ? Math.log2(v) : v);
  const xs = series.flatMap((s) => s.x ?? x).map(tx);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const px = (v: number) => L + ((tx(v) - x0) / (x1 - x0 || 1)) * (W - L - R);
  const py = (v: number) => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B);
  return (
    <div className="chart"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="line chart" width="100%">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => { const v = ymin + (ymax - ymin) * t, y = py(v); return <g key={t}><line x1={L} x2={W - R} y1={y} y2={y} stroke="var(--grid-major)" /><text x={L - 6} y={y + 4} fontSize="11" textAnchor="end" fill="var(--ink-3)">{v.toFixed(ymax <= 1.01 ? 2 : 1)}</text></g>; })}
      {x.length <= 12 && x.map((v) => <text key={v} x={px(v)} y={H - B + 16} fontSize="11" textAnchor="middle" fill="var(--ink-3)">{xFmt(v)}</text>)}
      {x.length > 12 && [0, 0.25, 0.5, 0.75, 1].map((t) => { const v = x[0] + (x[x.length - 1] - x[0]) * t; return <text key={t} x={px(v)} y={H - B + 16} fontSize="11" textAnchor="middle" fill="var(--ink-3)">{xFmt(Math.round(v * 100) / 100)}</text>; })}
      {diag && <line x1={px(0)} y1={py(0)} x2={px(1)} y2={py(1)} stroke="var(--ink-3)" strokeDasharray="4 4" />}
      {series.map((s) => {
        const xx = s.x ?? x;
        const d = s.y.map((v, i) => `${i ? "L" : "M"}${px(xx[i]).toFixed(1)},${py(v).toFixed(1)}`).join("");
        return <g key={s.name}><path d={d} fill="none" stroke={s.color} strokeWidth={s.w ?? 2.4} strokeDasharray={s.dash} strokeLinejoin="round" />{points && s.y.map((v, i) => <circle key={i} cx={px(xx[i])} cy={py(v)} r="3.4" fill={s.color}><title>{`${s.name}: ${v.toFixed(3)}`}</title></circle>)}</g>;
      })}
      {xLabel && <text x={(L + W - R) / 2} y={H - 6} fontSize="12" textAnchor="middle" fill="var(--ink-2)">{xLabel}</text>}
      {yLabel && <text transform={`translate(12 ${(T + H - B) / 2}) rotate(-90)`} fontSize="12" textAnchor="middle" fill="var(--ink-2)">{yLabel}</text>}
    </svg></div>
  );
}

/** Heat map with row/column labels (SVG). */
export function Heatmap({ data, rows, cols, color = ramp, cell = 22, vmin, vmax, rowLabel, colLabel, showValues = false, fmt = (v: number) => v.toFixed(2) }: {
  data: number[][]; rows: string[]; cols: string[]; color?: (t: number) => string; cell?: number; vmin?: number; vmax?: number; rowLabel?: string; colLabel?: string; showValues?: boolean; fmt?: (v: number) => string;
}) {
  const flat = data.flat();
  const lo = vmin ?? Math.min(...flat), hi = vmax ?? Math.max(...flat);
  const L = 54, T = 8, B = 34, cw = cell, ch = Math.min(cell, 26);
  const W = L + cols.length * cw + 6, H = T + rows.length * ch + B;
  return (
    <div className="chart" style={{ overflowX: "auto" }}><svg viewBox={`0 0 ${W} ${H}`} width={Math.max(W, 260)} role="img" aria-label="heat map">
      {data.map((row, r) => row.map((v, c) => (
        <g key={`${r}-${c}`}>
          <rect x={L + c * cw} y={T + r * ch} width={cw - 1} height={ch - 1} rx="2" fill={color((v - lo) / (hi - lo || 1))}><title>{`${rows[r]} · ${cols[c]}: ${fmt(v)}`}</title></rect>
          {showValues && <text x={L + c * cw + cw / 2} y={T + r * ch + ch / 2 + 4} fontSize="10" textAnchor="middle" fill={(v - lo) / (hi - lo || 1) > 0.55 ? "#fff" : "var(--ink)"}>{fmt(v)}</text>}
        </g>
      )))}
      {rows.map((r, i) => <text key={r} x={L - 5} y={T + i * ch + ch / 2 + 4} fontSize="10.5" textAnchor="end" fill="var(--ink-2)">{r}</text>)}
      {cols.map((c, i) => (i % Math.ceil(cols.length / 15) === 0) && <text key={c + i} x={L + i * cw + cw / 2} y={H - B + 14} fontSize="10.5" textAnchor="middle" fill="var(--ink-2)">{c}</text>)}
      {colLabel && <text x={L + (cols.length * cw) / 2} y={H - 4} fontSize="11.5" textAnchor="middle" fill="var(--ink-2)">{colLabel}</text>}
      {rowLabel && <text transform={`translate(10 ${T + (rows.length * ch) / 2}) rotate(-90)`} fontSize="11.5" textAnchor="middle" fill="var(--ink-2)">{rowLabel}</text>}
    </svg></div>
  );
}

/** Scalp topography: inverse-distance-weighted field on a head outline, 19 electrodes labelled. */
export function Scalp({ values, coords, names, size = 230, color = ramp, vmin, vmax, unit = "", highlight, onPick, lesion }: {
  values: number[]; coords: number[][]; names: string[]; size?: number; color?: (t: number) => string; vmin?: number; vmax?: number; unit?: string; highlight?: number; onPick?: (x: number, y: number) => void; lesion?: { x: number; y: number; sigma: number } | null;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const lo = vmin ?? Math.min(...values), hi = vmax ?? Math.max(...values);
  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext("2d")!, n = 56, dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = size * dpr; cv.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    const r = size * 0.4, cx = size / 2, cy = size / 2 + size * 0.03, cell = (2 * r) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const gx = (i + 0.5) / n * 2 - 1, gy = 1 - (j + 0.5) / n * 2;
      if (gx * gx + gy * gy > 1) continue;
      let wsum = 0, v = 0;
      for (let k = 0; k < coords.length; k++) { const d2 = (gx - coords[k][0]) ** 2 + (gy - coords[k][1]) ** 2 + 0.012; const w = 1 / (d2 * d2); wsum += w; v += w * values[k]; }
      ctx.fillStyle = color((v / wsum - lo) / (hi - lo || 1));
      ctx.fillRect(cx - r + i * cell, cy - r + j * cell, cell + 0.6, cell + 0.6);
    }
  }, [values, coords, size, lo, hi, color]);
  const r = size * 0.4, cx = size / 2, cy = size / 2 + size * 0.03;
  const handle = (e: React.MouseEvent) => {
    if (!onPick) return;
    const b = (e.currentTarget as SVGElement).getBoundingClientRect();
    onPick(((e.clientX - b.left) / b.width * size - cx) / r, -(((e.clientY - b.top) / b.height * size - cy) / r));
  };
  return (
    <div style={{ position: "relative", width: size, height: size, maxWidth: "100%" }}>
      <canvas ref={ref} style={{ width: size, height: size, position: "absolute", inset: 0 }} />
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size} style={{ position: "absolute", inset: 0, cursor: onPick ? "crosshair" : "default" }} onClick={handle} role="img" aria-label={`scalp map ${unit}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--ink)" strokeWidth="2" />
        <path d={`M${cx - 9},${cy - r + 1} L${cx},${cy - r - 13} L${cx + 9},${cy - r + 1}`} fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinejoin="round" />
        <path d={`M${cx - r},${cy - 8} q-9,8 0,16 M${cx + r},${cy - 8} q9,8 0,16`} fill="none" stroke="var(--ink)" strokeWidth="2" />
        {lesion && <circle cx={cx + lesion.x * r} cy={cy - lesion.y * r} r={lesion.sigma * r * 1.2} fill="none" stroke="var(--high)" strokeWidth="2" strokeDasharray="5 4" />}
        {coords.map((p, i) => (
          <g key={i}>
            <circle cx={cx + p[0] * r} cy={cy - p[1] * r} r={highlight === i ? 5.5 : 3.6} fill="var(--paper)" stroke="var(--ink)" strokeWidth={highlight === i ? 2.4 : 1.2}><title>{`${names[i]}: ${values[i].toFixed(4)}${unit}`}</title></circle>
            <text x={cx + p[0] * r} y={cy - p[1] * r - 7} fontSize="8.5" textAnchor="middle" fill="var(--ink)" fontWeight="600" style={{ paintOrder: "stroke", stroke: "var(--paper)", strokeWidth: 2.5 }}>{names[i]}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function ColorBar({ lo, hi, color = ramp, fmt = (v: number) => v.toFixed(3) }: { lo: number; hi: number; color?: (t: number) => string; fmt?: (v: number) => string }) {
  const stops = Array.from({ length: 11 }, (_, i) => color(i / 10)).join(",");
  return <div style={{ display: "grid", gap: 2, width: 200, maxWidth: "100%" }}><div style={{ height: 8, borderRadius: 4, background: `linear-gradient(90deg,${stops})` }} /><div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--ink-3)" }}><span>{fmt(lo)}</span><span>{fmt(hi)}</span></div></div>;
}

export function RiskBars({ probs, compact = false }: { probs: number[]; compact?: boolean }) {
  return (
    <div className="risk-bars">
      {CLASS_NAMES.map((n, i) => (
        <div className="rbar" key={n} style={compact ? { gridTemplateColumns: "104px 1fr 46px", fontSize: "0.88rem" } : undefined}>
          <span>{n}</span>
          <div className="track" role="progressbar" aria-valuenow={Math.round(probs[i] * 100)} aria-valuemin={0} aria-valuemax={100} aria-label={n}><div className="fill" style={{ width: `${probs[i] * 100}%`, background: CLASS_HEX[i] }} /></div>
          <span>{(probs[i] * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

export function ConfusionMatrix({ cm }: { cm: number[][] }) {
  const rowSum = cm.map((r) => r.reduce((a, b) => a + b, 0));
  return (
    <div className="chart"><svg viewBox="0 0 330 290" width="100%" style={{ maxWidth: 380 }} role="img" aria-label="confusion matrix">
      {cm.map((row, i) => row.map((v, j) => {
        const f = v / (rowSum[i] || 1);
        return <g key={`${i}${j}`}><rect x={92 + j * 78} y={26 + i * 66} width="74" height="62" rx="6" fill={i === j ? `color-mix(in srgb, var(--signal) ${Math.round(f * 100)}%, var(--paper))` : `color-mix(in srgb, var(--high) ${Math.round(f * 100)}%, var(--paper))`} stroke="var(--line)" /><text x={129 + j * 78} y={55 + i * 66} fontSize="17" fontWeight="700" textAnchor="middle" fill={f > 0.55 ? "#fff" : "var(--ink)"}>{v}</text><text x={129 + j * 78} y={73 + i * 66} fontSize="11" textAnchor="middle" fill={f > 0.55 ? "#fff" : "var(--ink-2)"}>{(f * 100).toFixed(0)}%</text></g>;
      }))}
      {["Normal", "Early", "High"].map((n, i) => <g key={n}><text x={129 + i * 78} y="16" fontSize="12" textAnchor="middle" fill="var(--ink-2)">{n}</text><text x="86" y={62 + i * 66} fontSize="12" textAnchor="end" fill="var(--ink-2)">{n}</text></g>)}
      <text x="210" y="284" fontSize="12" textAnchor="middle" fill="var(--ink-2)">Predicted class (columns) vs true class (rows)</text>
    </svg></div>
  );
}

/** Stateful tooltip helper for SVG/canvas charts. */
export function useTooltip() {
  const [tip, set] = useState<{ x: number; y: number; text: ReactNode } | null>(null);
  const el = tip ? <div className="tooltip" style={{ left: tip.x + 14, top: tip.y + 14 }}>{tip.text}</div> : null;
  return { show: (e: { clientX: number; clientY: number }, text: ReactNode) => set({ x: e.clientX, y: e.clientY, text }), hide: () => set(null), el };
}
