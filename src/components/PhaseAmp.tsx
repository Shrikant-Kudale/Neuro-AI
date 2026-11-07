import { useMemo } from "react";
import { analytic, filtfiltZeroPhase } from "../lib/dsp";
import { modulationIndex, DspConfig } from "../lib/pac";

/** Mean gamma amplitude per theta-phase bin for one channel and epoch — the picture behind the Modulation Index. */
export default function PhaseAmp({ eeg, cfg, channel, epoch, pair }: { eeg: Float64Array[]; cfg: DspConfig; channel: number; epoch: number; pair: "tg" | "ab" }) {
  const nb = cfg.n_bins;
  const { bars, mi } = useMemo(() => {
    const [pb, ab] = pair === "tg" ? (["theta", "gamma"] as const) : (["alpha", "beta"] as const);
    const x = eeg[channel], L = cfg.epoch_len, s = epoch * L;
    const fp = filtfiltZeroPhase(cfg.sos[pb], x).subarray(s, s + L), fa = filtfiltZeroPhase(cfg.sos[ab], x).subarray(s, s + L);
    const [pr, pi] = analytic(fp), [ar, ai] = analytic(fa);
    const ph = new Float64Array(L), am = new Float64Array(L), tot = new Float64Array(nb), cnt = new Float64Array(nb);
    for (let i = 0; i < L; i++) { ph[i] = Math.atan2(pi[i], pr[i]); am[i] = Math.hypot(ar[i], ai[i]); const b = Math.min(nb - 1, Math.max(0, Math.floor((ph[i] + Math.PI) / ((2 * Math.PI) / nb)))); tot[b] += am[i]; cnt[b]++; }
    const m = Array.from(tot, (t, b) => (cnt[b] ? t / cnt[b] : 0)), s2 = m.reduce((a, b) => a + b, 0);
    return { bars: m.map((v) => v / (s2 || 1)), mi: modulationIndex(ph, am, nb) };
  }, [eeg, cfg, channel, epoch, pair, nb]);
  const W = 420, H = 190, L = 36, B = 34, mx = Math.max(...bars, 1 / nb) * 1.15, bw = (W - L - 8) / nb;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Amplitude by phase, MI ${mi.toFixed(4)}`}>
      <line x1={L} x2={W - 8} y1={H - B - ((H - B - 10) * (1 / nb)) / mx} y2={H - B - ((H - B - 10) * (1 / nb)) / mx} stroke="var(--ink)" strokeDasharray="4 4" />
      <text x={W - 10} y={H - B - ((H - B - 10) * (1 / nb)) / mx - 5} fontSize="10.5" textAnchor="end" fill="var(--ink)">uniform (no coupling)</text>
      {bars.map((v, i) => { const h = ((H - B - 10) * v) / mx; return <rect key={i} x={L + i * bw + 1} y={H - B - h} width={bw - 2} height={h} rx="2" fill={pair === "tg" ? "var(--high)" : "var(--signal)"} opacity="0.88"><title>{`bin ${i + 1}: ${(v * 100).toFixed(2)}% of total amplitude`}</title></rect>; })}
      {[-180, -90, 0, 90, 180].map((d) => <text key={d} x={L + ((d + 180) / 360) * (W - L - 8)} y={H - B + 14} fontSize="10.5" textAnchor="middle" fill="var(--ink-3)">{d}°</text>)}
      <text x={(L + W) / 2} y={H - 4} fontSize="11.5" textAnchor="middle" fill="var(--ink-2)">{pair === "tg" ? "theta phase" : "alpha phase"} (18 bins of 20°)</text>
      <text x={L + 4} y="14" fontSize="12.5" fontWeight="700" fill="var(--ink)">MI = {mi.toFixed(4)}</text>
    </svg>
  );
}
