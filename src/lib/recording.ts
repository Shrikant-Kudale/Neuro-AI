import { resample } from "./resample";
import { analytic, filtfiltZeroPhase } from "./dsp";
import { modulationIndex, DspConfig } from "./pac";
import { RNG } from "./rng";

export interface RawRecording { names: string[]; data: Float64Array[]; fs: number[]; unitScale: number; source: string }

/** Map montage labels such as "EEG Fp1-REF", "T7", "P8-LE" onto the 19 channels of the 10–20 system used by the models. */
const ALIAS: Record<string, string> = { T7: "T3", T8: "T4", P7: "T5", P8: "T6", FP1: "Fp1", FP2: "Fp2" };
export function canonicalName(raw: string, known: string[]): string | null {
  let s = raw.replace(/^EEG[\s_-]*/i, "").replace(/[-_ ]*(REF|LE|AR|AVG|M1|M2|A1|A2|Pz)$/i, (m) => (/pz/i.test(m) ? m : "")).replace(/[^A-Za-z0-9]/g, "");
  const up = s.toUpperCase();
  const hit = known.find((k) => k.toUpperCase() === up) ?? (ALIAS[up] ? known.find((k) => k === ALIAS[up]) : undefined);
  return hit ?? null;
}

// ── parsers ─────────────────────────────────────────────────────────────────────
export function parseCSV(text: string, fsHint?: number): RawRecording {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length && !l.trim().startsWith("#"));
  if (lines.length < 3) throw new Error("The file has too few rows to be an EEG recording.");
  const delim = [",", "\t", ";"].find((d) => lines[0].includes(d)) ?? /\s+/;
  const split = (l: string) => l.trim().split(delim as any).map((c) => c.trim().replace(/^"|"$/g, ""));
  let head = split(lines[0]);
  const hasHeader = head.some((c) => c !== "" && !isFinite(Number(c)));
  const rows = (hasHeader ? lines.slice(1) : lines).map((l) => split(l).map(Number));
  if (!hasHeader) head = rows[0].map((_, i) => `Ch${i + 1}`);
  let cols = head.length, n = rows.length;
  let get = (r: number, c: number) => rows[r][c];
  let names = head;
  // transposed export (channels as rows): many more columns than rows
  if (!hasHeader && cols > n && n <= 128) { const t = rows; names = t.map((_, i) => `Ch${i + 1}`); cols = n; n = t[0].length; get = (r, c) => t[c][r]; }
  let tcol = names.findIndex((h) => /^(time|t|timestamp|sample|samples)$/i.test(h));
  let fs = fsHint ?? 0;
  if (tcol >= 0) {
    const dts: number[] = []; for (let r = 1; r < Math.min(n, 2000); r++) dts.push(get(r, tcol) - get(r - 1, tcol));
    dts.sort((a, b) => a - b); const dt = dts[dts.length >> 1];
    if (!fs && dt > 0) fs = dt > 1 ? 1000 / dt : 1 / dt;
  }
  const keep = names.map((_, i) => i).filter((i) => i !== tcol);
  const data = keep.map((c) => { const a = new Float64Array(n); for (let r = 0; r < n; r++) a[r] = get(r, c); return a; });
  if (data.some((a) => a.some((v) => !isFinite(v)))) throw new Error("The file contains non-numeric values in the signal columns.");
  if (!fs) throw new Error("Sampling rate is unknown: enter it in Hz (the file has no time column).");
  return { names: keep.map((i) => names[i]), data, fs: keep.map(() => fs), unitScale: 1, source: "CSV" };
}

export function parseEDF(buf: ArrayBuffer): RawRecording {
  const dv = new DataView(buf), td = new TextDecoder("ascii");
  const txt = (o: number, l: number) => td.decode(new Uint8Array(buf, o, l)).trim();
  if (buf.byteLength < 512 || txt(0, 8) !== "0") throw new Error("Not a valid EDF file (version field).");
  const nRec = parseInt(txt(236, 8)), dur = parseFloat(txt(244, 8)), ns = parseInt(txt(252, 4));
  if (!(nRec > 0) || !(dur > 0) || !(ns > 0)) throw new Error("EDF header is incomplete (continuous EDF+ with unknown record count is not supported).");
  let o = 256;
  const field = (len: number) => { const out: string[] = []; for (let i = 0; i < ns; i++) out.push(txt(o + i * len, len)); o += ns * len; return out; };
  const label = field(16); field(80); const dim = field(8), pmin = field(8).map(Number), pmax = field(8).map(Number), dmin = field(8).map(Number), dmax = field(8).map(Number); field(80);
  const nsamp = field(8).map(Number); field(32);
  const sig = label.map((l, i) => i).filter((i) => !/annotation/i.test(label[i]));
  const out = sig.map((i) => new Float64Array(nsamp[i] * nRec));
  let p = 256 + ns * 256;
  for (let r = 0; r < nRec; r++) for (let i = 0; i < ns; i++) {
    const idx = sig.indexOf(i), cnt = nsamp[i];
    if (idx >= 0) {
      const g = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]);
      for (let k = 0; k < cnt; k++) out[idx][r * cnt + k] = (dv.getInt16(p + 2 * k, true) - dmin[i]) * g + pmin[i];
    }
    p += 2 * cnt;
  }
  // convert to µV
  sig.forEach((i, idx) => { const d = dim[i].toLowerCase(); const s = d.startsWith("mv") ? 1e3 : d === "v" ? 1e6 : 1; if (s !== 1) for (let k = 0; k < out[idx].length; k++) out[idx][k] *= s; });
  return { names: sig.map((i) => label[i]), data: out, fs: sig.map((i) => nsamp[i] / dur), unitScale: 1, source: "EDF" };
}

// ── analysis ────────────────────────────────────────────────────────────────────
export interface PairResult { mi: Float32Array[]; meanMI: number[]; z: number[]; p: number[] }
export interface RecordingAnalysis {
  names: string[]; matched: (string | null)[]; nEpochs: number; seconds: number; flagged: boolean[][];
  tg: PairResult; ab: PairResult; resampled: boolean; signals: Float64Array[];
}

/** Analyse a recording: resample to 256 Hz, zero-phase band-pass, Hilbert phase/envelope, Tort MI per 2-s epoch, and a
 *  surrogate test (circular amplitude shifts) for each channel over the whole analysed segment. */
export function analyseRecording(rec: RawRecording, cfg: DspConfig, opt: { startSec: number; maxSec: number; include: boolean[]; artifactUv: number; nSurrogates: number; unit: number }): RecordingAnalysis {
  const FS = cfg.fs, L = cfg.epoch_len, rng = new RNG(12345);
  const idx = rec.names.map((_, i) => i).filter((i) => opt.include[i]);
  if (!idx.length) throw new Error("Select at least one channel.");
  const prepared = idx.map((i) => {
    let x = rec.data[i]; const f = rec.fs[i];
    const a = Math.floor(opt.startSec * f), b = Math.min(x.length, a + Math.floor(opt.maxSec * f));
    x = x.slice(a, b);
    for (let k = 0; k < x.length; k++) x[k] *= opt.unit;
    return Math.abs(f - FS) > 1e-6 ? resample(x, f, FS) : x;
  });
  const E = Math.floor(Math.min(...prepared.map((x) => x.length)) / L);
  if (E < 2) throw new Error("Need at least 4 seconds of data in the selected window.");
  const nS = E * L, nCh = prepared.length;
  const out: Record<"tg" | "ab", PairResult> = { tg: { mi: [], meanMI: [], z: [], p: [] }, ab: { mi: [], meanMI: [], z: [], p: [] } };
  const flagged: boolean[][] = [];
  const signals: Float64Array[] = [];
  const pairs = [["tg", "theta", "gamma"], ["ab", "alpha", "beta"]] as const;
  prepared.forEach((x0) => {
    const x = x0.slice(0, nS);
    signals.push(x);
    const fl = Array.from({ length: E }, (_, e) => { let m = 0; for (let i = e * L; i < (e + 1) * L; i++) m = Math.max(m, Math.abs(x[i])); return m > opt.artifactUv; });
    flagged.push(fl);
    for (const [key, pb, ab] of pairs) {
      const fp = filtfiltZeroPhase(cfg.sos[pb], x), fa = filtfiltZeroPhase(cfg.sos[ab], x);
      const ph = new Float64Array(nS), am = new Float64Array(nS), mi = new Float32Array(E), good: number[] = [];
      for (let e = 0; e < E; e++) {
        const [pr, pi] = analytic(fp.subarray(e * L, (e + 1) * L)), [ar, ai] = analytic(fa.subarray(e * L, (e + 1) * L));
        for (let i = 0; i < L; i++) { ph[e * L + i] = Math.atan2(pi[i], pr[i]); am[e * L + i] = Math.hypot(ar[i], ai[i]); }
        mi[e] = modulationIndex(ph.subarray(e * L, (e + 1) * L), am.subarray(e * L, (e + 1) * L), cfg.n_bins);
        if (!fl[e]) good.push(e);
      }
      // use only clean epochs for the channel summary and the surrogate test
      const cleanPh: number[] = [], cleanAm: number[] = [];
      for (const e of good) for (let i = 0; i < L; i++) { cleanPh.push(ph[e * L + i]); cleanAm.push(am[e * L + i]); }
      const P = Float64Array.from(cleanPh), A = Float64Array.from(cleanAm);
      const obs = P.length ? modulationIndex(P, A, cfg.n_bins) : NaN;
      let mu = 0, sd = 0, ge = 0; const sur: number[] = [];
      if (P.length > 4 * FS) for (let s = 0; s < opt.nSurrogates; s++) {
        const lag = Math.floor(rng.uniform(FS, P.length - FS)), sh = new Float64Array(A.length);
        for (let i = 0; i < A.length; i++) sh[i] = A[(i + lag) % A.length];
        const v = modulationIndex(P, sh, cfg.n_bins); sur.push(v); if (v >= obs) ge++;
      }
      if (sur.length) { mu = sur.reduce((a, b) => a + b, 0) / sur.length; sd = Math.sqrt(sur.reduce((a, b) => a + (b - mu) ** 2, 0) / sur.length) + 1e-12; }
      out[key].mi.push(mi); out[key].meanMI.push(obs);
      out[key].z.push(sur.length ? (obs - mu) / sd : NaN); out[key].p.push(sur.length ? (ge + 1) / (sur.length + 1) : NaN);
    }
  });
  const known = cfg.channels;
  return { names: idx.map((i) => rec.names[i]), matched: idx.map((i) => canonicalName(rec.names[i], known)), nEpochs: E, seconds: E * 2, flagged, tg: out.tg, ab: out.ab, resampled: idx.some((i) => Math.abs(rec.fs[i] - FS) > 1e-6), signals };
}

export function toCSV(a: RecordingAnalysis): string {
  const rows = ["channel,matched_10_20,theta_gamma_MI,theta_gamma_z,theta_gamma_p,alpha_beta_MI,alpha_beta_z,alpha_beta_p,rejected_epochs"];
  a.names.forEach((n, c) => rows.push([n, a.matched[c] ?? "", a.tg.meanMI[c], a.tg.z[c], a.tg.p[c], a.ab.meanMI[c], a.ab.z[c], a.ab.p[c], a.flagged[c].filter(Boolean).length].map((v) => (typeof v === "number" ? (Number.isFinite(v) ? v.toPrecision(5) : "") : v)).join(",")));
  rows.push("", "epoch," + a.names.map((n) => `${n}_TG`).join(",") + "," + a.names.map((n) => `${n}_AB`).join(","));
  for (let e = 0; e < a.nEpochs; e++) rows.push([e + 1, ...a.tg.mi.map((r) => r[e].toPrecision(5)), ...a.ab.mi.map((r) => r[e].toPrecision(5))].join(","));
  return rows.join("\n");
}
