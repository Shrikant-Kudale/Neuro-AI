import { analytic, filtfiltZeroPhase, SOS } from "./dsp";

export interface DspConfig {
  fs: number; n_samples: number; epoch_len: number; n_epochs: number; n_bins: number; artifact_uv: number;
  channels: string[]; coords: number[][]; sos: Record<"theta" | "gamma" | "alpha" | "beta", SOS>;
}

/** Tort et al. (2010) Modulation Index: 1 − H(P)/log N for the mean amplitude per phase bin. */
export function modulationIndex(phase: Float64Array, amp: Float64Array, nBins: number): number {
  const tot = new Float64Array(nBins), cnt = new Float64Array(nBins);
  const w = (2 * Math.PI) / nBins;
  for (let i = 0; i < phase.length; i++) {
    let b = Math.floor((phase[i] + Math.PI) / w);
    if (b < 0) b = 0; else if (b >= nBins) b = nBins - 1;
    tot[b] += amp[i]; cnt[b] += 1;
  }
  let s = 0;
  const m = new Float64Array(nBins);
  for (let b = 0; b < nBins; b++) { m[b] = cnt[b] > 0 ? tot[b] / cnt[b] : 0; s += m[b]; }
  let h = 0;
  for (let b = 0; b < nBins; b++) { const p = m[b] / (s + 1e-12); h -= p * Math.log(p + 1e-12); }
  return (Math.log(nBins) - h) / Math.log(nBins);
}

export interface PacResult {
  tg: Float32Array[];     // [channel][epoch]
  ab: Float32Array[];
  flagged: boolean[][];   // [channel][epoch]
  phaseAmp?: { theta: number[]; gamma: number[] };
}

function interpolateFlagged(row: Float32Array, bad: boolean[]): void {
  const good: number[] = [];
  bad.forEach((b, i) => { if (!b) good.push(i); });
  if (good.length === 0 || good.length === bad.length) return;
  for (let i = 0; i < row.length; i++) {
    if (!bad[i]) continue;
    let lo = -1, hi = -1;
    for (const g of good) { if (g < i) lo = g; if (g > i && hi < 0) hi = g; }
    if (lo < 0) row[i] = row[hi];
    else if (hi < 0) row[i] = row[lo];
    else row[i] = row[lo] + ((row[hi] - row[lo]) * (i - lo)) / (hi - lo);
  }
}

/** eeg[channel] (length n_samples) → theta–gamma and alpha–beta MI per (channel, epoch). */
export function pacMatrix(eeg: Float64Array[], cfg: DspConfig, reject = true): PacResult {
  const nCh = eeg.length, E = cfg.n_epochs, L = cfg.epoch_len;
  const tg = Array.from({ length: nCh }, () => new Float32Array(E));
  const ab = Array.from({ length: nCh }, () => new Float32Array(E));
  const flagged = Array.from({ length: nCh }, () => new Array<boolean>(E).fill(false));
  const pairs: [keyof DspConfig["sos"], keyof DspConfig["sos"], Float32Array[]][] = [["theta", "gamma", tg], ["alpha", "beta", ab]];
  for (let c = 0; c < nCh; c++) {
    for (const [phB, amB, out] of pairs) {
      const fp = filtfiltZeroPhase(cfg.sos[phB], eeg[c]);
      const fa = filtfiltZeroPhase(cfg.sos[amB], eeg[c]);
      for (let e = 0; e < E; e++) {
        const [pr, pi] = analytic(fp.subarray(e * L, (e + 1) * L));
        const [ar, ai] = analytic(fa.subarray(e * L, (e + 1) * L));
        const ph = new Float64Array(L), am = new Float64Array(L);
        for (let i = 0; i < L; i++) { ph[i] = Math.atan2(pi[i], pr[i]); am[i] = Math.hypot(ar[i], ai[i]); }
        out[c][e] = modulationIndex(ph, am, cfg.n_bins);
      }
    }
    if (reject) {
      for (let e = 0; e < E; e++) {
        let mx = 0;
        for (let i = e * L; i < (e + 1) * L; i++) mx = Math.max(mx, Math.abs(eeg[c][i]));
        flagged[c][e] = mx > cfg.artifact_uv;
      }
      if (flagged[c].some(Boolean)) { interpolateFlagged(tg[c], flagged[c]); interpolateFlagged(ab[c], flagged[c]); }
    }
  }
  return { tg, ab, flagged };
}

/** Channel×epoch MI matrices + behaviour → (15 tokens × 41) feature matrix (row-major Float32Array). */
export function toTokens(tg: Float32Array[], ab: Float32Array[], beh: number[]): Float32Array {
  const nCh = tg.length, E = tg[0].length, D = 2 * nCh + 3;
  const X = new Float32Array(E * D);
  for (let e = 0; e < E; e++) {
    for (let c = 0; c < nCh; c++) { X[e * D + c] = tg[c][e]; X[e * D + nCh + c] = ab[c][e]; }
    for (let j = 0; j < 3; j++) X[e * D + 2 * nCh + j] = beh[j];
  }
  return X;
}

/** Inverse of the 570-vector layout used by the Python exporter: [tg(19×15) | ab(19×15)] + 3 behaviour. */
export function tokensFromVector(v: ArrayLike<number>, nCh = 19, E = 15): { X: Float32Array; tg: Float32Array[]; ab: Float32Array[]; beh: number[] } {
  const tg = Array.from({ length: nCh }, (_, c) => Float32Array.from({ length: E }, (_, e) => v[c * E + e]));
  const ab = Array.from({ length: nCh }, (_, c) => Float32Array.from({ length: E }, (_, e) => v[nCh * E + c * E + e]));
  const beh = [v[2 * nCh * E], v[2 * nCh * E + 1], v[2 * nCh * E + 2]];
  return { X: toTokens(tg, ab, beh), tg, ab, beh };
}
