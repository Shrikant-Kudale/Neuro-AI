import { fft } from "./fft";
import { RNG } from "./rng";

export const PAD = 512;
export type SOS = number[][];  // rows [b0,b1,b2,a0,a1,a2]

/** scipy.signal.sosfilt (transposed direct-form II, zero initial state). */
export function sosfilt(sos: SOS, x: Float64Array): Float64Array {
  let y = Float64Array.from(x);
  for (const [b0, b1, b2, a0, a1, a2] of sos) {
    let z1 = 0, z2 = 0;
    for (let i = 0; i < y.length; i++) {
      const xi = y[i] / a0;
      const o = b0 * xi + z1;
      z1 = b1 * xi - (a1 / a0) * o + z2;
      z2 = b2 * xi - (a2 / a0) * o;
      y[i] = o;
    }
  }
  return y;
}

/** Zero-phase forward–backward SOS filter with odd-extension padding (mirrors dsp.filtfilt_zero_phase). */
export function filtfiltZeroPhase(sos: SOS, x: Float64Array): Float64Array {
  const n = x.length;
  const ext = new Float64Array(n + 2 * PAD);
  for (let i = 0; i < PAD; i++) ext[i] = 2 * x[0] - x[PAD - i];
  ext.set(x, PAD);
  for (let i = 0; i < PAD; i++) ext[PAD + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  let y = sosfilt(sos, ext);
  y.reverse();
  y = sosfilt(sos, y);
  y.reverse();
  return y.slice(PAD, PAD + n);
}

/** Analytic signal via FFT; returns [re, im] (length must be a power of two). */
export function analytic(x: ArrayLike<number>): [Float64Array, Float64Array] {
  const n = x.length;
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = x[i];
  fft(re, im);
  const half = n >> 1;
  for (let i = 1; i < half; i++) { re[i] *= 2; im[i] *= 2; }
  for (let i = half + 1; i < n; i++) { re[i] = 0; im[i] = 0; }
  fft(re, im, true);
  return [re, im];
}

/** Band-limited Gaussian-phase noise, spectral amplitude ∝ f^-slope, unit RMS over the first nOut samples. */
export function bandNoise(rng: RNG, lo: number, hi: number, nfft: number, fs: number, nOut: number, slope = 0): Float64Array {
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  const df = fs / nfft;
  for (let k = 0; k <= nfft / 2; k++) {
    const f = k * df;
    if (f < lo || f > hi) continue;
    const a = Math.pow(Math.max(f, 0.5), -slope);
    const ph = rng.uniform(0, 2 * Math.PI);
    re[k] = a * Math.cos(ph); im[k] = a * Math.sin(ph);
    if (k > 0 && k < nfft / 2) { re[nfft - k] = re[k]; im[nfft - k] = -im[k]; }
  }
  im[0] = 0; im[nfft / 2] = 0;
  fft(re, im, true);
  const out = re.slice(0, nOut);
  let m = 0; for (let i = 0; i < nOut; i++) m += out[i]; m /= nOut;
  let v = 0; for (let i = 0; i < nOut; i++) v += (out[i] - m) ** 2;
  const sd = Math.sqrt(v / nOut) + 1e-12;
  for (let i = 0; i < nOut; i++) out[i] /= sd;
  return out;
}
