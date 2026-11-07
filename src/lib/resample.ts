import { dft } from "./fft";

/** Band-limited (FFT) resampling to `fsOut`; content above the new Nyquist is discarded (ideal anti-alias).
 *  A linear trend is removed first so the implicit periodic extension does not ring at the edges. */
export function resample(x: Float64Array, fsIn: number, fsOut: number): Float64Array {
  const n1 = x.length;
  if (Math.abs(fsIn - fsOut) < 1e-9) return Float64Array.from(x);
  const n2 = Math.max(2, Math.round((n1 * fsOut) / fsIn));
  const a = x[0], b = (x[n1 - 1] - x[0]) / Math.max(1, n1 - 1);
  const re = new Float64Array(n1), im = new Float64Array(n1);
  for (let i = 0; i < n1; i++) re[i] = x[i] - (a + b * i);
  dft(re, im);
  const yr = new Float64Array(n2), yi = new Float64Array(n2);
  const h = Math.min(n1, n2) >> 1;
  for (let k = 0; k <= h; k++) { yr[k] = re[k]; yi[k] = im[k]; }
  for (let k = 1; k < h; k++) { yr[n2 - k] = re[n1 - k]; yi[n2 - k] = im[n1 - k]; }
  const s = n2 / n1;
  for (let k = 0; k < n2; k++) { yr[k] *= s; yi[k] *= s; }
  dft(yr, yi, true);
  const out = new Float64Array(n2), step = (n1 - 1) / Math.max(1, n2 - 1);
  for (let i = 0; i < n2; i++) out[i] = yr[i] + a + b * (i * step);
  return out;
}
