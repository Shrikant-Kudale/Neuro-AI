/** Sample-size mathematics for diagnostic-accuracy studies. */

/** Hanley & McNeil (1982) standard error of an AUC estimate. */
export function aucSE(A: number, nPos: number, nNeg: number): number {
  const q1 = A / (2 - A), q2 = (2 * A * A) / (1 + A);
  const v = (A * (1 - A) + (nPos - 1) * (q1 - A * A) + (nNeg - 1) * (q2 - A * A)) / (nPos * nNeg);
  return Math.sqrt(Math.max(v, 0));
}
/** Smallest total N (cases = N·prevalence) whose 95% CI half-width on the AUC is ≤ `halfWidth`. */
export function totalForAUC(A: number, halfWidth: number, prevalence: number, z = 1.96): number {
  for (let n = 20; n < 200000; n += 2) {
    const pos = Math.max(2, Math.round(n * prevalence)), neg = Math.max(2, n - pos);
    if (z * aucSE(A, pos, neg) <= halfWidth) return n;
  }
  return NaN;
}
/** Half-width of the 95% CI for given totals (used for the planning curve). */
export function halfWidth(A: number, N: number, prevalence: number, z = 1.96): number {
  const pos = Math.max(2, Math.round(N * prevalence)), neg = Math.max(2, N - pos);
  return z * aucSE(A, pos, neg);
}
/** Buderer (1996): subjects needed to estimate sensitivity and specificity with a given half-width. */
export function totalForSensSpec(sens: number, spec: number, halfW: number, prevalence: number, z = 1.96) {
  const nCases = Math.ceil((z * z * sens * (1 - sens)) / (halfW * halfW));
  const nControls = Math.ceil((z * z * spec * (1 - spec)) / (halfW * halfW));
  return { cases: nCases, controls: nControls, total: Math.ceil(Math.max(nCases / prevalence, nControls / (1 - prevalence))) };
}
export const inflate = (n: number, loss: number) => Math.ceil(n / (1 - loss));
