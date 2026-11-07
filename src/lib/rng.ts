/** Seeded PRNG (mulberry32) with Gaussian / uniform helpers. Deterministic across browsers. */
export class RNG {
  private s: number;
  private spare: number | null = null;
  constructor(seed: number) { this.s = seed >>> 0; }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(lo = 0, hi = 1): number { return lo + (hi - lo) * this.next(); }
  normal(mu = 0, sd = 1): number {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return mu + sd * v; }
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    while (v === 0) v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return mu + sd * r * Math.cos(2 * Math.PI * v);
  }
  /** Fisher–Yates sample of k distinct integers from [0, n). */
  choice(n: number, k: number): number[] {
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(this.next() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a.slice(0, k);
  }
  binomial(n: number, p: number): number { let c = 0; for (let i = 0; i < n; i++) if (this.next() < p) c++; return c; }
}
export const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
