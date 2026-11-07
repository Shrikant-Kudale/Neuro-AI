/** Forward pass of the trained NeuroTransformer in plain TypeScript (no ML runtime needed). */
export interface ModelManifest {
  arch: { in_dim: number; seq: number; d: number; heads: number; layers: number; ffn: number; n_classes: number };
  tensors: { name: string; shape: number[]; offset: number }[];
  scaler: { lo: number[]; scale: number[] };
  profile: string; fill: number[]; params: number; test_accuracy: number; class_names: string[]; seed: number;
}
export interface Prediction {
  probs: number[]; logits: number[]; pred: number;
  attnLast: Float32Array;       // heads × 16 × 16 (last layer)
  clsAttn: number[];            // CLS→epoch, mean over heads (15)
  rollout: number[];            // attention rollout CLS→epoch (15)
}

export class NeuroTransformer {
  private W: Record<string, Float32Array> = {};
  constructor(public manifest: ModelManifest, weights: Float32Array) {
    for (const t of manifest.tensors) {
      const n = t.shape.reduce((a, b) => a * b, 1);
      this.W[t.name] = weights.subarray(t.offset, t.offset + n);
    }
  }
  static async load(base: string): Promise<NeuroTransformer> {
    const [m, b] = await Promise.all([fetch(base + ".json").then((r) => r.json()), fetch(base + ".bin").then((r) => r.arrayBuffer())]);
    return new NeuroTransformer(m as ModelManifest, new Float32Array(b));
  }

  /** Min-max scale raw (15 × in_dim) tokens exactly as in training. */
  scale(raw: Float32Array): Float32Array {
    const D = this.manifest.scaler.lo.length, out = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) { const j = i % D; out[i] = (raw[i] - this.manifest.scaler.lo[j]) / this.manifest.scaler.scale[j]; }
    return out;
  }

  private linear(x: Float32Array, T: number, inD: number, w: Float32Array, b: Float32Array, outD: number): Float32Array {
    const y = new Float32Array(T * outD);
    for (let t = 0; t < T; t++) {
      for (let o = 0; o < outD; o++) {
        let s = b[o];
        const wo = o * inD, xo = t * inD;
        for (let i = 0; i < inD; i++) s += x[xo + i] * w[wo + i];
        y[t * outD + o] = s;
      }
    }
    return y;
  }
  private layerNorm(x: Float32Array, T: number, D: number, g: Float32Array, b: Float32Array): Float32Array {
    const y = new Float32Array(x.length);
    for (let t = 0; t < T; t++) {
      let m = 0; for (let i = 0; i < D; i++) m += x[t * D + i]; m /= D;
      let v = 0; for (let i = 0; i < D; i++) v += (x[t * D + i] - m) ** 2; v /= D;
      const inv = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < D; i++) y[t * D + i] = (x[t * D + i] - m) * inv * g[i] + b[i];
    }
    return y;
  }

  /** `scaled`: (15 × in_dim) already min-max scaled. */
  predictScaled(scaled: Float32Array): Prediction {
    const { d, heads, layers, ffn, seq, in_dim, n_classes } = this.manifest.arch;
    const T = seq + 1, dk = d / heads, W = this.W;
    const emb = this.linear(scaled, seq, in_dim, W["embed.weight"], W["embed.bias"], d);
    let h = new Float32Array(T * d);
    for (let i = 0; i < d; i++) h[i] = W["cls"][i];
    h.set(emb, d);
    for (let i = 0; i < T * d; i++) h[i] += W["pos"][i];
    const rolled: Float32Array[] = [];
    let attnLast = new Float32Array(heads * T * T);
    for (let l = 0; l < layers; l++) {
      const p = `blocks.${l}.`;
      const z = this.layerNorm(h, T, d, W[p + "ln1.weight"], W[p + "ln1.bias"]);
      const q = this.linear(z, T, d, W[p + "q.weight"], W[p + "q.bias"], d);
      const k = this.linear(z, T, d, W[p + "k.weight"], W[p + "k.bias"], d);
      const v = this.linear(z, T, d, W[p + "v.weight"], W[p + "v.bias"], d);
      const ctx = new Float32Array(T * d);
      const att = new Float32Array(heads * T * T);
      const sc = 1 / Math.sqrt(dk);
      for (let hd = 0; hd < heads; hd++) {
        for (let i = 0; i < T; i++) {
          const row = new Float32Array(T);
          let mx = -1e30;
          for (let j = 0; j < T; j++) {
            let s = 0;
            for (let u = 0; u < dk; u++) s += q[i * d + hd * dk + u] * k[j * d + hd * dk + u];
            row[j] = s * sc; if (row[j] > mx) mx = row[j];
          }
          let sum = 0; for (let j = 0; j < T; j++) { row[j] = Math.exp(row[j] - mx); sum += row[j]; }
          for (let j = 0; j < T; j++) { row[j] /= sum; att[(hd * T + i) * T + j] = row[j]; }
          for (let u = 0; u < dk; u++) { let s = 0; for (let j = 0; j < T; j++) s += row[j] * v[j * d + hd * dk + u]; ctx[i * d + hd * dk + u] = s; }
        }
      }
      const o = this.linear(ctx, T, d, W[p + "o.weight"], W[p + "o.bias"], d);
      for (let i = 0; i < T * d; i++) h[i] += o[i];
      const z2 = this.layerNorm(h, T, d, W[p + "ln2.weight"], W[p + "ln2.bias"]);
      const f1 = this.linear(z2, T, d, W[p + "f1.weight"], W[p + "f1.bias"], ffn);
      for (let i = 0; i < f1.length; i++) f1[i] = gelu(f1[i]);
      const f2 = this.linear(f1, T, ffn, W[p + "f2.weight"], W[p + "f2.bias"], d);
      for (let i = 0; i < T * d; i++) h[i] += f2[i];
      // head-mean attention for rollout
      const A = new Float32Array(T * T);
      for (let hd = 0; hd < heads; hd++) for (let i = 0; i < T * T; i++) A[i] += att[hd * T * T + i] / heads;
      rolled.push(A);
      if (l === layers - 1) attnLast = att;
    }
    const hf = this.layerNorm(h, T, d, W["ln_f.weight"], W["ln_f.bias"]);
    const cls = hf.subarray(0, d);
    const h1 = this.linear(cls, 1, d, W["head.0.weight"], W["head.0.bias"], 64);
    for (let i = 0; i < 64; i++) h1[i] = Math.max(0, h1[i]);
    const logits = Array.from(this.linear(h1, 1, 64, W["head.2.weight"], W["head.2.bias"], n_classes));
    const mx = Math.max(...logits), ex = logits.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((v) => v / s);

    const clsAttn = new Array(seq).fill(0);
    for (let hd = 0; hd < heads; hd++) for (let j = 1; j < T; j++) clsAttn[j - 1] += attnLast[(hd * T) * T + j] / heads;
    // attention rollout (Abnar & Zuidema): R = Π (0.5 A + 0.5 I), row-normalised
    let R: Float32Array | null = null;
    for (const A of rolled) {
      const M = new Float32Array(T * T);
      for (let i = 0; i < T; i++) { let rs = 0; for (let j = 0; j < T; j++) { M[i * T + j] = 0.5 * A[i * T + j] + (i === j ? 0.5 : 0); rs += M[i * T + j]; } for (let j = 0; j < T; j++) M[i * T + j] /= rs; }
      if (!R) R = M;
      else { const N2 = new Float32Array(T * T); for (let i = 0; i < T; i++) for (let k = 0; k < T; k++) { const a = M[i * T + k]; if (a) for (let j = 0; j < T; j++) N2[i * T + j] += a * R[k * T + j]; } R = N2; }
    }
    const rollout = Array.from({ length: seq }, (_, j) => R![j + 1]);
    return { probs, logits, pred: probs.indexOf(Math.max(...probs)), attnLast, clsAttn, rollout };
  }

  predictRaw(raw: Float32Array): Prediction { return this.predictScaled(this.scale(raw)); }

  /** Occlusion importance: drop in the predicted-class probability when a channel (both couplings) is replaced by its training mean. */
  channelOcclusion(raw: Float32Array, nCh: number, fill: number[], target: number): number[] {
    const base = this.predictRaw(raw).probs[target];
    const D = this.manifest.arch.in_dim, E = this.manifest.arch.seq;
    const out: number[] = [];
    for (let c = 0; c < nCh; c++) {
      const x = Float32Array.from(raw);
      for (let e = 0; e < E; e++) { x[e * D + c] = fill[c]; x[e * D + nCh + c] = fill[nCh + c]; }
      out.push(base - this.predictRaw(x).probs[target]);
    }
    return out;
  }
}

function gelu(x: number): number { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
/** erf with ~1e-7 absolute error (W. J. Cody-style rational approximation via Abramowitz–Stegun 7.1.26 refined). */
function erf(x: number): number {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
