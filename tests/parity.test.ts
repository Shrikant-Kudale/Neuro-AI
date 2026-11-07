import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { pacMatrix } from "../src/lib/pac";
import { NeuroTransformer } from "../src/lib/model";
import { RNG } from "../src/lib/rng";
import { sampleParams, synthesize } from "../src/lib/generator";

const manifest = JSON.parse(readFileSync("public/data/manifest.json", "utf8"));
const cfg = manifest.dsp;
const fx = JSON.parse(readFileSync("tests/fixtures/parity.json", "utf8"));
const buf = readFileSync("tests/fixtures/signal.f32");
const sig = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const eeg = Array.from({ length: 19 }, (_, c) => Float64Array.from(sig.subarray(c * 7680, (c + 1) * 7680)));

describe("Python ↔ TypeScript parity", () => {
  it("Modulation Index matches for every channel and epoch", () => {
    const r = pacMatrix(eeg, cfg);
    let worst = 0;
    for (let c = 0; c < 19; c++) for (let e = 0; e < 15; e++) {
      worst = Math.max(worst, Math.abs(r.tg[c][e] - fx.tg[c][e]), Math.abs(r.ab[c][e] - fx.ab[c][e]));
      expect(r.flagged[c][e]).toBe(fx.flagged[c][e]);
    }
    expect(worst).toBeLessThan(2e-4);
  });

  for (const prof of ["paper", "hard"]) {
    it(`Transformer (${prof}) logits and attention match PyTorch`, () => {
      const m = JSON.parse(readFileSync(`public/model/transformer_${prof}.json`, "utf8"));
      const w = readFileSync(`public/model/transformer_${prof}.bin`);
      const net = new NeuroTransformer(m, new Float32Array(w.buffer, w.byteOffset, w.byteLength / 4));
      const p = net.predictRaw(Float32Array.from(fx.raw));
      p.logits.forEach((v, i) => expect(Math.abs(v - fx.models[prof].logits[i])).toBeLessThan(2e-3));
      p.clsAttn.forEach((v, i) => expect(Math.abs(v - fx.models[prof].cls_attn[i])).toBeLessThan(2e-3));
    });
  }

  it("browser generator reproduces the class ordering of coupling", () => {
    const mean = (cls: number) => {
      let s = 0;
      for (let k = 0; k < 3; k++) {
        const p = sampleParams(cfg, "paper", cls, new RNG(50 + k));
        const r = pacMatrix(synthesize(cfg, p, new RNG(900 + k)), cfg);
        for (const row of r.tg) for (const v of row) s += v / (19 * 15);
      }
      return s / 3;
    };
    const [n, , h] = [mean(0), mean(1), mean(2)];
    expect(n).toBeGreaterThan(h);
  }, 60000);
});

describe("Browser generator → browser model (end-to-end distribution check)", () => {
  for (const prof of ["paper", "hard"]) {
    it(`${prof}: classification of freshly generated subjects is far above chance`, () => {
      const m = JSON.parse(readFileSync(`public/model/transformer_${prof}.json`, "utf8"));
      const w = readFileSync(`public/model/transformer_${prof}.bin`);
      const net = new NeuroTransformer(m, new Float32Array(w.buffer, w.byteOffset, w.byteLength / 4));
      let ok = 0, n = 0;
      for (let cls = 0; cls < 3; cls++) for (let k = 0; k < 8; k++) {
        const p = sampleParams(cfg, prof, cls, new RNG(7000 + cls * 100 + k));
        const r = pacMatrix(synthesize(cfg, p, new RNG(3000 + cls * 100 + k)), cfg);
        const raw = new Float32Array(15 * 41);
        for (let e = 0; e < 15; e++) { for (let c = 0; c < 19; c++) { raw[e * 41 + c] = r.tg[c][e]; raw[e * 41 + 19 + c] = r.ab[c][e]; } for (let j = 0; j < 3; j++) raw[e * 41 + 38 + j] = p.beh[j]; }
        if (net.predictRaw(raw).pred === cls) ok++;
        n++;
      }
      console.log(`${prof}: browser-generated accuracy ${ok}/${n}`);
      expect(ok / n).toBeGreaterThan(prof === "paper" ? 0.8 : 0.5);
    }, 120000);
  }
});
