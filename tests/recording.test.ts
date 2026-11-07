import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { dft } from "../src/lib/fft";
import { resample } from "../src/lib/resample";
import { parseCSV, parseEDF, analyseRecording, canonicalName } from "../src/lib/recording";
import { totalForAUC, totalForSensSpec, aucSE } from "../src/lib/planner";
import { RNG } from "../src/lib/rng";
import { bandNoise, analytic } from "../src/lib/dsp";

const cfg = JSON.parse(readFileSync("public/data/manifest.json", "utf8")).dsp;

describe("signal utilities", () => {
  it("Bluestein DFT equals the naive DFT for a non-power-of-two length", () => {
    const n = 150, r = new RNG(1), re = Float64Array.from({ length: n }, () => r.normal()), im = new Float64Array(n);
    const nr = new Float64Array(n), ni = new Float64Array(n);
    for (let k = 0; k < n; k++) for (let t = 0; t < n; t++) { const a = (-2 * Math.PI * k * t) / n; nr[k] += re[t] * Math.cos(a); ni[k] += re[t] * Math.sin(a); }
    const R = Float64Array.from(re), I = Float64Array.from(im); dft(R, I);
    for (let k = 0; k < n; k++) { expect(R[k]).toBeCloseTo(nr[k], 6); expect(I[k]).toBeCloseTo(ni[k], 6); }
    dft(R, I, true); for (let k = 0; k < n; k++) expect(R[k]).toBeCloseTo(re[k], 8);
  });
  it("resampling 500 Hz → 256 Hz preserves a 10 Hz sine", () => {
    const n = 5000, x = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * 10 * i) / 500));
    const y = resample(x, 500, 256), m = y.length;
    expect(m).toBe(Math.round((n * 256) / 500));
    let err = 0; for (let i = 100; i < m - 100; i++) err = Math.max(err, Math.abs(y[i] - Math.sin((2 * Math.PI * 10 * i) / 256)));
    expect(err).toBeLessThan(0.02);
  });
});

describe("file readers", () => {
  it("parses CSV with header, time column and infers the sampling rate", () => {
    const rows = ["time,Fp1,Cz"]; for (let i = 0; i < 100; i++) rows.push(`${(i / 250).toFixed(4)},${i},${-i}`);
    const r = parseCSV(rows.join("\n"));
    expect(r.names).toEqual(["Fp1", "Cz"]); expect(Math.round(r.fs[0])).toBe(250); expect(r.data[0].length).toBe(100);
  });
  it("round-trips a synthetic EDF file", () => {
    const ns = 2, nrec = 3, dur = 1, spr = 100, hdr = 256 + ns * 256, buf = new ArrayBuffer(hdr + nrec * ns * spr * 2), u8 = new Uint8Array(buf);
    const put = (o: number, s: string, l: number) => { s.padEnd(l).split("").forEach((c, i) => (u8[o + i] = c.charCodeAt(0))); };
    put(0, "0", 8); put(236, String(nrec), 8); put(244, String(dur), 8); put(252, String(ns), 4); put(184, String(hdr), 8);
    let o = 256; const f = (len: number, vals: string[]) => { vals.forEach((v, i) => put(o + i * len, v, len)); o += ns * len; };
    f(16, ["EEG Fp1-REF", "EEG Cz-REF"]); f(80, ["", ""]); f(8, ["uV", "uV"]); f(8, ["-500", "-500"]); f(8, ["500", "500"]); f(8, ["-32768", "-32768"]); f(8, ["32767", "32767"]); f(80, ["", ""]); f(8, [String(spr), String(spr)]); f(32, ["", ""]);
    const dv = new DataView(buf); let p = hdr;
    for (let r = 0; r < nrec; r++) for (let s = 0; s < ns; s++) for (let k = 0; k < spr; k++) { dv.setInt16(p, Math.round((s ? -1 : 1) * 10000 * Math.sin((2 * Math.PI * 5 * (r * spr + k)) / spr)), true); p += 2; }
    const rec = parseEDF(buf);
    expect(rec.names).toEqual(["EEG Fp1-REF", "EEG Cz-REF"]); expect(rec.fs[0]).toBe(100); expect(rec.data[0].length).toBe(300);
    expect(Math.max(...rec.data[0])).toBeCloseTo((10000 / 32767.5) * 500 * 1.0, 0);
    expect(canonicalName("EEG Fp1-REF", cfg.channels)).toBe("Fp1"); expect(canonicalName("P7", cfg.channels)).toBe("T5");
  });
});

describe("coupling analysis", () => {
  it("surrogate test separates coupled from uncoupled gamma", () => {
    const fs = 256, n = fs * 30, r = new RNG(3);
    const make = (depth: number) => {
      const th = bandNoise(r, 4, 8, 8192, fs, n), ga = bandNoise(r, 30, 80, 8192, fs, n);
      const pad = new Float64Array(8192); pad.set(th); const [re, im] = analytic(pad);
      const x = new Float64Array(n);
      for (let i = 0; i < n; i++) x[i] = 10 * th[i] + 3 * (1 + depth * Math.cos(Math.atan2(im[i], re[i]))) * ga[i] + r.normal(0, 0.5);
      return x;
    };
    const rec = { names: ["Cz", "Pz"], data: [make(0.9), make(0)], fs: [fs, fs], unitScale: 1, source: "t" };
    const a = analyseRecording(rec, cfg, { startSec: 0, maxSec: 30, include: [true, true], artifactUv: 150, nSurrogates: 100, unit: 1 });
    expect(a.tg.meanMI[0]).toBeGreaterThan(5 * a.tg.meanMI[1]);
    expect(a.tg.p[0]).toBeLessThan(0.05); expect(a.tg.p[1]).toBeGreaterThan(0.05);
  }, 60000);
});

describe("study planner", () => {
  it("matches published reference values", () => {
    expect(aucSE(0.8, 50, 50)).toBeCloseTo(0.0445, 3);          // hand calculation: Q1=0.667, Q2=0.711, Var=4.951/2500
    const s = totalForSensSpec(0.9, 0.9, 0.05, 0.5);
    expect(s.cases).toBe(139);                                   // 1.96²·0.09/0.0025 = 138.3 → 139
    expect(totalForAUC(0.8, 0.07, 0.2)).toBeGreaterThan(100);
  });
});
