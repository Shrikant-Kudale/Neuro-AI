import { RNG, clip } from "./rng";
import { sampleParams, synthesize, GenConfig, Lesion, SubjectParams } from "./generator";
import { pacMatrix, toTokens, DspConfig, PacResult } from "./pac";
import type { NeuroTransformer, Prediction } from "./model";
import type { ProfileName } from "./data";

export type Layout = "random" | "focal";
export type Temporal = "sustained" | "intermittent" | "ramp";
export interface StudioUI {
  cls: number; seed: number;
  red: number; gain: number; gboost: number;       // theta–gamma loss, alpha–beta gain (fractions), gamma power multiplier
  layout: Layout; nAff: number; lesion: Lesion;
  temporal: Temporal; beh: [number, number, number]; noise: number;
}
type Cfg = GenConfig & DspConfig;

export function presetUI(cfg: Cfg, profile: ProfileName, cls: number, seed: number): StudioUI {
  const p = sampleParams(cfg, profile, cls, new RNG(seed));
  const prof = cfg.profiles[profile];
  return {
    cls, seed, red: +p.red.toFixed(2), gain: +p.gain.toFixed(2), gboost: +p.gammaBoost.toFixed(2),
    layout: "random", nAff: p.w.filter((v) => v > 0.5).length, lesion: { x: -0.4, y: 0.45, sigma: 0.38 },
    temporal: prof.temporal && cls > 0 ? "intermittent" : "sustained", beh: p.beh.map((v) => +v.toFixed(3)) as [number, number, number], noise: 1,
  };
}

export function buildParams(cfg: Cfg, profile: ProfileName, ui: StudioUI): SubjectParams {
  const prof = cfg.profiles[profile], NC = cfg.coords.length, E = cfg.n_epochs;
  const base = sampleParams(cfg, profile, ui.cls, new RNG(ui.seed));
  const w = new Array(NC).fill(0);
  if (ui.layout === "focal") {
    for (let c = 0; c < NC; c++) { const d2 = (cfg.coords[c][0] - ui.lesion.x) ** 2 + (cfg.coords[c][1] - ui.lesion.y) ** 2; w[c] = Math.exp(-d2 / (2 * ui.lesion.sigma ** 2)); }
  } else for (const i of new RNG(ui.seed + 99).choice(NC, Math.round(ui.nAff))) w[i] = 1;
  let env: number[];
  if (ui.temporal === "sustained") env = new Array(E).fill(1);
  else if (ui.temporal === "ramp") env = Array.from({ length: E }, (_, t) => 0.2 + 0.8 * (t / (E - 1)));
  else { const r = new RNG(ui.seed + 5); const z = [r.normal()]; for (let t = 1; t < E; t++) z.push(0.7 * z[t - 1] + Math.sqrt(0.51) * r.normal()); env = z.map((v) => clip(0.7 + 0.3 * v, 0, 1)); }
  return { ...base, w, env, red: ui.red, gain: ui.gain, gammaBoost: ui.gboost, beh: ui.beh, whiteSd: prof.white_sd * ui.noise };
}

export interface Analysis {
  params: SubjectParams; eeg: Float64Array[]; pac: PacResult; raw: Float32Array; pred: Prediction; ms: number;
}
export function analyse(cfg: Cfg, model: NeuroTransformer, profile: ProfileName, ui: StudioUI): Analysis {
  const t0 = performance.now();
  const params = buildParams(cfg, profile, ui);
  const eeg = synthesize(cfg, params, new RNG(ui.seed * 7919 + 13));
  const pac = pacMatrix(eeg, cfg);
  const raw = toTokens(pac.tg, pac.ab, ui.beh);
  return { params, eeg, pac, raw, pred: model.predictRaw(raw), ms: performance.now() - t0 };
}

/** Re-score the same recording with new behavioural values (no re-synthesis needed). */
export function rescore(model: NeuroTransformer, a: Analysis, beh: number[]): Analysis {
  const raw = toTokens(a.pac.tg, a.pac.ab, beh);
  return { ...a, raw, pred: model.predictRaw(raw) };
}

export const mean = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
export const chMean = (m: Float32Array[]) => m.map((r) => mean(r));
