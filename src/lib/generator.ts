/** Virtual-subject generator — TypeScript twin of research/neuroai/generator.py. */
import { RNG, clip } from "./rng";
import { analytic, bandNoise } from "./dsp";

export type Pair = [number, number];
export interface Profile {
  description: string;
  beh: Record<string, { etvs: Pair; gai: Pair; sscs: Pair }>;
  tg_depth: Pair; ab_depth: Pair;
  n_affected: Record<string, number>;
  tg_scale: Record<string, Pair>; ab_scale: Record<string, Pair>;
  gamma_boost: Record<string, number>;
  latent_sd: number; temporal: boolean; beh_sd_mult: number; fatigue_sd: number;
  impedance_sd: number; artifact_rate: number; white_sd: number; spatial: string;
}
export interface GenConfig {
  fs: number; n_samples: number; nfft_synth: number; epoch_len: number; n_epochs: number; duration: number;
  coords: number[][]; component_amp: Record<string, Pair | number>; rms_uv: Pair; coupling_gain: number; max_depth: number;
  bands: Record<string, Pair>; profiles: Record<string, Profile>;
}
export type Lesion = { x: number; y: number; sigma: number };

/** All numbers needed to synthesise one subject; the Studio edits these directly. */
export interface SubjectParams {
  cls: number;
  severity: number; fatigue: number;
  env: number[];                  // disruption intensity per epoch [0,1]
  w: number[];                    // per-channel involvement [0,1]
  dTg: number; dAb: number;       // healthy coupling parameters (paper units)
  red: number;                    // fractional theta–gamma loss in affected channels
  gain: number;                   // fractional alpha–beta gain in affected channels
  gammaBoost: number;             // gamma power multiplier in affected channels (1 = none)
  jitTg: number[]; jitAb: number[];
  beh: [number, number, number];
  whiteSd: number; rms: number; impedance: number[];
  artifacts: { t0: number; amp: number }[];
}

const mean = (p: Pair) => p[0];

export function sampleParams(cfg: GenConfig, profileName: string, cls: number, rng: RNG,
                             opt: { spatial?: string; lesion?: Lesion; noiseMult?: number } = {}): SubjectParams {
  const prof = cfg.profiles[profileName];
  const NC = cfg.coords.length, E = cfg.n_epochs;
  const K = String(cls);
  let f = 1;
  if (prof.latent_sd > 0 && cls > 0) f = clip(1 + prof.latent_sd * rng.normal(), 0, 1.8);
  const fatigue = prof.fatigue_sd > 0 ? rng.normal(0, prof.fatigue_sd) : 0;

  let env = new Array(E).fill(1);
  if (prof.temporal && cls > 0) {
    const base = cls === 1 ? 0.55 : 0.85;
    const z = new Array(E).fill(0);
    z[0] = rng.normal();
    for (let t = 1; t < E; t++) z[t] = 0.7 * z[t - 1] + Math.sqrt(1 - 0.49) * rng.normal();
    env = z.map((v) => clip(base + 0.3 * v, 0, 1));
  }

  const w = new Array(NC).fill(0);
  const spatial = opt.spatial ?? prof.spatial;
  if (cls > 0) {
    if (spatial === "focal") {
      let les = opt.lesion;
      if (!les) {
        const ang = rng.uniform(0, 2 * Math.PI), rad = rng.uniform(0.2, 0.7);
        les = { x: rad * Math.cos(ang), y: rad * Math.sin(ang), sigma: cls === 2 ? 0.45 : 0.3 };
      }
      for (let c = 0; c < NC; c++) {
        const d2 = (cfg.coords[c][0] - les.x) ** 2 + (cfg.coords[c][1] - les.y) ** 2;
        w[c] = Math.exp(-d2 / (2 * les.sigma ** 2));
      }
    } else {
      let n = prof.n_affected[K];
      if (prof.latent_sd > 0) n = clip(Math.round(n * (0.5 + 0.5 * f)), 1, NC);
      for (const i of rng.choice(NC, n)) w[i] = 1;
    }
  }

  const dTg = Math.max(0.02, rng.normal(prof.tg_depth[0], prof.tg_depth[1]));
  const dAb = Math.max(0.02, rng.normal(prof.ab_depth[0], prof.ab_depth[1]));
  const sTg = rng.uniform(prof.tg_scale[K][0], prof.tg_scale[K][1]);
  const sAb = rng.uniform(prof.ab_scale[K][0], prof.ab_scale[K][1]);
  const jitTg = Array.from({ length: NC }, () => clip(rng.normal(1, 0.1), 0.5, 1.5));
  const jitAb = Array.from({ length: NC }, () => clip(rng.normal(1, 0.1), 0.5, 1.5));
  const rms = rng.uniform(cfg.rms_uv[0], cfg.rms_uv[1]);
  const impedance = Array.from({ length: NC }, () => (prof.impedance_sd > 0 ? clip(rng.normal(1, prof.impedance_sd), 0.5, 1.6) : 1));

  const artifacts: { t0: number; amp: number }[] = [];
  if (prof.artifact_rate > 0) {
    const n = rng.binomial(E, prof.artifact_rate);
    for (let i = 0; i < n; i++) artifacts.push({ t0: rng.uniform(0.3, cfg.duration - 0.3), amp: rng.uniform(180, 320) });
  }

  const beh = sampleBehaviour(prof, cls, rng, f, fatigue);
  return {
    cls, severity: f, fatigue, env, w, dTg, dAb,
    red: (1 - sTg) * f, gain: (sAb - 1) * f, gammaBoost: 1 + (prof.gamma_boost[K] - 1) * f,
    jitTg, jitAb, beh, whiteSd: prof.white_sd * (opt.noiseMult ?? 1), rms, impedance, artifacts,
  };
}

export function sampleBehaviour(prof: Profile, cls: number, rng: RNG, f: number, fatigue: number): [number, number, number] {
  const out: number[] = [];
  for (const k of ["etvs", "gai", "sscs"] as const) {
    const m0 = mean(prof.beh["0"][k]);
    const [mc, sd] = prof.beh[String(cls)][k];
    out.push(rng.normal(m0 + (mc - m0) * f, sd * prof.beh_sd_mult) + fatigue * sd);
  }
  return [clip(out[0], 0.5, 4), clip(out[1], 0, 0.5), clip(out[2], 0, 1)];
}

/** Turn parameters into a 19-channel, 30 s recording (µV). */
export function synthesize(cfg: GenConfig, p: SubjectParams, rng: RNG): Float64Array[] {
  const N = cfg.n_samples, NC = cfg.coords.length, NF = cfg.nfft_synth, FS = cfg.fs, EL = cfg.epoch_len, E = cfg.n_epochs;
  const A = cfg.component_amp as Record<string, Pair>;
  const pinkAmp = cfg.component_amp.pink as number;
  const B = cfg.bands;
  const envT = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    const x = i / EL - 0.5;
    if (x <= 0) envT[i] = p.env[0];
    else if (x >= E - 1) envT[i] = p.env[E - 1];
    else { const i0 = Math.floor(x), fr = x - i0; envT[i] = p.env[i0] * (1 - fr) + p.env[i0 + 1] * fr; }
  }
  const ampOf = (name: string) => Math.max(0.05, rng.normal(A[name][0], A[name][1]));

  const out: Float64Array[] = [];
  const G = cfg.coupling_gain, MD = cfg.max_depth;
  for (let c = 0; c < NC; c++) {
    const delta = bandNoise(rng, B.delta[0], B.delta[1], NF, FS, N), theta = bandNoise(rng, B.theta[0], B.theta[1], NF, FS, N);
    const alpha = bandNoise(rng, B.alpha[0], B.alpha[1], NF, FS, N), beta = bandNoise(rng, B.beta[0], B.beta[1], NF, FS, N);
    const gamma = bandNoise(rng, B.gamma[0], B.gamma[1], NF, FS, N), pink = bandNoise(rng, 1, 100, NF, FS, N, 1);
    const ad = ampOf("delta"), at = ampOf("theta"), aa = ampOf("alpha"), ab = ampOf("beta"), ag = ampOf("gamma");
    const phase = (x: Float64Array) => {
      const pad = new Float64Array(NF); pad.set(x);
      const [re, im] = analytic(pad);
      const ph = new Float64Array(N);
      for (let i = 0; i < N; i++) ph[i] = Math.atan2(im[i], re[i]);
      return ph;
    };
    const phT = phase(theta), phA = phase(alpha);
    const w = p.w[c], gb = 1 + (p.gammaBoost - 1) * w;
    const sig = new Float64Array(N);
    let m = 0;
    for (let i = 0; i < N; i++) {
      const mTg = clip(G * p.dTg * p.jitTg[c] * Math.max(0, 1 - p.red * w * envT[i]), 0, MD);
      const mAb = clip(G * p.dAb * p.jitAb[c] * (1 + p.gain * w * envT[i]), 0, MD);
      sig[i] = delta[i] * ad + theta[i] * at + alpha[i] * aa + beta[i] * ab * (1 + mAb * Math.cos(phA[i]))
        + gamma[i] * ag * gb * (1 + mTg * Math.cos(phT[i])) + pink[i] * pinkAmp + rng.normal(0, p.whiteSd);
      m += sig[i];
    }
    m /= N;
    let v = 0; for (let i = 0; i < N; i++) v += (sig[i] - m) ** 2;
    const k = (p.rms / (Math.sqrt(v / N) + 1e-12)) * p.impedance[c];
    for (let i = 0; i < N; i++) sig[i] *= k;
    out.push(sig);
  }
  // sporadic ocular artefacts on frontal leads (same loadings as the Python generator)
  const load = [1, 1, 0.6, 0.5, 0.4, 0.5, 0.6];
  for (const a of p.artifacts) {
    for (let c = 0; c < load.length; c++)
      for (let i = 0; i < N; i++) { const t = i / FS; out[c][i] += load[c] * a.amp * Math.exp(-0.5 * ((t - a.t0) / 0.12) ** 2); }
  }
  return out;
}
