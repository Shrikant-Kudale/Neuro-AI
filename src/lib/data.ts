import { useEffect, useState } from "react";
import { NeuroTransformer } from "./model";
import type { GenConfig } from "./generator";
import type { DspConfig } from "./pac";

export type ProfileName = "sim";
export const PROFILE: ProfileName = "sim";
export const CLASS_NAMES = ["Normal", "Early Disruption", "High Risk"] as const;
export const CLASS_COLORS = ["var(--normal)", "var(--early)", "var(--high)"] as const;
export const CLASS_HEX = ["#2f7d4f", "#a8690f", "#b3362f"] as const;
export const BEH_NAMES = ["ETVS", "GAI", "SSCS"] as const;

const cache = new Map<string, Promise<any>>();
export function getJSON<T = any>(url: string): Promise<T> {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); }));
  return cache.get(url)!;
}
export function getBuffer(url: string): Promise<ArrayBuffer> {
  if (!cache.has(url)) cache.set(url, fetch(url).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.arrayBuffer(); }));
  return cache.get(url)!;
}
const modelCache = new Map<string, Promise<NeuroTransformer>>();
export function loadModel(profile: ProfileName) {
  if (!modelCache.has(profile)) modelCache.set(profile, NeuroTransformer.load(`/model/transformer_${profile}`));
  return modelCache.get(profile)!;
}

export interface Manifest { dsp: DspConfig & GenConfig & { class_names: string[]; beh_long: string[] }; generated: string; profiles: string[]; seeds: number[] }

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): { data: T | null; error: string | null } {
  const [state, set] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let live = true;
    set({ data: null, error: null });
    fn().then((d) => live && set({ data: d, error: null })).catch((e) => live && set({ data: null, error: String(e.message ?? e) }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
export const useManifest = () => useAsync<Manifest>(() => getJSON("/data/manifest.json"), []);
export const useResults = (p: ProfileName) => useAsync<any>(() => getJSON(`/data/results_${p}.json`), [p]);
export const useModel = (p: ProfileName) => useAsync<NeuroTransformer>(() => loadModel(p), [p]);

export const pct = (x: number, d = 1) => (100 * x).toFixed(d) + "%";
export const f3 = (x: number) => x.toFixed(3);
export const argmax = (a: ArrayLike<number>) => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
