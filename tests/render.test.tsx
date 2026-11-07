// @vitest-environment jsdom
import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { render, screen, waitFor, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "../src/App";

beforeAll(() => {
  (globalThis as any).fetch = async (url: string) => {
    const buf = readFileSync("public" + url);
    return { ok: true, json: async () => JSON.parse(buf.toString()), arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
  };
  const ctx = new Proxy({}, { get: (_t, k) => (k === "measureText" ? () => ({ width: 10 }) : () => undefined), set: () => true });
  (HTMLCanvasElement.prototype as any).getContext = () => ctx;
  window.scrollTo = () => {};
  (window as any).matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.requestAnimationFrame = () => 0;
});

const routes: [string, RegExp][] = [["/", /How to use this workbench/], ["/lab", /Classifier output/], ["/analyse", /Load a recording/], ["/cohort", /The simulated cohort/], ["/results", /Model comparison/], ["/planner", /Participants required/], ["/method", /Responsible use/]];
describe("every page renders against the exported data", () => {
  for (const [path, text] of routes) {
    it(path, async () => {
      render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
      await waitFor(() => expect(screen.getAllByText(text).length).toBeGreaterThan(0), { timeout: 20000 });
      cleanup();
    }, 30000);
  }
});
