#!/usr/bin/env node
/**
 * Focused, dependency-free (beyond installed TypeScript) regression checks of
 * the actual api-jobs.ts module. No backend or browser server is required.
 * Compile to CommonJS in memory and control only its transport and timers.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, "../lib/api-jobs.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  reportDiagnostics: true,
});
const errors = compiled.diagnostics?.filter((item) => item.category === ts.DiagnosticCategory.Error) ?? [];
assert.equal(errors.length, 0, "Job module must transpile cleanly");

function job(status, extra = {}) {
  return {
    job_id: "job-1", kind: "automatic_set", status, progress: status === "completed" ? 1 : 0,
    stage: "planning", message: "", error: null, result: null, ...extra,
  };
}

function harness({ responses = [], sse = true } = {}) {
  const paths = [];
  const timers = new Map();
  const sources = [];
  let sequence = 0;
  const queue = [...responses];
  const transport = {
    API_BASE: "",
    request: async (path, init) => {
      paths.push({ path, method: init?.method ?? "GET" });
      if (!queue.length) throw new Error(`Unexpected request: ${path}`);
      const next = queue.shift();
      if (next instanceof Error) throw next;
      return next;
    },
  };
  class FakeEventSource {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.closed = false;
      sources.push(this);
    }
    close() { this.closed = true; }
    emit(value) { this.onmessage?.({ data: typeof value === "string" ? value : JSON.stringify(value) }); }
    fail() { this.onerror?.(new Error("disconnect")); }
  }
  const exports = {};
  vm.runInNewContext(compiled.outputText, {
    exports,
    require: (name) => {
      assert.equal(name, "./api-transport");
      return transport;
    },
    EventSource: sse ? FakeEventSource : undefined,
    window: {
      setInterval: (callback) => { const id = ++sequence; timers.set(id, callback); return id; },
      clearInterval: (id) => timers.delete(id),
    },
    setTimeout: (callback) => { queueMicrotask(callback); },
    Promise,
    Error,
    JSON,
  }, { filename: "api-jobs.cjs" });
  return { api: exports, paths, sources, timers, tick: async () => {
    for (const callback of [...timers.values()]) await callback();
  } };
}

test("already completed job returns without requests or SSE, notifying once", async () => {
  const h = harness();
  const updates = [];
  const initial = job("completed");
  assert.deepEqual(await h.api.waitForJob(initial, (value) => updates.push(value.status)), initial);
  assert.deepEqual(updates, ["completed"]);
  assert.equal(h.paths.length, 0);
  assert.equal(h.sources.length, 0);
});

test("failed and cancelled initial jobs preserve distinct errors", async () => {
  const h = harness();
  const updates = [];
  await assert.rejects(h.api.waitForJob(job("failed", { message: "unsafe" }), (value) => updates.push(value.status)), /unsafe/);
  await assert.rejects(h.api.waitForJob(job("cancelled", { message: "stopped" }), (value) => updates.push(value.status)),
    (error) => error.name === "JobCancelledError" && error.message === "stopped");
  assert.deepEqual(updates, ["failed", "cancelled"]);
  assert.equal(h.sources.length, 0);
});

test("HTTP polling fallback preserves status updates and server completion", async () => {
  const h = harness({ sse: false, responses: [job("running", { progress: 0.4 }), job("completed")] });
  const updates = [];
  const result = await h.api.waitForJob(job("queued"), (value) => updates.push(value.status), 1);
  assert.equal(result.status, "completed");
  assert.deepEqual(updates, ["queued", "running", "completed"]);
  assert.deepEqual(h.paths.map((p) => p.path), ["/api/jobs/job-1", "/api/jobs/job-1"]);
});

test("HTTP polling cancellation rejects with JobCancelledError", async () => {
  const h = harness({ sse: false, responses: [job("cancelled", { message: "cancel confirmed" })] });
  await assert.rejects(h.api.waitForJob(job("running"), undefined, 1),
    (error) => error.name === "JobCancelledError" && error.message === "cancel confirmed");
});

test("SSE sends partial progress but final authoritative GET determines completion", async () => {
  const h = harness({ responses: [job("completed", { message: "server done" })] });
  const updates = [];
  const pending = h.api.waitForJob(job("running"), (value) => updates.push(value));
  assert.equal(h.sources.length, 1);
  assert.equal(h.sources[0].options.withCredentials, true);
  assert.equal(h.sources[0].url, "/api/jobs/job-1/events");
  h.sources[0].emit({ status: "running", progress: 0.7, stage: "rehearsal" });
  assert.equal(updates.at(-1).progress, 0.7);
  h.sources[0].emit({ status: "completed" });
  const final = await pending;
  assert.equal(final.message, "server done");
  assert.equal(h.sources[0].closed, true);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(h.paths.map((p) => p.path), ["/api/jobs/job-1"]);
});

test("SSE disconnect falls back to polling and closes source", async () => {
  const h = harness({ responses: [job("running", { progress: 0.5 }), job("completed")] });
  const updates = [];
  const pending = h.api.waitForJob(job("queued"), (value) => updates.push(value.status), 1);
  h.sources[0].emit("invalid json");
  h.sources[0].fail();
  const final = await pending;
  assert.equal(final.status, "completed");
  assert.equal(h.sources[0].closed, true);
  assert.equal(h.timers.size, 0);
  assert.deepEqual(updates, ["queued", "queued", "running", "completed"]);
});

test("SSE terminal failure respects final server rejection", async () => {
  const h = harness({ responses: [job("failed", { error: "render safety rejected" })] });
  const pending = h.api.waitForJob(job("running"));
  h.sources[0].emit({ status: "failed", message: "rejected" });
  await assert.rejects(pending, /render safety rejected/);
  assert.equal(h.timers.size, 0);
});

test("cancellation action remains an explicit POST, never synthetic success", async () => {
  const returned = job("cancelled");
  const h = harness({ responses: [returned] });
  const result = await h.api.cancelJob("job-1");
  assert.deepEqual(result, returned);
  assert.deepEqual(h.paths, [{ path: "/api/jobs/job-1/cancel", method: "POST" }]);
});

test("diagnostic timer uses canonical job GET, not SSE payload as authority", async () => {
  const h = harness({ responses: [job("completed", { message: "confirmed" }), job("completed", { message: "confirmed" })] });
  const pending = h.api.waitForJob(job("running"));
  await h.tick();
  const result = await pending;
  assert.equal(result.status, "completed");
  assert.equal(h.sources[0].closed, true);
  assert.equal(h.timers.size, 0);
});
