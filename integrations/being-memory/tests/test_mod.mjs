import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { register } from "../hooks/register.mjs";

function host(env = {}) {
  const hooks = new Map(), calls = [], logs = [];
  register((name, matcher, handler) => hooks.set(name, handler || matcher));
  const $ = {
    env: { get: async (name) => env[name] },
    plugin: { root: "/plugin path" },
    command: { register: async (value) => calls.push({ command: value }) },
    session: { id: async () => "session-1" },
    clock: { now: async () => 1234 },
    ui: { log: async (text) => logs.push(text) },
    process: { run: async (argv, options) => {
      calls.push({ argv, options });
      return { exitCode: 0, stdout: JSON.stringify({ ok: true, text: "Attributed memory data.", revision: 0 }), stderr: "" };
    } },
  };
  return { $, hooks, calls, logs };
}
const configured = { LICORICE_BEING_BACKEND: "prototype", LICORICE_BEING_AGENT: "michelle", LICORICE_BEING_STORE: "/private store/being.sqlite" };

test("unconfigured mod forwards a prompt once without process or provider calls", async () => {
  const h = host();
  const e = Object.freeze({ text: "A real user prompt.", origin: { kind: "user" } });
  let n = 0;
  const result = await h.hooks.get("prompt.submit")(h.$, e, (x) => { n++; return x; });
  assert.equal(result, e);
  assert.equal(n, 1);
  assert.equal(h.calls.length, 0);
});

test("context preserves original prompt, origin, prior context and paths with spaces", async () => {
  const h = host(configured);
  const e = Object.freeze({ text: "Original.", origin: { kind: "plugin", name: "other" }, context: ["Prior context"] });
  const result = await h.hooks.get("prompt.submit")(h.$, e, (x) => x);
  assert.equal(result.text, e.text);
  assert.equal(result.origin, e.origin);
  assert.deepEqual(result.context, ["Prior context", "Attributed memory data."]);
  assert.equal(h.calls[0].argv[1], "/plugin path/skills/being-memory/scripts/memory.py");
  assert.ok(h.calls[0].argv.includes("/private store/being.sqlite"));
  assert.equal(h.calls[0].options.stdin, "");
});

test("bad configuration, process failure and oversized unicode packets degrade visibly", async () => {
  for (const mode of ["config", "process", "packet"]) {
    const h = host(mode === "config" ? { ...configured, LICORICE_BEING_AGENT: "../other" } : configured);
    if (mode === "process") h.$.process.run = async () => { throw new Error("private secret must not be logged"); };
    if (mode === "packet") h.$.process.run = async () => ({ exitCode: 0, stdout: JSON.stringify({ ok: true, text: "界".repeat(800) }) });
    const e = { text: "unchanged" };
    let n = 0;
    const result = await h.hooks.get("prompt.submit")(h.$, e, (x) => { n++; return x; });
    assert.equal(result, e);
    assert.equal(n, 1);
    assert.equal(h.logs.length, 1);
    assert.ok(!h.logs[0].includes("private secret"));
  }
});

test("downstream prompt failure is propagated without a duplicate next call", async () => {
  const h = host(configured);
  let n = 0;
  await assert.rejects(h.hooks.get("prompt.submit")(h.$, {}, () => { n++; throw new Error("downstream"); }), /downstream/);
  assert.equal(n, 1);
});

test("turn boundary captures only metadata, keeps downstream result and uses stable replay ID", async () => {
  const h = host(configured);
  const e = { turnId: "turn-1", text: "private answer that must not be captured" };
  const result = { done: true };
  assert.equal(await h.hooks.get("turn.complete")(h.$, e, () => result), result);
  await h.hooks.get("turn.complete")(h.$, e, () => result);
  const a = JSON.parse(h.calls[0].options.stdin), b = JSON.parse(h.calls[1].options.stdin);
  assert.deepEqual(a, b);
  assert.equal(a.kind, "boundary");
  assert.deepEqual(a.payload, { session_id: "session-1", turn_id: "turn-1", phase: "completed" });
  assert.ok(!h.calls[0].options.stdin.includes("private answer"));
});

test("compaction continues and status command does not submit a model prompt", async () => {
  const h = host(configured);
  let n = 0;
  await h.hooks.get("session.compact")(h.$, { trigger: "manual", messages: [{ role: "user", text: "Keep our working agreement.", toolUses: [] }] }, () => { n++; return {}; });
  assert.equal(n, 1);
  assert.equal(JSON.parse(h.calls[0].options.stdin).payload.phase, "compacting");
  const result = await h.hooks.get("command.run")(h.$);
  assert.equal(result.text, "Attributed memory data.");
});

test("boundary write failure preserves completed answer and emits a visible notice", async () => {
  const h = host(configured);
  h.$.process.run = async () => ({ exitCode: 1, stdout: "{}", stderr: "private failure" });
  const result = { text: "Completed answer" };
  assert.equal(await h.hooks.get("turn.complete")(h.$, { turnId: "turn-1" }, () => result), result);
  assert.equal(h.logs.length, 1);
});

test("Mod and skill CLI share actual SQLite state across fresh processes", async () => {
  const folder = mkdtempSync(`${tmpdir()}/being-memory-`);
  const root = fileURLToPath(new URL("../", import.meta.url)).replace(/\/$/, "");
  const store = `${folder}/shared.sqlite`;
  const script = `${root}/skills/being-memory/scripts/memory.py`;
  const prefix = [script, "--store", store, "--agent", "michelle"];
  const run = (command, payload) => {
    const r = spawnSync("python3", [...prefix, command], { input: payload ? JSON.stringify(payload) : "", encoding: "utf8" });
    assert.equal(r.status, 0, r.stderr || r.stdout);
    return JSON.parse(r.stdout);
  };
  try {
    run("init");
    run("apply", { request_id: "seed", kind: "propose", expected_revision: 0, payload: {
      claim_id: "method", category: "method", subject: "michelle", statement: "Preserve uncertainty through a context reset.",
      source_ids: [], credence: null, importance: 0.8, reason: "An explicit practice for this experiment.",
    } });
    run("apply", { request_id: "adopt", kind: "decide", expected_revision: 1, payload: {
      claim_id: "method", stance: "adopted", credence: null, source_ids: [], reason: "Choose this revisable practice.",
    } });
    const h = host({ ...configured, LICORICE_BEING_STORE: store });
    h.$.plugin.root = root;
    h.$.process.run = async (argv, options) => {
      const r = spawnSync(argv[0], argv.slice(1), { input: options.stdin, encoding: "utf8" });
      return { exitCode: r.status, stdout: r.stdout, stderr: r.stderr };
    };
    const forwarded = await h.hooks.get("prompt.submit")(h.$, { text: "Resume our work." }, (x) => x);
    assert.match(forwarded.context[0], /Preserve uncertainty through a context reset/);
    assert.match(forwarded.context[0], /"stance":"adopted"/);
    for (let i = 0; i < 2; i++) await h.hooks.get("turn.complete")(h.$, { turnId: "turn-1" }, () => ({}));
    const events = run("history").events;
    assert.equal(events.length, 3);
    assert.equal(events[2].request.kind, "boundary");
    assert.equal(h.logs.length, 0);
  } finally {
    rmSync(folder, { recursive: true, force: true });
  }
});
