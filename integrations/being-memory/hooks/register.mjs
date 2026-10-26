// Native Claude Code Mod. No Node imports, provider calls or permission changes.
const ID = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

async function config($) {
  const agent = await $.env.get("LICORICE_BEING_AGENT");
  const store = await $.env.get("LICORICE_BEING_STORE");
  const relationship = await $.env.get("LICORICE_BEING_RELATIONSHIP");
  const python = (await $.env.get("LICORICE_BEING_PYTHON")) || "python3";
  const backend = (await $.env.get("LICORICE_BEING_BACKEND")) || "native";
  const binary = await $.env.get("LICORICE_BEING_BIN");
  const models = await $.env.get("LICORICE_BEING_MODELS");
  const dimension = (await $.env.get("LICORICE_BEING_DIM")) || "256";
  const episodes = (await $.env.get("LICORICE_BEING_EPISODES")) || "0";
  if (!agent && !store) return null;
  if (!ID.test(agent || "") || typeof store !== "string" || !store.startsWith("/")) {
    throw new Error("invalid explicit memory configuration");
  }
  if (relationship && !ID.test(relationship)) throw new Error("invalid relationship");
  if (!["native", "prototype"].includes(backend)) throw new Error("unknown memory backend");
  if (backend === "native" && (typeof binary !== "string" || !binary.startsWith("/"))) throw new Error("set the native binary explicitly");
  if (!/^[1-9][0-9]{0,4}$/.test(dimension) || Number(dimension) > 65536) throw new Error("invalid embedding dimension");
  if (models && !models.startsWith("/")) throw new Error("models path must be absolute");
  if (!["0", "1"].includes(episodes)) throw new Error("episodes setting must be0or1");
  return { agent, store, relationship, python, backend, binary, models, dimension, includeEpisodes:episodes==="1" };
}

async function call($, cfg, command, payload) {
  const argv = cfg.backend === "native"
    ? [cfg.binary, command, "--store", cfg.store, "--agent", cfg.agent, "--embed-dim", cfg.dimension]
    : [cfg.python, `${$.plugin.root}/skills/being-memory/scripts/memory.py`, "--store", cfg.store, "--agent", cfg.agent, command];
  if (command === "context" && cfg.relationship) argv.push("--relationship", cfg.relationship);
  if (cfg.backend === "native" && command === "context" && cfg.models) argv.push("--models", cfg.models);
  let input = payload;
  if (cfg.backend === "native" && command === "context" && cfg.includeEpisodes) {
    input = { query: "current commitments and shared work", budget_tokens: 2000, ...payload, include_episodes: true };
  }
  if (cfg.backend === "native" && command === "apply") {
    if (payload.kind !== "boundary") throw new Error("automatic mod writes only boundary metadata");
    argv.push("--principal", "claude-code", "--role", "harness", "--session", payload.payload.session_id);
    input = { request_id: payload.request_id, expected_revision: null,
      operation: { kind: "boundary", payload: { turn: payload.payload.turn_id, phase: payload.payload.phase } } };
  }
  const result = await $.process.run(argv, {
    stdin: input ? JSON.stringify(input) : "",
    timeoutMs: 1500,
  });
  if (result.exitCode !== 0 || result.stdout.length > 1000000) throw new Error("memory process rejected request");
  const parsed = JSON.parse(result.stdout);
  if (parsed.ok !== true) throw new Error("memory process rejected request");
  return cfg.backend === "native" ? parsed.result : parsed;
}

function validatePacket(packet, cfg) {
  if (!packet || typeof packet.text !== "string") throw new Error("missing memory text");
  if (cfg.backend === "prototype") {
    if (new TextEncoder().encode(packet.text).length > 2000) throw new Error("oversized prototype packet");
  } else {
    const usage = packet.usage;
    if (!usage || usage.tokenizer !== "cl100k_base:ordinary" || !Number.isInteger(usage.tokens)
      || !Number.isInteger(usage.budget_tokens) || usage.tokens < 0 || usage.tokens > usage.budget_tokens
      || usage.budget_tokens > 2000 || usage.budget_tokens < 0) throw new Error("invalid native context budget receipt");
  }
}

async function notice($) {
  // Keep provider/user payloads and process stderr out of public notices.
  try { await $.ui.log("BEING memory unavailable; continuing without recalled state. Check the selected store, agent and native binary settings."); }
  catch { /* Memory diagnostics must not turn a tool or prompt into a retry. */ }
}

async function rememberBoundary($, e, phase) {
  const cfg = await config($);
  if (!cfg) return;
  const session = await $.session.id();
  let turn = e.turnId;
  if (!turn && phase === "compacting") {
    if (!Array.isArray(e.messages)) throw new Error("missing compaction transcript");
    // Opaque engine handles can change on replay. Fingerprint the semantic input
    // locally and retain only the digest, never its transcript or instructions.
    const material = { agent: e.agentId ?? null, trigger: e.trigger,
      instructions: e.instructions ?? null,
      messages: e.messages.map(m => ({ role: m.role, text: m.text, toolUses: m.toolUses || [], toolResults: m.toolResults || [] })) };
    const canonical = JSON.stringify(material, (_, value) =>
      value && typeof value === "object" && !Array.isArray(value)
        ? Object.fromEntries(Object.keys(value).sort().map(key => [key, value[key]])) : value);
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical));
    turn = `compact-${Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("")}`;
  }
  if (!ID.test(session) || !ID.test(turn)) throw new Error("unsupported host identity");
  const requestId = `b-${session}-${turn}-${phase}`;
  if (!ID.test(requestId)) throw new Error("host identity exceeds prototype limit");
  await call($, cfg, "apply", {
    request_id: requestId,
    kind: "boundary",
    payload: { session_id: session, turn_id: turn, phase },
  });
}

export function register(on) {
  on("session.start", async ($, e, next) => {
    await $.command.register({ name: "being-memory", description: "Inspect the configured local BEING memory without a model call" });
    return next(e);
  });

  on("command.run", { command: "being-memory" }, async ($) => {
    try {
      const cfg = await config($);
      if (!cfg) return { text: "Set LICORICE_BEING_AGENT, LICORICE_BEING_STORE and LICORICE_BEING_BIN, then explicitly initialize the native store." };
      const packet = await call($, cfg, "context");
      validatePacket(packet, cfg);
      return { text: packet.text };
    } catch {
      return { text: "BEING memory unavailable. Check store identity and initialization; no state was adopted." };
    }
  });

  on("prompt.submit", async ($, e, next) => {
    let forwarded = e;
    try {
      const cfg = await config($);
      if (cfg) {
        const input = cfg.backend === "native" ? { query: Array.from(e.text || "current shared work").slice(0, 2000).join(""), at: await $.clock.now(), budget_tokens: 2000 } : undefined;
        const packet = await call($, cfg, "context", input);
        validatePacket(packet, cfg);
        forwarded = { ...e, context: [...(e.context || []), packet.text] };
      }
    } catch { await notice($); }
    // Preserve the original user's text/origin and all prior mod context.
    return next(forwarded);
  });

  on("turn.complete", async ($, e, next) => {
    const result = await next(e);
    try { await rememberBoundary($, e, "completed"); }
    catch { await notice($); }
    // Only boundary metadata is persisted. Reflection/adoption remains explicit.
    return result;
  });

  on("session.compact", async ($, e, next) => {
    try { await rememberBoundary($, e, "compacting"); }
    catch { await notice($); }
    return next(e);
  });
}
