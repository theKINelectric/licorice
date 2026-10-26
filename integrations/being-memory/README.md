# Licorice BEING memory — native integration package

The Codex skill and native Claude Code Mod share Licorice's Rust event/stance
kernel. The [architecture](../../docs/being-memory.md) defines all five authorized
phases; [native status](../../docs/being-native.md) records what is implemented
and its limits. The initial five-phase build is complete. Separate Codex and
Claude identities are now configured for [ordinary use](../../docs/being-daily-use.md).

## Native setup

Build `cargo build --locked -p licorice --bin licorice-being`. Select an absolute
binary/store path and stable identity:

```sh
export LICORICE_BEING_BIN='/absolute/path/to/licorice-being'
export LICORICE_BEING_AGENT='chosen-agent-id'
export LICORICE_BEING_STORE='/absolute/private/path/chosen-agent.licorice'
"$LICORICE_BEING_BIN" init --store "$LICORICE_BEING_STORE" --agent "$LICORICE_BEING_AGENT"
```

The native backend is the Mod's default. Optional settings are
`LICORICE_BEING_RELATIONSHIP`, `LICORICE_BEING_DIM` (default256), and
`LICORICE_BEING_MODELS` (absolute local WordLlama directory). Without models,
recall explicitly returns continuity only; with models, it includes source-aware
factual retrieval within the same2k cap. Optional `LICORICE_BEING_EPISODES=1` adds recent scoped observations without
adopting them, inside the same packet. Native context is counted as ordinary
cl100k tokens, including all memory notices and citations.

The caller supplies author/session metadata separately from typed requests.
Local file access remains authority; this is not an authenticated remote service.
New CLI stores are private0600files on Unix. Foreign/prototype SQLite is refused
before migration; explicit import validates original receipts and commits atomically.
See [the native protocol](skills/being-memory/references/native.md).

## Codex

`.agents/skills/being-memory` links to this package's `skills/being-memory`.
Invoke `$being-memory` or use normal skill discovery. A skill is a workflow,
not an internal Codex loop callback.

This repository has a private ignored binding at `data/being/active.json` for
the actual `licorice-codex` development collaboration. The skill may read it as
one complete binding when no explicit environment selection is supplied. Keep
different identities in different stores. Claude has its own ignored binding,
`data/being/claude.json`, and its own `licorice-claude.sqlite` store. The launcher
`scripts/claude-being` explicitly selects that binding and the native Mod.
For existing sessions and other project directories, `scripts/being-session`
selects a registered identity profile and resumes the saved conversation. See
[attach/resume usage](../../docs/being-attach.md); the profile's complete binding
takes precedence over repository defaults.

## Native Claude Code Mod

The package has `.claude-plugin/plugin.json`, `hooks/hooks.json` with `modules`,
and `hooks/register.mjs` exporting `register(on)`. It appends bounded recall while
preserving the user prompt/origin and prior context, records only completion/
compaction boundary metadata, and adds `/being-memory` for inspection. No callback
calls a provider, persists full prompts/answers, automatically adopts a belief,
forces a continuation or changes tool permissions.

On a compatible host:

```sh
claude plugin validate /absolute/path/to/licorice/integrations/being-memory --strict
claude plugin test /absolute/path/to/licorice/integrations/being-memory
claude --plugin-dir /absolute/path/to/licorice/integrations/being-memory
```

The npm CLI on PATH remains old, but a persistent native Claude2.1.287 is installed
at `~/.local/bin/claude`; `scripts/claude-being` selects it explicitly. The official
Claude2.1.287 was also isolated under
`/private/tmp/licorice-claude-2.1.287` for qualification, with npm SHA512 integrity
checked. Eight actual SDK tests and strict validation pass, including stable compaction
replay as the clock advances. No global installation
or configuration was changed. SDK host services are stubbed; actual native-binary/
SQLite transport is separately exercised. Two fresh real Claude processes also passed a full loop against a loopback-only
synthetic Anthropic service, proving native packet injection and boundary writes.
The separate30request GLM behavior trial passed all10checkpoints with BEING state
and with episodic history. The operator subsequently assessed the demonstrated
Codex continuity as a useful starting point. Claude now begins its own attributed
history; no Codex stance, personality or autobiography was copied into it. See
the daily-use report for live-response and relay qualification limits.

## Legacy prototype

`LICORICE_BEING_BACKEND=prototype` explicitly selects the old standard-library
Python sidecar for compatibility/inspection. Its packet limit remains2,000UTF-8
bytes. Do not pass a native database to it, mix receipt formats, or automatically
fall back after an error. The [legacy protocol](skills/being-memory/references/protocol.md)
remains available for existing sidecars and native import preparation.

## Local checks

```sh
python3 -m unittest discover -s integrations/being-memory/tests -p 'test_memory.py' -v
node --test integrations/being-memory/tests/test_mod.mjs
node scripts/being-native-mod-smoke.mjs /absolute/path/to/licorice-being /absolute/path/to/report.json
```

The first checks retained prototype semantics. Node checks compatibility and the
adapter. The explicit smoke uses the actual native binary and SQLite. Native
Claude tests are in `tests/native-mod.test.ts`; Rust kernel/context controls are
in `crates/licorice/tests/being_memory.rs` and `being_context.rs`.
