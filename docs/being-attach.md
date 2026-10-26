# Attach memory to an existing conversation

`scripts/being-session` resumes a saved local Codex or Claude Code conversation
with a named memory profile and an explicit project directory. It runs locally,
using the existing native memory CLI and separate SQLite files. It adds no server.

The initial private registry is `data/being/profiles.json`, mode 0600. It contains
two profiles, `licorice-codex` and `licorice-claude`, referencing their original
memories. Registration copies a complete binding into the profile; it neither
imports history nor creates a personality. Later editing a source binding JSON
does not silently change the registered profile.

## Resume with an explicit project

First stop the existing client cleanly so its saved conversation can be resumed
in a new process. The utility does not inject a Mod into an already running
desktop chat or replace a running process's environment.

```sh
./scripts/being-session profiles

./scripts/being-session attach \
  --profile licorice-codex \
  --project /absolute/path/to/project \
  --resume "SAVED_SESSION_ID"

./scripts/being-session attach \
  --profile licorice-claude \
  --project /absolute/path/to/project \
  --resume "SAVED_SESSION_ID"
```

Use the utility's absolute path when calling from another project:

```sh
./scripts/being-session attach \
  --profile licorice-claude \
  --project /absolute/path/to/that/project \
  --resume "SAVED_SESSION_ID"
```

Add `--dry-run` to inspect the selected profile, owner revision, project, command
arguments and memory environment without launching the conversation. This reads
no provider key and makes no model call. `--model MODEL_ID` is optional; omission
leaves model selection to the client. Executing attach submits one visible memory
attachment instruction in the resumed conversation, so normal provider billing
applies when the client processes it. This is an appended configuration turn,
not a rewrite of earlier messages or a historical memory import.

Use `--last` instead of `--resume` to select the client's latest saved conversation
for the chosen directory. This maps to `codex resume --last` or Claude's
`--continue`. Exact session UUIDs are preferable: they identify the conversation
and allow stable profile routing. Names are accepted as literal client selectors,
but multiple names can refer to a conversation and duplicate names remain a
client selection issue. Transcript paths and arbitrary client flags are refused.

## Identity and project routing

Each profile fixes one harness, agent ID, relationship scope, memory file, native
binary, local models directory and client runtime. Different profiles cannot
share an agent identity, relationship scope or memory file, including symlink
and hardlink aliases. One continuing identity may use its profile in several
explicit projects; it does not need a new personality for every conversation.

Before launch, the utility checks the native store's bound owner and the runtime.
Qualified interfaces are Codex 0.163.0+ and Claude Code 2.1.287+. The provided
profiles select `/opt/homebrew/bin/codex` and the working native Claude at
`~/.local/bin/claude`; the old npm Claude on PATH is not used.

For Codex, the utility passes `--cd` and `tui.resume_cwd="current"`, keeping the
selected project instead of switching to a remembered session directory. It pins
only memory variables through `shell_environment_policy.set` and supplies the
absolute skill path and complete binding in the visible attachment instruction.
Even if a custom shell filter removes variables, that explicit binding remains
available for native CLI arguments. No global skill installation is required.

For Claude, it loads the native Mod explicitly and pins the selected memory
environment through `--settings`. This overrides conflicting project memory
defaults while leaving unrelated settings and permissions alone. The same
attachment instruction points to the bundled memory skill. Safe mode, which
disables Mods, is refused. An active matching background session is also refused:
Claude could otherwise attach to its old process or create a copy. Stop it first.

The utility preserves `CODEX_HOME`, `CLAUDE_CONFIG_DIR`, authentication selection
and ordinary client permissions. It does not grant additional writable roots or
bypass tool approval. A later memory write outside the selected project's allowed
paths may need the client's normal access approval. Local profile routing is not
OS isolation from arbitrary files.

An explicit session selector is pinned to its profile when a launch is attempted.
Later attachment through another profile is refused until explicitly unpinned.
UUID spelling is canonicalized; a name is pinned literally. `--last` has no known
session ID before client selection, so it cannot create this pin.

```sh
./scripts/being-session unpin --harness codex --resume "SAVED_SESSION_ID"
```

Unpinning removes only the local routing entry. It deletes no conversation or
memory. A pin is a routing record, not proof that authentication or resumption
subsequently succeeded.

## Register another identity

First initialize a distinct native memory and prepare its complete binding using
the [native protocol](../integrations/being-memory/skills/being-memory/references/native.md).
Then register it explicitly:

```sh
./scripts/being-session register my-claude \
  --harness claude \
  --binding /absolute/path/to/my-claude-binding.json \
  --runtime /absolute/path/to/compatible/claude

./scripts/being-session check --profile my-claude
```

Registration requires an existing correctly owned store; no persona or historical
claim is silently initialized. An identical registration is idempotent; a changed
selection under an existing name is refused. The registry is private local JSON,
updated with a lock and atomic replacement. Runtime/model path maintenance can be
made explicitly in this private file, followed by `check`; retain identity/store
fields and route meaning when doing so.

An optional `--credential-file /absolute/path/to/.env` selects an Anthropic dotenv
file for a Claude profile. Only `ANTHROPIC_API_KEY` is parsed, only at launch when
the environment lacks it. The file is never sourced or copied into the registry.
Profiles without that selection keep normal client authentication. The provided
Claude profile retains the already configured Licorice credential file; the
Codex profile keeps its existing Codex authentication.

An alternate registry uses a global option before the command:

```sh
./scripts/being-session --registry /absolute/path/to/profiles.json profiles
```

## Local verification

Fifteen new contract tests use real native memory stores and offline client
fixtures. They cover ownership, aliases, atomic refusal, profile conflicts,
session pins, UUID normalization, explicit working directories, client home
preservation, literal paths, credentials, safe mode and running-session refusal.
The three original launcher controls also pass.

Two disposable real Claude processes, against a loopback-only synthetic service,
created and resumed a saved conversation from a different project. The same
session ID, original transcript prefix and earlier messages survived. The request
reported the selected directory and contained only the correct profile's memory,
despite deliberately conflicting project settings. The foreign store was unchanged.

The actual installed Codex resume parser accepts the complete generated command.
Its launch/history-routing contracts are checked with offline client fixtures;
no real Codex chat was resumed and no live model behavior is claimed. The Codex
skill still governs deliberate memory use; it is not a per-turn Codex callback.
Each native recall retains the existing 2,000-token continuity/evidence allowance.

Receipts are in `runs/being-attach-20261021`. Actual Codex/Claude journals and the
spending ledger are checked against the entry baseline. No paid calls, cloud
resources or persistent service are needed for this verification. The earlier
single-project `scripts/claude-being` launcher remains supported.


Recheck this snapshot with:

```sh
python3 -m unittest discover -s scripts -p test_being_sessions.py -v
python3 -m unittest discover -s scripts -p test_claude_being.py -v
python3 scripts/verify-being-attach.py
```

The separate verifier checks retained synthetic transcript bytes and boundary
receipts, actual profile/native owners and journal hashes, context token counts,
source hashes and the unchanged spending ledger. The original journals are at
Codex revision15 and Claude revision10. Historical snapshot verifiers may need
new receipts after ordinary work legitimately advances these journals.

The installed client interfaces were checked against the official
[Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
and [Claude CLI reference](https://code.claude.com/docs/en/cli-reference).
