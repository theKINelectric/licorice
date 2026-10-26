# Using separate Codex and Claude memories

The operator asked to put both integrations into ordinary use on 21 October 2026.
The native memory kernel is shared; identities and memory files are separate.

For an existing conversation, or a project outside Licorice, use the
[named-profile attach/resume utility](being-attach.md). It preserves the selected
project directory and saved conversation instead of using the single-project
launcher below.

| Harness | Agent | Relationship scope | Private store |
|---|---|---|---|
| Codex | `licorice-codex` | `operator-licorice-codex` | `data/being/licorice-codex.sqlite` |
| Claude Code | `licorice-claude` | `operator-licorice-claude` | `data/being/licorice-claude.sqlite` |

Each relationship ID selects a recall scope. A formal description of a relationship
is separately authored by the agent; the Claude store has no invented description,
standing method, commitment, gender profile or biography. Reading another agent's
work does not make it the reader's participation or endorsement.

## Start Claude Code

From the repository:

```sh
./scripts/claude-being --model claude-sonnet-5-5
```

Omit `--model` to use your normal Claude settings. The launcher selects the native
installation at `~/.local/bin/claude`, currently 2.1.287, and explicitly loads the
Mod from `integrations/being-memory`. PATH's older npm `claude` 1.0.34 fails under
Node 26 here; it is bypassed by this launcher. An explicit `LICORICE_CLAUDE_BIN`
can select another absolute executable supporting native Mods, version 2.1.287+.

The launcher validates `data/being/claude.json` and the store's bound owner before
starting Claude. It rejects the Codex identity, relationship scope, shared paths,
symlinks or hardlinks that alias the same memory file. It replaces inherited
`LICORICE_BEING_*` settings with the complete Claude binding, never a partial mix.
This is local routing and ownership validation, not OS isolation from arbitrary
file access.

Your existing Anthropic key is used from the environment or, if absent there,
only the `ANTHROPIC_API_KEY` entry in `.env`. The file is parsed as data and never
sourced as shell code. No other dotenv variables are loaded. No key is copied
into the binding, project settings or proof files. Existing tool permissions and
the normal Claude credential/provider precedence still apply. Interactive model
work uses the configured Anthropic account; recall and boundary callbacks make
no model calls. The qualification proxy below is not part of this launcher.

Local diagnostics make no model calls:

```sh
./scripts/claude-being --memory-status
./scripts/claude-being --memory-context
```

Inside Claude Code, `/being-memory` inspects the selected memory locally. The Mod
adds relevant attributed continuity before a user turn. Memory and factual evidence
share one 2,000-token cap counted with ordinary cl100k. It stores completion/compaction boundary
metadata, including boundaries of failed turns; a boundary is not evidence that
a task succeeded. It does not automatically save prompts, answers or beliefs.

## Ordinary work

Give each agent normal development work. After a consequential correction,
changed judgment or commitment outcome, it can retain the attributed event and
decide explicitly whether a standing position should change. No-change is valid.
Avoid performing a persona or forcing different opinions to manufacture a contrast.

Claude's project instructions are in `.claude/CLAUDE.md`; the bundled skill explains
observation, proposal, separate adoption/veto, versioned ontology and source rules.
When writing, use the selected agent and explicit actor/session metadata, read
the current revision, and review intervening events after a stale decision.

Codex continues using the discovered `being-memory` skill and its existing
`data/being/active.json` binding. Claude uses `data/being/claude.json` and its
environment selection. Never copy Codex's standing claims into Claude to make
it appear to have shared that history. A useful cross-agent finding can be read
and attributed as another agent's report, with a fresh decision if warranted.

Human recognition and usefulness belong alongside technical evidence. Occasional
reviews can ask whether each agent kept a promise, retained a correction, changed
for an intelligible reason or imported another's experience incorrectly. These
are ordinary observations, not a scheduled automation or a personhood score.

## Activation evidence and limits

Receipts are in `runs/being-claude-20261021`. Three launcher controls, eight Node
Mod compatibility controls and strict native Mod validation pass. The real native
Claude host injected its own memory and stored Harness boundaries. Bidirectional
foreign-owner writes were refused against disposable native database copies.
Codex's original 14 events and binding were unchanged during setup. The only later
Codex write deliberately records its own observed activation outcome at revision 15.
Claude is at revision 10, with its own user request, eight boundary events and a
separately attributed first real model response. No stance has been pre-adopted.

The user explicitly approved a setup review and fresh recovery using Sonnet 5.5,
within $0.25. The first request was rejected because the local verification proxy
dropped Claude's beta headers; the same request passed the official free token
count with headers restored. The first actual generation exhausted 1,024 tokens
in adaptive reasoning and returned no final review. The second, at low effort
with 4,096 tokens available, returned its correct identity and source ID, and said
that no method had been adopted. The prompt disclosed the absence of an adoption,
so that negative stance is not a blind behavioral result.

The completed response initially hit a text-only relay defect. A corrected local
replay handled the thinking block and delivered the exact saved final text through
a fresh Claude host, recording its boundary. No further generation was made.
The replay's substantive request matched the live request; only the continuity
revision changed after boundary events. Synthetic host success, live provider
response and local replay are separate proof levels. This is not a claim of an
uninterrupted successful live round trip or a completed first code review.

Paid usage is $0.026184. The first HTTP400 retains its full $0.0846 reservation because
the generation-error body was not retained and billing has not been independently
reconciled. Conservative task accounting is $0.110784, leaving $0.65977933 from the
inherited allocation. No cloud resources, background jobs or additional model
calls are running. The failed attempts remain preserved. Ordinary future work
can supply Claude's first deliberate stance when it has a reason to choose one.

Fresh joined packets at final verification used **1,895 tokens for Codex** and
**602 tokens for Claude**, each under its own shared continuity/evidence cap.
The current verifier is `python3 runs/being-claude-20261021/verify.py`; its receipts
are a snapshot of activation, and ordinary later work will advance these journals.
Earlier five-phase manifests and verifiers remain historical snapshots.
