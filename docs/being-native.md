# Native BEING implementation — initial five-phase build complete

Current date: 21 October 2026. All five authorized phases have passed their initial
acceptance, including actual Codex use and the operator’s usefulness assessment.
The Codex skill and native Claude Mod share the kernel. Relationships, commitment
lifecycle, versioned ontology tools, optional reflection and a small registered
model-behavior trial are implemented. This document records implementation and
limits; [the architecture](being-memory.md) explains the product intent. The operator answered “Yes, useful starting
point” about the recovered commitment and verification practice. That attributed
response closes the initial usefulness gate. The native promise is fulfilled at
revision14; its source history and the adopted practice remain available.

## Native kernel

`crates/licorice/src/being` exports `Memory`, typed `Actor`, `Request`, `Operation`,
observations/sources/assurance, proposals, decisions, snapshots and receipts.
Schema 8 adds a bound identity, event journal, observation/claim projections,
source closures, participation bindings and preserved sidecar import receipts.
It leaves existing facts and their lifecycle untouched. Upgrading schema 7 neither
guesses an agent identity nor endorses a factual assertion.

`Memory::initialize` explicitly binds one agent. `attach` requires that binding.
`apply` takes actor/session metadata separately from the request body and commits
event, source/claim projection and receipt in the existing single-writer transaction.
Only the matching agent can propose/decide its stance. A human report, a harness
receipt and an imported document retain their distinct origin/author metadata.
The local CLI trusts the process that supplies transport flags; arbitrary file
access remains authority. This is not a remote authentication or tamper-proofing
claim against the file owner.

Every proposal starts `considering`. Credence and endorsement remain separate.
A decision can adopt, suspend, decline or retract, retaining earlier decisions
and sources. Request replay binds agent, actor, session, operation and import
identity; changed data conflicts. New decisions require the current revision.
Finite weights are checked before hashing, including native NaN/Infinity values
which JSON would otherwise serialize as null. A regression witness for that
cache-alias defect is retained.

Canonical source observations must match exact stored turn text, role and time
within the declared session. Autobiographical participation requires an explicit
harness/session binding in a session containing an assistant, or a harness tool
receipt whose BLAKE3 digest binds the exact observation body. These checks establish
source identity/attestation, not semantic truth or entailment of every claim.
Reading an autobiography cannot become this agent's participation source.

## Import and recovery

`import_sidecar` reads the prototype database read-only, checks its application
ID/schema, agent, event sequence, bounded original bodies and SHA256 receipts.
It atomically imports into an empty native journal, preserving original body
bytes, timestamps, SHA256 and the one-to-one native event mapping. Exact replay
returns the original import receipt; later native events are not overwritten.
Invalid late operations roll back earlier events/projections from the same import.
Source assurance remains `legacy_import`. An uncertified historical autobiography
can still be suspended or declined; its evidence is not silently promoted.

The binary `licorice-being` supplies init,apply,snapshot,context,history and
import-sidecar. It refuses unrelated SQLite/prototype files before native schema
migration. New stores created by the CLI have 0600 permissions on Unix. Existing
native store permissions are not silently changed.

## One context allowance

`being::context::continuity` ranks complete, attributed claim/source units and
records selected/omitted IDs, revision and exact ordinary-cl100k usage. Confidence
is not used as endorsement. Missing source closure fails rather than fabricating
a proof. A unit that cannot fit is omitted whole; later smaller units may fit.

`Memory::retrieve_context` admits continuity before ordinary retrieval, with a
default 400-token reservation and one total allowance <= 2000. Unused continuity
space remains available to evidence. Source-aware recovery receives the same
prefix and counts it, its scope notice, source text and citations exactly once.
The result carries final text/usage, continuity trace, primary retrieval and final
source-selection/citation trace. Empty prefix retains the old retrieval path.
Both ordinary and fixed-summary packers count a nonempty prefix at admission.

After evidence selection, unused space can admit additional whole continuity
units. Earlier memory units stay protected, and the evidence suffix remains
byte-for-byte identical. This fixes large obligations missing a 400-token initial
reservation despite ample free space. The final combined recount still caps 2000.

CLI `context --models ABS` uses local WordLlama/source-aware factual retrieval.
Without models it explicitly returns `continuity_only`, using the available cap
for that packet. It does not pretend factual retrieval ran. See the skill's
[native protocol](../integrations/being-memory/skills/being-memory/references/native.md).

## Harness integration and real project use

The repository skill remains registered through `.agents/skills/being-memory`.
It now routes native operations. The old Python script is an explicit prototype
and import aid. The Mod defaults to native with an explicit absolute
`LICORICE_BEING_BIN`; no error triggers a fallback to another backend/store.
`LICORICE_BEING_BACKEND=prototype` is retained for old fixtures only.

The Mod's native context envelope checks the ordinary-cl100k budget receipt.
It forwards the original user prompt/origin/prior context unchanged and uses a
bounded local query projection for retrieval. Completion/compaction callbacks
write only harness boundary metadata and do not persist full prompts/answers,
adopt beliefs, call a provider or force continuations.

An official macOS-arm64 Claude 2.1.287 binary is isolated under
`/private/tmp/licorice-claude-2.1.287`. Its npm SHA512 integrity was verified;
installer scripts/global updates were not run. Native strict plugin validation
and six native SDK tests pass with credential-free isolated configuration.
These engine tests caught a real command matcher defect missed by the earlier
Node-only harness. They test actual Mod dispatch with stubbed host services;
they do not establish a live model turn or a model-generated compaction summary.
The binary-through-Mod smoke separately exercises actual native processes and
SQLite, preserving state and idempotent boundaries.

The skill has also been used in this project. Ignored/private local binding
`data/being/active.json` selects `licorice-codex`, its native store and relationship
`operator-licorice-codex`. It records the actual user goal, a verification receipt,
an explicit commitment to all five phases, and an adopted practice to verify
real integration boundaries and retain counterexamples. A fresh process recovered
both adopted records in 554 tokens. No fictional autobiography or psychological/
gender traits were seeded. This is a technical collaboration identity; it does
not imply subjective experience. Human usefulness feedback is requested separately.

## Optional episode recall after completion audit

The audit found that stored observations without a claim reference could not
appear in continuity recall. Native context now accepts `include_episodes:true`,
or `LICORICE_BEING_EPISODES=1` in the Claude Mod. Codex's private local binding
explicitly selects it. Default false preserves the frozen trial protocol.

A bounded view adds up to 128 recent observations from global or the selected
relationship, with a candidate-saturation receipt. Equal-relevance episodes prefer
recent journal order. Whole units are labelled `unadopted_observation` and retain
origin/subject/source assurance. They share the same 2k allowance, make no writes
and never become adopted claims simply by being recalled. The real project now
recovers the latest audit alongside its standing commitment and verification
practice in 1,961 joined tokens, at journal revision 11.

An actual Claude SDK replay control also caught clock-derived compaction IDs
changing for an identical callback. Compaction now hashes its semantic input
locally, excluding opaque message handles. The journal retains only the digest
and boundary metadata. An actual native-process/SQLite test confirms the replay
returns the original receipt and does not persist transcript text.

The current suite passes 280 workspace tests, all-target Clippy, eight actual
Claude SDK controls and strict validation. Two actual Claude processes qualify
explicit episode injection against a local synthetic service. No new paid calls
were made. This optional extension was not part of the frozen 30-request model
comparison; its original results remain historical rather than being relabelled
as a new score. See `native/post-audit-qualification.json` and the current verifier.

## Relationships, obligations and revisable ontology

Schema 9 adds explicit versioned relationships and structured obligations. A
relationship records this agent's perspective, members, shared-work description,
boundaries, sources and reasons. It never infers the other person's feelings or
consent. Obligations refer to adopted owner commitment claims and an explicit
counterparty. Active promises may be fulfilled, released or broken, and terminal
ones may be explicitly reopened. Outcome sources are retained; fulfillment cannot
rely on reflection alone, and release requires an attributed counterparty report.
The original decisions and earlier relationship versions remain in the journal.

Schema 10 preserves exact concept definitions by ID/version. Split and merge use
explicit lineage; dependent positions keep their original meaning until reviewed.
Typed links distinguish dependency, support, objection, contradiction and meaning
use. Links can be explicitly retired, preserving the original record. Bounded
impact review reports candidates and saturation without cascading stance changes.
Exclusive belief alternatives carry an explicit unknown allocation; compatible
propositions retain independent nullable weights. Every estimate is labelled
uncalibrated. Values and commitments cannot be normalized as empirical hypotheses.

Context selection admits complete concept/hypothesis/dependency/source closures.
Superseded concepts remain in dependent closures rather than masquerading as
current standalone meanings. Oversized closures are omitted as whole units with
an omission trace; they are never clipped into apparent proof.

## Optional reflection

Schema 11 stores reflection plans, source groups and retained reservations. The
native planning path makes no provider calls. It accepts an explicit consequential
trigger, protects whole observations inside the same2k memory allowance, adds
continuity, and emits a replayable `ModelRequest` with 512 output tokens. Four jobs
may reserve per UTC day per store. Renaming an identical body does not produce
novel evidence; reflection-only triggers do not schedule a recursive reflection.

The injected `ChatModel` execution API atomically claims a plan and invokes one
completion call. A failed or invalid attempt keeps its reservation; this layer
has no retries or automatic recursion. An external executor must enforce a
separate durable monetary ceiling and preserve at-most-once execution receipts.
The CLI `reflect-plan` and `reflect-complete` allow a host to use its own authorized
model path. Neither command grants spending authority.

A completion contains a concise conclusion, alternatives, literal trigger-source
quotations and optional existing positions to review. A changed snapshot, forged
quote, actor/session mismatch, altered replay or excess output is refused. Even
an accepted review becomes an attributed observation; stance/weight changes still
require a separate deliberate decision. No hidden reasoning transcript is requested.

## Model behavior and evidence

The registered synthetic trial has 10 checkpoints across method revision, promise
lifecycle, imported-book attribution and relationship scope. All 30 requests use
GLM-5.3-Flash, identical system/style/settings and fresh model contexts. Memory is
capped 2,000 ordinary-cl100k tokens in every condition, with no reference answer sent
to the reader. Exact choice rules and error handling were frozen before execution.

| Condition | Correct actions | Cases |
|---|---|---|
| No earlier memory | 1 | 10 |
| Chronological attributed episodic history | 10 | 10 |
| Current BEING state | 10 | 10 |

These small authored cases show that retrieved state can guide action after reset,
including stability under a weak objection and deliberate revision under stronger
work evidence. They do **not** establish an accuracy advantage over episodic
history, personality, consciousness, calibrated belief, generalization or a memory
benchmark rank. The current-state packets used 232.7 tokens on average versus 883.8 for episodic
history (about 74% less memory). The [full trial report](eval/being-continuity-2026-10-21.md)
records exact usage.

The 30 live calls cost $0.0028654 at conservative regular rates, with no cloud
resources. Remaining inherited allocation is $0.77142728. All earlier 1,048 ledger
rows remain unchanged as verified by their decoded-row hash; 30 new calls were reserved before HTTP.
No retries, model substitutions or paid repairs were made. Rates/settings were
checked against the [primary pricing](https://docs.z.ai/guides/overview/pricing)
and [model documentation](https://docs.z.ai/guides/vlm/glm-5.3-flash).

Verification receipts are under `runs/being-memory-20261021/native`: native
kernel/import/context/relationship/ontology/reflection controls; full workspace
and all-target Clippy; actual isolated Claude SDK dispatch, strict validation,
real two-process Claude loop against a loopback synthetic service, and real native
CLI/SQLite imports. 48 frozen development packets match the phase-entry results;
historical databases remain unchanged. These are distinct proof levels, not a
claim of an independent reviewer. The exact current totals and final file hashes
are recorded in the final verification receipt rather than inferred from old logs.

At the original trial checkpoint, human usefulness feedback was pending. The real
Codex collaboration identity was already in use; Claude integration was qualified
in isolation. No real Claude persona has been selected or silently initialized.
A longer observed collaboration is needed before claiming sustained individuality.
The final acceptance below records the later human response.


The completion audit also ran six preregistered unrelated-task controls against
existing synthetic packets. Both simple tasks remained correct in all three
conditions (2/2 each), closing the initial behavioral noninterference check.
Those calls cost $0.00086395. Total goal external API spend is $0.00372935; the
remaining allocation is $0.77056333. The actual collaboration journal is now at
revision 12, with the control outcome retained as an unadopted ToolResult; a fresh
query includes it and both standing records in 1,974 joined tokens. The separate
current verifier checks all 36 paid-call receipts and leaves the original trial
scores intact. At revision12, human usefulness was the remaining unverified acceptance.


The pre-feedback acceptance audit found no additional required autonomous
engineering work. After three consecutive goal turns without human feedback, the
goal was marked blocked, preserving all five phases and their qualified evidence.
That historical audit remains in `native/blocked-audit.json`. The subsequent human
response resolves its sole blocker.

## Final acceptance and actual commitment closure

The operator’s actual response on 21 October was “Yes, useful starting point.”
The question concerned the recovered five-phase commitment and the adopted practice
to verify execution boundaries, retain counterexamples and explain revisions.
It is recorded as a relationship-scoped HumanReport, with the question and source
reference retained. It does not author a new psychological profile or worldview.

A fresh acceptance audit revalidated the 93-entry pre-feedback manifest and all
five phase evidence mappings. The agent deliberately fulfilled the promise using
the human response and an existing verification ToolResult as outcome sources.
Revision13 holds the response; revision14 records fulfillment. Exact replay
adds no event. The original12 events are unchanged, and the verification method
remains adopted. Fresh native processes recover the terminal promise and feedback
in 1,849 tokens; a practice-focused packet contains both standing claims and the
feedback in 1,744 tokens. Each packet obeys the same single 2,000-token cap.

`native/closure/verification.json` records the separate same-operator closure
check; `native/verification-current.json` joins it with the preserved technical
qualification. `native/closure/pre-feedback-verification.json` retains the exact
revision12 receipt. The original verifier targets that historical snapshot; use
`python3 runs/being-memory-20261021/native/closure/verify.py` to verify the final
actual journal without rewriting the frozen trial. Source code is unchanged
during closure, all1,084 spending rows match, and no further API/cloud spend was
incurred. Remaining allocation: $0.77056333. Longer-term individuality, harder
independent trials and actual model-generated Claude compaction remain future
observations, outside this completed initial build.
