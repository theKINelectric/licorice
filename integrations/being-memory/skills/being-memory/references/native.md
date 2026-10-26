# Native BEING protocol

The executable is `licorice-being`, built from the native `licorice` crate:
`cargo build --locked -p licorice --bin licorice-being`. Set `LICORICE_BEING_BIN`
to the resulting executable's absolute path. It does not read provider keys.
Choose an absolute native store path and a stable agent identity; the embedding
dimension is fixed by the space (default256; `--embed-dim` selects another).

```sh
"$LICORICE_BEING_BIN" init --store "$LICORICE_BEING_STORE" --agent "$LICORICE_BEING_AGENT"
"$LICORICE_BEING_BIN" context --store "$LICORICE_BEING_STORE" --agent "$LICORICE_BEING_AGENT"
```

Success is `{"ok":true,"result":...}`; failure is a nonzero exit with
`{"ok":false,"error":...}`. Context's result has `text`, `usage`, `continuity`,
and `evidence_mode`. Inject only `text`; inspection metadata is not extra context.
The whole text is counted as ordinary cl100k tokens under one cap<=2000.

`context` optionally reads a JSON object from stdin:

```json
{"query":"What should guide our current work?","at":1792595000000,"budget_tokens":2000,"continuity_tokens":400}
```

Omit stdin for a current-work query. `--relationship ID` includes that scoped
history alongside global state. With `--models /absolute/path/to/models/wordllama`,
context joins continuity with local source-aware factual retrieval inside the
same cap. Without models, it explicitly reports `continuity_only` and can use
the full allowance for continuity. There is no second hidden2k evidence block.
Question/system/tool framing and output tokens belong to the host's overall
context allowance and must be reserved separately.

`snapshot` returns the selected relationship's state and current revision;
`history` returns attributed event/decision history for local inspection.

## Attribution and mutation

`apply` reads one typed request from stdin. Transport flags supply the author:
`--principal ID --role agent|human|harness|importer --session ID`. An agent
principal must equal the bound agent. Only the agent can propose/adopt its stance.
These flags are trusted local-caller metadata, not remote authentication. A
caller with arbitrary access to the database retains that OS-level authority.

```json
{"request_id":"working-method-01","expected_revision":0,"operation":{"kind":"propose","payload":{"claim_id":"working-method","category":"method","subject":"chosen-agent-id","statement":"Retain a correction and the reason for changing a judgment.","source_ids":[],"credence":null,"importance":0.8,"reason":"A deliberate practice for our shared work.","relationship_id":null}}}
```

Proposals start `considering`. A separate decision uses the next current revision:

```json
{"request_id":"adoption-01","expected_revision":1,"operation":{"kind":"decide","payload":{"claim_id":"working-method","stance":"adopted","credence":null,"reason":"Adopt this as a revisable practice.","source_ids":[]}}}
```

Stances are `considering`, `adopted`, `suspended`, `declined`, `retracted`.
Categories currently include `belief`, `value`, `method`, `commitment`,
`self_description`, `relationship`, `autobiographical`. Dedicated commitment lifecycle and versioned ontology tools are described below.
Credence/importance must be finite [0,1]; null credence remains unknown.
Proposition text/subject/scope is immutable; changed meaning gets a new claim ID.
Replay requires the exact actor/session/request. Stale revisions require review.

An observation's operation is `observe`; its payload has `origin`, `subject`,
`text`, optional `relationship_id`/`occurred_at`, and typed `source`:

- `{"kind":"reported","reference":"source-description"}` retains a report.
- `{"kind":"native_turn","session":"external-session-id","turn":"external-turn-id"}`
  must match the canonical stored text, role and timestamp exactly.
- `{"kind":"host_receipt","receipt_id":"tool-id","body_blake3":"..."}`
  binds a harness-attested tool observation to the exact text digest.

Observation origins are human report, tool result, imported document, model
default report, agent reflection and direct interaction (snake_case on wire).
They have role-specific admission. Reading/import never silently becomes personal
participation. Autobiography requires this agent's canonical interaction in a
harness-bound assistant session, or a host tool receipt. Reports alone do not
certify that participation. Host attestation does not prove semantic truth.

## Legacy import

Initialize an explicit empty native journal for the same agent, then run
`import-sidecar --source ABS --import-id ID` with importer transport flags.
This validates original SHA256 receipts and sequence, retains original bytes,
timestamps and source mappings, and commits all events/projections atomically.
Imported source assurance stays `legacy_import`; it is never promoted to a live
canonical observation. Exact import replay is safe. Declining/suspending a legacy
autobiography remains possible even when its participation source is uncertified.

## Relationships and commitment lifecycle

Use agent-authored, revision-checked `define_relationship` with
`relationship_id`, next `version` (starts1), distinct `members` including the
agent, `description`, `boundaries`, `source_ids` and `reason`. The description is
this agent's perspective. Members are immutable; a changed boundary advances the
version and preserves earlier events. A scope tag alone does not register members
or assert reciprocal feelings, agreement or consent.

`register_commitment` takes `claim_id`, `counterparty`, `next_action`, optional
`due_at` and optional `acknowledgement_source`. The referenced claim must be this
agent's adopted `commitment`, scoped to a defined relationship containing that
counterparty. No acknowledgement is inferred when its source is absent.

`transition_commitment` takes `claim_id`, `status`, `reason`, `source_ids`.
Statuses are `active`, `fulfilled`, `released`, `broken`. Close an active promise
or explicitly reopen a terminal one; do not rewrite one terminal status into
another. Fulfillment needs a canonical interaction, a host tool receipt or an
attributed counterparty report. Release needs a counterparty report. An agent's
reflection alone cannot certify fulfillment/release. Admission establishes an
outcome-source identity, not that its interpretation is correct.

## Ontology tools

All mutations use `apply`, matching agent transport and current revision:

- `define_concept`: `concept_id`, `version`, `label`, `definition`, optional
  `relationship_id`, `supersedes`, `source_ids`, `reason`. `supersedes` contains
  exact versioned concept references. Redefinition includes the immediately prior
  version; a split makes new IDs pointing at the old meaning, and a merge points
  at each prior meaning. Original definitions remain readable.
- `define_hypotheses`: `set_id`, `version`, `label`, `mode`, `alternatives`,
  `unknown_weight`, optional `relationship_id`, `source_ids`, `reason`.
  Alternatives are `{claim_id,weight}` over attributed belief claims. Exclusive
  weights plus explicit unknown mass sum to1. Compatible propositions have
  independent nullable weights and no shared unknown allocation. These are
  explicitly uncalibrated estimates and never imply adoption.
- `link_positions`: `link_id`, `from`, `to`, `relation`, optional `relationship_id`,
  `source_ids`, `reason`. Dependent first, dependency second. Relations are
  `depends_on`, `supports`, `objects_to`, `contradicts`, `uses_meaning`. Source and
  position scopes cannot widen or cross relationships.
- `retire_link`: `link_id`, `reason`. A new event retires a dependency after
  deliberate review; history retains the original link. Create a new link when
  the basis changes.

Position references are `{kind:"claim",claim_id:"..."}`,
`{kind:"concept",concept_id:"...",version:1}`, or
`{kind:"hypotheses",set_id:"...",version:1}`.

`impact --relationship ID` reads optional
`{"roots":[...],"limits":{"nodes":32,"edges":128,"depth":4}}`.
Without roots it detects changed referenced positions. Maximum limits are
64nodes/256relevant edges/8levels. Saturation means further candidates may exist.
The result schedules review; it does not propagate stance, claim philosophical
truth or run a provider. Current context omits superseded top-level meanings but
keeps their exact definitions inside source-closed dependent units.

## Selective reflection

`reflect-plan` requires agent transport and reads:

```json
{"job_id":"review-01","trigger":"correction","source_ids":["observation-id"],"relationship_id":null,"question":"Does this observation warrant reconsidering our working practice?"}
```

Triggers are `correction`, `commitment_outcome`, `boundary_change`, `contradiction`,
`user_requested`. Their significance remains the caller's explicit judgment.
User-requested review needs a human report. Reflection-only triggers are skipped;
repeated text cannot become novel evidence by changing its reference. A ready
plan contains a replayable `ModelRequest` and exact memory-token usage. Trigger
sources and continuity together fit2000ordinary-cl100k tokens, output is capped
at512, and at most four jobs reserve per UTC day in one store. Failed/in-flight
jobs retain their reservations. No callback executes these plans automatically.

A host can run the plan through its authorized budgeted model path, or use native
`Memory::execute_reflection` with an injected `ChatModel`. That API claims a
reservation atomically and makes at most one completion attempt, without retries
or reflection recursion. The token reservation does not authorize monetary spend;
the selected provider must enforce its own durable monetary ceiling.

`reflect-complete` reads a `ModelResponse`:

```json
{"id":"review-01","output":{"action":"review","conclusion":"Reconsider the workshop schedule.","alternatives":["The report may describe a temporary exception."],"witnesses":[{"source_id":"observation-id","quote":"an exact literal substring"}],"positions":[]}}
```

`action` is `review` or `no_change`. Review needs literal trigger quotations;
position references must exist in the selected snapshot. Changed snapshot, actor,
invalid quotations, excess output or altered replay is refused. An accepted
completion records an attributed reflection observation, retaining its plan and
response. Adoption/weight changes still require a separate explicit decision.
Use the same actor/session when completing a reserved plan. CLI planning and
completion are transport primitives; external runners must retain at-most-once
execution receipts rather than paying again after an interrupted request.

## Remembering an observation without endorsing it

Context input can explicitly set `"include_episodes":true`. This adds a bounded
recent view of attributed observations without requiring a claim proposal or
adoption. Such units say `unadopted_observation`; their origin, subject, source
and assurance remain visible. Global observations and the selected relationship
are eligible; another relationship is excluded. Up to 128 recent scoped candidate
observations are inspected, with a saturation receipt. Equal-relevance episodes
prefer the more recently recorded event. `recorded_revision` identifies journal
order, not when the reported occurrence happened.

These whole units share the existing 2,000-token allowance with claims and factual
evidence. They are not a second context block. Recall makes no writes. The default
is false to preserve frozen context protocols. Codex may use the explicit local
binding's `include_episodes` value in its context input; Claude can opt in with
`LICORICE_BEING_EPISODES=1` (0 disables). This option does not share identities.

Compaction boundary replay uses a local SHA-256 digest of the semantic compaction
input, excluding opaque engine handles. Only the digest and boundary metadata are
persisted. Repeated callbacks for the same input retain one request identity even
when the clock advances; raw transcript text/instructions are not stored there.
