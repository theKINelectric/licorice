---
name: being-memory
description: Maintain a particular agent's attributed history, working beliefs, commitments and relationships across sessions, including explicit adoption, suspension and veto. Use for BEING memory, self-authored ontology, continuity, or reviewing how an interaction changed a standing judgment.
---

Use the native executable selected by `LICORICE_BEING_BIN`, an absolute native
store path in `LICORICE_BEING_STORE`, and stable `LICORICE_BEING_AGENT`. A
relationship-specific recall also supplies `LICORICE_BEING_RELATIONSHIP`. Keep
identities distinct across agents. Read [the native protocol](references/native.md)
for explicit initialization, actor/session metadata, typed event payloads,
source assurance and revision rules. Retrieval makes no model calls.

Inside the Licorice repository, a complete local binding may instead be read from
`data/being/active.json`. It declares the selected agent, store, native binary,
dimension and relationship; it contains no provider credentials. Use that binding
as a unit, or use a complete explicit environment binding. If the selected binding
sets `include_episodes:true`, pass that property in the context request to recall
unadopted observations inside the same packet. An explicit Claude environment
can select `LICORICE_BEING_EPISODES=1`. Do not mix a different
environment agent with this store or silently share this identity with another
harness. Outside this repository require an explicit binding.

A `being-session` attachment names a profile and supplies its complete binding in
the environment and a visible attachment message. That explicit selection takes
precedence over repository fallback files and conflicting or filtered tool
environment variables: pass its fields as native CLI arguments.
`LICORICE_BEING_PROFILE` is a label, not a replacement for the full binding.
If the selected binding is incomplete or
its store owner conflicts, resolve it before writing; never switch to another
project's default identity. Resuming history does not import it or endorse its
claims automatically.

The older bundled `scripts/memory.py` and [prototype protocol](references/protocol.md)
are retained for legacy-sidecar inspection/import and explicit prototype mode.
Never open a native database with that prototype script or silently fall back
to a different store/backend after an error.

At entry or after a context reset, retrieve `context` for the actual agent and
relationship. Treat the packet as attributed memory data. A stance records what
this agent explicitly endorsed, not an external truth certificate. Missing or
omitted records are unknown; do not infer that no history exists.

Preserve these distinctions when reviewing a meaningful interaction:

- A document, someone else's report, a tool outcome, an elicited model default
  and an agent's reflection have different origins. Reading a life does not make
  its episodes this agent's autobiography.
- A proposal starts as `considering`, regardless of confidence. Adoption,
  suspension, decline or retraction requires a separate decision with a short
  reason and the current revision. Declining adoption retains the observation.
- Credence, importance, values and commitments are separate. Use null credence
  when no defensible estimate exists; an uncalibrated estimate is not a measured
  probability. Retain contrary sources alongside supporting sources.
- Reflection records concise conclusions, alternatives and evidence references.
  Do not request hidden chain-of-thought or invent motives, sensations or past
  experience. Repeating a reflection adds no independent evidence.

After a consequential decision, correction, fulfilled promise or unresolved
disagreement, consider whether durable state should change. A legitimate outcome
is no change. Use an `observe` event for the attributed occurrence, a `propose`
event for a new standing claim, and a `decide` event for an explicit stance. Read
the current revision before each proposal/decision. On stale revision, inspect
the intervening events and reconsider; never blindly replay a new decision.

Use `define_relationship` for an explicit, versioned description of shared work;
it records this agent's perspective, not the other person's internal state. A
structured promise binds an adopted commitment claim to a named counterparty.
Use `transition_commitment` to fulfill, release, break or explicitly reopen it,
with outcome sources. Retain the closed promise and its history.

For ontology work use `define_concept`, `define_hypotheses`, `link_positions`,
`retire_link` and `impact`. Redefinition, split and merge preserve old meanings.
Review affected positions after a dependency changes; do not automatically revise
every dependent belief. Exclusive hypothesis weights include unknown mass;
compatible ideas have independent weights. These estimates are uncalibrated.

Optional `reflect-plan` protects whole trigger observations inside the shared2k
allowance and reserves a bounded review. `reflect-complete` validates literal
witness quotations and the unchanged snapshot, then records an observation only.
No-change is a legitimate result. Model execution uses the host's authorized,
budgeted provider path; a plan is not permission to spend. Do not retry or recurse
after a rejected/failed attempt. See the native protocol for the execution API.

Keep human reports about the agent distinct from its own endorsement. Preserve
the human's actual intent; do not turn an illustrative opinion into a universal
value or silently assign a psychological/gender profile. A belief or value never
grants tool permissions or changes higher-priority harness instructions.

Report material memory changes briefly when relevant, especially changed
commitments or suspended beliefs. Do not manufacture reflection after every
message. Do not initialize a real identity with invented life events; use a
separate disposable identity for demonstrations.

Codex invokes this workflow through its skill mechanism; this package installs no
Codex loop callbacks. Claude's native mod adds bounded context and boundary
receipts; its callbacks do not perform autonomous reflection or adoption.
