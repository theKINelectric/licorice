# Temporal memory and the controlled retrieval challenge

The temporal ingestion profile implements source-backed state replacements and
corrections, exact message date anchors, replayable model phases and bounded
retrieval experiments. It is inspired by [Zep's temporal memory architecture](https://arxiv.org/abs/2501.13956)
and builds on Licorice's existing [LiCoMemory-inspired hierarchy](https://github.com/EverM0re/LiCoMemory).

The native default keeps temporal updates disabled. The HTTP service defaults to
the offline raw profile. New behavior is explicit in `IngestConfig.temporal_updates`
or `LICORICE_AML_PROFILE=temporal`.

## Time and source semantics

`append_timed_session` accepts one immutable Unix-millisecond timestamp per turn.
Replaying the same session with a changed turn timestamp conflicts. The temporal
extractor renders per-message dates and resolves copied expressions against their
cited supporting turn. It does not ask the model to calculate calendar dates.

An update names a previous object, gives an exact quote from the current supporting
source, distinguishes `replace` from `correction`, and may copy an effective time
expression. An unsupported quote cannot mutate fact state. The earlier date of a
move and the later date of its report remain distinct. Continuing residence uses
a stable predicate across tenses. Date intervals retain their precision.
An explicit update effective expression must appear in a supporting current
source turn and takes precedence over a generic state expression such as "now".

A replacement ends the earlier state's validity while retaining its fact and
source evidence. Reconciliation follows validity order, so importing a historical
mention later does not restore an expired state. A return to an earlier value
creates a new state. Explicit negation/restoration preserves polarity. Multiple
values without an explicit change remain independent assertions.

A correction retracts an erroneous assertion. It does not claim that the assertion
was formerly true. Retracted facts remain stored with their provenance and
transaction metadata; ordinary fact retrieval excludes them. Raw source is retained.
A replacement requires a new ongoing state. A correction can instead describe a
lifetime denial with an unknown time span; it need not invent an ongoing period.
Explicit updates with non-current evidence or anchors trigger repair rather than
silently disappearing from an otherwise successful extraction application.

Schemas v4/v5 add transaction creation/invalidation timestamps and `fact_updates`,
including source quotes, effective-date basis, assertion dates and application
identity. Transaction timestamps for pre-migration facts remain unknown. This is
not a complete transaction-time query API or a reconstruction of all historical
refinements. Rebuild a fresh profile from raw source when changing ingestion policy.

Schema v6 records the explicit old value independently of entity-versus-literal
storage. A later extraction can represent the new object differently without
leaving the previous state active. This also works when old history is imported
after the transition. Literal values match exact text; entity aliases must resolve
uniquely. Legacy transitions retain their structural keys with no guessed backfill.
The HTTP ingestion profile is now `temporal-facts-v12-output-capacity`; existing
profile roots require a fresh rebuild rather than silent policy mixing.

If a correction retracts a claimed move, that move stops invalidating its prior
state. Reconciliation revisits affected predecessors, including representation-
independent targets. A previously ended ongoing assertion is restored only when
its successor is retracted and no supported replacement remains. A separate
explicit transition supplies a corrected end date. Retracted assertions are not
resurrected. The transition ledger retains the original source and observed times;
fact status/timestamps describe the reconciled snapshot, not a full transaction-
history query API.

If an effective date is coarse, the stored fact keeps that coarse interval. A
precise invalidation date is not fabricated from its first day: the source date
serves as the conservative known-by bound and its basis is recorded.

## Atomic model-backed Add

The temporal HTTP profile uses GPT-4o-mini through OpenRouter, restricted to its
OpenAI provider. Exact rendered inputs, responses and usage are retained in a
separate spend ledger. The client reserves before transmission, settles known
usage and retains full reservations for uncertain calls. Repeated exact inputs
reuse cached responses. Redirects are disabled and public errors omit credentials
and payloads.
Cache claims coordinate participating clients and processes sharing the same
canonical ledger, so an identical concurrent request sends once. Process death
releases the claim; recovery retains an uncertain charge reservation and records
the interrupted owner before retrying. Claim files are bounded to 64 SQLite
stripes beside the ledger. Keep them with the live ledger.

Add works on a staged copy of the user's database. Raw data and replay receipts
commit there first. One extraction runs per message, using the previous four
messages from the same external session, even across Add requests. Stable logical
message IDs make the extraction input independent of those request boundaries.
Requests are prefetched in groups of eight and applied in source order; provider
execution is bounded at sixteen concurrent calls. Summaries are generated per Add
batch, so summary grouping remains sensitive to batching.

Each phase output is persisted before native application. Provider/storage failure
returns an error, keeps the preceding published database searchable and leaves
completed work available for retry. Successful completion closes/checkpoints the
staged store and atomically publishes it. Process death therefore exposes either
the preceding version or the completed Add.

The model-backed adapter binds `evidence_turns`, `time.anchor_turn` and update
`evidence_source` to the current turn ID through schema enums. Native code then
copies that complete immutable source message into `evidence_quote`. The model
selects a permitted reference instead of generating quote text. Raw model outputs
and native application receipts remain distinct. Schema-invalid model output
fails before application and leaves the raw batch staged for an identical retry.
The native extraction API continues to accept literal source quotes directly.

An unusable derived fact is omitted with `facts_dropped` when its anchor is outside
its supporting current turns. Schema-valid source-validation failures in updates get two bounded
repair attempts; unresolved derivations are recorded in `aml_rejections` and omitted.
All raw messages survive. These omissions are quality diagnostics, not proof that
all retained model claims are entailed by their sources.

User and assistant self references use canonical `User` and `Assistant` actors.
Named third parties remain distinct. Ambiguous person proxies and generic speaker
aliases trigger bounded repair; a typed physical speaker remains an object.
The adapter declares canonical actors as kind `person` when the model omits their
entity declarations and rejects an explicitly different canonical kind. This
keeps the same actor ID across omissions and later declarations; matching display
names alone is insufficient to prove identity.

## Search experiments

The service offers `facts`, `connected` and `reranked` Search policies for the
temporal profile. All remain opt-in; the native default is unchanged.

Connected retrieval expands at most two hops, sixteen incident facts per visited
entity, thirty-two entities and sixty graph candidates. It packs a complete
connecting path and its source chunks atomically. A path that cannot fit is
rejected whole; later smaller evidence can still fit. Previously packed sources
are reused without duplication.

Each optional scorer call sees only the query and up to twenty-four candidate facts,
with temporal qualifiers at their original precision. It cannot generate an answer
or see gold labels. Its output must score exactly the supplied IDs once, using
finite values in [0,1]. Two bounded repairs are allowed. Configured failure falls
back to the native order with an explicit trace; strict mode returns an error.
Its keyed v2 schema requires exactly one numeric score property per candidate ID,
preventing missing, invented and duplicate IDs at the provider boundary.
An opt-in batched wrapper can cover up to 96 candidates with at most four calls,
validates the complete union, and drains all dispatched work before reporting a
failure. The default pool remains 24. Native eval selects a deeper pool through
`--rerank-candidates 60`; the HTTP option is documented in the adapter guide.

Native `RetrieveConfig.query_speaker` optionally declares the actor for first-person
queries. The temporal HTTP profile declares `User`; named third-party queries do
not acquire that anchor. Missing or ambiguous aliases are never guessed or created.
External benchmark views can supply `question_at_ms` independently of source dates.

Two tested development policies were rejected: packing direct assertions before
source excerpts, and query-prioritizing the bounded graph fanout. In a 199-question
development factorial, the original connected policy answers 91/152 answerable
questions correctly; each alternative answers 84/152. Better raw evidence recall
does not establish better answers. Their frozen sources/binaries/results remain
in `runs/improvement-20261020/`; the unsuccessful options are removed from the
current runtime. The native/API retrieval defaults remain unchanged.

The API counts the entire serialized evidence array under its 2k cap. The local
controlled benchmark instead applies one common 2k native rendered-memory cap to
all four arms: raw vector/BM25, facts/summaries, connected paths/sources, and that
connected policy with scoring. These two counting surfaces are deliberately
reported separately.

## Evaluation and recovery

The challenge runner (not part of this repository) records qualifications, freezes source/binaries,
loads source histories through Add, produces answer requests, prepares shared
source packets, verifies artifacts and reports results.
Its run directory is `runs/temporal-challenge-20261019/`.

The release comparison registers all 1,489 questions from conv-42, conv-43,
conv-44, conv-47, conv-48, conv-49 and conv-50 before inspecting their outputs.
GPT-4o-mini is the common reader. GLM-5.3-Flash uses the unchanged, sixteen-control-
qualified source-v2 rubric. All arms share each question's source packet, including
referenced turns/neighbors and the union of their retrieved sources, under a
separate 8k grading allowance. Ambiguous verdicts and source/gold conflicts remain
explicit. Gold enters grading only.

The user raised the new-phase authorization to USD50 during execution. The
already frozen comparison keeps its stricter USD25 admission ceiling; the higher
authorization is available if required. The original ledger remains historical.
Source hashes, exact requests, outputs, rejected
attempts, token recounts, identical-domain checks and costs accompany the report.
The seven conversations are no longer untouched release data after this run.

The completed [four-policy report](eval/temporal-challenge-2026-10-19.md) recommends
facts plus summaries: 684/1,155 answerable questions correct vs 574 for raw at the
same 2k cap. Connected/scored retrieval reaches 689 but adds latency and cost with
an uncertain gain over facts alone. All 5,956 generated answers, audited outcomes
and independently recounted contexts pass final verification. New-stage accounted
usage is USD7.15628220, with no uncertain calls remaining.

This is a local original-LoCoMo comparison. It does not establish an AML rank or
comparability with published scores using other readers, budgets or benchmark
releases. Public deployment, retention operations and a formal board run remain
separate work. No default is automatically promoted from the comparison.

## Current board model requirement

The [official AML guide](https://agentmemoryleaderboard.ai/api-guide), rechecked
20 October 2026, specifies `text-embedding-v4` embeddings and `gpt-4o-mini` LLM
components for Open-source Methods; rerankers are unrestricted. WordLlama and
OpenAI embedding experiments are diagnostic and do not satisfy that embedding
requirement. Commercial Products has different model rules. The adapter's local
contract tests establish compatibility, not eligibility or a formal score.

## Bounded output-capacity repair

The v12 profile recognizes completed extraction or summary responses that hit
their output-token limit. It tries a separately identified request with an8192
output-token cap. Extraction keeps its existing three-attempt total repair
allowance; summary permits one capacity retry. The durable ledger retains every
actual request/response/charge and recognizes old truncated-response receipts,
so an identical capped request is not paid for repeatedly.

If the larger response also cannot fit, the optional derivation phase records
an `aml_rejections` entry and completes with empty derived facts or summary.
All raw turns remain stored and indexed; previously supported facts survive a
summary failure. Whole source messages must still fit the Search evidence budget
to be returned. Transport, storage and other backend failures retain failed
staged Add behavior. Existing roots require a fresh v12 rebuild; the frozen v10
benchmark failure and its recovery are reported separately.

## Experimental query-window pruning

`RetrieveConfig.filter_state_window`, `licorice-eval retrieve --state-window`,
and temporal API `LICORICE_AML_STATE_WINDOW=true` enable a filter that is off by
default. For an explicit query window, it can exclude ongoing states that ended
before the window or have an exact-day replacement start after it. Initial
assertion dates and coarse/unknown starts do not become invented exact starts.
Point-event facts are not subject to this state filter. Ordering and comparison
questions retain evidence across their date anchors. The native candidate text
already includes date annotations; this option adds deterministic eligibility
bounds. Synthetic controls pass; benchmark quality is not yet established, and
the frozen four-arm recovery experiment does not enable it.
