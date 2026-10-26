# Local protocol v0.1

The script uses Python 3 standard library and a dedicated SQLite sidecar. The
sidecar is a prototype, not a schema migration or replacement for native Licorice.
It binds one stable agent ID to one store. Local file access is the trust boundary;
origin/subject labels are caller assertions, not authenticated provenance.

Set `LICORICE_BEING_AGENT` to the agreed stable identity and
`LICORICE_BEING_STORE` to an absolute path. Then invoke the script by its actual
installed skill path, always quoting paths/variables:

```sh
python3 /absolute/path/to/being-memory/scripts/memory.py \
  --store "$LICORICE_BEING_STORE" --agent "$LICORICE_BEING_AGENT" init
```

Use the same prefix with `context`, `history`, or `apply`. `apply` reads one JSON
request from stdin; use a literal heredoc or structured file, never shell-expand
its contents. `context --relationship ID` includes global claims plus that
relationship's claims. Omit the relationship to recall global claims only.

Every write has a unique `request_id`. Identical replay returns the original
receipt; changed content under the same ID is rejected. `propose` and `decide`
also require `expected_revision`, the current context/history event sequence.
Each accepted write increments that sequence. Observe/boundary writes may run
between review and adoption, so stale decisions must be reconsidered.

## Observation

```json
{"request_id":"read-01","kind":"observe","payload":{"origin":"imported_document","subject":"author-f","text":"The author reports enjoying experimental investigation.","source_ref":"owned-document:chapter-1"}}
```

Origins: `human_report`, `tool_result`, `imported_document`,
`model_default_report`, `agent_reflection`, `direct_interaction`.
An optional `relationship_id` makes the observation available only as evidence
for claims scoped to that relationship. Observation alone creates no adopted claim.

## Proposal

```json
{"request_id":"proposal-01","kind":"propose","expected_revision":1,"payload":{"claim_id":"practice-01","category":"method","subject":"demo-agent","statement":"Prefer an experiment that can distinguish the competing explanations.","source_ids":["read-01"],"credence":null,"importance":0.8,"reason":"A useful practice to consider, inspired by the attributed reading."}}
```

Categories: `belief`, `value`, `method`, `commitment`, `self_description`,
`relationship`, `autobiographical`. Belief, relationship and autobiographical
proposals require observations. An autobiographical claim requires the same
subject on sources and a participation/tool origin, not imported biography.
Self-description/autobiographical subjects must be the store's bound agent.
An optional `relationship_id` scopes a claim; that scope cannot be widened by a
decision. Values/methods may originate in a reasoned proposal without a source.

## Decision and revision

```json
{"request_id":"decision-01","kind":"decide","expected_revision":2,"payload":{"claim_id":"practice-01","stance":"adopted","credence":null,"reason":"Adopt this as a revisable method for our experimental work.","source_ids":[]}}
```

Stances: `considering`, `adopted`, `suspended`, `declined`, `retracted`.
Every decision carries credence (null or finite [0,1]), reason and zero to eight
new observation IDs. Prior and contrary source IDs are retained. A new decision
may change stance/credence but not silently rewrite the proposition or importance;
make a separately identified proposal for changed meaning. Explicit graph links
between changed propositions belong to the next native phase.

The whole rendered recall packet is at most 2,000 UTF-8 bytes, including its
attribution/scope notice. Ordinary cl100k byte-BPE tokens cannot exceed those
bytes. This is deliberately conservative and is not an exact token count or a
guarantee for arbitrary provider tokenizers. Complete claim/source units are
admitted or omitted. `omitted_claim_ids` is outside the injected text, available
to an inspecting caller. The packet always states its bounded coverage.

This prototype is limited to 4,096 events, 256 claims, eight sources per event,
and 32KiB request bodies. Projection replays the event stream; it is not an
indexed production ontology engine. It has no branch merge, semantic entailment,
automatic belief update, confidence calibration, outcome verification or deletion
API. History contains sensitive content: retain it locally and do not send it
to providers without the applicable authorization. Keep native factual evidence
and BEING context within one future combined 2k allowance, not two allowances.
