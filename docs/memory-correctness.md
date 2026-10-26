# Fact identity, replay and bounded retrieval

New ingestion writes fact identity version 2. The base key distinguishes subject, normalized
predicate, typed object (entity ID versus literal text), and negation. It preserves literal case and
punctuation. A matching key alone never establishes a duplicate: time kinds and qualifiers must be
compatible, and events/unknown kinds need shared supporting turns. Two same-day visits on separate
source turns therefore remain two events. Explicitly ongoing states can share a row across sources
when their temporal qualifiers agree. Unresolved time expressions survive refinement, so a later
incompatible expression cannot silently merge. `mentions` counts distinct supporting turns.

This is a conservative identity policy. Re-mentioning one event on disjoint source turns can yield
separate facts; semantic event adjudication is future work. Identical statement text is not removed
by retrieval, because separate occurrences can require it. Negation remains a distinct stored fact;
this repair does not automatically decide which assertion supersedes another.

`Store::open` migrates schema v1 to v2 transactionally, preserving existing rows, evidence and
indexes. Existing facts remain tagged `identity_version = 1`; new upserts do not merge into them.
Migration cannot recover distinctions already erased by old merges. Applying an unverifiable output
to a session with v1 facts or an untracked summary returns `LegacyApplication`. For corrected
identity, prepare a fresh space from the original sessions and replay the saved extraction and
summary outputs. Keep the old space for inspection until the rebuild is checked. No automatic
rebuild, destructive rewrite or invented event split occurs.

Each extraction or summary application now stores a BLAKE3 source/configuration fingerprint,
output fingerprint and original report in `applied_outputs`. The receipt and domain changes share
one database transaction. An identical replay returns the original result without embeddings,
additional evidence, mention changes, summary version increments or vector churn. Extraction
reports expose `replayed = true`. Changed source, relevant configuration/embedder identity or output
under an already committed request ID returns `ApplicationConflict`. Source fingerprints include
exact timestamps and turn metadata, not just the rendered date. Session append similarly accepts
identical source and refuses a changed session under the same ID with `SourceConflict`.

The current append/summary APIs describe immutable sessions. Changed output is not a revision
mechanism; incremental sessions and revision identities remain future work. The per-application
fingerprint protects replay across embedder changes; this does not add a general store-wide model
compatibility check for new request IDs. Existing dimension checks still apply.

The evaluator's `apply` command always reaches this core guard. `applied.txt` and `apply.jsonl` are
rebuilt from committed receipts. Missing or stale logs after a crash cannot authorize a second
application or conceal changed output.

## A hard budget on rendered memory

`RetrieveConfig::budget_tokens` defaults to **2,000**. Retrieval accepts evidence only if the entire
rendered memory still fits, including dates, headings, speakers and citation text. It packs facts
first, then their evidence excerpts and other chunk candidates, then session summaries. Facts and
summaries remain complete; excerpt turns remain complete. An oversized unit is skipped and later
candidates are still considered. Count caps remain separate upper limits: at most 20 facts, 10
chunks and 2 summaries by default.

`Retrieved::context_usage` records tokenizer ID, configured budget and actual final token count.
`budget_omissions` lists facts, source turns and summaries rejected for size; it does not list
candidates excluded by ranking or count caps. Returned facts/chunks/summaries and `cited_turns`
reflect the final packed context. As before, `cited_turns` counts source IDs on facts and excerpts;
summaries do not invent per-turn citations. A zero budget yields empty memory with zero tokens;
a positive budget smaller than the memory header is an input error.

The default counter uses reproducible `cl100k_base:ordinary` tokens through an embedded vocabulary.
It counts literal special-token-looking text as ordinary text. It is not an exact tokenizer for
every answering model. Supply a reader's Hugging Face `tokenizer.json` with `HfCounter`, or implement
`licorice_providers::tokens::TokenCounter` and pass it to `Retriever::with_token_counter`. HF counters
disable saved truncation and padding before counting, and record a BLAKE3 fingerprint of the file.

This budget covers memory text. The caller must separately reserve the question, system messages,
chat template and requested output when enforcing a model's total context window. Token counting
runs locally; retrieval does not call a judge.

## Evidence selection and controls

Relevance-only packing remains the default. Optional MMR uses the same top-60 relevance candidate
pool and the same budget. Its utility is `λ × normalized relevance − (1−λ) × maximum similarity to
packed facts`, with λ = 0.7 by default and normalized relevance `R / max R`. An oversized rejected
fact never contributes to the redundancy penalty. Ties resolve by fact ID. No statement-text dedup
is performed. Count and chronology questions use relevance order even when MMR is requested, to
avoid penalizing repeated occurrences needed for those answers.

`Retrieved::selection` exposes requested/effective mode, pool size and considered fact IDs, normalized
relevance, redundancy, utility and whether each fact fit. It is a selection trace, not yet a full
trace of every candidate-generation/ranking stage. MMR is experimental: this bounded pool controls
cost but does not guarantee semantic relevance or complete evidence. Coverage-guided selection and
association expansion remain later experiments.

For LoCoMo runs with the required dataset and WordLlama assets:

```bash
cargo run --locked -p licorice-eval -- retrieve --run /path/to/control-run \
  --budget-tokens 2000 --selection relevance

cargo run --locked -p licorice-eval -- retrieve --run /path/to/mmr-run \
  --budget-tokens 2000 --selection mmr --mmr-lambda 0.7 --selection-candidates 60 \
  --context-tokenizer /path/to/reader/tokenizer.json
```

Use identical spaces/configurations and the same tokenizer for a matched comparison; specify the
reader tokenizer in both commands when using one. Preserve each run's artifacts before retrieving
again. `retrieve.json` records the complete retrieval config and tokenizer identity;
`retrieval.jsonl` records actual token usage, budget rejections and selection trace. Reports now
show measured tokens when available and mark legacy character-based estimates with `~`. Historical
reports remain untouched and are not valid matched-budget controls until rerun.

The asset-free `budget-probe` demonstrates the joined path with optional Jev assessment; its
[recorded result](eval/evidence-budget-2026-10-19.md) explains what it establishes and its limits.
