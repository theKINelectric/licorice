# Memory Licorice

Long-term memory your agent can explain.

Licorice keeps what an agent saw, who said it, what the agent decided to
believe, and why it changed its mind. Reading an observation does not make it a
belief: adoption and revision are explicit, attributed, and kept as history.
Recall is bounded: a context packet never exceeds its token budget (2,000 by
default), and every item in it cites its source.

Each memory space is one SQLite file, with
[sqlite-vector](https://github.com/sqliteai/sqlite-vector) for vector search and
FTS5 for text. It runs locally; no chat model is needed to store or recall.

**Hosted:** [try the preview](https://memory.stellarmodal.online/) or
[create an account](https://memory.stellarmodal.online/account): passkey sign-in,
isolated projects, Add/Search, source corrections, exports and project-scoped API
keys.

## What's inside

| Crate | What it does |
|---|---|
| `crates/licorice` | Storage (one writer, a reader pool, vector kNN with INT8 quantization above a size threshold, FTS5), ingestion (turns, entities, facts with stable identity, session summaries, source-backed temporal corrections) and retrieval (rank fusion, temporal intent, optional MMR, cited evidence within a hard token budget). |
| `crates/licorice-providers` | Model requests and responses as data, with JSON Schema validation; the WordLlama embedder; token counters; judge clients. |
| `crates/licorice-service` | A self-hostable memory API with accounts and projects, the same service behind the hosted preview. |
| `crates/licorice-game` | Per-character memory for games: witnessed and reported perceptions, simulation time, checksummed saves and timeline forks. |
| `crates/licorice-aml` | An HTTP Add/Search adapter with user isolation and a 2k cap on Search results. |
| `crates/licorice-eval` | The LoCoMo evaluation harness. |

Integrations:

- [`integrations/being-memory`](integrations/being-memory/README.md): BEING memory for Claude Code (a native Mod) and Codex (a skill): attributed observations, deliberate adoption and revision, commitments and relationships across sessions.
- [`integrations/being-inspector`](integrations/being-inspector/README.md): a read-only inspector for a memory store: sources, judgments, commitments, and what the context packet selected or omitted.
- [`integrations/being-portable`](integrations/being-portable/README.md): consistent backup and restore of a memory store.
- [`integrations/local-embeddings`](integrations/local-embeddings): a local embedding service.

## Build

```bash
git clone https://github.com/theKINelectric/licorice && cd licorice
cargo test --workspace
```

A C compiler is required: `build.rs` compiles the vendored sqlite-vector 1.1.2
(`vendor/sqlite-vector`) against the SQLite that rusqlite bundles. On x86_64 the
AVX2 and AVX-512 kernels are built too, and one is picked at runtime. Set
`SQLITE_VECTOR_DIR` to build from a different sqlite-vector checkout.

The WordLlama embedder needs its model files: `scripts/fetch-wordllama.sh`
downloads them. Tests that need them skip with a message when they are absent.

## Use it with Claude Code or Codex

Start Claude Code with a memory identity:

```bash
./scripts/claude-being
```

Attach a named memory profile while resuming an existing Claude Code or Codex
conversation with [`scripts/being-session`](docs/being-attach.md). Day-to-day
setup is in [docs/being-daily-use.md](docs/being-daily-use.md).

## Results so far

On LoCoMo conversation 26, end to end with Claude Haiku answering and WordLlama
embeddings, Licorice scored 80.9% judged accuracy on categories 1–4 (95% CI
74–86, n = 152) and 91.5% on adversarial questions, with retrieval at about 9 ms
p50. conv-26 is the development conversation; one conversation is not a
benchmark result. See [docs/eval/locomo-conv-26.md](docs/eval/locomo-conv-26.md).

Quantized search keeps recall@10 ≥ 0.99 against exact search on 20k × 64-d
synthetic vectors. Quantized recall on real embeddings is not yet measured.

## Documentation

- [BEING memory: design](docs/being-memory.md) and [native implementation](docs/being-native.md)
- [Memory correctness](docs/memory-correctness.md): fact identity, atomic replay, budgeted evidence selection
- [Temporal memory](docs/temporal-memory.md): source-backed updates and corrections
- [Game memory](docs/game-memory.md)
- [Why SQLite](docs/adr/0001-storage-engine.md), with measurements

## License

Licorice's own sources are [Apache-2.0](LICENSE). Third-party components keep
their own licenses; see [NOTICE](NOTICE). The design acknowledges
[LiCoMemory](https://github.com/EverM0re/LiCoMemory) as a research reference.
