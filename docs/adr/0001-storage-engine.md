# ADR 0001: Storage engine

**Status:** Proposed, 2026-10-19
**Decision owner:** you
**Evidence:** `spikes/sqlite-vector-static/`, `spikes/turso-engine/`, `spikes/libsql-diskann/`, all run on the same machine.

## Question

Is C SQLite with sqlite-vector and FTS5 the right engine for Licorice, or is something better available, Turso in particular?

## What the engine has to do (from SPEC.md)

- **Relational core:** per-space entities, facts, evidence and summaries with indexed joins. Expect tens of thousands of rows per space, up to around 1M.
- **Vector kNN** over five kinds of item, often combined with relational filters (only active facts, a set of sessions, a time window).
- **BM25 lexical search** that sees new rows as soon as they commit.
- **Write pattern:** bursty and small, about one commit of tens of rows per extraction window. Each window waits seconds on an LLM call, so database commit latency (milliseconds) is never the bottleneck.
- **Deployment:** embedded in the agent's process; portable to laptop, server and edge. A shared server is possible later.

## Options considered

1. **SQLite + sqlite-vector + FTS5.** C SQLite through rusqlite, with sqlite-vector 1.1.2 statically linked. This is the current spec.
2. **Turso Database 0.8.3.** A SQLite-compatible engine rewritten in Rust (formerly Limbo).
3. **libSQL 0.9.30.** Turso's C fork of SQLite, with a built-in DiskANN vector index.
4. **Postgres + pgvector** (plus `tsvector` or ParadeDB for BM25).
5. **LanceDB**, an embedded Rust columnar vector store, possibly next to SQLite.
6. **An embedded graph database** (Kùzu and its forks).

## Measurements

**Setup.** 4-vCPU Intel Xeon @ 2.1 GHz (AVX-512), file-backed databases, 384-d random unit vectors, k = 20, medians of 21 queries. Random vectors are the worst case for quantized recall. This is one machine and a warm page cache, so read the numbers as ratios rather than absolutes.

### Vector search

| Engine / mode | 20k vectors | 100k vectors | Notes |
|---|---:|---:|---|
| sqlite-vector, exact `vector_full_scan` | 10.7 ms | 63.0 ms | SIMD (AVX-512) |
| sqlite-vector, INT8 `vector_quantize_scan`, preloaded | **0.91 ms** | **7.3 ms** | recall@20 0.990 / 0.981; re-quantize takes 0.14 s / 1.13 s |
| Turso, exact `ORDER BY vector_distance_cos` (f32) | 23.4 ms | 130 ms | 2.1–2.2× slower than sqlite-vector exact |
| Turso, `vector8` (8-bit) exact scan | 17.0 ms | 87.5 ms | Smaller storage; still a full scan; no vector index exists yet |
| libSQL, exact `ORDER BY vector_distance_cos` | 31.3 ms | n/a | |
| libSQL, DiskANN `vector_top_k` (default settings) | 13.0 ms | did not finish loading in > 7 min | recall@20 0.790; loading 20k took 277 s; database 1.72 GB; **50-row commit 1.6 s** |
| libSQL, DiskANN (`compress_neighbors=float8, max_neighbors=32`) | 5.7 ms | not run | recall@20 0.648; loading took 80 s; database 378 MB; 50-row commit 337 ms |

### Full-text search (20k twelve-word documents)

| | SQLite FTS5 | Turso FTS (tantivy; needs `experimental_index_method`) |
|---|---:|---:|
| Initial build | 0.42 s (index maintained during the load) | 0.28 s with `CREATE INDEX` after the load. **61.9 s** if the index exists during a one-transaction load. |
| **Commit of 50 new rows into the indexed table** | **1.33 ms** | **184 ms**, against 0.78 ms for the same commit with no FTS index |
| BM25 top-20 query | 0.49 ms | 0.15 ms |
| New row searchable right after commit | yes | yes |
| Works under MVCC | n/a | yes |

### Write throughput (1,600 transactions × 10 rows into an indexed table, no FTS)

| Configuration | rows/s |
|---|---:|
| SQLite, one writer, `synchronous=NORMAL` | 129,800 |
| SQLite, one writer, `synchronous=FULL` | 22,450 |
| Turso, one writer (WAL; Turso supports only `synchronous` OFF and FULL, so this is presumably FULL) | 12,200 |
| Turso WAL, 8 contending writers (`BEGIN IMMEDIATE` + retry) | 12,000, with 610k busy retries |
| Turso MVCC, 8 writers, `BEGIN CONCURRENT` | 48,500, with 0 retries |

Turso's source (`core/mvcc/persistent_storage/logical_log.rs`) says `Full` mode fsyncs the MVCC log on every commit. The MVCC row is therefore roughly comparable to SQLite `FULL`, assuming the default mode was `Full`; I did not set it explicitly.

## Findings by option

### Turso Database: the most likely long-term successor, not yet better for this workload

Turso is MIT-licensed, written in pure Rust, and async-native. The team describes it as running in production at several organisations, but it is pre-1.0 (0.8.3; the 0.8 line started 2026-09-28). The libSQL README now steers new projects to it.

**What it would give Licorice:**
- No C toolchain or FFI: no `build.rs` compiling C, and no `unsafe` registration of an extension.
- An async API that fits tokio.
- MVCC `BEGIN CONCURRENT`, about 4× the throughput of its own single writer in this test.
- Change data capture.
- Sync with offline support through Turso Cloud.
- WASM and browser builds.
- Encryption at rest (experimental).

**What it costs today:**
- **Vectors:** exact search only. "Vector indexing" is on the roadmap. Its exact scan is about 2× slower than sqlite-vector's, and 10–19× slower than sqlite-vector's quantized scan.
- **Extensions:** it cannot load SQLite C extensions, so sqlite-vector is not an option there.
- **FTS** is experimental and requires a feature flag. Each incremental commit costs about 180 ms per 50 rows, so real-time ingestion, which writes chunks and facts on every window, would pay roughly 140× SQLite FTS5's cost. Queries are fast.
- **MVCC mode limits:** no virtual tables, no indexes on `WITHOUT ROWID` tables, no materialized views, and `wal_checkpoint(TRUNCATE)` blocks readers and writers. Multi-process access is experimental.
- **Concurrent writes solve a problem Licorice doesn't have** inside one space. Commits take milliseconds and LLM calls take seconds, and a single SQLite writer at `synchronous=NORMAL` was still about 2.7× faster than Turso's 8 concurrent MVCC writers.

### libSQL: the only SQLite-family option with a real ANN index, at a steep price

- Production-grade, MIT-licensed, with FTS5 and embedded replicas.
- Its README says new features now go into Turso; libSQL is maintained, not developed.
- DiskANN gives true approximate search, but on this workload it lost on every axis.
- At default settings, 20k vectors took **1.72 GB on disk** (the raw vectors are about 30 MB) and 277 s to load.
- Its queries (13 ms, recall@20 0.79) were slower than sqlite-vector's *exact* scan.
- **A 50-row commit into the indexed table took 1.6 s.** Index maintenance sits on every insert, which is the hot path for real-time ingestion.
- Compressed neighbours cut the size to 378 MB and commits to 337 ms, but recall fell to 0.65.
- A 100k load did not finish in over 7 minutes.
- Random vectors are a hard case for graph indexes too, so real embeddings would recall better. The size and insert costs don't depend on the data.

### Postgres + pgvector: the right answer for a different deployment

- Mature, with HNSW and IVFFlat indexes, real concurrent writers, and replication.
- It is a server, not an embedded library. Multi-tenancy becomes row filters on shared indexes, and filtered HNSW search is the classic weak spot there.
- Choose it if Licorice becomes a hosted multi-tenant service with many writers per tenant. Not for an in-process memory library.

### LanceDB: a possible vector sidecar, not the primary store

- Embedded, Rust-native, Apache-2.0, with ANN indexes, full-text search and built-in hybrid fusion.
- It is a versioned columnar table format, not a transactional row store. The fact and supersession model (many small updates, joins, foreign keys, atomic multi-table commits) would end up split across two stores, with cross-store consistency to manage.
- Worth revisiting only if a single space outgrows about 1M vectors.

### Embedded graph database: not needed

- Licorice's graph work is one-hop expansion from linked entities, which is an indexed SQL join.
- Kùzu, the obvious embedded choice, was archived in October 2025 (its company was reportedly acquired); forks such as LadybugDB and RyuGraph continue it.
- No graph engine is justified by the retrieval design.

## Decision (proposed)

Keep **C SQLite + sqlite-vector + FTS5** for v1, and make Turso a cheap later switch:

- **Engine-specific code is confined** to one `store::Index` trait (`knn`, `lexical`) plus connection setup. The rest of the store uses SQL both engines accept:
  - no `WITHOUT ROWID` tables;
  - no triggers (FTS rows are written explicitly in the same transaction);
  - no other loadable extensions.

  SPEC.md §5 and §6 have been updated to match.
- **The store test suite is engine-agnostic**, so a Turso backend is a new `Index` implementation plus a CI job, not a rewrite.

**Revisit when** any of these happens:
1. Turso reaches 1.0.
2. Turso ships vector indexing, or its exact scan comes within 1.5× of sqlite-vector on `spikes/turso-engine`.
3. Turso FTS leaves experimental and a 50-row commit into a 20k-row index takes under 10 ms on the same benchmark.
4. The product needs browser/WASM, offline multi-device sync, or many concurrent writers per space. Then Turso's strengths outweigh the gaps, and exact-only vectors are acceptable at per-user scale.

## Consequences

- **We keep:** the fastest measured vector path (quantized, under 10 ms at 100k), mature FTS with millisecond incremental commits, and SQLite's reliability record.
- **We keep paying:** for a C build step and an `unsafe` extension registration, a young C extension (mitigated by vendoring, pinning, running its SIMD tests in CI, and a pure-Rust exact-scan fallback), and single-writer semantics (absorbed by the writer actor).
- **A future move to Turso** touches the `Index` implementations, connection setup and CI. It does not touch the schema, the ingestion pipeline or the retrieval logic.
- **Found in M0:** sqlite-vector's quantization parameters and preloads are per connection. A connection initialized before a quantization silently returns wrong results. Licorice works around this by replacing stale reader connections (SPEC §7.1, §7.3). This is one more reason to keep the vector code behind `Index`.
