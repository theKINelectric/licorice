# Spike: Turso Database (Rust rewrite) as the Licorice engine

`cargo run --release` with turso 0.8.3. Measures exact and 8-bit vector scans (20k / 100k × 384-d),
tantivy-backed FTS (bulk build, 50-row steady-state commits, query), single-writer WAL vs `BEGIN CONCURRENT`
(MVCC) write throughput, and FTS under MVCC. FTS needs `Builder::experimental_index_method(true)`.
Results: `../../docs/adr/0001-storage-engine.md`.
