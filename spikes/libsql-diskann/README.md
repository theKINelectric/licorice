# Spike: libSQL native vector index (DiskANN)

`cargo run --release` with libsql 0.9.30 (`core` feature only). Builds a `libsql_vector_idx` index over 20k × 384-d
vectors with default and compressed (`compress_neighbors=float8, max_neighbors=32`) settings; measures insert time,
database size, `vector_top_k` latency and recall@20 against an exact scan, and 50-row incremental commits.
Results: `../../docs/adr/0001-storage-engine.md`.
