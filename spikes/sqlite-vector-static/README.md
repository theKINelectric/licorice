# Spike: statically linking sqlite-vector into a Rust binary

Proves the integration path assumed in `../../SPEC.md` §7.

```bash
git clone --depth 1 --branch 1.1.2 https://github.com/sqliteai/sqlite-vector.git /path/to/sqlite-vector  # or main @ 0c2223a
SQLITE_VECTOR_DIR=/path/to/sqlite-vector cargo run --release            # behaviour checks
SQLITE_VECTOR_DIR=/path/to/sqlite-vector cargo run --release --bin bench # latency/recall at 20k and 100k rows
```

- `main.rs`: registers `sqlite3_vector_init` via `sqlite3_auto_extension`, checks exact search, INT8 quantized search on a
  second connection, visibility of rows inserted after `vector_quantize`, and streaming mode with a relational `WHERE`.
- `bench.rs`: median of 21 queries, k=20, 384-d random unit vectors, file-backed WAL database.

Note `libsqlite3-sys` is a *direct* dependency: Cargo only exposes `DEP_SQLITE3_INCLUDE` to crates that depend on the
`links = "sqlite3"` crate directly, and the C extension must compile against the same SQLite header it links with.
- `writes.rs`: C SQLite counterpart to the Turso spike: FTS5 bulk and 50-row steady-state commits, BM25 query, and
  single-writer throughput at `synchronous=NORMAL` and `FULL` (`cargo run --release --bin writes`).
