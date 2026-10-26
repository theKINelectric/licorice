//! SQLite-engine behaviour: schema, quantization lifecycle, reader recycling,
//! compaction, persistence, isolation between spaces.

mod common;

use common::{Rng, brute_force, clustered, normalized, open, put_facts, put_vectors, space};
use licorice::rusqlite::{Connection, OpenFlags, params};
use std::time::Duration;

use licorice::{Error, Filter, Index, LexicalKind, Store, StoreConfig, VectorKind};

fn quantizing(dim: usize) -> StoreConfig {
    let mut cfg = StoreConfig::new(dim);
    cfg.vector.exact_max_cells = 50_000; // force the quantized path at test sizes
    cfg.vector.requantize_min_delta = 100;
    cfg.readers = 2;
    // Tests quantize with explicit maintain() calls; a 500 ms idle timer firing first
    // on a slow runner would make their reports flaky.
    cfg.idle_maintenance = Duration::from_secs(3600);
    cfg
}

fn numbered(vectors: Vec<Vec<f32>>, first: i64) -> Vec<(i64, Vec<f32>)> {
    vectors.into_iter().enumerate().map(|(i, v)| (first + i as i64, v)).collect()
}

async fn owners(store: &Store, kind: VectorKind, q: &[f32], k: usize) -> Vec<i64> {
    store.knn(kind, q, k, Filter::None).await.unwrap().iter().map(|h| h.owner).collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn schema_is_created_and_dimension_is_fixed() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(16)).await;
    let (version, tables): (i64, i64) = store
        .read(|c| {
            Ok((
                c.pragma_query_value(None, "user_version", |r| r.get(0))?,
                c.query_row(
                    "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name LIKE 'vec_%'",
                    [],
                    |r| r.get(0),
                )?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(version, 11);
    assert_eq!(tables, 6, "five vector tables plus vec_state");
    store.close().await.unwrap();

    match Store::open(&path, StoreConfig::new(32)).await {
        Err(Error::DimensionMismatch { stored: 16, requested: 32 }) => {}
        Err(other) => panic!("unexpected error: {other}"),
        Ok(_) => panic!("opened with a different dimension"),
    }
    open(&path, StoreConfig::new(16)).await.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn ingest_lookups_use_indexes_not_table_scans() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let plans: Vec<String> = store
        .read(|c| {
            let queries = [
                "SELECT id FROM chunks WHERE session_id = (SELECT session_id FROM turns WHERE id = 1) \
                 AND first_turn <= 1 AND last_turn >= 1 ORDER BY id LIMIT 1",
                "SELECT EXISTS(SELECT 1 FROM facts f JOIN fact_evidence fe ON fe.fact_id = f.id \
                 JOIN turns t ON t.id = fe.turn_id JOIN sessions s ON s.id = t.session_id \
                 WHERE s.ext_id = 'D1' AND f.identity_version = 1)",
                "SELECT id FROM summary_keys WHERE session_id = 1 AND live = 1",
            ];
            let mut plans = Vec::new();
            for sql in queries {
                let mut stmt = c.prepare(&format!("EXPLAIN QUERY PLAN {sql}"))?;
                let detail: Vec<String> = stmt.query_map([], |r| r.get(3))?.collect::<Result<_, _>>()?;
                plans.push(detail.join("; "));
            }
            Ok(plans)
        })
        .await
        .unwrap();
    for plan in &plans {
        for table in ["chunks", "fe", "fact_evidence", "summary_keys", "facts", "turns"] {
            assert!(!plan.contains(&format!("SCAN {table}")), "full scan of {table}: {plan}");
        }
    }
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_vectors_are_rejected() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let wrong_dim = store.write(|w| w.insert_vector(VectorKind::Fact, 1, &[1.0, 2.0])).await;
    assert!(matches!(wrong_dim, Err(Error::Dimension { expected: 4, got: 2 })));
    let zero = store.write(|w| w.insert_vector(VectorKind::Fact, 1, &[0.0; 4])).await;
    assert!(matches!(zero, Err(Error::InvalidVector)));
    let nan = store.knn(VectorKind::Fact, &[f32::NAN, 0.0, 0.0, 0.0], 3, Filter::None).await;
    assert!(matches!(nan, Err(Error::InvalidVector)));
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_write_rolls_back() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let result: licorice::Result<()> = store
        .write(|w| {
            w.insert_vector(VectorKind::Fact, 1, &[1.0, 0.0, 0.0, 0.0])?;
            Err(Error::Task("caller gave up".into()))
        })
        .await;
    assert!(result.is_err());
    let (rows, live): (i64, i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM vec_fact", [], |r| r.get(0))?,
                c.query_row("SELECT live_rows FROM vec_state WHERE tbl = 'vec_fact'", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((rows, live), (0, 0));
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn quantized_search_keeps_recall_against_exact() {
    let (_dir, path) = space();
    let dim = 64;
    let store = open(&path, quantizing(dim)).await;
    let items = numbered(clustered(11, 20_000, dim, 200, 0.35), 1);
    put_vectors(&store, VectorKind::Fact, items.clone()).await;
    // Touch a reader before quantizing, so recall also covers replacing a stale connection.
    assert_eq!(store.knn(VectorKind::Fact, &items[0].1, 1, Filter::None).await.unwrap()[0].owner, 1);

    let report = store.maintain().await.unwrap();
    assert_eq!(report.requantized, vec![(VectorKind::Fact, 20_000)]);
    assert_eq!(store.quantization_generation(VectorKind::Fact).await.unwrap(), 1);

    let mut rng = Rng::new(5);
    let (mut found, mut total) = (0usize, 0usize);
    for _ in 0..50 {
        let noise = 0.3 / (dim as f32).sqrt();
        let q = items[(rng.next_u64() % items.len() as u64) as usize]
            .1
            .iter()
            .map(|x| x + noise * rng.gauss())
            .collect::<Vec<_>>();
        let exact = brute_force(&items, &q, 10);
        let got = owners(&store, VectorKind::Fact, &q, 10).await;
        found += exact.iter().filter(|o| got.contains(o)).count();
        total += exact.len();
    }
    let recall = found as f64 / total as f64;
    assert!(recall >= 0.99, "recall@10 = {recall}");
    assert!(store.reader_replacements() >= 1, "the reader opened before quantizing was replaced");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn rows_added_after_quantization_are_found() {
    let (_dir, path) = space();
    let dim = 32;
    let store = open(&path, quantizing(dim)).await;
    put_vectors(&store, VectorKind::Chunk, numbered(clustered(1, 3000, dim, 30, 0.3), 1)).await;
    store.maintain().await.unwrap();
    let watermark: i64 = store
        .read(|c| Ok(c.query_row("SELECT watermark FROM vec_state WHERE tbl = 'vec_chunk'", [], |r| r.get(0))?))
        .await
        .unwrap();
    assert_eq!(watermark, 3000);

    // New rows the quantization can't see; each must still find itself.
    let fresh = numbered(clustered(2, 20, dim, 20, 0.0), 10_001);
    put_vectors(&store, VectorKind::Chunk, fresh.clone()).await;
    for (owner, v) in &fresh {
        assert_eq!(owners(&store, VectorKind::Chunk, v, 1).await, vec![*owner]);
    }
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn requantization_replaces_stale_reader_connections() {
    let (_dir, path) = space();
    let dim = 32;
    let mut cfg = quantizing(dim);
    cfg.readers = 1; // one pooled connection, so the same one would be reused
    let store = open(&path, cfg).await;
    let first = numbered(clustered(21, 3000, dim, 30, 0.3), 1);
    put_vectors(&store, VectorKind::Fact, first.clone()).await;
    store.maintain().await.unwrap();
    assert_eq!(owners(&store, VectorKind::Fact, &first[7].1, 1).await, vec![first[7].0]);

    let second = numbered(clustered(22, 3000, dim, 30, 0.3), 100_001);
    put_vectors(&store, VectorKind::Fact, second.clone()).await;
    let report = store.maintain().await.unwrap();
    assert_eq!(report.requantized, vec![(VectorKind::Fact, 6000)]);
    assert_eq!(store.quantization_generation(VectorKind::Fact).await.unwrap(), 2);
    let replaced_before = store.reader_replacements();

    let all: Vec<(i64, Vec<f32>)> = first.iter().chain(second.iter()).cloned().collect();
    let mut rng = Rng::new(9);
    for _ in 0..20 {
        let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        let exact = brute_force(&all, &q, 5);
        let got = owners(&store, VectorKind::Fact, &q, 5).await;
        let overlap = exact.iter().filter(|o| got.contains(o)).count();
        assert!(overlap >= 4, "exact {exact:?} got {got:?}");
    }
    assert_eq!(store.reader_replacements(), replaced_before + 1, "the one pooled reader was replaced once");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_search_after_requantization_skips_every_stale_pooled_reader() {
    let (_dir, path) = space();
    let dim = 32;
    let mut cfg = quantizing(dim);
    cfg.readers = 4;
    let store = open(&path, cfg).await;
    let items = numbered(clustered(31, 3000, dim, 30, 0.3), 1);
    put_vectors(&store, VectorKind::Fact, items.clone()).await;
    // Four concurrent searches leave four idle connections, all at generation 0.
    let q = &items[3].1;
    let found = tokio::join!(
        owners(&store, VectorKind::Fact, q, 1),
        owners(&store, VectorKind::Fact, q, 1),
        owners(&store, VectorKind::Fact, q, 1),
        owners(&store, VectorKind::Fact, q, 1),
    );
    assert_eq!(found, (vec![4], vec![4], vec![4], vec![4]));

    store.maintain().await.unwrap(); // generation 1: every pooled reader is now behind
    assert_eq!(owners(&store, VectorKind::Fact, q, 1).await, vec![4]);
    assert_eq!(store.reader_replacements(), 1, "one stale reader dropped, then a fresh one opened");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn retired_vectors_disappear_and_get_compacted() {
    let (_dir, path) = space();
    let dim = 16;
    let store = open(&path, StoreConfig::new(dim)).await;
    let items = numbered(clustered(4, 200, dim, 10, 0.3), 1);
    put_vectors(&store, VectorKind::Entity, items.clone()).await;
    let retired: Vec<i64> = (1..=40).collect();
    let ids = retired.clone();
    store
        .write(move |w| {
            for o in &ids {
                assert_eq!(w.retire_vectors(VectorKind::Entity, *o)?, 1);
            }
            Ok(())
        })
        .await
        .unwrap();
    for (owner, v) in items.iter().take(40) {
        let got = owners(&store, VectorKind::Entity, v, 5).await;
        assert!(!got.contains(owner), "retired owner {owner} returned");
        assert_eq!(got.len(), 5);
    }
    let report = store.maintain().await.unwrap();
    assert_eq!(report.compacted, 40);
    let (rows, dead): (i64, i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM vec_entity", [], |r| r.get(0))?,
                c.query_row("SELECT dead_rows FROM vec_state WHERE tbl = 'vec_entity'", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((rows, dead), (160, 0));
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn compacting_below_the_exact_threshold_keeps_recall_when_rows_grow_back() {
    let (_dir, path) = space();
    let dim = 32; // quantizing(): exact scans up to 50_000 cells, i.e. 1562 rows
    let store = open(&path, quantizing(dim)).await;
    let mut rng = Rng::new(81);
    let probe = normalized((0..dim).map(|_| rng.gauss()).collect());
    let near = |rng: &mut Rng, spread: f32| -> Vec<f32> {
        normalized(probe.iter().map(|x| x + spread / (dim as f32).sqrt() * rng.gauss()).collect())
    };
    // 500 decoys hug the probe and will be retired; ten neighbours sit a little farther out.
    let decoys: Vec<(i64, Vec<f32>)> = (1..=500).map(|o| (o, near(&mut rng, 0.05))).collect();
    let neighbours: Vec<(i64, Vec<f32>)> = (1001..=1010).map(|o| (o, near(&mut rng, 0.5))).collect();
    let background = numbered(clustered(82, 1490, dim, 30, 0.3), 2001);
    let first: Vec<(i64, Vec<f32>)> = decoys.iter().chain(&neighbours).chain(&background).cloned().collect();
    put_vectors(&store, VectorKind::Fact, first).await;
    assert_eq!(store.maintain().await.unwrap().requantized, vec![(VectorKind::Fact, 2000)]);

    // Retiring the decoys drops the table to 1500 rows: back to exact scans, and compacted.
    store
        .write(|w| {
            for o in 1..=500 {
                w.retire_vectors(VectorKind::Fact, o)?;
            }
            Ok(())
        })
        .await
        .unwrap();
    let report = store.maintain().await.unwrap();
    assert_eq!(report.compacted, 500);
    assert!(report.requantized.is_empty(), "{report:?}");
    let (watermark, shadow): (i64, i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT watermark FROM vec_state WHERE tbl = 'vec_fact'", [], |r| r.get(0))?,
                c.query_row("SELECT count(*) FROM sqlite_master WHERE name = 'vector0_vec_fact_v'", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((watermark, shadow), (0, 0), "the stale quantization was dropped");

    // 100 new rows cross the threshold again with a delta too small to re-quantize.
    let fresh = numbered(clustered(83, 100, dim, 5, 0.3), 5001);
    put_vectors(&store, VectorKind::Fact, fresh.clone()).await;
    let live: Vec<(i64, Vec<f32>)> = neighbours.iter().chain(&background).chain(&fresh).cloned().collect();
    let want: Vec<i64> = (1001..=1010).collect();
    let mut exact = brute_force(&live, &probe, 10);
    exact.sort_unstable();
    assert_eq!(exact, want);
    let mut got = owners(&store, VectorKind::Fact, &probe, 10).await;
    got.sort_unstable();
    assert_eq!(got, want, "before maintenance");
    assert_eq!(store.maintain().await.unwrap().requantized, vec![(VectorKind::Fact, 1600)]);
    let mut got = owners(&store, VectorKind::Fact, &probe, 10).await;
    got.sort_unstable();
    assert_eq!(got, want, "after re-quantizing");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn replace_vector_moves_an_owner() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(3)).await;
    put_vectors(&store, VectorKind::Key, vec![(1, vec![1.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0])]).await;
    store.write(|w| w.replace_vector(VectorKind::Key, 1, &[0.0, 0.0, 1.0])).await.unwrap();
    let hits = store.knn(VectorKind::Key, &[0.0, 0.0, 1.0], 2, Filter::None).await.unwrap();
    assert_eq!(hits[0].owner, 1);
    assert!((hits[0].score - 1.0).abs() < 1e-5);
    assert_eq!(hits.len(), 2, "one hit per owner");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn two_spaces_in_one_process_stay_independent() {
    let (_a_dir, a_path) = space();
    let (_b_dir, b_path) = space();
    let dim = 32;
    let a = open(&a_path, quantizing(dim)).await;
    let b = open(&b_path, quantizing(dim)).await;
    let a_items = numbered(clustered(31, 2500, dim, 25, 0.3), 1);
    let b_items = numbered(clustered(32, 2500, dim, 25, 0.3), 50_001);
    put_vectors(&a, VectorKind::Fact, a_items.clone()).await;
    put_vectors(&b, VectorKind::Fact, b_items.clone()).await;
    a.maintain().await.unwrap();
    b.maintain().await.unwrap();
    for i in [0usize, 500, 2499] {
        assert_eq!(owners(&a, VectorKind::Fact, &a_items[i].1, 1).await, vec![a_items[i].0]);
        assert_eq!(owners(&b, VectorKind::Fact, &b_items[i].1, 1).await, vec![b_items[i].0]);
        assert!(owners(&a, VectorKind::Fact, &b_items[i].1, 3).await.iter().all(|o| *o < 50_001));
    }
    a.close().await.unwrap();
    b.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn other_sqlite_databases_in_the_process_are_left_alone() {
    let (dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let other = dir.path().join("host-app.db");
    let host = Connection::open(&other).unwrap();
    host.execute_batch("CREATE TABLE notes(body TEXT)").unwrap();
    let tables: Vec<String> = host
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(tables, ["notes"], "opening a Store must not reach the host's databases");
    drop(host);
    let read_only = Connection::open_with_flags(&other, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let notes: i64 = read_only.query_row("SELECT count(*) FROM notes", [], |r| r.get(0)).unwrap();
    assert_eq!(notes, 0);

    // A caller's own connection to a space opts in explicitly.
    let mine = Connection::open(&path).unwrap();
    assert!(mine.query_row("SELECT vector_version()", [], |r| r.get::<_, String>(0)).is_err());
    licorice::store::load_vector_extension(&mine).unwrap();
    licorice::store::load_vector_extension(&mine).unwrap(); // idempotent
    let version: String = mine.query_row("SELECT vector_version()", [], |r| r.get(0)).unwrap();
    assert_eq!(version, "1.1.2");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn quantization_survives_reopen() {
    let (_dir, path) = space();
    let dim = 32;
    let items = numbered(clustered(41, 3000, dim, 30, 0.3), 1);
    {
        let store = open(&path, quantizing(dim)).await;
        put_vectors(&store, VectorKind::Summary, items.clone()).await;
        store.maintain().await.unwrap();
        store.close().await.unwrap();
    }
    let store = open(&path, quantizing(dim)).await;
    assert_eq!(store.quantization_generation(VectorKind::Summary).await.unwrap(), 1);
    assert!(store.maintain().await.unwrap().requantized.is_empty(), "nothing new to quantize");
    for i in [3usize, 1500, 2999] {
        assert_eq!(owners(&store, VectorKind::Summary, &items[i].1, 1).await, vec![items[i].0]);
    }
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn preloaded_and_streamed_quantized_scans_agree() {
    let (_a_dir, a_path) = space();
    let dim = 32;
    let items = numbered(clustered(51, 4000, dim, 40, 0.3), 1);
    let mut cfg = quantizing(dim);
    let plain = open(&a_path, cfg.clone()).await;
    put_vectors(&plain, VectorKind::Fact, items.clone()).await;
    plain.maintain().await.unwrap();
    let mut rng = Rng::new(17);
    let queries: Vec<Vec<f32>> = (0..10).map(|_| (0..dim).map(|_| rng.gauss()).collect()).collect();
    let mut expected = Vec::new();
    for q in &queries {
        expected.push(owners(&plain, VectorKind::Fact, q, 10).await);
    }
    plain.close().await.unwrap();

    cfg.vector.preload = true;
    let preloaded = open(&a_path, cfg).await;
    for (q, want) in queries.iter().zip(&expected) {
        assert_eq!(&owners(&preloaded, VectorKind::Fact, q, 10).await, want);
    }
    preloaded.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn maintenance_runs_under_writes_that_never_pause() {
    let (_dir, path) = space();
    let dim = 32;
    let mut cfg = quantizing(dim); // never idle long enough for the idle timer
    cfg.max_maintenance_delay = Duration::ZERO;
    let store = open(&path, cfg).await;
    put_vectors(&store, VectorKind::Fact, numbered(clustered(71, 3000, dim, 30, 0.3), 1)).await;
    // The next write is queued behind maintenance, so by its reply the pass has run.
    store.write(|_| Ok(())).await.unwrap();
    assert_eq!(store.quantization_generation(VectorKind::Fact).await.unwrap(), 1);
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_panicking_write_rolls_back_and_the_writer_keeps_serving() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let result: licorice::Result<()> = store
        .write(|w| {
            w.insert_vector(VectorKind::Fact, 1, &[1.0, 0.0, 0.0, 0.0])?;
            panic!("bug in a write closure");
        })
        .await;
    assert!(matches!(&result, Err(Error::Task(m)) if m.contains("bug in a write closure")), "{result:?}");
    store.write(|w| w.insert_vector(VectorKind::Fact, 2, &[0.0, 1.0, 0.0, 0.0]).map(drop)).await.unwrap();
    let owners: Vec<i64> = store
        .read(|c| {
            Ok(c.prepare("SELECT owner FROM vec_fact")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(owners, [2], "the panicked insert rolled back");
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_holding_the_last_store_handle_still_commits() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let held = store.clone();
    let (started, wait_started) = std::sync::mpsc::channel::<()>();
    let (go, wait_go) = std::sync::mpsc::channel::<()>();
    let caller = store.clone();
    let task = tokio::spawn(async move {
        caller
            .write(move |w| {
                started.send(()).unwrap();
                wait_go.recv().unwrap();
                w.conn().execute("INSERT INTO meta(key, value) VALUES ('last', 'handle')", [])?;
                drop(held); // the last Store handle, dropped on the writer thread itself
                Ok(())
            })
            .await
    });
    tokio::task::spawn_blocking(move || wait_started.recv().unwrap()).await.unwrap();
    task.abort(); // the caller gives up; with `store` dropped, `held` is the last handle
    let _ = task.await;
    drop(store);
    go.send(()).unwrap();

    // The detached writer commits and exits; poll until a fresh open sees the row.
    for _ in 0..200 {
        let reopened = open(&path, StoreConfig::new(4)).await;
        let seen: Option<String> = reopened
            .read(|c| {
                Ok(c.query_row("SELECT value FROM meta WHERE key = 'last'", [], |r| r.get(0))
                    .map(Some)
                    .or_else(|e| if e == licorice::rusqlite::Error::QueryReturnedNoRows { Ok(None) } else { Err(e) })?)
            })
            .await
            .unwrap();
        reopened.close().await.unwrap();
        if seen.as_deref() == Some("handle") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("the write that dropped the last handle never committed");
}

#[tokio::test(flavor = "multi_thread")]
async fn reads_run_while_writes_queue() {
    let (_dir, path) = space();
    let dim = 16;
    let store = open(&path, StoreConfig::new(dim)).await;
    put_vectors(&store, VectorKind::Fact, numbered(clustered(61, 500, dim, 10, 0.3), 1)).await;
    let mut tasks = Vec::new();
    for t in 0..8u64 {
        let s = store.clone();
        tasks.push(tokio::spawn(async move {
            let mut rng = Rng::new(t);
            for _ in 0..25 {
                let q: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
                assert_eq!(s.knn(VectorKind::Fact, &q, 5, Filter::None).await.unwrap().len(), 5);
            }
        }));
    }
    for batch in 0..20i64 {
        let items = numbered(clustered(100 + batch as u64, 50, dim, 5, 0.3), 1000 + batch * 50);
        put_vectors(&store, VectorKind::Fact, items).await;
    }
    for t in tasks {
        t.await.unwrap();
    }
    let rows: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM vec_fact", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(rows, 1500);
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn unindexed_facts_leave_lexical_results() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let ids = put_facts(&store, vec!["User adopted a cat named Miso", "User visited Lisbon in May"]).await;
    assert_eq!(store.lexical(LexicalKind::Fact, "Miso", 5).await.unwrap()[0].owner, ids[0]);
    let id = ids[0];
    store.write(move |w| w.unindex_text(LexicalKind::Fact, id, "User adopted a cat named Miso")).await.unwrap();
    assert!(store.lexical(LexicalKind::Fact, "Miso", 5).await.unwrap().is_empty());
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn alias_search_tolerates_misspellings() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(4)).await;
    let ids: Vec<i64> = store
        .write(|w| {
            let mut ids = Vec::new();
            for (name, aliases) in [
                ("Melanie", vec!["Melanie", "Mel"]),
                ("Melany Ortiz", vec!["Melany Ortiz"]),
                ("Caroline", vec!["Caroline", "Caro"]),
            ] {
                w.conn().execute("INSERT INTO entities(name, kind) VALUES (?1, 'person')", params![name])?;
                let id = w.conn().last_insert_rowid();
                for a in aliases {
                    assert!(w.add_alias(id, &a.to_lowercase())?);
                }
                assert!(!w.add_alias(id, &aliases_first(name))?, "duplicate alias is ignored");
                ids.push(id);
            }
            Ok(ids)
        })
        .await
        .unwrap();
    let hits = store.lexical(LexicalKind::Alias, "Melanie", 5).await.unwrap();
    let got: Vec<i64> = hits.iter().map(|h| h.owner).collect();
    assert_eq!(got.first(), Some(&ids[0]), "{hits:?}");
    assert_eq!(got.get(1), Some(&ids[1]), "near spelling ranks next: {hits:?}");
    assert!(!got.contains(&ids[2]), "{hits:?}");
    let typo = store.lexical(LexicalKind::Alias, "Carolyne", 5).await.unwrap();
    assert_eq!(typo.first().map(|h| h.owner), Some(ids[2]), "{typo:?}");
    store.close().await.unwrap();
}

fn aliases_first(name: &str) -> String {
    name.to_lowercase()
}

#[tokio::test(flavor = "multi_thread")]
async fn knn_normalizes_queries() {
    let (_dir, path) = space();
    let store = open(&path, StoreConfig::new(2)).await;
    put_vectors(&store, VectorKind::Fact, vec![(1, normalized(vec![1.0, 1.0])), (2, vec![-1.0, 0.5])]).await;
    let hits = store.knn(VectorKind::Fact, &[100.0, 100.0], 1, Filter::None).await.unwrap();
    assert_eq!(hits[0].owner, 1);
    assert!((hits[0].score - 1.0).abs() < 1e-5);
    store.close().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn a_huge_k_returns_every_match() {
    let (_dir, path) = space();
    let dim = 32;
    let store = open(&path, quantizing(dim)).await;
    let items = numbered(clustered(91, 2000, dim, 20, 0.3), 1);
    put_vectors(&store, VectorKind::Fact, items.clone()).await;
    put_vectors(&store, VectorKind::Entity, items[..100].to_vec()).await;
    assert_eq!(store.maintain().await.unwrap().requantized, vec![(VectorKind::Fact, 2000)]);
    put_facts(&store, vec!["User adopted a cat named Miso", "Miso likes tuna"]).await;
    store
        .write(|w| {
            for (name, alias) in [("Melanie", "melanie"), ("Melany", "melany")] {
                w.conn().execute("INSERT INTO entities(name, kind) VALUES (?1, 'person')", params![name])?;
                let id = w.conn().last_insert_rowid();
                w.add_alias(id, alias)?;
            }
            Ok(())
        })
        .await
        .unwrap();

    let q = &items[0].1;
    // On 64-bit: k × oversample overflows; preallocating k slots aborts the process;
    // sqlite-vector reads k × oversample as a 32-bit int, which wraps to "no rows".
    for k in [usize::MAX, usize::MAX >> 20, usize::MAX >> 24] {
        let quantized = store.knn(VectorKind::Fact, q, k, Filter::None).await;
        assert_eq!(quantized.map(|h| h.len()).map_err(|e| e.to_string()), Ok(2000), "quantized knn, k = {k}");
        let exact = store.knn(VectorKind::Entity, q, k, Filter::None).await;
        assert_eq!(exact.map(|h| h.len()).map_err(|e| e.to_string()), Ok(100), "exact knn, k = {k}");
        let filtered = store.knn(VectorKind::Fact, q, k, Filter::Owners(vec![1, 2, 3])).await;
        assert_eq!(filtered.map(|h| h.len()).map_err(|e| e.to_string()), Ok(3), "filtered knn, k = {k}");
        let facts = store.lexical(LexicalKind::Fact, "Miso", k).await;
        assert_eq!(facts.map(|h| h.len()).map_err(|e| e.to_string()), Ok(2), "lexical facts, k = {k}");
        let aliases = store.lexical(LexicalKind::Alias, "Melanie", k).await;
        assert_eq!(aliases.map(|h| h.len()).map_err(|e| e.to_string()), Ok(2), "lexical aliases, k = {k}");
    }
    store.close().await.unwrap();
}
