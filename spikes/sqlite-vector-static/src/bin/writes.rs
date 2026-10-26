// Apples-to-apples with the Turso spike: FTS5 steady-state commits and single-writer throughput on C SQLite.
use rusqlite::{Connection, params};
use std::time::{Duration, Instant};
fn median(mut v: Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }
fn word(i: u64) -> String { let syl = ["ka","lo","mi","ra","tu","ne","so","vi","de","po"]; let mut x = i; let mut w = String::new(); for _ in 0..3 { w.push_str(syl[(x % 10) as usize]); x /= 10; } w }
fn sentence(seed: u64) -> String { let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493); (0..12).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; word(s % 1000) }).collect::<Vec<_>>().join(" ") }
fn tmp(name: &str) -> std::path::PathBuf { let p = std::env::temp_dir().join(name); for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p.display(), e)); } p }
fn main() -> rusqlite::Result<()> {
  let c = Connection::open(tmp("sf.db"))?; c.pragma_update(None, "journal_mode", "WAL")?; c.pragma_update(None, "synchronous", "NORMAL")?;
  c.execute_batch("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL);
    CREATE VIRTUAL TABLE docs_fts USING fts5(body, content='docs', content_rowid='id');
    CREATE TRIGGER docs_ai AFTER INSERT ON docs BEGIN INSERT INTO docs_fts(rowid, body) VALUES (new.id, new.body); END;")?;
  let t = Instant::now(); let tx = c.unchecked_transaction()?;
  { let mut st = tx.prepare("INSERT INTO docs(id, body) VALUES (?1, ?2)")?; for i in 1..=20_000u64 { st.execute(params![i as i64, sentence(i)])?; } } tx.commit()?;
  println!("[sqlite] FTS5 bulk insert 20000 rows (index maintained by trigger): {:?}", t.elapsed());
  let mut per = vec![];
  for b in 0..20u64 { let t = Instant::now(); let tx = c.unchecked_transaction()?;
    { let mut st = tx.prepare_cached("INSERT INTO docs(body) VALUES (?1)")?; for k in 0..50 { st.execute([sentence(1_000_000 + b * 50 + k)])?; } } tx.commit()?; per.push(t.elapsed()); }
  println!("[sqlite] FTS5 steady-state commit of 50 rows into indexed 20000-row table: median {:?}", median(per));
  let q = format!("{} OR {}", word(7), word(123)); let mut lat = vec![];
  for _ in 0..21 { let t = Instant::now(); let n: Vec<i64> = c.prepare_cached("SELECT rowid FROM docs_fts WHERE docs_fts MATCH ?1 ORDER BY bm25(docs_fts) LIMIT 20")?.query_map([&q], |r| r.get(0))?.collect::<Result<_,_>>()?; let _ = n; lat.push(t.elapsed()); }
  println!("[sqlite] FTS5 BM25 top-20 median {:?}", median(lat));
  for sync in ["NORMAL", "FULL"] {
  let c = Connection::open(tmp("sw.db"))?; c.pragma_update(None, "journal_mode", "WAL")?; c.pragma_update(None, "synchronous", sync)?;
  c.execute_batch("CREATE TABLE facts(id INTEGER PRIMARY KEY, session INTEGER, statement TEXT); CREATE INDEX facts_session ON facts(session);")?;
  let t = Instant::now();
  for tx_i in 0..1600u64 { let tx = c.unchecked_transaction()?; { let mut st = tx.prepare_cached("INSERT INTO facts(session, statement) VALUES (?1, ?2)")?;
    for k in 0..10 { st.execute(params![tx_i as i64, sentence(tx_i * 10 + k)])?; } } tx.commit()?; }
  let el = t.elapsed(); println!("[sqlite] single writer, 1600 txns x 10 rows (synchronous={sync}): {:?} ({:.0} rows/s)", el, 16000.0 / el.as_secs_f64());
  }
  Ok(())
}
