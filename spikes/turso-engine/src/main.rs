use std::time::{Duration, Instant};
use turso::{Builder, Error, Value};

fn blob(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unit(seed: u64, d: usize) -> Vec<f32> { let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  let mut v: Vec<f32> = (0..d).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 2000) as f32 / 1000.0) - 1.0 }).collect();
  let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn median(mut v: Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }
fn word(i: u64) -> String { let syl = ["ka","lo","mi","ra","tu","ne","so","vi","de","po"]; let mut x = i; let mut w = String::new(); for _ in 0..3 { w.push_str(syl[(x % 10) as usize]); x /= 10; } w }
fn sentence(seed: u64) -> String { let mut s = seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493); (0..12).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; word(s % 1000) }).collect::<Vec<_>>().join(" ") }
fn tmp(name: &str) -> String { let p = std::env::temp_dir().join(name); for ext in ["", "-wal", "-shm", "-log", ".tshm"] { let _ = std::fs::remove_file(format!("{}{}", p.display(), ext)); } p.display().to_string() }

async fn vectors(n: u64, dim: usize) -> Result<(), Error> {
  let db = Builder::new_local(&tmp(&format!("tv{n}.db"))).build().await?; let c = db.connect()?;
  c.execute("CREATE TABLE e(id INTEGER PRIMARY KEY, v BLOB NOT NULL)", ()).await?;
  let t = Instant::now(); c.execute("BEGIN", ()).await?;
  { let mut st = c.prepare("INSERT INTO e(id, v) VALUES (?1, vector32(?2))").await?;
    for i in 1..=n { st.execute((i as i64, Value::Blob(blob(&unit(i, dim))))).await?; st.reset()?; } }
  c.execute("COMMIT", ()).await?; let ins = t.elapsed();
  let mut lat = vec![]; let mut self_ok = true;
  for qi in 0..21u64 { let id = 1 + qi * (n / 21); let q = Value::Blob(blob(&unit(id, dim)));
    let t = Instant::now();
    let mut rows = c.query("SELECT id, vector_distance_cos(v, vector32(?1)) AS d FROM e ORDER BY d LIMIT 20", (q,)).await?;
    let mut ids = vec![]; while let Some(r) = rows.next().await? { ids.push(r.get::<i64>(0)?); }
    lat.push(t.elapsed()); if ids.first() != Some(&(id as i64)) { self_ok = false; } }
  println!("[vector] n={n:>6} dim={dim}: insert {:?} | exact top-20 via ORDER BY vector_distance_cos median {:?} | self-match top1 ok={self_ok}", ins, median(lat));
  Ok(())
}

async fn fts(n: u64) -> Result<(), Error> {
  let db = Builder::new_local(&tmp("tf.db")).experimental_index_method(true).build().await?; let c = db.connect()?;
  c.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL)", ()).await?;
  c.execute("CREATE INDEX docs_fts ON docs USING fts (body)", ()).await?;
  let t = Instant::now(); c.execute("BEGIN", ()).await?;
  { let mut st = c.prepare("INSERT INTO docs(id, body) VALUES (?1, ?2)").await?;
    for i in 1..=n { st.execute((i as i64, sentence(i))).await?; st.reset()?; } }
  c.execute("COMMIT", ()).await?; let ins = t.elapsed();
  let q = format!("{} {}", word(7), word(123));
  let mut lat = vec![]; let mut hits = 0;
  for _ in 0..21 { let t = Instant::now();
    let mut rows = c.query("SELECT id, fts_score(body, ?1) AS s FROM docs WHERE fts_match(body, ?1) ORDER BY s DESC LIMIT 20", (q.clone(),)).await?;
    hits = 0; while let Some(_r) = rows.next().await? { hits += 1; } lat.push(t.elapsed()); }
  // real-time: a row inserted after the bulk load must be searchable immediately
  c.execute("INSERT INTO docs(id, body) VALUES (?1, 'caroline borrowed an earring from melanie')", ((n + 1) as i64,)).await?;
  let mut rows = c.query("SELECT id FROM docs WHERE fts_match(body, 'earring')", ()).await?; let fresh = rows.next().await?.is_some();
  println!("[fts]    n={n:>6}: insert {:?} | BM25 top-20 median {:?} ({hits} hits) | fresh row searchable={fresh}", ins, median(lat));
  Ok(())
}

async fn fts_variants(n: u64) -> Result<(), Error> {
  // (a) bulk load, then build the index
  let db = Builder::new_local(&tmp("tf2.db")).experimental_index_method(true).build().await?; let c = db.connect()?;
  c.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL)", ()).await?;
  c.execute("BEGIN", ()).await?;
  { let mut st = c.prepare("INSERT INTO docs(id, body) VALUES (?1, ?2)").await?; for i in 1..=n { st.execute((i as i64, sentence(i))).await?; st.reset()?; } }
  c.execute("COMMIT", ()).await?;
  let t = Instant::now(); c.execute("CREATE INDEX docs_fts ON docs USING fts (body)", ()).await?;
  println!("[fts]    bulk-load {n} rows then CREATE INDEX: {:?}", t.elapsed());
  // (b) steady-state ingestion: 20 commits of 50 rows each on top of the indexed table
  let mut per = vec![];
  for b in 0..20u64 { let t = Instant::now(); c.execute("BEGIN", ()).await?;
    { let mut st = c.prepare("INSERT INTO docs(body) VALUES (?1)").await?; for k in 0..50 { st.execute((sentence(1_000_000 + b * 50 + k),)).await?; st.reset()?; } }
    c.execute("COMMIT", ()).await?; per.push(t.elapsed()); }
  println!("[fts]    steady-state commit of 50 rows into indexed {n}-row table: median {:?}", median(per));
  // (c) same 50-row commit without any FTS index, for comparison
  let db = Builder::new_local(&tmp("tf3.db")).build().await?; let c = db.connect()?;
  c.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL)", ()).await?;
  let mut per = vec![];
  for b in 0..20u64 { let t = Instant::now(); c.execute("BEGIN", ()).await?;
    { let mut st = c.prepare("INSERT INTO docs(body) VALUES (?1)").await?; for k in 0..50 { st.execute((sentence(b * 50 + k),)).await?; st.reset()?; } }
    c.execute("COMMIT", ()).await?; per.push(t.elapsed()); }
  println!("[fts]    same 50-row commit with no FTS index: median {:?}", median(per));
  Ok(())
}

async fn writers(mvcc: bool, tasks: usize, txns: usize) -> Result<(), Error> {
  let db = Builder::new_local(&tmp(&format!("tw{mvcc}.db"))).build().await?; let c = db.connect()?;
  if mvcc { c.pragma_update("journal_mode", "'mvcc'").await?; }
  c.execute("CREATE TABLE facts(id INTEGER PRIMARY KEY, session INTEGER, statement TEXT)", ()).await?;
  c.execute("CREATE INDEX facts_session ON facts(session)", ()).await?;
  let t = Instant::now(); let mut hs = vec![];
  for w in 0..tasks { let db = db.clone(); hs.push(tokio::spawn(async move {
    let c = db.connect()?; let mut retries = 0u64;
    for tx in 0..txns { loop {
      if c.execute(if mvcc { "BEGIN CONCURRENT" } else { "BEGIN IMMEDIATE" }, ()).await.is_err() { retries += 1; tokio::task::yield_now().await; continue; }
      let mut r = Ok(0);
      for k in 0..10 { r = c.execute("INSERT INTO facts(session, statement) VALUES (?1, ?2)", ((w * 1000 + tx) as i64, sentence((w * 100000 + tx * 10 + k) as u64))).await; if r.is_err() { break; } }
      let r = r.and(c.execute("COMMIT", ()).await);
      match r { Ok(_) => break, Err(_) => { let _ = c.execute("ROLLBACK", ()).await; retries += 1; tokio::task::yield_now().await; } } } }
    Ok::<u64, Error>(retries) })); }
  let mut retries = 0; for h in hs { retries += h.await.expect("join")?; }
  let el = t.elapsed();
  let mut rows = c.query("SELECT count(*) FROM facts", ()).await?; let cnt: i64 = rows.next().await?.unwrap().get(0)?;
  println!("[write]  mode={} tasks={tasks}: {cnt} rows in {:?} ({:.0} rows/s), retries={retries}", if mvcc { "MVCC BEGIN CONCURRENT" } else { "WAL BEGIN IMMEDIATE  " }, el, cnt as f64 / el.as_secs_f64());
  Ok(())
}

async fn fts_under_mvcc() -> Result<(), Error> {
  let db = Builder::new_local(&tmp("tfm.db")).experimental_index_method(true).build().await?; let c = db.connect()?;
  c.pragma_update("journal_mode", "'mvcc'").await?;
  c.execute("CREATE TABLE docs(id INTEGER PRIMARY KEY, body TEXT NOT NULL)", ()).await?;
  let r = c.execute("CREATE INDEX docs_fts ON docs USING fts (body)", ()).await;
  match r { Err(e) => println!("[fts+mvcc] create index failed: {e}"), Ok(_) => {
    c.execute("BEGIN CONCURRENT", ()).await?; c.execute("INSERT INTO docs(body) VALUES ('melanie borrowed an earring')", ()).await?; c.execute("COMMIT", ()).await?;
    let mut rows = c.query("SELECT id FROM docs WHERE fts_match(body, 'earring')", ()).await?;
    println!("[fts+mvcc] FTS index under MVCC works, hit={}", rows.next().await?.is_some()); } }
  Ok(())
}

async fn vectors8(n: u64, dim: usize) -> Result<(), Error> {
  let db = Builder::new_local(&tmp(&format!("tv8{n}.db"))).build().await?; let c = db.connect()?;
  c.execute("CREATE TABLE e(id INTEGER PRIMARY KEY, v BLOB NOT NULL)", ()).await?;
  c.execute("BEGIN", ()).await?;
  { let mut st = c.prepare("INSERT INTO e(id, v) VALUES (?1, vector8(vector32(?2)))").await?;
    for i in 1..=n { st.execute((i as i64, Value::Blob(blob(&unit(i, dim))))).await?; st.reset()?; } }
  c.execute("COMMIT", ()).await?;
  let mut lat = vec![]; let mut ok = 0;
  for qi in 0..21u64 { let id = 1 + qi * (n / 21); let q = Value::Blob(blob(&unit(id, dim)));
    let t = Instant::now();
    let r = c.query("SELECT id FROM e ORDER BY vector_distance_cos(v, vector8(vector32(?1))) LIMIT 20", (q,)).await;
    let mut rows = match r { Ok(r) => r, Err(e) => { println!("[vector8] unsupported: {e}"); return Ok(()); } };
    let mut ids = vec![]; while let Some(r) = rows.next().await? { ids.push(r.get::<i64>(0)?); }
    lat.push(t.elapsed()); if ids.first() == Some(&(id as i64)) { ok += 1; } }
  println!("[vector8] n={n:>6} dim={dim}: 8-bit quantized exact scan median {:?} | self-match top1 {ok}/21", median(lat));
  Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
  for n in [20_000u64, 100_000] { vectors(n, 384).await?; }
  for n in [20_000u64, 100_000] { vectors8(n, 384).await?; }
  fts_variants(20_000).await?;
  writers(false, 1, 1600).await?;
  writers(false, 8, 200).await?;
  writers(true, 8, 200).await?;
  fts_under_mvcc().await?;
  Ok(())
}
