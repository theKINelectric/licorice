use libsql::{Builder, Value};
use std::time::{Duration, Instant};
fn blob(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unit(seed: u64, d: usize) -> Vec<f32> { let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  let mut v: Vec<f32> = (0..d).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 2000) as f32 / 1000.0) - 1.0 }).collect();
  let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn median(mut v: Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }
fn tmp_path(name: &str) -> String { std::env::temp_dir().join(name).display().to_string() }
fn tmp(name: &str) -> String { let p = std::env::temp_dir().join(name); for e in ["", "-wal", "-shm"] { let _ = std::fs::remove_file(format!("{}{}", p.display(), e)); } p.display().to_string() }
#[tokio::main]
async fn main() -> Result<(), libsql::Error> {
  let dim = 384;
  for (n, opts) in [(20_000u64, "'metric=cosine'"), (20_000u64, "'metric=cosine', 'compress_neighbors=float8', 'max_neighbors=32'")] {
    let db = Builder::new_local(tmp(&format!("ls{n}_{}.db", opts.len()))).build().await?; let c = db.connect()?;
    c.execute("CREATE TABLE e(id INTEGER PRIMARY KEY, v F32_BLOB(384) NOT NULL)", ()).await?;
    c.execute(&format!("CREATE INDEX e_idx ON e(libsql_vector_idx(v, {opts}))"), ()).await?;
    let t = Instant::now(); c.execute("BEGIN", ()).await?;
    { let mut st = c.prepare("INSERT INTO e(id, v) VALUES (?1, vector32(?2))").await?;
      for i in 1..=n { st.execute((i as i64, Value::Blob(blob(&unit(i, dim))))).await?; st.reset(); } }
    c.execute("COMMIT", ()).await?; let ins = t.elapsed();
    let qs: Vec<Vec<u8>> = (0..21).map(|i| blob(&unit(1_000_000 + i, dim))).collect();
    let (mut ann_lat, mut ex_lat, mut hits) = (vec![], vec![], 0usize);
    for q in &qs {
      let t = Instant::now(); let mut rows = c.query("SELECT e.id FROM vector_top_k('e_idx', vector32(?1), 20) AS t JOIN e ON e.rowid = t.id", [Value::Blob(q.clone())]).await?;
      let mut ann = vec![]; while let Some(r) = rows.next().await? { ann.push(r.get::<i64>(0)?); } ann_lat.push(t.elapsed());
      let t = Instant::now(); let mut rows = c.query("SELECT id FROM e ORDER BY vector_distance_cos(v, vector32(?1)) LIMIT 20", [Value::Blob(q.clone())]).await?;
      let mut ex = vec![]; while let Some(r) = rows.next().await? { ex.push(r.get::<i64>(0)?); } ex_lat.push(t.elapsed());
      hits += ex.iter().filter(|x| ann.contains(x)).count(); }
    // incremental insert cost once the index is populated (real-time ingestion shape)
    let mut per = vec![];
    for b in 0..10u64 { let t = Instant::now(); c.execute("BEGIN", ()).await?;
      { let mut st = c.prepare("INSERT INTO e(v) VALUES (vector32(?1))").await?; for k in 0..50 { st.execute([Value::Blob(blob(&unit(5_000_000 + b * 50 + k, dim)))]).await?; st.reset(); } }
      c.execute("COMMIT", ()).await?; per.push(t.elapsed()); }
    let size = std::fs::metadata(tmp_path(&format!("ls{n}_{}.db", opts.len()))).map(|m| m.len()).unwrap_or(0);
    println!("[libsql] n={n:>6} dim={dim} idx({opts}) db={} MB: insert w/ DiskANN index {:?} | vector_top_k median {:?} recall@20 {:.3} | exact ORDER BY median {:?} | 50-row commit into indexed table median {:?}",
      size / 1_000_000, ins, median(ann_lat), hits as f64 / (20.0 * qs.len() as f64), median(ex_lat), median(per));
  }
  Ok(())
}
