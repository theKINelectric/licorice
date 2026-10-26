use rusqlite::{Connection, ffi, params};
use std::time::{Duration, Instant};
unsafe extern "C" { fn sqlite3_vector_init(db: *mut ffi::sqlite3, err: *mut *mut std::os::raw::c_char, api: *const ffi::sqlite3_api_routines) -> std::os::raw::c_int; }
fn blob(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unit(seed: u64, d: usize) -> Vec<f32> { let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
  let mut v: Vec<f32> = (0..d).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 2000) as f32 / 1000.0) - 1.0 }).collect();
  let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn median(mut v: Vec<Duration>) -> Duration { v.sort(); v[v.len() / 2] }
fn main() -> rusqlite::Result<()> {
  unsafe { ffi::sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vector_init as *const ()))); }
  let dim = 384;
  for &n in &[20_000u64, 100_000] {
    let path = std::env::temp_dir().join(format!("vecbench{n}.db")); let _ = std::fs::remove_file(&path);
    let c = Connection::open(&path)?; c.pragma_update(None, "journal_mode", "WAL")?;
    c.execute_batch("CREATE TABLE e(id INTEGER PRIMARY KEY AUTOINCREMENT, v BLOB NOT NULL);")?;
    c.query_row("SELECT vector_init('e','v','type=FLOAT32,dimension=384,distance=COSINE,normalized=1')", [], |_| Ok(()))?;
    let tx = c.unchecked_transaction()?; { let mut s = tx.prepare("INSERT INTO e(id,v) VALUES(?1,?2)")?; for i in 1..=n { s.execute(params![i as i64, blob(&unit(i, dim))])?; } } tx.commit()?;
    let qs: Vec<Vec<u8>> = (0..21).map(|i| blob(&unit(1_000_000 + i, dim))).collect();
    let mut full = vec![]; for q in &qs { let t = Instant::now(); let _: Vec<i64> = c.prepare_cached("SELECT rowid FROM vector_full_scan('e','v',?1,20)")?.query_map([q], |r| r.get(0))?.collect::<Result<_,_>>()?; full.push(t.elapsed()); }
    c.query_row("SELECT vector_quantize('e','v','qtype=INT8')", [], |_| Ok(()))?;
    let t = Instant::now(); c.query_row("SELECT vector_quantize('e','v','qtype=INT8')", [], |_| Ok(()))?; let requant = t.elapsed();
    c.query_row("SELECT vector_quantize_preload('e','v')", [], |_| Ok(()))?;
    let mut quant = vec![]; let mut hits = 0;
    for q in &qs { let ex: Vec<i64> = c.prepare_cached("SELECT rowid FROM vector_full_scan('e','v',?1,20)")?.query_map([q], |r| r.get(0))?.collect::<Result<_,_>>()?;
      let t = Instant::now(); let ap: Vec<i64> = c.prepare_cached("SELECT rowid FROM vector_quantize_scan('e','v',?1,20)")?.query_map([q], |r| r.get(0))?.collect::<Result<_,_>>()?; quant.push(t.elapsed());
      hits += ex.iter().filter(|x| ap.contains(x)).count(); }
    // Rust-side exact scan over a 2,000-row "delta" (rows newer than the quantization watermark)
    let delta: Vec<(i64, Vec<f32>)> = c.prepare("SELECT id, v FROM e WHERE id > ?1")?.query_map([(n - 2000) as i64], |r| { let b: Vec<u8> = r.get(1)?; Ok((r.get(0)?, b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())) })?.collect::<Result<_,_>>()?;
    let qv = unit(1_000_000, dim); let t = Instant::now();
    let mut best: Vec<(f32, i64)> = delta.iter().map(|(id, v)| (v.iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>(), *id)).collect(); best.sort_by(|a, b| b.0.total_cmp(&a.0)); let dscan = t.elapsed();
    println!("n={n:>6} dim={dim}: full_scan(exact) median {:?} | INT8 quantize_scan preloaded median {:?} recall@20 {:.3} | re-quantize {:?} | Rust exact scan of 2k delta rows {:?}",
      median(full), median(quant), hits as f64 / (20.0 * qs.len() as f64), requant, dscan);
  }
  Ok(())
}
