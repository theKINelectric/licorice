use rusqlite::{Connection, ffi, params};

unsafe extern "C" {
    fn sqlite3_vector_init(db: *mut ffi::sqlite3, err: *mut *mut std::os::raw::c_char, api: *const ffi::sqlite3_api_routines) -> std::os::raw::c_int;
}

fn f32_blob(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn unit(seed: u64, d: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v: Vec<f32> = (0..d).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s % 2000) as f32 / 1000.0) - 1.0 }).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v
}

fn main() -> rusqlite::Result<()> {
    unsafe { ffi::sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vector_init as *const ()))); }
    let path = std::env::temp_dir().join("vecspike.db");
    let _ = std::fs::remove_file(&path);
    let dim = 384;
    let w = Connection::open(&path)?;
    w.pragma_update(None, "journal_mode", "WAL")?;
    let ver: String = w.query_row("SELECT vector_version() || ' / ' || vector_backend()", [], |r| r.get(0))?;
    println!("sqlite {} | sqlite-vector {}", rusqlite::version(), ver);
    w.execute_batch("CREATE TABLE emb_fact(id INTEGER PRIMARY KEY AUTOINCREMENT, v BLOB NOT NULL);
                     CREATE TABLE facts(id INTEGER PRIMARY KEY, status TEXT NOT NULL);")?;
    w.query_row("SELECT vector_init('emb_fact','v','type=FLOAT32,dimension=384,distance=COSINE,normalized=1')", [], |_| Ok(()))?;
    let n = 20_000u64;
    let tx = w.unchecked_transaction()?;
    { let mut ins = tx.prepare("INSERT INTO emb_fact(id, v) VALUES (?1, ?2)")?;
      let mut f = tx.prepare("INSERT INTO facts(id, status) VALUES (?1, ?2)")?;
      for i in 1..=n { ins.execute(params![i as i64, f32_blob(&unit(i, dim))])?; f.execute(params![i as i64, if i % 10 == 0 { "superseded" } else { "active" }])?; } }
    tx.commit()?;
    let q = f32_blob(&unit(4242, dim));
    let t = std::time::Instant::now();
    let exact: Vec<(i64, f64)> = w.prepare("SELECT rowid, distance FROM vector_full_scan('emb_fact','v',?1,10)")?
        .query_map([&q], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_,_>>()?;
    println!("full_scan top1 {:?} ({} rows) in {:?}", exact[0], n, t.elapsed());
    assert_eq!(exact[0].0, 4242, "self-match should be nearest");
    let qn: i64 = w.query_row("SELECT vector_quantize('emb_fact','v','qtype=INT8')", [], |r| r.get(0))?;
    w.query_row("SELECT vector_quantize_preload('emb_fact','v')", [], |_| Ok(()))?;
    // Rows inserted after quantize: are they visible to quantize_scan?
    let late = f32_blob(&unit(999_999, dim));
    w.execute("INSERT INTO emb_fact(id, v) VALUES (?1, ?2)", params![(n + 1) as i64, late])?;
    w.execute("INSERT INTO facts(id, status) VALUES (?1, 'active')", params![(n + 1) as i64])?;
    // Second connection: must call vector_init itself; quantization is shared.
    let r = Connection::open(&path)?;
    r.query_row("SELECT vector_init('emb_fact','v','type=FLOAT32,dimension=384,distance=COSINE,normalized=1')", [], |_| Ok(()))?;
    let t = std::time::Instant::now();
    let approx: Vec<i64> = r.prepare("SELECT rowid FROM vector_quantize_scan('emb_fact','v',?1,10)")?
        .query_map([&q], |r| r.get(0))?.collect::<Result<_,_>>()?;
    println!("quantize_scan (INT8, {} quantized) top1 {} in {:?} on 2nd connection", qn, approx[0], t.elapsed());
    let recall = exact.iter().filter(|(id, _)| approx.contains(id)).count();
    println!("recall@10 vs exact: {}/10", recall);
    let late_hit: Option<i64> = r.query_row("SELECT rowid FROM vector_quantize_scan('emb_fact','v',?1,1)", [&late], |r| r.get(0)).ok();
    println!("row inserted after quantize, quantize_scan top1 for its own vector = {:?} (expected != {} if unindexed)", late_hit, n + 1);
    let late_full: i64 = r.query_row("SELECT rowid FROM vector_full_scan('emb_fact','v',?1,1)", [&late], |r| r.get(0))?;
    println!("same via full_scan = {}", late_full);
    // Streaming mode + relational filter
    let filtered: Vec<(i64, String)> = r.prepare("SELECT f.id, f.status FROM vector_quantize_scan('emb_fact','v',?1) AS v JOIN facts f ON f.id = v.rowid WHERE f.status='active' ORDER BY v.distance LIMIT 5")?
        .query_map([&q], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_,_>>()?;
    println!("streaming+filter: {:?}", filtered);
    Ok(())
}
