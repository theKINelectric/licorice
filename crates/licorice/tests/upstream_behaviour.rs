//! Pins the sqlite-vector 1.1.2 behaviour the reader pool is designed around: a
//! connection's quantization parameters are read once, when `vector_init` runs, and a
//! later `vector_quantize` on another connection does not reach it. If an upgrade
//! changes this, this test says so.

mod common;

use common::{clustered, put_vectors, space};
use licorice::rusqlite::{Connection, params};
use licorice::{Store, StoreConfig, VectorKind};

fn blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn init(conn: &Connection, dim: usize) {
    licorice::store::load_vector_extension(conn).unwrap();
    let opts = format!("type=FLOAT32,dimension={dim},distance=COSINE,normalized=1");
    conn.query_row("SELECT vector_init('vec_fact', 'v', ?1)", [opts], |_| Ok(())).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn quantization_parameters_are_per_connection() {
    let (_dir, path) = space();
    let dim = 16;
    let store = Store::open(&path, StoreConfig::new(dim)).await.unwrap();
    let items: Vec<(i64, Vec<f32>)> =
        clustered(3, 500, dim, 10, 0.3).into_iter().enumerate().map(|(i, v)| (i as i64 + 1, v)).collect();
    put_vectors(&store, VectorKind::Fact, items.clone()).await;
    store.close().await.unwrap();

    let early = Connection::open(&path).unwrap();
    init(&early, dim); // before any quantization exists

    let writer = Connection::open(&path).unwrap();
    init(&writer, dim);
    let n: i64 = writer.query_row("SELECT vector_quantize('vec_fact', 'v', 'qtype=INT8')", [], |r| r.get(0)).unwrap();
    assert_eq!(n, 500);

    let late = Connection::open(&path).unwrap();
    init(&late, dim);

    let q = blob(&items[42].1);
    let scan = |conn: &Connection| -> Result<Vec<i64>, String> {
        let mut stmt = conn
            .prepare("SELECT rowid FROM vector_quantize_scan('vec_fact', 'v', ?1, 5)")
            .map_err(|e| e.to_string())?;
        let rows = stmt.query_map(params![q], |r| r.get::<_, i64>(0)).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    };
    let late_result = scan(&late).expect("a connection initialized after quantizing can scan");
    assert_eq!(late_result.first(), Some(&43));
    let early_result = scan(&early);
    eprintln!("connection initialized before quantizing: {early_result:?}");
    assert_ne!(
        early_result.as_ref().ok(),
        Some(&late_result),
        "upstream now refreshes stale connections; revisit the reader pool"
    );
}
