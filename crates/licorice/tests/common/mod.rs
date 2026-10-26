//! Shared test helpers: deterministic vectors and fixture writers.
#![allow(dead_code)]

use std::path::PathBuf;

use licorice::{LexicalKind, Store, StoreConfig, VectorKind};
use tempfile::TempDir;

pub fn space() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("space.licorice");
    (dir, path)
}

pub async fn open(path: &PathBuf, cfg: StoreConfig) -> Store {
    Store::open(path, cfg).await.expect("open store")
}

/// xorshift64*: deterministic, dependency-free.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal via Box–Muller.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

pub fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

/// `n` vectors around `clusters` random centroids: closer to real embeddings than
/// uniform noise, which is the worst case for quantization. `spread` is the noise norm
/// relative to the (unit) centroid.
pub fn clustered(seed: u64, n: usize, dim: usize, clusters: usize, spread: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let spread = spread / (dim as f32).sqrt();
    let centroids: Vec<Vec<f32>> = (0..clusters).map(|_| normalized((0..dim).map(|_| rng.gauss()).collect())).collect();
    (0..n)
        .map(|i| {
            let c = &centroids[i % clusters];
            normalized(c.iter().map(|x| x + spread * rng.gauss()).collect())
        })
        .collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Exact top-k owners by cosine similarity.
pub fn brute_force(items: &[(i64, Vec<f32>)], q: &[f32], k: usize) -> Vec<i64> {
    let q = normalized(q.to_vec());
    let mut scored: Vec<(f32, i64)> = items.iter().map(|(o, v)| (dot(&q, &normalized(v.clone())), *o)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, o)| o).collect()
}

pub async fn put_vectors(store: &Store, kind: VectorKind, items: Vec<(i64, Vec<f32>)>) {
    store
        .write(move |w| {
            for (owner, v) in &items {
                w.insert_vector(kind, *owner, v)?;
            }
            Ok(())
        })
        .await
        .expect("insert vectors");
}

/// Inserts one entity and a fact per statement, indexing each for lexical search.
/// Returns the fact ids in order.
pub async fn put_facts(store: &Store, statements: Vec<&'static str>) -> Vec<i64> {
    store
        .write(move |w| {
            w.conn().execute("INSERT INTO entities(name, kind) VALUES ('User', 'person')", [])?;
            let subject = w.conn().last_insert_rowid();
            let mut ids = Vec::new();
            for (i, s) in statements.iter().enumerate() {
                w.conn().execute(
                    "INSERT INTO facts(subject_id, predicate, object_value, statement, recorded_first, recorded_last, dedup_key) \
                     VALUES (?1, 'states', ?2, ?2, 0, 0, ?3)",
                    licorice::rusqlite::params![subject, s, format!("k{i}")],
                )?;
                let id = w.conn().last_insert_rowid();
                w.index_text(LexicalKind::Fact, id, s)?;
                ids.push(id);
            }
            Ok(ids)
        })
        .await
        .expect("insert facts")
}
