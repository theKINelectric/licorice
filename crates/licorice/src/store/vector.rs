//! Vector kNN over sqlite-vector (SPEC §7.2, §7.3).
//!
//! Each embedded kind has an append-only table `(vid, owner, live, v)`. Below
//! `exact_max_cells` (rows × dim) every search is an exact SIMD scan. Above it, the
//! sealed part is searched through the INT8 quantization and the delta (rows with
//! `vid > watermark`, which the quantization cannot see) is scanned exactly in Rust;
//! all candidates are then rescored exactly from their FLOAT32 vectors.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params, params_from_iter};

use super::readers::ReaderConn;
use super::{Filter, Hit, VectorConfig};
use crate::{Error, Result};

/// An embedded kind of item, each with its own vector table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VectorKind {
    Fact,
    Entity,
    Chunk,
    /// Summary keys and theme titles.
    Key,
    Summary,
}

impl VectorKind {
    pub const COUNT: usize = 5;
    pub const ALL: [VectorKind; Self::COUNT] =
        [VectorKind::Fact, VectorKind::Entity, VectorKind::Chunk, VectorKind::Key, VectorKind::Summary];

    pub fn table(self) -> &'static str {
        match self {
            VectorKind::Fact => "vec_fact",
            VectorKind::Entity => "vec_entity",
            VectorKind::Chunk => "vec_chunk",
            VectorKind::Key => "vec_key",
            VectorKind::Summary => "vec_summary",
        }
    }

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// Owner ids per statement when filtering or rescoring by id, well under SQLite's
/// variable limit.
const ID_BATCH: usize = 500;

/// Returns `v` scaled to unit length.
pub(crate) fn normalize(v: &[f32], dim: usize) -> Result<Vec<f32>> {
    if v.len() != dim {
        return Err(Error::Dimension { expected: dim, got: v.len() });
    }
    let norm = dot(v, v).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(Error::InvalidVector);
    }
    Ok(v.iter().map(|x| x / norm).collect())
}

pub(crate) fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Dot product with eight independent accumulators, which the compiler vectorizes.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (a8, a_rest) = a.split_at(a.len() - a.len() % 8);
    let (b8, b_rest) = b.split_at(a8.len());
    for (x, y) in a8.chunks_exact(8).zip(b8.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let tail: f32 = a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum();
    acc.iter().sum::<f32>() + tail
}

pub(crate) fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

// ---- writer side -------------------------------------------------------------------

/// Appends a vector for `owner`; returns its vid.
pub(crate) fn insert(conn: &Connection, kind: VectorKind, owner: i64, v: &[f32], dim: usize) -> Result<i64> {
    let v = normalize(v, dim)?;
    conn.execute(&format!("INSERT INTO {}(owner, v) VALUES (?1, ?2)", kind.table()), params![owner, to_blob(&v)])?;
    let vid = conn.last_insert_rowid();
    conn.execute("UPDATE vec_state SET live_rows = live_rows + 1 WHERE tbl = ?1", [kind.table()])?;
    Ok(vid)
}

/// Marks every live vector of `owner` dead; returns how many.
pub(crate) fn retire(conn: &Connection, kind: VectorKind, owner: i64) -> Result<usize> {
    let n = conn.execute(&format!("UPDATE {} SET live = 0 WHERE owner = ?1 AND live = 1", kind.table()), [owner])?;
    if n > 0 {
        conn.execute(
            "UPDATE vec_state SET live_rows = live_rows - ?1, dead_rows = dead_rows + ?1 WHERE tbl = ?2",
            params![n as i64, kind.table()],
        )?;
    }
    Ok(n)
}

struct State {
    watermark: i64,
    generation: i64,
    live_rows: i64,
    dead_rows: i64,
}

fn state(conn: &Connection, kind: VectorKind) -> Result<State> {
    Ok(conn.query_row(
        "SELECT watermark, generation, live_rows, dead_rows FROM vec_state WHERE tbl = ?1",
        [kind.table()],
        |r| Ok(State { watermark: r.get(0)?, generation: r.get(1)?, live_rows: r.get(2)?, dead_rows: r.get(3)? }),
    )?)
}

/// What one maintenance pass did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaintenanceReport {
    /// Kinds that were re-quantized, with the number of rows quantized.
    pub requantized: Vec<(VectorKind, i64)>,
    /// Dead vector rows physically removed.
    pub compacted: i64,
}

/// Re-quantizes kinds whose delta or dead fraction crossed the configured thresholds
/// (SPEC §7.3). Runs on the writer connection, one transaction per kind.
pub(crate) fn maintain(conn: &mut Connection, cfg: &VectorConfig, dim: usize) -> Result<MaintenanceReport> {
    let mut report = MaintenanceReport::default();
    for kind in VectorKind::ALL {
        let st = state(conn, kind)?;
        let cells = st.live_rows as u64 * dim as u64;
        let total = st.live_rows + st.dead_rows;
        let dead_due = total > 0 && st.dead_rows as f64 > cfg.requantize_dead_frac * total as f64;
        if cells <= cfg.exact_max_cells {
            // Exact scans don't need a quantization; still drop dead rows once they pile up.
            if dead_due {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
                report.compacted += compact(&tx, kind)?;
                if st.watermark != 0 {
                    // The quantization still holds the deleted rows, which the dead-row
                    // over-fetch no longer covers. Drop it, so that growing past the
                    // threshold again re-quantizes instead of scanning stale vids.
                    tx.query_row("SELECT vector_quantize_cleanup(?1, 'v')", [kind.table()], |_| Ok(()))?;
                    tx.execute(
                        "UPDATE vec_state SET watermark = 0, qtype = NULL, quantized_at = NULL WHERE tbl = ?1",
                        [kind.table()],
                    )?;
                }
                tx.commit()?;
            }
            continue;
        }
        let delta: i64 = conn.query_row(
            &format!("SELECT count(*) FROM {} WHERE vid > ?1 AND live = 1", kind.table()),
            [st.watermark],
            |r| r.get(0),
        )?;
        let sealed = (st.live_rows - delta).max(0);
        let delta_limit = (cfg.requantize_min_delta as f64).max(cfg.requantize_delta_frac * sealed as f64);
        if st.watermark == 0 || delta as f64 > delta_limit || dead_due {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            report.compacted += compact(&tx, kind)?;
            let quantized: i64 =
                tx.query_row("SELECT vector_quantize(?1, 'v', 'qtype=INT8')", [kind.table()], |r| r.get(0))?;
            let watermark: i64 =
                tx.query_row(&format!("SELECT coalesce(max(vid), 0) FROM {}", kind.table()), [], |r| r.get(0))?;
            tx.execute(
                "UPDATE vec_state SET watermark = ?1, generation = generation + 1, qtype = 'INT8', quantized_at = ?2 \
                 WHERE tbl = ?3",
                params![watermark, now_ms(), kind.table()],
            )?;
            tx.commit()?;
            tracing::debug!(table = kind.table(), quantized, watermark, "re-quantized");
            report.requantized.push((kind, quantized));
        }
    }
    Ok(report)
}

fn compact(tx: &Transaction<'_>, kind: VectorKind) -> Result<i64> {
    let n = tx.execute(&format!("DELETE FROM {} WHERE live = 0", kind.table()), [])? as i64;
    tx.execute("UPDATE vec_state SET dead_rows = 0 WHERE tbl = ?1", [kind.table()])?;
    Ok(n)
}

// ---- reader side -------------------------------------------------------------------

/// k nearest owners by cosine similarity, best first. `query` must already be normalized.
/// Any `k` is accepted: there are never more hits than live rows.
pub(crate) fn knn(
    rc: &mut ReaderConn,
    cfg: &VectorConfig,
    dim: usize,
    kind: VectorKind,
    query: &[f32],
    k: usize,
    filter: &Filter,
) -> Result<Vec<Hit>> {
    if k == 0 {
        return Ok(Vec::new());
    }
    let tx = rc.conn.unchecked_transaction()?;
    let st = state(&tx, kind)?;
    if st.generation != rc.generations[kind.index()] {
        return Err(Error::StaleReader);
    }
    // Capping `k` at the live rows also bounds every count and allocation derived from it.
    let k = k.min(usize::try_from(st.live_rows).unwrap_or(0));
    if k == 0 {
        return Ok(Vec::new());
    }
    let blob = to_blob(query);
    let table = kind.table();
    let mut best = Best::default();

    match filter {
        Filter::Owners(owners) => {
            for batch in owners.chunks(ID_BATCH) {
                let sql =
                    format!("SELECT owner, v FROM {table} WHERE live = 1 AND owner IN ({})", placeholders(batch.len()));
                let mut stmt = tx.prepare_cached(&sql)?;
                let mut rows = stmt.query(params_from_iter(batch))?;
                while let Some(row) = rows.next()? {
                    let v: Vec<u8> = row.get(1)?;
                    best.keep(row.get(0)?, dot(query, &from_blob(&v)));
                }
            }
        }
        Filter::None => {
            // No scan returns more candidates than the table has rows, and sqlite-vector
            // allocates a slot for every candidate asked for.
            let oversample = i64::try_from(cfg.oversample.max(1)).unwrap_or(i64::MAX);
            let wanted =
                (k as i64).saturating_mul(oversample).saturating_add(st.dead_rows).min(st.live_rows + st.dead_rows);
            let cells = st.live_rows as u64 * dim as u64;
            if st.watermark == 0 || cells <= cfg.exact_max_cells {
                exact_scan(&tx, table, &blob, wanted, &mut best)?;
                if best.len() < k && (best.len() as i64) < st.live_rows {
                    // Dead rows or duplicate owners crowded the top-k: stream the whole table.
                    exact_stream(&tx, table, &blob, wanted.max(k as i64 * 4), &mut best)?;
                }
            } else {
                if cfg.preload && rc.preloaded[kind.index()] != st.generation {
                    tx.query_row("SELECT vector_quantize_preload(?1, 'v')", [table], |_| Ok(()))?;
                    rc.preloaded[kind.index()] = st.generation;
                }
                let mut sealed: Vec<i64> = Vec::with_capacity(wanted as usize);
                {
                    let mut stmt =
                        tx.prepare_cached(&format!("SELECT rowid FROM vector_quantize_scan('{table}', 'v', ?1, ?2)"))?;
                    let mut rows = stmt.query(params![blob, wanted])?;
                    while let Some(row) = rows.next()? {
                        sealed.push(row.get(0)?);
                    }
                }
                for batch in sealed.chunks(ID_BATCH) {
                    let sql = format!(
                        "SELECT owner, v FROM {table} WHERE live = 1 AND vid IN ({})",
                        placeholders(batch.len())
                    );
                    let mut stmt = tx.prepare_cached(&sql)?;
                    let mut rows = stmt.query(params_from_iter(batch))?;
                    while let Some(row) = rows.next()? {
                        let v: Vec<u8> = row.get(1)?;
                        best.keep(row.get(0)?, dot(query, &from_blob(&v)));
                    }
                }
                let mut stmt =
                    tx.prepare_cached(&format!("SELECT owner, v FROM {table} WHERE vid > ?1 AND live = 1"))?;
                let mut rows = stmt.query([st.watermark])?;
                while let Some(row) = rows.next()? {
                    let v: Vec<u8> = row.get(1)?;
                    best.keep(row.get(0)?, dot(query, &from_blob(&v)));
                }
            }
        }
    }
    tx.commit()?;
    let mut hits: Vec<Hit> = best.0.into_iter().map(|(owner, sim)| Hit { owner, score: sim as f64 }).collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.owner.cmp(&b.owner)));
    hits.truncate(k);
    Ok(hits)
}

/// Best similarity per owner (an owner can have several live vectors).
#[derive(Default)]
struct Best(HashMap<i64, f32>);

impl Best {
    fn keep(&mut self, owner: i64, sim: f32) {
        self.0.entry(owner).and_modify(|s| *s = s.max(sim)).or_insert(sim);
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn exact_scan(tx: &Transaction<'_>, table: &str, blob: &[u8], wanted: i64, best: &mut Best) -> Result<()> {
    let mut stmt = tx.prepare_cached(&format!(
        "SELECT t.owner, s.distance FROM vector_full_scan('{table}', 'v', ?1, ?2) AS s \
         JOIN {table} AS t ON t.vid = s.rowid WHERE t.live = 1"
    ))?;
    let mut rows = stmt.query(params![blob, wanted])?;
    while let Some(row) = rows.next()? {
        let distance: f64 = row.get(1)?;
        best.keep(row.get(0)?, (1.0 - distance) as f32);
    }
    Ok(())
}

fn exact_stream(tx: &Transaction<'_>, table: &str, blob: &[u8], limit: i64, best: &mut Best) -> Result<()> {
    let mut stmt = tx.prepare_cached(&format!(
        "SELECT t.owner, s.distance FROM vector_full_scan('{table}', 'v', ?1) AS s \
         JOIN {table} AS t ON t.vid = s.rowid WHERE t.live = 1 ORDER BY s.distance LIMIT ?2"
    ))?;
    let mut rows = stmt.query(params![blob, limit])?;
    while let Some(row) = rows.next()? {
        let distance: f64 = row.get(1)?;
        best.keep(row.get(0)?, (1.0 - distance) as f32);
    }
    Ok(())
}

fn placeholders(n: usize) -> String {
    let mut s = String::with_capacity(n * 2);
    for i in 0..n {
        if i > 0 {
            s.push(',');
        }
        s.push('?');
    }
    s
}

/// The generation a kind is currently at, for tests and diagnostics.
pub(crate) fn generation(conn: &Connection, kind: VectorKind) -> Result<i64> {
    Ok(conn
        .query_row("SELECT generation FROM vec_state WHERE tbl = ?1", [kind.table()], |r| r.get(0))
        .optional()?
        .unwrap_or(0))
}
