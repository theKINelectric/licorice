//! The storage layer: one SQLite file per memory space (SPEC §5–§7).
//!
//! A [`Store`] owns a single writer connection on its own thread and a pool of reader
//! connections. Writes are closures run in one immediate transaction; reads run on
//! blocking threads. Vector and lexical search go through the [`Index`] trait, which is
//! the only engine-specific surface (see `docs/adr/0001-storage-engine.md`).

mod conn;
mod lexical;
mod migrate;
mod readers;
mod vector;
mod writer;

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use rusqlite::Connection;
use tokio::sync::oneshot;

pub use conn::load_vector_extension;
pub use lexical::LexicalKind;
pub(crate) use vector::now_ms;
pub use vector::{MaintenanceReport, VectorKind};

use crate::{Error, Result};
use readers::ReaderPool;
use writer::{Msg, WriterState};

/// Vector search and quantization settings (SPEC §7.2–§7.3, Appendix A).
#[derive(Clone, Debug)]
pub struct VectorConfig {
    /// Rows × dimensions at or below which searches are exact scans.
    pub exact_max_cells: u64,
    /// Candidates fetched from the quantized scan per requested result.
    pub oversample: usize,
    /// Re-quantize once this many rows sit above the watermark …
    pub requantize_min_delta: u64,
    /// … or this fraction of the sealed rows, whichever is larger.
    pub requantize_delta_frac: f64,
    /// Compact (and re-quantize) once dead rows exceed this fraction.
    pub requantize_dead_frac: f64,
    /// Preload quantized data on each reader connection. Costs about rows × dim bytes
    /// per reader per kind; off by default.
    pub preload: bool,
}

impl Default for VectorConfig {
    fn default() -> Self {
        VectorConfig {
            exact_max_cells: 8_000_000,
            oversample: 4,
            requantize_min_delta: 2000,
            requantize_delta_frac: 0.05,
            requantize_dead_frac: 0.10,
            preload: false,
        }
    }
}

/// Store settings.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    /// Embedding dimension, fixed per space.
    pub embed_dim: usize,
    /// Maximum concurrent reader connections.
    pub readers: usize,
    pub vector: VectorConfig,
    /// Writer idle time before it runs vector maintenance.
    pub idle_maintenance: Duration,
    /// Longest that changed vectors wait for maintenance when writes never leave the
    /// writer idle; it then runs between two writes.
    pub max_maintenance_delay: Duration,
}

impl StoreConfig {
    pub fn new(embed_dim: usize) -> StoreConfig {
        StoreConfig {
            embed_dim,
            readers: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            vector: VectorConfig::default(),
            idle_maintenance: Duration::from_millis(500),
            max_maintenance_delay: Duration::from_secs(30),
        }
    }
}

/// One search result: the owning row's id and its score (higher is better).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub owner: i64,
    pub score: f64,
}

/// Restricts a vector search.
#[derive(Clone, Debug, Default)]
pub enum Filter {
    #[default]
    None,
    /// Only these owners; scored exactly.
    Owners(Vec<i64>),
}

/// Vector and lexical search: the engine-specific part of the store.
pub trait Index {
    /// The `k` owners whose vectors are most similar (cosine) to `query`, best first.
    fn knn(
        &self,
        kind: VectorKind,
        query: &[f32],
        k: usize,
        filter: Filter,
    ) -> impl Future<Output = Result<Vec<Hit>>> + Send;

    /// The `k` best BM25 matches for free text, best first.
    fn lexical(&self, kind: LexicalKind, query: &str, k: usize) -> impl Future<Output = Result<Vec<Hit>>> + Send;
}

/// A write transaction, handed to [`Store::write`] closures.
pub struct WriteTx<'a> {
    conn: &'a Connection,
    dim: usize,
    touched_vectors: bool,
}

impl WriteTx<'_> {
    /// The transaction's connection, for any SQL.
    pub fn conn(&self) -> &Connection {
        self.conn
    }

    /// Appends a vector for `owner` (normalized here); returns its vid.
    pub fn insert_vector(&mut self, kind: VectorKind, owner: i64, v: &[f32]) -> Result<i64> {
        let vid = vector::insert(self.conn, kind, owner, v, self.dim)?;
        self.touched_vectors = true;
        Ok(vid)
    }

    /// Retires the owner's live vectors; returns how many.
    pub fn retire_vectors(&mut self, kind: VectorKind, owner: i64) -> Result<usize> {
        let n = vector::retire(self.conn, kind, owner)?;
        if n > 0 {
            self.touched_vectors = true;
        }
        Ok(n)
    }

    /// Retires the owner's live vectors and appends `v`; returns the new vid.
    pub fn replace_vector(&mut self, kind: VectorKind, owner: i64, v: &[f32]) -> Result<i64> {
        self.retire_vectors(kind, owner)?;
        self.insert_vector(kind, owner, v)
    }

    /// Adds `text` to the lexical index under the source row's id
    /// (for [`LexicalKind::Alias`], the `entity_aliases` rowid).
    pub fn index_text(&mut self, kind: LexicalKind, rowid: i64, text: &str) -> Result<()> {
        lexical::index(self.conn, kind, rowid, text)
    }

    /// Removes a lexical entry; `text` must be exactly what was indexed.
    pub fn unindex_text(&mut self, kind: LexicalKind, rowid: i64, text: &str) -> Result<()> {
        lexical::unindex(self.conn, kind, rowid, text)
    }

    /// Records `alias` for an entity and indexes it; returns false if it already existed.
    pub fn add_alias(&mut self, entity_id: i64, alias: &str) -> Result<bool> {
        let added = self.conn.execute(
            "INSERT OR IGNORE INTO entity_aliases(alias, entity_id) VALUES (?1, ?2)",
            rusqlite::params![alias, entity_id],
        )? > 0;
        if added {
            let rowid = self.conn.last_insert_rowid();
            lexical::index(self.conn, LexicalKind::Alias, rowid, alias)?;
        }
        Ok(added)
    }
}

struct Inner {
    cfg: Arc<StoreConfig>,
    writer: Mutex<Option<Sender<Msg>>>,
    writer_thread: Mutex<Option<JoinHandle<()>>>,
    readers: Arc<ReaderPool>,
}

impl Inner {
    fn shutdown(&self) {
        drop(self.writer.lock().expect("writer handle poisoned").take());
        if let Some(handle) = self.writer_thread.lock().expect("writer thread poisoned").take() {
            // A write closure can hold the last handle, so this may run on the writer
            // thread itself. Joining would deadlock; its loop ends after this job.
            if handle.thread().id() != std::thread::current().id() {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// A memory space: one SQLite file.
#[derive(Clone)]
pub struct Store {
    inner: Arc<Inner>,
}

impl Store {
    /// Opens (creating if needed) the space at `path` and applies migrations.
    pub async fn open(path: impl AsRef<Path>, cfg: StoreConfig) -> Result<Store> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let cfg = Arc::new(cfg);
        let (writer_conn, path) = {
            let cfg = Arc::clone(&cfg);
            tokio::task::spawn_blocking(move || -> Result<(Connection, PathBuf)> {
                let mut conn = conn::open(&path, conn::Role::Writer)?;
                migrate::migrate(&mut conn)?;
                migrate::check_dimension(&conn, cfg.embed_dim)?;
                conn::init_vectors(&conn, cfg.embed_dim)?;
                Ok((conn, path))
            })
            .await
            .map_err(|e| Error::Task(e.to_string()))??
        };
        let (tx, rx) = mpsc::channel();
        let state = WriterState { conn: writer_conn, cfg: Arc::clone(&cfg), dirty: true };
        let handle = std::thread::Builder::new()
            .name("licorice-writer".into())
            .spawn(move || writer::run(state, rx))
            .map_err(|e| Error::Task(e.to_string()))?;
        let readers = Arc::new(ReaderPool::new(path, cfg.embed_dim, cfg.readers));
        Ok(Store {
            inner: Arc::new(Inner {
                cfg,
                writer: Mutex::new(Some(tx)),
                writer_thread: Mutex::new(Some(handle)),
                readers,
            }),
        })
    }

    pub fn config(&self) -> &StoreConfig {
        &self.inner.cfg
    }

    /// Runs `f` in one immediate transaction on the writer thread. If `f` fails, the
    /// transaction rolls back and the error is returned.
    pub async fn write<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut WriteTx<'_>) -> Result<T> + Send + 'static,
    {
        self.on_writer(move |state| state.write(f)).await
    }

    /// Runs vector maintenance now instead of waiting for the writer to go idle.
    pub async fn maintain(&self) -> Result<MaintenanceReport> {
        self.on_writer(|state| state.maintain()).await
    }

    async fn on_writer<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut WriterState) -> Result<T> + Send + 'static,
    {
        let (reply, rx) = oneshot::channel();
        let job: writer::Job = Box::new(move |state| {
            let _ = reply.send(writer::contain(state, f));
        });
        {
            let guard = self.inner.writer.lock().expect("writer handle poisoned");
            let sender = guard.as_ref().ok_or(Error::Closed)?;
            sender.send(Msg::Job(job)).map_err(|_| Error::Closed)?;
        }
        rx.await.map_err(|_| Error::Closed)?
    }

    /// Runs a read-only closure on a pooled reader connection.
    pub async fn read<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: Fn(&Connection) -> Result<T> + Send + Sync + 'static,
    {
        self.inner.readers.run(move |rc| f(&rc.conn)).await
    }

    /// The current quantization generation of a vector kind (0 = never quantized).
    pub async fn quantization_generation(&self, kind: VectorKind) -> Result<i64> {
        self.read(move |conn| vector::generation(conn, kind)).await
    }

    /// Reader connections replaced so far because they were behind a re-quantization.
    pub fn reader_replacements(&self) -> u64 {
        self.inner.readers.replaced()
    }

    /// Closes the store, waiting for queued writes to finish.
    pub async fn close(self) -> Result<()> {
        let inner = Arc::clone(&self.inner);
        drop(self);
        tokio::task::spawn_blocking(move || inner.shutdown()).await.map_err(|e| Error::Task(e.to_string()))
    }
}

impl Index for Store {
    async fn knn(&self, kind: VectorKind, query: &[f32], k: usize, filter: Filter) -> Result<Vec<Hit>> {
        let dim = self.inner.cfg.embed_dim;
        let query = vector::normalize(query, dim)?;
        let cfg = self.inner.cfg.vector.clone();
        self.inner.readers.run(move |rc| vector::knn(rc, &cfg, dim, kind, &query, k, &filter)).await
    }

    async fn lexical(&self, kind: LexicalKind, query: &str, k: usize) -> Result<Vec<Hit>> {
        let query = query.to_owned();
        self.inner.readers.run(move |rc| lexical::search(&rc.conn, kind, &query, k)).await
    }
}
