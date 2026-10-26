//! Reader connection pool.
//!
//! Each connection records the quantization generation of every vector table at the
//! moment `vector_init` ran on it. sqlite-vector never reloads those parameters on an
//! existing connection, so a reader that falls behind a re-quantization is dropped and
//! replaced rather than reused (see `conn::init_vectors`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rusqlite::Connection;
use tokio::sync::Semaphore;

use super::conn::{self, Role};
use super::vector::VectorKind;
use crate::{Error, Result};

pub(crate) struct ReaderConn {
    pub(crate) conn: Connection,
    pub(crate) generations: [i64; VectorKind::COUNT],
    /// Generation whose quantization this connection has preloaded, per kind; -1 if none.
    pub(crate) preloaded: [i64; VectorKind::COUNT],
}

impl ReaderConn {
    fn open(path: &Path, dim: usize) -> Result<ReaderConn> {
        let conn = conn::open(path, Role::Reader)?;
        let generations = conn::init_vectors(&conn, dim)?;
        Ok(ReaderConn { conn, generations, preloaded: [-1; VectorKind::COUNT] })
    }
}

pub(crate) struct ReaderPool {
    path: PathBuf,
    dim: usize,
    idle: Mutex<Vec<ReaderConn>>,
    permits: Semaphore,
    replaced: AtomicU64,
}

/// Attempts per read before a stale-generation race is reported as an error.
const STALE_RETRIES: usize = 3;

impl ReaderPool {
    pub(crate) fn new(path: PathBuf, dim: usize, size: usize) -> ReaderPool {
        ReaderPool {
            path,
            dim,
            idle: Mutex::new(Vec::new()),
            permits: Semaphore::new(size.max(1)),
            replaced: AtomicU64::new(0),
        }
    }

    /// Runs `f` on a pooled connection on a blocking thread. If `f` reports the
    /// connection stale, it is dropped and `f` runs again on a freshly opened one: the
    /// other idle connections predate the same re-quantization, so they are no better.
    pub(crate) async fn run<T, F>(self: &Arc<Self>, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: Fn(&mut ReaderConn) -> Result<T> + Send + Sync + 'static,
    {
        let _permit = self.permits.acquire().await.map_err(|_| Error::Closed)?;
        let f = Arc::new(f);
        let mut fresh = false;
        for _ in 0..STALE_RETRIES {
            let pool = Arc::clone(self);
            let f = Arc::clone(&f);
            let pooled = if fresh { None } else { self.idle.lock().expect("reader pool poisoned").pop() };
            let (conn, result) = tokio::task::spawn_blocking(move || {
                let mut rc = match pooled {
                    Some(rc) => rc,
                    None => match ReaderConn::open(&pool.path, pool.dim) {
                        Ok(rc) => rc,
                        Err(e) => return (None, Err(e)),
                    },
                };
                let result = f(&mut rc);
                (Some(rc), result)
            })
            .await
            .map_err(|e| Error::Task(e.to_string()))?;
            match result {
                Err(Error::StaleReader) => {
                    // Drop the connection; a fresh one loads the current generation.
                    self.replaced.fetch_add(1, Ordering::Relaxed);
                    fresh = true;
                    continue;
                }
                other => {
                    if let Some(rc) = conn {
                        self.idle.lock().expect("reader pool poisoned").push(rc);
                    }
                    return other;
                }
            }
        }
        Err(Error::Task("reader stayed behind the quantization generation after retries".into()))
    }

    /// Connections replaced because they were behind a re-quantization.
    pub(crate) fn replaced(&self) -> u64 {
        self.replaced.load(Ordering::Relaxed)
    }
}
