//! The single writer: one connection on a dedicated thread, fed by a channel.
//!
//! Every write runs as one `BEGIN IMMEDIATE` transaction on this thread, so callers
//! never see `SQLITE_BUSY` from write contention. When the channel has been idle for
//! `idle_maintenance` and vectors changed since the last pass, the thread runs
//! re-quantization and compaction (SPEC §7.3). Under writes that never pause, it runs
//! them anyway once vectors have waited `max_maintenance_delay`.
//!
//! A panic in a write closure is contained: its transaction rolls back and the caller
//! gets an error, while the thread keeps serving every other handle.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::Instant;

use rusqlite::{Connection, TransactionBehavior};

use super::vector::{self, MaintenanceReport};
use super::{StoreConfig, WriteTx};
use crate::{Error, Result};

pub(crate) type Job = Box<dyn FnOnce(&mut WriterState) + Send>;

pub(crate) enum Msg {
    Job(Job),
}

pub(crate) struct WriterState {
    pub(crate) conn: Connection,
    pub(crate) cfg: Arc<StoreConfig>,
    /// Vectors changed since the last maintenance pass.
    pub(crate) dirty: bool,
}

impl WriterState {
    /// Runs `f` inside one immediate transaction; rolls back if `f` fails.
    pub(crate) fn write<T>(&mut self, f: impl FnOnce(&mut WriteTx<'_>) -> Result<T>) -> Result<T> {
        let tx = self.conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut w = WriteTx { conn: &tx, dim: self.cfg.embed_dim, touched_vectors: false };
        let out = f(&mut w)?;
        let touched = w.touched_vectors;
        tx.commit()?;
        if touched {
            self.dirty = true;
        }
        Ok(out)
    }

    pub(crate) fn maintain(&mut self) -> Result<MaintenanceReport> {
        let report = vector::maintain(&mut self.conn, &self.cfg.vector, self.cfg.embed_dim)?;
        self.dirty = false;
        Ok(report)
    }
}

/// Runs `f` on the writer state, turning a panic into an error. The panicking write's
/// transaction rolls back as the stack unwinds; anything still open is rolled back here.
pub(crate) fn contain<T>(state: &mut WriterState, f: impl FnOnce(&mut WriterState) -> Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(|| f(&mut *state))) {
        Ok(result) => result,
        Err(panic) => {
            if !state.conn.is_autocommit() {
                let _ = state.conn.execute_batch("ROLLBACK");
            }
            Err(Error::Task(format!("write panicked: {}", panic_message(&*panic))))
        }
    }
}

fn panic_message(panic: &(dyn Any + Send)) -> &str {
    panic.downcast_ref::<&str>().copied().or_else(|| panic.downcast_ref::<String>().map(String::as_str)).unwrap_or("?")
}

fn maintain(state: &mut WriterState) {
    if let Err(e) = contain(state, WriterState::maintain) {
        tracing::warn!(error = %e, "vector maintenance failed");
    }
}

/// The writer thread's loop. Returns when every sender is gone.
pub(crate) fn run(mut state: WriterState, rx: Receiver<Msg>) {
    let idle = state.cfg.idle_maintenance;
    // When vectors last went from maintained to changed.
    let mut dirty_since: Option<Instant> = None;
    loop {
        match rx.recv_timeout(idle) {
            Ok(Msg::Job(job)) => {
                job(&mut state);
                if state.dirty
                    && dirty_since.get_or_insert_with(Instant::now).elapsed() >= state.cfg.max_maintenance_delay
                {
                    maintain(&mut state);
                    dirty_since = Some(Instant::now()); // after a failure, wait again before retrying
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if state.dirty {
                    maintain(&mut state);
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if !state.dirty {
            dirty_since = None;
        }
    }
}
