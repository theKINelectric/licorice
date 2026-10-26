//! Licorice: long-term conversational memory with a CogniGraph-style hierarchy
//! (session summaries → entities → facts → raw turns) on SQLite, sqlite-vector and FTS5.
//!
//! Modules: [`store`] (schema, the single-writer / reader-pool connection model, vector
//! kNN with a quantization watermark, lexical search), [`ingest`] (turns, chunks, facts,
//! entities, summaries), [`temporal`] (deterministic date resolution), [`retrieve`]
//! (candidates, ranking, context assembly) and [`answer`] (the answering request).

pub mod answer;
pub mod assess;
pub mod being;
mod error;
pub mod ingest;
pub mod retrieve;
pub mod store;
pub mod temporal;

pub use error::{Error, Result};
/// Re-exported because [`WriteTx::conn`] and [`Store::read`] hand out its types.
pub use rusqlite;
pub use store::{
    Filter, Hit, Index, LexicalKind, MaintenanceReport, Store, StoreConfig, VectorConfig, VectorKind, WriteTx,
};
