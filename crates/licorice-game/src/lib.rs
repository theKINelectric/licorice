//! Engine-independent character continuity. The host supplies perception and owns
//! world truth, permissions, atomic game saves, and scheduling outside frame work.
//!
//! Native BEING projections are disposable caches. [`Checkpoint`] is the complete
//! game-owned journal; restoring it never opens a character's abandoned future.
mod checkpoint;
pub mod hostile_hospitality;
mod runtime;

pub use checkpoint::{Checkpoint, Timeline};
pub use licorice::being;
pub use runtime::GameMemory;
use serde::{Deserialize, Serialize};

pub const MAX_CHARACTERS: usize = 16;
pub const MAX_OPERATIONS: usize = 4096;
pub const MAX_SAVE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_MEMORY_TOKENS: usize = 2000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("game memory: {0}")]
    Invalid(String),
    #[error(transparent)]
    Native(#[from] licorice::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn invalid(message: &str) -> Error {
    Error::Invalid(message.into())
}
pub(crate) fn identifier(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 96
        || !s.as_bytes()[0].is_ascii_alphanumeric()
        || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
    {
        return Err(invalid("IDs require 1..96 ASCII identifier bytes"));
    }
    Ok(())
}

/// What the game host says this character could perceive. A report attests that
/// someone said something; its contents are not promoted to world truth.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Channel {
    Witnessed,
    Report { speaker: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Perception {
    pub event_id: String,
    /// Simulation tick, deliberately not a Unix timestamp.
    pub tick: u64,
    pub place: String,
    pub subject: String,
    pub text: String,
    pub channel: Channel,
    /// An engine-defined semantic cue; observations do not adopt a method.
    pub cue: Option<String>,
    pub relationship_id: Option<String>,
    /// Only the trusted host can assert an actual matching commitment outcome.
    /// Reports cannot carry this proof.
    pub fulfills: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Action {
    Perceive { character: String, perception: Perception },
    Deliberate { character: String, request_id: String, expected_revision: i64, operation: being::Operation },
}
impl Action {
    pub(crate) fn character(&self) -> &str {
        match self {
            Self::Perceive { character, .. } | Self::Deliberate { character, .. } => character,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Entry {
    pub timeline: String,
    pub tick: u64,
    pub action: Action,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub source_id: Option<String>,
    pub revision: i64,
    pub replayed: bool,
}

/// Callback validity is local to this live runtime as well as the saved timeline.
/// It is a freshness token, not authentication or tool authorization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionStamp {
    pub world: String,
    pub timeline: String,
    pub runtime: String,
    pub character: String,
    pub revision: i64,
    pub policy_revision: u64,
    pub based_on_tick: u64,
    pub expires_at_tick: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Recall {
    /// Inject this text only. Metadata is not an additional hidden memory block.
    pub text: String,
    pub tokens: usize,
    pub budget_tokens: usize,
    pub tokenizer: String,
    pub source_ids: Vec<String>,
    pub episode_candidates_saturated: bool,
    pub stamp: DecisionStamp,
}
