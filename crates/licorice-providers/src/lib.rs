//! Model providers for Licorice.
//!
//! [`judge`] speaks the System One wire format (typed `noul` / `choice` / `score`
//! questions about one `state`) used by TypeSafe's Jev and by Coffee Cat (SPEC §10.2).
//! [`embed`] holds text embedders and [`chat`] the model request types (SPEC §10.1).

pub mod chat;
pub mod chat_http;
pub mod embed;
pub mod judge;
pub mod rerank;
pub mod tokens;
