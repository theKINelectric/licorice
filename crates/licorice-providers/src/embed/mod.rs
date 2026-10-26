//! Text embedders (SPEC §10.1).

mod dashscope;
mod onnx_http;
mod wordllama;

pub use dashscope::DashScopeV4;
pub use onnx_http::LocalOnnx;
pub use wordllama::WordLlama;

/// Whether a text is a query or a stored document, for embedders with asymmetric
/// prefixes. WordLlama treats both alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Query,
    Document,
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("could not load the embedding model: {0}")]
    Load(String),
    #[error("could not embed: {0}")]
    Embed(String),
}

/// Turns texts into vectors of a fixed dimension.
#[async_trait::async_trait]
pub trait Embedder: Send + Sync {
    /// Model identity, recorded in the space's metadata.
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError>;
}
