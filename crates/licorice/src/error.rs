/// Errors returned by the store.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("application {id} was already committed with different source, configuration or output")]
    ApplicationConflict { id: String },

    #[error(
        "application {id} targets a session with legacy facts or an untracked summary; rebuild that session in a fresh space from source outputs"
    )]
    LegacyApplication { id: String },

    #[error("session {session} already exists with different source content")]
    SourceConflict { session: String },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("vector has {got} dimensions; this space uses {expected}")]
    Dimension { expected: usize, got: usize },

    #[error("space was created with embedding dimension {stored}; opened with {requested}")]
    DimensionMismatch { stored: usize, requested: usize },

    #[error("vector must be finite and non-zero")]
    InvalidVector,

    #[error("sqlite-vector failed to load: {0}")]
    Extension(String),

    #[error("sqlite-vector {found} is linked; {expected} is required")]
    ExtensionVersion { expected: &'static str, found: String },

    #[error("schema version {found} is newer than this build supports ({supported})")]
    SchemaTooNew { found: i64, supported: i64 },

    #[error("store is closed")]
    Closed,

    #[error("background task failed: {0}")]
    Task(String),

    /// A reader connection was loaded at an older quantization generation. Handled
    /// inside the reader pool (the connection is replaced); never returned to callers.
    #[doc(hidden)]
    #[error("reader connection is behind the quantization generation")]
    StaleReader,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
