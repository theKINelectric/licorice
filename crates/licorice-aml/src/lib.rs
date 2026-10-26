//! AML Textual contract adapter. The `raw-dialogue-v1` profile makes no LLM calls.
//! Source rows, FTS, vectors and the replay receipt share one native transaction.

mod model;

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use licorice::retrieve::{Query, RetrieveConfig, Retriever};
use licorice::rusqlite::{Connection, OptionalExtension, params};
use licorice::temporal::Day;
use licorice::{LexicalKind, Store, StoreConfig, VectorKind};
use licorice_providers::chat::ChatModel;
use licorice_providers::embed::{EmbedKind, Embedder};
use licorice_providers::rerank::Reranker;
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{Mutex, Semaphore};

const PROFILE: &str = "aml-textual/raw-dialogue-v1";
const BODY_LIMIT: usize = 1024 * 1024;
const STRIPES: usize = 64;

/// Kept separate from the evaluation harness and native retrieval defaults.
/// Debug is deliberately absent because this configuration contains a secret.
pub struct Config {
    pub data_dir: PathBuf,
    pub api_key: String,
    /// Entire compact JSON `data` array, including IDs, dates and escaping.
    /// Must be in 64..=2000. The platform's Answer prompt is outside this budget.
    pub budget_tokens: usize,
}

#[derive(Clone)]
pub struct Adapter(Arc<Shared>);
struct Shared {
    config: Config,
    embedder: Arc<dyn Embedder>,
    model: Option<Arc<dyn ChatModel>>,
    search_policy: Option<RetrieveConfig>,
    reranker: Option<Arc<dyn Reranker>>,
    locks: Vec<Mutex<()>>,
    capacity: Semaphore,
    // An exclusive rollback-journal transaction prevents two processes owning the
    // same directory. SQLite/OS releases it on crash; no stale lock-file recovery.
    _owner: StdMutex<Connection>,
}

#[derive(Clone, Deserialize, Serialize)]
struct AddRequest {
    request_id: String,
    messages: Vec<Message>,
    user_id: String,
    session_id: String,
}
#[derive(Clone, Deserialize, Serialize)]
struct Message {
    role: MessageRole,
    #[serde(default)]
    timestamp: Option<i64>,
    content: String,
}
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum MessageRole {
    User,
    Assistant,
    System,
    Participant,
    Tool,
}
impl MessageRole {
    fn name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
            Self::Participant => "participant",
            Self::Tool => "tool",
        }
    }
    fn stored(self) -> &'static str {
        if matches!(self, Self::Tool) { "participant" } else { self.name() }
    }
}
#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    user_id: String,
    top_k: usize,
    #[serde(default, rename = "options")]
    _options: Option<Vec<String>>,
}
#[derive(Serialize)]
struct Memory {
    id: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
}
#[derive(Serialize)]
struct SearchResponse {
    data: Vec<Memory>,
}

struct ApiError(StatusCode, &'static str);
impl ApiError {
    fn invalid(reason: &'static str) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, reason)
    }
    fn unavailable() -> Self {
        Self(StatusCode::SERVICE_UNAVAILABLE, "memory operation failed; retry the same request")
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"detail":{"reason":self.1}}))).into_response()
    }
}
type ApiResult<T> = Result<T, ApiError>;

impl Adapter {
    /// Open a dedicated data root. This profile never reads `.env` or calls a model.
    pub fn open(config: Config, embedder: Arc<dyn Embedder>) -> Result<Self, String> {
        Self::configured(config, embedder, None)
    }
    /// Isolated opt-in profile; cached phase outputs and a staged database make
    /// synchronous Add atomic across remote model calls and local applications.
    pub fn open_with_model(
        config: Config,
        embedder: Arc<dyn Embedder>,
        model: Arc<dyn ChatModel>,
    ) -> Result<Self, String> {
        Self::configured(config, embedder, Some(model))
    }
    fn configured(
        config: Config,
        embedder: Arc<dyn Embedder>,
        model: Option<Arc<dyn ChatModel>>,
    ) -> Result<Self, String> {
        if config.api_key.is_empty() || !config.api_key.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("AML API key must be nonempty visible ASCII without spaces".into());
        }
        if !(64..=2000).contains(&config.budget_tokens) || embedder.dim() == 0 {
            return Err("budget must be 64..=2000 tokens and embedding dimension must be positive".into());
        }
        std::fs::create_dir_all(&config.data_dir).map_err(|e| format!("create data root: {e}"))?;
        let owner = Connection::open(config.data_dir.join(".aml-owner.sqlite"))
            .map_err(|_| "could not open data-root ownership lock")?;
        owner
            .execute_batch("PRAGMA busy_timeout=0; BEGIN EXCLUSIVE;")
            .map_err(|_| "data root already owned by another adapter; use one process per root")?;
        Ok(Self(Arc::new(Shared {
            config,
            embedder,
            model,
            search_policy: None,
            reranker: None,
            locks: (0..STRIPES).map(|_| Mutex::new(())).collect(),
            capacity: Semaphore::new(8),
            _owner: StdMutex::new(owner),
        })))
    }

    /// Configure a fixed Search policy before cloning or serving the adapter.
    pub fn with_search_policy(
        mut self,
        config: RetrieveConfig,
        reranker: Option<Arc<dyn Reranker>>,
    ) -> Result<Self, String> {
        let shared = Arc::get_mut(&mut self.0).ok_or("configure Search before sharing the adapter")?;
        if config.budget_tokens != shared.config.budget_tokens {
            return Err("Search policy must use the configured memory budget".into());
        }
        if !(1..=96).contains(&config.rerank_candidates) {
            return Err("Search scorer pool must be in 1..=96".into());
        }
        shared.search_policy = Some(config);
        shared.reranker = reranker;
        Ok(self)
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/add", post(add))
            .route("/search", post(search))
            .route_layer(middleware::from_fn_with_state(self.clone(), authenticate))
            .route(
                "/health",
                get(|State(adapter): State<Adapter>| async move {
                    Json(json!({"status":"ok","profile":adapter.profile()}))
                }),
            )
            .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "unknown endpoint") })
            .method_not_allowed_fallback(|| async { ApiError(StatusCode::METHOD_NOT_ALLOWED, "unsupported method") })
            .layer(DefaultBodyLimit::max(BODY_LIMIT))
            .with_state(self.clone())
    }

    fn path(&self, user: &str) -> PathBuf {
        self.0.config.data_dir.join(format!("{}.sqlite3", blake3::hash(user.as_bytes()).to_hex()))
    }
    fn stripe(&self, user: &str) -> usize {
        blake3::hash(user.as_bytes()).as_bytes()[0] as usize % STRIPES
    }
    async fn store(&self, user: &str) -> ApiResult<Store> {
        self.store_at(self.path(user)).await
    }
    async fn store_at(&self, path: PathBuf) -> ApiResult<Store> {
        let mut cfg = StoreConfig::new(self.0.embedder.dim());
        cfg.readers = 2;
        let store = Store::open(path, cfg).await.map_err(|_| ApiError::unavailable())?;
        let identity = format!("{}:{}", self.profile(), self.0.embedder.id());
        let setup = store
            .write(move |w| {
                w.conn().execute_batch(
                    "CREATE TABLE IF NOT EXISTS aml_requests(
                request_id TEXT PRIMARY KEY, payload TEXT NOT NULL, added_at INTEGER NOT NULL);",
                )?;
                let existing: Option<String> = w
                    .conn()
                    .query_row("SELECT value FROM meta WHERE key='aml_profile'", [], |r| r.get(0))
                    .optional()?;
                match existing {
                    Some(value) if value != identity => {
                        return Err(licorice::Error::InvalidInput("adapter profile mismatch".into()));
                    }
                    None => {
                        w.conn().execute("INSERT INTO meta(key,value) VALUES('aml_profile',?1)", [&identity])?;
                    }
                    _ => {}
                }
                Ok(())
            })
            .await;
        if setup.is_err() {
            let _ = store.close().await;
            return Err(ApiError::unavailable());
        }
        Ok(store)
    }

    async fn add(&self, request: AddRequest) -> ApiResult<serde_json::Value> {
        check_id(&request.user_id)?;
        check_id(&request.session_id)?;
        check_id(&request.request_id)?;
        if request.messages.is_empty() || request.messages.len() > 20 {
            return Err(ApiError::invalid("messages must contain 1..=20 Textual messages"));
        }
        for message in &request.messages {
            if message.content.trim().is_empty() {
                return Err(ApiError::invalid("message content must be nonempty text"));
            }
            // Civil years 0001..9999; avoids native temporal arithmetic overflow.
            if message.timestamp.is_some_and(|ms| !(-62_135_596_800_000..=253_402_300_799_999).contains(&ms)) {
                return Err(ApiError::invalid("timestamp must be Unix milliseconds in civil years 0001..9999"));
            }
        }
        let _capacity = self.0.capacity.acquire().await.map_err(|_| ApiError::unavailable())?;
        let _user = self.0.locks[self.stripe(&request.user_id)].lock().await;
        if self.0.model.is_some() {
            return self.model_add(request).await;
        }
        let store = self.store(&request.user_id).await?;
        let result = self.append(&store, &request).await;
        let closed = store.close().await;
        result?;
        closed.map_err(|_| ApiError::unavailable())?;
        Ok(
            json!({"success":true,"request_id":request.request_id,"user_id":request.user_id,"session_id":request.session_id}),
        )
    }

    async fn append(&self, store: &Store, request: &AddRequest) -> ApiResult<()> {
        let payload = serde_json::to_string(request).map_err(|_| ApiError::unavailable())?;
        let request_id = request.request_id.clone();
        let stored: Option<String> = store
            .read(move |conn| {
                Ok(conn
                    .query_row("SELECT payload FROM aml_requests WHERE request_id=?1", [&request_id], |r| r.get(0))
                    .optional()?)
            })
            .await
            .map_err(|_| ApiError::unavailable())?;
        if let Some(old) = stored {
            return if old == payload {
                Ok(())
            } else {
                Err(ApiError(StatusCode::CONFLICT, "request_id already exists with different content or session"))
            };
        }
        let texts: Vec<String> = request.messages.iter().map(|m| format!("{}: {}", m.role.name(), m.content)).collect();
        let vectors = self.0.embedder.embed(&texts, EmbedKind::Document).await.map_err(|_| ApiError::unavailable())?;
        if vectors.len() != texts.len() {
            return Err(ApiError::unavailable());
        }
        let input = request.clone();
        let logical = if self.0.model.is_some() {
            let session_id = request.session_id.clone();
            let start:i64=store.read(move |c| Ok(c.query_row(
                "SELECT count(*) FROM turns t JOIN sessions s ON s.id=t.session_id WHERE json_extract(s.meta,'$.session_id')=?1",[&session_id],|r|r.get(0))?))
                .await.map_err(|_|ApiError::unavailable())?;
            Some((
                blake3::hash(serde_json::to_string(&(&request.user_id, &request.session_id)).unwrap().as_bytes())
                    .to_hex()[..16]
                    .to_string(),
                start,
            ))
        } else {
            None
        };
        let now = now_ms();
        let source_id =
            blake3::hash(&serde_json::to_vec(&(&request.user_id, &request.request_id)).unwrap()).to_hex().to_string();
        // All source/index mutations and the receipt roll back together, including
        // an invalid later vector. User serialization protects the preflight replay.
        store
            .write(move |w| {
                w.conn().execute(
                    "INSERT INTO sessions(ext_id,started_at,meta) VALUES(?1,?2,?3)",
                    params![
                        source_id,
                        input.messages[0].timestamp.unwrap_or(now),
                        json!({"session_id":input.session_id,"request_id":input.request_id}).to_string()
                    ],
                )?;
                let session = w.conn().last_insert_rowid();
                for (index, ((message, text), vector)) in input.messages.iter().zip(&texts).zip(&vectors).enumerate() {
                    let id = match &logical {
                        Some((session, start)) => format!("m:{session}:{}", start + index as i64),
                        None => format!("{source_id}:{index}"),
                    };
                    let ts = message.timestamp.unwrap_or(now);
                    w.conn().execute(
                        "INSERT INTO turns(session_id,seq,speaker,role,ts,text,ext_id) VALUES(?1,?2,?3,?4,?5,?6,?7)",
                        params![
                            session,
                            index as i64,
                            message.role.name(),
                            message.role.stored(),
                            ts,
                            message.content,
                            id
                        ],
                    )?;
                    let turn = w.conn().last_insert_rowid();
                    w.conn().execute(
                        "INSERT INTO chunks(session_id,first_turn,last_turn,ts,text,hash) VALUES(?1,?2,?2,?3,?4,?5)",
                        params![session, turn, ts, text, blake3::hash(id.as_bytes()).as_bytes().as_slice()],
                    )?;
                    let chunk = w.conn().last_insert_rowid();
                    w.index_text(LexicalKind::Chunk, chunk, text)?;
                    w.insert_vector(VectorKind::Chunk, chunk, vector)?;
                }
                w.conn().execute(
                    "INSERT INTO aml_requests(request_id,payload,added_at) VALUES(?1,?2,?3)",
                    params![input.request_id, payload, now],
                )?;
                Ok(())
            })
            .await
            .map_err(|_| ApiError::unavailable())
    }

    async fn search(&self, request: SearchRequest) -> ApiResult<SearchResponse> {
        check_id(&request.user_id)?;
        if request.query.trim().is_empty() || request.top_k > 1000 {
            return Err(ApiError::invalid("query must be nonempty text; top_k must be in 0..=1000"));
        }
        if request.top_k == 0 {
            return Ok(SearchResponse { data: vec![] });
        }
        let _capacity = self.0.capacity.acquire().await.map_err(|_| ApiError::unavailable())?;
        let _user = self.0.locks[self.stripe(&request.user_id)].lock().await;
        // An unseen user must not create a database or retrieve another user's space.
        if !self.path(&request.user_id).try_exists().map_err(|_| ApiError::unavailable())? {
            return Ok(SearchResponse { data: vec![] });
        }
        let store = self.store(&request.user_id).await?;
        let result = self.retrieve(&store, request).await;
        let closed = store.close().await;
        let response = result?;
        closed.map_err(|_| ApiError::unavailable())?;
        Ok(response)
    }

    async fn retrieve_context(
        &self,
        store: &Store,
        query: &str,
        at: i64,
        config: RetrieveConfig,
    ) -> ApiResult<licorice::retrieve::Retrieved> {
        let mut retriever = Retriever::new(store, self.0.embedder.as_ref(), config);
        if let Some(reranker) = &self.0.reranker {
            retriever = retriever.with_reranker(reranker.as_ref());
        }
        retriever.retrieve(&Query { text: query.into(), at }).await.map_err(|_| ApiError::unavailable())
    }

    async fn retrieve(&self, store: &Store, request: SearchRequest) -> ApiResult<SearchResponse> {
        let mut cfg = self.0.search_policy.clone().unwrap_or_else(|| RetrieveConfig {
            budget_tokens: self.0.config.budget_tokens,
            final_facts: if self.0.model.is_some() { 20 } else { 0 },
            summaries: if self.0.model.is_some() { 2 } else { 0 },
            chunks: 100,
            chunk_candidates: 100,
            ..RetrieveConfig::default()
        });
        if self.0.model.is_some() && cfg.query_speaker.is_none() {
            cfg.query_speaker = Some("User".into());
        }
        let at = now_ms();
        let mut retrieved = self.retrieve_context(store, &request.query, at, cfg.clone()).await?;
        if self.0.model.is_some() {
            let mut budget = self.0.config.budget_tokens;
            loop {
                if retrieved.facts.is_empty() && retrieved.chunks.is_empty() && retrieved.sessions.is_empty() {
                    return Ok(SearchResponse { data: vec![] });
                }
                let data = vec![Memory {
                    id: blake3::hash(retrieved.context.as_bytes()).to_hex().to_string(),
                    content: retrieved.context,
                    created_at: None,
                }];
                let count =
                    Cl100kCounter.count(&serde_json::to_string(&data).unwrap()).map_err(|_| ApiError::unavailable())?;
                if count <= self.0.config.budget_tokens {
                    return Ok(SearchResponse { data });
                }
                if budget < 96 {
                    return Ok(SearchResponse { data: vec![] });
                }
                budget = budget.saturating_sub(count - self.0.config.budget_tokens + 8);
                let mut policy = cfg.clone();
                policy.budget_tokens = budget;
                retrieved = self.retrieve_context(store, &request.query, at, policy).await?;
            }
        }
        // Batch exact timestamp lookups by chunk primary key. Looking up each
        // external turn ID would scan the entire turn table per returned record.
        let chunk_ids: Vec<i64> = retrieved.chunks.iter().map(|c| c.id).collect();
        let timestamps = store
            .read(move |conn| {
                let mut statement = conn.prepare("SELECT ts FROM chunks WHERE id=?1")?;
                chunk_ids
                    .iter()
                    .copied()
                    .map(|id| statement.query_row([id], |r| r.get::<_, i64>(0)).map(|ts| (id, ts)))
                    .collect::<Result<std::collections::HashMap<_, _>, _>>()
                    .map_err(Into::into)
            })
            .await
            .map_err(|_| ApiError::unavailable())?;
        let mut data = vec![];
        for chunk in retrieved.chunks {
            let ts = timestamps[&chunk.id];
            for (id, speaker, text) in chunk.turns {
                data.push(Memory {
                    content: format!("[{}] {speaker}: {text}", Day::from_ms(ts).iso()),
                    id,
                    created_at: Some(iso_timestamp(ts)),
                });
                let wire = serde_json::to_string(&data).map_err(|_| ApiError::unavailable())?;
                if Cl100kCounter.count(&wire).map_err(|_| ApiError::unavailable())? > self.0.config.budget_tokens {
                    data.pop(); // Never cut a source message; try a smaller later record.
                }
                if data.len() == request.top_k {
                    break;
                }
            }
            if data.len() == request.top_k {
                break;
            }
        }
        Ok(SearchResponse { data })
    }
}

async fn authenticate(State(adapter): State<Adapter>, request: Request, next: Next) -> Response {
    let headers = request.headers();
    let candidate = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|value| {
            value.split_once(' ').and_then(|(scheme, token)| {
                (scheme.eq_ignore_ascii_case("Bearer") || scheme.eq_ignore_ascii_case("Token")).then_some(token)
            })
        })
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    let valid = candidate.is_some_and(|token| {
        let a = blake3::hash(token.as_bytes());
        let b = blake3::hash(adapter.0.config.api_key.as_bytes());
        a.as_bytes().iter().zip(b.as_bytes()).fold(0_u8, |diff, (a, b)| diff | (a ^ b)) == 0
    });
    if valid {
        next.run(request).await
    } else {
        ApiError(StatusCode::UNAUTHORIZED, "invalid API authentication").into_response()
    }
}
async fn add(
    State(adapter): State<Adapter>,
    request: Result<Json<AddRequest>, JsonRejection>,
) -> ApiResult<Json<serde_json::Value>> {
    Ok(Json(adapter.add(json_body(request)?).await?))
}
async fn search(
    State(adapter): State<Adapter>,
    request: Result<Json<SearchRequest>, JsonRejection>,
) -> ApiResult<Json<SearchResponse>> {
    Ok(Json(adapter.search(json_body(request)?).await?))
}
fn json_body<T>(body: Result<Json<T>, JsonRejection>) -> ApiResult<T> {
    body.map(|Json(value)| value).map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError(StatusCode::PAYLOAD_TOO_LARGE, "request body exceeds 1 MiB")
        } else {
            ApiError::invalid("invalid Textual request schema or JSON")
        }
    })
}
fn check_id(value: &str) -> ApiResult<()> {
    if value.trim().is_empty() || value.len() > 4096 {
        Err(ApiError::invalid("identifiers must be nonempty and at most 4096 UTF-8 bytes"))
    } else {
        Ok(())
    }
}
fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock must follow Unix epoch").as_millis() as i64
}
fn iso_timestamp(ms: i64) -> String {
    let day = Day::from_ms(ms);
    let time = ms.rem_euclid(86_400_000);
    format!(
        "{}T{:02}:{:02}:{:02}.{:03}Z",
        day.iso(),
        time / 3_600_000,
        time / 60_000 % 60,
        time / 1000 % 60,
        time % 1000
    )
}
