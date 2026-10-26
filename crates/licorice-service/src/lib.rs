//! A deliberately bounded, anonymous trial of the native attributed-memory kernel.
//! A capability cookie owns one store; request JSON never chooses actors or paths.
mod accounts;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use licorice::{
    Store, StoreConfig,
    being::{self, Actor, ActorRole, Memory, Observation, Operation, Origin, Source},
    rusqlite::{Connection, OptionalExtension, params},
};
use licorice_providers::{embed::Embedder, tokens::Cl100kCounter};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Mutex, Semaphore};

const COOKIE: &str = "licorice_trial";
const SECURE_COOKIE: &str = "__Host-licorice_trial";
const AGENT: &str = "trial-being";
const MESSAGE_BYTES: usize = 4000;
const BUDGET: usize = 2000;
const STRIPES: usize = 64;

/// No Debug implementation: configuration may contain provider objects elsewhere.
pub struct Config {
    pub data_dir: PathBuf,
    pub public_origin: String,
    pub secure_cookie: bool,
    pub retention: Duration,
    pub accounts_enabled: bool,
    pub max_trials: usize,
    pub max_trial_creations: usize,
    pub max_observations: usize,
    pub max_concurrent: usize,
    /// No external provider calls; optionally an operator-owned private embedding sidecar.
    pub embedder: Option<Arc<dyn Embedder>>,
    pub semantic: being::context::SemanticContinuityConfig,
    /// Representation window only; complete admitted source text remains intact.
    pub representation_token_limit: Option<usize>,
}
impl Config {
    pub fn new(data_dir: impl Into<PathBuf>, public_origin: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            public_origin: public_origin.into(),
            secure_cookie: true,
            retention: Duration::from_secs(24 * 3600),
            accounts_enabled: false,
            max_trials: 100,
            max_trial_creations: 1000,
            max_observations: 40,
            max_concurrent: 8,
            embedder: None,
            representation_token_limit: None,
            semantic: being::context::SemanticContinuityConfig::default(),
        }
    }
}
#[derive(Clone)]
pub struct Service(Arc<Shared>);
struct Shared {
    config: Config,
    accounts: Option<accounts::Accounts>,
    registry: StdMutex<Connection>,
    _owner: StdMutex<Connection>,
    locks: Vec<Mutex<()>>,
    maintenance: Mutex<()>,
    capacity: Semaphore,
}

#[derive(Debug)]
pub struct ApiError(StatusCode, &'static str, &'static str);
impl ApiError {
    fn unavailable() -> Self {
        Self(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            "Memory operation unavailable; inspect state before starting a new request.",
        )
    }
    fn invalid(message: &'static str) -> Self {
        Self(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", message)
    }
    fn quota() -> Self {
        Self(StatusCode::TOO_MANY_REQUESTS, "trial_limit", "This workspace has reached its current limit. Export or delete it.")
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error":{"code":self.1,"message":self.2}}))).into_response()
    }
}
type Result<T> = std::result::Result<T, ApiError>;
fn native<T>(r: licorice::Result<T>) -> Result<T> {
    r.map_err(|_| ApiError::unavailable())
}
fn sql<T>(r: std::result::Result<T, licorice::rusqlite::Error>) -> Result<T> {
    r.map_err(|_| ApiError::unavailable())
}
fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}
fn random_id() -> std::io::Result<String> {
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|v| format!("{v:02x}")).collect())
}
fn id() -> Result<String> {
    random_id().map_err(|_| ApiError::unavailable())
}
fn private_dir(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

impl Service {
    pub fn open(config: Config) -> std::result::Result<Self, String> {
        let uri: axum::http::Uri = config.public_origin.parse().map_err(|_| "invalid public origin")?;
        let host = uri.host().ok_or("origin needs a host")?;
        if uri.path_and_query().is_some_and(|p| p.as_str() != "/")
            || uri.authority().is_none()
            || host.contains('@')
            || !matches!(uri.scheme_str(), Some("http" | "https"))
            || config.public_origin.ends_with('/')
        {
            return Err("origin must be an absolute HTTP(S) origin without path or trailing slash".into());
        }
        if !config.secure_cookie
            && !(uri.scheme_str() == Some("http") && matches!(host, "localhost" | "127.0.0.1" | "[::1]"))
        {
            return Err("insecure cookies are allowed only for local HTTP development".into());
        }
        if config.retention.is_zero()
            || config.retention > Duration::from_secs(24 * 3600)
            || config.max_trials == 0
            || config.max_trials > 1000
            || config.max_trial_creations == 0
            || config.max_trial_creations > 10000
            || config.max_concurrent == 0
            || config.max_concurrent > 32
            || config.max_observations == 0
            || config.max_observations > 40
        {
            return Err("trial limits exceed the qualified bounded profile".into());
        }
        if config.representation_token_limit.is_some_and(|n| n == 0 || config.embedder.is_none())
            || config.embedder.as_ref().is_some_and(|e| e.dim() == 0)
            || !config.semantic.minimum_cosine.is_finite()
            || !(0.0 < config.semantic.minimum_cosine && config.semantic.minimum_cosine <= 1.0)
            || !(1..=128).contains(&config.semantic.maximum_observations)
        {
            return Err("invalid semantic retrieval policy".into());
        }
        private_dir(&config.data_dir).map_err(|e| format!("private data directory: {e}"))?;
        private_dir(&config.data_dir.join("trials")).map_err(|e| format!("private trial directory: {e}"))?;
        let owner = Connection::open(config.data_dir.join(".owner.sqlite")).map_err(|_| "owner lock failed")?;
        owner
            .execute_batch("PRAGMA busy_timeout=0; BEGIN EXCLUSIVE;")
            .map_err(|_| "data directory already owned by another process")?;
        let registry = Connection::open(config.data_dir.join("registry.sqlite")).map_err(|_| "registry failed")?;
        registry.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS trials(token_hash TEXT PRIMARY KEY,expires_at INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS counts(name TEXT PRIMARY KEY,value INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS api_keys(key_hash TEXT PRIMARY KEY,workspace_hash TEXT UNIQUE NOT NULL); INSERT OR IGNORE INTO counts VALUES('created',0);").map_err(|_|"registry initialization failed")?;
        let capacity = Semaphore::new(config.max_concurrent);
        let accounts = if config.accounts_enabled { Some(accounts::Accounts::open(&config)?) } else { None };
        let service = Self(Arc::new(Shared {
            config,
            accounts,
            registry: StdMutex::new(registry),
            _owner: StdMutex::new(owner),
            locks: (0..STRIPES).map(|_| Mutex::new(())).collect(),
            maintenance: Mutex::new(()),
            capacity,
        }));
        service.cleanup_orphans().map_err(|_| "orphan cleanup failed")?;
        Ok(service)
    }
    pub fn router(&self) -> Router {
        Router::new()
            .merge(accounts::router())
            .route("/api/v1/workspaces", post(create_trial))
            .route("/api/v1/workspace", get(trial_state).delete(delete_trial))
            .route("/api/v1/sessions", post(conversation))
            .route("/api/v1/history", get(history))
            .route("/api/v1/keys", post(rotate_key))
            .route("/api/v1/trials", post(create_trial))
            .route("/api/v1/trial", get(trial_state).delete(delete_trial))
            .route("/api/v1/observations", post(observation))
            .route("/api/v1/conversations", post(conversation))
            .route("/api/v1/recall", post(recall))
            .route("/api/v1/export", get(export))
            .route("/health", get(|State(service):State<Service>| async move { Json(json!({"status":"ok","service":"licorice-trial","version":"v1","retrieval":service.retrieval_profile(),"accounts_enabled":service.0.config.accounts_enabled})) }))
            .route("/", get(|| async { Html(include_str!("../../../integrations/hosted-trial/index.html")) }))
            .route(
                "/app.js",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                        include_str!("../../../integrations/hosted-trial/app.js"),
                    )
                }),
            )
            .route(
                "/styles.css",
                get(|| async {
                    (
                        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                        include_str!("../../../integrations/hosted-trial/styles.css"),
                    )
                }),
            )
            .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "not_found", "Unknown endpoint.") })
            .layer(DefaultBodyLimit::max(8192))
            .layer(middleware::from_fn_with_state(self.clone(), boundary))
            .with_state(self.clone())
    }
    fn path(&self, hash: &str) -> PathBuf {
        self.0.config.data_dir.join("trials").join(hash)
    }
    fn stripe(&self, hash: &str) -> usize {
        hash.as_bytes()[0] as usize % STRIPES
    }
    fn token(&self, headers: &HeaderMap) -> Result<String> {
        let bearer =
            headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer "));
        let raw = bearer.or_else(|| {
            headers.get(header::COOKIE).and_then(|v| v.to_str().ok()).and_then(|s| {
                s.split(';').map(str::trim).find_map(|v| v.strip_prefix(&format!("{}=", self.cookie_name())))
            })
        });
        let raw = raw.filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())).ok_or(ApiError(
            StatusCode::UNAUTHORIZED,
            "no_trial",
            "Create or reopen a workspace in this browser.",
        ))?;
        let hash = blake3::hash(raw.as_bytes()).to_hex().to_string();
        if bearer.is_none() {
            return Ok(hash);
        }
        let conn = self.0.registry.lock().map_err(|_| ApiError::unavailable())?;
        sql(conn.query_row("SELECT workspace_hash FROM api_keys WHERE key_hash=?1", [hash], |r| r.get(0)).optional())?
            .ok_or(ApiError(StatusCode::UNAUTHORIZED, "invalid_api_key", "This API key is invalid or revoked."))
    }
    fn cookie_name(&self) -> &'static str {
        if self.0.config.secure_cookie { SECURE_COOKIE } else { COOKIE }
    }
    fn authenticate(&self, hash: &str) -> Result<i64> {
        let conn = self.0.registry.lock().map_err(|_| ApiError::unavailable())?;
        let expiry = sql(conn
            .query_row("SELECT expires_at FROM trials WHERE token_hash=?1", [hash], |r| r.get::<_, i64>(0))
            .optional())?
        .ok_or(ApiError(StatusCode::UNAUTHORIZED, "no_trial", "This trial no longer exists."))?;
        if expiry <= now_ms() {
            return Err(ApiError(StatusCode::GONE, "trial_expired", "This trial expired. Create a new trial."));
        }
        Ok(expiry)
    }
    fn authenticate_request(&self, hash: &str, headers: &HeaderMap) -> Result<i64> {
        // Credential mapping must be read again while the workspace stripe is held.
        // A queued request cannot survive API-key rotation merely by having resolved its workspace earlier.
        if self.token(headers)? != hash {
            return Err(ApiError(StatusCode::UNAUTHORIZED, "invalid_api_key", "This API key is invalid or revoked."));
        }
        self.authenticate(hash)
    }
    fn cleanup_orphans(&self) -> Result<usize> {
        let mut removed = 0;
        for entry in std::fs::read_dir(self.0.config.data_dir.join("trials")).map_err(|_| ApiError::unavailable())? {
            let entry = entry.map_err(|_| ApiError::unavailable())?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.len() != 64
                || !name.bytes().all(|b| b.is_ascii_digit() || b"abcdef".contains(&b))
                || !entry.file_type().map_err(|_| ApiError::unavailable())?.is_dir()
            {
                continue;
            }
            let exists = {
                let c = self.0.registry.lock().map_err(|_| ApiError::unavailable())?;
                sql(c.query_row("SELECT EXISTS(SELECT 1 FROM trials WHERE token_hash=?1)", [name], |r| {
                    r.get::<_, bool>(0)
                }))?
            };
            if !exists {
                std::fs::remove_dir_all(entry.path()).map_err(|_| ApiError::unavailable())?;
                removed += 1;
            }
        }
        Ok(removed)
    }
    /// Called by the server's periodic maintenance loop and before trial creation.
    pub async fn cleanup_durable(&self) -> Result<usize> { accounts::cleanup(self).await }
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let _maintenance = self.0.maintenance.lock().await;
        self.cleanup_orphans()?;
        let expired = {
            let conn = self.0.registry.lock().map_err(|_| ApiError::unavailable())?;
            let mut stmt = sql(conn.prepare("SELECT token_hash FROM trials WHERE expires_at<=?1"))?;
            sql(stmt.query_map([now_ms()], |r| r.get::<_, String>(0)))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| ApiError::unavailable())?
        };
        for hash in &expired {
            let _guard = self.0.locks[self.stripe(hash)].lock().await;
            self.erase(hash)?;
        }
        Ok(expired.len())
    }
    fn erase(&self, hash: &str) -> Result<()> {
        let path = self.path(hash);
        if path.exists() {
            std::fs::remove_dir_all(path).map_err(|_| ApiError::unavailable())?;
        }
        let mut conn = self.0.registry.lock().map_err(|_| ApiError::unavailable())?;
        let tx = sql(conn.transaction())?;
        sql(tx.execute("DELETE FROM api_keys WHERE workspace_hash=?1", [hash]))?;
        sql(tx.execute("DELETE FROM trials WHERE token_hash=?1", [hash]))?;
        sql(tx.commit())?;
        Ok(())
    }
    async fn store(&self, hash: &str) -> Result<(Store, Memory)> {
        let mut config = StoreConfig::new(1);
        config.readers = 1;
        let store = native(Store::open(self.path(hash).join("memory.sqlite"), config).await)?;
        let memory = native(Memory::attach(&store, AGENT).await)?;
        Ok((store, memory))
    }
    fn limits(&self) -> Value {
        json!({"message_bytes":MESSAGE_BYTES,"observations":self.0.config.max_observations,"context_tokens":BUDGET,"operations":120})
    }
    fn retrieval_profile(&self) -> Value {
        match &self.0.config.embedder {
            Some(model) => {
                json!({"mode":"hybrid_semantic","model":model.id(),"dimensions":model.dim(),"minimum_cosine":self.0.config.semantic.minimum_cosine,"source_kind":"explicit_attributed_reports","automatic_extraction":false,"enumeration_policy":"same_cosine_eligibility_as_other_queries","representation_token_limit":self.0.config.representation_token_limit,"whole_source_evidence":true,"evidence_semantics":"candidate_reports_not_answer_certification"})
            }
            None => json!({"mode":"lexical","source_kind":"explicit_attributed_reports","automatic_extraction":false}),
        }
    }
    fn details(&self, expiry: i64, conversation: &str) -> Value {
        json!({"expires_at":expiry,"conversation_id":conversation,"limits":self.limits(),"retention_hours":self.0.config.retention.as_secs_f64()/3600.0,"retrieval":self.retrieval_profile()})
    }
}

async fn boundary(State(service): State<Service>, request: Request, next: Next) -> Response {
    let mutation = !matches!(*request.method(), Method::GET | Method::HEAD | Method::OPTIONS);
    let bearer = request.headers().contains_key(header::AUTHORIZATION) && (service.token(request.headers()).is_ok() || accounts::valid_bearer(&service, request.headers()));
    if mutation
        && !(bearer && !request.headers().contains_key(header::ORIGIN))
        && request.headers().get(header::ORIGIN).and_then(|h| h.to_str().ok())
            != Some(service.0.config.public_origin.as_str())
    {
        return ApiError(StatusCode::FORBIDDEN, "origin_rejected", "Use the configured same-origin application.")
            .into_response();
    }
    if request.headers().get("sec-fetch-site").and_then(|h| h.to_str().ok()) == Some("cross-site") {
        return ApiError(StatusCode::FORBIDDEN, "origin_rejected", "Cross-site requests are not accepted.")
            .into_response();
    }
    let _permit = match service.0.capacity.try_acquire() {
        Ok(v) => v,
        Err(_) => {
            return ApiError(StatusCode::TOO_MANY_REQUESTS, "busy", "The trial service is busy; try again shortly.")
                .into_response();
        }
    };
    let (parts, body) = request.into_parts();
    let bytes = match tokio::time::timeout(Duration::from_secs(5), axum::body::to_bytes(body, 8192)).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(_)) => {
            return ApiError(StatusCode::PAYLOAD_TOO_LARGE, "body_limit", "Request body exceeds 8192 bytes.")
                .into_response();
        }
        Err(_) => {
            return ApiError(
                StatusCode::REQUEST_TIMEOUT,
                "body_timeout",
                "Request body was not received within five seconds.",
            )
            .into_response();
        }
    };
    let request = Request::from_parts(parts, axum::body::Body::from(bytes));
    let mut response = match tokio::time::timeout(Duration::from_secs(45), next.run(request)).await {
        Ok(v) => v,
        Err(_) => ApiError(
            StatusCode::GATEWAY_TIMEOUT,
            "request_timeout",
            "Request timed out. Inspect state before starting a new operation.",
        )
        .into_response(),
    };
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert("x-content-type-options", HeaderValue::from_static("nosniff"));
    headers.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
    headers.insert("content-security-policy",HeaderValue::from_static("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"));
    response
}

async fn create_trial(State(service): State<Service>, headers: HeaderMap) -> Result<Response> {
    // Repeated create from a browser with a live capability resumes its trial.
    if let Ok(hash) = service.token(&headers) {
        let _guard = service.0.locks[service.stripe(&hash)].lock().await;
        if let Ok(expiry) = service.authenticate_request(&hash, &headers) {
            let (store, _) = service.store(&hash).await?;
            let conversation = current_conversation(&store).await?;
            native(store.close().await)?;
            return Ok(Json(service.details(expiry, &conversation)).into_response());
        }
    }
    service.cleanup_expired().await?;
    let _maintenance = service.0.maintenance.lock().await;
    {
        let conn = service.0.registry.lock().map_err(|_| ApiError::unavailable())?;
        let active: i64 = sql(conn.query_row("SELECT count(*) FROM trials", [], |r| r.get(0)))?;
        let created: i64 = sql(conn.query_row("SELECT value FROM counts WHERE name='created'", [], |r| r.get(0)))?;
        if active >= service.0.config.max_trials as i64 || created >= service.0.config.max_trial_creations as i64 {
            return Err(ApiError::quota());
        }
    }
    let token = id()?;
    let api_key = id()?;
    let key_hash = blake3::hash(api_key.as_bytes()).to_hex().to_string();
    let hash = blake3::hash(token.as_bytes()).to_hex().to_string();
    let conversation = id()?;
    let expiry = now_ms() + service.0.config.retention.as_millis() as i64;
    {
        let mut conn = service.0.registry.lock().map_err(|_| ApiError::unavailable())?;
        let tx = sql(conn.transaction())?;
        sql(tx.execute("INSERT INTO trials VALUES(?1,?2)", params![hash, expiry]))?;
        sql(tx.execute("INSERT INTO api_keys VALUES(?1,?2)", params![key_hash, hash]))?;
        sql(tx.execute("UPDATE counts SET value=value+1 WHERE name='created'", []))?;
        sql(tx.commit())?;
    }
    private_dir(&service.path(&hash)).map_err(|_| ApiError::unavailable())?;
    let mut config = StoreConfig::new(1);
    config.readers = 1;
    let store = native(Store::open(service.path(&hash).join("memory.sqlite"), config).await)?;
    native(Memory::initialize(&store, AGENT).await)?;
    let c = conversation.clone();
    native(store.write(move |w|{w.conn().execute_batch("CREATE TABLE trial_meta(name TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE trial_requests(request_id TEXT PRIMARY KEY,digest TEXT NOT NULL,status TEXT NOT NULL,response TEXT); INSERT INTO trial_meta VALUES('observations','0');")?;w.conn().execute("INSERT INTO trial_meta VALUES('conversation',?1)",[c])?;Ok(())}).await)?;
    native(store.close().await)?;
    let mut details = service.details(expiry, &conversation);
    details["api_key"] = json!(api_key);
    let mut response = Json(details).into_response();
    let secure = if service.0.config.secure_cookie { "; Secure" } else { "" };
    let cookie = format!(
        "{}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={}{}",
        service.cookie_name(),
        service.0.config.retention.as_secs(),
        secure
    );
    response
        .headers_mut()
        .insert(header::SET_COOKIE, HeaderValue::from_str(&cookie).map_err(|_| ApiError::unavailable())?);
    Ok(response)
}
async fn current_conversation(store: &Store) -> Result<String> {
    native(
        store
            .read(|c| Ok(c.query_row("SELECT value FROM trial_meta WHERE name='conversation'", [], |r| r.get(0))?))
            .await,
    )
}
async fn state_value(service: &Service, store: &Store, memory: &Memory, expiry: i64) -> Result<Value> {
    let conversation = current_conversation(store).await?;
    let mut value = service.details(expiry, &conversation);
    value["memory"] = json!(native(memory.snapshot_with_episodes(None).await)?.0);
    value["supersessions"] = json!(supersessions(memory).await?);
    Ok(value)
}
async fn trial_state(State(service): State<Service>, headers: HeaderMap) -> Result<Json<Value>> {
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    let expiry = service.authenticate_request(&hash, &headers)?;
    let (store, memory) = service.store(&hash).await?;
    let result = state_value(&service, &store, &memory, expiry).await;
    native(store.close().await)?;
    Ok(Json(result?))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationInput {
    request_id: String,
    text: String,
    #[serde(default)]
    supersedes: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestId {
    request_id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecallInput {
    query: String,
}
fn body<T>(input: std::result::Result<Json<T>, JsonRejection>) -> Result<T> {
    input.map(|Json(v)| v).map_err(|e| {
        if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError(StatusCode::PAYLOAD_TOO_LARGE, "body_limit", "Request body exceeds 8192 bytes.")
        } else {
            ApiError::invalid("Malformed JSON or unexpected fields.")
        }
    })
}
fn check_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 96 || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)) {
        Err(ApiError::invalid("request_id must be 1..96 ASCII letters, digits, underscore or hyphen."))
    } else {
        Ok(())
    }
}
fn check_text(text: &str) -> Result<()> {
    if text.trim().is_empty() || text.len() > MESSAGE_BYTES {
        Err(ApiError::invalid("Text must contain 1..4000 UTF-8 bytes."))
    } else {
        Ok(())
    }
}
async fn reserve(store: &Store, request_id: &str, kind: &str, text: &str) -> Result<Option<Value>> {
    check_id(request_id)?;
    let request_id = request_id.to_owned();
    let digest = blake3::hash(serde_json::to_string(&(kind, text)).unwrap().as_bytes()).to_hex().to_string();
    let result = store
        .write(move |w| {
            let prior: Option<(String, String, Option<String>)> = w
                .conn()
                .query_row(
                    "SELECT digest,status,response FROM trial_requests WHERE request_id=?1",
                    [&request_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                )
                .optional()?;
            if let Some(prior) = prior {
                return Ok(Some(prior));
            }
            let count: i64 = w.conn().query_row("SELECT count(*) FROM trial_requests", [], |r| r.get(0))?;
            if count >= 120 {
                return Err(licorice::Error::InvalidInput("operation quota".into()));
            }
            w.conn()
                .execute("INSERT INTO trial_requests VALUES(?1,?2,'reserved',NULL)", params![request_id, digest])?;
            Ok(None)
        })
        .await
        .map_err(|error| match error {
            licorice::Error::InvalidInput(ref reason) if reason == "operation quota" => ApiError::quota(),
            _ => ApiError::unavailable(),
        })?;
    match result {
        Some((old, status, response)) => {
            if old != blake3::hash(serde_json::to_string(&(kind, text)).unwrap().as_bytes()).to_hex().to_string() {
                Err(ApiError(
                    StatusCode::CONFLICT,
                    "request_conflict",
                    "request_id was used for different content or operation.",
                ))
            } else if status == "complete" {
                Ok(Some(serde_json::from_str(&response.unwrap_or_default()).map_err(|_| ApiError::unavailable())?))
            } else {
                Err(ApiError(
                    StatusCode::CONFLICT,
                    "operation_incomplete",
                    "This request is incomplete. Inspect memory and history before starting a new operation.",
                ))
            }
        }
        None => Ok(None),
    }
}
async fn finish(store: &Store, request_id: &str, response: Value) -> Result<Value> {
    let id = request_id.to_owned();
    let text = response.to_string();
    native(
        store
            .write(move |w| {
                w.conn().execute(
                    "UPDATE trial_requests SET status='complete',response=?2 WHERE request_id=?1",
                    params![id, text],
                )?;
                Ok(())
            })
            .await,
    )?;
    Ok(response)
}
async fn observe(
    memory: &Memory,
    request_id: &str,
    text: &str,
    session: &str,
    supersedes: Option<&str>,
) -> Result<i64> {
    let origin = Origin::HumanReport;
    let actor = Actor { principal: "trial-visitor".into(), role: ActorRole::Human, session: session.into() };
    let r = native(
        memory
            .apply(
                actor,
                being::Request {
                    request_id: request_id.into(),
                    expected_revision: None,
                    operation: Operation::Observe(Observation {
                        origin,
                        subject: "trial-visitor".into(),
                        text: text.into(),
                        relationship_id: None,
                        occurred_at: None,
                        source: Source::Reported {
                            reference: Some(json!({"session":session,"supersedes":supersedes}).to_string()),
                        },
                    }),
                },
            )
            .await,
    )?;
    Ok(r.revision)
}
async fn increment(store: &Store, key: &str, limit: usize) -> Result<()> {
    let key = key.to_owned();
    let used = native(
        store
            .write(move |w| {
                let value: i64 =
                    w.conn().query_row("SELECT CAST(value AS INTEGER) FROM trial_meta WHERE name=?1", [&key], |r| {
                        r.get(0)
                    })?;
                if value >= limit as i64 {
                    return Ok(false);
                }
                w.conn().execute("UPDATE trial_meta SET value=CAST(value AS INTEGER)+1 WHERE name=?1", [key])?;
                Ok(true)
            })
            .await,
    )?;
    if used { Ok(()) } else { Err(ApiError::quota()) }
}
fn source_supersessions(snapshot: &being::Snapshot) -> Vec<(String, String)> {
    snapshot
        .observations
        .iter()
        .filter_map(|o| {
            let Source::Reported { reference: Some(reference) } = &o.observation.source else {
                return None;
            };
            let value: Value = serde_json::from_str(reference).ok()?;
            Some((value.get("supersedes")?.as_str()?.to_owned(), o.source_id.clone()))
        })
        .collect()
}
async fn supersessions(memory: &Memory) -> Result<Vec<(String, String)>> {
    Ok(source_supersessions(&native(memory.snapshot_with_episodes(None).await)?.0))
}
async fn add_observation(service: &Service, store: &Store, memory: &Memory, input: ObservationInput) -> Result<Value> {
    check_text(&input.text)?;
    let payload = json!({"text":input.text,"supersedes":input.supersedes}).to_string();
    if let Some(response) = reserve(store, &input.request_id, "observation", &payload).await? {
        return Ok(response);
    }
    if let Some(old) = &input.supersedes {
        let snapshot = native(memory.snapshot_with_episodes(None).await)?.0;
        if !snapshot.observations.iter().any(|o| &o.source_id == old)
            || supersessions(memory).await?.iter().any(|(id, _)| id == old)
        {
            return Err(ApiError::invalid("supersedes must name an active source in this workspace."));
        }
    }
    increment(store, "observations", service.0.config.max_observations).await?;
    let conversation = current_conversation(store).await?;
    let source_id = format!("human:{}", input.request_id);
    let revision = observe(memory, &source_id, &input.text, &conversation, input.supersedes.as_deref()).await?;
    finish(store,&input.request_id,json!({"request_id":input.request_id,"memory_revision":revision,"source_id":source_id,"supersedes":input.supersedes})).await
}
async fn observation(
    State(service): State<Service>,
    headers: HeaderMap,
    input: std::result::Result<Json<ObservationInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    let (store, memory) = service.store(&hash).await?;
    let result = add_observation(&service, &store, &memory, input).await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn packet(service: &Service, memory: &Memory, query: &str) -> Result<Value> {
    if query.chars().count() > 2000 {
        return Err(ApiError::invalid("Recall query must contain at most 2000 characters."));
    }
    let (mut snapshot, report) = native(memory.snapshot_with_episodes(None).await)?;
    let superseded = source_supersessions(&snapshot);
    snapshot.observations.retain(|o| !superseded.iter().any(|(old, _)| old == &o.source_id));
    let mut packet = match &service.0.config.embedder {
        Some(embedder) => native(
            being::context::semantic_continuity_with_config(
                &snapshot,
                query,
                embedder.as_ref(),
                &Cl100kCounter,
                BUDGET,
                service.0.config.semantic,
            )
            .await,
        )?,
        _ => native(being::context::continuity(&snapshot, query, &Cl100kCounter, BUDGET))?,
    };
    packet.episode_search = Some(report);
    let mut value = json!(packet);
    value["supersessions"] = json!(superseded);
    value["retrieval"] = service.retrieval_profile();
    Ok(value)
}
async fn recall(
    State(service): State<Service>,
    headers: HeaderMap,
    input: std::result::Result<Json<RecallInput>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    check_text(&input.query)?;
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    let (store, memory) = service.store(&hash).await?;
    let result = packet(&service, &memory, &input.query).await;
    native(store.close().await)?;
    Ok(Json(result?))
}
async fn conversation(
    State(service): State<Service>,
    headers: HeaderMap,
    input: std::result::Result<Json<RequestId>, JsonRejection>,
) -> Result<Json<Value>> {
    let input = body(input)?;
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    let (store, _) = service.store(&hash).await?;
    let result = async {
        if let Some(response) = reserve(&store, &input.request_id, "conversation", "").await? {
            return Ok(response);
        }
        let conversation = id()?;
        let c = conversation.clone();
        native(
            store
                .write(move |w| {
                    w.conn().execute("UPDATE trial_meta SET value=?1 WHERE name='conversation'", [c])?;
                    Ok(())
                })
                .await,
        )?;
        finish(&store, &input.request_id, json!({"conversation_id":conversation})).await
    }
    .await;
    native(store.close().await)?;
    Ok(Json(result?))
}

async fn export(State(service): State<Service>, headers: HeaderMap) -> Result<Response> {
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    let expiry = service.authenticate_request(&hash, &headers)?;
    let (store, memory) = service.store(&hash).await?;
    let result = async {
        let state = state_value(&service, &store, &memory, expiry).await?;
        Ok::<_, ApiError>(
            json!({"schema":"licorice-trial-export/1","state":state,"history":native(memory.history().await)?}),
        )
    }
    .await;
    native(store.close().await)?;
    Ok(([(header::CONTENT_DISPOSITION, "attachment; filename=licorice-trial.json")], Json(result?)).into_response())
}
async fn history(State(service): State<Service>, headers: HeaderMap) -> Result<Json<Value>> {
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    let (store, memory) = service.store(&hash).await?;
    let result = native(memory.history().await);
    native(store.close().await)?;
    Ok(Json(json!({"history":result?})))
}
async fn delete_trial(State(service): State<Service>, headers: HeaderMap) -> Result<Response> {
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    service.erase(&hash)?;
    let mut response = Json(json!({"deleted":true})).into_response();
    let cookie = format!(
        "{}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0{}",
        service.cookie_name(),
        if service.0.config.secure_cookie { "; Secure" } else { "" }
    );
    response
        .headers_mut()
        .insert(header::SET_COOKIE, HeaderValue::from_str(&cookie).map_err(|_| ApiError::unavailable())?);
    Ok(response)
}

async fn rotate_key(State(service): State<Service>, headers: HeaderMap) -> Result<Json<Value>> {
    let hash = service.token(&headers)?;
    let _guard = service.0.locks[service.stripe(&hash)].lock().await;
    service.authenticate_request(&hash, &headers)?;
    let api_key = id()?;
    let key_hash = blake3::hash(api_key.as_bytes()).to_hex().to_string();
    let mut conn = service.0.registry.lock().map_err(|_| ApiError::unavailable())?;
    let tx = sql(conn.transaction())?;
    sql(tx.execute("DELETE FROM api_keys WHERE workspace_hash=?1", [&hash]))?;
    sql(tx.execute("INSERT INTO api_keys VALUES(?1,?2)", params![key_hash, hash]))?;
    sql(tx.commit())?;
    Ok(Json(json!({"api_key":api_key,"revoked_previous":true})))
}

#[cfg(test)]
mod revocation_tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use std::{
        future::Future,
        task::{Context, Poll, Waker},
    };
    use tower::ServiceExt;
    #[tokio::test]
    async fn a_queued_old_key_is_rejected_after_rotation() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::new(tmp.path(), "http://localhost:8080");
        config.secure_cookie = false;
        let service = Service::open(config).unwrap();
        let app = service.router();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/workspaces")
                    .header("origin", "http://localhost:8080")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let cookie = response.headers()[header::SET_COOKIE].to_str().unwrap().split(';').next().unwrap().to_owned();
        let body: Value = serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(header::COOKIE, cookie.parse().unwrap());
        let hash = service.token(&headers).unwrap();
        let guard = service.0.locks[service.stripe(&hash)].lock().await;
        let mut request = Box::pin(
            app.oneshot(
                Request::builder()
                    .uri("/api/v1/workspace")
                    .header("authorization", format!("Bearer {}", body["api_key"].as_str().unwrap()))
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        // Poll synchronously until it queues on the held stripe, after the first credential lookup.
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(request.as_mut().poll(&mut context), Poll::Pending));
        {
            let conn = service.0.registry.lock().unwrap();
            conn.execute("DELETE FROM api_keys WHERE workspace_hash=?1", [&hash]).unwrap();
        }
        drop(guard);
        assert_eq!(request.await.unwrap().status(), StatusCode::UNAUTHORIZED);
        assert!(service.authenticate_request(&hash, &headers).unwrap() > now_ms());
    }
}
