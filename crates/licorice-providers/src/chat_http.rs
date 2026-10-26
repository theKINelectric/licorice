//! Budgeted OpenRouter GPT-4o-mini execution with durable exact-input caching.
use crate::chat::{ChatError, ChatModel, ModelRequest, validate};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

type Result<T> = std::result::Result<T, ChatError>;
const MODEL: &str = "openai/gpt-4o-mini";
const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const INPUT: f64 = 0.15;
const OUTPUT: f64 = 0.60;

/// A rollback-journal claim releases on normal drop, cancellation or process
/// death. It never holds the spend ledger's write lock across HTTP.
struct CacheClaim(Connection);
impl Drop for CacheClaim {
    fn drop(&mut self) {
        let _ = self.0.execute_batch("ROLLBACK");
    }
}

pub struct BudgetChat {
    key: String,
    client: reqwest::Client,
    ledger: Mutex<Connection>,
    locks: Vec<tokio::sync::Mutex<()>>,
    capacity: tokio::sync::Semaphore,
    ceiling: f64,
    endpoint: String,
    claim_dir: PathBuf,
}
impl BudgetChat {
    pub fn open(key: String, path: &Path, ceiling: f64) -> Result<Self> {
        Self::at_endpoint(key, path, ceiling, ENDPOINT)
    }
    /// Alternate endpoint is limited to loopback for offline transport tests.
    pub fn local_test(key: String, path: &Path, ceiling: f64, endpoint: &str) -> Result<Self> {
        let url = reqwest::Url::parse(endpoint).map_err(|_| err("invalid local endpoint"))?;
        if url.scheme() != "http" || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")) {
            return Err(err("test endpoint must be HTTP loopback"));
        }
        Self::at_endpoint(key, path, ceiling, endpoint)
    }
    pub(crate) fn at_endpoint(key: String, path: &Path, ceiling: f64, endpoint: &str) -> Result<Self> {
        if key.trim().is_empty() || !ceiling.is_finite() || ceiling <= 0. {
            return Err(err("credential and positive finite ceiling required"));
        }
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|_| err("create ledger directory failed"))?;
        }
        let conn = Connection::open(path).map_err(|_| err("open ledger failed"))?;
        conn.busy_timeout(Duration::from_secs(10)).map_err(db)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1),ceiling REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,hash TEXT NOT NULL,request TEXT NOT NULL,reservation REAL NOT NULL,cost REAL,raw TEXT,output TEXT,error TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
          CREATE INDEX IF NOT EXISTS calls_hash ON calls(hash);").map_err(db)?;
        conn.execute("INSERT OR IGNORE INTO budget(id,ceiling) VALUES(1,?1)", [ceiling]).map_err(db)?;
        let declared: f64 = conn.query_row("SELECT ceiling FROM budget WHERE id=1", [], |r| r.get(0)).map_err(db)?;
        if ceiling > declared {
            return Err(err("requested ceiling exceeds the ledger's original ceiling"));
        }
        let canonical = std::fs::canonicalize(path).map_err(|_| err("canonicalize ledger failed"))?;
        let mut claim_name = canonical.file_name().ok_or_else(|| err("ledger filename required"))?.to_os_string();
        claim_name.push(".cache-locks");
        let claim_dir = canonical.with_file_name(claim_name);
        std::fs::create_dir_all(&claim_dir).map_err(|_| err("create cache claim directory failed"))?;
        let mut builder =
            reqwest::Client::builder().timeout(Duration::from_secs(120)).redirect(reqwest::redirect::Policy::none());
        if endpoint != ENDPOINT {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|_| err("HTTP client setup failed"))?;
        Ok(Self {
            key,
            client,
            ledger: Mutex::new(conn),
            locks: (0..64).map(|_| tokio::sync::Mutex::new(())).collect(),
            ceiling,
            capacity: tokio::sync::Semaphore::new(16),
            endpoint: endpoint.into(),
            claim_dir,
        })
    }
    async fn claim(&self, stripe: u8) -> Result<CacheClaim> {
        let path = self.claim_dir.join(format!("{:02}.sqlite", stripe as usize % 64));
        tokio::task::spawn_blocking(move || {
            let conn = Connection::open(path).map_err(db)?;
            // Four bounded HTTP attempts can outlive a single 120-second call.
            // Waiting never steals a live owner's claim or sends another call.
            conn.busy_timeout(Duration::from_secs(500)).map_err(db)?;
            conn.execute_batch("CREATE TABLE IF NOT EXISTS cache_claim(id INTEGER PRIMARY KEY); BEGIN IMMEDIATE;")
                .map_err(db)?;
            Ok(CacheClaim(conn))
        })
        .await
        .map_err(|_| err("cache claim worker failed"))?
    }
    fn retain_interrupted(&self, hash: &str) -> Result<()> {
        // Owning this shared claim proves no participating client still owns an
        // unclosed request for this hash. Preserve its unknown charge on recovery.
        self.ledger.lock().map_err(|_|err("ledger lock poisoned"))?.execute(
            "UPDATE calls SET error='interrupted request owner; reservation retained' WHERE hash=?1 AND output IS NULL AND cost IS NULL AND error IS NULL",[hash]).map_err(db)?;
        Ok(())
    }
    pub fn accounted(&self) -> Result<f64> {
        self.ledger
            .lock()
            .map_err(|_| err("ledger lock poisoned"))?
            .query_row("SELECT coalesce(sum(coalesce(cost,reservation)),0) FROM calls", [], |r| r.get(0))
            .map_err(db)
    }
    fn reserve(&self, hash: &str, identity: &Value, reservation: f64) -> Result<i64> {
        let mut conn = self.ledger.lock().map_err(|_| err("ledger lock poisoned"))?;
        let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(db)?;
        let used: f64 = tx
            .query_row("SELECT coalesce(sum(coalesce(cost,reservation)),0) FROM calls", [], |r| r.get(0))
            .map_err(db)?;
        if used + reservation > self.ceiling {
            return Err(err("model spend admission refused: ceiling reached"));
        }
        tx.execute(
            "INSERT INTO calls(hash,request,reservation) VALUES(?1,?2,?3)",
            params![hash, identity.to_string(), reservation],
        )
        .map_err(db)?;
        let id = tx.last_insert_rowid();
        tx.commit().map_err(db)?;
        Ok(id)
    }
    fn cached(&self, hash: &str) -> Result<Option<Value>> {
        let text: Option<String> = self
            .ledger
            .lock()
            .map_err(|_| err("ledger lock poisoned"))?
            .query_row(
                "SELECT output FROM calls WHERE hash=?1 AND output IS NOT NULL ORDER BY id DESC LIMIT 1",
                [hash],
                |r| r.get(0),
            )
            .optional()
            .map_err(db)?;
        if let Some(text) = text {
            return serde_json::from_str(&text).map(Some).map_err(|_| err("cached output is invalid"));
        }
        // A completed response cut at a fixed token cap cannot be repaired by
        // paying for the identical request again. Recognize legacy receipts too.
        let limited: bool = self.ledger.lock().map_err(|_|err("ledger lock poisoned"))?.query_row(
            "SELECT EXISTS(SELECT 1 FROM calls WHERE hash=?1 AND output IS NULL AND error IN ('model output token limit reached','model response refused or incomplete') AND json_extract(raw,'$.choices[0].finish_reason')='length' AND json_extract(raw,'$.choices[0].message.refusal') IS NULL)",
            [hash],|r|r.get(0)).map_err(db)?;
        if limited {
            return Err(ChatError::output_limit());
        }
        Ok(None)
    }
    fn settle(
        &self,
        id: i64,
        raw: Option<&Value>,
        output: Option<&Value>,
        cost: Option<f64>,
        error: Option<&str>,
    ) -> Result<()> {
        self.ledger
            .lock()
            .map_err(|_| err("ledger lock poisoned"))?
            .execute(
                "UPDATE calls SET raw=?1,output=?2,cost=?3,error=?4 WHERE id=?5",
                params![raw.map(Value::to_string), output.map(Value::to_string), cost, error, id],
            )
            .map_err(db)?;
        Ok(())
    }
    /// Shared durable JSON transport. Public provider wrappers validate their
    /// own endpoint, request format, pricing and response before exposing this.
    pub(crate) async fn json_request<P, C>(
        &self,
        identity: &Value,
        body: &Value,
        reservation: f64,
        parse_output: P,
        cost_for: C,
    ) -> Result<Value>
    where
        P: Fn(&Value) -> Result<Value> + Send + Sync,
        C: Fn(&Value) -> Option<f64> + Send + Sync,
    {
        if !reservation.is_finite() || reservation <= 0. {
            return Err(err("positive finite provider reservation required"));
        }
        let digest = blake3::hash(identity.to_string().as_bytes());
        let _guard = self.locks[digest.as_bytes()[0] as usize % self.locks.len()].lock().await;
        let hash = digest.to_hex().to_string();
        if let Some(output) = self.cached(&hash)? {
            return Ok(output);
        }
        let _shared_claim = self.claim(digest.as_bytes()[0]).await?;
        if let Some(output) = self.cached(&hash)? {
            return Ok(output);
        }
        self.retain_interrupted(&hash)?;
        let _capacity = self.capacity.acquire().await.map_err(|_| err("model capacity is closed"))?;
        for attempt in 0..4 {
            let id = self.reserve(&hash, identity, reservation)?;
            let sent = self.client.post(&self.endpoint).bearer_auth(&self.key).json(body).send().await;
            let response = match sent {
                Ok(response) => response,
                Err(_) => {
                    self.settle(id, None, None, None, Some("transport failure; reservation retained"))?;
                    return Err(err("model transport failure; reservation retained"));
                }
            };
            let status = response.status();
            if !status.is_success() {
                let known_rejection = matches!(status.as_u16(), 400 | 401 | 402 | 403 | 404 | 429);
                let error_raw: Option<Value> = response.json().await.ok();
                self.settle(
                    id,
                    error_raw.as_ref(),
                    None,
                    known_rejection.then_some(0.),
                    Some(&format!("HTTP {}", status.as_u16())),
                )?;
                if (status.as_u16() == 429 || status.is_server_error()) && attempt < 3 {
                    tokio::time::sleep(Duration::from_millis(250 * (1 << attempt))).await;
                    continue;
                }
                return Err(err(&format!("provider returned HTTP {}", status.as_u16())));
            }
            let raw: Value = match response.json().await {
                Ok(raw) => raw,
                Err(_) => {
                    self.settle(id, None, None, None, Some("invalid response; reservation retained"))?;
                    return Err(err("invalid provider response; reservation retained"));
                }
            };
            let cost = cost_for(&raw);
            let output = parse_output(&raw);
            let failure = output.as_ref().err().map(|e| e.0.as_str());
            self.settle(id, Some(&raw), output.as_ref().ok(), cost, failure)?;
            if cost.is_none() {
                return Err(err("provider usage is missing; reservation retained"));
            }
            if cost.is_some_and(|actual| actual > reservation) {
                return Err(err("provider cost exceeded reservation; actual cost retained"));
            }
            return output;
        }
        unreachable!()
    }
}
fn chat_cost(raw: &Value) -> Option<f64> {
    raw["usage"]["cost"].as_f64().filter(|v| v.is_finite() && *v >= 0.).or_else(|| {
        let input = raw["usage"]["prompt_tokens"].as_u64()?;
        let output = raw["usage"]["completion_tokens"].as_u64()?;
        Some((input as f64 * INPUT + output as f64 * OUTPUT) / 1e6)
    })
}

fn err(message: &str) -> ChatError {
    ChatError(message.into())
}
fn db(_: rusqlite::Error) -> ChatError {
    err("durable ledger operation failed")
}

/// Strict output requires every object property, including nullable optionals.
fn strict_schema(schema: &Value) -> Value {
    let mut result = schema.clone();
    if let Some(map) = result.as_object_mut() {
        for value in map.values_mut() {
            if value.is_object() {
                *value = strict_schema(value);
            } else if let Some(items) = value.as_array_mut() {
                for item in items {
                    if item.is_object() {
                        *item = strict_schema(item);
                    }
                }
            }
        }
        if schema["type"] == "object" {
            let old = schema["required"].as_array().cloned().unwrap_or_default();
            if let Some(properties) = map.get_mut("properties").and_then(Value::as_object_mut) {
                for (name, value) in properties.iter_mut() {
                    if !old.contains(&json!(name)) {
                        *value = json!({"anyOf":[value.clone(),{"type":"null"}]});
                    }
                }
                let required: Vec<_> = properties.keys().cloned().collect();
                map.insert("required".into(), json!(required));
                map.insert("additionalProperties".into(), json!(false));
            }
        }
    }
    result
}
fn body(request: &ModelRequest) -> Value {
    let mut messages = Vec::new();
    if !request.system.is_empty() {
        messages.push(json!({"role":"system","content":request.system}));
    }
    messages.push(json!({"role":"user","content":request.user}));
    let mut result = json!({"model":MODEL,"temperature":0,"max_tokens":request.max_tokens,
       "provider":{"order":["OpenAI"],"allow_fallbacks":false,"require_parameters":true},
       "messages":messages});
    if let Some(schema) = &request.schema {
        result["response_format"] = json!({"type":"json_schema","json_schema":{"name":"licorice_output","strict":true,"schema":strict_schema(schema)}});
    }
    result
}
fn parse(request: &ModelRequest, raw: &Value) -> Result<Value> {
    if raw["model"].as_str().is_some_and(|model| !model.contains("gpt-4o-mini")) {
        return Err(err("provider returned a different model"));
    }
    let choice = &raw["choices"][0];
    if choice["message"]["refusal"].is_null() && choice["finish_reason"] == "length" {
        return Err(ChatError::output_limit());
    }
    if choice["finish_reason"] != "stop" || !choice["message"]["refusal"].is_null() {
        return Err(err("model response refused or incomplete"));
    }
    let text = choice["message"]["content"]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| err("empty model response"))?;
    if let Some(schema) = &request.schema {
        let value = serde_json::from_str(text).map_err(|_| err("model response is not JSON"))?;
        validate(schema, &value).map_err(|_| err("model response failed schema validation"))?;
        Ok(value)
    } else {
        Ok(json!(text))
    }
}
#[async_trait::async_trait]
impl ChatModel for BudgetChat {
    fn id(&self) -> &str {
        "openrouter/openai/gpt-4o-mini:temperature=0:openai-only:v1"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<Value> {
        if request.max_tokens == 0 || request.max_tokens > 8192 {
            return Err(err("max_tokens must be in 1..=8192"));
        }
        let body = body(request);
        let identity = json!({"provider":"openrouter","endpoint":self.endpoint,"model":MODEL,"request_id":request.id,"task":request.task,"body":body,
            "input_rate":INPUT,"output_rate":OUTPUT,"price_source":"https://openrouter.ai/openai/gpt-4o-mini"});
        // UTF-8 byte length bounds ordinary tokens; fixed framing allowance plus
        // strict-schema bytes and a 25% margin cover provider tokenization overhead.
        let reservation =
            ((body.to_string().len() + 512) as f64 * INPUT + request.max_tokens as f64 * OUTPUT) * 1.25 / 1e6;
        self.json_request(&identity, &body, reservation, |raw| parse(request, raw), chat_cost).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grading_can_preserve_a_user_only_prompt_and_the_explicit_completion_limit() {
        let mut request = ModelRequest {
            id: "grade".into(),
            task: "grade".into(),
            system: String::new(),
            user: "published rubric".into(),
            schema: None,
            max_tokens: 10,
        };
        let wire = body(&request);
        assert_eq!(wire["messages"], json!([{"role":"user","content":"published rubric"}]));
        assert_eq!(wire["temperature"], 0);
        assert_eq!(wire["max_tokens"], 10);
        assert!(wire.get("response_format").is_none());
        request.system = "instructions".into();
        assert_eq!(body(&request)["messages"][0], json!({"role":"system","content":"instructions"}));
    }
    #[test]
    fn strict_schema_preserves_optional_meaning_and_enforces_required_properties() {
        let schema = json!({"type":"object","required":["x"],"properties":{"x":{"type":"integer"},"note":{"type":"string"}},"additionalProperties":false});
        let strict = strict_schema(&schema);
        assert_eq!(strict["required"], json!(["note", "x"]));
        assert!(validate(&strict, &json!({"x":1,"note":null})).is_ok());
        assert!(validate(&strict, &json!({"x":1,"note":3})).is_err());
    }
    #[test]
    fn parser_rejects_truncation_and_wrong_models() {
        let request = ModelRequest {
            id: "r".into(),
            task: "answer".into(),
            system: "s".into(),
            user: "u".into(),
            schema: None,
            max_tokens: 10,
        };
        assert!(
            parse(
                &request,
                &json!({"model":"other","choices":[{"finish_reason":"stop","message":{"content":"answer"}}]})
            )
            .is_err()
        );
        assert!(
            parse(
                &request,
                &json!({"model":"gpt-4o-mini","choices":[{"finish_reason":"length","message":{"content":"partial"}}]})
            )
            .is_err()
        );
    }
}
