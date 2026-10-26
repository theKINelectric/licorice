//! Synthetic HTTP-level controls. No model service or evaluation dataset is used.
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use licorice_aml::{Adapter, Config};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Default)]
struct LocalEmbedder {
    calls: AtomicUsize,
    bad_vector: AtomicBool,
}
#[async_trait::async_trait]
impl Embedder for LocalEmbedder {
    fn id(&self) -> &str {
        "contract/bag-of-words-v1"
    }
    fn dim(&self) -> usize {
        32
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut vectors: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| {
                let mut vector = vec![0.0; 32];
                for word in text.to_lowercase().split_whitespace() {
                    vector[blake3::hash(word.as_bytes()).as_bytes()[0] as usize % 32] += 1.0;
                }
                vector
            })
            .collect();
        if self.bad_vector.swap(false, Ordering::SeqCst) {
            // Fail after the first source/FTS/vector rows have been staged.
            *vectors.last_mut().unwrap() = vec![f32::NAN; 32];
        }
        Ok(vectors)
    }
}
fn config(dir: &tempfile::TempDir, budget: usize) -> Config {
    Config { data_dir: dir.path().into(), api_key: "synthetic-test-secret".into(), budget_tokens: budget }
}
fn add(user: &str, request: &str, session: &str, text: &str) -> Value {
    json!({"request_id":request,"user_id":user,"session_id":session,"messages":[
        {"role":"user","timestamp":1683504000000_i64,"content":text}
    ]})
}
fn search(user: &str, top_k: usize) -> Value {
    json!({"query":"Where did Ada travel?","user_id":user,"top_k":top_k})
}
async fn call(adapter: &Adapter, path: &str, body: Value) -> (StatusCode, Value) {
    raw(adapter, path, body.to_string(), Some(("authorization", "Bearer synthetic-test-secret"))).await
}
async fn raw(adapter: &Adapter, path: &str, body: String, auth: Option<(&str, &str)>) -> (StatusCode, Value) {
    let mut builder = Request::builder().method("POST").uri(path).header("content-type", "application/json");
    if let Some((name, value)) = auth {
        builder = builder.header(name, value);
    }
    let response = adapter.router().oneshot(builder.body(Body::from(body)).unwrap()).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 2 * 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).expect("all API errors and responses are JSON"))
}
fn data(value: &Value) -> &Vec<Value> {
    value["data"].as_array().unwrap()
}

#[tokio::test]
async fn synchronous_add_echoes_exact_ids_and_search_is_evidence_only() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let request = add(" user:../Ada 🍌 ", " request-1 ", "session:🧬", "Ada traveled to Oslo. <|endoftext|>");
    let (status, response) = call(&adapter, "/add", request.clone()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(
        response,
        json!({"success":true,"request_id":request["request_id"],
        "user_id":request["user_id"],"session_id":request["session_id"]})
    );
    let mut query = search(request["user_id"].as_str().unwrap(), 100);
    let (status, response) = call(&adapter, "/search", query.clone()).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    assert_eq!(data(&response).len(), 1);
    let memory = &data(&response)[0];
    assert!(!memory["id"].as_str().unwrap().is_empty());
    assert!(memory["content"].as_str().unwrap().contains("Ada traveled to Oslo. <|endoftext|>"));
    assert_eq!(memory["created_at"], "2023-05-08T00:00:00.000Z");
    query["options"] = json!(["INJECTED_OPTION_NEVER_STORED", "Paris"]);
    assert_eq!(call(&adapter, "/search", query).await.1, response);
    assert_eq!(call(&adapter, "/search", search("user:../Ada 🍌", 100)).await.1, json!({"data":[]}));
    assert_eq!(call(&adapter, "/search", search("other", 100)).await.1, json!({"data":[]}));
}

#[tokio::test]
async fn replay_conflicts_and_restart_preserve_memory_and_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = Arc::new(LocalEmbedder::default());
    let adapter = Adapter::open(config(&dir, 2000), embedder.clone()).unwrap();
    let mut request = add("u", "r", "s", "Ada traveled to Oslo.");
    request["messages"][0].as_object_mut().unwrap().remove("timestamp");
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::OK);
    let first = call(&adapter, "/search", search("u", 100)).await.1;
    let calls = embedder.calls.load(Ordering::SeqCst);
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::OK);
    assert_eq!(embedder.calls.load(Ordering::SeqCst), calls, "replay must not embed again");
    let mut changed = request.clone();
    changed["messages"][0]["content"] = json!("Ada traveled to Paris.");
    assert_eq!(call(&adapter, "/add", changed).await.0, StatusCode::CONFLICT);
    let mut changed_session = request.clone();
    changed_session["session_id"] = json!("different");
    assert_eq!(call(&adapter, "/add", changed_session).await.0, StatusCode::CONFLICT);
    drop(adapter);
    let adapter = Adapter::open(config(&dir, 2000), embedder.clone()).unwrap();
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    assert_eq!(call(&adapter, "/search", search("u", 100)).await.1, first);
    assert_eq!(call(&adapter, "/add", add("u", "r2", "s", "Ada traveled to Rome.")).await.0, StatusCode::OK);
    let found = call(&adapter, "/search", search("u", 100)).await.1;
    assert_eq!(data(&found).len(), 2, "chunks in the same external session must coexist");
}

#[tokio::test]
async fn auth_health_and_validation_are_explicit_and_do_not_mutate_memory() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let health =
        adapter.router().oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    for header in
        [None, Some(("authorization", "Bearer wrong")), Some(("authorization", "Basic synthetic-test-secret"))]
    {
        let (status, value) = raw(&adapter, "/add", "{bad-json".into(), header).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert!(!value.to_string().contains("synthetic-test-secret"));
    }
    for header in [
        ("authorization", "Token synthetic-test-secret"),
        ("authorization", "Bearer synthetic-test-secret"),
        ("x-api-key", "synthetic-test-secret"),
    ] {
        assert_eq!(raw(&adapter, "/search", search("u", 100).to_string(), Some(header)).await.0, StatusCode::OK);
    }
    for invalid in [
        json!({}),
        json!({"request_id":"r","user_id":"u","session_id":"s","messages":[]}),
        add(" ", "r", "s", "text"),
        add("u", "r", "s", "  "),
        json!({"request_id":"r","user_id":"u","session_id":"s","messages":[{"role":"bogus","content":"x"}]}),
        json!({"request_id":"r","user_id":"u","session_id":"s","messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}),
    ] {
        assert_eq!(call(&adapter, "/add", invalid).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
    for invalid in [
        json!({"query":"q","user_id":"u"}),
        search("u", 1001),
        json!({"query":[],"user_id":"u","top_k":1}),
        json!({"query":" ","user_id":"u","top_k":1}),
    ] {
        assert_eq!(call(&adapter, "/search", invalid).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
    assert_eq!(call(&adapter, "/search", search("u", 100)).await.1, json!({"data":[]}));
}

#[tokio::test]
async fn failed_vector_write_rolls_back_source_fts_vectors_and_receipt_then_retry_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = Arc::new(LocalEmbedder::default());
    embedder.bad_vector.store(true, Ordering::SeqCst);
    let adapter = Adapter::open(config(&dir, 2000), embedder).unwrap();
    let mut request = add("u", "r", "s", "Ada traveled to Oslo.");
    request["messages"].as_array_mut().unwrap().push(json!({"role":"assistant","content":"Good trip."}));
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(call(&adapter, "/search", search("u", 100)).await.1, json!({"data":[]}));
    let path = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
        .unwrap();
    let db = licorice::rusqlite::Connection::open(path).unwrap();
    for table in ["turns", "sessions", "chunks", "vec_chunk", "aml_requests", "fts_chunk"] {
        let count: i64 = db.query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0, "partial transaction in {table}");
    }
    let indexed: i64 =
        db.query_row("SELECT count(*) FROM fts_chunk WHERE fts_chunk MATCH 'Oslo'", [], |r| r.get(0)).unwrap();
    assert_eq!(indexed, 0, "FTS inverted-index entries must also roll back");
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    assert_eq!(data(&call(&adapter, "/search", search("u", 100)).await.1).len(), 2);
}

#[tokio::test]
async fn concurrent_retries_and_users_do_not_duplicate_or_leak() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let request = add("u", "r", "s", "ONLY_USER_U Oslo");
    let (a, b, c) = tokio::join!(
        call(&adapter, "/add", request.clone()),
        call(&adapter, "/add", request),
        call(&adapter, "/add", add("v", "r", "s", "ONLY_USER_V Rome"))
    );
    for (status, value) in [a, b, c] {
        assert_eq!(status, StatusCode::OK, "{value}");
    }
    let u = call(&adapter, "/search", search("u", 100)).await.1;
    let v = call(&adapter, "/search", search("v", 100)).await.1;
    assert_eq!(data(&u).len(), 1);
    assert_eq!(data(&v).len(), 1);
    assert!(!u.to_string().contains("ONLY_USER_V"));
    assert!(!v.to_string().contains("ONLY_USER_U"));
    assert_ne!(data(&u)[0]["id"], data(&v)[0]["id"]);
}

#[tokio::test]
async fn whole_records_top_k_and_full_wire_budget_hold_under_unicode_pressure() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 200), Arc::new(LocalEmbedder::default())).unwrap();
    let mut request = add("u", "r", "s", &"oversized 🍌 日本語 café ".repeat(300));
    request["messages"].as_array_mut().unwrap().extend([
        json!({"role":"user","timestamp":1683504000001_i64,"content":"Ada traveled to Oslo. 🍌 日本語 café"}),
        json!({"role":"assistant","timestamp":1683590400002_i64,"content":"Ada traveled to Rome. 🍌 日本語 café"}),
    ]);
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    for top_k in [0, 1, 2, 100] {
        let (status, value) = call(&adapter, "/search", search("u", top_k)).await;
        assert_eq!(status, StatusCode::OK, "{value}");
        assert!(data(&value).len() <= top_k);
        let wire = serde_json::to_string(&value["data"]).unwrap();
        assert!(Cl100kCounter.count(&wire).unwrap() <= 200);
        for row in data(&value) {
            let text = row["content"].as_str().unwrap();
            assert!(!text.contains("oversized"), "oversized turns must be omitted whole");
            assert!(text.ends_with("🍌 日本語 café"), "complete evidence must survive");
        }
        if top_k > 0 {
            assert!(!data(&value).is_empty(), "smaller later evidence must still fit");
        }
    }
}

#[test]
fn ownership_and_configuration_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    assert!(Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).is_err());
    drop(adapter);
    assert!(Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).is_ok());
    let mut cfg = config(&dir, 2000);
    cfg.api_key.clear();
    assert!(Adapter::open(cfg, Arc::new(LocalEmbedder::default())).is_err());
    assert!(Adapter::open(config(&dir, 2001), Arc::new(LocalEmbedder::default())).is_err());
}

#[tokio::test]
async fn request_limits_are_json_errors_and_unknown_users_do_not_allocate_spaces() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let baseline = std::fs::read_dir(dir.path()).unwrap().count();
    for user in ["unknown", "../escaped", "Ω"] {
        assert_eq!(call(&adapter, "/search", search(user, 100)).await.1, json!({"data":[]}));
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), baseline);
    let mut excessive = add("u", "r", "s", "small");
    let message = excessive["messages"][0].clone();
    excessive["messages"] = json!(vec![message; 21]);
    assert_eq!(call(&adapter, "/add", excessive).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    for timestamp in [i64::MIN, i64::MAX, -62_135_596_800_001, 253_402_300_800_000] {
        let mut invalid = add("u", "r", "s", "small");
        invalid["messages"][0]["timestamp"] = json!(timestamp);
        assert_eq!(call(&adapter, "/add", invalid).await.0, StatusCode::UNPROCESSABLE_ENTITY);
    }
    let oversized = add("u", "r", "s", &"a".repeat(1024 * 1024));
    assert_eq!(call(&adapter, "/add", oversized).await.0, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), baseline);
}

#[tokio::test]
async fn search_preserves_relevance_order_and_each_source_timestamp() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let mut request = add("u", "r", "s", "gardening fertilizer potatoes");
    request["messages"].as_array_mut().unwrap().extend([
        json!({"role":"system","timestamp":-1,"content":"calendar archive"}),
        json!({"role":"tool","timestamp":1683590400123_i64,"content":"Ada travel Oslo"}),
    ]);
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    let mut query = search("u", 100);
    query["query"] = json!("Ada travel Oslo");
    let (status, value) = call(&adapter, "/search", query.clone()).await;
    assert_eq!(status, StatusCode::OK, "{value}");
    assert_eq!(data(&value).len(), 3);
    assert_eq!(data(&value)[0]["content"], "[2023-05-09] tool: Ada travel Oslo");
    assert_eq!(data(&value)[0]["created_at"], "2023-05-09T00:00:00.123Z");
    assert!(data(&value).iter().any(|r| r["created_at"] == "1969-12-31T23:59:59.999Z"));
    query["top_k"] = json!(1);
    assert_eq!(data(&call(&adapter, "/search", query).await.1), &vec![data(&value)[0].clone()]);
}

struct DifferentModel;
#[async_trait::async_trait]
impl Embedder for DifferentModel {
    fn id(&self) -> &str {
        "a-different-model-with-the-same-dimension"
    }
    fn dim(&self) -> usize {
        32
    }
    async fn embed(&self, _: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        panic!("mismatched model must be rejected before embedding");
    }
}
#[tokio::test]
async fn reopening_with_a_different_embedding_identity_fails_closed() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    let request = add("u", "r", "s", "Ada traveled to Oslo.");
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::OK);
    let expected = call(&adapter, "/search", search("u", 100)).await.1;
    drop(adapter);
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(DifferentModel)).unwrap();
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(call(&adapter, "/search", search("u", 100)).await.0, StatusCode::SERVICE_UNAVAILABLE);
    drop(adapter);
    let adapter = Adapter::open(config(&dir, 2000), Arc::new(LocalEmbedder::default())).unwrap();
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    assert_eq!(call(&adapter, "/search", search("u", 100)).await.1, expected);
}

#[derive(Default)]
struct ContextModel {
    requests: std::sync::Mutex<Vec<licorice_providers::chat::ModelRequest>>,
    fail_summary: AtomicBool,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for ContextModel {
    fn id(&self) -> &str {
        "offline-context-model-v1"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        self.requests.lock().unwrap().push(request.clone());
        if request.task == "summarize" {
            if self.fail_summary.swap(false, Ordering::SeqCst) {
                return Err(licorice_providers::chat::ChatError("injected summary failure".into()));
            }
            return Ok(json!({"keys":[],"themes":[]}));
        }
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        if current.contains("She lives in Bergen") {
            assert!(request.user.contains("My sister Mira"), "cross-Add context was lost");
            Ok(json!({"entities":[{"name":"Mira","kind":"person","aliases":[]}],"facts":[{
                "subject":"Mira","predicate":"lives in","object":"Bergen","object_is_entity":false,
                "statement":"Mira lives in Bergen.","negated":false,"salience":8,
                "time":{"kind":"ongoing","expression":null,"anchor_turn":id},"update":null,"evidence_turns":[id]}]}))
        } else {
            Ok(json!({"entities":[],"facts":[]}))
        }
    }
}
#[tokio::test]
async fn model_add_preserves_extraction_inputs_across_batch_boundaries_and_restart() {
    let a = tempfile::tempdir().unwrap();
    let b = tempfile::tempdir().unwrap();
    let model_a = Arc::new(ContextModel::default());
    let model_b = Arc::new(ContextModel::default());
    let adapter_a =
        Adapter::open_with_model(config(&a, 2000), Arc::new(LocalEmbedder::default()), model_a.clone()).unwrap();
    let adapter_b =
        Adapter::open_with_model(config(&b, 2000), Arc::new(LocalEmbedder::default()), model_b.clone()).unwrap();
    let mut both = add("u", "together", "s", "My sister Mira moved recently.");
    let next = json!({"role":"user","timestamp":1683590400000_i64,"content":"She lives in Bergen."});
    both["messages"].as_array_mut().unwrap().push(next.clone());
    assert_eq!(call(&adapter_a, "/add", both.clone()).await.0, StatusCode::OK);
    let first = add("u", "first", "s", "My sister Mira moved recently.");
    assert_eq!(call(&adapter_b, "/add", first).await.0, StatusCode::OK);
    drop(adapter_b);
    let adapter_b =
        Adapter::open_with_model(config(&b, 2000), Arc::new(LocalEmbedder::default()), model_b.clone()).unwrap();
    let second = json!({"request_id":"second","session_id":"s","user_id":"u","messages":[next]});
    assert_eq!(call(&adapter_b, "/add", second.clone()).await.0, StatusCode::OK);
    let mut left: Vec<_> =
        model_a.requests.lock().unwrap().iter().filter(|r| r.task == "extract").map(|r| r.user.clone()).collect();
    let mut right: Vec<_> =
        model_b.requests.lock().unwrap().iter().filter(|r| r.task == "extract").map(|r| r.user.clone()).collect();
    left.sort();
    right.sort();
    assert_eq!(left, right, "batch boundaries must not change the model's extraction input");
    let count = model_b.requests.lock().unwrap().len();
    assert_eq!(call(&adapter_b, "/add", second).await.0, StatusCode::OK);
    assert_eq!(model_b.requests.lock().unwrap().len(), count, "successful replay cannot call a model");
    for adapter in [&adapter_a, &adapter_b] {
        let (status, response) =
            call(adapter, "/search", json!({"user_id":"u","query":"Where does Mira live?","top_k":100})).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert!(response.to_string().contains("Mira lives in Bergen."));
        assert!(Cl100kCounter.count(&serde_json::to_string(&response["data"]).unwrap()).unwrap() <= 2000);
    }
}
#[tokio::test]
async fn failed_model_phase_stays_invisible_and_retry_reuses_applied_work() {
    let dir = tempfile::tempdir().unwrap();
    let model = Arc::new(ContextModel::default());
    let adapter =
        Adapter::open_with_model(config(&dir, 2000), Arc::new(LocalEmbedder::default()), model.clone()).unwrap();
    assert_eq!(
        call(&adapter, "/add", add("u", "first", "s", "My sister Mira moved recently.")).await.0,
        StatusCode::OK
    );
    let before = call(&adapter, "/search", search("u", 100)).await.1;
    model.fail_summary.store(true, Ordering::SeqCst);
    let request = add("u", "failing", "s", "NEW_PENDING_MESSAGE");
    assert_eq!(call(&adapter, "/add", request.clone()).await.0, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        call(&adapter, "/search", search("u", 100)).await.1,
        before,
        "failed Add must not change published evidence"
    );
    assert_eq!(call(&adapter, "/add", add("u", "different", "s", "different")).await.0, StatusCode::CONFLICT);
    let calls = model.requests.lock().unwrap().len();
    drop(adapter);
    let adapter =
        Adapter::open_with_model(config(&dir, 2000), Arc::new(LocalEmbedder::default()), model.clone()).unwrap();
    assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
    assert_eq!(model.requests.lock().unwrap().len(), calls + 1, "only the failed summary should run again");
    let after = call(&adapter, "/search", search("u", 100)).await.1;
    assert!(after.to_string().contains("NEW_PENDING_MESSAGE"));
}

struct UnsupportedDerivation;
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for UnsupportedDerivation {
    fn id(&self) -> &str {
        "unsupported-derivation-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        Ok(
            json!({"entities":[],"facts":[{"subject":"Ada","predicate":"lives in","object":"Rome","object_is_entity":false,"statement":"Ada lives in Rome.","negated":false,"salience":7,
    "time":{"expression":null,"kind":"ongoing","anchor_turn":id},"evidence_turns":[id],
    "update":{"kind":"replace","previous_object":"Oslo","effective_expression":"1 March 2024","evidence_source":id}}]}),
        )
    }
}
#[tokio::test]
async fn unusable_derived_updates_are_recorded_after_bounded_repairs_while_raw_source_survives() {
    let dir = tempfile::tempdir().unwrap();
    let adapter = Adapter::open_with_model(
        config(&dir, 2000),
        Arc::new(LocalEmbedder::default()),
        Arc::new(UnsupportedDerivation),
    )
    .unwrap();
    assert_eq!(call(&adapter, "/add", add("u", "r", "s", "SOURCE_ONLY_MARKER")).await.0, StatusCode::OK);
    let (status, value) = call(&adapter, "/search", search("u", 100)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(value.to_string().contains("SOURCE_ONLY_MARKER"));
    assert!(!value.to_string().contains("Ada lives in Rome"));
    let path = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
        .unwrap();
    let db = licorice::rusqlite::Connection::open(&path).unwrap();
    assert_eq!(db.query_row("SELECT count(*) FROM aml_rejections", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    let reason: String = db.query_row("SELECT reason FROM aml_rejections", [], |r| r.get(0)).unwrap();
    assert_eq!(
        reason,
        "source validation failed after two repairs: an update effective_expression must copy exact text from a supporting current turn"
    );
    assert_eq!(db.query_row("SELECT count(*) FROM facts", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
    assert_eq!(
        db.query_row("SELECT count(*) FROM aml_models WHERE json_extract(request,'$.task')='extract'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        3
    );
    drop(db);
    assert_eq!(call(&adapter, "/add", add("u", "r", "s", "SOURCE_ONLY_MARKER")).await.0, StatusCode::OK);
    let db = licorice::rusqlite::Connection::open(path).unwrap();
    assert_eq!(db.query_row("SELECT reason FROM aml_rejections", [], |r| r.get::<_, String>(0)).unwrap(), reason);
    assert_eq!(
        db.query_row("SELECT count(*) FROM aml_models WHERE json_extract(request,'$.task')='extract'", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        3,
        "successful replay retains the original rejection cause and saved model work"
    );
}

struct SpeakerIdentityModel {
    calls: AtomicUsize,
    physical_object: bool,
    alias_proxy: bool,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for SpeakerIdentityModel {
    fn id(&self) -> &str {
        "speaker-identity-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        let repaired = request.id.contains(":repair:");
        let subject = if repaired || self.alias_proxy { "User" } else { "the speaker" };
        let entities = if self.physical_object {
            json!([{"name":"the speaker","kind":"object","aliases":[]}])
        } else if repaired {
            json!([{"name":"User","kind":"person","aliases":[]}])
        } else if self.alias_proxy {
            json!([{"name":"User","kind":"person","aliases":["the speaker"]}])
        } else {
            json!([])
        };
        Ok(
            json!({"entities":entities,"facts":[{"subject":subject,"predicate":"lives in","object":"Oslo","object_is_entity":false,
            "statement":format!("{subject} lives in Oslo."),"negated":false,"salience":7,"time":{"expression":null,"kind":"ongoing","anchor_turn":id},"update":null,"evidence_turns":[id]}]}),
        )
    }
}

#[tokio::test]
async fn ambiguous_person_speaker_proxies_are_repaired_but_typed_physical_speakers_are_preserved() {
    for (physical_object, alias_proxy) in [(false, false), (false, true), (true, false)] {
        let dir = tempfile::tempdir().unwrap();
        let model = Arc::new(SpeakerIdentityModel { calls: AtomicUsize::new(0), physical_object, alias_proxy });
        let adapter =
            Adapter::open_with_model(config(&dir, 2000), Arc::new(LocalEmbedder::default()), model.clone()).unwrap();
        let text = if physical_object { "The speaker is in Oslo." } else { "I live in Oslo." };
        assert_eq!(call(&adapter, "/add", add("u", "r", "s", text)).await.0, StatusCode::OK);
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
            .unwrap();
        let db = licorice::rusqlite::Connection::open(path).unwrap();
        let subject: String =
            db.query_row("SELECT e.name FROM facts f JOIN entities e ON e.id=f.subject_id", [], |r| r.get(0)).unwrap();
        assert_eq!(subject, if physical_object { "the speaker" } else { "User" });
        assert_eq!(model.calls.load(Ordering::SeqCst), if physical_object { 1 } else { 2 });
        assert_eq!(db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    }
}

struct SourceReferenceModel {
    invalid: bool,
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for SourceReferenceModel {
    fn id(&self) -> &str {
        "source-reference-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        let fact = &request.schema.as_ref().unwrap()["properties"]["facts"]["items"];
        let update = &fact["properties"]["update"]["anyOf"][1];
        assert_eq!(update["properties"]["evidence_source"]["enum"], json!([id]));
        assert!(update["properties"].get("evidence_quote").is_none());
        assert_eq!(fact["properties"]["time"]["properties"]["anchor_turn"]["enum"], json!([id]));
        assert_eq!(fact["properties"]["evidence_turns"]["items"]["enum"], json!([id]));
        let reference = if self.invalid { "earlier-or-invented-source" } else { id };
        Ok(
            json!({"entities":[],"facts":[{"subject":"User","predicate":"lives in","object":"Rome","object_is_entity":false,"statement":"User lives in Rome.","negated":false,"salience":8,
            "time":{"expression":null,"kind":"ongoing","anchor_turn":id},"evidence_turns":[id],"update":{"kind":"replace","previous_object":"Oslo","effective_expression":null,"evidence_source":reference}}]}),
        )
    }
}
#[tokio::test]
async fn update_source_references_bind_native_quotes_without_generating_or_guessing_text() {
    for invalid in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let model = Arc::new(SourceReferenceModel { invalid, calls: AtomicUsize::new(0) });
        let adapter =
            Adapter::open_with_model(config(&dir, 2000), Arc::new(LocalEmbedder::default()), model.clone()).unwrap();
        let source = "I moved from Oslo to Rome. I love café music 🎵.";
        assert_eq!(
            call(&adapter, "/add", add("u", "r", "s", source)).await.0,
            if invalid { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK }
        );
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
            .unwrap();
        let db = licorice::rusqlite::Connection::open(&path).unwrap();
        assert_eq!(db.query_row("SELECT text FROM turns", [], |r| r.get::<_, String>(0)).unwrap(), source);
        if invalid {
            assert_eq!(
                model.calls.load(Ordering::SeqCst),
                1,
                "schema-invalid provider output fails before native application"
            );
            assert_eq!(db.query_row("SELECT count(*) FROM fact_updates", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
            assert_eq!(db.query_row("SELECT count(*) FROM aml_models", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
            assert!(path.to_string_lossy().contains(".pending."), "raw source remains staged for an identical retry");
        } else {
            assert_eq!(model.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                db.query_row("SELECT evidence_quote FROM fact_updates", [], |r| r.get::<_, String>(0)).unwrap(),
                source
            );
            let raw: String = db
                .query_row("SELECT output FROM aml_models WHERE json_extract(request,'$.task')='extract'", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert!(raw.contains("evidence_source"));
            assert!(!raw.contains("evidence_quote"));
        }
    }
}

struct CanonicalActorModel {
    wrong_kind: bool,
    decorated: bool,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for CanonicalActorModel {
    fn id(&self) -> &str {
        "canonical-actor-kind-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        let moved = current.contains("moved from Oslo to Rome");
        let actor = if self.decorated && !moved && !request.id.contains(":repair:") { "The User." } else { "User" };
        let entities = if moved || request.id.contains(":repair:") {
            json!([{"name":"User","kind":"person","aliases":[]}])
        } else if self.wrong_kind {
            json!([{"name":actor,"kind":"concept","aliases":[]}])
        } else {
            json!([])
        };
        let place = if moved { "Rome" } else { "Oslo" };
        Ok(
            json!({"entities":entities,"facts":[{"subject":actor,"predicate":"lives in","object":place,"object_is_entity":false,"statement":format!("User lives in {place}."),"negated":false,"salience":8,
            "time":{"expression":null,"kind":"ongoing","anchor_turn":id},"evidence_turns":[id],"update":if moved {json!({"kind":"replace","previous_object":"Oslo","effective_expression":null,"evidence_source":id})} else {Value::Null}}]}),
        )
    }
}
#[tokio::test]
async fn canonical_actor_identity_survives_missing_or_changed_model_entity_kinds() {
    for (wrong_kind, decorated) in [(false, false), (true, false), (false, true), (true, true)] {
        let dir = tempfile::tempdir().unwrap();
        let adapter = Adapter::open_with_model(
            config(&dir, 2000),
            Arc::new(LocalEmbedder::default()),
            Arc::new(CanonicalActorModel { wrong_kind, decorated }),
        )
        .unwrap();
        assert_eq!(call(&adapter, "/add", add("u", "first", "s", "I live in Oslo.")).await.0, StatusCode::OK);
        assert_eq!(call(&adapter, "/add", add("u", "move", "s", "I moved from Oslo to Rome.")).await.0, StatusCode::OK);
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
            .unwrap();
        let db = licorice::rusqlite::Connection::open(path).unwrap();
        assert_eq!(
            db.query_row("SELECT count(DISTINCT subject_id) FROM facts", [], |r| r.get::<_, i64>(0)).unwrap(),
            1,
            "the same canonical name must keep the same entity ID across omitted/declared kind"
        );
        assert_eq!(
            db.query_row("SELECT kind FROM entities WHERE name='User'", [], |r| r.get::<_, String>(0)).unwrap(),
            "person"
        );
        assert_eq!(
            db.query_row("SELECT status FROM facts WHERE object_value='Oslo'", [], |r| r.get::<_, String>(0)).unwrap(),
            "superseded"
        );
        assert_eq!(
            db.query_row("SELECT status FROM facts WHERE object_value='Rome'", [], |r| r.get::<_, String>(0)).unwrap(),
            "active"
        );
    }
}

struct CrashPhaseModel {
    root: std::path::PathBuf,
    stall: bool,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for CrashPhaseModel {
    fn id(&self) -> &str {
        "process-death-phase-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        use std::io::Write;
        let mut calls =
            std::fs::OpenOptions::new().create(true).append(true).open(self.root.join("model-calls.jsonl")).unwrap();
        writeln!(calls, "{}", serde_json::to_string(&request.id).unwrap()).unwrap();
        calls.sync_all().unwrap();
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        if self.stall && current.contains("WAIT_FOR_PARENT") {
            std::fs::write(self.root.join("waiting-for-parent"), b"waiting").unwrap();
            std::future::pending::<()>().await;
        }
        let moved = current.contains("moved from Oslo to Rome");
        let (predicate, object, statement) = if moved {
            ("lives in", "Rome", "User lives in Rome.")
        } else if current.contains("WAIT_FOR_PARENT") {
            ("owns", "bicycle", "User owns a green bicycle.")
        } else {
            ("lives in", "Oslo", "User lives in Oslo.")
        };
        Ok(
            json!({"entities":[],"facts":[{"subject":"User","predicate":predicate,"object":object,"object_is_entity":false,"statement":statement,"negated":false,"salience":8,
            "time":{"expression":null,"kind":"ongoing","anchor_turn":id},"evidence_turns":[id],"update":if moved {json!({"kind":"replace","previous_object":"Oslo","effective_expression":null,"evidence_source":id})} else {Value::Null}}]}),
        )
    }
}
fn crash_phase_config(root: &std::path::Path) -> Config {
    Config { api_key: "synthetic-test-secret".into(), data_dir: root.into(), budget_tokens: 2000 }
}
fn crash_phase_add() -> Value {
    json!({"request_id":"interrupted-add","user_id":"u","session_id":"s","messages":[
        {"role":"user","timestamp":1683590400000i64,"content":"I moved from Oslo to Rome."},
        {"role":"user","timestamp":1683676800000i64,"content":"WAIT_FOR_PARENT: I own a green bicycle."}]})
}
#[tokio::test]
async fn process_death_during_model_prefetch_keeps_publication_atomic_and_reuses_saved_outputs() {
    const CHILD: &str = "_LICORICE_AML_CRASH_PHASE_CHILD";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = std::path::PathBuf::from(root);
        let adapter = Adapter::open_with_model(
            crash_phase_config(&root),
            Arc::new(LocalEmbedder::default()),
            Arc::new(CrashPhaseModel { root: root.clone(), stall: true }),
        )
        .unwrap();
        let _ = call(&adapter, "/add", crash_phase_add()).await;
        panic!("controlled child must be killed while its second model response is pending");
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let adapter = Adapter::open_with_model(
        crash_phase_config(root),
        Arc::new(LocalEmbedder::default()),
        Arc::new(CrashPhaseModel { root: root.into(), stall: false }),
    )
    .unwrap();
    assert_eq!(call(&adapter, "/add", add("u", "base", "s", "I live in Oslo.")).await.0, StatusCode::OK);
    drop(adapter);
    let main = std::fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
        .unwrap();
    let pending = main.with_extension("pending.sqlite3");
    struct ChildGuard(std::process::Child);
    impl Drop for ChildGuard {
        fn drop(&mut self) {
            if matches!(self.0.try_wait(), Ok(None)) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }
    let log = std::fs::File::create(root.join("child.log")).unwrap();
    let mut child = ChildGuard(
        std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "process_death_during_model_prefetch_keeps_publication_atomic_and_reuses_saved_outputs",
                "--nocapture",
            ])
            .env(CHILD, root)
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .unwrap(),
    );
    let started = std::time::Instant::now();
    loop {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "child failed before controlled interruption; inspect child.log"
        );
        let saved = if pending.exists() {
            licorice::rusqlite::Connection::open(&pending)
                .ok()
                .and_then(|db| {
                    db.query_row(
                        "SELECT count(*) FROM aml_models WHERE json_extract(request,'$.task')='extract'",
                        [],
                        |r| r.get::<_, i64>(0),
                    )
                    .ok()
                })
                .is_some_and(|n| n >= 2)
        } else {
            false
        };
        if saved && root.join("waiting-for-parent").exists() {
            break;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(20), "no saved first phase before interruption");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    child.0.kill().unwrap();
    let status = child.0.wait().unwrap();
    assert!(!status.success());
    let db = licorice::rusqlite::Connection::open(&main).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0)).unwrap(),
        1,
        "the incomplete batch must stay unpublished"
    );
    drop(db);
    let adapter = Adapter::open_with_model(
        crash_phase_config(root),
        Arc::new(LocalEmbedder::default()),
        Arc::new(CrashPhaseModel { root: root.into(), stall: false }),
    )
    .unwrap();
    assert_eq!(call(&adapter, "/add", crash_phase_add()).await.0, StatusCode::OK);
    let before = std::fs::read_to_string(root.join("model-calls.jsonl")).unwrap();
    assert_eq!(call(&adapter, "/add", crash_phase_add()).await.0, StatusCode::OK);
    assert_eq!(
        std::fs::read_to_string(root.join("model-calls.jsonl")).unwrap(),
        before,
        "a completed replay cannot issue more model calls"
    );
    let db = licorice::rusqlite::Connection::open(&main).unwrap();
    assert_eq!(db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0)).unwrap(), 3);
    assert_eq!(db.query_row("SELECT count(*) FROM fact_updates", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    assert_eq!(
        db.query_row("SELECT status FROM facts WHERE object_value='Oslo'", [], |r| r.get::<_, String>(0)).unwrap(),
        "superseded"
    );
    assert!(!pending.exists());
    let calls: Vec<String> = before.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let extract: Vec<_> = calls.iter().filter(|id| id.starts_with("extract:")).collect();
    assert_eq!(extract.len(), 4, "base once, first interrupted phase once, second interrupted phase twice");
    let counts = extract.iter().fold(std::collections::HashMap::new(), |mut m, id| {
        *m.entry(id).or_insert(0) += 1;
        m
    });
    let mut counts: Vec<_> = counts.into_values().collect();
    counts.sort();
    assert_eq!(counts, vec![1, 1, 2]);
}

struct ManyFactsModel;
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for ManyFactsModel {
    fn id(&self) -> &str {
        "sixty-candidate-http-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        if request.task == "summarize" {
            return Ok(json!({"keys":[],"themes":[]}));
        }
        let current = request.user.split("Turns to extract from:\n").nth(1).unwrap().lines().next().unwrap();
        let id = current.split_once('[').unwrap().1.split_once(']').unwrap().0;
        Ok(
            json!({"entities":[],"facts":(0..64).map(|i|json!({"subject":"User","predicate":"likes","object":format!("item{i}"),"object_is_entity":false,"statement":format!("User likes item{i}."),"negated":false,"salience":8,"time":{"expression":null,"kind":"ongoing","anchor_turn":id},"evidence_turns":[id],"update":null})).collect::<Vec<_>>()}),
        )
    }
}
#[derive(Default)]
struct HttpScorerPool(std::sync::Mutex<Vec<usize>>);
#[async_trait::async_trait]
impl licorice_providers::rerank::Reranker for HttpScorerPool {
    fn id(&self) -> &str {
        "http-pool-control"
    }
    async fn rank(
        &self,
        _: &str,
        rows: &[licorice_providers::rerank::Candidate],
    ) -> Result<Vec<licorice_providers::rerank::Ranked>, licorice_providers::chat::ChatError> {
        assert!(rows.len() <= 24);
        self.0.lock().unwrap().push(rows.len());
        Ok(rows.iter().map(|r| licorice_providers::rerank::Ranked { id: r.id, score: 0.5 }).collect())
    }
}
#[tokio::test]
async fn deeper_scorer_search_keeps_per_call_limits_and_the_entire_wire_budget() {
    let dir = tempfile::tempdir().unwrap();
    let scorer = Arc::new(HttpScorerPool::default());
    let policy = licorice::retrieve::RetrieveConfig {
        budget_tokens: 600,
        rerank_candidates: 60,
        graph_hops: 2,
        connected_evidence: true,
        ..Default::default()
    };
    let adapter =
        Adapter::open_with_model(config(&dir, 600), Arc::new(LocalEmbedder::default()), Arc::new(ManyFactsModel))
            .unwrap()
            .with_search_policy(policy, Some(Arc::new(licorice_providers::rerank::BatchedReranker(scorer.clone()))))
            .unwrap();
    let text = format!("I like {}.", (0..64).map(|i| format!("item{i}")).collect::<Vec<_>>().join(", "));
    assert_eq!(call(&adapter, "/add", add("u", "r", "s", &text)).await.0, StatusCode::OK);
    let (status, response) =
        call(&adapter, "/search", json!({"user_id":"u","query":"What do I like?","top_k":100})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(data(&response).len(), 1);
    let wire = serde_json::to_string(data(&response)).unwrap();
    assert!(Cl100kCounter.count(&wire).unwrap() <= 600);
    assert!(response.to_string().contains("User likes item"));
    let batches = scorer.0.lock().unwrap();
    assert!(!batches.is_empty());
    assert_eq!(batches.len() % 3, 0);
    for batch in batches.chunks(3) {
        let mut lengths = batch.to_vec();
        lengths.sort();
        assert_eq!(lengths, vec![12, 24, 24]);
    }
    for pool in [0, 97] {
        let dir = tempfile::tempdir().unwrap();
        let adapter = Adapter::open(config(&dir, 600), Arc::new(LocalEmbedder::default())).unwrap();
        assert!(
            adapter
                .with_search_policy(
                    licorice::retrieve::RetrieveConfig {
                        budget_tokens: 600,
                        rerank_candidates: pool,
                        ..Default::default()
                    },
                    None
                )
                .is_err()
        );
    }
}

#[tokio::test]
async fn a_fitting_temporal_search_retrieves_once_before_wire_validation() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = Arc::new(LocalEmbedder::default());
    let scorer = Arc::new(HttpScorerPool::default());
    let adapter = Adapter::open_with_model(
        config(&dir, 2000),
        embedder.clone(),
        Arc::new(CanonicalActorModel { wrong_kind: false, decorated: false }),
    )
    .unwrap()
    .with_search_policy(licorice::retrieve::RetrieveConfig::default(), Some(scorer.clone()))
    .unwrap();
    assert_eq!(call(&adapter, "/add", add("u", "r", "s", "I live in Oslo.")).await.0, StatusCode::OK);
    let before = embedder.calls.load(Ordering::SeqCst);
    let (status, response) =
        call(&adapter, "/search", json!({"user_id":"u","query":"Where do I live?","top_k":100})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(data(&response).len(), 1);
    assert!(response.to_string().contains("Oslo"));
    assert_eq!(
        embedder.calls.load(Ordering::SeqCst) - before,
        1,
        "a fitting context must not repeat embedding, graph retrieval or paid scoring"
    );
    assert_eq!(*scorer.0.lock().unwrap(), vec![1], "a fitting Search must score its candidate pool once");
    assert!(Cl100kCounter.count(&serde_json::to_string(data(&response)).unwrap()).unwrap() <= 2000);
}

#[tokio::test]
async fn wire_reduction_that_excludes_all_source_returns_no_header_only_memory() {
    let dir = tempfile::tempdir().unwrap();
    let embedder = Arc::new(LocalEmbedder::default());
    let adapter =
        Adapter::open_with_model(config(&dir, 200), embedder.clone(), Arc::new(ContextModel::default())).unwrap();
    // Backslashes fit cheaply as native text but expand when JSON-escaped.
    let text = "\\".repeat(440);
    assert_eq!(call(&adapter, "/add", add("u", "r", "s", &text)).await.0, StatusCode::OK);
    let before = embedder.calls.load(Ordering::SeqCst);
    let (status, response) = call(&adapter, "/search", search("u", 100)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(embedder.calls.load(Ordering::SeqCst) - before >= 2, "wire escaping must exercise native-budget reduction");
    assert_eq!(
        response,
        json!({"data":[]}),
        "an empty evidence selection must not publish its decorative header as a memory"
    );
}

struct OutputCapacityModel {
    extraction: bool,
    summary: bool,
    never_fits: bool,
    calls: std::sync::Mutex<Vec<(String, u32)>>,
}
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for OutputCapacityModel {
    fn id(&self) -> &str {
        "offline-output-capacity-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        self.calls.lock().unwrap().push((request.task.clone(), request.max_tokens));
        let limited = if request.task == "summarize" { self.summary } else { self.extraction };
        if limited && (self.never_fits || request.max_tokens < 8192) {
            return Err(licorice_providers::chat::ChatError("model output token limit reached".into()));
        }
        CanonicalActorModel { wrong_kind: false, decorated: false }.complete(request).await
    }
}
#[tokio::test]
async fn output_capacity_repairs_are_bounded_and_preserve_whole_source_when_derivations_cannot_fit() {
    for (extraction, summary, never_fits) in
        [(true, false, false), (true, false, true), (false, true, false), (false, true, true)]
    {
        let dir = tempfile::tempdir().unwrap();
        let model = Arc::new(OutputCapacityModel { extraction, summary, never_fits, calls: Default::default() });
        let adapter =
            Adapter::open_with_model(config(&dir, 2000), Arc::new(LocalEmbedder::default()), model.clone()).unwrap();
        let request = add("u", "r", "s", "I live in Oslo.");
        let (status, response) = call(&adapter, "/add", request.clone()).await;
        assert_eq!(status, StatusCode::OK, "known output capacity must not permanently block Add: {response}");
        let before = model.calls.lock().unwrap().len();
        assert_eq!(call(&adapter, "/add", request).await.0, StatusCode::OK);
        assert_eq!(model.calls.lock().unwrap().len(), before);
        let (status, response) =
            call(&adapter, "/search", json!({"user_id":"u","query":"Where do I live?","top_k":100})).await;
        assert_eq!(status, StatusCode::OK);
        assert!(response.to_string().contains("Oslo"));
        let path = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "sqlite3"))
            .unwrap();
        let db = licorice::rusqlite::Connection::open(path).unwrap();
        assert_eq!(db.query_row("SELECT count(*) FROM turns", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
        assert_eq!(
            db.query_row("SELECT count(*) FROM facts", [], |r| r.get::<_, i64>(0)).unwrap(),
            if extraction && never_fits { 0 } else { 1 }
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM aml_rejections", [], |r| r.get::<_, i64>(0)).unwrap(),
            i64::from(never_fits)
        );
        let calls = model.calls.lock().unwrap();
        let limited_task = if extraction { "extract" } else { "summarize" };
        let caps: Vec<_> = calls.iter().filter(|(task, _)| task == limited_task).map(|(_, cap)| *cap).collect();
        assert_eq!(caps.len(), 2, "one original call and one8192-token capacity repair");
        assert!(caps[0] < 8192);
        assert_eq!(caps[1], 8192);
    }
}
