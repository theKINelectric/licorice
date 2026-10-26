use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_service::{Config, Service};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tower::ServiceExt;

const ORIGIN: &str = "http://localhost:8080";

struct SynonymModel(Mutex<Vec<String>>, bool);
#[async_trait::async_trait]
impl Embedder for SynonymModel {
    fn id(&self) -> &str {
        "test-synonym-model"
    }
    fn dim(&self) -> usize {
        2
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0.lock().unwrap().extend(texts.iter().cloned());
        if self.1 {
            return Err(EmbedError::Embed("deliberate failure".into()));
        }
        Ok(texts
            .iter()
            .map(|s| if s.contains("locomotive") || s.contains("railway") { vec![1., 0.] } else { vec![0., 1.] })
            .collect())
    }
}

#[tokio::test]
async fn semantic_service_recovers_synonym_and_filters_old_source_before_embedding() {
    let tmp = tempfile::tempdir().unwrap();
    let model = Arc::new(SynonymModel(Mutex::new(vec![]), false));
    let mut c = config(tmp.path());
    c.embedder = Some(model.clone());
    let service = Service::open(c).unwrap();
    let app = service.router();
    let (key, _, state) = workspace(&app).await;
    assert_eq!(state["retrieval"]["mode"], "hybrid_semantic");
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"old","text":"The locomotive departs Tuesday."})),
    )
    .await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"current","text":"The locomotive departs Thursday.","supersedes":"human:old"})),
    )
    .await;
    let (status, packet, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"railway timetable"}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(packet["source_ids"], json!(["human:current"]));
    assert!(packet["text"].as_str().unwrap().contains("Thursday"));
    assert!(!packet["text"].as_str().unwrap().contains("Tuesday"));
    assert!(packet["usage"]["tokens"].as_u64().unwrap() <= 2000);
    assert!(
        !model.0.lock().unwrap().iter().any(|s| s.contains("Tuesday")),
        "stale source must not enter semantic embedding batch"
    );
}

#[tokio::test]
async fn configured_model_failure_is_not_mislabeled_lexical_success() {
    let tmp = tempfile::tempdir().unwrap();
    let mut c = config(tmp.path());
    c.embedder = Some(Arc::new(SynonymModel(Mutex::new(vec![]), true)));
    let app = Service::open(c).unwrap().router();
    let (key, _, _) = workspace(&app).await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"one","text":"The locomotive departs Tuesday."})),
    )
    .await;
    let (status, body, _) = call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"locomotive"}))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert!(body.get("text").is_none());
}

#[tokio::test]
async fn explicit_enumeration_obeys_semantic_eligibility() {
    let tmp = tempfile::tempdir().unwrap();
    let model = Arc::new(SynonymModel(Mutex::new(vec![]), false));
    let mut c = config(tmp.path());
    c.embedder = Some(model.clone());
    let app = Service::open(c).unwrap().router();
    let (key, _, _) = workspace(&app).await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"one","text":"Lina handles the locomotive timetable."})),
    )
    .await;
    let (status, body, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"List all locomotive duties"}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["source_ids"], json!(["human:one"]));
    assert_eq!(body["retrieval"]["mode"], "hybrid_semantic");
    assert!(body.get("semantic_search").is_some());
    assert!(!model.0.lock().unwrap().is_empty());
}
fn config(path: &std::path::Path) -> Config {
    let mut c = Config::new(path, ORIGIN);
    c.secure_cookie = false;
    c
}
async fn call(
    app: &Router,
    method: &str,
    path: &str,
    credential: Option<&str>,
    body: Option<Value>,
) -> (StatusCode, Value, Option<String>) {
    let mut req = Request::builder().method(method).uri(path).header("origin", ORIGIN);
    if let Some(credential) = credential {
        if credential.starts_with("licorice_trial=") {
            req = req.header("cookie", credential);
        } else {
            req = req.header("authorization", format!("Bearer {credential}"));
        }
    }
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let response = app
        .clone()
        .oneshot(req.body(Body::from(body.map(|v| v.to_string()).unwrap_or_default())).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let cookie =
        response.headers().get("set-cookie").map(|v| v.to_str().unwrap().split(';').next().unwrap().to_owned());
    let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null), cookie)
}
async fn workspace(app: &Router) -> (String, String, Value) {
    let (status, state, cookie) = call(app, "POST", "/api/v1/workspaces", None, None).await;
    assert_eq!(status, StatusCode::OK, "{state}");
    (state["api_key"].as_str().unwrap().to_owned(), cookie.unwrap(), state)
}

#[tokio::test]
async fn own_memory_survives_session_and_restart_with_idempotency() {
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (key, _, initial) = workspace(&app).await;
    let input = json!({"request_id":"memory-1","text":"My favorite launch snack is a blue plum."});
    let (status, first, _) = call(&app, "POST", "/api/v1/observations", Some(&key), Some(input.clone())).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (_, repeat, _) = call(&app, "POST", "/api/v1/observations", Some(&key), Some(input)).await;
    assert_eq!(repeat, first);
    let (status, _, _) = call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"memory-1","text":"Different"})),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT);
    let (_, session, _) =
        call(&app, "POST", "/api/v1/sessions", Some(&key), Some(json!({"request_id":"session-1"}))).await;
    assert_ne!(session["conversation_id"], initial["conversation_id"]);
    let (_, recall, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"favorite launch snack"}))).await;
    assert!(recall["text"].as_str().unwrap().contains("blue plum"));
    assert_eq!(recall["source_ids"], json!(["human:memory-1"]));
    drop(app);
    drop(service);
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (_, recalled, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"favorite launch snack"}))).await;
    assert_eq!(recall, recalled);
    let (_, history, _) = call(&app, "GET", "/api/v1/history", Some(&key), None).await;
    assert_eq!(history["history"].as_array().unwrap().len(), 1);
    assert_eq!(history["history"][0]["actor"]["role"], "human");
    let (_, state, _) = call(&app, "GET", "/api/v1/workspace", Some(&key), None).await;
    assert!(state["memory"]["claims"].as_array().unwrap().is_empty());
    assert_eq!(state["memory"]["observations"][0]["observation"]["origin"], "human_report");
}

#[tokio::test]
async fn isolation_key_rotation_and_deletion_are_enforced() {
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (a, cookie, _) = workspace(&app).await;
    let (b, _, _) = workspace(&app).await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&a),
        Some(json!({"request_id":"a","text":"Secret violet marmalade"})),
    )
    .await;
    let (_, recall, _) =
        call(&app, "POST", "/api/v1/recall", Some(&b), Some(json!({"query":"violet marmalade"}))).await;
    assert!(!recall["text"].as_str().unwrap().contains("Secret violet"));
    let (_, rotated, _) = call(&app, "POST", "/api/v1/keys", Some(&cookie), None).await;
    let new = rotated["api_key"].as_str().unwrap();
    assert_ne!(new, a);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&a), None).await.0, StatusCode::UNAUTHORIZED);
    let (_, state, _) = call(&app, "GET", "/api/v1/workspace", Some(new), None).await;
    assert_eq!(state["memory"]["observations"].as_array().unwrap().len(), 1);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&cookie), None).await.0, StatusCode::OK);
    let (_, export, _) = call(&app, "GET", "/api/v1/export", Some(new), None).await;
    assert!(!export.to_string().contains(new));
    assert_eq!(export["history"].as_array().unwrap().len(), 1);
    assert_eq!(call(&app, "DELETE", "/api/v1/workspace", Some(new), None).await.0, StatusCode::OK);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&cookie), None).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(new), None).await.0, StatusCode::UNAUTHORIZED);
    assert_eq!(std::fs::read_dir(tmp.path().join("trials")).unwrap().count(), 1);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&b), None).await.0, StatusCode::OK);
}

#[tokio::test]
async fn explicit_updates_preserve_history_but_recall_only_current_sources() {
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (key, _, _) = workspace(&app).await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"first","text":"My favorite color is red."})),
    )
    .await;
    let update = json!({"request_id":"updated","text":"My favorite color is green.","supersedes":"human:first"});
    let (status, receipt, _) = call(&app, "POST", "/api/v1/observations", Some(&key), Some(update.clone())).await;
    assert_eq!(status, StatusCode::OK, "{receipt}");
    assert_eq!(call(&app, "POST", "/api/v1/observations", Some(&key), Some(update)).await.1, receipt);
    let (_, recall, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"favorite color"}))).await;
    let text = recall["text"].as_str().unwrap();
    assert!(text.contains("green"));
    assert!(!text.contains("color is red"));
    assert_eq!(recall["supersessions"], json!([["human:first", "human:updated"]]));
    let (_, state, _) = call(&app, "GET", "/api/v1/workspace", Some(&key), None).await;
    assert_eq!(state["memory"]["observations"].as_array().unwrap().len(), 2);
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&key),
            Some(json!({"request_id":"bad-update","text":"Nope","supersedes":"human:first"}))
        )
        .await
        .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
    let (b, _, _) = workspace(&app).await;
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&b),
            Some(json!({"request_id":"foreign","text":"Nope","supersedes":"human:updated"}))
        )
        .await
        .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn native_update_remains_atomic_when_application_receipt_is_missing() {
    use licorice::{
        Store, StoreConfig,
        being::{Actor, ActorRole, Memory, Observation, Operation, Origin, Request as NativeRequest, Source},
    };
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (key, cookie, _) = workspace(&app).await;
    call(
        &app,
        "POST",
        "/api/v1/observations",
        Some(&key),
        Some(json!({"request_id":"first","text":"My favorite color is red."})),
    )
    .await;
    let raw = cookie.split('=').nth(1).unwrap();
    let hash = blake3::hash(raw.as_bytes()).to_hex().to_string();
    let mut cfg = StoreConfig::new(1);
    cfg.readers = 1;
    let store = Store::open(tmp.path().join("trials").join(hash).join("memory.sqlite"), cfg).await.unwrap();
    let memory = Memory::attach(&store, "trial-being").await.unwrap();
    // Simulate the only durable native write surviving a crash before the application receipt.
    memory
        .apply(
            Actor { principal: "trial-visitor".into(), role: ActorRole::Human, session: "interrupted-session".into() },
            NativeRequest {
                request_id: "human:interrupted-update".into(),
                expected_revision: None,
                operation: Operation::Observe(Observation {
                    origin: Origin::HumanReport,
                    subject: "trial-visitor".into(),
                    text: "My favorite color is amber.".into(),
                    relationship_id: None,
                    occurred_at: None,
                    source: Source::Reported {
                        reference: Some(
                            json!({"session":"interrupted-session","supersedes":"human:first"}).to_string(),
                        ),
                    },
                }),
            },
        )
        .await
        .unwrap();
    store.close().await.unwrap();
    drop(app);
    drop(service);
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (_, recall, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"favorite color"}))).await;
    let text = recall["text"].as_str().unwrap();
    assert!(text.contains("amber"));
    assert!(!text.contains("color is red"));
    assert_eq!(recall["source_ids"], json!(["human:interrupted-update"]));
}

#[tokio::test]
async fn bounds_origin_roles_and_capacity_are_checked() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(tmp.path());
    cfg.max_observations = 2;
    cfg.max_trials = 1;
    cfg.max_trial_creations = 1;
    let service = Service::open(cfg).unwrap();
    let app = service.router();
    let (key, _, _) = workspace(&app).await;
    let bad = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/observations")
                .header("origin", "https://evil.example")
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(Body::from("{} "))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(bad.status(), StatusCode::FORBIDDEN);
    let none = app
        .clone()
        .oneshot(Request::builder().method("POST").uri("/api/v1/workspaces").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(none.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&key),
            Some(json!({"request_id":"spoof","text":"I rule","role":"agent","path":"../../secrets"}))
        )
        .await
        .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&key),
            Some(json!({"request_id":"large","text":"🍌".repeat(1001)}))
        )
        .await
        .0,
        StatusCode::UNPROCESSABLE_ENTITY
    );
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&key),
            Some(json!({"request_id":"oversized","text":"x".repeat(9000)}))
        )
        .await
        .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
    for n in 0..2 {
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/observations",
                Some(&key),
                Some(json!({"request_id":format!("m-{n}"),"text":"A violet berry"}))
            )
            .await
            .0,
            StatusCode::OK
        );
    }
    assert_eq!(
        call(
            &app,
            "POST",
            "/api/v1/observations",
            Some(&key),
            Some(json!({"request_id":"third","text":"A violet berry"}))
        )
        .await
        .0,
        StatusCode::TOO_MANY_REQUESTS
    );
    assert_eq!(call(&app, "POST", "/api/v1/workspaces", None, None).await.0, StatusCode::TOO_MANY_REQUESTS);
    assert!(Service::open(config(tmp.path())).is_err());
    call(&app, "DELETE", "/api/v1/workspace", Some(&key), None).await;
    assert_eq!(call(&app, "POST", "/api/v1/workspaces", None, None).await.0, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn expiry_cleanup_removes_private_disk_data() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(tmp.path());
    cfg.retention = Duration::from_millis(100);
    let service = Service::open(cfg).unwrap();
    let app = service.router();
    let (key, _, _) = workspace(&app).await;
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&key), None).await.0, StatusCode::GONE);
    assert_eq!(service.cleanup_expired().await.unwrap(), 1);
    assert_eq!(std::fs::read_dir(tmp.path().join("trials")).unwrap().count(), 0);
    assert_eq!(call(&app, "GET", "/api/v1/workspace", Some(&key), None).await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn complete_rendered_recall_never_exceeds_two_thousand_tokens() {
    use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (key, _, _) = workspace(&app).await;
    for n in 0..12 {
        let text = format!(
            "Violet trial memory {n}: {}",
            "many important facts about bananas and violet berries. ".repeat(60)
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/api/v1/observations",
                Some(&key),
                Some(json!({"request_id":format!("m-{n}"),"text":text}))
            )
            .await
            .0,
            StatusCode::OK
        );
    }
    let (_, recall, _) =
        call(&app, "POST", "/api/v1/recall", Some(&key), Some(json!({"query":"violet berries"}))).await;
    let tokens = Cl100kCounter.count(recall["text"].as_str().unwrap()).unwrap();
    assert!(tokens <= 2000);
    assert_eq!(tokens as u64, recall["usage"]["tokens"].as_u64().unwrap());
    assert!(!recall["omitted_unit_ids"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn bearer_cli_does_not_need_browser_origin_and_no_model_exists() {
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    let app = service.router();
    let (key, _, _) = workspace(&app).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/observations")
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(Body::from(json!({"request_id":"cli","text":"My favorite fruit is blueberry"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        call(&app, "POST", "/api/v1/messages", Some(&key), Some(json!({"request_id":"chat","message":"hello"})))
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let response = app.clone().oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().get("access-control-allow-origin").is_none());
    let mut insecure = config(tmp.path());
    insecure.public_origin = "http://example.com".into();
    assert!(Service::open(insecure).is_err());
}

#[tokio::test]
async fn bounded_operation_ledger_and_orphan_cleanup_survive_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let orphan = tmp.path().join("trials").join("a".repeat(64));
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("incomplete.sqlite"), b"abandoned").unwrap();
    let unrelated = tmp.path().join("trials").join("operator-files");
    std::fs::create_dir_all(&unrelated).unwrap();
    let service = Service::open(config(tmp.path())).unwrap();
    assert!(!orphan.exists());
    assert!(unrelated.exists());
    let app = service.router();
    let (key, _, state) = workspace(&app).await;
    assert_eq!(state["limits"]["operations"], 120);
    for n in 0..120 {
        let (status, body, _) =
            call(&app, "POST", "/api/v1/sessions", Some(&key), Some(json!({"request_id":format!("s-{n}")}))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    assert_eq!(
        call(&app, "POST", "/api/v1/sessions", Some(&key), Some(json!({"request_id":"over"}))).await.0,
        StatusCode::TOO_MANY_REQUESTS
    );
    // Completed replays still work at capacity.
    assert_eq!(
        call(&app, "POST", "/api/v1/sessions", Some(&key), Some(json!({"request_id":"s-0"}))).await.0,
        StatusCode::OK
    );
    std::fs::create_dir_all(&orphan).unwrap();
    assert_eq!(service.cleanup_expired().await.unwrap(), 0);
    assert!(!orphan.exists());
    assert!(unrelated.exists());
}

#[tokio::test]
async fn public_https_uses_host_only_cookie_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    let service = Service::open(Config::new(tmp.path(), "https://memory.example")).unwrap();
    let app = service.router();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/v1/workspaces")
                .header("origin", "https://memory.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let cookie = response.headers()["set-cookie"].to_str().unwrap();
    assert!(cookie.starts_with("__Host-licorice_trial="));
    assert!(cookie.contains("; Secure"));
    assert!(cookie.contains("; Path=/"));
    assert!(cookie.contains("; HttpOnly"));
    assert!(!cookie.contains("Domain="));
}
