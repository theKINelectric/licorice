use axum::{Json, Router, extract::State, routing::post};
use licorice_providers::chat::{ChatModel, ModelRequest};
use licorice_providers::chat_http::BudgetChat;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
async fn handler(State(calls): State<Arc<AtomicUsize>>, Json(body): Json<Value>) -> Json<Value> {
    calls.fetch_add(1, Ordering::SeqCst);
    assert_eq!(body["model"], "openai/gpt-4o-mini");
    assert_eq!(body["provider"]["allow_fallbacks"], false);
    Json(
        json!({"model":"openai/gpt-4o-mini-2024-07-18","choices":[{"finish_reason":"stop","message":{"content":"synthetic answer"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"cost":0.0000033}}),
    )
}
#[tokio::test]
async fn durable_cache_deduplicates_concurrent_calls_and_budget_admission_precedes_http() {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/chat", listener.local_addr().unwrap());
    let app = Router::new().route("/chat", post(handler)).with_state(calls.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spend.sqlite");
    let model = BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap();
    let req = ModelRequest {
        id: "one".into(),
        task: "answer".into(),
        system: "s".into(),
        user: "u".into(),
        schema: None,
        max_tokens: 30,
    };
    let (a, b) = tokio::join!(model.complete(&req), model.complete(&req));
    assert_eq!(a.unwrap(), json!("synthetic answer"));
    assert_eq!(b.unwrap(), json!("synthetic answer"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!((model.accounted().unwrap() - 0.0000033).abs() < 1e-12);
    drop(model);
    let model = BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap();
    assert!(model.complete(&req).await.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let tiny =
        BudgetChat::local_test("synthetic-key".into(), &dir.path().join("tiny.sqlite"), 0.000001, &endpoint).unwrap();
    assert!(tiny.complete(&req).await.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(tiny.accounted().unwrap(), 0.);
    server.abort();
}

async fn delayed_handler(state: State<Arc<AtomicUsize>>, body: Json<Value>) -> Json<Value> {
    tokio::time::sleep(std::time::Duration::from_millis(120)).await;
    handler(state, body).await
}

#[tokio::test]
async fn independent_clients_of_one_ledger_share_an_inflight_success() {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/chat", listener.local_addr().unwrap());
    let app = Router::new().route("/chat", post(delayed_handler)).with_state(calls.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.sqlite");
    let a = BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap();
    let b = BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap();
    let request = ModelRequest {
        id: "same-exact-input".into(),
        task: "answer".into(),
        system: "s".into(),
        user: "u".into(),
        schema: None,
        max_tokens: 30,
    };
    let (left, right) = tokio::join!(a.complete(&request), b.complete(&request));
    assert_eq!(left.unwrap(), right.unwrap());
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a shared ledger must deduplicate independent clients, not just one client's tasks"
    );
    let db = rusqlite::Connection::open(path).unwrap();
    assert_eq!(db.query_row("SELECT count(*) FROM calls", [], |r| r.get::<_, i64>(0)).unwrap(), 1);
    server.abort();
}

fn process_request(crash: bool) -> ModelRequest {
    ModelRequest {
        id: if crash { "crash-owner" } else { "shared-process-input" }.into(),
        task: "answer".into(),
        system: "s".into(),
        user: if crash { "crash-input" } else { "same input" }.into(),
        schema: None,
        max_tokens: 30,
    }
}
struct ProcessState {
    calls: AtomicUsize,
    crashed_once: AtomicBool,
    crash_started: tokio::sync::Notify,
}
async fn process_handler(State(state): State<Arc<ProcessState>>, Json(body): Json<Value>) -> Json<Value> {
    state.calls.fetch_add(1, Ordering::SeqCst);
    if body["messages"][1]["content"] == "crash-input" && !state.crashed_once.swap(true, Ordering::SeqCst) {
        state.crash_started.notify_one();
        std::future::pending::<()>().await;
    }
    tokio::time::sleep(Duration::from_millis(120)).await;
    Json(
        json!({"model":"openai/gpt-4o-mini","choices":[{"finish_reason":"stop","message":{"content":"synthetic answer"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"cost":0.0000033}}),
    )
}

#[tokio::test]
async fn cross_process_cache_claims_release_on_owner_death_and_retain_unknown_charges() {
    const CHILD: &str = "_LICORICE_CACHE_CONTROL_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let endpoint = std::env::var("_LICORICE_CACHE_CONTROL_ENDPOINT").unwrap();
        // Relative and absolute ledger spellings must acquire the same claim.
        let model =
            BudgetChat::local_test("synthetic-key".into(), std::path::Path::new("shared.sqlite"), 1., &endpoint)
                .unwrap();
        std::fs::write(format!("ready-{mode}"), b"ready").unwrap();
        for _ in 0..1000 {
            if std::path::Path::new("start").exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(std::path::Path::new("start").exists());
        assert_eq!(model.complete(&process_request(mode == "crash")).await.unwrap(), json!("synthetic answer"));
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.sqlite");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/chat", listener.local_addr().unwrap());
    let state = Arc::new(ProcessState {
        calls: AtomicUsize::new(0),
        crashed_once: AtomicBool::new(false),
        crash_started: tokio::sync::Notify::new(),
    });
    let app = Router::new().route("/chat", post(process_handler)).with_state(state.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    // Initialize schema once; the child test processes then race only on requests.
    drop(BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap());
    let spawn = |mode: &str| {
        let mut command = tokio::process::Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "cross_process_cache_claims_release_on_owner_death_and_retain_unknown_charges",
                "--nocapture",
            ])
            .env(CHILD, mode)
            .env("_LICORICE_CACHE_CONTROL_ENDPOINT", &endpoint)
            .current_dir(dir.path())
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        command.spawn().unwrap()
    };
    let a = spawn("a");
    let b = spawn("b");
    tokio::time::timeout(Duration::from_secs(10), async {
        while !dir.path().join("ready-a").exists() || !dir.path().join("ready-b").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    std::fs::write(dir.path().join("start"), b"start").unwrap();
    let (a, b) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(a.wait_with_output(), b.wait_with_output())
    })
    .await
    .unwrap();
    for output in [a.unwrap(), b.unwrap()] {
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    }
    assert_eq!(state.calls.load(Ordering::SeqCst), 1, "separate processes must share one successful request");
    let mut dying = spawn("crash");
    tokio::time::timeout(Duration::from_secs(10), state.crash_started.notified()).await.unwrap();
    dying.start_kill().unwrap();
    let terminal = dying.wait().await.unwrap();
    assert!(!terminal.success(), "owner death is established by process exit, not an observation timeout");
    let recovered = BudgetChat::local_test("synthetic-key".into(), &path, 1., &endpoint).unwrap();
    let answer = tokio::time::timeout(Duration::from_secs(10), recovered.complete(&process_request(true)))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(answer, json!("synthetic answer"));
    assert_eq!(state.calls.load(Ordering::SeqCst), 3);
    let db = rusqlite::Connection::open(path).unwrap();
    let (reservation, cost, error): (f64, Option<f64>, String) = db
        .query_row("SELECT reservation,cost,error FROM calls WHERE output IS NULL", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert!(reservation > 0. && cost.is_none() && error.contains("interrupted"));
    assert!((recovered.accounted().unwrap() - reservation - 0.0000066).abs() < 1e-12);
    assert!(recovered.complete(&process_request(false)).await.is_ok());
    assert_eq!(state.calls.load(Ordering::SeqCst), 3, "successful replay after the crash must stay cached");
    server.abort();
}

#[tokio::test]
async fn completed_output_limit_receipts_are_not_rebilled_and_larger_caps_have_their_own_identity() {
    use axum::{Json, Router, extract::State, routing::post};
    use licorice_providers::chat::{ChatModel, ModelRequest};
    use licorice_providers::chat_http::BudgetChat;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let app=Router::new().route("/chat",post(|State(calls):State<Arc<AtomicUsize>>,Json(body):Json<serde_json::Value>|async move {
        calls.fetch_add(1,Ordering::SeqCst);
        Json(serde_json::json!({"model":"openai/gpt-4o-mini","choices":[{"finish_reason":if body["max_tokens"]==4096 {"length"}else{"stop"},"message":{"content":"synthetic recovered output"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"cost":0.0000033}}))
    })).with_state(calls.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/chat", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.sqlite");
    let request = ModelRequest {
        id: "capacity".into(),
        task: "extract".into(),
        system: String::new(),
        user: "synthetic source".into(),
        schema: None,
        max_tokens: 4096,
    };
    let model = BudgetChat::local_test("synthetic".into(), &path, 1., &endpoint).unwrap();
    assert!(model.complete(&request).await.unwrap_err().is_output_limit());
    assert!(model.complete(&request).await.unwrap_err().is_output_limit());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(model);
    // Existing frozen v10 ledgers used this less specific error text.
    rusqlite::Connection::open(&path)
        .unwrap()
        .execute("UPDATE calls SET error='model response refused or incomplete'", [])
        .unwrap();
    let model = BudgetChat::local_test("synthetic".into(), &path, 1., &endpoint).unwrap();
    assert!(model.complete(&request).await.unwrap_err().is_output_limit());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let mut repaired = request;
    repaired.max_tokens = 8192;
    repaired.id.push_str(":capacity:8192");
    assert_eq!(model.complete(&repaired).await.unwrap(), serde_json::json!("synthetic recovered output"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!((model.accounted().unwrap() - 0.0000066).abs() < 1e-12);
    server.abort();
}
