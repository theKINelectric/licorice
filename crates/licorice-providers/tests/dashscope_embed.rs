//! Local-only contracts for board-specified embeddings. No live credentials.
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::post,
};
use licorice_providers::embed::{DashScopeV4, EmbedKind, Embedder};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct Mock {
    requests: Mutex<Vec<Value>>,
    mode: AtomicUsize,
    redirected: AtomicUsize,
}
async fn embedding(
    State(state): State<Arc<Mock>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> axum::response::Response {
    assert_eq!(headers.get("authorization").unwrap(), "Bearer synthetic-embedding-key");
    state.requests.lock().unwrap().push(body.clone());
    if state.mode.load(Ordering::SeqCst) == 3 {
        return (StatusCode::FOUND, [("location", "/leak")], "redirect").into_response();
    }
    assert_eq!(body["model"], "text-embedding-v4");
    assert_eq!(body["parameters"]["output_type"], "dense");
    let texts = body["input"]["texts"].as_array().unwrap();
    assert!(texts.len() <= 10);
    let dimension = body["parameters"]["dimension"].as_u64().unwrap() as usize;
    let mut rows: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            let n = text.as_str().unwrap().trim_start_matches("doc-").parse::<usize>().unwrap_or(0);
            let mut vector = vec![0.; dimension];
            vector[n % dimension] = 2.;
            json!({"text_index":i,"embedding":vector})
        })
        .collect();
    if state.mode.load(Ordering::SeqCst) == 1 && rows.len() > 1 {
        rows[1]["text_index"] = json!(0);
    }
    rows.reverse();
    let mut result =
        json!({"request_id":"synthetic","output":{"embeddings":rows},"usage":{"total_tokens":texts.len()*3}});
    if state.mode.load(Ordering::SeqCst) == 2 {
        result.as_object_mut().unwrap().remove("usage");
    }
    Json(result).into_response()
}
async fn server() -> (Arc<Mock>, String, tokio::task::JoinHandle<()>) {
    let state = Arc::new(Mock::default());
    let router = Router::new()
        .route("/embed", post(embedding))
        .route("/chat", post(|| async { Json(json!({"model":"openai/gpt-4o-mini","usage":{"prompt_tokens":3,"completion_tokens":2},"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]})) }))
        .route(
            "/leak",
            post(|State(s): State<Arc<Mock>>| async move {
                s.redirected.fetch_add(1, Ordering::SeqCst);
                StatusCode::NO_CONTENT
            }),
        )
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/embed", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (state, endpoint, task)
}
fn client(endpoint: &str, path: &std::path::Path, ceiling: f64) -> DashScopeV4 {
    DashScopeV4::local_test("synthetic-embedding-key".into(), endpoint, path, ceiling, 64).unwrap()
}

#[tokio::test]
async fn complete_batches_preserve_order_kind_normalization_and_persistent_cache() {
    let (mock, endpoint, task) = server().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spend.sqlite");
    let texts: Vec<String> = (0..23).map(|i| format!("doc-{i}")).collect();
    let first = client(&endpoint, &path, 1.);
    let vectors = first.embed(&texts, EmbedKind::Document).await.unwrap();
    assert_eq!(vectors.len(), 23);
    for (n, v) in vectors.iter().enumerate() {
        assert_eq!(v.len(), 64);
        assert_eq!(v[n], 1.);
        assert_eq!(v.iter().sum::<f32>(), 1.);
    }
    assert_eq!(
        mock.requests.lock().unwrap().iter().map(|v| v["input"]["texts"].as_array().unwrap().len()).collect::<Vec<_>>(),
        vec![10, 10, 3]
    );
    drop(first);
    let reopened = client(&endpoint, &path, 1.);
    assert_eq!(reopened.embed(&texts, EmbedKind::Document).await.unwrap(), vectors);
    assert_eq!(mock.requests.lock().unwrap().len(), 3);
    reopened.embed(&texts[..1], EmbedKind::Query).await.unwrap();
    {
        let requests = mock.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0]["parameters"]["text_type"], "document");
        assert_eq!(requests[3]["parameters"]["text_type"], "query");
    }
    let before = mock.requests.lock().unwrap().len();
    let mut invalid = texts.clone();
    invalid[22] = " ".into();
    assert!(reopened.embed(&invalid, EmbedKind::Document).await.is_err());
    assert_eq!(mock.requests.lock().unwrap().len(), before);
    assert_eq!(reopened.embed(&[], EmbedKind::Query).await.unwrap(), Vec::<Vec<f32>>::new());
    task.abort();
}
#[tokio::test]
async fn independent_clients_share_one_owned_embedding_call() {
    let (mock, endpoint, task) = server().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spend.sqlite");
    let a = client(&endpoint, &path, 1.);
    let b = client(&endpoint, &path, 1.);
    let texts = vec!["doc-4".into(), "doc-8".into()];
    let (left, right) = tokio::join!(a.embed(&texts, EmbedKind::Query), b.embed(&texts, EmbedKind::Query));
    assert_eq!(left.unwrap(), right.unwrap());
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
    task.abort();
}
#[tokio::test]
async fn malformed_mapping_cannot_publish_vectors_and_known_usage_is_charged() {
    let (mock, endpoint, task) = server().await;
    mock.mode.store(1, Ordering::SeqCst);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spend.sqlite");
    let provider = client(&endpoint, &path, 1.);
    assert!(provider.embed(&["doc-1".into(), "doc-2".into()], EmbedKind::Document).await.is_err());
    let db = rusqlite::Connection::open(&path).unwrap();
    let row: (Option<f64>, Option<String>, Option<String>) =
        db.query_row("SELECT cost,output,error FROM calls", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
    assert!(row.0.unwrap() > 0.);
    assert!(row.1.is_none());
    assert!(row.2.is_some());
    task.abort();
}
#[tokio::test]
async fn admission_and_unknown_charges_preserve_the_cumulative_ceiling() {
    let (mock, endpoint, task) = server().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("denied.sqlite");
    let denied = client(&endpoint, &path, 0.00000001);
    assert!(denied.embed(&["doc-1".into()], EmbedKind::Document).await.is_err());
    assert!(mock.requests.lock().unwrap().is_empty());
    let path = dir.path().join("uncertain.sqlite");
    mock.mode.store(2, Ordering::SeqCst);
    let uncertain = client(&endpoint, &path, 1.);
    assert!(uncertain.embed(&["doc-2".into()], EmbedKind::Document).await.is_err());
    let db = rusqlite::Connection::open(path).unwrap();
    let (reservation, cost): (f64, Option<f64>) =
        db.query_row("SELECT reservation,cost FROM calls", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert!(cost.is_none());
    assert_eq!(uncertain.accounted().unwrap(), reservation);
    assert!(reservation > 0.);
    task.abort();
}
#[tokio::test]
async fn redirects_are_not_followed_and_cannot_forward_authentication() {
    let (mock, endpoint, task) = server().await;
    mock.mode.store(3, Ordering::SeqCst);
    let dir = tempfile::tempdir().unwrap();
    let provider = client(&endpoint, &dir.path().join("spend.sqlite"), 1.);
    assert!(provider.embed(&["doc-1".into()], EmbedKind::Query).await.is_err());
    assert_eq!(mock.requests.lock().unwrap().len(), 1);
    assert_eq!(mock.redirected.load(Ordering::SeqCst), 0);
    assert!(provider.accounted().unwrap() > 0.);
    task.abort();
}

#[tokio::test]
async fn chat_and_embedding_share_one_budget_and_cannot_raise_its_ceiling() {
    use licorice_providers::{
        chat::{ChatModel, ModelRequest},
        chat_http::BudgetChat,
    };
    let (_, endpoint, task) = server().await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("spend.sqlite");
    let embedder = client(&endpoint, &path, 1.);
    let chat_endpoint = endpoint.replace("/embed", "/chat");
    let chat = BudgetChat::local_test("synthetic-chat-key".into(), &path, 1., &chat_endpoint).unwrap();
    embedder.embed(&["doc-1".into()], EmbedKind::Document).await.unwrap();
    let embedded_cost = embedder.accounted().unwrap();
    assert_eq!(chat.accounted().unwrap(), embedded_cost);
    chat.complete(&ModelRequest {
        id: "synthetic".into(),
        task: "answer".into(),
        system: String::new(),
        user: "fixture".into(),
        schema: None,
        max_tokens: 10,
    })
    .await
    .unwrap();
    assert!(chat.accounted().unwrap() > embedded_cost);
    assert_eq!(chat.accounted().unwrap(), embedder.accounted().unwrap());
    assert!(DashScopeV4::local_test("synthetic".into(), &endpoint, &path, 2., 64).is_err());
    assert!(BudgetChat::local_test("synthetic".into(), &path, 2., &chat_endpoint).is_err());
    task.abort();
}
