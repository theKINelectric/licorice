//! Contract tests for the System One HTTP judge against a local mock server.
//! Response bodies follow TypeSafe's v1 OpenAPI examples (typesafe-sdk 0.7.5) and the
//! shape Coffee Cat's research engine (`coffee_cat.py`, 0.2) returns.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::post;
use indexmap::IndexMap;
use licorice_providers::judge::{
    Answer, HttpJudge, HttpJudgeConfig, HttpJudgeProtocol, Judge, JudgeError, Point, Question,
};
use serde_json::{Value, json};

/// One recorded request: its `Authorization` header and JSON body.
type Recorded = (Option<String>, Value);

#[derive(Clone, Default)]
struct Seen {
    requests: Arc<Mutex<Vec<Recorded>>>,
}

/// Starts a server on an ephemeral port that records each request and answers with `reply`.
async fn serve<F>(path: &'static str, reply: F) -> (String, Seen)
where
    F: Fn(&Value) -> (StatusCode, Value) + Clone + Send + Sync + 'static,
{
    let seen = Seen::default();
    let app = Router::new()
        .route(
            path,
            post(move |State(seen): State<Seen>, headers: HeaderMap, body: axum::Json<Value>| {
                let reply = reply.clone();
                async move {
                    let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).map(str::to_owned);
                    seen.requests.lock().unwrap().push((auth, body.0.clone()));
                    let (status, value) = reply(&body.0);
                    if value.get("_sleep_ms").is_some() {
                        tokio::time::sleep(Duration::from_millis(value["_sleep_ms"].as_u64().unwrap())).await;
                    }
                    (status, axum::Json(value))
                }
            }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), seen)
}

fn config(base_url: String, path: &str) -> HttpJudgeConfig {
    HttpJudgeConfig {
        base_url,
        path: path.into(),
        protocol: HttpJudgeProtocol::Jev,
        model: None,
        api_key: None,
        max_questions_per_request: 15,
        max_concurrent: 1,
    }
}

fn ticket_questions() -> IndexMap<String, Question> {
    let mut q = IndexMap::new();
    q.insert("billing".into(), Question::noul("Is this message about billing?"));
    q.insert(
        "area".into(),
        Question::choice(
            "Which product area is this about?",
            [("refund", Some("A billing dispute or refund request".to_owned())), ("card", None), ("other", None)],
        ),
    );
    q.insert(
        "urgency".into(),
        Question::score(
            "How urgent is this message?",
            ["Can wait", "Needs attention this week", "Needs attention today"],
        ),
    );
    q
}

const SECOND: Duration = Duration::from_secs(1);

#[tokio::test]
async fn mercury_uses_the_decision_wire_without_coffee_cat_conversion_or_deadline() {
    let (url, seen) = serve("/v1/decisions", |body| {
        assert_eq!(body["model"], "mercury-decide");
        assert!(body.get("deadline_ms").is_none());
        assert_eq!(body["questions"]["evidence"]["criteria"]["no"], Value::Null);
        (
            StatusCode::OK,
            json!({"model":"mercury-decide","answers":{
            "evidence":{"type":"choice","choice":"yes","confidence":0.9,"probabilities":{"yes":0.9,"no":0.1}}
        },"usage":{"input_tokens":120,"output_tokens":1}}),
        )
    })
    .await;
    let mut cfg = HttpJudgeConfig::mercury_with_key("synthetic-mercury-key").unwrap();
    assert!(!format!("{cfg:?}").contains("synthetic-mercury-key"));
    cfg.base_url = url;
    let judge = HttpJudge::new(cfg).unwrap();
    let mut questions = IndexMap::new();
    questions.insert(
        "evidence".into(),
        Question::choice("Supported?", [("yes", Some("Supported".to_owned())), ("no", None)]),
    );
    let result = judge.ask(Point::J10, &json!({"memory":"Synthetic evidence"}), &questions, SECOND).await.unwrap();
    assert_eq!(result.usage.input_tokens, Some(120));
    assert_eq!(result.answers["evidence"].as_choice(), Some(("yes", 0.9)));
    let requests = seen.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0.as_deref(), Some("Bearer synthetic-mercury-key"));
}

#[tokio::test]
async fn mercury_refuses_model_substitution_without_retrying() {
    let (url, seen) =
        serve("/v1/decisions", |_| (StatusCode::OK, json!({"model":"other","answers":{},"usage":{}}))).await;
    let mut cfg = HttpJudgeConfig::mercury_with_key("synthetic-key").unwrap();
    cfg.base_url = url;
    let mut questions = IndexMap::new();
    questions.insert("check".into(), Question::noul("Supported?"));
    assert!(HttpJudge::new(cfg).unwrap().ask(Point::J10, &json!({}), &questions, SECOND).await.is_err());
    assert_eq!(seen.requests.lock().unwrap().len(), 1);
    assert!(HttpJudgeConfig::mercury_with_key(" ").is_err());
}

#[test]
fn coffee_cat_defaults_match_the_production_service() {
    let cfg = HttpJudgeConfig::coffee_cat();
    assert_eq!(cfg.path, "/api/v1/decisions");
}

#[test]
fn configuration_debug_does_not_disclose_credentials() {
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.api_key = Some("synthetic-test-secret".into());
    assert!(!format!("{cfg:?}").contains("synthetic-test-secret"));
}

#[tokio::test]
async fn v1_request_and_response_round_trip() {
    let (url, seen) = serve("/v1/systemone", |_| {
        (
            StatusCode::OK,
            json!({
                "model": "jev-1.14",
                "answers": {
                    "billing": {"type": "noul", "noul": 0.98},
                    "area": {"type": "choice", "choice": "refund", "confidence": 0.84,
                             "probabilities": {"refund": 0.89, "card": 0.04, "other": 0.07}},
                    "urgency": {"type": "score", "score": 1.7, "confidence": 0.9,
                                "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
                                "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}}
                },
                "usage": {"input_tokens": 120, "output_tokens": 12}
            }),
        )
    })
    .await;
    let mut cfg = config(url, "/v1/systemone");
    cfg.model = Some("jev-latest".into());
    cfg.api_key = Some("test-key".into());
    let judge = HttpJudge::new(cfg).unwrap();
    let state = json!({"document": "I was charged twice. Please fix this ASAP."});
    let j = judge.ask(Point::J8, &state, &ticket_questions(), SECOND).await.unwrap();

    assert_eq!(j.model, "jev-1.14");
    assert_eq!(j.answers["billing"], Answer::Noul { noul: 0.98 });
    assert_eq!(j.answers["area"].as_choice(), Some(("refund", 0.84)));
    assert_eq!(j.answers["urgency"].as_score(), Some(1.7));
    assert_eq!(j.usage.input_tokens, Some(120));
    assert_eq!(j.answers.keys().collect::<Vec<_>>(), ["billing", "area", "urgency"], "question order kept");

    let requests = seen.requests.lock().unwrap();
    let (auth, body) = &requests[0];
    assert_eq!(auth.as_deref(), Some("Bearer test-key"));
    assert_eq!(body["model"], "jev-latest");
    assert_eq!(body["state"], state);
    assert_eq!(body["questions"]["billing"], json!({"type": "noul", "instructions": "Is this message about billing?"}));
    assert_eq!(body["questions"]["area"]["criteria"]["card"], Value::Null);
    assert_eq!(body["questions"]["urgency"]["criteria"][2], "Needs attention today");
}

#[tokio::test]
async fn coffee_cat_research_shape_is_understood() {
    // coffee-cat/0.2: no `type`, model under `usage`, a choice distribution under another key.
    let (url, seen) = serve("/api/alpha/decisions", |_| {
        (
            StatusCode::OK,
            json!({
                "answers": {
                    "billing": {"noul": 0.91, "confidence": 0.91},
                    "area": {"choice": "card", "confidence": 0.6, "distribution": {"refund": 0.3, "card": 0.6, "other": 0.1}},
                    "urgency": {"score": 1.2, "choice": "Needs attention this week", "confidence": 0.7}
                },
                "usage": {"model": "coffee-cat/0.2", "cost": 0.0, "ms": 48}
            }),
        )
    })
    .await;
    let judge = HttpJudge::new(config(url, "/api/alpha/decisions")).unwrap();
    let j = judge.ask(Point::J1, &json!("state"), &ticket_questions(), SECOND).await.unwrap();
    assert_eq!(j.model, "coffee-cat/0.2");
    assert_eq!(j.answers["billing"].as_noul(), Some(0.91));
    match &j.answers["area"] {
        Answer::Choice { choice, probabilities, .. } => {
            assert_eq!(choice, "card");
            assert_eq!(probabilities["card"], 0.6);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(j.answers["urgency"].as_score(), Some(1.2));
    assert_eq!(j.usage.input_tokens, None);
    let requests = seen.requests.lock().unwrap();
    assert!(requests[0].0.is_none(), "no auth header without a key");
    assert!(requests[0].1.get("model").is_none(), "no model sent when none is configured");
}

#[tokio::test]
async fn deadline_is_enforced() {
    let (url, _) = serve("/v1/systemone", |_| (StatusCode::OK, json!({"_sleep_ms": 600, "answers": {}}))).await;
    let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
    let mut q = IndexMap::new();
    q.insert("gate".to_owned(), Question::noul("Worth remembering?"));
    let started = Instant::now();
    let err = judge.ask(Point::J1, &json!({}), &q, Duration::from_millis(100)).await.unwrap_err();
    assert!(matches!(err, JudgeError::Timeout(_)), "{err}");
    assert!(started.elapsed() < Duration::from_millis(400), "returned after {:?}", started.elapsed());
}

#[tokio::test]
async fn http_errors_are_reported_with_their_body() {
    let (url, _) = serve("/v1/systemone", |_| {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            json!({"detail": [{"loc": ["body", "state"], "msg": "STATE_TRUNCATED", "type": "value_error"}]}),
        )
    })
    .await;
    let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
    let mut q = IndexMap::new();
    q.insert("gate".to_owned(), Question::noul("Worth remembering?"));
    match judge.ask(Point::J1, &json!({}), &q, SECOND).await {
        Err(JudgeError::Http { status: 422, body }) => assert!(body.contains("STATE_TRUNCATED"), "{body}"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn large_requests_are_split_and_merged_in_order() {
    let (url, seen) = serve("/v1/systemone", |body| {
        let answers: serde_json::Map<String, Value> = body["questions"]
            .as_object()
            .unwrap()
            .keys()
            .map(|k| (k.clone(), json!({"type": "noul", "noul": 0.5})))
            .collect();
        (
            StatusCode::OK,
            json!({"model": "jev-1.14", "answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}}),
        )
    })
    .await;
    let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
    let q: IndexMap<String, Question> = (0..20)
        .map(|i| (format!("candidate_{i:02}"), Question::noul(format!("Candidate {i} is the same person"))))
        .collect();
    let j = judge.ask(Point::J2, &json!({"mention": "Mel"}), &q, SECOND).await.unwrap();
    assert_eq!(j.answers.len(), 20);
    assert_eq!(j.answers.keys().cloned().collect::<Vec<_>>(), q.keys().cloned().collect::<Vec<_>>());
    assert_eq!(j.usage.input_tokens, Some(20), "usage summed over both requests");
    let sizes: Vec<usize> =
        seen.requests.lock().unwrap().iter().map(|(_, b)| b["questions"].as_object().unwrap().len()).collect();
    assert_eq!(sizes, vec![15, 5]);
}

#[tokio::test]
async fn malformed_answers_are_rejected() {
    let cases: Vec<(&str, Value)> = vec![
        ("missing answer", json!({"answers": {}})),
        ("wrong type", json!({"answers": {"gate": {"type": "choice", "choice": "a", "confidence": 1.0}}})),
        ("probability out of range", json!({"answers": {"gate": {"noul": 1.7}}})),
        ("not json answers", json!({"answers": "nope"})),
    ];
    for (label, reply) in cases {
        let (url, _) = serve("/v1/systemone", move |_| (StatusCode::OK, reply.clone())).await;
        let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
        let mut q = IndexMap::new();
        q.insert("gate".to_owned(), Question::noul("Worth remembering?"));
        let err = judge.ask(Point::J1, &json!({}), &q, SECOND).await.unwrap_err();
        assert!(matches!(err, JudgeError::Decode(_)), "{label}: {err}");
    }
}

#[tokio::test]
async fn choices_outside_the_options_are_rejected() {
    let (url, _) = serve("/v1/systemone", |_| {
        (StatusCode::OK, json!({"answers": {"pick": {"choice": "zebra", "confidence": 0.9}}}))
    })
    .await;
    let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
    let mut q = IndexMap::new();
    q.insert("pick".to_owned(), Question::choice("Pick one", [("a", None), ("b", None)]));
    assert!(matches!(judge.ask(Point::J7, &json!({}), &q, SECOND).await, Err(JudgeError::Decode(_))));
}

#[test]
fn noul_criteria_serialize_as_true_and_false() {
    let q = Question::noul_with("Is this spam?", "Unsolicited advertising", "A legitimate conversation");
    assert_eq!(
        serde_json::to_value(&q).unwrap(),
        json!({"type": "noul", "instructions": "Is this spam?",
               "criteria": {"true": "Unsolicited advertising", "false": "A legitimate conversation"}})
    );
    let back: Question = serde_json::from_value(serde_json::to_value(&q).unwrap()).unwrap();
    assert_eq!(back, q);
}

#[tokio::test]
async fn production_coffee_cat_receives_compatible_questions_and_deadline() {
    let (url, seen) = serve("/api/v1/decisions", |body| {
        assert!((100..=1000).contains(&body["deadline_ms"].as_u64().unwrap()));
        assert!(body["questions"]["billing"]["instructions"].is_string());
        assert_eq!(body["questions"]["area"]["criteria"]["card"], "card");
        (
            StatusCode::OK,
            json!({"model": "coffee-cat-0.3", "answers": {
                "billing": {"type": "noul", "noul": 0.9},
                "area": {"type": "choice", "choice": "card", "confidence": 0.8},
                "urgency": {"type": "score", "score": 1.4}
            }}),
        )
    })
    .await;
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.base_url = url;
    let judge = HttpJudge::new(cfg).unwrap();
    assert!(judge.is_loopback());
    let mut questions = ticket_questions();
    if let Question::Noul { instructions, .. } = questions.get_mut("billing").unwrap() {
        *instructions = json!({"question": "Is this about billing?", "reference": ["invoices", "cards"]});
    }
    let judgment = judge.ask(Point::J10, &json!({"message": "Help with my card"}), &questions, SECOND).await.unwrap();
    assert_eq!(judgment.model, "coffee-cat-0.3");
    assert_eq!(seen.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn coffee_cat_deadline_decreases_between_batches() {
    let (url, seen) = serve("/api/v1/decisions", |body| {
        let answers: serde_json::Map<String, Value> =
            body["questions"].as_object().unwrap().keys().map(|name| (name.clone(), json!({"noul": 0.5}))).collect();
        (StatusCode::OK, json!({"model": "coffee-cat-0.3", "answers": answers, "_sleep_ms": 40}))
    })
    .await;
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.base_url = url;
    cfg.max_questions_per_request = 1;
    let judge = HttpJudge::new(cfg).unwrap();
    let questions = IndexMap::from([
        ("one".into(), Question::noul("Is one present?")),
        ("two".into(), Question::noul("Is two present?")),
    ]);
    judge.ask(Point::J10, &json!({}), &questions, SECOND).await.unwrap();
    let requests = seen.requests.lock().unwrap();
    let first = requests[0].1["deadline_ms"].as_u64().unwrap();
    let second = requests[1].1["deadline_ms"].as_u64().unwrap();
    assert!(second < first, "{first} then {second}");
}

#[tokio::test]
async fn coffee_cat_never_extends_a_deadline_below_the_server_minimum() {
    let (url, seen) = serve("/api/v1/decisions", |_| (StatusCode::OK, json!({}))).await;
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.base_url = url;
    let judge = HttpJudge::new(cfg).unwrap();
    let questions = IndexMap::from([("gate".into(), Question::noul("Is this present?"))]);
    let response = judge.ask(Point::J10, &json!({}), &questions, Duration::from_millis(50)).await;
    assert!(matches!(response, Err(JudgeError::Timeout(_))));
    assert!(seen.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn queued_requests_are_included_in_the_client_deadline() {
    let (url, seen) =
        serve("/api/v1/decisions", |_| (StatusCode::OK, json!({"answers": {"gate": {"noul": 0.9}}, "_sleep_ms": 250})))
            .await;
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.base_url = url;
    let judge = Arc::new(HttpJudge::new(cfg).unwrap());
    let questions = IndexMap::from([("gate".into(), Question::noul("Is this present?"))]);
    let first = {
        let judge = Arc::clone(&judge);
        let questions = questions.clone();
        tokio::spawn(async move { judge.ask(Point::J10, &json!({}), &questions, SECOND).await })
    };
    tokio::time::timeout(SECOND, async {
        while seen.requests.lock().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = judge.ask(Point::J10, &json!({}), &questions, Duration::from_millis(100)).await;
    assert!(matches!(second, Err(JudgeError::Timeout(_))));
    first.await.unwrap().unwrap();
    assert_eq!(seen.requests.lock().unwrap().len(), 1, "expired request never left the queue");
}

#[tokio::test]
async fn multibyte_http_error_body_is_bounded_without_panicking() {
    let (url, _) = serve("/v1/systemone", |_| (StatusCode::BAD_REQUEST, json!({"detail": "€".repeat(2000)}))).await;
    let judge = HttpJudge::new(config(url, "/v1/systemone")).unwrap();
    let questions = IndexMap::from([("gate".into(), Question::noul("Is this present?"))]);
    match judge.ask(Point::J10, &json!({}), &questions, SECOND).await {
        Err(JudgeError::Http { status: 400, body }) => assert!(body.len() <= 2048),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn loopback_requests_do_not_follow_redirects() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let forwarded = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&forwarded);
    let app =
        Router::new().route("/start", post(|| async { axum::response::Redirect::temporary("/forwarded") })).route(
            "/forwarded",
            post(move || {
                seen.fetch_add(1, Ordering::SeqCst);
                async { axum::Json(json!({"answers": {"gate": {"noul": 1.0}}})) }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let judge = HttpJudge::new(config(url, "/start")).unwrap();
    let questions = IndexMap::from([("gate".into(), Question::noul("Is this present?"))]);
    assert!(matches!(
        judge.ask(Point::J10, &json!({}), &questions, SECOND).await,
        Err(JudgeError::Http { status: 307, .. })
    ));
    assert_eq!(forwarded.load(Ordering::SeqCst), 0);
}

#[test]
fn unknown_hosts_and_blank_credentials_are_not_local_defaults() {
    for host in ["http://127.0.0.1", "http://[::1]", "http://localhost"] {
        assert!(HttpJudge::new(config(host.into(), "/judge")).unwrap().is_loopback());
    }
    for host in ["https://api.typesafe.ai", "http://127.0.0.1.example.invalid"] {
        assert!(!HttpJudge::new(config(host.into(), "/judge")).unwrap().is_loopback());
    }
    let mut cfg = HttpJudgeConfig::coffee_cat();
    cfg.api_key = Some(" ".into());
    assert!(matches!(HttpJudge::new(cfg), Err(JudgeError::Invalid(_))));
}

#[tokio::test]
async fn openrouter_uses_its_own_host_key_and_system_one_protocol() {
    let (url, seen) = serve("/api/v1/systemone", |body| {
        assert_eq!(body["model"], "jev-latest");
        assert!(body.get("deadline_ms").is_none());
        (
            StatusCode::OK,
            json!({
                "id": "synthetic-provider-request", "provider": "TypeSafe",
                "model": "typesafe/jev-1.14-test",
                "answers": {"gate": {"type": "noul", "noul": 0.97}},
                "usage": {"input_tokens": 300, "output_tokens": 2, "cost": 0.0000126}
            }),
        )
    })
    .await;
    let mut cfg = HttpJudgeConfig::openrouter_with_key("synthetic-openrouter-key").unwrap();
    assert_eq!(cfg.base_url, "https://openrouter.ai/api");
    assert_eq!(cfg.path, "/v1/systemone");
    assert_eq!(cfg.protocol, HttpJudgeProtocol::Jev);
    cfg.base_url = format!("{url}/api");
    let judge = HttpJudge::new(cfg).unwrap();
    let questions = IndexMap::from([("gate".into(), Question::noul_with("Is this present?", "Present", "Absent"))]);
    let judgment = judge.ask(Point::J10, &json!({"memory": "synthetic"}), &questions, SECOND).await.unwrap();
    assert_eq!(judgment.model, "typesafe/jev-1.14-test");
    assert_eq!(judgment.usage.cost, Some(0.0000126));
    assert_eq!(seen.requests.lock().unwrap()[0].0.as_deref(), Some("Bearer synthetic-openrouter-key"));
}
