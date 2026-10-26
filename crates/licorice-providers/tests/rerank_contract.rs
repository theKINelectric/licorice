use licorice_providers::chat::{ChatError, ChatModel, ModelRequest, validate};
use licorice_providers::rerank::{Candidate, ChatReranker, Reranker};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

struct Scripted {
    outputs: Mutex<VecDeque<Value>>,
    requests: Mutex<Vec<ModelRequest>>,
}
impl Scripted {
    fn new(outputs: Vec<Value>) -> Arc<Self> {
        Arc::new(Self { outputs: Mutex::new(outputs.into()), requests: Mutex::new(Vec::new()) })
    }
}
#[async_trait::async_trait]
impl ChatModel for Scripted {
    fn id(&self) -> &str {
        "scripted-rerank-contract"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<Value, ChatError> {
        self.requests.lock().unwrap().push(request.clone());
        self.outputs.lock().unwrap().pop_front().ok_or_else(|| ChatError("unexpected model call".into()))
    }
}
fn candidates() -> Vec<Candidate> {
    vec![
        Candidate { id: 17, text: "Ada's sister is Mira.".into() },
        Candidate { id: 91, text: "Mira lives in Bergen.".into() },
    ]
}

#[tokio::test]
async fn keyed_scores_bind_the_output_schema_to_the_exact_supplied_ids() {
    let output = json!({"scores":{"91":0.9,"17":0.2}});
    let model = Scripted::new(vec![output.clone()]);
    let ranked = ChatReranker(model.clone()).rank("Where does Ada's sister live?", &candidates()).await.unwrap();
    assert_eq!(ranked.iter().map(|r| r.id).collect::<Vec<_>>(), vec![91, 17]);
    let requests = model.requests.lock().unwrap();
    assert_eq!(requests.len(), 1, "valid exact-set output needs no paid repair");
    let schema = requests[0].schema.as_ref().unwrap();
    assert!(validate(schema, &output).is_ok());
    assert!(validate(schema, &json!({"scores":{"17":0.2}})).is_err(), "missing candidate is schema-invalid");
    assert!(
        validate(schema, &json!({"scores":{"17":0.2,"91":0.9,"1":1.0}})).is_err(),
        "ordinal/invented IDs are schema-invalid"
    );
    assert!(validate(schema, &json!({"scores":{"17":0.2,"91":1.1}})).is_err());
    assert!(requests[0].max_tokens <= 1024);
}

#[tokio::test]
async fn duplicate_candidate_inputs_fail_before_any_model_call() {
    let model = Scripted::new(vec![]);
    let mut input = candidates();
    input[1].id = input[0].id;
    let error = ChatReranker(model.clone()).rank("query", &input).await.unwrap_err();
    assert!(error.to_string().contains("unique"));
    assert!(model.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn malformed_injected_outputs_get_at_most_two_repairs_and_never_fabricate_scores() {
    let model = Scripted::new(vec![json!({"scores":{"17":0.3}}); 3]);
    assert!(ChatReranker(model.clone()).rank("query", &candidates()).await.is_err());
    assert_eq!(model.requests.lock().unwrap().len(), 3);
    let model = Scripted::new(vec![json!({"scores":{"17":0.3}}), json!({"scores":{"17":0.3,"91":0.3}})]);
    let ranked = ChatReranker(model.clone()).rank("query", &candidates()).await.unwrap();
    assert_eq!(ranked.iter().map(|r| r.id).collect::<Vec<_>>(), vec![17, 91], "ties are stable by ID");
    assert_eq!(model.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn empty_and_oversized_inputs_do_not_spend_model_calls() {
    let model = Scripted::new(vec![]);
    assert!(ChatReranker(model.clone()).rank("query", &[]).await.unwrap().is_empty());
    let input: Vec<_> = (0..25).map(|id| Candidate { id, text: "source".into() }).collect();
    assert!(ChatReranker(model.clone()).rank("query", &input).await.is_err());
    assert!(model.requests.lock().unwrap().is_empty());
}

/// Explicitly invoked paid qualification; ordinary workspace tests remain offline.
#[tokio::test]
#[ignore = "paid OpenRouter qualification; requires an explicit ledger and ceiling"]
async fn live_keyed_twenty_four_candidates_have_one_exact_set_response() {
    use licorice_providers::chat_http::BudgetChat;
    let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY is required");
    let ledger = std::env::var("LICORICE_RERANK_PROBE_LEDGER").expect("explicit probe ledger required");
    let ceiling: f64 =
        std::env::var("LICORICE_RERANK_PROBE_CEILING").expect("explicit ceiling required").parse().unwrap();
    let model = Arc::new(BudgetChat::open(key, std::path::Path::new(&ledger), ceiling).unwrap());
    let candidates: Vec<_> = (0..24)
        .map(|index| Candidate {
            id: 1009 + index * 37,
            text: match index {
                0 => "Ada's sister is Mira.".into(),
                1 => "Mira lives in Bergen.".into(),
                _ => format!("Person {index} enjoys gardening and has a blue bicycle."),
            },
        })
        .collect();
    let query = "Where does Ada's sister live?";
    let fixture_id = format!(
        "rerank-v2:{}",
        blake3::hash(json!({"query":query,"candidates":candidates}).to_string().as_bytes()).to_hex()
    );
    let rows = ChatReranker(model.clone()).rank(query, &candidates).await.unwrap();
    assert_eq!(rows.len(), 24);
    assert_eq!(
        rows.iter().map(|r| r.id).collect::<std::collections::HashSet<_>>(),
        candidates.iter().map(|c| c.id).collect()
    );
    let db = rusqlite::Connection::open(ledger).unwrap();
    let (calls, complete): (i64, i64) = db
        .query_row(
            "SELECT count(*),sum(output IS NOT NULL) FROM calls WHERE json_extract(request,'$.task')='rerank' AND json_extract(request,'$.request_id') LIKE ?1",
            [format!("{fixture_id}%")],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!((calls, complete), (1, 1), "qualification uses one exact-set response and no repairs");
    println!(
        "{}",
        json!({"qualification":"passed","candidate_count":24,"calls":calls,"accounted_usd":model.accounted().unwrap()})
    );
}

#[tokio::test]
async fn contextual_requests_bind_clock_and_actor_without_changing_legacy_wire() {
    use licorice_providers::rerank::RerankContext;
    let output = json!({"scores":{"17":0.2,"91":0.9}});
    let model = Scripted::new(vec![output; 5]);
    let scorer = ChatReranker(model.clone());
    let context = RerankContext { reference_date_utc: "2023-01-02".into(), speaker: Some("Ada".into()) };
    let query = "What did I do yesterday?";
    let input = candidates();
    scorer.rank(query, &input).await.unwrap();
    scorer.rank_with_context(query, &input, &context).await.unwrap();
    scorer.rank_with_context(query, &input, &context).await.unwrap();
    scorer
        .rank_with_context(query, &input, &RerankContext { reference_date_utc: "2023-01-03".into(), ..context.clone() })
        .await
        .unwrap();
    scorer.rank_with_context(query, &input, &RerankContext { speaker: Some("Mira".into()), ..context }).await.unwrap();
    let calls = model.requests.lock().unwrap();
    let legacy = json!({"query":query,"candidates":input}).to_string();
    assert_eq!(calls[0].user, legacy);
    assert_eq!(calls[0].id, format!("rerank-v2:{}", blake3::hash(legacy.as_bytes()).to_hex()));
    assert_eq!(
        calls[0].system,
        "Score every candidate's usefulness as evidence for the query from 0 to 1. Preserve connecting relationships needed for multi-hop questions. Do not answer the question. Use only candidate text; do not invent facts. Return scores as an object keyed by the exact supplied candidate IDs, with every required key once."
    );
    assert_eq!(calls[1].user, calls[2].user);
    assert_eq!(calls[1].id, calls[2].id);
    for index in [0, 3, 4] {
        assert_ne!(calls[1].id, calls[index].id);
    }
    assert!(calls.iter().all(|c| c.max_tokens == 1024));
}

struct ContextEcho(Mutex<Vec<ModelRequest>>);
#[async_trait::async_trait]
impl ChatModel for ContextEcho {
    fn id(&self) -> &str {
        "context-batch-control"
    }
    async fn complete(&self, request: &ModelRequest) -> Result<Value, ChatError> {
        self.0.lock().unwrap().push(request.clone());
        let payload: Value = serde_json::from_str(&request.user).unwrap();
        let scores: serde_json::Map<String, Value> = payload["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| (c["id"].as_i64().unwrap().to_string(), json!(0.5)))
            .collect();
        Ok(json!({"scores":scores}))
    }
}
#[tokio::test]
async fn all_sixty_candidate_chunks_receive_the_same_context_and_invalid_pools_make_no_calls() {
    use licorice_providers::rerank::{BatchedReranker, RerankContext};
    let model = Arc::new(ContextEcho(Mutex::new(vec![])));
    let scorer = BatchedReranker(Arc::new(ChatReranker(model.clone())));
    let context = RerankContext { reference_date_utc: "2023-01-02".into(), speaker: Some("Ada".into()) };
    let mut input: Vec<_> = (0..60).map(|id| Candidate { id, text: format!("evidence{id}") }).collect();
    assert_eq!(scorer.rank_with_context("natural query", &input, &context).await.unwrap().len(), 60);
    {
        let calls = model.0.lock().unwrap();
        let mut sizes = vec![];
        for call in calls.iter() {
            let payload: Value = serde_json::from_str(&call.user).unwrap();
            assert_eq!(payload["query"], "natural query");
            assert_eq!(payload["context"], serde_json::to_value(&context).unwrap());
            sizes.push(payload["candidates"].as_array().unwrap().len());
        }
        sizes.sort();
        assert_eq!(sizes, vec![12, 24, 24]);
    }
    input[25].id = input[0].id;
    assert!(scorer.rank_with_context("q", &input, &context).await.is_err());
    assert_eq!(model.0.lock().unwrap().len(), 3);
}
