use licorice::being::reflection::*;
use licorice::being::*;
use licorice::{Store, StoreConfig};
use licorice_providers::chat::{ChatError, ChatModel, ModelRequest, ModelResponse};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

fn actor() -> Actor {
    Actor { principal: "agent".into(), role: ActorRole::Agent, session: "s".into() }
}
async fn fixture() -> (tempfile::TempDir, Store, Memory) {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::initialize(&s, "agent").await.unwrap();
    (d, s, m)
}
async fn observe(m: &Memory, id: &str, origin: Origin) {
    let role = if origin == Origin::HumanReport { ActorRole::Human } else { ActorRole::Agent };
    m.apply(
        Actor { principal: if role == ActorRole::Human { "human" } else { "agent" }.into(), role, session: "s".into() },
        Request {
            request_id: id.into(),
            expected_revision: None,
            operation: Operation::Observe(Observation {
                origin,
                subject: "human".into(),
                text: format!("Correction {id}: the workshop closes on Monday."),
                relationship_id: None,
                occurred_at: None,
                source: Source::Reported { reference: Some(format!("source:{id}")) },
            }),
        },
    )
    .await
    .unwrap();
}
fn input(job: &str, source: &str) -> ReflectionInput {
    ReflectionInput {
        job_id: job.into(),
        trigger: Trigger::Correction,
        source_ids: vec![source.into()],
        relationship_id: None,
        question: "Does this warrant a review of workshop scheduling?".into(),
    }
}
fn response(job: &str, source: &str) -> ModelResponse {
    ModelResponse {
        id: job.into(),
        output: json!({"action":"review","conclusion":"Review workshop scheduling against the attributed correction.","alternatives":["The report could describe a temporary closure."],"witnesses":[{"source_id":source,"quote":"the workshop closes on Monday."}],"positions":[]}),
    }
}

#[tokio::test]
async fn reflection_quotes_replay_and_reservations_survive_reopen_without_adopting_stance() {
    let (d, s, m) = fixture().await;
    observe(&m, "correction", Origin::HumanReport).await;
    let PlanResult::Ready { plan } = m.plan_reflection(actor(), input("job", "correction")).await.unwrap() else {
        panic!()
    };
    assert!(plan.memory_tokens <= 2000);
    assert_eq!(plan.request.max_tokens, 512);
    let mut bad = response("job", "correction");
    bad.output["witnesses"][0]["quote"] = json!("The report definitely proves every Monday forever.");
    assert!(m.complete_reflection(actor(), bad).await.is_err());
    assert_eq!(m.revision().await.unwrap(), 1);
    s.close().await.unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::attach(&s, "agent").await.unwrap();
    m.complete_reflection(actor(), response("job", "correction")).await.unwrap();
    assert!(m.complete_reflection(actor(), response("job", "correction")).await.unwrap().replayed);
    assert!(m.snapshot(None).await.unwrap().claims.is_empty());
    assert!(matches!(
        m.plan_reflection(actor(), input("duplicate", "correction")).await.unwrap(),
        PlanResult::Skipped { .. }
    ));
    m.apply(
        Actor { principal: "human".into(), role: ActorRole::Human, session: "s".into() },
        Request {
            request_id: "renamed-copy".into(),
            expected_revision: None,
            operation: Operation::Observe(Observation {
                origin: Origin::HumanReport,
                subject: "human".into(),
                text: "Correction correction: the workshop closes on Monday.".into(),
                relationship_id: None,
                occurred_at: None,
                source: Source::Reported { reference: Some("renamed:source".into()) },
            }),
        },
    )
    .await
    .unwrap();
    assert!(matches!(
        m.plan_reflection(actor(), input("copied-source", "renamed-copy")).await.unwrap(),
        PlanResult::Skipped { .. }
    ));
    assert!(matches!(
        m.plan_reflection(actor(), input("self-loop", "reflect-job")).await.unwrap(),
        PlanResult::Skipped { .. }
    ));
    s.close().await.unwrap();
}

#[tokio::test]
async fn stale_snapshot_and_overbudget_output_cannot_change_memory() {
    let (_d, s, m) = fixture().await;
    observe(&m, "correction", Origin::HumanReport).await;
    m.plan_reflection(actor(), input("job", "correction")).await.unwrap();
    let mut bad = response("job", "correction");
    bad.output["conclusion"] = json!("wide ".repeat(900));
    assert!(m.complete_reflection(actor(), bad).await.is_err());
    observe(&m, "new-source", Origin::HumanReport).await;
    assert!(m.complete_reflection(actor(), response("job", "correction")).await.is_err());
    assert_eq!(m.revision().await.unwrap(), 2);
    s.close().await.unwrap();
}
struct FailingModel(AtomicUsize);
#[async_trait::async_trait]
impl ChatModel for FailingModel {
    fn id(&self) -> &str {
        "reflection-failure-control"
    }
    async fn complete(&self, _: &ModelRequest) -> Result<Value, ChatError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Err(ChatError("synthetic failure".into()))
    }
}
#[tokio::test]
async fn failure_has_one_attempt_and_daily_budget_is_persistent_and_finite() {
    let (d, s, m) = fixture().await;
    observe(&m, "c0", Origin::HumanReport).await;
    let model = FailingModel(AtomicUsize::new(0));
    assert!(m.execute_reflection(actor(), input("j0", "c0"), &model).await.is_err());
    assert!(m.execute_reflection(actor(), input("j0", "c0"), &model).await.unwrap().is_none());
    assert_eq!(model.0.load(Ordering::SeqCst), 1);
    for i in 1..4 {
        let source = format!("c{i}");
        observe(&m, &source, Origin::HumanReport).await;
        assert!(matches!(
            m.plan_reflection(actor(), input(&format!("j{i}"), &source)).await.unwrap(),
            PlanResult::Ready { .. }
        ));
    }
    s.close().await.unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::attach(&s, "agent").await.unwrap();
    observe(&m, "c4", Origin::HumanReport).await;
    assert!(matches!(m.plan_reflection(actor(), input("j4", "c4")).await.unwrap(), PlanResult::Skipped { .. }));
    s.close().await.unwrap();
}
