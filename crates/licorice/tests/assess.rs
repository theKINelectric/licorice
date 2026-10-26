mod common;

use std::sync::Mutex;
use std::time::Duration;

use indexmap::IndexMap;
use licorice::assess::{
    AssessConfig, AssessmentStatus, EvidenceAssessor, allow_remote_judgments, remote_judgments_allowed,
};
use licorice::{Store, StoreConfig};
use licorice_providers::judge::{Answer, Judge, JudgeError, Judgment, Point, Question, Usage};
use serde_json::{Value, json};

enum Reply {
    Valid,
    Error,
    Slow,
    Invalid,
}

struct TestJudge {
    local: bool,
    reply: Reply,
    seen: Mutex<Vec<Value>>,
}

impl TestJudge {
    fn new(local: bool, reply: Reply) -> Self {
        Self { local, reply, seen: Mutex::new(Vec::new()) }
    }
}

#[async_trait::async_trait]
impl Judge for TestJudge {
    fn id(&self) -> &str {
        "synthetic-test-judge"
    }
    fn requested_model(&self) -> Option<&str> {
        Some("synthetic-alias")
    }
    fn is_loopback(&self) -> bool {
        self.local
    }
    async fn ask(
        &self,
        point: Point,
        state: &Value,
        questions: &IndexMap<String, Question>,
        _: Duration,
    ) -> Result<Judgment, JudgeError> {
        assert_eq!(point, Point::J10);
        self.seen.lock().unwrap().push(json!({"state": state, "questions": questions}));
        match self.reply {
            Reply::Error => return Err(JudgeError::Transport("synthetic outage".into())),
            Reply::Slow => tokio::time::sleep(Duration::from_secs(1)).await,
            _ => {}
        }
        Ok(Judgment {
            model: "synthetic-model-1".into(),
            answers: IndexMap::from([
                (
                    "sufficient".into(),
                    Answer::Noul { noul: if matches!(self.reply, Reply::Invalid) { 1.5 } else { 0.8 } },
                ),
                ("contradicted".into(), Answer::Noul { noul: 0.1 }),
            ]),
            usage: Usage { input_tokens: Some(100), output_tokens: Some(2), cost: Some(0.0001) },
            latency: Duration::ZERO,
        })
    }
}

#[tokio::test]
async fn assessment_uses_exact_context_and_persists_its_request_and_result() {
    let (_dir, path) = common::space();
    let store = Store::open(path, StoreConfig::new(2)).await.unwrap();
    let judge = TestJudge::new(true, Reply::Valid);
    let assessor = EvidenceAssessor::new(&store, &judge, AssessConfig::default()).unwrap();
    let question = "Where does Ada live?";
    let context = "Ada lives in Oslo. [S1:1]";
    let assessment = assessor.assess(question, context).await.unwrap();
    assert_eq!(assessment.status, AssessmentStatus::Judged);
    assert_eq!(assessment.sufficient, Some(0.8));
    assert_eq!(assessment.contradicted, Some(0.1));
    assert_eq!(assessment.model.as_deref(), Some("synthetic-model-1"));
    assert_eq!(assessment.input_tokens, Some(100));
    assert_eq!(assessment.cost, Some(0.0001));
    {
        let seen = judge.seen.lock().unwrap();
        assert_eq!(seen[0]["state"], json!({"question": question, "memory": context}));
        assert_eq!(seen[0]["questions"].as_object().unwrap().len(), 2);
    }
    let id = assessment.judgment_id.unwrap();
    store
        .read(move |c| {
            let (point, request, answers, hash): (String, String, String, Vec<u8>) =
                c.query_row("SELECT point, request, answers, request_hash FROM judgments WHERE id = ?1", [id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })?;
            assert_eq!(point, "J10");
            assert_eq!(hash, blake3::hash(request.as_bytes()).as_bytes());
            let request: Value = serde_json::from_str(&request).unwrap();
            assert_eq!(request["requested_model"], "synthetic-alias");
            assert_eq!(request["deadline_ms"], 1000);
            assert_eq!(serde_json::from_str::<Value>(&answers).unwrap()["assessment"]["sufficient"], 0.8);
            Ok(())
        })
        .await
        .unwrap();
    store.close().await.unwrap();
}

#[tokio::test]
async fn remote_permission_is_per_space_persistent_and_revocable() {
    let (_dir, path) = common::space();
    let store = Store::open(&path, StoreConfig::new(2)).await.unwrap();
    let judge = TestJudge::new(false, Reply::Valid);
    assert!(!remote_judgments_allowed(&store).await.unwrap());
    let assessment =
        EvidenceAssessor::new(&store, &judge, AssessConfig::default()).unwrap().assess("q", "memory").await.unwrap();
    assert_eq!(assessment.status, AssessmentStatus::NotAllowed);
    assert!(assessment.judgment_id.is_none());
    assert!(judge.seen.lock().unwrap().is_empty());
    allow_remote_judgments(&store, true).await.unwrap();
    store.close().await.unwrap();
    let store = Store::open(&path, StoreConfig::new(2)).await.unwrap();
    assert!(remote_judgments_allowed(&store).await.unwrap());
    let assessor = EvidenceAssessor::new(&store, &judge, AssessConfig::default()).unwrap();
    assert_eq!(assessor.assess("q", "memory").await.unwrap().status, AssessmentStatus::Judged);
    allow_remote_judgments(&store, false).await.unwrap();
    assert_eq!(assessor.assess("q", "memory").await.unwrap().status, AssessmentStatus::NotAllowed);
    assert_eq!(judge.seen.lock().unwrap().len(), 1);
    store.close().await.unwrap();
}

#[tokio::test]
async fn failed_slow_or_invalid_judges_preserve_unknown_scores_and_audit_failure() {
    for reply in [Reply::Error, Reply::Slow, Reply::Invalid] {
        let (_dir, path) = common::space();
        let store = Store::open(path, StoreConfig::new(2)).await.unwrap();
        let judge = TestJudge::new(true, reply);
        let cfg = AssessConfig { deadline: Duration::from_millis(20), ..AssessConfig::default() };
        let assessment = EvidenceAssessor::new(&store, &judge, cfg).unwrap().assess("q", "memory").await.unwrap();
        assert_eq!(assessment.status, AssessmentStatus::Unavailable);
        assert!(assessment.sufficient.is_none() && assessment.contradicted.is_none());
        assert!(assessment.error.is_some() && assessment.judgment_id.is_some());
        let count: i64 =
            store.read(|c| Ok(c.query_row("SELECT count(*) FROM judgments", [], |r| r.get(0))?)).await.unwrap();
        assert_eq!(count, 1);
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn oversized_context_is_never_silently_truncated_or_sent() {
    let (_dir, path) = common::space();
    let store = Store::open(path, StoreConfig::new(2)).await.unwrap();
    let judge = TestJudge::new(true, Reply::Valid);
    let cfg = AssessConfig { max_input_bytes: 8, ..AssessConfig::default() };
    let assessment =
        EvidenceAssessor::new(&store, &judge, cfg).unwrap().assess("q", "a context beyond the bound").await.unwrap();
    assert_eq!(assessment.status, AssessmentStatus::Unavailable);
    assert!(judge.seen.lock().unwrap().is_empty());
    assert!(assessment.judgment_id.is_none());
    store.close().await.unwrap();
}
