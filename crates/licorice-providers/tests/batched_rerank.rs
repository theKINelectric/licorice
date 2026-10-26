//! Larger scoring pools retain per-call bounds and fail without partial rankings.
use licorice_providers::chat::ChatError;
use licorice_providers::rerank::{BatchedReranker, Candidate, Ranked, Reranker};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
struct BoundedScorer {
    calls: Mutex<Vec<Vec<i64>>>,
    active: AtomicUsize,
    peak: AtomicUsize,
    fail: bool,
}
impl BoundedScorer {
    fn new(fail: bool) -> Self {
        Self { calls: Mutex::new(vec![]), active: AtomicUsize::new(0), peak: AtomicUsize::new(0), fail }
    }
}
#[async_trait::async_trait]
impl Reranker for BoundedScorer {
    fn id(&self) -> &str {
        "bounded-24-control"
    }
    async fn rank(&self, _: &str, candidates: &[Candidate]) -> Result<Vec<Ranked>, ChatError> {
        if candidates.len() > 24 {
            return Err(ChatError("per-call cap exceeded".into()));
        }
        self.calls.lock().unwrap().push(candidates.iter().map(|c| c.id).collect());
        let n = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(n, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if self.fail && candidates.iter().any(|c| c.id == 1024) {
            return Ok(vec![Ranked { id: -1, score: 0.5 }]);
        }
        Ok(candidates.iter().map(|c| Ranked { id: c.id, score: 0.5 }).rev().collect())
    }
}
fn candidates(n: usize) -> Vec<Candidate> {
    (0..n).map(|i| Candidate { id: 1000 + i as i64, text: format!("evidence {i}") }).collect()
}
#[tokio::test]
async fn sixty_candidates_are_scored_once_in_bounded_batches_and_merged_stably() {
    let inner = Arc::new(BoundedScorer::new(false));
    let scorer = BatchedReranker(inner.clone());
    let input = candidates(60);
    let ranked = scorer.rank("query", &input).await.unwrap();
    assert_eq!(ranked.iter().map(|r| r.id).collect::<Vec<_>>(), input.iter().map(|c| c.id).collect::<Vec<_>>());
    let mut calls = inner.calls.lock().unwrap().clone();
    calls.sort_by_key(|c| c[0]);
    assert_eq!(calls.iter().map(Vec::len).collect::<Vec<_>>(), vec![24, 24, 12]);
    assert!(inner.peak.load(Ordering::SeqCst) <= 4);
    assert_eq!(inner.active.load(Ordering::SeqCst), 0);
}
#[tokio::test]
async fn invalid_pool_inputs_make_no_calls_and_the_largest_allowed_pool_stays_bounded() {
    let inner = Arc::new(BoundedScorer::new(false));
    let scorer = BatchedReranker(inner.clone());
    assert!(scorer.rank("q", &candidates(97)).await.is_err());
    let mut repeated = candidates(60);
    repeated[25].id = repeated[0].id;
    assert!(scorer.rank("q", &repeated).await.is_err());
    assert!(scorer.rank("q", &[]).await.unwrap().is_empty());
    assert!(inner.calls.lock().unwrap().is_empty());
    assert_eq!(scorer.rank("q", &candidates(96)).await.unwrap().len(), 96);
    assert_eq!(inner.calls.lock().unwrap().len(), 4);
}
#[tokio::test]
async fn one_invalid_batch_rejects_the_entire_ranking_and_drains_started_work() {
    let inner = Arc::new(BoundedScorer::new(true));
    let scorer = BatchedReranker(inner.clone());
    assert!(scorer.rank("q", &candidates(60)).await.is_err());
    assert_eq!(inner.calls.lock().unwrap().len(), 3, "all dispatched batches must reach a terminal state");
    assert_eq!(inner.active.load(Ordering::SeqCst), 0, "no unobserved provider work may outlive a failed ranking");
}
