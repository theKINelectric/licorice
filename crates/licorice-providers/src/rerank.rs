//! Optional bounded query/candidate scoring; neither answers nor gold labels enter.
use crate::chat::{ChatError, ChatModel, ModelRequest, validate as validate_schema};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub id: i64,
    pub text: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ranked {
    pub id: i64,
    pub score: f64,
}
/// Query interpretation metadata, never additional source evidence.
/// Day precision matches native temporal retrieval and avoids cache churn within a day.
#[derive(Clone, Debug, Serialize)]
pub struct RerankContext {
    pub reference_date_utc: String,
    pub speaker: Option<String>,
}
#[async_trait::async_trait]
pub trait Reranker: Send + Sync {
    fn id(&self) -> &str;
    async fn rank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Ranked>, ChatError>;
    /// Custom scorers retain their existing raw-query contract by default.
    async fn rank_with_context(
        &self,
        query: &str,
        candidates: &[Candidate],
        _context: &RerankContext,
    ) -> Result<Vec<Ranked>, ChatError> {
        self.rank(query, candidates).await
    }
}
pub struct ChatReranker(pub Arc<dyn ChatModel>);
/// Optional larger pool, preserving the underlying scorer's 24-candidate grammar.
pub struct BatchedReranker(pub Arc<dyn Reranker>);
#[async_trait::async_trait]
impl Reranker for BatchedReranker {
    fn id(&self) -> &str {
        "query-fact-scorer-batched-v1/24x4"
    }
    async fn rank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Ranked>, ChatError> {
        self.rank_optional_context(query, candidates, None).await
    }
    async fn rank_with_context(
        &self,
        query: &str,
        candidates: &[Candidate],
        context: &RerankContext,
    ) -> Result<Vec<Ranked>, ChatError> {
        self.rank_optional_context(query, candidates, Some(context)).await
    }
}
impl BatchedReranker {
    async fn rank_optional_context(
        &self,
        query: &str,
        candidates: &[Candidate],
        context: Option<&RerankContext>,
    ) -> Result<Vec<Ranked>, ChatError> {
        if candidates.len() > 96 {
            return Err(ChatError("batched reranker candidate limit is 96".into()));
        }
        if candidates.iter().map(|c| c.id).collect::<std::collections::HashSet<_>>().len() != candidates.len() {
            return Err(ChatError("reranker candidate IDs must be unique".into()));
        }
        let mut pending = tokio::task::JoinSet::new();
        for chunk in candidates.chunks(24) {
            let (scorer, query, chunk) = (self.0.clone(), query.to_owned(), chunk.to_vec());
            let context = context.cloned();
            pending.spawn(async move {
                let ranked = match context {
                    Some(context) => scorer.rank_with_context(&query, &chunk, &context).await?,
                    None => scorer.rank(&query, &chunk).await?,
                };
                validate(&chunk, ranked)
            });
        }
        let (mut rows, mut error) = (Vec::new(), None);
        // Drain all started work, including on a failed provider/contract result.
        // A caller either receives the complete pool or one explicit failure.
        while let Some(result) = pending.join_next().await {
            match result {
                Ok(Ok(batch)) => rows.extend(batch),
                Ok(Err(failed)) => {
                    if error.is_none() {
                        error = Some(failed);
                    }
                }
                Err(_) => {
                    if error.is_none() {
                        error = Some(ChatError("scorer batch task failed".into()));
                    }
                }
            }
        }
        if let Some(error) = error {
            return Err(error);
        }
        validate(candidates, rows)
    }
}
#[async_trait::async_trait]
impl Reranker for ChatReranker {
    fn id(&self) -> &str {
        "query-fact-scorer-v2-keyed/gpt-4o-mini"
    }
    async fn rank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Ranked>, ChatError> {
        self.rank_optional_context(query, candidates, None).await
    }
    async fn rank_with_context(
        &self,
        query: &str,
        candidates: &[Candidate],
        context: &RerankContext,
    ) -> Result<Vec<Ranked>, ChatError> {
        self.rank_optional_context(query, candidates, Some(context)).await
    }
}
impl ChatReranker {
    async fn rank_optional_context(
        &self,
        query: &str,
        candidates: &[Candidate],
        context: Option<&RerankContext>,
    ) -> Result<Vec<Ranked>, ChatError> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        if candidates.len() > 24 {
            return Err(ChatError("reranker candidate limit is 24".into()));
        }
        if candidates.iter().map(|c| c.id).collect::<std::collections::HashSet<_>>().len() != candidates.len() {
            return Err(ChatError("reranker candidate IDs must be unique".into()));
        }
        // Bind the grammar to the supplied IDs. A free-ID array can satisfy its
        // schema while omitting candidates, duplicating IDs or inventing ordinals.
        let properties: serde_json::Map<String, serde_json::Value> =
            candidates.iter().map(|c| (c.id.to_string(), json!({"type":"number","minimum":0,"maximum":1}))).collect();
        let required: Vec<_> = properties.keys().cloned().collect();
        let mut payload = json!({"query":query,"candidates":candidates});
        if let Some(context) = context {
            payload["context"] = serde_json::to_value(context).expect("serializable query context");
        }
        let user = payload.to_string();
        let request=ModelRequest{id:format!("rerank-v2:{}",blake3::hash(user.as_bytes()).to_hex()),task:"rerank".into(),
    system:"Score every candidate's usefulness as evidence for the query from 0 to 1. Preserve connecting relationships needed for multi-hop questions. Do not answer the question. Use only candidate text; do not invent facts. Return scores as an object keyed by the exact supplied candidate IDs, with every required key once.".into(),user,
    schema:Some(json!({"type":"object","additionalProperties":false,"required":["scores"],"properties":{"scores":{"type":"object","additionalProperties":false,"required":required,"properties":properties}}})),max_tokens:1024};
        let mut request = request;
        if context.is_some() {
            request.system.push_str(" The context reference_date_utc anchors relative query dates; context speaker, when present, is the uniquely resolved first-person referent. These fields interpret the query and are not source evidence.");
        }
        let original = request;
        let mut request = original.clone();
        let mut last_error = ChatError("reranker validation failed".into());
        for attempt in 0..3 {
            let output = self.0.complete(&request).await?;
            let result = validate_schema(original.schema.as_ref().unwrap(), &output)
                .map_err(|_| {
                    ChatError("reranker output does not match the exact candidate keys and score bounds".into())
                })
                .and_then(|()| {
                    let rows = candidates
                        .iter()
                        .map(|c| Ranked {
                            id: c.id,
                            // Schema validation establishes the exact numeric map.
                            score: output["scores"][c.id.to_string()].as_f64().unwrap(),
                        })
                        .collect();
                    validate(candidates, rows)
                });
            match result {
                Ok(rows) => return Ok(rows),
                Err(error) => {
                    last_error = error;
                    request = original.clone();
                    request.id = format!("{}:repair:{}", original.id, attempt + 1);
                    request.user.push_str("\nReturn the required scores object with every original candidate ID as a key, no other keys, and finite numeric values between 0 and 1.");
                }
            }
        }
        Err(last_error)
    }
}
pub fn validate(candidates: &[Candidate], mut rows: Vec<Ranked>) -> Result<Vec<Ranked>, ChatError> {
    let wanted: std::collections::HashSet<_> = candidates.iter().map(|c| c.id).collect();
    let actual: std::collections::HashSet<_> = rows.iter().map(|c| c.id).collect();
    if wanted.len() != candidates.len()
        || rows.len() != candidates.len()
        || actual != wanted
        || rows.iter().any(|r| !r.score.is_finite() || !(0.0..=1.).contains(&r.score))
    {
        return Err(ChatError("reranker must score the exact candidate set once with finite scores in [0,1]".into()));
    }
    rows.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
    Ok(rows)
}
