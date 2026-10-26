//! Offline, exact-input embeddings for controlled model comparisons.
//! A missing vector is an error, never a fallback to another embedding space.
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
struct File {
    model_id: String,
    dimensions: usize,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    kind: String,
    text: String,
    embedding: Vec<f32>,
}
pub struct CachedEmbedder {
    id: String,
    dim: usize,
    rows: HashMap<(bool, String), Vec<f32>>,
}
impl CachedEmbedder {
    pub fn load(path: &Path) -> Result<Self, EmbedError> {
        let text =
            std::fs::read_to_string(path).map_err(|_| EmbedError::Load("embedding cache could not be read".into()))?;
        Self::parse(&text)
    }
    fn parse(text: &str) -> Result<Self, EmbedError> {
        let file: File =
            serde_json::from_str(text).map_err(|_| EmbedError::Load("invalid embedding cache JSON".into()))?;
        if file.model_id.trim().is_empty() || file.dimensions == 0 || file.dimensions > 4096 {
            return Err(EmbedError::Load("embedding cache needs a model ID and dimension in 1..=4096".into()));
        }
        let mut rows = HashMap::new();
        for row in file.rows {
            let query = match row.kind.as_str() {
                "query" => true,
                "document" => false,
                _ => return Err(EmbedError::Load("unknown embedding kind".into())),
            };
            let norm = row.embedding.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>().sqrt();
            if row.embedding.len() != file.dimensions || !norm.is_finite() || norm == 0.0 {
                return Err(EmbedError::Load(
                    "embedding cache has a nonfinite, zero or wrong-dimensional vector".into(),
                ));
            }
            let vector = row.embedding.into_iter().map(|v| (f64::from(v) / norm) as f32).collect();
            if rows.insert((query, row.text), vector).is_some() {
                return Err(EmbedError::Load("duplicate embedding cache input".into()));
            }
        }
        Ok(Self { id: file.model_id, dim: file.dimensions, rows })
    }
}
#[async_trait::async_trait]
impl Embedder for CachedEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts
            .iter()
            .map(|text| {
                self.rows.get(&(kind == EmbedKind::Query, text.clone())).cloned().ok_or_else(|| {
                    EmbedError::Embed("exact input missing from the declared offline embedding cache".into())
                })
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn exact_text_and_kind_are_required_and_vectors_are_normalized() {
        let cache = CachedEmbedder::parse(
            &json!({"model_id":"fixture","dimensions":2,"rows":[{"kind":"query","text":"café","embedding":[3.,4.]}]})
                .to_string(),
        )
        .unwrap();
        assert_eq!(cache.embed(&["café".into()], EmbedKind::Query).await.unwrap(), vec![vec![0.6, 0.8]]);
        assert!(cache.embed(&["café".into()], EmbedKind::Document).await.is_err());
        assert!(cache.embed(&["cafe".into()], EmbedKind::Query).await.is_err());
    }
    #[test]
    fn malformed_vectors_and_duplicate_inputs_fail_before_retrieval() {
        for vector in [json!([0., 0.]), json!([1.]), json!([1e100, 1.])] {
            assert!(CachedEmbedder::parse(&json!({"model_id":"fixture","dimensions":2,"rows":[{"kind":"query","text":"q","embedding":vector}]}).to_string()).is_err());
        }
        let row = json!({"kind":"query","text":"q","embedding":[1.,0.]});
        assert!(
            CachedEmbedder::parse(&json!({"model_id":"fixture","dimensions":2,"rows":[row,row]}).to_string()).is_err()
        );
    }
}
