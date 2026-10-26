//! Explicit, budgeted text-embedding-v4. Never substitutes another model.
use super::{EmbedError, EmbedKind, Embedder};
use crate::chat::ChatError;
use crate::chat_http::BudgetChat;
use serde_json::{Value, json};
use std::path::Path;

const MODEL: &str = "text-embedding-v4";
const PATH: &str = "/api/v1/services/embeddings/text-embedding/text-embedding";
const PRICE_SOURCE: &str = "https://www.alibabacloud.com/help/en/model-studio/text-embedding-synchronous-api";

pub struct DashScopeV4 {
    transport: BudgetChat,
    endpoint: String,
    id: String,
    dimensions: usize,
    input_rate: f64,
}
impl DashScopeV4 {
    /// `base_url` is a documented workspace/region URL, ending in `/api/v1`
    /// or `/compatible-mode/v1`. The native endpoint distinguishes query/document.
    pub fn open(
        key: String,
        base_url: &str,
        ledger: &Path,
        ceiling: f64,
        dimensions: usize,
    ) -> Result<Self, EmbedError> {
        let (endpoint, rate) = endpoint(base_url)?;
        Self::configured(key, &endpoint, ledger, ceiling, dimensions, rate)
    }
    /// Test transport is restricted to HTTP loopback and cannot select a model.
    pub fn local_test(
        key: String,
        endpoint: &str,
        ledger: &Path,
        ceiling: f64,
        dimensions: usize,
    ) -> Result<Self, EmbedError> {
        let url = reqwest::Url::parse(endpoint).map_err(|_| load("invalid local endpoint"))?;
        if url.scheme() != "http"
            || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(load("test endpoint must be HTTP loopback without credentials, query or fragment"));
        }
        Self::configured(key, endpoint, ledger, ceiling, dimensions, 0.072)
    }
    fn configured(
        key: String,
        endpoint: &str,
        ledger: &Path,
        ceiling: f64,
        dimensions: usize,
        input_rate: f64,
    ) -> Result<Self, EmbedError> {
        if ![64, 128, 256, 512, 768, 1024, 1536, 2048].contains(&dimensions) {
            return Err(load("text-embedding-v4 dimension must be 64,128,256,512,768,1024,1536 or2048"));
        }
        let transport = BudgetChat::at_endpoint(key, ledger, ceiling, endpoint).map_err(convert)?;
        let host = reqwest::Url::parse(endpoint)
            .map_err(|_| load("invalid endpoint"))?
            .host_str()
            .unwrap_or_default()
            .to_owned();
        Ok(Self {
            transport,
            endpoint: endpoint.into(),
            id: format!("dashscope/{MODEL}/{dimensions}/query-document-dense-v1/{host}"),
            dimensions,
            input_rate,
        })
    }
    pub fn accounted(&self) -> Result<f64, EmbedError> {
        self.transport.accounted().map_err(convert)
    }
    async fn batch(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = json!({"model":MODEL,"input":{"texts":texts},"parameters":{"text_type":if kind==EmbedKind::Query {"query"} else {"document"},"dimension":self.dimensions,"output_type":"dense"}});
        let identity = json!({"provider":"dashscope","task":"embedding-v4-native-v1","endpoint":self.endpoint,"model":MODEL,"body":body,"input_rate":self.input_rate,"price_source":PRICE_SOURCE});
        // Byte count deliberately bounds token usage without claiming a different
        // model's tokenizer. Provider enforces its8192-token limit. Full inputs
        // are sent; rejection never triggers truncation or another model.
        let reservation = (body.to_string().len() + 2048) as f64 * self.input_rate * 1.25 / 1e6;
        let output = self
            .transport
            .json_request(
                &identity,
                &body,
                reservation,
                |raw| {
                    parse(raw, texts.len(), self.dimensions).and_then(|v| {
                        serde_json::to_value(v).map_err(|_| ChatError("embedding serialization failed".into()))
                    })
                },
                |raw| raw["usage"]["total_tokens"].as_u64().map(|n| n as f64 * self.input_rate / 1e6),
            )
            .await
            .map_err(convert)?;
        let vectors: Vec<Vec<f32>> =
            serde_json::from_value(output).map_err(|_| failure("invalid cached embedding output"))?;
        if vectors.len() != texts.len()
            || vectors
                .iter()
                .any(|v| v.len() != self.dimensions || v.iter().any(|n| !n.is_finite()) || !v.iter().any(|n| *n != 0.))
        {
            return Err(failure("invalid cached embedding shape"));
        }
        Ok(vectors)
    }
}
#[async_trait::async_trait]
impl Embedder for DashScopeV4 {
    fn id(&self) -> &str {
        &self.id
    }
    fn dim(&self) -> usize {
        self.dimensions
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        // Validate the entire operation before sending its first paid batch.
        if texts.iter().any(|text| text.trim().is_empty()) {
            return Err(failure("embedding inputs must be nonempty text"));
        }
        let mut result = Vec::with_capacity(texts.len());
        for batch in texts.chunks(10) {
            result.extend(self.batch(batch, kind).await?);
        }
        Ok(result)
    }
}
fn endpoint(base: &str) -> Result<(String, f64), EmbedError> {
    let mut url = reqwest::Url::parse(base).map_err(|_| load("invalid DashScope base URL"))?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || url.port().is_some_and(|p| p != 443)
    {
        return Err(load("DashScope requires an official HTTPS workspace base without credentials, query or fragment"));
    }
    let host = url.host_str().unwrap_or_default();
    let suffixes = [
        (".ap-southeast-1.maas.aliyuncs.com", 0.07),
        (".cn-beijing.maas.aliyuncs.com", 0.072),
        (".cn-hongkong.maas.aliyuncs.com", 0.07),
    ];
    let rate = suffixes
        .into_iter()
        .find_map(|(suffix, rate)| {
            let workspace = host.strip_suffix(suffix)?;
            (!workspace.is_empty()
                && workspace.len() <= 63
                && workspace.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'))
            .then_some(rate)
        })
        .ok_or_else(|| load("DashScope base must use a documented workspace and region host"))?;
    if !matches!(url.path().trim_end_matches('/'), "/api/v1" | "/compatible-mode/v1") && url.path() != PATH {
        return Err(load("DashScope base path must be /api/v1 or /compatible-mode/v1"));
    }
    url.set_path(PATH);
    Ok((url.into(), rate))
}
fn parse(raw: &Value, count: usize, dimensions: usize) -> Result<Vec<Vec<f32>>, ChatError> {
    let error = || ChatError("invalid text-embedding-v4 response indices, dimensions or values".into());
    if raw.get("model").is_some_and(|v| v != MODEL)
        || raw.get("status_code").is_some_and(|v| v != 200)
        || raw.get("code").is_some_and(|v| !v.is_null() && v != "")
    {
        return Err(error());
    }
    let rows = raw["output"]["embeddings"].as_array().ok_or_else(error)?;
    if rows.len() != count {
        return Err(error());
    }
    let mut result = vec![None; count];
    for row in rows {
        let index = row["text_index"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n < count)
            .ok_or_else(error)?;
        if result[index].is_some() {
            return Err(error());
        }
        let values = row["embedding"].as_array().filter(|v| v.len() == dimensions).ok_or_else(error)?;
        let vector: Vec<f64> =
            values.iter().map(|v| v.as_f64().filter(|n| n.is_finite()).ok_or_else(error)).collect::<Result<_, _>>()?;
        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0. {
            return Err(error());
        }
        result[index] = Some(vector.into_iter().map(|v| (v / norm) as f32).collect());
    }
    result.into_iter().map(|v| v.ok_or_else(error)).collect()
}
fn convert(error: ChatError) -> EmbedError {
    EmbedError::Embed(error.0)
}
fn load(message: &str) -> EmbedError {
    EmbedError::Load(message.into())
}
fn failure(message: &str) -> EmbedError {
    EmbedError::Embed(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_hosts_paths_and_regional_rates_are_explicit() {
        for (region, rate) in [("ap-southeast-1", 0.07), ("cn-beijing", 0.072), ("cn-hongkong", 0.07)] {
            let base = format!("https://workspace.{region}.maas.aliyuncs.com/compatible-mode/v1");
            let (url, actual) = endpoint(&base).unwrap();
            assert!(url.ends_with(PATH));
            assert_eq!(actual, rate);
        }
        for base in [
            "http://workspace.cn-beijing.maas.aliyuncs.com/api/v1",
            "https://evil.example/api/v1",
            "https://workspace.cn-beijing.maas.aliyuncs.com.evil.example/api/v1",
            "https://user:secret@workspace.cn-beijing.maas.aliyuncs.com/api/v1",
            "https://workspace.cn-beijing.maas.aliyuncs.com/api/v1?key=secret",
            "https://workspace.cn-beijing.maas.aliyuncs.com/other",
        ] {
            assert!(endpoint(base).is_err());
        }
    }
    #[test]
    fn exact_index_mapping_normalization_and_rejection() {
        let raw = json!({"output":{"embeddings":[{"text_index":1,"embedding":[0.,2.]},{"text_index":0,"embedding":[3.,4.]}]}});
        assert_eq!(parse(&raw, 2, 2).unwrap(), vec![vec![0.6, 0.8], vec![0., 1.]]);
        for bad in [
            json!({"output":{"embeddings":[{"text_index":0,"embedding":[1.,0.]},{"text_index":0,"embedding":[0.,1.]}]}}),
            json!({"output":{"embeddings":[{"text_index":0,"embedding":[0.,0.]}]}}),
            json!({"output":{"embeddings":[{"text_index":0,"embedding":[1.]}]}}),
            json!({"code":"InvalidApiKey","output":{"embeddings":[{"text_index":0,"embedding":[1.,0.]}]}}),
        ] {
            assert!(parse(&bad, 1, 2).is_err());
        }
    }
}
