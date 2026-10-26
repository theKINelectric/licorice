//! Bounded transport to an operator-owned, local ONNX embedding sidecar.
//! There is no model download, provider credential or external API fallback here.
use super::{EmbedError, EmbedKind, Embedder};
use serde_json::{Value, json};
use std::{net::IpAddr, time::Duration};

const MAX_TEXT_BYTES: usize = 4000;
const MAX_TEXTS: usize = 128;
const BATCH: usize = 32;
const MAX_OPERATION_BYTES: usize = 512 * 1024;
const MAX_RESPONSE_BYTES: usize = 4 * 1024 * 1024;
const DEADLINE: Duration = Duration::from_secs(10);

/// OpenAI-compatible `/v1/embeddings` JSON, restricted to literal private IPs.
/// The sidecar must echo the exact configured model and reject unsupported input
/// rather than substitute models. Prefixes are part of the recorded identity.
pub struct LocalOnnx {
    client: reqwest::Client,
    endpoint: reqwest::Url,
    model: String,
    dimensions: usize,
    query_prefix: String,
    document_prefix: String,
    id: String,
}

impl LocalOnnx {
    pub fn new(
        origin: &str,
        model: &str,
        dimensions: usize,
        query_prefix: &str,
        document_prefix: &str,
    ) -> Result<Self, EmbedError> {
        let mut endpoint = reqwest::Url::parse(origin).map_err(|_| load("invalid local embedding origin"))?;
        let host = endpoint.host_str().unwrap_or_default().trim_start_matches('[').trim_end_matches(']');
        let private = host.parse::<IpAddr>().ok().is_some_and(private_address);
        if endpoint.scheme() != "http"
            || !private
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || !matches!(endpoint.path(), "" | "/")
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            return Err(load(
                "local embedding origin must be HTTP at a literal loopback or private IP, without credentials, path, query or fragment",
            ));
        }
        if model.is_empty()
            || model.len() > 256
            || !model.bytes().all(|b| b.is_ascii_alphanumeric() || b"/_.:-".contains(&b))
            || !(1..=4096).contains(&dimensions)
            || query_prefix.len() > 512
            || document_prefix.len() > 512
        {
            return Err(load("invalid local embedding model, dimensions or prefixes"));
        }
        endpoint.set_path("/v1/embeddings");
        // Environment proxy settings must not reroute private memory off-host.
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .connect_timeout(Duration::from_secs(3))
            .timeout(DEADLINE)
            .build()
            .map_err(|_| load("local embedding transport initialization failed"))?;
        let prefix_identity = serde_json::to_vec(&(query_prefix, document_prefix))
            .map_err(|_| load("local embedding prefix serialization failed"))?;
        let digest = blake3::hash(&prefix_identity).to_hex();
        Ok(Self {
            client,
            endpoint,
            model: model.into(),
            dimensions,
            query_prefix: query_prefix.into(),
            document_prefix: document_prefix.into(),
            id: format!("local-onnx/{model}/{dimensions}/prefixes-{digest}"),
        })
    }

    async fn batches(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut output = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let body = json!({"model":self.model,"input":batch,"encoding_format":"float"});
            let mut response = self
                .client
                .post(self.endpoint.clone())
                .json(&body)
                .send()
                .await
                .map_err(|_| failure("local embedding request failed"))?;
            if response.status() != reqwest::StatusCode::OK {
                return Err(failure("local embedding sidecar returned an unsuccessful status"));
            }
            if response.content_length().is_some_and(|n| n > MAX_RESPONSE_BYTES as u64) {
                return Err(failure("local embedding response exceeds byte limit"));
            }
            let mut bytes = Vec::new();
            while let Some(chunk) =
                response.chunk().await.map_err(|_| failure("local embedding response read failed"))?
            {
                if bytes.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                    return Err(failure("local embedding response exceeds byte limit"));
                }
                bytes.extend_from_slice(&chunk);
            }
            let raw: Value = serde_json::from_slice(&bytes).map_err(|_| failure("invalid local embedding JSON"))?;
            output.extend(parse(&raw, &self.model, batch.len(), self.dimensions)?);
        }
        Ok(output)
    }
}

#[async_trait::async_trait]
impl Embedder for LocalOnnx {
    fn id(&self) -> &str {
        &self.id
    }
    fn dim(&self) -> usize {
        self.dimensions
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.len() > MAX_TEXTS || texts.iter().any(|s| s.trim().is_empty() || s.len() > MAX_TEXT_BYTES) {
            return Err(failure("local embedding inputs exceed nonempty text or batch bounds"));
        }
        let prefix = if kind == EmbedKind::Query { &self.query_prefix } else { &self.document_prefix };
        let inputs = texts.iter().map(|text| format!("{prefix}{text}")).collect::<Vec<_>>();
        if inputs.iter().map(String::len).sum::<usize>() > MAX_OPERATION_BYTES {
            return Err(failure("local embedding inputs exceed total UTF-8 byte limit"));
        }
        tokio::time::timeout(DEADLINE, self.batches(&inputs))
            .await
            .map_err(|_| failure("local embedding operation deadline exceeded"))?
    }
}

fn private_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback() || ip.is_private(),
        IpAddr::V6(ip) => {
            ip.is_loopback()
                || ip.is_unique_local()
                || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback() || v4.is_private())
        }
    }
}

fn parse(raw: &Value, model: &str, count: usize, dimensions: usize) -> Result<Vec<Vec<f32>>, EmbedError> {
    let bad = || failure("local embedding response model, indices, dimensions or vector values mismatch");
    if raw.get("model").and_then(Value::as_str) != Some(model) || raw.get("error").is_some() {
        return Err(bad());
    }
    let rows = raw.get("data").and_then(Value::as_array).filter(|rows| rows.len() == count).ok_or_else(bad)?;
    let mut result = vec![None; count];
    for row in rows {
        let index = row
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|index| *index < count)
            .ok_or_else(bad)?;
        if result[index].is_some() {
            return Err(bad());
        }
        let values =
            row.get("embedding").and_then(Value::as_array).filter(|v| v.len() == dimensions).ok_or_else(bad)?;
        let values = values
            .iter()
            .map(|v| v.as_f64().filter(|n| n.is_finite()).ok_or_else(bad))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = values.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(bad());
        }
        result[index] = Some(values.into_iter().map(|v| (v / norm) as f32).collect::<Vec<_>>());
    }
    result.into_iter().map(|v| v.ok_or_else(bad)).collect()
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
    use axum::{Json, Router, body::Body, http::StatusCode, response::Response, routing::post};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (origin, task)
    }

    #[test]
    fn local_origin_and_configuration_are_explicit() {
        for origin in [
            "http://127.0.0.1:8081",
            "http://10.10.0.2:8081",
            "http://172.20.0.2:8081",
            "http://192.168.2.4:8081",
            "http://[::1]:8081",
            "http://[fd00::2]:8081",
        ] {
            assert!(LocalOnnx::new(origin, "BAAI/bge-small-en-v1.5", 384, "query: ", "").is_ok(), "{origin}");
        }
        for origin in [
            "https://127.0.0.1",
            "http://localhost:8081",
            "http://8.8.8.8:8081",
            "http://169.254.169.254",
            "http://172.32.0.2",
            "http://0.0.0.0",
            "http://user:password@127.0.0.1",
            "http://127.0.0.1/path",
            "http://127.0.0.1?redirect=1",
            "http://127.0.0.1#fragment",
        ] {
            assert!(LocalOnnx::new(origin, "model", 384, "", "").is_err(), "{origin}");
        }
        assert!(LocalOnnx::new("http://127.0.0.1", "model", 0, "", "").is_err());
        assert!(LocalOnnx::new("http://127.0.0.1", "model", 4097, "", "").is_err());
        assert!(LocalOnnx::new("http://127.0.0.1", "model", 384, &"x".repeat(513), "").is_err());
        assert_ne!(
            LocalOnnx::new("http://127.0.0.1", "model", 384, "query: ", "").unwrap().id(),
            LocalOnnx::new("http://127.0.0.1", "model", 384, "", "").unwrap().id()
        );
    }

    #[test]
    fn exact_model_index_mapping_and_vector_validation() {
        let valid = json!({"model":"m","data":[{"index":1,"embedding":[0.0,2.0]},{"index":0,"embedding":[3.0,4.0]}]});
        assert_eq!(parse(&valid, "m", 2, 2).unwrap(), [vec![0.6, 0.8], vec![0.0, 1.0]]);
        for bad in [
            json!({"model":"other","data":[{"index":0,"embedding":[1,0]}]}),
            json!({"data":[{"index":0,"embedding":[1,0]}]}),
            json!({"model":"m","data":[{"index":1,"embedding":[1,0]}]}),
            json!({"model":"m","data":[{"index":0,"embedding":[0,0]}]}),
            json!({"model":"m","data":[{"index":0,"embedding":[1]}]}),
            json!({"model":"m","data":[{"index":0,"embedding":["1",0]}]}),
            json!({"model":"m","error":"failed","data":[{"index":0,"embedding":[1,0]}]}),
        ] {
            assert!(parse(&bad, "m", 1, 2).is_err());
        }
        assert!(
            parse(
                &json!({"model":"m","data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]}),
                "m",
                2,
                2
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn http_batches_preserve_prefixes_indices_and_reject_all_invalid_input_before_io() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let capture = seen.clone();
        let app = Router::new().route(
            "/v1/embeddings",
            post(move |Json(body): Json<Value>| {
                let capture = capture.clone();
                async move {
                    capture.lock().unwrap().push(body.clone());
                    let rows = body["input"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .rev()
                        .map(|(i, _)| json!({"index":i,"embedding":[3,4]}))
                        .collect::<Vec<_>>();
                    Json(json!({"model":"m","data":rows}))
                }
            }),
        );
        let (origin, task) = serve(app).await;
        let model = LocalOnnx::new(&origin, "m", 2, "query: ", "document: ").unwrap();
        let inputs = (0..50).map(|i| format!("report {i}")).collect::<Vec<_>>();
        assert_eq!(model.embed(&inputs, EmbedKind::Document).await.unwrap(), vec![vec![0.6, 0.8]; 50]);
        model.embed(&["question".into()], EmbedKind::Query).await.unwrap();
        let calls = seen.lock().unwrap().clone();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0]["input"].as_array().unwrap().len(), 32);
        assert_eq!(calls[1]["input"].as_array().unwrap().len(), 18);
        assert_eq!(calls[0]["input"][0], "document: report 0");
        assert_eq!(calls[2]["input"][0], "query: question");
        assert_eq!(calls[0]["encoding_format"], "float");
        assert!(model.embed(&vec!["small".into(); 129], EmbedKind::Document).await.is_err());
        assert!(model.embed(&["ok".into(), " ".into()], EmbedKind::Document).await.is_err());
        assert!(model.embed(&["界".repeat(1334)], EmbedKind::Document).await.is_err());
        assert!(model.embed(&[], EmbedKind::Document).await.unwrap().is_empty());
        assert_eq!(seen.lock().unwrap().len(), 3, "invalid/empty input sends nothing");
        task.abort();
    }

    #[tokio::test]
    async fn redirects_are_never_followed_and_error_bodies_are_not_disclosed() {
        let followed = Arc::new(AtomicUsize::new(0));
        let capture = followed.clone();
        let app = Router::new()
            .route(
                "/v1/embeddings",
                post(|| async {
                    Response::builder()
                        .status(StatusCode::TEMPORARY_REDIRECT)
                        .header("location", "/redirect-target")
                        .body(Body::from("PRIVATE_INPUT_WITNESS"))
                        .unwrap()
                }),
            )
            .route(
                "/redirect-target",
                post(move || {
                    let capture = capture.clone();
                    async move {
                        capture.fetch_add(1, Ordering::SeqCst);
                        Json(json!({"model":"m","data":[{"index":0,"embedding":[1,0]}]}))
                    }
                }),
            );
        let (origin, task) = serve(app).await;
        let model = LocalOnnx::new(&origin, "m", 2, "", "").unwrap();
        let error = model.embed(&["text".into()], EmbedKind::Query).await.unwrap_err();
        assert!(!error.to_string().contains("PRIVATE_INPUT_WITNESS"));
        assert_eq!(followed.load(Ordering::SeqCst), 0);
        task.abort();
    }

    #[tokio::test]
    async fn response_limit_counts_streamed_bytes_without_content_length() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut input = [0u8; 8192];
            let _ = socket.read(&mut input).await;
            if socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").await.is_err(){return;}
            let chunk = vec![b' '; 4096];
            for _ in 0..=MAX_RESPONSE_BYTES / 4096 {
                if socket.write_all(b"1000\r\n").await.is_err()
                    || socket.write_all(&chunk).await.is_err()
                    || socket.write_all(b"\r\n").await.is_err()
                {
                    return;
                }
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        });
        let model = LocalOnnx::new(&origin, "m", 2, "", "").unwrap();
        let error = model.embed(&["text".into()], EmbedKind::Query).await.unwrap_err();
        assert!(error.to_string().contains("byte limit"), "{error}");
        task.abort();
    }

    #[tokio::test]
    async fn whole_operation_has_a_real_deadline() {
        let app = Router::new().route(
            "/v1/embeddings",
            post(|| async {
                tokio::time::sleep(Duration::from_secs(11)).await;
                Json(json!({"model":"m","data":[{"index":0,"embedding":[1,0]}]}))
            }),
        );
        let (origin, task) = serve(app).await;
        let model = LocalOnnx::new(&origin, "m", 2, "", "").unwrap();
        let before = std::time::Instant::now();
        assert!(model.embed(&["text".into()], EmbedKind::Query).await.is_err());
        assert!(before.elapsed() < Duration::from_secs(11));
        task.abort();
    }
}
