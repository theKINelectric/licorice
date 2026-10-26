//! HTTP judge for the System One wire format.
//!
//! Decoding is driven by the question that was asked, not by the answer's `type`
//! field, so the same client reads TypeSafe's v1 answers and Coffee Cat's research
//! engine (`coffee-cat/0.2`, which omits `type` and reports its model under `usage`).

use std::fmt;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::sync::Semaphore;

use super::{Answer, Judge, JudgeError, Judgment, Point, Question, Usage};

/// Where and how to reach a judge.
#[derive(Clone)]
pub struct HttpJudgeConfig {
    /// For example `http://127.0.0.1:4777` or `https://api.typesafe.ai`.
    pub base_url: String,
    /// `/v1/systemone` for Jev; `/api/v1/decisions` for production Coffee Cat.
    pub path: String,
    pub protocol: HttpJudgeProtocol,
    /// Sent as `model`; `None` lets the server use its default.
    pub model: Option<String>,
    /// Sent as `Authorization: Bearer …` when set.
    pub api_key: Option<String>,
    /// Larger requests are split (SPEC §10.2: Coffee Cat is fastest up to ~15).
    pub max_questions_per_request: usize,
    /// Requests in flight at once; local engines run one forward pass at a time.
    pub max_concurrent: usize,
}

/// Coffee Cat 0.3.1 accepts string instructions/criteria and a server deadline.
/// Jev also accepts structured instructions and does not document `deadline_ms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpJudgeProtocol {
    Jev,
    CoffeeCat,
    /// Inception's typed decisions endpoint; no Coffee Cat deadline field.
    Mercury,
}

impl fmt::Debug for HttpJudgeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpJudgeConfig")
            .field("base_url", &self.base_url)
            .field("path", &self.path)
            .field("protocol", &self.protocol)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("max_questions_per_request", &self.max_questions_per_request)
            .field("max_concurrent", &self.max_concurrent)
            .finish()
    }
}

impl HttpJudgeConfig {
    /// Coffee Cat's decision service (release 0.3.1) on its default local port.
    pub fn coffee_cat() -> HttpJudgeConfig {
        HttpJudgeConfig {
            base_url: "http://127.0.0.1:4777".into(),
            path: "/api/v1/decisions".into(),
            protocol: HttpJudgeProtocol::CoffeeCat,
            model: None,
            api_key: None,
            max_questions_per_request: 15,
            max_concurrent: 1,
        }
    }

    /// TypeSafe's hosted Jev. The key is read from the environment, never from config
    /// files (SPEC §10.2: hosted judgments are opt-in per space).
    pub fn jev_from_env(var: &str) -> Result<HttpJudgeConfig, JudgeError> {
        let key = std::env::var(var).map_err(|_| JudgeError::Invalid(format!("{var} is not set")))?;
        if key.trim().is_empty() {
            return Err(JudgeError::Invalid(format!("{var} is empty")));
        }
        Self::jev_with_key(key)
    }

    pub fn jev_with_key(key: impl Into<String>) -> Result<HttpJudgeConfig, JudgeError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(JudgeError::Invalid("API key is empty".into()));
        }
        Ok(HttpJudgeConfig {
            base_url: "https://api.typesafe.ai".into(),
            path: "/v1/systemone".into(),
            protocol: HttpJudgeProtocol::Jev,
            model: Some("jev-latest".into()),
            api_key: Some(key.trim().to_owned()),
            max_questions_per_request: 64,
            max_concurrent: 8,
        })
    }

    /// OpenRouter's documented TypeSafe-compatible System One endpoint. The
    /// credential belongs to OpenRouter, never to the direct TypeSafe host.
    pub fn openrouter_with_key(key: impl Into<String>) -> Result<HttpJudgeConfig, JudgeError> {
        let mut cfg = Self::jev_with_key(key)?;
        cfg.base_url = "https://openrouter.ai/api".into();
        Ok(cfg)
    }

    /// Mercury Decide shares the typed System One question/answer grammar.
    /// The caller owns credential loading and any spend reservation.
    pub fn mercury_with_key(key: impl Into<String>) -> Result<HttpJudgeConfig, JudgeError> {
        let mut cfg = Self::jev_with_key(key)?;
        cfg.base_url = "https://api.inceptionlabs.ai".into();
        cfg.path = "/v1/decisions".into();
        cfg.protocol = HttpJudgeProtocol::Mercury;
        cfg.model = Some("mercury-decide".into());
        cfg.max_questions_per_request = 15;
        cfg.max_concurrent = 1;
        Ok(cfg)
    }
}

pub struct HttpJudge {
    cfg: HttpJudgeConfig,
    url: String,
    client: reqwest::Client,
    permits: Semaphore,
    loopback: bool,
}

impl HttpJudge {
    pub fn new(cfg: HttpJudgeConfig) -> Result<HttpJudge, JudgeError> {
        if cfg.max_questions_per_request == 0 {
            return Err(JudgeError::Invalid("max_questions_per_request must be at least 1".into()));
        }
        let url = format!("{}{}", cfg.base_url.trim_end_matches('/'), cfg.path);
        let parsed = reqwest::Url::parse(&url).map_err(|e| JudgeError::Invalid(e.to_string()))?;
        if !matches!(parsed.scheme(), "http" | "https")
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
        {
            return Err(JudgeError::Invalid("judge URL must be HTTP(S) without credentials, query or fragment".into()));
        }
        if cfg.api_key.as_ref().is_some_and(|k| k.trim().is_empty()) {
            return Err(JudgeError::Invalid("api_key is empty".into()));
        }
        let loopback = parsed.host_str().is_some_and(|host| {
            host == "localhost" || host.trim_matches(['[', ']']).parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
        });
        // A loopback service must not redirect a memory payload to a remote host.
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if loopback {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|e| JudgeError::Transport(e.to_string()))?;
        let permits = Semaphore::new(cfg.max_concurrent.max(1));
        Ok(HttpJudge { cfg, url, client, permits, loopback })
    }

    async fn send(
        &self,
        state: &Value,
        questions: &[(&String, &Question)],
        started: Instant,
        deadline: Duration,
    ) -> Result<(String, IndexMap<String, Answer>, Usage), JudgeError> {
        let mut qs = Map::new();
        for (name, q) in questions {
            let mut value = serde_json::to_value(q).map_err(|e| JudgeError::Invalid(e.to_string()))?;
            if self.cfg.protocol == HttpJudgeProtocol::CoffeeCat {
                coffee_cat_question(&mut value);
            }
            qs.insert((*name).clone(), value);
        }
        let mut body = json!({ "state": state, "questions": qs });
        if let Some(model) = &self.cfg.model {
            body["model"] = Value::String(model.clone());
        }
        let _permit = self.permits.acquire().await.map_err(|e| JudgeError::Transport(e.to_string()))?;
        if self.cfg.protocol == HttpJudgeProtocol::CoffeeCat {
            let remaining = deadline.saturating_sub(started.elapsed());
            // Production rejects deadlines below 100 ms or above 120 s. Never
            // extend a caller's deadline to satisfy the server's minimum.
            if remaining < Duration::from_millis(100) {
                return Err(JudgeError::Timeout(deadline));
            }
            body["deadline_ms"] = json!(remaining.as_millis().min(120_000) as u64);
        }
        let mut request = self.client.post(&self.url).json(&body);
        if let Some(key) = &self.cfg.api_key {
            request = request.bearer_auth(key);
        }
        let response = request.send().await.map_err(|e| JudgeError::Transport(e.to_string()))?;
        let status = response.status();
        let text = response.text().await.map_err(|e| JudgeError::Transport(e.to_string()))?;
        if !status.is_success() {
            let mut body = text;
            let mut end = body.len().min(2048);
            while !body.is_char_boundary(end) {
                end -= 1;
            }
            body.truncate(end);
            return Err(JudgeError::Http { status: status.as_u16(), body });
        }
        let value: Value = serde_json::from_str(&text).map_err(|e| JudgeError::Decode(e.to_string()))?;
        if self.cfg.protocol == HttpJudgeProtocol::Mercury
            && value.get("model").and_then(Value::as_str) != self.cfg.model.as_deref()
        {
            return Err(JudgeError::Decode("Mercury returned a different model".into()));
        }
        decode(&value, questions)
    }
}

#[async_trait::async_trait]
impl Judge for HttpJudge {
    fn id(&self) -> &str {
        &self.url
    }

    fn requested_model(&self) -> Option<&str> {
        self.cfg.model.as_deref()
    }

    fn is_loopback(&self) -> bool {
        self.loopback
    }

    async fn ask(
        &self,
        point: Point,
        state: &Value,
        questions: &IndexMap<String, Question>,
        deadline: Duration,
    ) -> Result<Judgment, JudgeError> {
        if questions.is_empty() {
            return Err(JudgeError::Invalid("at least one question is required".into()));
        }
        let started = Instant::now();
        let all: Vec<(&String, &Question)> = questions.iter().collect();
        let work = async {
            let mut model = String::new();
            let mut answers = IndexMap::with_capacity(all.len());
            let mut usage = Usage::default();
            for batch in all.chunks(self.cfg.max_questions_per_request) {
                let (m, a, u) = self.send(state, batch, started, deadline).await?;
                if model.is_empty() {
                    model = m;
                }
                answers.extend(a);
                usage.input_tokens = add(usage.input_tokens, u.input_tokens);
                usage.output_tokens = add(usage.output_tokens, u.output_tokens);
                usage.cost = match (usage.cost, u.cost) {
                    (None, None) => None,
                    (a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
                };
            }
            Ok::<_, JudgeError>((model, answers, usage))
        };
        let (model, answers, usage) =
            tokio::time::timeout(deadline, work).await.map_err(|_| JudgeError::Timeout(deadline))??;
        let latency = started.elapsed();
        tracing::debug!(%point, %model, questions = questions.len(), latency_ms = latency.as_millis() as u64, "judgment");
        Ok(Judgment { model, answers, usage, latency })
    }
}

pub(super) fn coffee_cat_question(question: &mut Value) {
    let string_value = |v: &Value| match v {
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    };
    if let Some(instructions) = question.get_mut("instructions") {
        *instructions = Value::String(string_value(instructions));
    }
    match question.get_mut("criteria") {
        Some(Value::Object(criteria)) => {
            for (name, description) in criteria {
                *description =
                    Value::String(if description.is_null() { name.clone() } else { string_value(description) });
            }
        }
        Some(Value::Array(levels)) => {
            for level in levels {
                *level = Value::String(string_value(level));
            }
        }
        _ => {}
    }
}

fn add(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

/// Decodes a response body against the questions that were asked.
pub(super) fn decode(
    body: &Value,
    questions: &[(&String, &Question)],
) -> Result<(String, IndexMap<String, Answer>, Usage), JudgeError> {
    let answers = body.get("answers").and_then(Value::as_object).ok_or_else(|| decode_err("no `answers` object"))?;
    let mut out = IndexMap::with_capacity(questions.len());
    for (name, question) in questions {
        let raw = answers.get(name.as_str()).ok_or_else(|| decode_err(format!("no answer for `{name}`")))?;
        out.insert((*name).clone(), decode_answer(name, question, raw)?);
    }
    let usage = body.get("usage");
    let model = body
        .get("model")
        .or_else(|| usage.and_then(|u| u.get("model")))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let usage = Usage {
        input_tokens: usage.and_then(|u| u.get("input_tokens")).and_then(Value::as_u64),
        output_tokens: usage.and_then(|u| u.get("output_tokens")).and_then(Value::as_u64),
        cost: usage.and_then(|u| u.get("cost")).and_then(Value::as_f64).filter(|cost| cost.is_finite() && *cost >= 0.0),
    };
    Ok((model, out, usage))
}

fn decode_answer(name: &str, question: &Question, raw: &Value) -> Result<Answer, JudgeError> {
    if let Some(kind) = raw.get("type").and_then(Value::as_str) {
        if kind != question.kind() {
            return Err(decode_err(format!("`{name}` was asked as {} but answered as {kind}", question.kind())));
        }
    }
    let number = |key: &str| raw.get(key).and_then(Value::as_f64);
    let probability = |key: &str| -> Result<f64, JudgeError> {
        let p = number(key).ok_or_else(|| decode_err(format!("`{name}` has no numeric `{key}`")))?;
        if (0.0..=1.0).contains(&p) { Ok(p) } else { Err(decode_err(format!("`{name}`.{key} = {p} is outside 0..1"))) }
    };
    let distribution = || -> IndexMap<String, f64> {
        ["probabilities", "distribution"]
            .iter()
            .find_map(|k| raw.get(*k).and_then(Value::as_object))
            .map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|p| (k.clone(), p))).collect())
            .unwrap_or_default()
    };
    match question {
        Question::Noul { .. } => Ok(Answer::Noul { noul: probability("noul")? }),
        Question::Choice { criteria, .. } => {
            let choice = raw
                .get("choice")
                .and_then(Value::as_str)
                .ok_or_else(|| decode_err(format!("`{name}` has no `choice`")))?
                .to_owned();
            if !criteria.contains_key(&choice) {
                return Err(decode_err(format!("`{name}` chose `{choice}`, which was not an option")));
            }
            Ok(Answer::Choice { choice, confidence: probability("confidence")?, probabilities: distribution() })
        }
        Question::Score { criteria, .. } => {
            let score = number("score").ok_or_else(|| decode_err(format!("`{name}` has no numeric `score`")))?;
            let top = criteria.len().saturating_sub(1) as f64;
            if !(0.0..=top).contains(&score) {
                return Err(decode_err(format!("`{name}` score {score} is outside 0..{top}")));
            }
            Ok(Answer::Score { score, confidence: number("confidence"), probabilities: distribution() })
        }
    }
}

fn decode_err(msg: impl Into<String>) -> JudgeError {
    JudgeError::Decode(msg.into())
}
