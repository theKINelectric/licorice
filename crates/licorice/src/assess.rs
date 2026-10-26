//! Optional evidence assessment (J10). Scores are observational: they never remove
//! evidence or suppress an answer. Normal retrieval/answering remains the fallback.

use std::time::{Duration, Instant};

use indexmap::IndexMap;
use licorice_providers::judge::{Answer, Judge, JudgeError, Point, Question};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::store::now_ms;
use crate::{Error, Result, Store};

const REMOTE_PERMISSION: &str = "judge.send_to_hosted";

/// Explicit permission for this space. Non-loopback endpoints, including a house
/// service on another machine, require opt-in. Revocation applies to later calls;
/// a request already in flight cannot be recalled.
pub async fn allow_remote_judgments(store: &Store, allowed: bool) -> Result<()> {
    store
        .write(move |w| {
            w.conn().execute(
                "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![REMOTE_PERMISSION, if allowed { "true" } else { "false" }],
            )?;
            Ok(())
        })
        .await
}

pub async fn remote_judgments_allowed(store: &Store) -> Result<bool> {
    store
        .read(|conn| {
            let value: Option<String> = conn
                .query_row("SELECT value FROM meta WHERE key = ?1", [REMOTE_PERMISSION], |r| r.get(0))
                .optional()?;
            Ok(value.as_deref() == Some("true"))
        })
        .await
}

#[derive(Clone, Debug)]
pub struct AssessConfig {
    /// Includes provider queueing. The caller enforces this even if a custom
    /// Judge implementation neglects its own deadline.
    pub deadline: Duration,
    /// Oversized inputs are reported as unavailable, never silently truncated.
    pub max_input_bytes: usize,
}

impl Default for AssessConfig {
    fn default() -> Self {
        AssessConfig { deadline: Duration::from_secs(1), max_input_bytes: 32 * 1024 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssessmentStatus {
    Judged,
    Unavailable,
    NotAllowed,
}

/// Raw per-task scores, not a calibrated probability that an answer is correct.
/// None means no valid assessment; it never means a score of zero.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceAssessment {
    pub status: AssessmentStatus,
    pub sufficient: Option<f64>,
    pub contradicted: Option<f64>,
    pub engine: String,
    pub model: Option<String>,
    pub latency_ms: f64,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cost: Option<f64>,
    pub error: Option<String>,
    pub judgment_id: Option<i64>,
}

pub struct EvidenceAssessor<'a> {
    store: &'a Store,
    judge: &'a dyn Judge,
    cfg: AssessConfig,
}

impl<'a> EvidenceAssessor<'a> {
    pub fn new(store: &'a Store, judge: &'a dyn Judge, cfg: AssessConfig) -> Result<Self> {
        if cfg.deadline.is_zero() || cfg.max_input_bytes == 0 {
            return Err(Error::Task("assessment deadline and input limit must be positive".into()));
        }
        Ok(Self { store, judge, cfg })
    }

    /// Assess the exact context sent to the answering model. Do not pass gold
    /// answers or hidden benchmark evidence here. Every attempted judgment,
    /// including failure, is recorded in the space; permission refusals are not.
    pub async fn assess(&self, question: &str, context: &str) -> Result<EvidenceAssessment> {
        let mut result = EvidenceAssessment {
            status: AssessmentStatus::Unavailable,
            sufficient: None,
            contradicted: None,
            engine: self.judge.id().to_owned(),
            model: None,
            latency_ms: 0.0,
            input_tokens: None,
            output_tokens: None,
            cost: None,
            error: None,
            judgment_id: None,
        };
        if !self.judge.is_loopback() && !remote_judgments_allowed(self.store).await? {
            result.status = AssessmentStatus::NotAllowed;
            result.error = Some("this space has not enabled remote judgments".into());
            return Ok(result);
        }
        if question.len().saturating_add(context.len()) > self.cfg.max_input_bytes {
            result.error = Some("assessment input exceeds max_input_bytes; context was not truncated".into());
            return Ok(result);
        }

        let state = json!({"question": question, "memory": context});
        let questions = evidence_questions();
        let request = json!({
            "version": 1, "engine": self.judge.id(), "requested_model": self.judge.requested_model(),
            "deadline_ms": self.cfg.deadline.as_millis() as u64, "state": state, "questions": questions,
        })
        .to_string();
        let hash = blake3::hash(request.as_bytes()).as_bytes().to_vec();
        let started = Instant::now();
        let response =
            tokio::time::timeout(self.cfg.deadline, self.judge.ask(Point::J10, &state, &questions, self.cfg.deadline))
                .await
                .unwrap_or_else(|_| Err(JudgeError::Timeout(self.cfg.deadline)));
        result.latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut answers = json!({});
        match response {
            Ok(judgment) => {
                result.model = Some(judgment.model);
                result.input_tokens = judgment.usage.input_tokens;
                result.output_tokens = judgment.usage.output_tokens;
                result.cost = judgment.usage.cost;
                answers = serde_json::to_value(&judgment.answers).map_err(|e| Error::Task(e.to_string()))?;
                match (probability(&judgment.answers, "sufficient"), probability(&judgment.answers, "contradicted")) {
                    (Some(s), Some(c)) => {
                        result.status = AssessmentStatus::Judged;
                        result.sufficient = Some(s);
                        result.contradicted = Some(c);
                    }
                    _ => result.error = Some("judge did not return both valid noul assessments".into()),
                }
            }
            Err(error) => result.error = Some(error.to_string()),
        }
        let model = result.model.clone().unwrap_or_else(|| result.engine.clone());
        let receipt = json!({"assessment": result, "answers": answers}).to_string();
        let latency = result.latency_ms.ceil() as i64;
        let id = self
            .store
            .write(move |w| {
                w.conn().execute(
                    "INSERT INTO judgments(point, model, request_hash, request, answers, latency_ms, created_at) \
                 VALUES ('J10', ?1, ?2, ?3, ?4, ?5, ?6)",
                    params![model, hash, request, receipt, latency, now_ms()],
                )?;
                Ok(w.conn().last_insert_rowid())
            })
            .await?;
        result.judgment_id = Some(id);
        Ok(result)
    }
}

fn probability(answers: &IndexMap<String, Answer>, name: &str) -> Option<f64> {
    answers.get(name)?.as_noul().filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
}

fn evidence_questions() -> IndexMap<String, Question> {
    IndexMap::from([
        (
            "sufficient".into(),
            Question::noul_with(
                "Does the supplied memory contain all the information needed to answer the supplied question?",
                "The answer can be supported using only this memory, including every requested item or reasoning step.",
                "Some required information is absent. Related information alone is not sufficient.",
            ),
        ),
        (
            "contradicted".into(),
            Question::noul_with(
                "Does the supplied memory explicitly contradict a factual assumption in the supplied question?",
                "The memory states something incompatible with a premise of the question.",
                "No explicit contradiction is present. Absence of evidence alone is not a contradiction.",
            ),
        ),
    ])
}
