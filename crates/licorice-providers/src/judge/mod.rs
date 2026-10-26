//! System One judgments (SPEC §10.2).
//!
//! A judge answers named, typed questions about one `state` and returns probabilities
//! rather than text. The types mirror TypeSafe's v1 wire format (`POST /v1/systemone`,
//! from typesafe-sdk 0.7.5), which Coffee Cat also serves, over HTTP ([`HttpJudge`]) or
//! from its native CLI ([`ProcessJudge`]).

mod http;
mod process;

use std::fmt;
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use http::{HttpJudge, HttpJudgeConfig, HttpJudgeProtocol};
pub use process::{ProcessJudge, ProcessJudgeConfig};

/// Where in Licorice a judgment is asked (SPEC §10.2 decision table). Logged with every
/// judgment; the wire request doesn't carry it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Point {
    /// Write gate: is this window worth extracting?
    J1,
    /// Entity resolution in the ambiguous band.
    J2,
    /// Supersession: does the new fact replace the old one?
    J3,
    /// Is a predicate single-valued?
    J4,
    /// Is a fact supported by its evidence turns?
    J5,
    /// Salience.
    J6,
    /// Which candidate interval a time expression means.
    J7,
    /// Query analysis: temporal intent and subject.
    J8,
    /// Rerank candidate facts.
    J9,
    /// Is the assembled context sufficient to answer?
    J10,
    /// Dev-loop grader.
    J11,
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// What counts as yes and no for a [`Question::Noul`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

/// A typed question. Question names (the map keys in a request) are for code and are
/// not shown to the model, so `instructions` and `criteria` carry the full meaning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Probability that a statement is true.
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One option out of `criteria` (name → description, or `None` for the name alone).
    Choice { instructions: Value, criteria: IndexMap<String, Option<Value>> },
    /// Expected position on ordered levels; level `i` is `criteria[i]`.
    Score { instructions: Value, criteria: Vec<Value> },
}

impl Question {
    pub fn noul(instructions: impl Into<String>) -> Question {
        Question::Noul { instructions: Value::String(instructions.into()), criteria: None }
    }

    pub fn noul_with(instructions: impl Into<String>, yes: impl Into<String>, no: impl Into<String>) -> Question {
        Question::Noul {
            instructions: Value::String(instructions.into()),
            criteria: Some(NoulCriteria { yes: Some(Value::String(yes.into())), no: Some(Value::String(no.into())) }),
        }
    }

    pub fn choice<N: Into<String>>(
        instructions: impl Into<String>,
        options: impl IntoIterator<Item = (N, Option<String>)>,
    ) -> Question {
        Question::Choice {
            instructions: Value::String(instructions.into()),
            criteria: options.into_iter().map(|(n, d)| (n.into(), d.map(Value::String))).collect(),
        }
    }

    pub fn score<L: Into<String>>(instructions: impl Into<String>, levels: impl IntoIterator<Item = L>) -> Question {
        Question::Score {
            instructions: Value::String(instructions.into()),
            criteria: levels.into_iter().map(|l| Value::String(l.into())).collect(),
        }
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Question::Noul { .. } => "noul",
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
        }
    }
}

/// A decoded answer, typed like its question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// Probability of yes, 0..1.
    Noul { noul: f64 },
    /// The likeliest option, its probability-based confidence, and the distribution
    /// (empty when the engine didn't send one).
    Choice { choice: String, confidence: f64, probabilities: IndexMap<String, f64> },
    /// Expected level (0-based); level probabilities keyed "0", "1", … when sent.
    Score { score: f64, confidence: Option<f64>, probabilities: IndexMap<String, f64> },
}

impl Answer {
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Answer::Noul { noul } => Some(*noul),
            _ => None,
        }
    }

    pub fn as_choice(&self) -> Option<(&str, f64)> {
        match self {
            Answer::Choice { choice, confidence, .. } => Some((choice, *confidence)),
            _ => None,
        }
    }

    pub fn as_score(&self) -> Option<f64> {
        match self {
            Answer::Score { score, .. } => Some(*score),
            _ => None,
        }
    }
}

/// Token counts, when the engine reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// Provider-reported cost, when present (OpenRouter reports USD).
    pub cost: Option<f64>,
}

/// One judgment: every answer, in question order.
#[derive(Clone, Debug, PartialEq)]
pub struct Judgment {
    /// The model as reported by the server (may differ from the alias requested).
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: Usage,
    pub latency: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum JudgeError {
    #[error("no answer within {0:?}")]
    Timeout(Duration),
    #[error("judge returned HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("transport: {0}")]
    Transport(String),
    #[error("could not decode the judge's answer: {0}")]
    Decode(String),
    #[error("invalid request: {0}")]
    Invalid(String),
    /// A local engine could not be started or could not load its model.
    #[error("judge could not start: {0}")]
    Startup(String),
    /// The engine accepted the request but failed while answering it.
    #[error("judge failed: {0}")]
    Engine(String),
}

/// A System One judge. Callers own the fallback: on any error they use their rule.
#[async_trait::async_trait]
pub trait Judge: Send + Sync {
    /// Identifies the engine in logs (its configured name or URL).
    fn id(&self) -> &str;

    /// Configured model/alias, if any; separate from the model reported in a reply.
    fn requested_model(&self) -> Option<&str> {
        None
    }

    /// True only when requests stay on this machine. Unknown implementations are
    /// remote by default; callers must obtain space-level permission before use.
    fn is_loopback(&self) -> bool {
        false
    }

    /// Starts a local engine (process, model load) so the first `ask` deadline isn't
    /// spent on it. Remote engines have nothing to do.
    async fn warm_up(&self) -> Result<(), JudgeError> {
        Ok(())
    }

    /// Asks every question about `state` within `deadline`.
    async fn ask(
        &self,
        point: Point,
        state: &Value,
        questions: &IndexMap<String, Question>,
        deadline: Duration,
    ) -> Result<Judgment, JudgeError>;
}
