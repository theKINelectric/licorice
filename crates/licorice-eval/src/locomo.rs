//! LoCoMo (Maharana et al., 2024, `locomo10.json`): conversations, questions, the pinned
//! category mapping and the dataset's date format.

use std::path::Path;

use licorice::ingest::{Role, SessionInput, TurnInput};
use licorice::temporal::Day;
use serde_json::Value;

/// LoCoMo's numeric question categories. Follow-up work often mislabels them (SPEC §12),
/// so the mapping is pinned here and checked against the questions in the tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    MultiHop,
    Temporal,
    OpenDomain,
    SingleHop,
    Adversarial,
}

impl Category {
    pub fn from_id(id: i64) -> Option<Category> {
        Some(match id {
            1 => Category::MultiHop,
            2 => Category::Temporal,
            3 => Category::OpenDomain,
            4 => Category::SingleHop,
            5 => Category::Adversarial,
            _ => return None,
        })
    }

    pub fn id(self) -> i64 {
        match self {
            Category::MultiHop => 1,
            Category::Temporal => 2,
            Category::OpenDomain => 3,
            Category::SingleHop => 4,
            Category::Adversarial => 5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Category::MultiHop => "multi-hop",
            Category::Temporal => "temporal",
            Category::OpenDomain => "open-domain",
            Category::SingleHop => "single-hop",
            Category::Adversarial => "adversarial",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Qa {
    /// Position in the conversation's `qa` list; request ids are built from it.
    pub index: usize,
    pub question: String,
    /// The gold answer; absent for all but two adversarial questions in the dataset.
    pub answer: Option<String>,
    /// For adversarial questions: the misleading answer the question invites.
    pub adversarial_answer: Option<String>,
    /// Gold evidence turns (`dia_id`s such as "D1:3").
    pub evidence: Vec<String>,
    pub category: Category,
}

#[derive(Clone, Debug)]
pub struct Conversation {
    pub id: String,
    pub speakers: Vec<String>,
    pub sessions: Vec<SessionInput>,
    pub qa: Vec<Qa>,
    /// Explicit benchmark question clock, when it differs from the last source
    /// session (for example a converted LongMemEval instance). Unix milliseconds.
    pub question_at_ms: Option<i64>,
}

impl Conversation {
    /// LoCoMo defaults to the last session; external views can preserve their
    /// actual question time without changing source-message timestamps.
    pub fn question_time(&self) -> i64 {
        self.question_at_ms.unwrap_or_else(|| self.sessions.iter().map(|s| s.started_at).max().unwrap_or(0))
    }

    pub fn has_turn(&self, dia_id: &str) -> bool {
        self.sessions.iter().any(|s| s.turns.iter().any(|t| t.ext_id == dia_id))
    }
}

/// Loads conversation `id` (a `sample_id` such as "conv-26").
pub fn load(path: &Path, id: &str) -> Result<Conversation, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let all: Vec<Value> = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let sample =
        all.iter().find(|s| s["sample_id"] == id).ok_or_else(|| format!("{}: no conversation {id}", path.display()))?;
    parse(sample)
}

pub fn parse(sample: &Value) -> Result<Conversation, String> {
    let id = sample["sample_id"].as_str().unwrap_or_default().to_owned();
    let question_at_ms = match sample.get("question_at_ms") {
        None | Some(Value::Null) => None,
        Some(value) => {
            Some(value.as_i64().filter(|ms| (-62_135_596_800_000..=253_402_300_799_999).contains(ms)).ok_or_else(
                || format!("{id}: question_at_ms must be integer Unix milliseconds in civil years 0001..9999"),
            )?)
        }
    };
    let conv = &sample["conversation"];
    let speakers: Vec<String> =
        ["speaker_a", "speaker_b"].iter().filter_map(|k| conv[k].as_str().map(str::to_owned)).collect();
    let mut sessions = Vec::new();
    for n in 1.. {
        let Some(turns) = conv[format!("session_{n}")].as_array() else { break };
        let date = conv[format!("session_{n}_date_time")].as_str().unwrap_or_default();
        let started_at = parse_date(date).ok_or_else(|| format!("{id} session {n}: bad date {date:?}"))?;
        let turns = turns
            .iter()
            .map(|t| {
                let mut text = t["text"].as_str().unwrap_or_default().to_owned();
                if let Some(caption) = t["blip_caption"].as_str() {
                    text.push_str(&format!(" [shares a photo: {caption}]"));
                }
                let role = match t.get("role") {
                    None => Role::Participant,
                    Some(Value::String(role)) if role == "participant" => Role::Participant,
                    Some(Value::String(role)) if role == "user" => Role::User,
                    Some(Value::String(role)) if role == "assistant" => Role::Assistant,
                    Some(_) => return Err(format!("{id} session {n}: invalid declared source role")),
                };
                Ok(TurnInput {
                    ext_id: t["dia_id"].as_str().unwrap_or_default().to_owned(),
                    speaker: t["speaker"].as_str().unwrap_or_default().to_owned(),
                    role,
                    text,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        sessions.push(SessionInput { ext_id: format!("D{n}"), started_at, turns });
    }
    let qa = sample["qa"]
        .as_array()
        .ok_or_else(|| format!("{id}: no qa"))?
        .iter()
        .enumerate()
        .map(|(index, q)| {
            let category = q["category"]
                .as_i64()
                .and_then(Category::from_id)
                .ok_or_else(|| format!("{id} qa {index}: bad category {}", q["category"]))?;
            Ok(Qa {
                index,
                question: q["question"].as_str().unwrap_or_default().to_owned(),
                answer: text_of(&q["answer"]),
                adversarial_answer: text_of(&q["adversarial_answer"]),
                evidence: evidence_ids(&q["evidence"]),
                category,
            })
        })
        .collect::<Result<_, String>>()?;
    Ok(Conversation { id, speakers, sessions, qa, question_at_ms })
}

/// Answers are strings or, for some questions, bare numbers.
fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Evidence entries are usually single `dia_id`s, but some hold several ("D8:6; D9:17").
fn evidence_ids(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for entry in v.as_array().into_iter().flatten().filter_map(Value::as_str) {
        for part in entry.split([';', ',', ' ']).map(str::trim) {
            let valid = part.strip_prefix('D').and_then(|r| r.split_once(':')).is_some_and(|(s, t)| {
                !s.is_empty()
                    && !t.is_empty()
                    && s.bytes().all(|b| b.is_ascii_digit())
                    && t.bytes().all(|b| b.is_ascii_digit())
            });
            if valid && !out.iter().any(|e| e == part) {
                out.push(part.to_owned());
            }
        }
    }
    out
}

const MONTHS: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Parses LoCoMo's session time, "1:56 pm on 8 May, 2023", to Unix milliseconds (UTC).
pub fn parse_date(s: &str) -> Option<i64> {
    let (time, date) = s.trim().split_once(" on ")?;
    let (clock, meridiem) = time.trim().split_once(' ')?;
    let (h, m) = clock.split_once(':')?;
    let (mut h, m): (i64, i64) = (h.parse().ok()?, m.parse().ok()?);
    match meridiem.to_ascii_lowercase().as_str() {
        "am" if h == 12 => h = 0,
        "pm" if h < 12 => h += 12,
        "am" | "pm" => {}
        _ => return None,
    }
    let parts: Vec<String> = date.replace(',', " ").split_whitespace().map(str::to_lowercase).collect();
    let [d, month, y] = parts.as_slice() else { return None };
    let month = MONTHS.iter().position(|n| n == month)? as u32 + 1;
    let day = Day::from_ymd(y.parse().ok()?, month, d.parse().ok()?)?;
    Some(day.start_ms() + (h * 60 + m) * 60_000)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn declared_roles_survive_without_guessing_from_speaker_names() {
        let sample = json!({"sample_id":"role-control","conversation":{
            "session_1_date_time":"9:00 am on 2 June, 2023","session_1":[
                {"speaker":"Person A","role":"user","dia_id":"D1:1","text":"I bought a mug."},
                {"speaker":"Helper","role":"assistant","dia_id":"D1:2","text":"Some advice."},
                {"speaker":"assistant","dia_id":"D1:3","text":"A human with this name."}
            ]},"qa":[]});
        let c = parse(&sample).unwrap();
        assert_eq!(
            c.sessions[0].turns.iter().map(|t| t.role).collect::<Vec<_>>(),
            [Role::User, Role::Assistant, Role::Participant]
        );
        assert_eq!(c.sessions[0].turns[0].speaker, "Person A");
        for invalid in [json!(null), json!("system"), json!("tool"), json!(42), json!("USER")] {
            let mut bad = sample.clone();
            bad["conversation"]["session_1"][0]["role"] = invalid;
            assert!(parse(&bad).is_err(), "declared roles must never silently become participant");
        }
    }

    #[test]
    fn dates() {
        let t = parse_date("1:56 pm on 8 May, 2023").unwrap();
        assert_eq!(Day::from_ms(t), Day::from_ymd(2023, 5, 8).unwrap());
        assert_eq!((t - Day::from_ymd(2023, 5, 8).unwrap().start_ms()) / 60_000, 13 * 60 + 56);
        let midnight = parse_date("12:05 am on 1 January, 2024").unwrap();
        assert_eq!(midnight - Day::from_ymd(2024, 1, 1).unwrap().start_ms(), 5 * 60_000);
        assert!(parse_date("8 May 2023").is_none());
    }

    #[test]
    fn explicit_question_clock_is_distinct_from_source_session_dates() {
        let sample = json!({"sample_id":"external-clock-control","question_at_ms":1688256000000_i64,
            "conversation":{"session_1_date_time":"9:00 am on 2 June, 2023","session_1":[{"speaker":"User","dia_id":"D1:1","text":"I moved yesterday."}]},
            "qa":[{"question":"Where was I last month?","answer":"Oslo","evidence":["D1:1"],"category":2}]});
        let parsed = parse(&sample).unwrap();
        assert_eq!(
            parsed.question_time(),
            1688256000000_i64,
            "relative question time must use the explicit benchmark question timestamp"
        );
        assert_eq!(
            parsed.sessions[0].started_at,
            parse_date("9:00 am on 2 June, 2023").unwrap(),
            "source message clocks remain immutable"
        );
        let intent = licorice::retrieve::Intent::of("Where was I last month?", Day::from_ms(parsed.question_time()));
        let licorice::retrieve::Intent::Window(window) = intent else {
            panic!("relative month must resolve to a window")
        };
        assert_eq!(
            window.from,
            Day::from_ymd(2023, 6, 1).unwrap(),
            "question date is in July; source date in June must not shift the window to May"
        );
    }

    #[test]
    fn malformed_explicit_question_clocks_fail_instead_of_using_a_source_date() {
        for value in [json!("2023-06-23"), json!(1.5), json!(i64::MAX)] {
            let sample = json!({"sample_id":"bad-clock-control","question_at_ms":value,"conversation":{},"qa":[]});
            assert!(parse(&sample).is_err(), "bad explicit clock must not silently fall back: {value}");
        }
    }

    #[test]
    fn parses_the_dataset_shape() {
        let sample = json!({
            "sample_id": "conv-x",
            "conversation": {
                "speaker_a": "Ann", "speaker_b": "Bo",
                "session_1_date_time": "9:00 am on 2 June, 2023",
                "session_1": [
                    {"speaker": "Ann", "dia_id": "D1:1", "text": "Look!", "blip_caption": "a photo of a cat"},
                    {"speaker": "Bo", "dia_id": "D1:2", "text": "Cute."}
                ],
                "session_2_date_time": "4:10 pm on 9 June, 2023",
                "session_2": [{"speaker": "Bo", "dia_id": "D2:1", "text": "Hi"}]
            },
            "qa": [
                {"question": "When did Ann share a cat photo?", "answer": "2 June 2023", "evidence": ["D1:1"], "category": 2},
                {"question": "How many cats?", "answer": 1, "evidence": ["D1:1; D1:2", "D"], "category": 1},
                {"question": "What did Bo's cat do?", "adversarial_answer": "slept", "evidence": [], "category": 5}
            ]
        });
        let c = parse(&sample).unwrap();
        assert_eq!(c.speakers, ["Ann", "Bo"]);
        assert_eq!(c.sessions.len(), 2);
        assert_eq!(c.sessions[1].ext_id, "D2");
        assert_eq!(c.sessions[0].turns[0].text, "Look! [shares a photo: a photo of a cat]");
        assert_eq!(Day::from_ms(c.question_time()), Day::from_ymd(2023, 6, 9).unwrap());
        assert_eq!(c.qa[1].answer.as_deref(), Some("1"));
        assert_eq!(c.qa[1].evidence, ["D1:1", "D1:2"]);
        assert_eq!(c.qa[2].category, Category::Adversarial);
        assert_eq!(c.qa[2].adversarial_answer.as_deref(), Some("slept"));
        assert!(c.has_turn("D2:1") && !c.has_turn("D2:2"));
    }

    /// Checks the pinned mapping against the real data when `LOCOMO_PATH` names
    /// `locomo10.json`: temporal questions ask about time, adversarial ones carry the
    /// misleading answer, and single-hop is the largest category.
    #[test]
    fn category_mapping_matches_the_dataset() {
        let Ok(path) = std::env::var("LOCOMO_PATH") else { return };
        let c = load(Path::new(&path), "conv-26").unwrap();
        let temporal: Vec<&Qa> = c.qa.iter().filter(|q| q.category == Category::Temporal).collect();
        let timey = temporal
            .iter()
            .filter(|q| {
                let q = q.question.to_lowercase();
                ["when", "how long", "what year", "which year", "what month", "date"].iter().any(|w| q.contains(w))
            })
            .count();
        assert!(timey * 10 >= temporal.len() * 8, "{timey} of {}", temporal.len());
        let adversarial: Vec<&Qa> = c.qa.iter().filter(|q| q.category == Category::Adversarial).collect();
        assert!(adversarial.iter().all(|q| q.adversarial_answer.is_some()));
        // Two of them also carry a gold "No".
        assert_eq!(adversarial.iter().filter(|q| q.answer.is_some()).count(), 2);
        let single = c.qa.iter().filter(|q| q.category == Category::SingleHop).count();
        assert!(
            (1..=5).filter_map(Category::from_id).all(|k| c.qa.iter().filter(|q| q.category == k).count() <= single)
        );
    }
}
