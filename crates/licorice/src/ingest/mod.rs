//! Ingestion (SPEC §8): turns and chunks at append time; facts, entities and session
//! summaries from model output.
//!
//! Model calls are not made here. [`Ingestor`] renders each call as a
//! [`ModelRequest`] and applies the matching output, so the same code serves a live
//! model and replayed outputs.

mod application;
mod extract;
mod prompts;
mod summary;
mod updates;

use std::collections::HashSet;

use licorice_providers::chat::ModelRequest;
use licorice_providers::embed::{EmbedKind, Embedder};
use rusqlite::{OptionalExtension, params};
use serde::Serialize;

pub use extract::ExtractionReport;

use crate::store::{LexicalKind, Store, VectorKind};
use crate::{Error, Result};

/// Who spoke a turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    /// A named participant in a conversation between people (LoCoMo).
    Participant,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Participant => "participant",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TurnInput {
    /// Caller's id for the turn (LoCoMo `dia_id`, e.g. "D1:3"); used in prompts and evidence.
    pub ext_id: String,
    pub speaker: String,
    pub role: Role,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct SessionInput {
    pub ext_id: String,
    /// Unix milliseconds; every turn in the session gets this time.
    pub started_at: i64,
    pub turns: Vec<TurnInput>,
}

#[derive(Clone, Debug, Serialize)]
pub struct IngestConfig {
    /// Turns per extraction window (SPEC §8.2; parity mode uses 2).
    pub window_turns: usize,
    /// Preceding turns shown as context only.
    pub context_turns: usize,
    /// Turns per raw-dialogue chunk.
    pub chunk_turns: usize,
    /// Similarity at or above which an extracted entity is linked to an existing one.
    pub merge_threshold: f64,
    /// Opt-in source-backed state replacements and message-specific date anchors.
    #[serde(skip_serializing_if = "is_false")]
    pub temporal_updates: bool,
}

fn is_false(value: &bool) -> bool {
    !value
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig {
            window_turns: 8,
            context_turns: 2,
            chunk_turns: 2,
            merge_threshold: 0.92,
            temporal_updates: false,
        }
    }
}

/// Ingests sessions into one memory space.
pub struct Ingestor<'a> {
    pub(crate) store: &'a Store,
    pub(crate) embedder: &'a dyn Embedder,
    pub(crate) cfg: IngestConfig,
}

/// A stored turn, as prompts see it.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct StoredTurn {
    pub(crate) id: i64,
    pub(crate) seq: i64,
    pub(crate) ext_id: String,
    pub(crate) speaker: String,
    pub(crate) role: String,
    pub(crate) ts: Option<i64>,
    pub(crate) text: String,
}

impl<'a> Ingestor<'a> {
    pub fn new(store: &'a Store, embedder: &'a dyn Embedder, cfg: IngestConfig) -> Ingestor<'a> {
        Ingestor { store, embedder, cfg }
    }

    /// Stores a session's turns, its raw-dialogue chunks (embedded and FTS-indexed) and a
    /// pinned entity per speaker. After this the turns are searchable as raw text.
    pub async fn append_session(&self, session: SessionInput) -> Result<i64> {
        let timestamps = vec![session.started_at; session.turns.len()];
        self.append_timed_session(session, timestamps).await
    }

    /// Append immutable source with an exact timestamp for each message.
    pub async fn append_timed_session(&self, session: SessionInput, timestamps: Vec<i64>) -> Result<i64> {
        if timestamps.len() != session.turns.len() {
            return Err(Error::InvalidInput("one timestamp is required per turn".into()));
        }
        if session.ext_id.trim().is_empty() {
            return Err(Error::InvalidInput("session id is empty".into()));
        }
        let mut seen = HashSet::new();
        if session.turns.iter().any(|t| t.ext_id.trim().is_empty() || !seen.insert(&t.ext_id)) {
            return Err(Error::InvalidInput("turn ids must be nonempty and unique within the session".into()));
        }
        let source = session.clone();
        let source_times = timestamps.clone();
        if let Some(id) = self.store.read(move |conn| existing_session(conn, &source, &source_times)).await? {
            return Ok(id);
        }
        let chunk_texts: Vec<String> = session
            .turns
            .chunks(self.cfg.chunk_turns.max(1))
            .map(|c| c.iter().map(|t| format!("{}: {}", t.speaker, t.text)).collect::<Vec<_>>().join("\n"))
            .collect();
        let chunk_vectors = self.embed(&chunk_texts, EmbedKind::Document).await?;
        let speakers: Vec<String> = {
            let mut seen = HashSet::new();
            session.turns.iter().map(|t| t.speaker.clone()).filter(|s| seen.insert(s.clone())).collect()
        };
        let speaker_vectors = self
            .embed(&speakers.iter().map(|s| entity_text(s, "person", None)).collect::<Vec<_>>(), EmbedKind::Document)
            .await?;
        let chunk_turns = self.cfg.chunk_turns.max(1);
        self.store
            .write(move |w| {
                let conn = w.conn();
                if let Some(id)=existing_session(conn,&session,&timestamps)? {
                    return Ok(id);
                }
                conn.execute(
                    "INSERT INTO sessions(ext_id, started_at) VALUES (?1, ?2)",
                    params![session.ext_id, session.started_at],
                )?;
                let session_id = conn.last_insert_rowid();
                let mut turn_ids = Vec::with_capacity(session.turns.len());
                for (seq, t) in session.turns.iter().enumerate() {
                    conn.execute(
                        "INSERT INTO turns(session_id, seq, speaker, role, ts, text, ext_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![session_id, seq as i64, t.speaker, t.role.as_str(), timestamps[seq], t.text, t.ext_id],
                    )?;
                    turn_ids.push(conn.last_insert_rowid());
                }
                for (i, (text, vector)) in chunk_texts.iter().zip(&chunk_vectors).enumerate() {
                    let first = turn_ids[i * chunk_turns];
                    let last = turn_ids[((i + 1) * chunk_turns).min(turn_ids.len()) - 1];
                    let hash = fnv1a(format!("{}|{first}|{last}|{text}", session.ext_id).as_bytes());
                    w.conn().execute(
                        "INSERT INTO chunks(session_id, first_turn, last_turn, ts, text, hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                        params![session_id, first, last, timestamps[i * chunk_turns], text, hash.to_le_bytes().to_vec()],
                    )?;
                    let chunk_id = w.conn().last_insert_rowid();
                    w.index_text(LexicalKind::Chunk, chunk_id, text)?;
                    w.insert_vector(VectorKind::Chunk, chunk_id, vector)?;
                }
                for (name, vector) in speakers.iter().zip(&speaker_vectors) {
                    let alias = normalize(name);
                    if alias.is_empty() {
                        continue;
                    }
                    // Only a person answers for a speaker: a place called "Jordan" must
                    // not stop the speaker Jordan from being pinned.
                    let existing: Option<i64> = w
                        .conn()
                        .query_row(
                            "SELECT a.entity_id FROM entity_aliases a JOIN entities e ON e.id = a.entity_id \
                             WHERE a.alias = ?1 AND (e.kind = 'person' OR e.pinned = 1) LIMIT 1",
                            [&alias],
                            |r| r.get(0),
                        )
                        .optional()?;
                    if existing.is_none() {
                        w.conn().execute(
                            "INSERT INTO entities(name, kind, pinned, first_seen, last_seen) VALUES (?1, 'person', 1, ?2, ?2)",
                            params![name, session.started_at],
                        )?;
                        let id = w.conn().last_insert_rowid();
                        w.add_alias(id, &alias)?;
                        w.insert_vector(VectorKind::Entity, id, vector)?;
                    }
                }
                Ok(session_id)
            })
            .await
    }

    /// One extraction request per window of the session's turns.
    pub async fn extraction_requests(&self, session_ext: &str) -> Result<Vec<ModelRequest>> {
        let (started_at, turns) = self.session_turns(session_ext).await?;
        let speakers = self.speakers(session_ext).await?;
        Ok(extract::requests(session_ext, started_at, &turns, &speakers, &self.cfg))
    }

    /// Applies the model's output for one extraction request.
    pub async fn apply_extraction(&self, request_id: &str, output: &serde_json::Value) -> Result<ExtractionReport> {
        extract::apply(self, request_id, output, "").await
    }

    /// Apply a response whose extra preceding-source context is part of replay identity.
    pub async fn apply_extraction_with_context(
        &self,
        request_id: &str,
        output: &serde_json::Value,
        context: &str,
    ) -> Result<ExtractionReport> {
        extract::apply(self, request_id, output, context).await
    }

    /// The summary request for a session.
    pub async fn summary_request(&self, session_ext: &str) -> Result<ModelRequest> {
        let (started_at, turns) = self.session_turns(session_ext).await?;
        Ok(summary::request(session_ext, started_at, &turns, self.cfg.temporal_updates))
    }

    /// Applies a session summary.
    pub async fn apply_summary(&self, request_id: &str, output: &serde_json::Value) -> Result<()> {
        summary::apply(self, request_id, output).await
    }

    pub(crate) async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.embedder.embed(texts, kind).await.map_err(|e| Error::Task(e.to_string()))?;
        if vectors.len() != texts.len() {
            return Err(Error::Task(format!("embedder returned {} vectors for {} texts", vectors.len(), texts.len())));
        }
        Ok(vectors)
    }

    pub(crate) async fn session_turns(&self, session_ext: &str) -> Result<(i64, Vec<StoredTurn>)> {
        let ext = session_ext.to_owned();
        self.store
            .read(move |conn| {
                let (session_id, started_at): (i64, i64) = conn
                    .query_row("SELECT id, coalesce(started_at, 0) FROM sessions WHERE ext_id = ?1", [&ext], |r| {
                        Ok((r.get(0)?, r.get(1)?))
                    })
                    .optional()?
                    .ok_or_else(|| Error::Task(format!("no session {ext}")))?;
                let mut stmt = conn.prepare(
                    "SELECT id, seq, coalesce(ext_id, ''), speaker, text,role,ts FROM turns WHERE session_id = ?1 ORDER BY seq",
                )?;
                let turns = stmt
                    .query_map([session_id], |r| {
                        Ok(StoredTurn {
                            id: r.get(0)?,
                            seq: r.get(1)?,
                            ext_id: r.get(2)?,
                            speaker: r.get(3)?,
                            text: r.get(4)?,
                            role:r.get(5)?,
                            ts:r.get(6)?,
                        })
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((started_at, turns))
            })
            .await
    }

    async fn speakers(&self, session_ext: &str) -> Result<Vec<String>> {
        let ext = session_ext.to_owned();
        self.store
            .read(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT DISTINCT t.speaker FROM turns t JOIN sessions s ON s.id = t.session_id WHERE s.ext_id = ?1 ORDER BY t.seq",
                )?;
                Ok(stmt.query_map([&ext], |r| r.get(0))?.collect::<Result<Vec<String>, _>>()?)
            })
            .await
    }
}

fn existing_session(conn: &rusqlite::Connection, source: &SessionInput, timestamps: &[i64]) -> Result<Option<i64>> {
    let stored: Option<(i64, Option<i64>)> = conn
        .query_row("SELECT id,started_at FROM sessions WHERE ext_id=?1", [&source.ext_id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    let Some((id, time)) = stored else { return Ok(None) };
    let mut stmt =
        conn.prepare("SELECT coalesce(ext_id,''),speaker,role,text,ts FROM turns WHERE session_id=?1 ORDER BY seq")?;
    let turns: Vec<(String, String, String, String, Option<i64>)> = stmt
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?
        .collect::<std::result::Result<_, _>>()?;
    let equal = time == Some(source.started_at)
        && turns.len() == source.turns.len()
        && turns.iter().zip(&source.turns).zip(timestamps).all(|(((ext, speaker, role, text, ts), t), expected)| {
            ext == &t.ext_id
                && speaker == &t.speaker
                && role == t.role.as_str()
                && text == &t.text
                && *ts == Some(*expected)
        });
    if !equal {
        return Err(Error::SourceConflict { session: source.ext_id.clone() });
    }
    Ok(Some(id))
}

/// The canonical text an entity is embedded as (SPEC §8.6).
pub(crate) fn entity_text(name: &str, kind: &str, description: Option<&str>) -> String {
    match description {
        Some(d) if !d.trim().is_empty() => format!("{name} ({kind}): {d}"),
        _ => format!("{name} ({kind})"),
    }
}

/// Alias normal form: lower case; letters, digits, apostrophes and hyphens, plus the
/// symbols that tell names apart ("C++", "C#", "Node.js"); single spaces; no leading
/// article. A name with none of these (emoji only) keeps its trimmed lower case, so it
/// never collapses to the empty alias.
pub fn normalize(name: &str) -> String {
    let folded = fold(name, true);
    if folded.is_empty() { name.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ") } else { folded }
}

/// The predicate part of a fact identity key. Frozen at the original alias rules so
/// stored facts keep matching their re-mentions when [`normalize`] changes.
pub(crate) fn predicate_key(predicate: &str) -> String {
    fold(predicate, false)
}

/// `symbols`: keep `+`/`#` right after a letter or digit ("c++", "c#") and `.` between
/// two ("node.js"); anywhere else they separate words.
fn fold(text: &str, symbols: bool) -> String {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut cleaned = String::with_capacity(chars.len());
    for (i, &c) in chars.iter().enumerate() {
        let kept = c.is_alphanumeric()
            || c == '\''
            || c == '-'
            || (symbols
                && match c {
                    '+' | '#' => {
                        cleaned.chars().next_back().is_some_and(|p| p.is_alphanumeric() || p == '+' || p == '#')
                    }
                    '.' => {
                        cleaned.chars().next_back().is_some_and(char::is_alphanumeric)
                            && chars.get(i + 1).is_some_and(|n| n.is_alphanumeric())
                    }
                    _ => false,
                });
        cleaned.push(if kept { c } else { ' ' });
    }
    let mut words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.len() > 1 && matches!(words[0], "the" | "a" | "an") {
        words.remove(0);
    }
    words.join(" ")
}

/// FNV-1a, 64-bit: a stable content hash (std's hasher may change between releases).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::{normalize, predicate_key};

    #[test]
    fn alias_normal_form() {
        assert_eq!(normalize("  The LGBTQ Support-Group! "), "lgbtq support-group");
        assert_eq!(normalize("Caroline's mom"), "caroline's mom");
        assert_eq!(normalize("A"), "a");
    }

    #[test]
    fn symbols_that_distinguish_names_survive() {
        assert_eq!([normalize("C++"), normalize("C#"), normalize("C")], ["c++", "c#", "c"]);
        assert_eq!(normalize("Node.js"), "node.js");
        assert_eq!(normalize("St. Louis"), "st louis", "a trailing dot still separates");
        assert_eq!(normalize("Rock + Roll"), "rock roll");
        assert_eq!(normalize("#MeToo"), "metoo", "a leading hash is punctuation");
        assert_eq!(normalize("Caroline."), "caroline");
    }

    #[test]
    fn symbol_only_names_keep_an_alias_and_predicates_stay_frozen() {
        assert_eq!(normalize("🙂"), "🙂");
        assert_ne!(normalize("🙂"), normalize("🎸"));
        assert_eq!(normalize("   "), "");
        assert_eq!(predicate_key("Writes C++ in"), "writes c in", "identity keys must not move");
        assert_eq!(predicate_key("  Is Swamped With! "), "is swamped with");
    }
}
