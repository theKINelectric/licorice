//! Lexical search over FTS5 (SPEC §7.6).
//!
//! The FTS tables use external content, and the store writes and deletes their rows
//! explicitly in the same transaction as the source rows (no triggers, which keeps the
//! schema portable to engines without them).

use std::collections::HashSet;

use rusqlite::{Connection, params};

use super::Hit;
use crate::Result;

/// A lexically indexed kind of item.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LexicalKind {
    /// `facts.statement`; hits are fact ids.
    Fact,
    /// `chunks.text`; hits are chunk ids.
    Chunk,
    /// `entity_aliases.alias`, trigram-tokenized for fuzzy names; hits are entity ids.
    Alias,
}

impl LexicalKind {
    fn table(self) -> &'static str {
        match self {
            LexicalKind::Fact => "fts_fact",
            LexicalKind::Chunk => "fts_chunk",
            LexicalKind::Alias => "fts_alias",
        }
    }

    fn column(self) -> &'static str {
        match self {
            LexicalKind::Fact => "statement",
            LexicalKind::Chunk => "text",
            LexicalKind::Alias => "alias",
        }
    }
}

/// Adds `text` under `rowid` (the source row's id; for aliases, `entity_aliases.rowid`).
pub(crate) fn index(conn: &Connection, kind: LexicalKind, rowid: i64, text: &str) -> Result<()> {
    conn.execute(
        &format!("INSERT INTO {t}(rowid, {c}) VALUES (?1, ?2)", t = kind.table(), c = kind.column()),
        params![rowid, text],
    )?;
    Ok(())
}

/// Removes the entry for `rowid`. External-content FTS5 needs the exact text that was
/// indexed, so callers pass the old text.
pub(crate) fn unindex(conn: &Connection, kind: LexicalKind, rowid: i64, text: &str) -> Result<()> {
    conn.execute(
        &format!("INSERT INTO {t}({t}, rowid, {c}) VALUES ('delete', ?1, ?2)", t = kind.table(), c = kind.column()),
        params![rowid, text],
    )?;
    Ok(())
}

/// Best BM25 matches for free text, best first; scores are `-bm25` (higher is better).
/// Any `k` is accepted: there are never more hits than matches.
pub(crate) fn search(conn: &Connection, kind: LexicalKind, query: &str, k: usize) -> Result<Vec<Hit>> {
    let Some(expr) = match_expression(kind, query) else { return Ok(Vec::new()) };
    if k == 0 {
        return Ok(Vec::new());
    }
    let table = kind.table();
    // `k` can far exceed the matches, so `hits` grows as they arrive and the limit saturates.
    let limit = i64::try_from(k).unwrap_or(i64::MAX);
    let mut hits = Vec::new();
    match kind {
        LexicalKind::Fact | LexicalKind::Chunk => {
            let mut stmt = conn.prepare_cached(&format!(
                "SELECT rowid, bm25({table}) FROM {table} WHERE {table} MATCH ?1 ORDER BY bm25({table}) LIMIT ?2"
            ))?;
            let mut rows = stmt.query(params![expr, limit])?;
            while let Some(row) = rows.next()? {
                let bm25: f64 = row.get(1)?;
                hits.push(Hit { owner: row.get(0)?, score: -bm25 });
            }
        }
        LexicalKind::Alias => {
            // Several aliases can belong to one entity: over-fetch, keep the best per entity.
            let mut stmt = conn.prepare_cached(
                "SELECT a.entity_id, bm25(fts_alias) AS b FROM fts_alias JOIN entity_aliases AS a \
                 ON a.rowid = fts_alias.rowid WHERE fts_alias MATCH ?1 ORDER BY b LIMIT ?2",
            )?;
            let mut rows = stmt.query(params![expr, limit.saturating_mul(4)])?;
            let mut seen: HashSet<i64> = HashSet::new();
            while let Some(row) = rows.next()? {
                let owner: i64 = row.get(0)?;
                if seen.insert(owner) {
                    let bm25: f64 = row.get(1)?;
                    hits.push(Hit { owner, score: -bm25 });
                    if hits.len() == k {
                        break;
                    }
                }
            }
        }
    }
    Ok(hits)
}

/// Turns free text into a safe FTS5 expression, or `None` if nothing searchable remains.
///
/// Only letters and digits survive, so no user input can reach FTS5 query syntax
/// (quotes, `AND`/`OR`/`NOT`/`NEAR`, `*`, `-`, column filters). Words are OR-ed so a
/// partial match still ranks. Aliases are matched by their trigrams, which lets
/// "Melany" find "Melanie".
pub(crate) fn match_expression(kind: LexicalKind, query: &str) -> Option<String> {
    let words = words(query);
    let mut terms: Vec<String> = Vec::new();
    match kind {
        LexicalKind::Fact | LexicalKind::Chunk => {
            for w in words {
                if w.chars().count() >= 2 && !STOPWORDS.contains(&w.as_str()) && !terms.contains(&w) {
                    terms.push(w);
                }
            }
        }
        LexicalKind::Alias => {
            for w in words {
                let chars: Vec<char> = w.chars().collect();
                for tri in chars.windows(3) {
                    let tri: String = tri.iter().collect();
                    if !terms.contains(&tri) {
                        terms.push(tri);
                    }
                }
            }
        }
    }
    terms.truncate(64);
    if terms.is_empty() {
        return None;
    }
    Some(terms.iter().map(|t| format!("\"{t}\"")).collect::<Vec<_>>().join(" OR "))
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

const STOPWORDS: &[&str] = &[
    "a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "been", "before", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "than", "that", "the",
    "their", "them", "then", "there", "they", "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
];
