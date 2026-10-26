//! Session summaries (SPEC §8.5): the keys-and-themes layer used for session scores.

use licorice_providers::chat::ModelRequest;
use licorice_providers::embed::EmbedKind;
use rusqlite::{OptionalExtension, params};
use serde_json::{Value, json};

use super::application::Application;
use super::extract::transcript_line;
use super::{Ingestor, StoredTurn, prompts};
use crate::store::{VectorKind, now_ms};
use crate::temporal::Day;
use crate::{Error, Result};

pub(crate) fn request(session_ext: &str, started_at: i64, turns: &[StoredTurn], temporal: bool) -> ModelRequest {
    let mut user = format!("Session {session_ext}, {}\n\n", Day::from_ms(started_at).human());
    for t in turns {
        if temporal {
            user.push_str(&format!(
                "[{}] [UTC date {}] {}: {}",
                t.ext_id,
                Day::from_ms(t.ts.unwrap_or(started_at)).iso(),
                t.speaker,
                t.text
            ));
        } else {
            user.push_str(&transcript_line(t));
        }
        user.push('\n');
    }
    user.push('\n');
    user.push_str(prompts::SUMMARY_INSTRUCTIONS);
    ModelRequest {
        id: format!("summary:{session_ext}"),
        task: "summarize".into(),
        system: prompts::SUMMARY_SYSTEM.into(),
        user,
        schema: Some(prompts::summary_schema()),
        max_tokens: 2048,
    }
}

pub(crate) async fn apply(ing: &Ingestor<'_>, request_id: &str, output: &Value) -> Result<()> {
    let session_ext = request_id
        .strip_prefix("summary:")
        .ok_or_else(|| Error::Task(format!("not a summary request id: {request_id}")))?
        .to_owned();
    let (started_at, turns) = ing.session_turns(&session_ext).await?;
    let application = Application::new(
        request_id,
        "summarize",
        &session_ext,
        &json!({
            "version":2,"request":request(&session_ext,started_at,&turns,ing.cfg.temporal_updates),"started_at":started_at,"turns":turns,
            "embedder":ing.embedder.id(),"embed_dim":ing.embedder.dim(),
        }),
        output,
    );
    if application.lookup(ing.store).await?.is_some() {
        return Ok(());
    }
    let keys: Vec<String> = output
        .get("keys")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|k| k.trim().to_owned())
        .filter(|k| !k.is_empty())
        .take(5)
        .collect();
    let themes: Vec<(String, String)> = output
        .get("themes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| {
            let title = t.get("title").and_then(Value::as_str)?.trim().to_owned();
            let summary = t.get("summary").and_then(Value::as_str)?.trim().to_owned();
            (!title.is_empty() || !summary.is_empty()).then_some((title, summary))
        })
        .take(5)
        .collect();
    let date = Day::from_ms(started_at).human();
    let text = format!(
        "Session {session_ext} ({date}): {}",
        themes.iter().map(|(t, s)| format!("{t}: {s}")).collect::<Vec<_>>().join(" ")
    );
    let theme_texts: Vec<String> = themes.iter().map(|(t, s)| format!("{t}: {s}")).collect();
    let mut texts: Vec<String> = keys.clone();
    texts.extend(theme_texts.iter().cloned());
    texts.push(text.clone());
    let vectors = ing.embed(&texts, EmbedKind::Document).await?;
    let body = json!({"keys": keys, "themes": themes.iter().map(|(t, s)| json!({"title": t, "summary": s})).collect::<Vec<_>>()});
    ing.store
        .write(move |w| {
            if application.check(w.conn())?.is_some() {
                return Ok(());
            }
            let session_id: i64 = w
                .conn()
                .query_row("SELECT id FROM sessions WHERE ext_id = ?1", [&session_ext], |r| r.get(0))
                .optional()?
                .ok_or_else(|| Error::Task(format!("no session {session_ext}")))?;
            let version: i64 = w
                .conn()
                .query_row("SELECT version FROM summaries WHERE session_id = ?1", [session_id], |r| r.get(0))
                .optional()?
                .unwrap_or(0);
            w.conn().execute(
                "INSERT INTO summaries(session_id, version, body, text, updated_at) VALUES (?1, ?2, ?3, ?4, ?5) \
                 ON CONFLICT(session_id) DO UPDATE SET version = excluded.version, body = excluded.body, \
                 text = excluded.text, updated_at = excluded.updated_at",
                params![session_id, version + 1, body.to_string(), text, now_ms()],
            )?;
            // Replace the previous version's keys and themes.
            let old: Vec<i64> = {
                let mut stmt = w.conn().prepare("SELECT id FROM summary_keys WHERE session_id = ?1 AND live = 1")?;
                stmt.query_map([session_id], |r| r.get(0))?.collect::<Result<_, _>>()?
            };
            for id in old {
                w.conn().execute("UPDATE summary_keys SET live = 0 WHERE id = ?1", [id])?;
                w.retire_vectors(VectorKind::Key, id)?;
            }
            w.retire_vectors(VectorKind::Summary, session_id)?;
            let mut v = vectors.iter();
            for (kind, entry) in keys.iter().map(|k| ("key", k)).chain(theme_texts.iter().map(|t| ("theme", t))) {
                w.conn().execute(
                    "INSERT INTO summary_keys(session_id, kind, text) VALUES (?1, ?2, ?3)",
                    params![session_id, kind, entry],
                )?;
                let id = w.conn().last_insert_rowid();
                w.insert_vector(VectorKind::Key, id, v.next().expect("one vector per key"))?;
            }
            w.insert_vector(VectorKind::Summary, session_id, v.next().expect("one vector for the text"))?;
            application.finish(w.conn(), &json!({"version":version+1}))?;
            Ok(())
        })
        .await
}
