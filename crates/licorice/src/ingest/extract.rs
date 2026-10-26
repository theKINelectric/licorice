//! Windowed fact extraction (SPEC §8.2–§8.4): render a window as a model request, then
//! resolve entities and store facts from the model's JSON.

use std::collections::HashMap;

use licorice_providers::chat::ModelRequest;
use licorice_providers::embed::EmbedKind;
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::application::Application;
use super::{IngestConfig, Ingestor, StoredTurn, entity_text, normalize, prompts, updates};
use crate::store::{Filter, Index, LexicalKind, VectorKind};
use crate::temporal::{self, Day};
use crate::{Error, Result};

const KINDS: [&str; 6] = ["person", "organization", "location", "event", "concept", "object"];

/// What applying one extraction did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtractionReport {
    /// The original report was recovered; no writes or embeddings were needed.
    pub replayed: bool,
    pub entities_created: usize,
    pub entities_linked: usize,
    pub facts_created: usize,
    pub facts_merged: usize,
    /// Facts dropped for citing no turn inside the window, or for being malformed.
    pub facts_dropped: usize,
    /// Facts whose time expression resolved to a date interval.
    pub facts_dated: usize,
    /// Source-backed state changes accepted by the opt-in temporal profile.
    #[serde(default)]
    pub updates_applied: usize,
}

pub(crate) fn request_id(session_ext: &str, first_seq: i64, last_seq: i64) -> String {
    format!("extract:{session_ext}:{first_seq}-{last_seq}")
}

fn parse_request_id(id: &str) -> Result<(String, i64, i64)> {
    let bad = || Error::Task(format!("not an extraction request id: {id}"));
    let rest = id.strip_prefix("extract:").ok_or_else(bad)?;
    let (session, range) = rest.rsplit_once(':').ok_or_else(bad)?;
    let (a, b) = range.split_once('-').ok_or_else(bad)?;
    Ok((session.to_owned(), a.parse().map_err(|_| bad())?, b.parse().map_err(|_| bad())?))
}

pub(crate) fn transcript_line(t: &StoredTurn) -> String {
    format!("[{}] {}: {}", t.ext_id, t.speaker, t.text)
}

fn dated_line(t: &StoredTurn) -> String {
    format!("[{}] [UTC date {}] {}: {}", t.ext_id, Day::from_ms(t.ts.unwrap_or(0)).iso(), t.speaker, t.text)
}

pub(crate) fn requests(
    session_ext: &str,
    started_at: i64,
    turns: &[StoredTurn],
    speakers: &[String],
    cfg: &IngestConfig,
) -> Vec<ModelRequest> {
    let date = Day::from_ms(started_at).human();
    let w = cfg.window_turns.max(1);
    (0..turns.len())
        .step_by(w)
        .map(|start| {
            let end = (start + w).min(turns.len());
            let context = &turns[start.saturating_sub(cfg.context_turns)..start];
            let mut user = if cfg.temporal_updates {
                "Each turn carries its own UTC date.\n\n".into()
            } else {
                format!("Session date: {date}\nSpeakers: {}\n\n", speakers.join(", "))
            };
            let line = |t: &StoredTurn| if cfg.temporal_updates { dated_line(t) } else { transcript_line(t) };
            if !context.is_empty() {
                user.push_str("Earlier turns (context only):\n");
                for t in context {
                    user.push_str(&line(t));
                    user.push('\n');
                }
                user.push('\n');
            }
            user.push_str("Turns to extract from:\n");
            for t in &turns[start..end] {
                user.push_str(&line(t));
                user.push('\n');
            }
            user.push('\n');
            user.push_str(prompts::EXTRACT_INSTRUCTIONS);
            if cfg.temporal_updates {
                user.push_str(prompts::TEMPORAL_INSTRUCTIONS);
            }
            ModelRequest {
                id: request_id(session_ext, turns[start].seq, turns[end - 1].seq),
                task: "extract".into(),
                system: prompts::EXTRACT_SYSTEM.into(),
                user,
                schema: Some(if cfg.temporal_updates { prompts::temporal_schema() } else { prompts::extract_schema() }),
                max_tokens: 4096,
            }
        })
        .collect()
}

struct EntityIn {
    name: String,
    kind: String,
    aliases: Vec<String>,
    description: Option<String>,
}

struct FactIn {
    subject: String,
    predicate: String,
    object: Option<String>,
    object_is_entity: bool,
    statement: String,
    negated: bool,
    salience: Option<f64>,
    time_expression: Option<String>,
    time_kind: String,
    evidence: Vec<i64>,
    anchor: i64,
    recorded_first: i64,
    recorded_last: i64,
    update: Option<(String, String, i64, String)>,
}

fn string(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned)
}

/// Where an entity name resolved.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    Existing(i64),
    New(usize),
}

pub(crate) async fn apply(
    ing: &Ingestor<'_>,
    request_id: &str,
    output: &Value,
    extra_context: &str,
) -> Result<ExtractionReport> {
    let (session_ext, first_seq, last_seq) = parse_request_id(request_id)?;
    let (started_at, turns) = ing.session_turns(&session_ext).await?;
    if first_seq < 0 || last_seq < first_seq || last_seq >= turns.len() as i64 {
        return Err(Error::InvalidInput("extraction range is outside the session".into()));
    }
    let context_first = first_seq.saturating_sub(i64::try_from(ing.cfg.context_turns).unwrap_or(i64::MAX)).max(0);
    let source: Vec<_> = turns.iter().filter(|t| context_first <= t.seq && t.seq <= last_seq).collect();
    let mut application_input = json!({
        "identity_version":2,"date":started_at,"turns":source,"config":ing.cfg,
        "embedder":ing.embedder.id(),"embed_dim":ing.embedder.dim(),
        "system":prompts::EXTRACT_SYSTEM,"instructions": if ing.cfg.temporal_updates {format!("{}{}",prompts::EXTRACT_INSTRUCTIONS,prompts::TEMPORAL_INSTRUCTIONS)} else {prompts::EXTRACT_INSTRUCTIONS.to_owned()},
    });
    if !extra_context.is_empty() {
        application_input["earlier_source_context"] = json!(extra_context);
    }
    let application = Application::new(request_id, "extract", &session_ext, &application_input, output);
    if let Some(receipt) = application.lookup(ing.store).await? {
        return replay_report(receipt);
    }
    let window: HashMap<String, i64> =
        turns.iter().filter(|t| (first_seq..=last_seq).contains(&t.seq)).map(|t| (t.ext_id.clone(), t.id)).collect();
    let current_ids: Vec<_> =
        turns.iter().filter(|t| (first_seq..=last_seq).contains(&t.seq)).map(|t| t.ext_id.as_str()).collect();
    let temporal = ing.cfg.temporal_updates;
    let mut report = ExtractionReport::default();

    // ---- parse --------------------------------------------------------------------
    let mut entities: Vec<EntityIn> = Vec::new();
    for e in output.get("entities").and_then(Value::as_array).into_iter().flatten() {
        let Some(name) = string(e, "name").filter(|n| n.chars().count() <= 120) else { continue };
        let kind = string(e, "kind").filter(|k| KINDS.contains(&k.as_str())).unwrap_or_else(|| "concept".into());
        let aliases = e
            .get("aliases")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        entities.push(EntityIn { name, kind, aliases, description: string(e, "description") });
    }
    let mut facts: Vec<FactIn> = Vec::new();
    for f in output.get("facts").and_then(Value::as_array).into_iter().flatten() {
        let (Some(subject), Some(statement)) = (string(f, "subject"), string(f, "statement")) else {
            report.facts_dropped += 1;
            continue;
        };
        let evidence: Vec<i64> = f
            .get("evidence_turns")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter_map(|id| window.get(id.trim().trim_matches(['[', ']'])).copied())
            .collect();
        if evidence.is_empty() {
            if temporal && f.get("update").is_some_and(|u| !u.is_null()) {
                return Err(Error::InvalidInput(format!(
                    "update evidence_turns must cite a current supporting turn, not earlier context; current turn IDs: {current_ids:?}"
                )));
            }
            report.facts_dropped += 1;
            continue;
        }
        let support: Vec<_> = turns.iter().filter(|t| evidence.contains(&t.id)).collect();
        let effective_expression =
            if temporal { f.get("update").and_then(|u| string(u, "effective_expression")) } else { None };
        if effective_expression
            .as_ref()
            .is_some_and(|expression| !support.iter().any(|turn| turn.text.contains(expression)))
        {
            return Err(Error::InvalidInput(
                "an update effective_expression must copy exact text from a supporting current turn".into(),
            ));
        }
        // A current-state assertion can say "now" even when its explicit change
        // took effect earlier. Preserve the dated transition, not the report date.
        let time_expression = effective_expression.or_else(|| f.get("time").and_then(|t| string(t, "expression")));
        let selected = f.get("time").and_then(|t| string(t, "anchor_turn"));
        let anchor_turn = match selected {
            Some(ref id) if temporal => match support.iter().find(|t| t.ext_id == *id) {
                Some(turn) => turn,
                None => {
                    if f.get("update").is_some_and(|u| !u.is_null()) {
                        return Err(Error::InvalidInput(format!(
                            "an update time.anchor_turn must be both current and cited in evidence_turns; current turn IDs: {current_ids:?}"
                        )));
                    }
                    report.facts_dropped += 1;
                    continue;
                }
            },
            _ => &support[0],
        };
        let anchor = if temporal { anchor_turn.ts.unwrap_or(started_at) } else { started_at };
        let recorded_first =
            if temporal { support.iter().map(|t| t.ts.unwrap_or(started_at)).min().unwrap() } else { started_at };
        let recorded_last =
            if temporal { support.iter().map(|t| t.ts.unwrap_or(started_at)).max().unwrap() } else { started_at };
        let update = if temporal {
            f.get("update")
                .filter(|u| !u.is_null())
                .map(|u| {
                    let previous = string(u, "previous_object")
                        .ok_or_else(|| Error::InvalidInput("update requires a previous object".into()))?;
                    let quote = string(u, "evidence_quote")
                        .filter(|q| q.chars().count() >= 8)
                        .ok_or_else(|| Error::InvalidInput("update requires an exact source quote".into()))?;
                    let source = support
                        .iter()
                        .find(|t| t.text.contains(&quote))
                        .ok_or_else(|| Error::InvalidInput("update quote is absent from supporting turns".into()))?;
                    let kind = string(u, "kind").unwrap_or_else(|| "replace".into());
                    if !matches!(kind.as_str(), "replace" | "correction") {
                        return Err(Error::InvalidInput("invalid state update kind".into()));
                    }
                    Ok::<_, Error>((previous, quote, source.id, kind))
                })
                .transpose()?
        } else {
            None
        };
        let predicate = {
            let predicate = string(f, "predicate").unwrap_or_else(|| "relates to".into());
            if temporal && matches!(super::predicate_key(&predicate).as_str(), "lived in" | "resides in" | "resided in")
            {
                "lives in".into()
            } else {
                predicate
            }
        };
        let object = string(f, "object");
        let time_kind = f
            .get("time")
            .and_then(|t| string(t, "kind"))
            .filter(|k| matches!(k.as_str(), "instant" | "interval" | "ongoing" | "unknown"))
            .unwrap_or_else(|| "unknown".into());
        if temporal && super::predicate_key(&predicate) == "lives in" {
            if object.is_none() {
                return Err(Error::InvalidInput(
                    "a residence fact requires the source's place in object; null cannot identify a prior home".into(),
                ));
            }
            if time_kind == "instant" {
                return Err(Error::InvalidInput("residence predicate 'lives in' is a state, not an instant event: use ongoing for current/continuing residence, interval for explicitly bounded past residence; a separate move event may be instant".into()));
            }
        }
        facts.push(FactIn {
            subject,
            predicate,
            object,
            object_is_entity: f.get("object_is_entity").and_then(Value::as_bool).unwrap_or(false),
            statement,
            negated: f.get("negated").and_then(Value::as_bool).unwrap_or(false),
            salience: f.get("salience").and_then(Value::as_f64).map(|s| (s / 10.0).clamp(0.0, 1.0)),
            time_expression,
            time_kind,
            evidence,
            anchor,
            recorded_first,
            recorded_last,
            update,
        });
    }

    // ---- resolve entities -----------------------------------------------------------
    // Every name a fact uses must resolve, including names the model left out of `entities`.
    let mut wanted: Vec<EntityIn> = entities;
    let mut known: std::collections::HashSet<String> = wanted.iter().map(|e| normalize(&e.name)).collect();
    for f in &facts {
        let mut names = vec![f.subject.clone()];
        if f.object_is_entity {
            names.extend(f.object.clone());
            names.extend(f.update.as_ref().map(|u| u.0.clone()));
        }
        for n in names {
            if known.insert(normalize(&n)) {
                wanted.push(EntityIn { name: n, kind: "concept".into(), aliases: Vec::new(), description: None });
            }
        }
    }
    let texts: Vec<String> = wanted.iter().map(|e| entity_text(&e.name, &e.kind, e.description.as_deref())).collect();
    let vectors = ing.embed(&texts, EmbedKind::Document).await?;
    let mut by_name: HashMap<String, Target> = HashMap::new();
    let mut new_entities: Vec<usize> = Vec::new();
    // Surface forms of linked entities that aren't aliases yet (SPEC §8.3).
    let mut extra_aliases: Vec<(i64, String)> = Vec::new();
    // Normalized aliases accepted for each new entity, by index into `wanted`.
    let mut new_aliases: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, e) in wanted.iter().enumerate() {
        let norm = normalize(&e.name);
        if by_name.contains_key(&norm) {
            continue;
        }
        let target = match lookup_alias(ing, &norm, &e.kind).await? {
            Some(id) => Target::Existing(id),
            None => match similar_entity(ing, &vectors[i], &e.kind).await? {
                Some(id) => Target::Existing(id),
                None => {
                    new_entities.push(i);
                    Target::New(i)
                }
            },
        };
        if let Target::Existing(id) = target {
            report.entities_linked += 1;
            extra_aliases.push((id, norm.clone()));
        }
        by_name.insert(norm, target);
        // An alias names this entity only if no other stored entity already answers to
        // it: a new "Mel" listing "Melanie" must not capture the existing Melanie.
        let mut accepted = Vec::new();
        for a in &e.aliases {
            let a = normalize(a);
            if a.is_empty() || by_name.contains_key(&a) {
                continue;
            }
            if let Some(owner) = lookup_alias(ing, &a, &e.kind).await? {
                if target != Target::Existing(owner) {
                    continue;
                }
            }
            by_name.insert(a.clone(), target);
            match target {
                Target::Existing(id) => extra_aliases.push((id, a)),
                Target::New(_) => accepted.push(a),
            }
        }
        new_aliases.insert(i, accepted);
    }

    // ---- facts: embed statements, then one write transaction --------------------------
    let statements: Vec<String> = facts.iter().map(|f| f.statement.clone()).collect();
    let fact_vectors = ing.embed(&statements, EmbedKind::Document).await?;
    let mut prepared = Vec::with_capacity(facts.len());
    for (f, v) in facts.into_iter().zip(fact_vectors) {
        let interval = f.time_expression.as_deref().and_then(|e| temporal::resolve(e, Day::from_ms(f.anchor)));
        if interval.is_some() {
            report.facts_dated += 1;
        }
        prepared.push((f, v, interval));
    }
    let new_list: Vec<(EntityIn, Vec<f32>)> = new_entities
        .iter()
        .map(|&i| {
            let e = &wanted[i];
            let copy = EntityIn {
                name: e.name.clone(),
                kind: e.kind.clone(),
                aliases: new_aliases.remove(&i).unwrap_or_default(),
                description: e.description.clone(),
            };
            (copy, vectors[i].clone())
        })
        .collect();
    let new_index: HashMap<usize, usize> = new_entities.iter().enumerate().map(|(pos, &i)| (i, pos)).collect();
    let resolved: HashMap<String, Target> = by_name;

    let application_id = request_id.to_owned();
    ing
        .store
        .write(move |w| {
            if let Some(receipt)=application.check(w.conn())? {
                return replay_report(receipt);
            }
            let mut ids: Vec<i64> = Vec::with_capacity(new_list.len());
            for (EntityIn { name, kind, aliases, description }, vector) in &new_list {
                // Another application may have inserted the same alias while the
                // embeddings were in flight. Resolve again under the writer lock.
                let existing:Option<i64>=w.conn().query_row(
                    "SELECT coalesce(e.merged_into,e.id) FROM entity_aliases a JOIN entities e ON e.id=a.entity_id \
                     WHERE a.alias=?1 AND (e.kind=?2 OR e.pinned=1) ORDER BY (e.kind=?2) DESC,e.id LIMIT 1",
                    params![normalize(name),kind],|r|r.get(0)).optional()?;
                if let Some(id)=existing {
                    ids.push(id);
                    report.entities_linked+=1;
                    for alias in aliases {w.add_alias(id,alias)?;}
                    continue;
                }
                w.conn().execute(
                    "INSERT INTO entities(name, kind, description, first_seen, last_seen) VALUES (?1, ?2, ?3, ?4, ?4)",
                    params![name, kind, description, started_at],
                )?;
                let id = w.conn().last_insert_rowid();
                w.add_alias(id, &normalize(name))?;
                for a in aliases {
                    w.add_alias(id, a)?; // normalized and non-empty
                }
                w.insert_vector(VectorKind::Entity, id, vector)?;
                ids.push(id);
                report.entities_created+=1;
            }
            for (id, alias) in &extra_aliases {
                w.add_alias(*id, alias)?; // no-op when it already exists
            }
            let id_of = |name: &str| -> Option<i64> {
                match resolved.get(&normalize(name))? {
                    Target::Existing(id) => Some(*id),
                    Target::New(i) => new_index.get(i).map(|pos| ids[*pos]),
                }
            };
            let (mut created, mut merged) = (0usize, 0usize);
            for (f, vector, interval) in &prepared {
                let Some(subject_id) = id_of(&f.subject) else { continue };
                let object_id = if f.object_is_entity { f.object.as_deref().and_then(id_of) } else { None };
                let object_value = if object_id.is_some() { None } else { Some(f.object.clone().unwrap_or_default()) };
                // Structured keys distinguish an entity id from a literal that
                // happens to contain that id, and preserve literal punctuation.
                let dedup_key=updates::key(subject_id,&f.predicate,object_id,&object_value,f.negated);
                let existing=matching_fact(w.conn(),&dedup_key,f,*interval)?;
                let fact_id = match existing {
                    Some(id) => {
                        w.conn().execute(
                            "UPDATE facts SET recorded_first=min(recorded_first,?1),recorded_last=max(recorded_last,?2) WHERE id=?3",
                            params![f.recorded_first,f.recorded_last, id],
                        )?;
                        // Refinement can remain unresolved. Keep the expression
                        // so a later, incompatible qualifier cannot match a NULL.
                        w.conn().execute("UPDATE facts SET time_text=coalesce(time_text,?1) WHERE id=?2",params![f.time_expression,id])?;
                        if let Some(interval)=interval {
                            w.conn().execute("UPDATE facts SET occurred_from=?1,occurred_to=?2,time_precision=?3,time_text=?4 \
                                WHERE id=?5 AND occurred_from IS NULL",
                                params![interval.from.start_ms(),interval.to.end_ms(),interval.precision.as_str(),f.time_expression,id])?;
                        }
                        if f.time_kind!="unknown" {
                            w.conn().execute("UPDATE facts SET time_kind=?1 WHERE id=?2 AND time_kind='unknown'",params![f.time_kind,id])?;
                        }
                        merged += 1;
                        id
                    }
                    None => {
                        w.conn().execute(
                            "INSERT INTO facts(subject_id, predicate, object_id, object_value, statement, negated, salience, \
                             occurred_from, occurred_to, time_precision, time_text, recorded_first, recorded_last, dedup_key,identity_version,time_kind) \
                             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12, ?13,2,?14)",
                            params![
                                subject_id,
                                f.predicate,
                                object_id,
                                object_value,
                                f.statement,
                                f.negated as i64,
                                f.salience,
                                interval.map(|i| i.from.start_ms()),
                                interval.map(|i| i.to.end_ms()),
                                interval.map(|i| i.precision.as_str()),
                                f.time_expression,
                                f.recorded_first,
                                dedup_key,
                                f.time_kind
                            ],
                        )?;
                        let id = w.conn().last_insert_rowid();
                        w.conn().execute("UPDATE facts SET recorded_last=?1,transaction_created_at=?2 WHERE id=?3",
                            params![f.recorded_last,crate::store::now_ms(),id])?;
                        w.index_text(LexicalKind::Fact, id, &f.statement)?;
                        w.insert_vector(VectorKind::Fact, id, vector)?;
                        created += 1;
                        id
                    }
                };
                for turn_id in &f.evidence {
                    let chunk: Option<i64> = w
                        .conn()
                        .query_row(
                            "SELECT id FROM chunks WHERE session_id = (SELECT session_id FROM turns WHERE id = ?1) \
                             AND first_turn <= ?1 AND last_turn >= ?1 ORDER BY id LIMIT 1",
                            [turn_id],
                            |r| r.get(0),
                        )
                        .optional()?;
                    w.conn().execute(
                        "INSERT OR IGNORE INTO fact_evidence(fact_id, turn_id, chunk_id) VALUES (?1, ?2, ?3)",
                        params![fact_id, turn_id, chunk],
                    )?;
                }
                if temporal {
                    let now=crate::store::now_ms();
                    if let Some((previous,quote,source,kind))=&f.update {
                        let same_object=updates::matches_object(w.conn(),object_id,&object_value,previous)?;
                        if (kind=="replace" && f.time_kind!="ongoing") || (f.negated && !same_object) {
                            return Err(Error::InvalidInput("a replacement requires a new ongoing state; a negative update must target the same object; a correction need not invent an ongoing time span".into()));
                        }
                        let previous_id=if f.object_is_entity {id_of(previous)} else {None};
                        let previous_value=if previous_id.is_some(){None}else{Some(previous.clone())};
                        let previous_negated=same_object && !f.negated;
                        let previous_key=updates::key(subject_id,&f.predicate,previous_id,&previous_value,previous_negated);
                        let target_keys=updates::target_keys(w,subject_id,&f.predicate,previous,previous_negated)?;
                        if previous_negated && target_keys.is_empty() {
                            return Err(Error::InvalidInput("a same-value positive update needs an existing opposite-polarity assertion".into()));
                        }
                        let precise=interval.filter(|i|i.precision==temporal::Precision::Day);
                        let effective=precise.map(|i|i.from.start_ms()).unwrap_or(f.anchor);
                        let basis=if precise.is_some(){"explicit_day"}else{"known_by_source_date"};
                        let inserted=w.conn().execute("INSERT OR IGNORE INTO fact_updates(previous_key,next_fact_id,effective_at,effective_basis,source_turn_id,evidence_quote,observed_at,application_id,kind,assertion_at,subject_id,predicate_key,previous_object,previous_negated) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
                            params![previous_key,fact_id,effective,basis,source,quote,now,application_id,kind,f.anchor,subject_id,super::predicate_key(&f.predicate),previous,previous_negated])?;
                        for target in target_keys {updates::refresh(w,&target,now)?;}
                        report.updates_applied+=inserted;
                    }
                    updates::refresh(w,&dedup_key,now)?;
                }
                // Mentions count independent supporting turns, not repeated model
                // paraphrases or overlapping extraction windows.
                w.conn().execute("UPDATE facts SET mentions=(SELECT count(*) FROM fact_evidence WHERE fact_id=?1) WHERE id=?1",[fact_id])?;
            }
            report.facts_created=created;
            report.facts_merged=merged;
            application.finish(w.conn(),&json!(report))?;
            Ok(report)
        })
        .await
}

fn replay_report(receipt: Value) -> Result<ExtractionReport> {
    let mut report: ExtractionReport = serde_json::from_value(receipt).map_err(|e| Error::Task(e.to_string()))?;
    report.replayed = true;
    Ok(report)
}

fn matching_fact(
    conn: &rusqlite::Connection,
    key: &str,
    fact: &FactIn,
    interval: Option<temporal::Interval>,
) -> Result<Option<i64>> {
    let mut stmt = conn.prepare(
        "SELECT id,occurred_from,occurred_to,time_text,time_kind FROM facts \
        WHERE dedup_key=?1 AND identity_version=2 AND status='active' ORDER BY id",
    )?;
    let candidates = stmt.query_map([key], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Option<i64>>(1)?,
            r.get::<_, Option<i64>>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, String>(4)?,
        ))
    })?;
    for candidate in candidates {
        let (id, from, to, text, kind) = candidate?;
        if kind != fact.time_kind && kind != "unknown" && fact.time_kind != "unknown" {
            continue;
        }
        let compatible = match (from, interval) {
            (Some(a), Some(b)) => a == b.from.start_ms() && to == Some(b.to.end_ms()),
            (Some(_), None) => fact.time_expression.is_none(),
            (None, Some(_)) => text.is_none(),
            (None, None) => text == fact.time_expression || text.is_none() || fact.time_expression.is_none(),
        };
        if !compatible {
            continue;
        }
        let mut shared = false;
        for turn in &fact.evidence {
            if conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM fact_evidence WHERE fact_id=?1 AND turn_id=?2)",
                params![id, turn],
                |r| r.get::<_, bool>(0),
            )? {
                shared = true;
                break;
            }
        }
        let same_state = kind == "ongoing"
            && fact.time_kind == "ongoing"
            && match (from, interval) {
                (Some(a), Some(b)) => a == b.from.start_ms() && to == Some(b.to.end_ms()),
                (None, None) => text == fact.time_expression,
                _ => false,
            };
        if shared || same_state {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

/// An existing entity with this alias: same kind first, then any kind for a pinned
/// (speaker) entity.
async fn lookup_alias(ing: &Ingestor<'_>, alias: &str, kind: &str) -> Result<Option<i64>> {
    let (alias, kind) = (alias.to_owned(), kind.to_owned());
    ing.store
        .read(move |conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT coalesce(e.merged_into, e.id), e.kind, e.pinned FROM entity_aliases a \
                 JOIN entities e ON e.id = a.entity_id WHERE a.alias = ?1 ORDER BY e.id",
            )?;
            let rows: Vec<(i64, String, i64)> =
                stmt.query_map([&alias], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<Result<_, _>>()?;
            Ok(rows
                .iter()
                .find(|(_, k, _)| *k == kind)
                .or_else(|| rows.iter().find(|(_, _, pinned)| *pinned == 1))
                .map(|(id, _, _)| *id))
        })
        .await
}

/// The nearest existing entity of the same kind, if it clears the merge threshold.
async fn similar_entity(ing: &Ingestor<'_>, vector: &[f32], kind: &str) -> Result<Option<i64>> {
    let hits = ing.store.knn(VectorKind::Entity, vector, 5, Filter::None).await?;
    let threshold = ing.cfg.merge_threshold;
    let candidates: Vec<i64> = hits.iter().filter(|h| h.score >= threshold).map(|h| h.owner).collect();
    if candidates.is_empty() {
        return Ok(None);
    }
    let kind = kind.to_owned();
    ing.store
        .read(move |conn| {
            for id in &candidates {
                let k: Option<String> =
                    conn.query_row("SELECT kind FROM entities WHERE id = ?1", [id], |r| r.get(0)).optional()?;
                if k.as_deref() == Some(kind.as_str()) {
                    return Ok(Some(*id));
                }
            }
            Ok(None)
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_ids_round_trip() {
        let id = request_id("D12", 8, 15);
        assert_eq!(parse_request_id(&id).unwrap(), ("D12".to_owned(), 8, 15));
        assert!(parse_request_id("summary:D1").is_err());
    }
}
