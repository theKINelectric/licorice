//! Explicit state transitions. Similarity and recency never imply replacement.
use crate::Result;
use crate::store::WriteTx;
use rusqlite::{Connection, params};
use serde_json::json;

pub(super) fn key(
    subject: i64,
    predicate: &str,
    object_id: Option<i64>,
    object_value: &Option<String>,
    negated: bool,
) -> String {
    blake3::hash(
        json!([subject, super::predicate_key(predicate), object_id, object_value, negated]).to_string().as_bytes(),
    )
    .to_hex()
    .to_string()
}

/// Literal values retain exact spelling. Entity references may use an alias,
/// but only if it resolves uniquely; an ambiguous name cannot invalidate a fact.
pub(super) fn matches_object(
    conn: &Connection,
    id: Option<i64>,
    value: &Option<String>,
    previous: &str,
) -> Result<bool> {
    let Some(id) = id else { return Ok(value.as_deref() == Some(previous)) };
    let mut stmt = conn.prepare("SELECT DISTINCT coalesce(e.merged_into,e.id) FROM entity_aliases a JOIN entities e ON e.id=a.entity_id WHERE a.alias=?1 LIMIT 2")?;
    let owners: Vec<i64> = stmt.query_map([super::normalize(previous)], |r| r.get(0))?.collect::<Result<_, _>>()?;
    let canonical: i64 =
        conn.query_row("SELECT coalesce(merged_into,id) FROM entities WHERE id=?1", [id], |r| r.get(0))?;
    Ok(owners.as_slice() == [canonical])
}

/// Locate every stored representation of an explicitly named old state.
/// This also serves the opposite-polarity guard without assuming the next
/// assertion's object representation describes the previous assertion.
pub(super) fn target_keys(
    w: &WriteTx<'_>,
    subject: i64,
    predicate: &str,
    previous: &str,
    negated: bool,
) -> Result<Vec<String>> {
    let mut stmt = w.conn().prepare("SELECT DISTINCT dedup_key,predicate,object_id,object_value FROM facts WHERE subject_id=?1 AND negated=?2 AND identity_version=2")?;
    let rows = stmt.query_map(params![subject, negated], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, Option<i64>>(2)?,
            r.get::<_, Option<String>>(3)?,
        ))
    })?;
    let mut keys = Vec::new();
    for row in rows {
        let (key, p, id, value) = row?;
        if super::predicate_key(&p) == super::predicate_key(predicate)
            && matches_object(w.conn(), id, &value, previous)?
        {
            keys.push(key);
        }
    }
    Ok(keys)
}

/// Reconcile by validity order, including facts imported after the transition.
/// Source and transactional history remain separate; no evidence is deleted.
pub(super) fn refresh(w: &WriteTx<'_>, key: &str, observed: i64) -> Result<()> {
    let mut pending = std::collections::VecDeque::from([key.to_owned()]);
    while let Some(key) = pending.pop_front() {
        refresh_key(w, &key, observed, &mut pending)?;
    }
    Ok(())
}

fn refresh_key(
    w: &WriteTx<'_>,
    key: &str,
    observed: i64,
    pending: &mut std::collections::VecDeque<String>,
) -> Result<()> {
    let facts = {
        let mut statement=w.conn().prepare("SELECT id,coalesce(occurred_from,recorded_first),time_kind,subject_id,predicate,object_id,object_value,negated FROM facts WHERE dedup_key=?1 AND identity_version=2 AND status!='retracted'")?;
        statement
            .query_map([key], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<i64>>(5)?,
                    r.get::<_, Option<String>>(6)?,
                    r.get::<_, bool>(7)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
    };
    for (id, start, time_kind, subject, predicate, object_id, object_value, negated) in facts {
        let mut stmt=w.conn().prepare("SELECT u.next_fact_id,u.effective_at,u.kind,coalesce(u.assertion_at,u.effective_at),u.previous_key,u.previous_object,u.observed_at,u.id FROM fact_updates u JOIN facts n ON n.id=u.next_fact_id WHERE n.status!='retracted' AND u.next_fact_id!=?1 AND (u.previous_key=?2 OR (u.subject_id=?3 AND u.predicate_key=?4 AND u.previous_negated=?5)) ORDER BY u.effective_at,u.id")?;
        let rows = stmt.query_map(params![id, key, subject, super::predicate_key(&predicate), negated], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, i64>(7)?,
            ))
        })?;
        let mut correction = None;
        let mut replacement = None;
        for row in rows {
            let (next, effective, kind, asserted, previous_key, previous, observed_at, update_id) = row?;
            // New records use their declared target even if a structural key
            // happens to match; legacy records retain exact-key semantics.
            let matches = match previous {
                Some(previous) => matches_object(w.conn(), object_id, &object_value, &previous)?,
                None => previous_key == key,
            };
            if !matches {
                continue;
            }
            if kind == "correction" && asserted >= start {
                let candidate = (observed_at, update_id, next);
                if correction.is_none_or(|current| candidate < current) {
                    correction = Some(candidate);
                }
            }
            if kind == "replace" && effective >= start && replacement.is_none() {
                replacement = Some((effective, next));
            }
        }
        if let Some((_, _, next)) = correction {
            w.conn().execute("UPDATE facts SET status='retracted',superseded_by=?1,transaction_invalidated_at=coalesce(transaction_invalidated_at,?2) WHERE id=?3",params![next,observed,id])?;
            // A false move cannot keep ending its predecessor. Only newly
            // retracted facts enqueue dependents; retractions are monotonic, so
            // cycles terminate without recursion or a time-based lease.
            let mut stmt=w.conn().prepare("SELECT previous_key,subject_id,predicate_key,previous_object,previous_negated FROM fact_updates WHERE next_fact_id=?1")?;
            let targets = stmt.query_map([id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<i64>>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, Option<bool>>(4)?,
                ))
            })?;
            for target in targets {
                let (key, subject, predicate, previous, negated) = target?;
                pending.push_back(key);
                if let (Some(subject), Some(predicate), Some(previous), Some(negated)) =
                    (subject, predicate, previous, negated)
                {
                    pending.extend(target_keys(w, subject, &predicate, &previous, negated)?);
                }
            }
            continue;
        }
        if time_kind != "ongoing" {
            continue;
        }
        if let Some((until, next)) = replacement {
            w.conn().execute("UPDATE facts SET status='superseded',valid_until=?1,superseded_by=?2,transaction_invalidated_at=coalesce(transaction_invalidated_at,?3) WHERE id=?4",
                params![until,next,observed,id])?;
        } else {
            // Restore only a state ended by a now-retracted successor. Keep
            // manual/legacy invalidations with no explicit dependency intact.
            w.conn().execute("UPDATE facts SET status='active',valid_until=NULL,superseded_by=NULL,transaction_invalidated_at=NULL WHERE id=?1 AND status='superseded' AND superseded_by IN (SELECT id FROM facts WHERE status='retracted')",[id])?;
        }
    }
    Ok(())
}
