use super::*;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::collections::HashSet;

pub(super) fn identifier(s: &str) -> Result<()> {
    if s.is_empty()
        || s.len() > 192
        || !s.as_bytes()[0].is_ascii_alphanumeric()
        || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
    {
        return Err(invalid("invalid BEING identifier"));
    }
    Ok(())
}

pub(super) fn text(s: &str, bytes: usize) -> Result<()> {
    if s.trim().is_empty() || s.len() > bytes {
        return Err(invalid("empty or oversized BEING text"));
    }
    Ok(())
}

fn weight(n: f64) -> Result<()> {
    if !n.is_finite() || !(0.0..=1.0).contains(&n) {
        return Err(invalid("BEING weight must be finite in [0,1]"));
    }
    Ok(())
}

pub(super) fn encode<T: Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(value).map_err(|e| Error::Task(e.to_string()))
}

pub(super) fn decode<T: DeserializeOwned>(s: &str) -> Result<T> {
    serde_json::from_str(s).map_err(|e| Error::Task(format!("invalid stored BEING value: {e}")))
}

pub(super) fn owner(conn: &Connection, agent: &str) -> Result<()> {
    let stored: Option<String> =
        conn.query_row("SELECT agent_id FROM being_owner WHERE singleton=1", [], |r| r.get(0)).optional()?;
    if stored.as_deref() != Some(agent) {
        return Err(invalid("BEING identity is unbound or belongs to another agent"));
    }
    Ok(())
}

pub(super) fn revision(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("SELECT COALESCE(MAX(seq),0) FROM being_events", [], |r| r.get(0))?)
}

pub(super) fn load_observation(conn: &Connection, id: &str) -> Result<StoredObservation> {
    let row: Option<(String, String)> = conn
        .query_row("SELECT body,assurance FROM being_observations WHERE source_id=?1", [id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    let (body, assurance) = row.ok_or_else(|| invalid("unknown BEING source observation"))?;
    Ok(StoredObservation {
        source_id: id.into(),
        observation: decode(&body)?,
        assurance: decode(&assurance)?,
        recorded_revision: None,
    })
}

pub(super) fn load_claim(conn: &Connection, id: &str) -> Result<Claim> {
    let body: Option<String> =
        conn.query_row("SELECT body FROM being_claims WHERE claim_id=?1", [id], |r| r.get(0)).optional()?;
    decode(&body.ok_or_else(|| invalid("unknown BEING claim"))?)
}

pub(super) fn sources(
    conn: &Connection,
    ids: &[String],
    relationship: Option<&str>,
    autobiography: bool,
    subject: &str,
    imported: bool,
) -> Result<()> {
    if ids.len() > 4096 || ids.iter().collect::<HashSet<_>>().len() != ids.len() {
        return Err(invalid("duplicate or excessive BEING sources"));
    }
    for id in ids {
        identifier(id)?;
        let stored = load_observation(conn, id)?;
        let o = stored.observation;
        if o.relationship_id.as_deref().is_some_and(|r| Some(r) != relationship) {
            return Err(invalid("relationship evidence cannot be widened or cross-scoped"));
        }
        if autobiography
            && (o.subject != subject
                || !matches!(o.origin, Origin::DirectInteraction | Origin::ToolResult)
                || (!imported && !matches!(stored.assurance, Assurance::CanonicalTurn | Assurance::HostReceipt)))
        {
            return Err(invalid("autobiography needs this agent's source-bound participation"));
        }
    }
    Ok(())
}

fn observation(conn: &Connection, agent: &str, actor: &Actor, o: &Observation, imported: bool) -> Result<Assurance> {
    identifier(&o.subject)?;
    text(&o.text, 32768)?;
    if let Some(r) = &o.relationship_id {
        identifier(r)?;
    }
    if imported {
        return Ok(Assurance::LegacyImport);
    }
    let allowed = match o.origin {
        Origin::AgentReflection => actor.role == ActorRole::Agent,
        Origin::ModelDefaultReport => matches!(actor.role, ActorRole::Agent | ActorRole::Harness),
        Origin::ImportedDocument => matches!(actor.role, ActorRole::Importer | ActorRole::Agent),
        Origin::HumanReport => matches!(actor.role, ActorRole::Human | ActorRole::Harness),
        Origin::DirectInteraction | Origin::ToolResult => actor.role == ActorRole::Harness,
    };
    if !allowed {
        return Err(invalid("observation origin is not allowed for the transport actor"));
    }
    match &o.source {
        Source::Reported { reference } => {
            if let Some(r) = reference {
                text(r, 2048)?;
            }
            if matches!(o.origin, Origin::DirectInteraction | Origin::ToolResult) {
                return Err(invalid("participation requires a canonical turn or host receipt"));
            }
            Ok(Assurance::Reported)
        }
        Source::HostReceipt { receipt_id, body_blake3 } => {
            identifier(receipt_id)?;
            if actor.role != ActorRole::Harness
                || o.origin != Origin::ToolResult
                || blake3::hash(o.text.as_bytes()).to_hex().as_str() != body_blake3
            {
                return Err(invalid("host receipt must bind the exact observed tool text"));
            }
            Ok(Assurance::HostReceipt)
        }
        Source::NativeTurn { session, turn } => {
            identifier(session)?;
            identifier(turn)?;
            let mut stmt = conn.prepare("SELECT t.text,t.role,t.ts,s.id FROM turns t JOIN sessions s ON s.id=t.session_id WHERE s.ext_id=?1 AND t.ext_id=?2 LIMIT 2")?;
            let rows = stmt
                .query_map(params![session, turn], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Option<i64>>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if rows.len() != 1 {
                return Err(invalid("native source must resolve uniquely within its session"));
            }
            let (body, role, at, session_id) = &rows[0];
            if body != &o.text || at != &o.occurred_at {
                return Err(invalid("native source text/time mismatch"));
            }
            if o.origin == Origin::HumanReport && !matches!(role.as_str(), "user" | "participant") {
                return Err(invalid("assistant text cannot become a human report"));
            }
            if o.origin == Origin::DirectInteraction {
                let bound: bool = conn.query_row(
                    "SELECT EXISTS(SELECT 1 FROM being_session_bindings WHERE session_id=?1)",
                    [session_id],
                    |r| r.get(0),
                )?;
                if !bound || o.subject != agent || !matches!(role.as_str(), "user" | "assistant") {
                    return Err(invalid("this interaction lacks an explicit agent/session participation binding"));
                }
            }
            if !matches!(o.origin, Origin::HumanReport | Origin::DirectInteraction) {
                return Err(invalid("source role is incompatible with observation origin"));
            }
            Ok(Assurance::CanonicalTurn)
        }
    }
}

fn validate(
    conn: &Connection,
    agent: &str,
    actor: &Actor,
    request: &Request,
    imported: bool,
) -> Result<Option<Assurance>> {
    identifier(&actor.principal)?;
    identifier(&actor.session)?;
    if actor.role == ActorRole::Agent && actor.principal != agent {
        return Err(invalid("agent principal cannot author another agent's state"));
    }
    let changes_stance = matches!(
        request.operation,
        Operation::Propose(_)
            | Operation::Decide(_)
            | Operation::DefineRelationship(_)
            | Operation::RegisterCommitment(_)
            | Operation::TransitionCommitment(_)
            | Operation::DefineConcept(_)
            | Operation::DefineHypotheses(_)
            | Operation::LinkPositions(_)
            | Operation::RetireLink { .. }
    );
    if changes_stance && !imported && actor.role != ActorRole::Agent {
        return Err(invalid("the transport actor cannot adopt or propose this agent's stance"));
    }
    if changes_stance && request.expected_revision != Some(revision(conn)?) {
        return Err(invalid("stale BEING revision: reread and reconsider"));
    }
    if !changes_stance && request.expected_revision.is_some() {
        return Err(invalid("revision preconditions belong to stance changes"));
    }
    match &request.operation {
        Operation::Observe(o) => return observation(conn, agent, actor, o, imported).map(Some),
        Operation::Propose(p) => {
            if p.source_ids.len() > 16 {
                return Err(invalid("too many sources in one proposal"));
            }
            identifier(&p.claim_id)?;
            identifier(&p.subject)?;
            text(&p.statement, 4096)?;
            text(&p.reason, 2048)?;
            weight(p.importance)?;
            if let Some(n) = p.credence {
                weight(n)?;
            }
            if let Some(r) = &p.relationship_id {
                identifier(r)?;
            }
            let exists: bool =
                conn.query_row("SELECT EXISTS(SELECT 1 FROM being_claims WHERE claim_id=?1)", [&p.claim_id], |r| {
                    r.get(0)
                })?;
            if exists {
                return Err(invalid("claim ID is immutable; propose changed meaning with a new ID"));
            }
            if matches!(p.category, Category::Autobiographical | Category::SelfDescription) && p.subject != agent {
                return Err(invalid("self records must name the bound agent"));
            }
            if matches!(p.category, Category::Belief | Category::Relationship | Category::Autobiographical)
                && p.source_ids.is_empty()
            {
                return Err(invalid("this claim category requires attributed sources"));
            }
            sources(
                conn,
                &p.source_ids,
                p.relationship_id.as_deref(),
                p.category == Category::Autobiographical,
                &p.subject,
                imported,
            )?;
        }
        Operation::Decide(d) => {
            if d.source_ids.len() > 16 {
                return Err(invalid("too many new sources in one decision"));
            }
            identifier(&d.claim_id)?;
            text(&d.reason, 2048)?;
            if let Some(n) = d.credence {
                weight(n)?;
            }
            let c = load_claim(conn, &d.claim_id)?;
            // Declining/suspending a legacy autobiography must remain possible
            // even when its old participation assertion cannot be certified.
            let needs_participation = c.proposal.category == Category::Autobiographical && d.stance == Stance::Adopted;
            let ids = c
                .proposal
                .source_ids
                .iter()
                .chain(&d.source_ids)
                .cloned()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            // Duplicate new IDs are invalid even when the union would hide them.
            sources(
                conn,
                &d.source_ids,
                c.proposal.relationship_id.as_deref(),
                needs_participation,
                &c.proposal.subject,
                imported,
            )?;
            sources(
                conn,
                &ids,
                c.proposal.relationship_id.as_deref(),
                needs_participation,
                &c.proposal.subject,
                imported,
            )?;
        }
        Operation::DefineRelationship(_) | Operation::RegisterCommitment(_) | Operation::TransitionCommitment(_) => {
            if imported {
                return Err(invalid("legacy import cannot introduce structured obligations"));
            }
            super::relationships::validate(conn, agent, &request.operation)?;
        }
        Operation::BindSession { session } => {
            identifier(session)?;
            if actor.role != ActorRole::Harness || imported {
                return Err(invalid("session participation requires a live harness binding"));
            }
            let valid:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM sessions s JOIN turns t ON t.session_id=s.id WHERE s.ext_id=?1 AND t.role='assistant')",[session],|r|r.get(0))?;
            if !valid {
                return Err(invalid("participation binding requires an actual assistant session"));
            }
            let bound:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM being_session_bindings b JOIN sessions s ON s.id=b.session_id WHERE s.ext_id=?1)",[session],|r|r.get(0))?;
            if bound {
                return Err(invalid("session already bound; replay its original binding request"));
            }
        }
        Operation::DefineConcept(_)
        | Operation::DefineHypotheses(_)
        | Operation::LinkPositions(_)
        | Operation::RetireLink { .. } => {
            if imported {
                return Err(invalid("legacy import cannot introduce ontology operations"));
            }
            super::ontology::validate(conn, agent, &request.operation)?;
        }
        Operation::Boundary { turn, .. } => {
            identifier(turn)?;
            if !imported && actor.role != ActorRole::Harness {
                return Err(invalid("loop boundaries are authored by the harness"));
            }
        }
    }
    Ok(None)
}

pub(super) fn apply(
    conn: &Connection,
    agent: &str,
    actor: &Actor,
    request: &Request,
    imported_from: Option<&str>,
    recorded_at: i64,
) -> Result<Receipt> {
    owner(conn, agent)?;
    identifier(&request.request_id)?;
    super::ontology::finite(&request.operation)?;
    // serde_json renders native NaN/Infinity as null. Reject them before hashing
    // or cache lookup so an invalid native value cannot alias an unknown value.
    match &request.operation {
        Operation::Propose(p) => {
            weight(p.importance)?;
            if let Some(n) = p.credence {
                weight(n)?;
            }
        }
        Operation::Decide(d) => {
            if let Some(n) = d.credence {
                weight(n)?;
            }
        }
        _ => {}
    }
    let request_text = encode(request)?;
    if request_text.len() > 65536 {
        return Err(invalid("BEING request exceeds bounded admission"));
    }
    let actor_text = encode(actor)?;
    let digest = blake3::hash(encode(&(agent, actor, request, imported_from))?.as_bytes()).to_hex().to_string();
    let prior: Option<(i64, String)> = conn
        .query_row("SELECT seq,digest FROM being_events WHERE request_id=?1", [&request.request_id], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    if let Some((revision, stored)) = prior {
        if stored != digest {
            return Err(Error::ApplicationConflict { id: request.request_id.clone() });
        }
        return Ok(Receipt { revision, replayed: true, request_digest: digest });
    }
    let assurance = validate(conn, agent, actor, request, imported_from.is_some())?;
    conn.execute(
        "INSERT INTO being_events(request_id,actor,request,digest,recorded_at,imported_from) VALUES(?1,?2,?3,?4,?5,?6)",
        params![request.request_id, actor_text, request_text, digest, recorded_at, imported_from],
    )?;
    let seq = conn.last_insert_rowid();
    match &request.operation {
        Operation::Observe(o) => {
            conn.execute(
                "INSERT INTO being_observations(source_id,event_seq,body,assurance) VALUES(?1,?2,?3,?4)",
                params![request.request_id, seq, encode(o)?, encode(&assurance.expect("observation assurance"))?],
            )?;
        }
        Operation::Propose(p) => {
            let c = Claim {
                proposal: p.clone(),
                stance: Stance::Considering,
                credence: p.credence,
                reason: p.reason.clone(),
                revision: seq,
            };
            conn.execute(
                "INSERT INTO being_claims(claim_id,body,revision) VALUES(?1,?2,?3)",
                params![p.claim_id, encode(&c)?, seq],
            )?;
            for id in &p.source_ids {
                conn.execute("INSERT INTO being_claim_sources VALUES(?1,?2,?3)", params![p.claim_id, id, seq])?;
            }
        }
        Operation::Decide(d) => {
            let mut c = load_claim(conn, &d.claim_id)?;
            c.stance = d.stance;
            c.credence = d.credence;
            c.reason = d.reason.clone();
            c.revision = seq;
            for id in &d.source_ids {
                if !c.proposal.source_ids.contains(id) {
                    c.proposal.source_ids.push(id.clone());
                }
                conn.execute(
                    "INSERT OR IGNORE INTO being_claim_sources VALUES(?1,?2,?3)",
                    params![d.claim_id, id, seq],
                )?;
            }
            conn.execute(
                "UPDATE being_claims SET body=?1,revision=?2 WHERE claim_id=?3",
                params![encode(&c)?, seq, d.claim_id],
            )?;
        }
        Operation::DefineRelationship(_) | Operation::RegisterCommitment(_) | Operation::TransitionCommitment(_) => {
            super::relationships::project(conn, agent, &request.operation, seq)?;
        }
        Operation::BindSession { session } => {
            conn.execute(
                "INSERT INTO being_session_bindings SELECT id,?1 FROM sessions WHERE ext_id=?2",
                params![seq, session],
            )?;
        }
        Operation::DefineConcept(_)
        | Operation::DefineHypotheses(_)
        | Operation::LinkPositions(_)
        | Operation::RetireLink { .. } => {
            super::ontology::project(conn, &request.operation, seq)?;
        }
        Operation::Boundary { .. } => {}
    }
    Ok(Receipt { revision: seq, replayed: false, request_digest: digest })
}

pub(super) fn snapshot(conn: &Connection, agent: &str, relationship: Option<&str>) -> Result<Snapshot> {
    let mut claims = Vec::new();
    let mut ids = HashSet::new();
    let mut stmt = conn.prepare("SELECT body FROM being_claims ORDER BY claim_id")?;
    for row in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let c: Claim = decode(&row?)?;
        if c.proposal.relationship_id.as_deref().is_some_and(|r| Some(r) != relationship) {
            continue;
        }
        ids.extend(c.proposal.source_ids.iter().cloned());
        claims.push(c);
    }
    let (relationships, commitments) = super::relationships::snapshot(conn, relationship, &mut ids)?;
    let mut ids = ids.into_iter().collect::<Vec<_>>();
    ids.sort();
    let observations = ids.iter().map(|id| load_observation(conn, id)).collect::<Result<Vec<_>>>()?;
    let mut out = Snapshot {
        agent_id: agent.into(),
        revision: revision(conn)?,
        observations,
        claims,
        relationships,
        commitments,
        concepts: vec![],
        hypotheses: vec![],
        links: vec![],
    };
    super::ontology::snapshot(conn, relationship, &mut out)?;
    Ok(out)
}
