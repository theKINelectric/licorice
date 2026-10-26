//! Explicit relationships and obligation lifecycle. These are the agent's
//! attributed judgments; admission validates source identity, not fulfillment.
use super::kernel::*;
use super::*;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::HashSet;

fn relationship(conn: &Connection, id: &str) -> Result<Option<Relationship>> {
    let body: Option<String> = conn
        .query_row("SELECT body FROM being_relationships WHERE relationship_id=?1", [id], |r| r.get(0))
        .optional()?;
    body.map(|b| decode(&b)).transpose()
}
fn commitment(conn: &Connection, id: &str) -> Result<Commitment> {
    let body: Option<String> =
        conn.query_row("SELECT body FROM being_commitments WHERE claim_id=?1", [id], |r| r.get(0)).optional()?;
    decode(&body.ok_or_else(|| invalid("unknown structured commitment"))?)
}
fn counterparty_report(conn: &Connection, id: &str, counterparty: &str) -> Result<()> {
    let o = load_observation(conn, id)?;
    if o.observation.origin != Origin::HumanReport || o.observation.subject != counterparty {
        return Err(invalid("acknowledgement/release requires an attributed counterparty report"));
    }
    Ok(())
}

pub(super) fn validate(conn: &Connection, agent: &str, operation: &Operation) -> Result<()> {
    match operation {
        Operation::DefineRelationship(d) => {
            identifier(&d.relationship_id)?;
            text(&d.description, 2048)?;
            text(&d.reason, 2048)?;
            if !(2..=16).contains(&d.members.len())
                || d.members.iter().collect::<HashSet<_>>().len() != d.members.len()
                || !d.members.iter().any(|m| m == agent)
                || d.boundaries.len() > 8
                || d.source_ids.is_empty()
                || d.source_ids.len() > 16
            {
                return Err(invalid("relationship needs owner, distinct members, bounded boundaries and sources"));
            }
            for m in &d.members {
                identifier(m)?;
            }
            for b in &d.boundaries {
                text(b, 1024)?;
            }
            let prior = relationship(conn, &d.relationship_id)?;
            if let Some(old) = prior {
                let old_members = old.definition.members.iter().collect::<HashSet<_>>();
                if d.version != old.definition.version.saturating_add(1)
                    || d.members.iter().collect::<HashSet<_>>() != old_members
                {
                    return Err(invalid("relationship revision must preserve members and advance one version"));
                }
            } else if d.version != 1 {
                return Err(invalid("new relationship starts at version1"));
            }
            sources(conn, &d.source_ids, Some(&d.relationship_id), false, agent, false)?;
        }
        Operation::RegisterCommitment(d) => {
            identifier(&d.claim_id)?;
            identifier(&d.counterparty)?;
            text(&d.next_action, 2048)?;
            let c = load_claim(conn, &d.claim_id)?;
            if c.proposal.subject != agent || c.proposal.category != Category::Commitment || c.stance != Stance::Adopted
            {
                return Err(invalid("structured commitment must bind an adopted owner commitment claim"));
            }
            let scope = c
                .proposal
                .relationship_id
                .as_deref()
                .ok_or_else(|| invalid("commitment needs a relationship scope"))?;
            let r =
                relationship(conn, scope)?.ok_or_else(|| invalid("define the relationship before its commitments"))?;
            if d.counterparty == agent || !r.definition.members.contains(&d.counterparty) {
                return Err(invalid("commitment counterparty must be another relationship member"));
            }
            let exists: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM being_commitments WHERE claim_id=?1)",
                [&d.claim_id],
                |r| r.get(0),
            )?;
            if exists {
                return Err(invalid("commitment definition is immutable; transition its lifecycle"));
            }
            if let Some(id) = &d.acknowledgement_source {
                sources(conn, std::slice::from_ref(id), Some(scope), false, agent, false)?;
                counterparty_report(conn, id, &d.counterparty)?;
            }
        }
        Operation::TransitionCommitment(d) => {
            text(&d.reason, 2048)?;
            let old = commitment(conn, &d.claim_id)?;
            let c = load_claim(conn, &d.claim_id)?;
            if old.status == d.status
                || (old.status != CommitmentStatus::Active && d.status != CommitmentStatus::Active)
            {
                return Err(invalid("commitment transition must close an active promise or explicitly reopen it"));
            }
            if d.status == CommitmentStatus::Active && c.stance != Stance::Adopted {
                return Err(invalid("reopening requires current adoption of the commitment"));
            }
            if d.source_ids.is_empty() || d.source_ids.len() > 16 {
                return Err(invalid("commitment transitions need bounded outcome sources"));
            }
            sources(conn, &d.source_ids, c.proposal.relationship_id.as_deref(), false, agent, false)?;
            if d.status == CommitmentStatus::Released {
                let supported =
                    d.source_ids.iter().any(|id| counterparty_report(conn, id, &old.definition.counterparty).is_ok());
                if !supported {
                    return Err(invalid("release needs a counterparty report, not unilateral erasure"));
                }
            }
            if d.status == CommitmentStatus::Fulfilled {
                let supported =
                    d.source_ids.iter().map(|id| load_observation(conn, id)).collect::<Result<Vec<_>>>()?.iter().any(
                        |o| {
                            (o.observation.origin == Origin::ToolResult && o.assurance == Assurance::HostReceipt)
                                || (o.observation.origin == Origin::DirectInteraction
                                    && o.assurance == Assurance::CanonicalTurn)
                                || (o.observation.origin == Origin::HumanReport
                                    && o.observation.subject == old.definition.counterparty)
                        },
                    );
                if !supported {
                    return Err(invalid("fulfillment needs an outcome witness beyond the agent's own reflection"));
                }
            }
        }
        _ => return Err(invalid("not a relationship operation")),
    }
    Ok(())
}

pub(super) fn project(conn: &Connection, agent: &str, op: &Operation, seq: i64) -> Result<()> {
    match op {
        Operation::DefineRelationship(d) => {
            let r = Relationship { definition: d.clone(), revision: seq, perspective: agent.into() };
            conn.execute("INSERT INTO being_relationships VALUES(?1,?2,?3) ON CONFLICT(relationship_id) DO UPDATE SET body=excluded.body,revision=excluded.revision",params![d.relationship_id,encode(&r)?,seq])?;
        }
        Operation::RegisterCommitment(d) => {
            let c = Commitment {
                definition: d.clone(),
                status: CommitmentStatus::Active,
                reason: "Explicitly registered after adoption.".into(),
                source_ids: d.acknowledgement_source.iter().cloned().collect(),
                revision: seq,
            };
            conn.execute("INSERT INTO being_commitments VALUES(?1,?2,?3)", params![d.claim_id, encode(&c)?, seq])?;
        }
        Operation::TransitionCommitment(d) => {
            let mut c = commitment(conn, &d.claim_id)?;
            c.status = d.status;
            c.reason = d.reason.clone();
            c.revision = seq;
            for id in &d.source_ids {
                if !c.source_ids.contains(id) {
                    c.source_ids.push(id.clone());
                }
            }
            conn.execute(
                "UPDATE being_commitments SET body=?1,revision=?2 WHERE claim_id=?3",
                params![encode(&c)?, seq, d.claim_id],
            )?;
        }
        _ => return Err(invalid("not a relationship operation")),
    }
    Ok(())
}

pub(super) fn snapshot(
    conn: &Connection,
    scope: Option<&str>,
    ids: &mut HashSet<String>,
) -> Result<(Vec<Relationship>, Vec<Commitment>)> {
    let mut relationships = Vec::new();
    let mut commitments = Vec::new();
    if let Some(id) = scope {
        if let Some(r) = relationship(conn, id)? {
            ids.extend(r.definition.source_ids.iter().cloned());
            relationships.push(r);
        }
        let mut stmt = conn.prepare(
            "SELECT c.body FROM being_commitments c JOIN being_claims p ON p.claim_id=c.claim_id ORDER BY c.claim_id",
        )?;
        for body in stmt.query_map([], |r| r.get::<_, String>(0))? {
            let c: Commitment = decode(&body?)?;
            if load_claim(conn, &c.definition.claim_id)?.proposal.relationship_id.as_deref() == Some(id) {
                ids.extend(c.source_ids.iter().cloned());
                commitments.push(c);
            }
        }
    }
    Ok((relationships, commitments))
}
