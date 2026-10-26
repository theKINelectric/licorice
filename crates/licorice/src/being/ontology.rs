//! Versioned meaning and bounded review, without automatic belief propagation.
use super::kernel::*;
use super::*;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

fn versioned<T: serde::de::DeserializeOwned>(
    conn: &Connection,
    table: &str,
    key: &str,
    id: &str,
    version: u32,
) -> Result<T> {
    identifier(id)?;
    let body: Option<String> = conn
        .query_row(&format!("SELECT body FROM {table} WHERE {key}=?1 AND version=?2"), params![id, version], |r| {
            r.get(0)
        })
        .optional()?;
    decode(&body.ok_or_else(|| invalid("unknown versioned position"))?)
}
fn latest(conn: &Connection, table: &str, key: &str, id: &str) -> Result<u32> {
    Ok(conn.query_row(&format!("SELECT COALESCE(MAX(version),0) FROM {table} WHERE {key}=?1"), [id], |r| r.get(0))?)
}
fn link(conn: &Connection, id: &str) -> Result<PositionLink> {
    identifier(id)?;
    let body: Option<String> =
        conn.query_row("SELECT body FROM being_links WHERE link_id=?1", [id], |r| r.get(0)).optional()?;
    decode(&body.ok_or_else(|| invalid("unknown position link"))?)
}
fn node(conn: &Connection, p: &PositionRef) -> Result<(Option<String>, i64)> {
    match p {
        PositionRef::Claim { claim_id } => {
            identifier(claim_id)?;
            let c = load_claim(conn, claim_id)?;
            Ok((c.proposal.relationship_id, c.revision))
        }
        PositionRef::Concept { concept_id, version } => {
            let c: Concept = versioned(conn, "being_concepts", "concept_id", concept_id, *version)?;
            Ok((c.definition.relationship_id, c.revision))
        }
        PositionRef::Hypotheses { set_id, version } => {
            let h: HypothesisSet = versioned(conn, "being_hypotheses", "set_id", set_id, *version)?;
            Ok((h.definition.relationship_id, h.revision))
        }
    }
}
fn in_scope(scope: Option<&str>, target: Option<&str>) -> Result<()> {
    if target.is_some() && target != scope {
        return Err(invalid("position dependency cannot widen or cross relationship scope"));
    }
    Ok(())
}
fn common(conn: &Connection, agent: &str, scope: Option<&str>, ids: &[String], reason: &str) -> Result<()> {
    if let Some(s) = scope {
        identifier(s)?;
    }
    if ids.len() > 16 {
        return Err(invalid("too many position sources"));
    }
    text(reason, 2048)?;
    sources(conn, ids, scope, false, agent, false)
}
/// Reject native nonfinite values before JSON hashing can alias them with null.
pub(super) fn finite(op: &Operation) -> Result<()> {
    if let Operation::DefineHypotheses(h) = op {
        for w in h.alternatives.iter().filter_map(|a| a.weight).chain(h.unknown_weight) {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                return Err(invalid("hypothesis weights must be finite in [0,1]"));
            }
        }
    }
    Ok(())
}
pub(super) fn validate(conn: &Connection, agent: &str, op: &Operation) -> Result<()> {
    finite(op)?;
    match op {
        Operation::DefineConcept(c) => {
            identifier(&c.concept_id)?;
            text(&c.label, 256)?;
            text(&c.definition, 4096)?;
            common(conn, agent, c.relationship_id.as_deref(), &c.source_ids, &c.reason)?;
            let last = latest(conn, "being_concepts", "concept_id", &c.concept_id)?;
            if c.version != last.checked_add(1).ok_or_else(|| invalid("concept version exhausted"))?
                || c.supersedes.len() > 8
                || c.supersedes.iter().collect::<HashSet<_>>().len() != c.supersedes.len()
            {
                return Err(invalid("concept versions advance once and have bounded unique lineage"));
            }
            if last > 0
                && !c.supersedes.contains(&PositionRef::Concept { concept_id: c.concept_id.clone(), version: last })
            {
                return Err(invalid("redefinition must retain the previous meaning in its lineage"));
            }
            for p in &c.supersedes {
                if !matches!(p, PositionRef::Concept { .. }) {
                    return Err(invalid("concept lineage references only concept meanings"));
                }
                let (scope, _) = node(conn, p)?;
                in_scope(c.relationship_id.as_deref(), scope.as_deref())?;
            }
        }
        Operation::DefineHypotheses(h) => {
            identifier(&h.set_id)?;
            text(&h.label, 256)?;
            common(conn, agent, h.relationship_id.as_deref(), &h.source_ids, &h.reason)?;
            if h.version
                != latest(conn, "being_hypotheses", "set_id", &h.set_id)?
                    .checked_add(1)
                    .ok_or_else(|| invalid("hypothesis version exhausted"))?
                || !(2..=16).contains(&h.alternatives.len())
                || h.alternatives.iter().map(|a| &a.claim_id).collect::<HashSet<_>>().len() != h.alternatives.len()
            {
                return Err(invalid("hypothesis sets need a new version and distinct bounded alternatives"));
            }
            if h.version > 1 {
                let old: HypothesisSet = versioned(conn, "being_hypotheses", "set_id", &h.set_id, h.version - 1)?;
                if old.definition.relationship_id != h.relationship_id || old.definition.mode != h.mode {
                    return Err(invalid("changed hypothesis scope/mode needs a new set identity"));
                }
            }
            for a in &h.alternatives {
                identifier(&a.claim_id)?;
                let c = load_claim(conn, &a.claim_id)?;
                if c.proposal.category != Category::Belief {
                    return Err(invalid("empirical hypothesis weights cannot normalize values or promises"));
                }
                in_scope(h.relationship_id.as_deref(), c.proposal.relationship_id.as_deref())?;
            }
            match h.mode {
                HypothesisMode::Exclusive => {
                    if h.unknown_weight.is_none() || h.alternatives.iter().any(|a| a.weight.is_none()) {
                        return Err(invalid("exclusive sets require explicit weights including unknown alternatives"));
                    }
                    let sum = h.alternatives.iter().filter_map(|a| a.weight).sum::<f64>() + h.unknown_weight.unwrap();
                    if (sum - 1.0).abs() > 1e-9 {
                        return Err(invalid("exclusive weights and unknown mass must sum to1"));
                    }
                }
                HypothesisMode::Compatible => {
                    if h.unknown_weight.is_some() {
                        return Err(invalid(
                            "compatible propositions have independent weights, no shared unknown mass",
                        ));
                    }
                }
            }
        }
        Operation::LinkPositions(l) => {
            identifier(&l.link_id)?;
            common(conn, agent, l.relationship_id.as_deref(), &l.source_ids, &l.reason)?;
            if l.from == l.to {
                return Err(invalid("position cannot link to itself"));
            }
            for p in [&l.from, &l.to] {
                let (scope, _) = node(conn, p)?;
                in_scope(l.relationship_id.as_deref(), scope.as_deref())?;
            }
            let exists: bool =
                conn.query_row("SELECT EXISTS(SELECT 1 FROM being_links WHERE link_id=?1)", [&l.link_id], |r| {
                    r.get(0)
                })?;
            if exists {
                return Err(invalid("link identity is immutable"));
            }
        }
        Operation::RetireLink { link_id, reason } => {
            text(reason, 2048)?;
            if link(conn, link_id)?.retired_at.is_some() {
                return Err(invalid("link already retired"));
            }
        }
        _ => return Err(invalid("not an ontology operation")),
    }
    Ok(())
}
pub(super) fn project(conn: &Connection, op: &Operation, seq: i64) -> Result<()> {
    match op {
        Operation::DefineConcept(c) => {
            let c = Concept { definition: c.clone(), revision: seq };
            conn.execute(
                "INSERT INTO being_concepts VALUES(?1,?2,?3,?4)",
                params![c.definition.concept_id, c.definition.version, encode(&c)?, seq],
            )?;
        }
        Operation::DefineHypotheses(h) => {
            let h = HypothesisSet {
                definition: h.clone(),
                revision: seq,
                calibration: "uncalibrated_explicit_estimate".into(),
            };
            conn.execute(
                "INSERT INTO being_hypotheses VALUES(?1,?2,?3,?4)",
                params![h.definition.set_id, h.definition.version, encode(&h)?, seq],
            )?;
        }
        Operation::LinkPositions(l) => {
            let l = PositionLink {
                definition: l.clone(),
                revision: seq,
                from_revision: node(conn, &l.from)?.1,
                to_revision: node(conn, &l.to)?.1,
                retired_at: None,
                retirement_reason: None,
            };
            conn.execute("INSERT INTO being_links VALUES(?1,?2,?3)", params![l.definition.link_id, encode(&l)?, seq])?;
        }
        Operation::RetireLink { link_id, reason } => {
            let mut l = link(conn, link_id)?;
            l.retired_at = Some(seq);
            l.retirement_reason = Some(reason.clone());
            conn.execute("UPDATE being_links SET body=?1 WHERE link_id=?2", params![encode(&l)?, link_id])?;
        }
        _ => return Err(invalid("not an ontology operation")),
    }
    Ok(())
}
pub(super) fn snapshot(conn: &Connection, scope: Option<&str>, out: &mut Snapshot) -> Result<()> {
    let mut ids = HashSet::new();
    let mut stmt = conn.prepare("SELECT body FROM being_concepts ORDER BY concept_id,version")?;
    for body in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let c: Concept = decode(&body?)?;
        if in_scope(scope, c.definition.relationship_id.as_deref()).is_ok() {
            ids.extend(c.definition.source_ids.iter().cloned());
            out.concepts.push(c);
        }
    }
    let mut stmt = conn.prepare("SELECT body FROM being_hypotheses ORDER BY set_id,version")?;
    for body in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let h: HypothesisSet = decode(&body?)?;
        if in_scope(scope, h.definition.relationship_id.as_deref()).is_ok() {
            ids.extend(h.definition.source_ids.iter().cloned());
            out.hypotheses.push(h);
        }
    }
    let mut stmt = conn.prepare("SELECT body FROM being_links ORDER BY link_id")?;
    for body in stmt.query_map([], |r| r.get::<_, String>(0))? {
        let l: PositionLink = decode(&body?)?;
        if in_scope(scope, l.definition.relationship_id.as_deref()).is_ok() {
            ids.extend(l.definition.source_ids.iter().cloned());
            out.links.push(l);
        }
    }
    for id in ids {
        if !out.observations.iter().any(|o| o.source_id == id) {
            out.observations.push(load_observation(conn, &id)?);
        }
    }
    out.observations.sort_by(|a, b| a.source_id.cmp(&b.source_id));
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewLimits {
    pub nodes: usize,
    pub edges: usize,
    pub depth: usize,
}
impl Default for ReviewLimits {
    fn default() -> Self {
        Self { nodes: 32, edges: 128, depth: 4 }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImpactReview {
    pub roots: Vec<PositionRef>,
    pub affected: Vec<PositionRef>,
    pub link_ids: Vec<String>,
    pub inspected_edges: usize,
    pub saturated: bool,
    pub conclusion: String,
}
fn present(s: &Snapshot, p: &PositionRef) -> bool {
    match p {
        PositionRef::Claim { claim_id } => s.claims.iter().any(|c| &c.proposal.claim_id == claim_id),
        PositionRef::Concept { concept_id, version } => {
            s.concepts.iter().any(|c| &c.definition.concept_id == concept_id && &c.definition.version == version)
        }
        PositionRef::Hypotheses { set_id, version } => {
            s.hypotheses.iter().any(|h| &h.definition.set_id == set_id && &h.definition.version == version)
        }
    }
}
/// Incoming dependencies and outgoing evidence/objection links trigger review.
/// Traversal never changes stance; saturation means further candidates may exist.
pub fn impact(s: &Snapshot, roots: &[PositionRef], limits: ReviewLimits) -> Result<ImpactReview> {
    if !(1..=64).contains(&limits.nodes)
        || !(1..=256).contains(&limits.edges)
        || !(1..=8).contains(&limits.depth)
        || roots.is_empty()
        || roots.len() > 8
        || roots.len() > limits.nodes
        || roots.iter().any(|r| !present(s, r))
    {
        return Err(invalid("impact review needs visible roots and bounded node/edge/depth limits"));
    }
    let mut seen = roots.iter().cloned().collect::<HashSet<_>>();
    let mut queue = roots.iter().cloned().map(|r| (r, 0)).collect::<VecDeque<_>>();
    let mut out = ImpactReview {
        roots: roots.to_vec(),
        affected: vec![],
        link_ids: vec![],
        inspected_edges: 0,
        saturated: false,
        conclusion: "Candidates for deliberate review; no automatic stance or probability update.".into(),
    };
    let mut examined = HashSet::new();
    while let Some((p, depth)) = queue.pop_front() {
        for l in &s.links {
            if l.retired_at.is_some() {
                continue;
            }
            let d = &l.definition;
            let next = if d.to == p {
                Some(&d.from)
            } else if d.from == p
                && matches!(
                    d.relation,
                    PositionRelation::Supports | PositionRelation::ObjectsTo | PositionRelation::Contradicts
                )
            {
                Some(&d.to)
            } else {
                None
            };
            let Some(next) = next else { continue };
            if examined.contains(&d.link_id) {
                continue;
            }
            if out.inspected_edges == limits.edges {
                out.saturated = true;
                continue;
            }
            examined.insert(d.link_id.clone());
            out.inspected_edges += 1;
            if seen.contains(next) {
                continue;
            }
            if depth == limits.depth || seen.len() == limits.nodes {
                out.saturated = true;
                continue;
            }
            seen.insert(next.clone());
            out.affected.push(next.clone());
            out.link_ids.push(d.link_id.clone());
            queue.push_back((next.clone(), depth + 1));
        }
    }
    Ok(out)
}

/// Repeated reflections never increase weight. Supersession schedules review
/// without deleting old meaning, and changed claims preserve their decision log.
pub fn changed_dependencies(s: &Snapshot) -> Vec<PositionRef> {
    let mut roots = Vec::new();
    for l in &s.links {
        if l.retired_at.is_some() {
            continue;
        }
        let p = &l.definition.to;
        let changed = match p {
            PositionRef::Claim { claim_id } => {
                s.claims.iter().any(|c| &c.proposal.claim_id == claim_id && c.revision > l.to_revision)
            }
            PositionRef::Concept { .. } => s.concepts.iter().any(|c| c.definition.supersedes.contains(p)),
            PositionRef::Hypotheses { set_id, version } => {
                s.hypotheses.iter().any(|h| &h.definition.set_id == set_id && h.definition.version > *version)
            }
        };
        if changed && !roots.contains(p) {
            roots.push(p.clone());
        }
    }
    roots.sort();
    roots
}

/// Whole dependency closure for one context unit. None means the bounded unit
/// is too large, so the caller omits it rather than clipping a dependency.
pub(crate) fn bundle(s: &Snapshot, root: PositionRef) -> Result<Option<(serde_json::Value, Vec<String>)>> {
    use serde_json::json;
    let mut queue = VecDeque::from([root]);
    let mut seen = HashSet::new();
    let mut positions = Vec::new();
    let mut links = Vec::new();
    let mut ids = Vec::new();
    while let Some(p) = queue.pop_front() {
        if !seen.insert(p.clone()) {
            continue;
        }
        if seen.len() > 32 {
            return Ok(None);
        }
        let mut add_sources = |sources: &[String]| {
            for id in sources {
                if !ids.contains(id) {
                    ids.push(id.clone());
                }
            }
        };
        match &p {
            PositionRef::Claim { claim_id } => {
                let c = s
                    .claims
                    .iter()
                    .find(|c| &c.proposal.claim_id == claim_id)
                    .ok_or_else(|| invalid("missing claim in context closure"))?;
                add_sources(&c.proposal.source_ids);
                positions.push(json!({"position":p,"claim":c}));
                if let Some(c) = s.commitments.iter().find(|c| &c.definition.claim_id == claim_id) {
                    add_sources(&c.source_ids);
                    positions.push(json!({"commitment":c}));
                }
            }
            PositionRef::Concept { concept_id, version } => {
                let c = s
                    .concepts
                    .iter()
                    .find(|c| &c.definition.concept_id == concept_id && &c.definition.version == version)
                    .ok_or_else(|| invalid("missing concept in context closure"))?;
                add_sources(&c.definition.source_ids);
                positions.push(json!({"position":p,"concept":c}));
                queue.extend(c.definition.supersedes.iter().cloned());
                for successor in &s.concepts {
                    if successor.definition.supersedes.contains(&p) {
                        queue.push_back(PositionRef::Concept {
                            concept_id: successor.definition.concept_id.clone(),
                            version: successor.definition.version,
                        });
                    }
                }
            }
            PositionRef::Hypotheses { set_id, version } => {
                let h = s
                    .hypotheses
                    .iter()
                    .find(|h| &h.definition.set_id == set_id && &h.definition.version == version)
                    .ok_or_else(|| invalid("missing hypothesis set in context closure"))?;
                add_sources(&h.definition.source_ids);
                positions.push(json!({"position":p,"hypotheses":h}));
                queue.extend(
                    h.definition.alternatives.iter().map(|a| PositionRef::Claim { claim_id: a.claim_id.clone() }),
                );
            }
        }
        for l in &s.links {
            if l.retired_at.is_none() && l.definition.from == p {
                add_sources(&l.definition.source_ids);
                links.push(l.clone());
                queue.push_back(l.definition.to.clone());
            }
        }
    }
    let changed = changed_dependencies(s).into_iter().filter(|p| seen.contains(p)).collect::<Vec<_>>();
    Ok(Some((json!({"positions":positions,"links":links,"changed_dependencies_require_review":changed}), ids)))
}
