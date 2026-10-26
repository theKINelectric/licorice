//! Opt-in per-turn selection over canonical, dated source. No model or gold labels.
//! Sidecar provenance/coverage is diagnostic; only `context` is model evidence.
use super::{Retrieved, RetrievedChunk, complementary::EvidenceContext, packing::ContextPack};
use crate::{Error, Result, store::Store, temporal::Day};
use licorice_providers::tokens::TokenCounter;
use rusqlite::params;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
type CurrentAssertions = BTreeMap<(i64, String), Vec<(i64, String, bool)>>;
const SEEDS: usize = 96;
const SOURCE_LIMIT: usize = 512;
const UNIT_LIMIT: usize = 32;
const STOP: &str = "a an and are as at be been by did do does for from has have he her him his how i in is it its me my of on or our she that the their them they this to was were what when where which who why with would you your many much few";
fn words(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_lowercase).collect()
}
#[derive(Debug, Serialize)]
pub struct SelectionReport {
    pub policy: &'static str,
    pub status: &'static str,
    pub search_scope: &'static str,
    pub semantic_completeness: &'static str,
    pub raw_claim_conflicts: &'static str,
    pub candidate_limit_reached: bool,
    pub unit_limit_reached: bool,
    pub inspected_turns: usize,
    pub protected_primary_turn_ids: Vec<String>,
    pub unresolved_dependencies: Vec<String>,
    pub ambiguous_dependencies: Vec<String>,
    pub budget_omitted_turn_ids: Vec<String>,
    pub competing_claims: Vec<CompetingClaims>,
    pub selected_sources: Vec<SourceWitness>,
}
#[derive(Debug, Serialize)]
pub struct CompetingClaims {
    pub subject_id: i64,
    pub predicate: String,
    pub fact_ids: Vec<i64>,
    pub reason: &'static str,
}
#[derive(Debug, Serialize)]
pub struct SourceWitness {
    pub source_id: String,
    pub session: String,
    pub speaker: String,
    pub role: String,
    pub said_at: i64,
    pub body_blake3: String,
    pub verification: &'static str,
}
#[derive(Clone)]
struct Turn {
    id: String,
    session: String,
    speaker: String,
    role: String,
    text: String,
    seq: i64,
    said_at: i64,
    day: Day,
    chunk: i64,
    primary: bool,
}
impl Turn {
    fn chunk(&self) -> RetrievedChunk {
        RetrievedChunk {
            id: self.chunk,
            session: self.session.clone(),
            day: self.day,
            turns: vec![(self.id.clone(), self.speaker.clone(), self.text.clone())],
        }
    }
}
#[derive(Clone)]
struct Unit {
    ids: Vec<String>,
    primary: bool,
    score: f64,
    topics: BTreeSet<String>,
}
fn book_reference(t: &str) -> bool {
    let t = t.to_lowercase();
    (t.contains("recommended") || t.contains("suggested"))
        && ["the book", "the novel", "it you", "one you"].iter().any(|s| t.contains(s))
}
fn short_reference(t: &str) -> bool {
    let t = t.trim().to_lowercase();
    t.split_whitespace().count() < 32
        && (t == "yes"
            || t == "no"
            || ["yes, that", "no, that", "it ", "that one", "those ", "its title", "they are"]
                .iter()
                .any(|s| t.starts_with(s)))
}
pub async fn recover(
    store: &Store,
    primary: &Retrieved,
    question: &str,
    at: i64,
    counter: &dyn TokenCounter,
    budget: usize,
) -> Result<(EvidenceContext, SelectionReport)> {
    recover_with_prefix(store, primary, question, at, counter, RecoveryBudget { tokens: budget, prefix: "" }).await
}

/// Continuity and evidence share this allowance. The prefix is admitted before
/// primary evidence; all scope notices and citations count inside `tokens`.
pub struct RecoveryBudget<'a> {
    pub tokens: usize,
    pub prefix: &'a str,
}

pub async fn recover_with_prefix(
    store: &Store,
    primary: &Retrieved,
    question: &str,
    at: i64,
    counter: &dyn TokenCounter,
    allocation: RecoveryBudget<'_>,
) -> Result<(EvidenceContext, SelectionReport)> {
    let budget = allocation.tokens;
    if question.trim().is_empty()
        || question.chars().count() > 2000
        || !(64..=2000).contains(&budget)
        || primary.facts.len() > 64
        || primary.chunks.len() > 128
    {
        return Err(Error::InvalidInput(
            "source-aware retrieval requires bounded primary input, question and64..2000tokens".into(),
        ));
    }
    let primary_ids: HashSet<String> = primary
        .chunks
        .iter()
        .flat_map(|c| c.turns.iter().map(|t| t.0.clone()))
        .chain(primary.facts.iter().flat_map(|f| f.evidence_turns.clone()))
        .collect();
    if primary_ids.len() > 224 {
        return Err(Error::InvalidInput("primary source reference limit224".into()));
    }
    let query = question.to_owned();
    let pids = primary_ids.clone();
    let pacts = primary.facts.clone();
    // All candidate and primary-source validation uses a single SQLite snapshot.
    let (turns, saturated, canonical_facts, attributes, anchors) = store.read(move |c| {
        let tx=c.unchecked_transaction()?;
        let normalized=format!(" {} ",words(&query).into_iter().collect::<Vec<_>>().join(" "));
        let mut st=tx.prepare("SELECT DISTINCT a.alias FROM entity_aliases a JOIN entities e ON e.id=a.entity_id WHERE e.kind='person' AND e.merged_into IS NULL AND instr(?1,' '||lower(a.alias)||' ')>0 ORDER BY a.alias LIMIT 16")?;
        let anchors=st.query_map([normalized],|r|r.get::<_,String>(0))?.collect::<std::result::Result<Vec<_>,_>>()?;
        let terms=super::cross_session::topic_terms(&query,&anchors);
        let expr=terms.iter().map(|s|format!("\"{}\"",s.replace('"',"\"\""))).collect::<Vec<_>>().join(" OR ");
        let mut out:BTreeMap<String,Turn>=BTreeMap::new();let mut ambiguous=HashSet::new();
        let mut admit=|t:Turn| {
            if ambiguous.contains(&t.id) {return;}
            if let Some(old)=out.get(&t.id) {
                if old.session!=t.session || old.seq!=t.seq {out.remove(&t.id);ambiguous.insert(t.id);}
            } else { out.insert(t.id.clone(),t); }
        };
        let mut seed_ids=Vec::new();
        if !expr.is_empty() {
            let mut st=tx.prepare("SELECT k.id FROM fts_chunk JOIN chunks k ON k.id=fts_chunk.rowid JOIN sessions s ON s.id=k.session_id JOIN turns t ON t.id=k.first_turn WHERE fts_chunk MATCH ?1 AND s.started_at<=?2 AND coalesce(t.ts,s.started_at)<=?2 ORDER BY bm25(fts_chunk),k.id LIMIT 97")?;
            seed_ids=st.query_map(params![expr,at],|r|r.get::<_,i64>(0))?.collect::<std::result::Result<Vec<_>,_>>()?;
        }
        let saturated=seed_ids.len()>SEEDS;seed_ids.truncate(SEEDS);
        for cid in seed_ids {
            let (sid,seq):(i64,i64)=tx.query_row("SELECT t.session_id,t.seq FROM chunks k JOIN turns t ON t.id=k.first_turn WHERE k.id=?1",[cid],|r|Ok((r.get(0)?,r.get(1)?)))?;
            let mut st=tx.prepare("SELECT coalesce(t.ext_id,cast(t.id AS TEXT)),s.ext_id,t.speaker,t.role,t.text,t.seq,coalesce(t.ts,s.started_at),s.started_at FROM turns t JOIN sessions s ON s.id=t.session_id WHERE t.session_id=?1 AND t.seq BETWEEN ?2 AND ?3 AND s.started_at<=?4 AND coalesce(t.ts,s.started_at)<=?4 ORDER BY t.seq")?;
            let rows=st.query_map(params![sid,seq.saturating_sub(1),seq+1,at],|r|Ok(Turn{id:r.get(0)?,session:r.get(1)?,speaker:r.get(2)?,role:r.get(3)?,text:r.get(4)?,seq:r.get(5)?,said_at:r.get(6)?,day:Day::from_ms(r.get(7)?),chunk:cid,primary:false}))?;
            for t in rows {let mut t=t?;t.primary=pids.contains(&t.id);admit(t);}
        }
        let mut ordered:Vec<_>=pids.iter().collect();ordered.sort();
        for id in ordered {
            let mut st=tx.prepare("SELECT coalesce(t.ext_id,cast(t.id AS TEXT)),s.ext_id,t.speaker,t.role,t.text,t.seq,coalesce(t.ts,s.started_at),s.started_at FROM turns t JOIN sessions s ON s.id=t.session_id WHERE t.ext_id=?1 AND s.started_at<=?2 AND coalesce(t.ts,s.started_at)<=?2 ORDER BY t.id LIMIT 2")?;
            let rows=st.query_map(params![id,at],|r|Ok(Turn{id:r.get(0)?,session:r.get(1)?,speaker:r.get(2)?,role:r.get(3)?,text:r.get(4)?,seq:r.get(5)?,said_at:r.get(6)?,day:Day::from_ms(r.get(7)?),chunk:0,primary:true}))?.collect::<std::result::Result<Vec<_>,_>>()?;
            if rows.len()==1 {admit(rows[0].clone());}
        }
        // Source IDs must be unique across eligible canonical rows, including
        // duplicates outside the lexical pool. Never certify one arbitrary row.
        let mut duplicates=Vec::new();
        for id in out.keys() {
            let count:i64=tx.query_row("SELECT count(*) FROM (SELECT 1 FROM turns t JOIN sessions s ON s.id=t.session_id WHERE t.ext_id=?1 AND s.started_at<=?2 AND coalesce(t.ts,s.started_at)<=?2 LIMIT 2)",params![id,at],|r|r.get(0))?;
            if count!=1 {duplicates.push(id.clone());}
        }
        for id in duplicates {out.remove(&id);}
        let mut canonical_facts=Vec::new();let mut attributes=BTreeMap::new();
        for f in &pacts {
            let mut st=tx.prepare("SELECT statement FROM facts WHERE id=?1 AND status!='retracted' AND recorded_first<=?2")?;
            let body=st.query_map(params![f.id,at],|r|r.get::<_,String>(0))?.next().transpose()?;
            if body.as_deref()!=Some(&f.statement) {continue;}
            let mut st=tx.prepare("SELECT coalesce(t.ext_id,cast(t.id AS TEXT)) FROM fact_evidence e JOIN turns t ON t.id=e.turn_id WHERE e.fact_id=?1 ORDER BY t.id")?;
            let evidence=st.query_map([f.id],|r|r.get::<_,String>(0))?.collect::<std::result::Result<BTreeSet<_>,_>>()?;
            if !evidence.is_empty() && evidence==f.evidence_turns.iter().cloned().collect() && evidence.iter().all(|i|out.contains_key(i)) {
                canonical_facts.push(f.clone());
                let mut st=tx.prepare("SELECT subject_id,predicate,coalesce('entity:'||cast(object_id AS TEXT),'value:'||object_value),negated FROM facts WHERE id=?1 AND time_kind='ongoing' AND status='active'")?;
                if let Some(row)=st.query_map([f.id],|r|Ok((r.get::<_,i64>(0)?,r.get::<_,String>(1)?,r.get::<_,String>(2)?,r.get::<_,bool>(3)?)))?.next().transpose()? {attributes.insert(f.id,row);}
            }
        }
        Ok((out,saturated,canonical_facts,attributes,anchors))
    }).await?;
    if turns.len() > SOURCE_LIMIT {
        return Err(Error::InvalidInput("source candidate limit512".into()));
    }
    let stop = words(STOP);
    let direct: BTreeSet<_> = words(question).difference(&stop).cloned().collect();
    let expanded: BTreeSet<_> = super::cross_session::topic_terms(question, &[]).into_iter().collect();
    let mut units = Vec::new();
    let mut missing = Vec::new();
    let mut ambiguous = Vec::new();
    for turn in turns.values() {
        if !matches!(turn.role.as_str(), "user" | "assistant" | "participant") {
            return Err(Error::InvalidInput("invalid stored source role".into()));
        }
        if !anchors.is_empty()
            && !anchors.iter().any(|a| a.eq_ignore_ascii_case(&turn.speaker) || words(a).is_subset(&words(&turn.text)))
        {
            continue;
        }
        let ws = words(&turn.text);
        let topics: BTreeSet<_> = expanded.intersection(&ws).cloned().collect();
        if topics.is_empty() {
            continue;
        }
        let mut ids = vec![turn.id.clone()];
        if book_reference(&turn.text) {
            let mut antecedents: Vec<_> = turns
                .values()
                .filter(|t| {
                    t.id != turn.id
                        && (t.said_at < turn.said_at
                            || (t.session == turn.session && t.seq < turn.seq && t.said_at <= turn.said_at))
                        && !book_reference(&t.text)
                        && words(&t.text).iter().any(|w| w.starts_with("recommend") || w.starts_with("suggest"))
                })
                .collect();
            antecedents.sort_by(|a, b| b.said_at.cmp(&a.said_at).then(a.id.cmp(&b.id)));
            if antecedents.is_empty() {
                missing.push(turn.id.clone());
                continue;
            }
            if antecedents.len() > 1 {
                ambiguous.push(turn.id.clone());
            }
            ids.extend(antecedents.into_iter().take(2).map(|t| t.id.clone()));
        } else if short_reference(&turn.text) {
            if let Some(previous) = turns.values().find(|t| t.session == turn.session && t.seq + 1 == turn.seq) {
                ids.push(previous.id.clone());
            } else {
                missing.push(turn.id.clone());
                continue;
            }
        }
        let mut tokens = 0;
        for id in &ids {
            tokens += counter.count(&turns[id].text).map_err(|e| Error::Task(e.to_string()))?;
        }
        let weight = if turn.role == "assistant" { 1.0 } else { 1.5 };
        let score = (ws.intersection(&direct).count() as f64 + topics.len() as f64 * 0.15) * weight
            / (16.0 + tokens as f64).sqrt();
        units.push(Unit { ids, primary: turn.primary && turn.role != "assistant", score, topics });
    }
    let notice = format!(
        "{}## Retrieval scope\nBounded search; omissions and unresolved references may remain. Missing here does not establish global absence.\n",
        allocation.prefix
    );
    let mut pack =
        ContextPack::new(Day::from_ms(at), counter, budget)?.compact_excerpt_headers(true).with_notice(&notice)?;
    let mut seen = HashSet::new();
    let mut protected = Vec::new();
    for f in canonical_facts {
        pack.fact(f)?;
    }
    let mut covered = BTreeSet::new();
    let mut selected_units = 0;
    while !units.is_empty() && selected_units < UNIT_LIMIT {
        units.sort_by(|a, b| {
            let score = |u: &Unit| u.score * (1.0 + 0.15 * u.topics.difference(&covered).count() as f64);
            b.primary.cmp(&a.primary).then(score(b).total_cmp(&score(a))).then(a.ids.cmp(&b.ids))
        });
        let unit = units.remove(0);
        let ids: Vec<_> = unit.ids.iter().filter(|id| !seen.contains(*id)).cloned().collect();
        if ids.is_empty() {
            continue;
        }
        let chunks = ids.iter().map(|id| turns[id].chunk()).collect();
        if pack.bundle(vec![], chunks)? {
            for id in &ids {
                seen.insert(id.clone());
            }
            if unit.primary {
                protected.extend(ids);
            }
            covered.extend(unit.topics);
            selected_units += 1;
        }
    }
    let unit_limit_reached = !units.is_empty();
    let mut cited: BTreeSet<_> = pack.facts.iter().flat_map(|f| f.evidence_turns.clone()).collect();
    cited.extend(seen);
    let sources = cited
        .iter()
        .map(|id| {
            let t = &turns[id];
            SourceWitness {
                source_id: id.clone(),
                session: t.session.clone(),
                speaker: t.speaker.clone(),
                role: t.role.clone(),
                said_at: t.said_at,
                body_blake3: blake3::hash(t.text.as_bytes()).to_hex().to_string(),
                verification: "canonical_turn_identity_and_body_only",
            }
        })
        .collect();
    // Selected structured claims retain attributes from the SAME read snapshot.
    // Different values can coexist; no semantic exclusivity or raw-language proof.
    let mut groups: CurrentAssertions = BTreeMap::new();
    for f in &pack.facts {
        if let Some((subject, predicate, value, negated)) = attributes.get(&f.id) {
            groups.entry((*subject, crate::ingest::predicate_key(predicate))).or_default().push((
                f.id,
                value.clone(),
                *negated,
            ));
        }
    }
    let competing: Vec<_> = groups
        .into_iter()
        .filter_map(|((subject_id, predicate), v)| {
            let alternatives: BTreeSet<_> = v.iter().map(|(_, val, neg)| (val, neg)).collect();
            (alternatives.len() > 1).then(|| CompetingClaims {
                subject_id,
                predicate,
                fact_ids: v.iter().map(|r| r.0).collect(),
                reason: "different_current_assertions_require_semantic_review_and_may_coexist",
            })
        })
        .collect();
    let usage = pack.usage()?;
    let omitted = pack
        .omissions
        .turn_ids
        .iter()
        .filter(|id| !cited.contains(*id))
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    pack.omissions.turn_ids = omitted.clone();
    let status = if !competing.is_empty() || !ambiguous.is_empty() {
        "requires_review"
    } else if saturated
        || unit_limit_reached
        || !missing.is_empty()
        || !omitted.is_empty()
        || !pack.omissions.fact_ids.is_empty()
    {
        "incomplete"
    } else {
        "bounded"
    };
    let report = SelectionReport {
        policy: "role-turn-v1",
        status,
        search_scope: "bounded_candidates_not_exhaustive",
        semantic_completeness: "not_established",
        raw_claim_conflicts: "not_evaluated",
        candidate_limit_reached: saturated,
        unit_limit_reached,
        inspected_turns: turns.len(),
        protected_primary_turn_ids: protected,
        unresolved_dependencies: missing,
        ambiguous_dependencies: ambiguous,
        budget_omitted_turn_ids: omitted,
        competing_claims: competing,
        selected_sources: sources,
    };
    Ok((
        EvidenceContext {
            context: pack.context,
            context_usage: usage,
            cited_turns: cited.into_iter().collect(),
            budget_omissions: pack.omissions,
        },
        report,
    ))
}
