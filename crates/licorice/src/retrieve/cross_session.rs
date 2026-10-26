//! Opt-in lexical recall across sessions, preserving short dialogue bundles.
//! Query expansion uses a small general vocabulary, never benchmark annotations.
use super::{Retrieved, RetrievedChunk, RetrievedFact, complementary::EvidenceContext, packing::ContextPack};
use crate::store::{Index, LexicalKind, Store};
use crate::temporal::Day;
use crate::{Error, Result};
use licorice_providers::tokens::TokenCounter;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const STOP: &str = "a about after all an and are as at be been before by did do does for from has have her him his how i in is it its me my of on or our she some that the their them they this to was were what when where which who why with would you your s";
// Concept-level alternatives. No retrieved answer, proper name or dataset id
// may be inserted into this vocabulary or into the generated search expression.
const TOPICS: &[&str] = &[
    "pet pets cat cats dog dogs kitten kittens puppy puppies animal animals rabbit rabbits",
    "book books read reading recommend recommended recommendation novel title suggest suggested suggestion",
    "bought buy buying purchase purchased purchases shopping ordered",
    "pottery ceramic ceramics clay workshop class",
    "paint painted painting paintings artwork art",
    "destress stress relax relaxing calming therapeutic therapy",
    "instrument instruments musical band bands artists concert music play plays playing",
    "camp camped camping hike hikes hiking tent trail outdoors",
    "identity gender transition transgender trans",
    "move moved moving relocate relocated country hometown birthplace origin roots",
    "support supports supported supporting supporter supporters",
    "activity activities participate participates participated participating participation attend attends attended attending volunteer volunteers volunteered volunteering organize organizes organized organizing organised organising join joined joining went host hosted hosting arrange arranged arranging prepare prepared preparing putting",
];

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()).map(str::to_lowercase).collect()
}

// Conservative plural normalization is shared by question and evidence matching.
// Keep proper names untouched in the actual lexical query and source text.
fn singular(word: &str) -> &str {
    if word.len() > 4 && word.ends_with('s') && !word.ends_with("ss") && !word.ends_with("us") && !word.ends_with("is")
    {
        &word[..word.len() - 1]
    } else {
        word
    }
}

fn topic_key(word: &str) -> String {
    singular(word).to_owned()
}

fn activity_keys(text: &str) -> BTreeSet<String> {
    let mut keys: BTreeSet<_> = words(text)
        .into_iter()
        .filter_map(|w| {
            if w.starts_with("organiz") || w.starts_with("organis") {
                Some("organize")
            } else if w.starts_with("volunteer") {
                Some("volunteer")
            } else if w.starts_with("mentor") {
                Some("mentor")
            } else if w.starts_with("join") {
                Some("join")
            } else if w.starts_with("attend") || w == "went" {
                Some("attend")
            } else if w.starts_with("advoca") {
                Some("advocate")
            } else {
                None
            }
        })
        .map(str::to_owned)
        .collect();
    if text.to_lowercase().contains("putting together") {
        keys.insert("organize".into());
    }
    keys
}

/// Remove person anchors from the topical search, so a frequently named person
/// cannot fill the lexical pool with every unrelated activity in their history.
pub fn topic_terms(question: &str, anchors: &[String]) -> Vec<String> {
    let excluded: HashSet<_> = anchors.iter().flat_map(|a| words(a)).chain(words(STOP)).collect();
    let original: Vec<_> = words(question).into_iter().filter(|w| w.len() > 1 && !excluded.contains(w)).collect();
    let mut terms = original.clone();
    terms.extend(original.iter().map(|w| singular(w).to_owned()));
    for group in TOPICS {
        let alternatives = words(group);
        if original.iter().any(|w| alternatives.contains(w)) {
            terms.extend(alternatives);
        }
    }
    terms.sort();
    terms.dedup();
    terms.truncate(64);
    terms
}

#[derive(Debug, Serialize)]
pub struct RecoveryTrace {
    pub anchors: Vec<String>,
    pub terms: Vec<String>,
    pub candidate_windows: usize,
    pub admitted_windows: usize,
    pub admitted_sessions: Vec<String>,
    pub retained_primary_fact_ids: Vec<i64>,
    pub protected_primary_fact_ids: Vec<i64>,
    pub protected_primary_turn_ids: Vec<String>,
    pub linked_windows: usize,
}

#[derive(Clone)]
struct Window {
    chunk: RetrievedChunk,
    score: f64,
    topics: BTreeSet<String>,
    activities: BTreeSet<String>,
}

struct PrimaryUnit {
    fact: Option<RetrievedFact>,
    chunk: Option<RetrievedChunk>,
    topics: BTreeSet<String>,
    ordinal: usize,
}

/// Bound both work and output: <=96 lexical seed hits, at most three same-batch
/// source turns per seed, <=16 admitted windows, and <=2000 rendered tokens.
/// Protect topical primary witnesses first, then add complementary bundles in the
/// remaining space. No admitted primary unit is evicted. A rejected bundle never
/// leaves only its reply without context; irrelevant primary rows backfill last.
pub async fn recover(
    store: &Store,
    primary: &Retrieved,
    question: &str,
    at: i64,
    counter: &dyn TokenCounter,
    budget: usize,
) -> Result<(EvidenceContext, RecoveryTrace)> {
    if question.trim().is_empty() || question.chars().count() > 2000 || !(64..=2000).contains(&budget) {
        return Err(Error::InvalidInput(
            "cross-session recovery requires a bounded question and 64..2000 tokens".into(),
        ));
    }
    let normalized = format!(" {} ", words(question).join(" "));
    let anchors: Vec<String> = store.read(move |c| {
        let mut stmt=c.prepare("SELECT DISTINCT a.alias FROM entity_aliases a JOIN entities e ON e.id=a.entity_id WHERE e.kind='person' AND e.merged_into IS NULL AND instr(?1,' '||lower(a.alias)||' ')>0 ORDER BY a.alias LIMIT 16")?;
        Ok(stmt.query_map([&normalized],|r|r.get(0))?.collect::<std::result::Result<_,_>>()?)
    }).await?;
    let terms = topic_terms(question, &anchors);
    let seeds = if terms.is_empty() { vec![] } else { store.lexical(LexicalKind::Chunk, &terms.join(" "), 96).await? };
    let ids: Vec<i64> = seeds.iter().map(|h| h.owner).collect();
    let search_terms: BTreeSet<_> = terms.iter().map(|t| topic_key(t)).collect();
    let names = anchors.clone();
    let query_words = words(question);
    let want_names = query_words.iter().any(|w| matches!(w.as_str(), "name" | "names"));
    let want_activities =
        query_words.iter().any(|w| w.starts_with("participat") || matches!(w.as_str(), "activity" | "activities"));
    let windows:Vec<Window>=store.read(move |c| {
        let mut out=Vec::new();
        for (rank,id) in ids.iter().enumerate() {
            let (sid,session,day,seq):(i64,String,i64,i64)=c.query_row(
                "SELECT t.session_id,s.ext_id,s.started_at,t.seq FROM chunks k JOIN turns t ON t.id=k.first_turn JOIN sessions s ON s.id=t.session_id WHERE k.id=?1 AND s.started_at<=?2 AND coalesce(t.ts,s.started_at)<=?2",
                rusqlite::params![id,at],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).or_else(|e|if matches!(e,rusqlite::Error::QueryReturnedNoRows){Ok((-1,String::new(),0,0))}else{Err(e)})?;
            if sid<0 {continue;}
            let mut stmt=c.prepare_cached("SELECT coalesce(t.ext_id,cast(t.id AS TEXT)),t.speaker,t.text FROM turns t WHERE t.session_id=?1 AND t.seq BETWEEN ?2 AND ?3 AND coalesce(t.ts,?4)<=?5 ORDER BY t.seq")?;
            let turns:Vec<(String,String,String)>=stmt.query_map(rusqlite::params![sid,seq.saturating_sub(1),seq+1,day,at],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?.collect::<std::result::Result<_,_>>()?;
            let text=turns.iter().map(|t|format!("{} {}",t.1,t.2)).collect::<Vec<_>>().join(" ");
            let ws:HashSet<_>=words(&text).into_iter().collect();
            let keys:BTreeSet<_>=ws.iter().map(|w|topic_key(w)).collect();
            let matched:BTreeSet<_>=search_terms.intersection(&keys).cloned().collect();
            let anchor_matches=names.iter().filter(|a| words(a).iter().all(|w|ws.contains(w))).count();
            if !names.is_empty() && anchor_matches==0 {continue;}
            let naming=want_names && ws.iter().any(|w|matches!(w.as_str(),"name"|"names"|"named"|"called"));
            let action=want_activities && turns.iter().any(|t| {
                let body=t.2.to_lowercase();
                ["i went", "i attended", "i joined", "i volunteered", "i organized", "i organised", "i am organizing", "i'm organizing", "i'm putting together", "i am putting together"].iter().any(|p|body.contains(p))
            });
            let score=1.0/(20.0+rank as f64) + matched.len() as f64*0.04 + anchor_matches as f64*0.12 + if naming {0.25} else {0.0} + if action {0.24} else {0.0};
            let activities=if want_activities {activity_keys(&text)} else {BTreeSet::new()};
            out.push(Window{chunk:RetrievedChunk{id:*id,session,day:Day::from_ms(day),turns},score,topics:matched,activities});
        }
        Ok(out)
    }).await?;
    let candidate_windows = windows.len();
    let day = Day::from_ms(at);
    let source_ids: HashSet<String> = primary
        .facts
        .iter()
        .flat_map(|f| f.evidence_turns.clone())
        .chain(primary.chunks.iter().flat_map(|c| c.turns.iter().map(|t| t.0.clone())))
        .collect();
    let eligible:HashSet<String>=store.read(move |c| {
        let mut out=HashSet::new();let mut stmt=c.prepare_cached("SELECT coalesce(t.ts,s.started_at),s.started_at FROM turns t JOIN sessions s ON s.id=t.session_id WHERE t.ext_id=?1")?;
        for id in &source_ids {
            let mut rows=stmt.query([id])?;let mut count=0;let mut allowed=true;
            while let Some(row)=rows.next()? {count+=1;let ts:Option<i64>=row.get(0)?;let session:Option<i64>=row.get(1)?;allowed &= ts.is_some_and(|v|v<=at) && session.is_some_and(|v|v<=at);}
            if count==1 && allowed {out.insert(id.clone());}
        } Ok(out)
    }).await?;
    let overlap = |text: &str| {
        let ws: BTreeSet<_> = words(text).iter().map(|w| topic_key(w)).collect();
        let query: BTreeSet<_> = terms.iter().map(|t| topic_key(t)).collect();
        query.intersection(&ws).cloned().collect::<BTreeSet<_>>()
    };
    let mut units = Vec::new();
    for chunk in &primary.chunks {
        if chunk.day > day {
            continue;
        }
        let mut chunk = chunk.clone();
        chunk.turns.retain(|t| eligible.contains(&t.0));
        let topics = overlap(&chunk.turns.iter().map(|t| t.2.as_str()).collect::<Vec<_>>().join(" "));
        if !chunk.turns.is_empty() && !topics.is_empty() {
            units.push(PrimaryUnit { fact: None, chunk: Some(chunk), topics, ordinal: units.len() });
        }
    }
    for fact in &primary.facts {
        let topics = overlap(&fact.statement);
        if fact.day <= day
            && !fact.evidence_turns.is_empty()
            && fact.evidence_turns.iter().all(|id| eligible.contains(id))
            && !topics.is_empty()
        {
            units.push(PrimaryUnit { fact: Some(fact.clone()), chunk: None, topics, ordinal: units.len() });
        }
    }
    // Compact primary facts precede raw chunks. Raw witnesses for topics not yet
    // represented are protected next; same-topic raw detail competes with other
    // complementary evidence instead of exhausting the packet first. There is no
    // fixed token partition and no admitted primary witness is evicted.
    let mut frequency = HashMap::new();
    for unit in &units {
        for topic in &unit.topics {
            *frequency.entry(topic.clone()).or_insert(0usize) += 1;
        }
    }
    units.sort_by(|a, b| {
        let score = |u: &PrimaryUnit| u.topics.iter().map(|t| 1.0 / (1.0 + frequency[t] as f64)).sum::<f64>();
        b.fact.is_some().cmp(&a.fact.is_some()).then(score(b).total_cmp(&score(a))).then(a.ordinal.cmp(&b.ordinal))
    });
    let mut pack = ContextPack::new(day, counter, budget)?.compact_excerpt_headers(true);
    let mut seen = HashSet::new();
    let mut retained = Vec::new();
    let mut protected_primary_turn_ids = Vec::new();
    let mut primary_topics = BTreeSet::new();
    for unit in units {
        if let Some(fact) = unit.fact {
            if !fact.evidence_turns.iter().all(|id| seen.contains(id)) && pack.fact(fact.clone())? {
                retained.push(fact.id);
                primary_topics.extend(unit.topics);
            }
        } else if let Some(mut chunk) = unit.chunk {
            if unit.topics.is_subset(&primary_topics) {
                continue;
            }
            chunk.turns.retain(|t| !seen.contains(&t.0));
            if !chunk.turns.is_empty() && pack.bundle(vec![], vec![chunk.clone()])? {
                protected_primary_turn_ids.extend(chunk.turns.iter().map(|t| t.0.clone()));
                seen.extend(chunk.turns.iter().map(|t| t.0.clone()));
                primary_topics.extend(unit.topics);
            }
        }
    }
    let protected_primary_fact_ids = retained.clone();
    let mut sessions: HashMap<String, usize> = HashMap::new();
    let mut covered = overlap(&pack.context);
    let mut covered_activities = activity_keys(&pack.context);
    let mut remaining = windows;
    let mut admitted = 0;
    let mut linked_windows = 0;
    // References and antecedents compete badly with repeated single-session
    // mentions. Offer up to two dated recommendation/reference bundles first.
    // This retrieves alternatives; it does not assert that the link is unique.
    let qwords = words(question);
    if qwords.iter().any(|w| w.starts_with("recommend") || w.starts_with("suggest"))
        && qwords.iter().any(|w| matches!(w.as_str(), "read" | "reading"))
    {
        let role = |w: &Window| {
            let ws = words(&w.chunk.turns.iter().map(|t| t.2.as_str()).collect::<Vec<_>>().join(" "));
            let recommendation = ws.iter().any(|w| w.starts_with("recommend"));
            let reading = ws.iter().any(|w| matches!(w.as_str(), "read" | "reading"));
            (recommendation && reading, recommendation && !reading)
        };
        let mut refs: Vec<_> = remaining.iter().filter(|w| role(w).0).collect();
        refs.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk.id.cmp(&b.chunk.id)));
        let mut offered = 0;
        for reference in refs.into_iter().take(4) {
            let mut antecedents: Vec<_> =
                remaining.iter().filter(|w| role(w).1 && w.chunk.day <= reference.chunk.day).collect();
            antecedents.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk.id.cmp(&b.chunk.id)));
            for antecedent in antecedents.into_iter().take(2) {
                if offered == 2 {
                    break;
                }
                offered += 1;
                let mut local_seen = seen.clone();
                let mut bundle = Vec::new();
                for window in [antecedent, reference] {
                    let mut chunk = window.chunk.clone();
                    chunk.turns.retain(|t| local_seen.insert(t.0.clone()));
                    if !chunk.turns.is_empty() {
                        bundle.push(chunk);
                    }
                }
                if !bundle.is_empty() && pack.bundle(vec![], bundle.clone())? {
                    for chunk in &bundle {
                        *sessions.entry(chunk.session.clone()).or_default() += 1;
                        seen.extend(chunk.turns.iter().map(|t| t.0.clone()));
                    }
                    covered.extend(antecedent.topics.iter().chain(reference.topics.iter()).cloned());
                    admitted += bundle.len();
                    linked_windows += bundle.len();
                }
            }
            if offered == 2 {
                break;
            }
        }
    }
    while !remaining.is_empty() && admitted < 16 {
        remaining.sort_by(|a, b| {
            let value = |w: &Window| {
                w.score
                    + 0.025 * w.topics.difference(&covered).count() as f64
                    + 0.20 * w.activities.difference(&covered_activities).count() as f64
                    - 0.035 * sessions.get(&w.chunk.session).copied().unwrap_or(0) as f64
                    - 0.15 * w.chunk.turns.iter().filter(|t| seen.contains(&t.0)).count() as f64
            };
            value(b).total_cmp(&value(a)).then(a.chunk.id.cmp(&b.chunk.id))
        });
        let mut window = remaining.remove(0);
        window.chunk.turns.retain(|t| !seen.contains(&t.0));
        if window.chunk.turns.is_empty() {
            continue;
        }
        if pack.bundle(vec![], vec![window.chunk.clone()])? {
            seen.extend(window.chunk.turns.iter().map(|t| t.0.clone()));
            covered.extend(window.topics);
            covered_activities.extend(window.activities);
            *sessions.entry(window.chunk.session.clone()).or_default() += 1;
            admitted += 1;
        }
    }
    for fact in &primary.facts {
        if fact.day <= day
            && !retained.contains(&fact.id)
            && !fact.evidence_turns.iter().all(|id| seen.contains(id))
            && !fact.evidence_turns.is_empty()
            && fact.evidence_turns.iter().all(|id| eligible.contains(id))
            && pack.fact(fact.clone())?
        {
            retained.push(fact.id);
        }
    }
    for chunk in &primary.chunks {
        if chunk.day > day {
            continue;
        }
        let mut chunk = chunk.clone();
        chunk.turns.retain(|t| !seen.contains(&t.0) && eligible.contains(&t.0));
        if !chunk.turns.is_empty() && pack.bundle(vec![], vec![chunk.clone()])? {
            seen.extend(chunk.turns.iter().map(|t| t.0.clone()));
        }
    }
    let usage = pack.usage()?;
    let mut cited: Vec<_> = pack
        .facts
        .iter()
        .flat_map(|f| f.evidence_turns.clone())
        .chain(pack.chunks.iter().flat_map(|c| c.turns.iter().map(|t| t.0.clone())))
        .collect();
    cited.sort();
    cited.dedup();
    let admitted_sessions: BTreeMap<_, _> = sessions.into_iter().collect();
    Ok((
        EvidenceContext {
            context: pack.context,
            context_usage: usage,
            cited_turns: cited,
            budget_omissions: pack.omissions,
        },
        RecoveryTrace {
            anchors,
            terms,
            candidate_windows,
            admitted_windows: admitted,
            admitted_sessions: admitted_sessions.into_keys().collect(),
            retained_primary_fact_ids: retained,
            protected_primary_fact_ids,
            protected_primary_turn_ids,
            linked_windows,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn topical_query_does_not_let_the_named_person_dominate() {
        let terms = topic_terms("What are Noa's pets' names?", &["Noa".into()]);
        assert!(!terms.contains(&"noa".into()));
        assert!(terms.contains(&"cat".into()));
        assert!(terms.contains(&"rabbit".into()));
        assert!(!terms.contains(&"luna".into()));
    }
    #[test]
    fn purchase_expansion_does_not_invent_the_missing_item() {
        let terms = topic_terms("What has Omar bought?", &["Omar".into()]);
        assert!(terms.contains(&"purchased".into()));
        assert!(!terms.contains(&"figurines".into()));
    }
}
