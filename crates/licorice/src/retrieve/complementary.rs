//! One bounded round of complementary retrieval. Model planning lives outside the
//! store; this packer accepts only retrieved, attributed evidence, never model prose.
use super::{BudgetOmissions, ContextUsage, Retrieved, RetrievedChunk, packing::ContextPack};
use crate::{Error, Result, temporal::Day};
use licorice_providers::tokens::TokenCounter;
use serde::Serialize;
use std::collections::HashSet;

#[derive(Debug, Serialize)]
pub struct EvidenceContext {
    pub context: String,
    pub context_usage: ContextUsage,
    pub cited_turns: Vec<String>,
    pub budget_omissions: BudgetOmissions,
}

/// Keep a bounded primary prefix, admit complementary whole source turns, then
/// backfill with primary evidence. Recount the entire rendered context on every
/// admission. No-query plans preserve the original context byte for byte.
pub fn merge(
    primary: &Retrieved,
    followups: &[Retrieved],
    at: Day,
    counter: &dyn TokenCounter,
    budget: usize,
    primary_budget: usize,
) -> Result<EvidenceContext> {
    if followups.len() > 2 || budget > 2000 || primary_budget > budget {
        return Err(Error::InvalidInput("complementary retrieval requires <=2 searches and <=2000 tokens".into()));
    }
    if followups.is_empty() {
        if counter.count(&primary.context).map_err(|e| Error::Task(e.to_string()))? > budget {
            return Err(Error::InvalidInput("primary context exceeds budget".into()));
        }
        return Ok(EvidenceContext {
            context: primary.context.clone(),
            context_usage: ContextUsage { budget_tokens: budget, ..primary.context_usage.clone() },
            cited_turns: primary.cited_turns.clone(),
            budget_omissions: primary.budget_omissions.clone(),
        });
    }
    let mut prefix = ContextPack::new(at, counter, primary_budget)?;
    for f in &primary.facts {
        prefix.fact(f.clone())?;
    }
    for c in &primary.chunks {
        prefix.chunk(c.clone())?;
    }
    for s in &primary.sessions {
        prefix.session(s.clone())?;
    }
    let mut pack = ContextPack::new(at, counter, budget)?;
    let mut facts = HashSet::new();
    let mut turns = HashSet::new();
    let mut sessions = HashSet::new();
    for f in prefix.facts {
        if pack.fact(f.clone())? {
            facts.insert(f.id);
        }
    }
    for c in prefix.chunks {
        add_chunk(&mut pack, c, &mut turns)?;
    }
    for s in prefix.sessions {
        pack.session(s.clone())?;
        sessions.insert(s.session);
    }
    // Round robin prevents query one from monopolizing the complementary space.
    for i in 0..followups.iter().map(|r| r.chunks.len()).max().unwrap_or(0) {
        for r in followups {
            if let Some(c) = r.chunks.get(i) {
                add_chunk(&mut pack, c.clone(), &mut turns)?;
            }
        }
    }
    for f in &primary.facts {
        if !facts.contains(&f.id) && pack.fact(f.clone())? {
            facts.insert(f.id);
        }
    }
    for c in &primary.chunks {
        add_chunk(&mut pack, c.clone(), &mut turns)?;
    }
    for s in &primary.sessions {
        if !sessions.contains(&s.session) {
            pack.session(s.clone())?;
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
    Ok(EvidenceContext {
        context: pack.context,
        context_usage: usage,
        cited_turns: cited,
        budget_omissions: pack.omissions,
    })
}

fn add_chunk(pack: &mut ContextPack<'_>, mut chunk: RetrievedChunk, seen: &mut HashSet<String>) -> Result<()> {
    chunk.turns.retain(|t| !seen.contains(&t.0));
    if chunk.turns.is_empty() {
        return Ok(());
    }
    pack.chunk(chunk)?;
    seen.extend(pack.chunks.iter().flat_map(|c| c.turns.iter().map(|t| t.0.clone())));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{Intent, SelectionTrace};
    use super::*;
    use licorice_providers::tokens::Cl100kCounter;
    fn day() -> Day {
        Day::from_ymd(2023, 5, 8).unwrap()
    }
    fn retrieved(turns: Vec<(&str, String)>) -> Retrieved {
        let mut p = ContextPack::new(day(), &Cl100kCounter, 2000).unwrap();
        for (i, (id, text)) in turns.into_iter().enumerate() {
            p.chunk(RetrievedChunk {
                id: i as i64,
                session: "s".into(),
                day: day(),
                turns: vec![(id.into(), "Ada".into(), text)],
            })
            .unwrap();
        }
        let usage = p.usage().unwrap();
        let cited_turns = p.chunks.iter().flat_map(|c| c.turns.iter().map(|t| t.0.clone())).collect();
        Retrieved {
            intent: Intent::None,
            linked_entities: vec![],
            facts: p.facts,
            chunks: p.chunks,
            sessions: p.sessions,
            context: p.context,
            cited_turns,
            context_usage: usage,
            budget_omissions: p.omissions,
            selection: SelectionTrace::default(),
            fixed_summary: None,
            graph_paths: vec![],
            reranking: None,
            rerank_error: None,
        }
    }
    #[test]
    fn preserves_no_query_and_refuses_unbounded_work() {
        let r = retrieved(vec![("t1", "Ada plays piano.".into())]);
        assert_eq!(merge(&r, &[], day(), &Cl100kCounter, 2000, 1400).unwrap().context, r.context);
        assert!(merge(&r, &[], day(), &Cl100kCounter, 2001, 1400).is_err());
        assert!(merge(&r, &[], day(), &Cl100kCounter, 20, 1400).is_err());
        assert!(
            merge(&r, &[retrieved(vec![]), retrieved(vec![]), retrieved(vec![])], day(), &Cl100kCounter, 2000, 1400)
                .is_err()
        );
    }
    #[test]
    fn deduplicates_whole_turns_and_inserts_missing_premise() {
        let r = retrieved(vec![("t1", "Ada's sister is Mira.".into())]);
        let f = retrieved(vec![("t1", "Ada's sister is Mira.".into()), ("t2", "Mira lives in Bergen.".into())]);
        let m = merge(&r, &[f], day(), &Cl100kCounter, 2000, 1400).unwrap();
        assert!(m.context.contains("Mira lives in Bergen."));
        assert_eq!(m.context.matches("[t1]").count(), 1);
        assert_eq!(m.cited_turns, vec!["t1", "t2"]);
    }
    #[test]
    fn overflow_skips_whole_turn_then_admits_smaller_unicode_source() {
        let r = retrieved(vec![("t1", "Ada's sister is Mira.".into())]);
        let f = retrieved(vec![("t2", "long ".repeat(500)), ("t3", "Mira lives in 日本語 café 🍌.".into())]);
        let m = merge(&r, &[f], day(), &Cl100kCounter, 140, 70).unwrap();
        assert!(m.context.contains("Mira lives in 日本語 café 🍌."));
        assert!(!m.context.contains("long "));
        assert_eq!(m.context_usage.tokens, Cl100kCounter.count(&m.context).unwrap());
        assert!(m.context_usage.tokens <= 140);
        assert!(m.budget_omissions.turn_ids.contains(&"t2".to_string()));
    }
}
