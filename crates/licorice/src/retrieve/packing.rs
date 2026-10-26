//! Pack whole evidence units, testing the fully rendered text each time. Counting
//! fragments separately is unsound for tokenizers that merge across boundaries.

use licorice_providers::tokens::TokenCounter;
use serde::Serialize;

use super::{RetrievedChunk, RetrievedFact, RetrievedSession, assemble};
use crate::temporal::Day;
use crate::{Error, Result};

#[derive(Clone, Debug, Default, Serialize)]
pub struct ContextUsage {
    pub tokenizer: String,
    pub budget_tokens: usize,
    pub tokens: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BudgetOmissions {
    pub fact_ids: Vec<i64>,
    pub turn_ids: Vec<String>,
    pub session_ids: Vec<String>,
    pub theme_ids: Vec<i64>,
}

pub(super) struct ContextPack<'a> {
    at: Day,
    counter: &'a dyn TokenCounter,
    budget: usize,
    compact_excerpt_headers: bool,
    notice: String,
    pub facts: Vec<RetrievedFact>,
    pub chunks: Vec<RetrievedChunk>,
    pub sessions: Vec<RetrievedSession>,
    pub context: String,
    pub omissions: BudgetOmissions,
}

impl<'a> ContextPack<'a> {
    pub fn new(at: Day, counter: &'a dyn TokenCounter, budget: usize) -> Result<Self> {
        let context = if budget == 0 { String::new() } else { assemble(at, &[], &[], &[], false) };
        let tokens = count(counter, &context)?;
        if tokens > budget {
            return Err(Error::InvalidInput(format!("context budget must fit its {tokens}-token header, or be zero")));
        }
        Ok(Self {
            at,
            counter,
            budget,
            compact_excerpt_headers: false,
            notice: String::new(),
            facts: vec![],
            chunks: vec![],
            sessions: vec![],
            context,
            omissions: BudgetOmissions::default(),
        })
    }

    pub fn compact_excerpt_headers(mut self, compact: bool) -> Self {
        self.compact_excerpt_headers = compact;
        self
    }

    /// A source-scope notice is part of the SAME fully rendered token budget.
    /// Set it before admitting evidence so protection never needs eviction.
    pub fn with_notice(mut self, notice: &str) -> Result<Self> {
        if !self.facts.is_empty() || !self.chunks.is_empty() || !self.sessions.is_empty() {
            return Err(Error::InvalidInput("context notice must precede evidence admission".into()));
        }
        let text = format!("{notice}{}", self.context);
        if count(self.counter, &text)? > self.budget {
            return Err(Error::InvalidInput("context budget cannot fit its scope notice".into()));
        }
        self.notice = notice.to_owned();
        self.context = text;
        Ok(self)
    }

    fn accept(&mut self) -> Result<bool> {
        let candidate = format!(
            "{}{}",
            self.notice,
            assemble(self.at, &self.sessions, &self.facts, &self.chunks, self.compact_excerpt_headers)
        );
        if count(self.counter, &candidate)? <= self.budget {
            self.context = candidate;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn fact(&mut self, fact: RetrievedFact) -> Result<bool> {
        let id = fact.id;
        self.facts.push(fact);
        if self.accept()? {
            Ok(true)
        } else {
            self.facts.pop();
            self.omissions.fact_ids.push(id);
            Ok(false)
        }
    }

    pub fn bundle(&mut self, facts: Vec<RetrievedFact>, sources: Vec<RetrievedChunk>) -> Result<bool> {
        let before = self.facts.len();
        let before_chunks = self.chunks.len();
        let ids: Vec<_> = facts.iter().map(|f| f.id).collect();
        let turn_ids: Vec<_> = sources.iter().flat_map(|c| c.turns.iter().map(|t| t.0.clone())).collect();
        self.facts.extend(facts);
        self.chunks.extend(sources);
        if self.accept()? {
            Ok(true)
        } else {
            self.facts.truncate(before);
            self.chunks.truncate(before_chunks);
            self.omissions.fact_ids.extend(ids);
            self.omissions.turn_ids.extend(turn_ids);
            Ok(false)
        }
    }

    pub fn chunk(&mut self, chunk: RetrievedChunk) -> Result<()> {
        let RetrievedChunk { id, session, day, turns } = chunk;
        self.chunks.push(RetrievedChunk { id, session, day, turns: vec![] });
        for turn in turns {
            let id = turn.0.clone();
            self.chunks.last_mut().unwrap().turns.push(turn);
            if !self.accept()? {
                self.chunks.last_mut().unwrap().turns.pop();
                self.omissions.turn_ids.push(id);
            }
        }
        if self.chunks.last().unwrap().turns.is_empty() {
            self.chunks.pop();
        }
        Ok(())
    }

    pub fn session(&mut self, session: RetrievedSession) -> Result<()> {
        let id = session.session.clone();
        self.sessions.push(session);
        if !self.accept()? {
            self.sessions.pop();
            self.omissions.session_ids.push(id);
        }
        Ok(())
    }

    /// Append a whole theme within its session, restoring the prior rendered
    /// context and row on failure so a later, smaller theme can still fit.
    pub fn theme(&mut self, id: i64, session: RetrievedSession) -> Result<bool> {
        if let Some(index) = self.sessions.iter().position(|s| s.session == session.session) {
            let original = self.sessions[index].summary.clone();
            let text = session.summary.split_once("): ").map(|(_, t)| t).unwrap_or(&session.summary);
            self.sessions[index].summary.push(' ');
            self.sessions[index].summary.push_str(text);
            if self.accept()? {
                return Ok(true);
            }
            self.sessions[index].summary = original;
        } else {
            self.sessions.push(session);
            if self.accept()? {
                return Ok(true);
            }
            self.sessions.pop();
        }
        self.omissions.theme_ids.push(id);
        Ok(false)
    }

    pub fn usage(&self) -> Result<ContextUsage> {
        let tokens = count(self.counter, &self.context)?;
        if tokens > self.budget {
            return Err(Error::Task("token counter changed while packing context".into()));
        }
        Ok(ContextUsage { tokenizer: self.counter.id().into(), budget_tokens: self.budget, tokens })
    }
}

fn count(counter: &dyn TokenCounter, text: &str) -> Result<usize> {
    counter.count(text).map_err(|e| Error::Task(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use licorice_providers::tokens::Cl100kCounter;

    fn day() -> Day {
        Day::from_ymd(2023, 5, 8).unwrap()
    }
    fn fact(id: i64, text: &str) -> RetrievedFact {
        RetrievedFact {
            id,
            statement: text.into(),
            day: day(),
            dated: false,
            occurred: None,
            superseded: false,
            valid_until: None,
            session: "S1".into(),
            evidence_turns: vec![format!("S1:{id}")],
            score: 1.0,
        }
    }

    #[test]
    fn exact_boundary_counts_headings_dates_citations_and_unicode() {
        let f = fact(1, "Ada likes 🍌, 日本語 and café.");
        let rendered = assemble(day(), &[], std::slice::from_ref(&f), &[], false);
        let required = count(&Cl100kCounter, &rendered).unwrap();
        let mut exact = ContextPack::new(day(), &Cl100kCounter, required).unwrap();
        assert!(exact.fact(f.clone()).unwrap());
        assert_eq!(exact.context, rendered);
        assert_eq!(exact.usage().unwrap().tokens, required);
        let mut short = ContextPack::new(day(), &Cl100kCounter, required - 1).unwrap();
        assert!(!short.fact(f).unwrap());
        assert!(!short.context.contains("S1:1"));
        assert_eq!(short.omissions.fact_ids, vec![1]);
        assert!(short.facts.is_empty());
    }

    #[test]
    fn failed_theme_append_restores_prior_content_and_later_theme_fits() {
        let mut pack = ContextPack::new(day(), &Cl100kCounter, 90).unwrap();
        let theme = |text: String| RetrievedSession {
            session: "S1".into(),
            day: day(),
            summary: format!("Session S1 (date): {text}"),
            score: 1.0,
        };
        assert!(pack.theme(1, theme("Travel: Ada visited Oslo.".into())).unwrap());
        let prior = pack.context.clone();
        assert!(!pack.theme(2, theme("large ".repeat(1000))).unwrap());
        assert_eq!(pack.context, prior);
        assert!(!pack.sessions[0].summary.contains("large"));
        assert!(pack.theme(3, theme("Music: Ada plays piano.".into())).unwrap());
        assert!(pack.context.contains("Travel: Ada visited Oslo. Music: Ada plays piano."));
        assert_eq!(pack.omissions.theme_ids, vec![2]);
        assert_eq!(pack.sessions.len(), 1);
    }

    #[test]
    fn zero_is_empty_and_too_small_positive_budget_is_an_error() {
        let mut pack = ContextPack::new(day(), &Cl100kCounter, 0).unwrap();
        assert!(!pack.fact(fact(1, "Ada lives in Oslo.")).unwrap());
        assert_eq!(pack.context, "");
        assert_eq!(pack.usage().unwrap().tokens, 0);
        assert!(matches!(ContextPack::new(day(), &Cl100kCounter, 1), Err(Error::InvalidInput(_))));
    }

    fn compact_sources() -> Vec<RetrievedChunk> {
        (1..=2)
            .map(|id| RetrievedChunk {
                id,
                session: "0123456789abcdef".repeat(4),
                day: day(),
                turns: vec![(
                    format!("m:fedcba9876543210:{id}"),
                    "Ada".into(),
                    format!("Whole source {id}: café, 日本語 and 🍌."),
                )],
            })
            .collect()
    }

    #[test]
    fn compact_headers_pack_more_whole_turns_without_rewriting_citations_or_source() {
        let sources = compact_sources();
        let rendered = assemble(day(), &[], &[], &sources, true);
        let required = count(&Cl100kCounter, &rendered).unwrap();
        let mut compact = ContextPack::new(day(), &Cl100kCounter, required).unwrap().compact_excerpt_headers(true);
        let mut legacy = ContextPack::new(day(), &Cl100kCounter, required).unwrap();
        for source in sources.clone() {
            compact.chunk(source.clone()).unwrap();
            legacy.chunk(source).unwrap();
        }
        assert_eq!(compact.context, rendered);
        assert_eq!(compact.usage().unwrap().tokens, required);
        assert!(legacy.chunks.len() < compact.chunks.len());
        for (actual, original) in compact.chunks.iter().zip(&sources) {
            assert_eq!(actual.session, original.session);
            assert_eq!(actual.turns, original.turns);
            assert!(compact.context.contains(&format!("[{}] Ada: {}", original.turns[0].0, original.turns[0].2)));
        }
        assert!(!compact.context.contains(&sources[0].session));
        assert!(!super::super::RetrieveConfig::default().compact_excerpt_headers);
    }

    #[test]
    fn compact_connected_bundles_still_fit_atomically_or_omit_the_entire_path() {
        let sources = compact_sources();
        let mut facts = vec![fact(1, "Ada's sister is Mira."), fact(2, "Mira lives in Bergen.")];
        for (fact, source) in facts.iter_mut().zip(&sources) {
            fact.evidence_turns = vec![source.turns[0].0.clone()];
        }
        let rendered = assemble(day(), &[], &facts, &sources, true);
        let required = count(&Cl100kCounter, &rendered).unwrap();
        let mut exact = ContextPack::new(day(), &Cl100kCounter, required).unwrap().compact_excerpt_headers(true);
        assert!(exact.bundle(facts.clone(), sources.clone()).unwrap());
        assert_eq!(exact.context, rendered);
        let mut short = ContextPack::new(day(), &Cl100kCounter, required - 1).unwrap().compact_excerpt_headers(true);
        assert!(!short.bundle(facts, sources).unwrap());
        assert!(short.facts.is_empty() && short.chunks.is_empty());
        assert_eq!(short.omissions.fact_ids, vec![1, 2]);
        assert_eq!(short.omissions.turn_ids, vec!["m:fedcba9876543210:1", "m:fedcba9876543210:2"]);
    }

    #[test]
    fn oversized_facts_turns_and_summaries_do_not_block_later_units() {
        let mut pack = ContextPack::new(day(), &Cl100kCounter, 150).unwrap();
        assert!(!pack.fact(fact(1, &"long ".repeat(1000))).unwrap());
        assert!(pack.fact(fact(2, "Ada lives in Oslo.")).unwrap());
        pack.chunk(RetrievedChunk {
            id: 1,
            session: "S1".into(),
            day: day(),
            turns: vec![
                ("S1:3".into(), "Ada".into(), "banana ".repeat(1000)),
                ("S1:4".into(), "Ada".into(), "My favorite fruit is banana.".into()),
            ],
        })
        .unwrap();
        pack.session(RetrievedSession { session: "S2".into(), day: day(), summary: "long ".repeat(1000), score: 1.0 })
            .unwrap();
        pack.session(RetrievedSession { session: "S3".into(), day: day(), summary: "Ada's home.".into(), score: 0.9 })
            .unwrap();
        assert_eq!(pack.facts.len(), 1);
        assert_eq!(pack.chunks[0].turns.len(), 1);
        assert_eq!(pack.chunks[0].turns[0].2, "My favorite fruit is banana.");
        assert_eq!(pack.sessions[0].session, "S3");
        assert!(!pack.context.contains("S1:3"));
        assert_eq!(pack.omissions.turn_ids, vec!["S1:3"]);
        assert_eq!(pack.omissions.session_ids, vec!["S2"]);
        assert!(pack.usage().unwrap().tokens <= 150);
    }
}
