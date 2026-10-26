//! Optional maximal marginal relevance on a bounded, already relevance-ranked pool.
//! Only evidence that actually fits the context participates in the redundancy term.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::Intent;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    #[default]
    Relevance,
    Mmr,
}

impl SelectionMode {
    pub(super) fn effective(self, intent: Intent, question: &str) -> Self {
        // Similar statements can represent separate occurrences needed for a count
        // or chronology. Do not penalize their similarity for those questions.
        if intent == Intent::Ordering || Intent::needs_occurrences(question) { Self::Relevance } else { self }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct SelectionTrace {
    pub requested: SelectionMode,
    pub effective: SelectionMode,
    pub candidates: usize,
    pub considered: Vec<SelectionDecision>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectionDecision {
    pub fact_id: i64,
    pub relevance: f64,
    pub redundancy: f64,
    pub utility: f64,
    pub packed: bool,
}

pub(super) struct Selector {
    mode: SelectionMode,
    lambda: f64,
    max_score: f64,
    candidates: Vec<(i64, f64)>,
    vectors: HashMap<i64, Vec<f32>>,
}

impl Selector {
    pub fn new(mode: SelectionMode, lambda: f64, candidates: Vec<(i64, f64)>, vectors: HashMap<i64, Vec<f32>>) -> Self {
        let max_score = candidates.iter().map(|(_, s)| *s).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
        Self { mode, lambda, max_score, candidates, vectors }
    }

    pub fn next(&mut self, selected: &[i64]) -> Option<SelectionDecision> {
        let decision = |&(id, score): &(i64, f64)| {
            let redundancy = if self.mode == SelectionMode::Mmr {
                selected.iter().map(|other| self.similarity(id, *other)).fold(0.0, f64::max)
            } else {
                0.0
            };
            let relevance = (score / self.max_score).clamp(0.0, 1.0);
            let utility = if self.mode == SelectionMode::Mmr {
                self.lambda * relevance - (1.0 - self.lambda) * redundancy
            } else {
                relevance
            };
            SelectionDecision { fact_id: id, relevance, redundancy, utility, packed: false }
        };
        let (index, best) = self
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, decision(c)))
            .max_by(|(_, a), (_, b)| a.utility.total_cmp(&b.utility).then(b.fact_id.cmp(&a.fact_id)))?;
        self.candidates.remove(index);
        Some(best)
    }

    fn similarity(&self, a: i64, b: i64) -> f64 {
        match (self.vectors.get(&a), self.vectors.get(&b)) {
            (Some(a), Some(b)) => {
                a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum::<f64>().clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn selector(mode: SelectionMode) -> Selector {
        Selector::new(
            mode,
            0.7,
            vec![(1, 1.0), (2, 0.99), (3, 0.85)],
            HashMap::from([(1, vec![1.0, 0.0]), (2, vec![1.0, 0.0]), (3, vec![0.0, 1.0])]),
        )
    }
    #[test]
    fn mmr_selects_complementary_evidence_at_the_same_item_limit() {
        let mut relevance = selector(SelectionMode::Relevance);
        let mut mmr = selector(SelectionMode::Mmr);
        assert_eq!(relevance.next(&[]).unwrap().fact_id, 1);
        assert_eq!(mmr.next(&[]).unwrap().fact_id, 1);
        assert_eq!(relevance.next(&[1]).unwrap().fact_id, 2);
        assert_eq!(mmr.next(&[1]).unwrap().fact_id, 3);
    }
    #[test]
    fn rejected_evidence_does_not_suppress_later_candidates() {
        let mut mmr = selector(SelectionMode::Mmr);
        assert_eq!(mmr.next(&[]).unwrap().fact_id, 1);
        // Candidate 1 did not fit: the packed set is still empty.
        assert_eq!(mmr.next(&[]).unwrap().fact_id, 2);
    }
    #[test]
    fn ties_are_stable_and_occurrence_questions_keep_relevance_order() {
        assert_eq!(SelectionMode::Mmr.effective(Intent::Ordering, "How many visits?"), SelectionMode::Relevance);
        let mut s = Selector::new(SelectionMode::Mmr, 0.7, vec![(7, 1.0), (3, 1.0)], HashMap::new());
        assert_eq!(s.next(&[]).unwrap().fact_id, 3);
        assert_eq!(s.next(&[3]).unwrap().fact_id, 7);
        assert!(s.next(&[3, 7]).is_none());
    }
}
