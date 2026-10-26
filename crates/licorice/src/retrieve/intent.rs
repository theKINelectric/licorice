//! Temporal intent of a question (SPEC §9.2, §9.4), found by rules, not a model.

use crate::temporal::{self, Day, Interval};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Intent {
    /// No time cue: a mild recency prior (CogniRank, weighted; see `temporal_factor`).
    None,
    /// An explicit time window ("last weekend", "in May 2023").
    Window(Interval),
    /// About the present state ("currently", "still", "now").
    Current,
    /// "When did …": the date of a past event; no recency decay.
    When,
    /// Order or counts across time ("first", "how many times", "how long"); no decay.
    Ordering,
}

const CURRENT: [&str; 6] = ["currently", "now", "still", "nowadays", "anymore", "presently"];
const ORDERING: [&str; 7] = ["first", "before", "after", "earlier", "later", "latest", "recent"];

impl Intent {
    pub fn of(question: &str, at: Day) -> Intent {
        if let Some((interval, _)) = temporal::find(question, at) {
            return Intent::Window(interval);
        }
        let q = question.to_lowercase();
        let words: Vec<&str> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
        let has = |w: &str| words.contains(&w);
        if Self::needs_occurrences(question) {
            return Intent::Ordering;
        }
        if words.first() == Some(&"when")
            || q.contains("what date")
            || q.contains("which year")
            || q.contains("what year")
        {
            return Intent::When;
        }
        if CURRENT.iter().any(|w| has(w)) || q.contains("these days") || q.contains("at the moment") {
            return Intent::Current;
        }
        Intent::None
    }

    /// Selection needs question shape independently of the time window used for
    /// ranking: "how many visits in May" still needs every occurrence.
    pub(crate) fn needs_occurrences(question: &str) -> bool {
        let q = question.to_lowercase();
        let words: Vec<&str> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
        q.contains("how many")
            || q.contains("how often")
            || q.contains("how long")
            || ORDERING.iter().any(|w| words.contains(w))
    }

    /// T(f) in SPEC §9.4. `occurred` is the fact's event time if known, `recorded` when it
    /// was said; both Unix milliseconds.
    ///
    /// With no time cue, CogniRank's exp(−(Δτ/τ̂)^k) is blended in with `recency_weight`
    /// w, giving 1 − w·(1 − CogniRank), between 1 − w and 1. With k = 0.1 CogniRank
    /// is 1.0 at Δτ = 0 but already ~0.5 at two days, so unweighted (w = 1, LiCoMemory) it
    /// lifts whatever was said last over everything else (docs/eval/locomo-conv-26.md).
    pub(crate) fn temporal_factor(
        &self,
        occurred: Option<i64>,
        recorded: i64,
        at: i64,
        median_gap: f64,
        k: f64,
        recency_weight: f64,
    ) -> f64 {
        let day_ms = 86_400_000.0;
        match self {
            Intent::Window(i) => {
                let t = occurred.unwrap_or(recorded);
                let (from, to) = (i.from.start_ms(), i.to.end_ms());
                if (from..=to).contains(&t) {
                    1.0
                } else {
                    let distance = if t < from { from - t } else { t - to } as f64 / day_ms;
                    let sigma = (((to - from) as f64 / day_ms) / 2.0).max(1.0);
                    (-distance / sigma).exp()
                }
            }
            Intent::When => {
                if occurred.is_some() {
                    1.0
                } else {
                    0.9
                }
            }
            Intent::Ordering | Intent::Current => 1.0,
            Intent::None => {
                let gap = ((at - occurred.unwrap_or(recorded)) as f64 / day_ms).max(0.0);
                let cognirank = (-(gap / median_gap).powf(k)).exp();
                1.0 - recency_weight * (1.0 - cognirank)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at() -> Day {
        Day::from_ymd(2023, 10, 22).unwrap()
    }

    #[test]
    fn intents() {
        assert!(matches!(Intent::of("What did Caroline do last weekend?", at()), Intent::Window(_)));
        assert_eq!(Intent::of("When did Caroline go to the LGBTQ support group?", at()), Intent::When);
        assert_eq!(Intent::of("How many times has Melanie gone to the beach?", at()), Intent::Ordering);
        assert_eq!(Intent::of("Where does Caroline currently work?", at()), Intent::Current);
        assert_eq!(Intent::of("What is Caroline's identity?", at()), Intent::None);
    }

    #[test]
    fn window_factor_peaks_inside() {
        let i = temporal::resolve("May 2023", at()).unwrap();
        let w = Intent::Window(i);
        let inside = Day::from_ymd(2023, 5, 7).unwrap().start_ms();
        let near = Day::from_ymd(2023, 6, 2).unwrap().start_ms();
        let far = Day::from_ymd(2023, 9, 1).unwrap().start_ms();
        assert_eq!(w.temporal_factor(Some(inside), inside, 0, 1.0, 0.1, 1.0), 1.0);
        let (n, f) = (
            w.temporal_factor(Some(near), near, 0, 1.0, 0.1, 1.0),
            w.temporal_factor(Some(far), far, 0, 1.0, 0.1, 1.0),
        );
        assert!(n < 1.0 && f < n, "{n} {f}");
    }

    #[test]
    fn recency_prior_is_bounded_by_its_weight() {
        let at = Day::from_ymd(2023, 10, 22).unwrap().start_ms();
        let day = 86_400_000;
        let t = |gap_days: i64, w: f64| Intent::None.temporal_factor(None, at - gap_days * day, at, 70.0, 0.1, w);
        assert_eq!(t(0, 0.2), 1.0);
        for gap in [1, 2, 60, 365] {
            assert!(t(gap, 0.2) >= 0.8 && t(gap, 0.2) < 1.0, "{gap}: {}", t(gap, 0.2));
            assert!(t(gap, 0.2) > t(gap, 1.0));
        }
        assert!((t(2, 1.0) - 0.4962).abs() < 1e-3, "w = 1 is CogniRank: {}", t(2, 1.0));
        assert_eq!(t(60, 0.0), 1.0, "w = 0 turns recency off");
    }
}
