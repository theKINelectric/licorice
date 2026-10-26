//! Retrieval (SPEC §9, default mode): analyse the question, gather candidates from six
//! lists, score with session, semantic, lexical and temporal evidence, and assemble a
//! chronological, cited context. No model call is made here.

mod graph;
mod intent;
pub use graph::GraphPath;
use licorice_providers::rerank::{Candidate, Ranked, RerankContext, Reranker};
pub mod complementary;
pub mod cross_session;
mod packing;
mod selection;
pub mod source_aware;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryMode {
    #[default]
    Sessions,
    Themes,
}

use std::collections::{HashMap, HashSet};

use licorice_providers::embed::{EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use rusqlite::params_from_iter;
use serde::{Deserialize, Serialize};

pub use intent::Intent;
use packing::ContextPack;
pub use packing::{BudgetOmissions, ContextUsage};
use selection::Selector;
pub use selection::{SelectionMode, SelectionTrace};

use crate::ingest::normalize;
use crate::store::{Filter, Index, LexicalKind, Store, VectorKind};
use crate::temporal::{Day, Interval, Precision};
use crate::{Error, Result};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrieveConfig {
    /// Entire rendered memory, including headings/dates/citations, counted by the
    /// selected tokenizer. Question/system/output tokens are separate.
    pub budget_tokens: usize,
    pub selection: SelectionMode,
    /// Bounded pool shared by relevance and MMR controls, before packing.
    pub selection_candidates: usize,
    /// MMR relevance weight in [0,1]; the remainder penalizes redundancy.
    pub mmr_lambda: f64,
    pub linked_entities: usize,
    /// Optional declared source speaker for English first-person references.
    /// Only an existing, uniquely resolved alias is linked; no entity is created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_speaker: Option<String>,
    pub final_facts: usize,
    pub chunks: usize,
    pub summaries: usize,
    /// Optional query-only summaries packed before facts/excerpts. Includes the
    /// memory/session headers; unused allowance is shared by both selectors.
    /// None retains the original fact-weighted, summaries-last policy.
    pub fixed_summary_tokens: Option<usize>,
    pub summary_mode: SummaryMode,
    /// Session-score weight in the semantic score (LiCoMemory: 0.2).
    pub w_session: f64,
    /// Fact-score weight (LiCoMemory: 0.8).
    pub w_fact: f64,
    /// Share of the fact score taken from cosine similarity; the rest is rank fusion.
    pub alpha: f64,
    pub rrf_k: f64,
    /// Opt-in bounded graph candidate expansion, 0..=2 hops.
    pub graph_hops: usize,
    pub graph_candidates: usize,
    /// Pack each expanded fact with its connecting facts as one complete unit.
    pub connected_evidence: bool,
    /// Experimental: keep ongoing states compatible with an explicit time window.
    /// Only recorded ends and exact-day replacement starts are hard bounds;
    /// unknown/coarse starts and ordering/comparison queries are left untouched.
    pub filter_state_window: bool,
    /// Omit redundant batch/session hashes from excerpt headings. Full turn
    /// citations, source text and structured session metadata remain intact.
    pub compact_excerpt_headers: bool,
    /// Maximum query/candidate scores requested from an injected reranker.
    pub rerank_candidates: usize,
    /// Optional reranking may fall back to the unchanged native order, with an error trace.
    pub rerank_fallback: bool,
    /// CogniRank exponent for the recency factor when the question has no time intent.
    pub recency_k: f64,
    /// How much that recency factor counts: the factor ranges over [1 − w, 1]. 1.0 is
    /// LiCoMemory's CogniRank (parity mode); 0.0 turns recency off.
    pub recency_weight: f64,
    pub superseded_current: f64,
    pub superseded_other: f64,
    pub vector_facts: usize,
    pub entity_facts: usize,
    pub lexical_facts: usize,
    pub window_facts: usize,
    pub chunk_candidates: usize,
}

impl Default for RetrieveConfig {
    fn default() -> Self {
        RetrieveConfig {
            budget_tokens: 2000,
            selection: SelectionMode::Relevance,
            selection_candidates: 60,
            mmr_lambda: 0.7,
            linked_entities: 8,
            query_speaker: None,
            final_facts: 20,
            chunks: 10,
            summaries: 2,
            fixed_summary_tokens: None,
            summary_mode: SummaryMode::Sessions,
            w_session: 0.2,
            w_fact: 0.8,
            alpha: 0.5,
            rrf_k: 60.0,
            graph_hops: 0,
            graph_candidates: 60,
            connected_evidence: false,
            filter_state_window: false,
            compact_excerpt_headers: false,
            rerank_candidates: 24,
            rerank_fallback: false,
            recency_k: 0.1,
            recency_weight: 0.2,
            superseded_current: 0.35,
            superseded_other: 0.8,
            vector_facts: 100,
            entity_facts: 500,
            lexical_facts: 50,
            window_facts: 100,
            chunk_candidates: 30,
        }
    }
}

/// A question asked at a point in time.
#[derive(Clone, Debug)]
pub struct Query {
    pub text: String,
    /// Unix milliseconds; relative time words in the question resolve against it.
    pub at: i64,
}

#[derive(Clone, Debug)]
pub struct RetrievedFact {
    pub id: i64,
    pub statement: String,
    /// When it happened if known, otherwise when it was said.
    pub day: Day,
    pub dated: bool,
    /// When it happened, at the precision it was resolved to ("last summer" is a
    /// season, not its first day); `None` if undated.
    pub occurred: Option<Interval>,
    pub superseded: bool,
    /// Exclusive end of a source-backed state validity interval.
    pub valid_until: Option<Day>,
    pub session: String,
    pub evidence_turns: Vec<String>,
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct RetrievedChunk {
    pub id: i64,
    pub session: String,
    pub day: Day,
    /// (turn id, speaker, text)
    pub turns: Vec<(String, String, String)>,
}

#[derive(Clone, Debug)]
pub struct RetrievedSession {
    pub session: String,
    pub day: Day,
    pub summary: String,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FixedSummaryTrace {
    pub budget_tokens: usize,
    /// Fully rendered prefix, including the common memory/session headings.
    pub tokens: usize,
    pub session_ids: Vec<String>,
    pub mode: SummaryMode,
    pub theme_ids: Vec<i64>,
}

/// What retrieval returns: the pieces, the assembled context, and the turns it cites.
#[derive(Clone, Debug)]
pub struct Retrieved {
    pub intent: Intent,
    pub linked_entities: Vec<String>,
    pub facts: Vec<RetrievedFact>,
    pub chunks: Vec<RetrievedChunk>,
    pub sessions: Vec<RetrievedSession>,
    pub context: String,
    /// Every turn id the context draws on (fact evidence and excerpts), for recall metrics.
    pub cited_turns: Vec<String>,
    pub context_usage: ContextUsage,
    pub budget_omissions: BudgetOmissions,
    pub selection: SelectionTrace,
    pub fixed_summary: Option<FixedSummaryTrace>,
    pub graph_paths: Vec<GraphPath>,
    pub reranking: Option<Vec<Ranked>>,
    pub rerank_error: Option<String>,
}

pub struct Retriever<'a> {
    store: &'a Store,
    embedder: &'a dyn Embedder,
    cfg: RetrieveConfig,
    counter: &'a dyn TokenCounter,
    reranker: Option<&'a dyn Reranker>,
    context_prefix: &'a str,
}

struct FactRow {
    statement: String,
    retracted: bool,
    valid_until: Option<i64>,
    time_kind: String,
    superseded: bool,
    occurred: Option<i64>,
    occurred_to: Option<i64>,
    precision: Option<String>,
    recorded: i64,
    session_id: Option<i64>,
    session: String,
    evidence: Vec<String>,
}

impl FactRow {
    fn interval(&self) -> Option<Interval> {
        let from = Day::from_ms(self.occurred?);
        let precision = self.precision.as_deref().and_then(Precision::parse).unwrap_or(Precision::Day);
        let to = self.occurred_to.map(Day::from_ms).filter(|to| *to >= from).unwrap_or(from);
        Some(Interval { from, to, precision })
    }
}

impl<'a> Retriever<'a> {
    pub fn new(store: &'a Store, embedder: &'a dyn Embedder, cfg: RetrieveConfig) -> Retriever<'a> {
        Retriever { store, embedder, cfg, counter: &Cl100kCounter, reranker: None, context_prefix: "" }
    }

    /// Admit caller-selected continuity before evidence, inside the same full
    /// rendered budget. Empty retains the byte-identical legacy retrieval path.
    pub fn with_context_prefix(mut self, prefix: &'a str) -> Self {
        self.context_prefix = prefix;
        self
    }

    pub fn with_reranker(mut self, reranker: &'a dyn Reranker) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn with_token_counter(mut self, counter: &'a dyn TokenCounter) -> Self {
        self.counter = counter;
        self
    }

    pub async fn retrieve(&self, query: &Query) -> Result<Retrieved> {
        let cfg = &self.cfg;
        if cfg.query_speaker.as_ref().is_some_and(|name| name.trim().is_empty()) {
            return Err(Error::InvalidInput("query_speaker must be a nonempty declared alias".into()));
        }
        if cfg.selection_candidates == 0 || !cfg.mmr_lambda.is_finite() || !(0.0..=1.0).contains(&cfg.mmr_lambda) {
            return Err(Error::InvalidInput(
                "selection_candidates must be positive and mmr_lambda must be finite in [0,1]".into(),
            ));
        }
        if cfg.fixed_summary_tokens.is_some_and(|budget| budget == 0 || budget > cfg.budget_tokens) {
            return Err(Error::InvalidInput(
                "fixed summary allowance must be positive and fit the total context budget".into(),
            ));
        }
        if cfg.summary_mode == SummaryMode::Themes && cfg.fixed_summary_tokens.is_none() {
            return Err(Error::InvalidInput("theme selection requires a fixed summary allowance".into()));
        }
        if cfg.graph_hops > 2
            || cfg.graph_candidates > 100
            || cfg.graph_candidates == 0
            || cfg.rerank_candidates == 0
            || cfg.rerank_candidates > 96
        {
            return Err(Error::InvalidInput(
                "graph hops must be 0..=2, graph candidates 1..=100 and reranker candidates 1..=96".into(),
            ));
        }
        if cfg.connected_evidence && cfg.graph_hops == 0 {
            return Err(Error::InvalidInput("connected evidence requires graph expansion".into()));
        }
        let at = Day::from_ms(query.at);
        let mut packed =
            ContextPack::new(at, self.counter, cfg.budget_tokens)?.compact_excerpt_headers(cfg.compact_excerpt_headers);
        if !self.context_prefix.is_empty() {
            packed = packed.with_notice(self.context_prefix)?;
        }
        let intent = Intent::of(&query.text, at);
        let q = self
            .embedder
            .embed(std::slice::from_ref(&query.text), EmbedKind::Query)
            .await
            .map_err(|e| Error::Task(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Task("embedder returned no query vector".into()))?;

        // ---- entity linking (SPEC §9.2) ----------------------------------------------
        let (entity_ids, linked_entities, query_speaker) = self.link_entities(&query.text).await?;

        // ---- candidate facts: L1 vector, L2 entity-incident, L3 lexical, L4 window -----
        let l1 = self.store.knn(VectorKind::Fact, &q, cfg.vector_facts, Filter::None).await?;
        let l2_ids = self.entity_facts(&entity_ids).await?;
        let l3 = self.store.lexical(LexicalKind::Fact, &query.text, cfg.lexical_facts).await?;
        let l4_ids = match intent {
            Intent::Window(i) => self.window_facts(i.from, i.to).await?,
            _ => Vec::new(),
        };
        let graph_paths = graph::expand(self.store, &entity_ids, cfg.graph_hops, cfg.graph_candidates).await?;
        let mut graph_ids: Vec<i64> = graph_paths.keys().copied().collect();
        graph_ids.sort_unstable();
        let mut all: Vec<i64> = l1.iter().map(|h| h.owner).collect();
        all.extend(&l2_ids);
        all.extend(&graph_ids);
        all.extend(l3.iter().map(|h| h.owner));
        all.extend(&l4_ids);
        let mut seen = HashSet::new();
        all.retain(|id| seen.insert(*id));
        // Exact cosine for every candidate (L2–L4 arrive without one).
        let cos: HashMap<i64, f64> = self
            .store
            .knn(VectorKind::Fact, &q, all.len().max(1), Filter::Owners(all.clone()))
            .await?
            .into_iter()
            .map(|h| (h.owner, h.score))
            .collect();
        let by_cos = |ids: &[i64]| -> Vec<i64> {
            let mut v: Vec<i64> = ids.to_vec();
            v.sort_by(|a, b| cos.get(b).unwrap_or(&-1.0).total_cmp(cos.get(a).unwrap_or(&-1.0)));
            v
        };
        let lists: Vec<Vec<i64>> = vec![
            l1.iter().map(|h| h.owner).collect(),
            by_cos(&l2_ids),
            l3.iter().map(|h| h.owner).collect(),
            by_cos(&l4_ids),
            by_cos(&graph_ids),
        ];
        let mut fused: HashMap<i64, f64> = HashMap::new();
        for list in &lists {
            for (rank, id) in list.iter().enumerate() {
                *fused.entry(*id).or_default() += 1.0 / (cfg.rrf_k + rank as f64 + 1.0);
            }
        }
        let max_fused = fused.values().cloned().fold(0.0, f64::max).max(f64::MIN_POSITIVE);

        // ---- sessions (L5) ------------------------------------------------------------
        let session_scores = self.session_scores(&q).await?;

        // Freeze summaries from query-only scores before either selector chooses
        // facts. Both modes start with the same prefix and remaining allowance.
        let fixed_summary = if let Some(budget_tokens) = cfg.fixed_summary_tokens {
            let mut prefix =
                ContextPack::new(at, self.counter, budget_tokens)?.compact_excerpt_headers(cfg.compact_excerpt_headers);
            if !self.context_prefix.is_empty() {
                prefix = prefix.with_notice(self.context_prefix)?;
            }
            let mut theme_ids = Vec::new();
            if cfg.summary_mode == SummaryMode::Themes {
                let mut per_session: HashMap<String, usize> = HashMap::new();
                for (id, session) in self.theme_rows(&q).await? {
                    let count = per_session.get(&session.session).copied().unwrap_or(0);
                    if count >= 2 || (count == 0 && per_session.len() >= cfg.summaries) {
                        continue;
                    }
                    let ext = session.session.clone();
                    if prefix.theme(id, session)? {
                        *per_session.entry(ext).or_default() += 1;
                        theme_ids.push(id);
                    }
                }
            } else {
                for session in self.session_rows(&session_scores, &HashMap::new()).await? {
                    if prefix.sessions.len() >= cfg.summaries {
                        break;
                    }
                    prefix.session(session)?;
                }
            }
            let trace = FixedSummaryTrace {
                budget_tokens,
                mode: cfg.summary_mode,
                theme_ids,
                tokens: prefix.usage()?.tokens,
                session_ids: prefix.sessions.iter().map(|s| s.session.clone()).collect(),
            };
            packed.omissions.session_ids.extend(prefix.omissions.session_ids);
            packed.omissions.theme_ids.extend(prefix.omissions.theme_ids);
            for session in prefix.sessions {
                packed.session(session)?;
            }
            Some(trace)
        } else {
            None
        };

        // ---- score facts --------------------------------------------------------------
        let rows = self.fact_rows(&all).await?;
        let state_window = if cfg.filter_state_window && !Intent::needs_occurrences(&query.text) {
            if let Intent::Window(window) = intent { Some(window) } else { None }
        } else {
            None
        };
        let state_starts = if state_window.is_some() { self.exact_state_starts(&all).await? } else { HashMap::new() };
        let time_of = |r: &FactRow| r.occurred.unwrap_or(r.recorded);
        let median_gap = {
            let mut gaps: Vec<f64> =
                rows.values().map(|r| ((query.at - time_of(r)) as f64 / 86_400_000.0).max(0.0)).collect();
            gaps.sort_by(f64::total_cmp);
            gaps.get(gaps.len() / 2).copied().unwrap_or(1.0).max(1.0)
        };
        let mut scored: Vec<(i64, f64)> = all
            .iter()
            .filter_map(|id| {
                let r = rows.get(id)?;
                if r.retracted {
                    return None;
                }
                if r.time_kind == "ongoing"
                    && state_window.is_some_and(|window| {
                        r.valid_until.is_some_and(|end| end <= window.from.start_ms())
                            || state_starts.get(id).is_some_and(|start| *start > window.to.end_ms())
                    })
                {
                    return None;
                }
                if r.time_kind == "ongoing"
                    && intent == Intent::Current
                    && (time_of(r) > query.at || r.valid_until.is_some_and(|until| until <= query.at))
                {
                    return None;
                }
                let st = cos.get(id).copied().unwrap_or(0.0).max(0.0);
                let f = fused.get(id).copied().unwrap_or(0.0) / max_fused;
                let ss = r.session_id.and_then(|s| session_scores.get(&s)).copied().unwrap_or(0.0).max(0.0);
                let semantic = cfg.w_session * ss + cfg.w_fact * (cfg.alpha * st + (1.0 - cfg.alpha) * f);
                let t = intent.temporal_factor(
                    r.occurred,
                    r.recorded,
                    query.at,
                    median_gap,
                    cfg.recency_k,
                    cfg.recency_weight,
                );
                let v = match (r.superseded, &intent) {
                    (false, _) => 1.0,
                    (true, Intent::Current) => cfg.superseded_current,
                    (true, _) => cfg.superseded_other,
                };
                Some((*id, semantic * t * v))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        // Ranking-pool truncation is not a truth/temporal eligibility decision.
        // A selected graph endpoint may need a valid, lower-ranked bridge.
        let eligible_scores: HashMap<i64, f64> = scored.iter().copied().collect();
        scored.truncate(cfg.selection_candidates);
        let mut rerank_error = None;
        let reranking = if let Some(reranker) = self.reranker {
            let candidates: Vec<_> = scored
                .iter()
                .take(cfg.rerank_candidates)
                .map(|(id, _)| {
                    let row = &rows[id];
                    Candidate {
                        id: *id,
                        text: format!(
                            "{} [event date: {}; stated: {}; known superseded by: {}]",
                            row.statement,
                            row.interval().map(|interval| interval.human()).unwrap_or_else(|| "undated".into()),
                            Day::from_ms(row.recorded).iso(),
                            row.valid_until
                                .map(|ms| Day::from_ms(ms).iso())
                                .unwrap_or_else(|| "not established".into())
                        ),
                    }
                })
                .collect();
            let result = reranker
                .rank_with_context(
                    &query.text,
                    &candidates,
                    &RerankContext { reference_date_utc: at.iso(), speaker: query_speaker },
                )
                .await
                .and_then(|rows| licorice_providers::rerank::validate(&candidates, rows));
            let ranked = match result {
                Ok(rows) => Some(rows),
                Err(error) if cfg.rerank_fallback => {
                    rerank_error = Some(error.to_string());
                    None
                }
                Err(error) => return Err(Error::Task(error.to_string())),
            };
            if let Some(ranked) = ranked {
                // Map the new order to strictly descending scores above the unranked
                // tail. Equal native scores must not undo reranking through ID ties.
                let floor = scored.get(ranked.len()).map(|(_, s)| *s).unwrap_or(0.);
                let top = scored.first().map(|(_, s)| *s).unwrap_or(0.).max(floor + 1e-6);
                let step = (top - floor) / (ranked.len() + 1) as f64;
                for (index, row) in ranked.iter().enumerate() {
                    scored[index] = (row.id, top - index as f64 * step);
                }
                Some(ranked)
            } else {
                None
            }
        } else {
            None
        };
        let mode = cfg.selection.effective(intent, &query.text);
        let vectors = if mode == SelectionMode::Mmr {
            let mut vector_ids: Vec<_> = scored.iter().map(|(id, _)| *id).collect();
            if cfg.connected_evidence {
                vector_ids
                    .extend(graph_paths.values().flatten().filter(|id| eligible_scores.contains_key(id)).copied());
                vector_ids.sort_unstable();
                vector_ids.dedup();
            }
            self.fact_vectors(&vector_ids).await?
        } else {
            HashMap::new()
        };
        let scores: HashMap<i64, f64> = scored.iter().copied().collect();
        let mut selection =
            SelectionTrace { requested: cfg.selection, effective: mode, candidates: scored.len(), considered: vec![] };
        let mut selector = Selector::new(mode, cfg.mmr_lambda, scored, vectors);
        while packed.facts.len() < cfg.final_facts {
            let Some(mut decision) = selector.next(&packed.facts.iter().map(|f| f.id).collect::<Vec<_>>()) else {
                break;
            };
            let id = decision.fact_id;
            if packed.facts.iter().any(|f| f.id == id) {
                continue;
            }
            let make_fact = |id: i64| {
                let r = &rows[&id];
                RetrievedFact {
                    id,
                    statement: r.statement.clone(),
                    day: Day::from_ms(time_of(r)),
                    dated: r.occurred.is_some(),
                    occurred: r.interval(),
                    superseded: r.superseded,
                    valid_until: r.valid_until.map(Day::from_ms),
                    session: r.session.clone(),
                    evidence_turns: r.evidence.clone(),
                    score: scores.get(&id).copied().unwrap_or(eligible_scores[&id]),
                }
            };
            let ids = if cfg.connected_evidence {
                graph_paths.get(&id).cloned().unwrap_or_else(|| vec![id])
            } else {
                vec![id]
            };
            let ids: Vec<_> = ids.into_iter().filter(|id| !packed.facts.iter().any(|f| f.id == *id)).collect();
            // A path cannot smuggle an ineligible/filtered state back into context.
            if ids.iter().all(|id| eligible_scores.contains_key(id))
                && packed.facts.len() + ids.len() <= cfg.final_facts
            {
                decision.packed = if cfg.connected_evidence {
                    let source_ids = self.evidence_chunks(&ids).await?;
                    let sources = self
                        .chunk_rows(&source_ids)
                        .await?
                        .into_iter()
                        .filter(|c| !packed.chunks.iter().any(|old| old.id == c.id))
                        .collect::<Vec<_>>();
                    let complete_sources = ids.iter().flat_map(|id| rows[id].evidence.iter()).all(|turn| {
                        packed.chunks.iter().chain(sources.iter()).any(|chunk| chunk.turns.iter().any(|t| &t.0 == turn))
                    });
                    if complete_sources && packed.chunks.len() + sources.len() <= cfg.chunks {
                        packed.bundle(ids.into_iter().map(make_fact).collect(), sources)?
                    } else {
                        false
                    }
                } else {
                    packed.fact(make_fact(id))?
                };
            }
            selection.considered.push(decision);
        }

        // ---- chunks: evidence first, then L6 ------------------------------------------
        let mut chunk_ids: Vec<i64> =
            self.evidence_chunks(&packed.facts.iter().map(|f| f.id).collect::<Vec<_>>()).await?;
        let l6v = self.store.knn(VectorKind::Chunk, &q, cfg.chunk_candidates, Filter::None).await?;
        let l6l = self.store.lexical(LexicalKind::Chunk, &query.text, cfg.chunk_candidates).await?;
        let mut chunk_fused: HashMap<i64, f64> = HashMap::new();
        for list in [&l6v, &l6l] {
            for (rank, h) in list.iter().enumerate() {
                *chunk_fused.entry(h.owner).or_default() += 1.0 / (cfg.rrf_k + rank as f64 + 1.0);
            }
        }
        let mut l6: Vec<(i64, f64)> = chunk_fused.into_iter().collect();
        l6.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        chunk_ids.truncate(cfg.chunk_candidates.max(cfg.chunks));
        for (id, _) in l6 {
            if !chunk_ids.contains(&id) {
                chunk_ids.push(id);
            }
        }
        for chunk in self.chunk_rows(&chunk_ids).await? {
            if packed.chunks.iter().any(|existing| existing.id == chunk.id) {
                continue;
            }
            if packed.chunks.len() >= cfg.chunks {
                break;
            }
            packed.chunk(chunk)?;
        }

        // ---- sessions to show: session score plus the weight of their selected facts ---
        if fixed_summary.is_none() {
            let mut by_session: HashMap<String, f64> = HashMap::new();
            for f in &packed.facts {
                *by_session.entry(f.session.clone()).or_default() += f.score;
            }
            for session in self.session_rows(&session_scores, &by_session).await? {
                if packed.sessions.len() >= cfg.summaries {
                    break;
                }
                packed.session(session)?;
            }
        }

        let context_usage = packed.usage()?;
        let ContextPack { facts, chunks, sessions, context, omissions: budget_omissions, .. } = packed;
        let mut cited: Vec<String> = Vec::new();
        let mut cited_seen = HashSet::new();
        for t in facts
            .iter()
            .flat_map(|f| f.evidence_turns.iter())
            .chain(chunks.iter().flat_map(|c| c.turns.iter().map(|t| &t.0)))
        {
            if cited_seen.insert(t.clone()) {
                cited.push(t.clone());
            }
        }
        Ok(Retrieved {
            intent,
            linked_entities,
            facts,
            chunks,
            sessions,
            context,
            cited_turns: cited,
            context_usage,
            budget_omissions,
            selection,
            fixed_summary,
            graph_paths: {
                let mut paths: Vec<_> =
                    graph_paths.into_iter().map(|(target, facts)| GraphPath { target, facts }).collect();
                paths.sort_by_key(|p| p.target);
                paths
            },
            reranking,
            rerank_error,
        })
    }

    /// Stored fact vectors are normalized. Use the latest live vector per owner
    /// and bound materialization to the selection pool, never the whole space.
    async fn fact_vectors(&self, ids: &[i64]) -> Result<HashMap<i64, Vec<f32>>> {
        let ids = ids.to_vec();
        let dim = self.embedder.dim();
        self.store
            .read(move |conn| {
                let mut out = HashMap::new();
                for batch in ids.chunks(400) {
                    let sql = format!(
                        "SELECT owner,v FROM vec_fact WHERE live=1 AND owner IN ({}) ORDER BY vid",
                        vec!["?"; batch.len()].join(",")
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    let mut rows = stmt.query(params_from_iter(batch))?;
                    while let Some(row) = rows.next()? {
                        let bytes: Vec<u8> = row.get(1)?;
                        if bytes.len() != dim * 4 {
                            return Err(Error::InvalidVector);
                        }
                        let vector: Vec<f32> =
                            bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        if vector.iter().any(|v| !v.is_finite()) {
                            return Err(Error::InvalidVector);
                        }
                        out.insert(row.get(0)?, vector);
                    }
                }
                Ok(out)
            })
            .await
    }

    /// Exact alias matches on the question's 1–4-grams, then fuzzy alias matches.
    async fn link_entities(&self, text: &str) -> Result<(Vec<i64>, Vec<String>, Option<String>)> {
        let first_person = text
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| ["i", "me", "my", "mine", "myself"].iter().any(|pronoun| word.eq_ignore_ascii_case(pronoun)));
        let speaker_alias = if first_person { self.cfg.query_speaker.as_deref().map(normalize) } else { None };
        let words: Vec<String> = normalize(text).split(' ').map(|w| w.trim_end_matches("'s").to_owned()).collect();
        let mut grams: Vec<String> = Vec::new();
        for n in 1..=4 {
            for w in words.windows(n) {
                grams.push(w.join(" "));
            }
        }
        let limit = self.cfg.linked_entities;
        let fuzzy = self.store.lexical(LexicalKind::Alias, text, limit).await?;
        self.store
            .read(move |conn| {
                let mut ids: Vec<i64> = Vec::new();
                let mut speaker_id = None;
                if limit>0 {
                    if let Some(alias)=speaker_alias.as_deref() {
                        let mut statement=conn.prepare("SELECT DISTINCT coalesce(e.merged_into,e.id) FROM entity_aliases a JOIN entities e ON e.id=a.entity_id WHERE a.alias=?1 LIMIT 2")?;
                        let matches=statement.query_map([alias],|row|row.get::<_,i64>(0))?.collect::<Result<Vec<_>,_>>()?;
                        // An ambiguous or absent role alias cannot justify guessing
                        // a referent. Literal/fuzzy linking remains available below.
                        if matches.len()==1 {ids.push(matches[0]);speaker_id=Some(matches[0]);}
                    }
                }
                for batch in grams.chunks(500) {
                    let sql = format!(
                        "SELECT DISTINCT coalesce(e.merged_into, e.id) FROM entity_aliases a JOIN entities e ON e.id = a.entity_id \
                         WHERE a.alias IN ({}) AND length(a.alias) > 2",
                        vec!["?"; batch.len()].join(",")
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    for id in stmt.query_map(params_from_iter(batch), |r| r.get::<_, i64>(0))? {
                        let id = id?;
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
                for h in &fuzzy {
                    if ids.len() >= limit {
                        break;
                    }
                    if !ids.contains(&h.owner) {
                        ids.push(h.owner);
                    }
                }
                ids.truncate(limit);
                let mut names: Vec<String> = Vec::with_capacity(ids.len());
                for id in &ids {
                    names.push(conn.query_row("SELECT name FROM entities WHERE id = ?1", [id], |r| r.get(0))?);
                }
                let speaker = speaker_id.and_then(|actor| ids.iter().position(|id| *id == actor)).map(|index| names[index].clone());
                Ok((ids, names, speaker))
            })
            .await
    }

    async fn entity_facts(&self, entity_ids: &[i64]) -> Result<Vec<i64>> {
        if entity_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids = entity_ids.to_vec();
        let limit = self.cfg.entity_facts as i64;
        self.store
            .read(move |conn| {
                let marks = vec!["?"; ids.len()].join(",");
                let sql = format!(
                    "SELECT id FROM facts WHERE subject_id IN ({marks}) OR object_id IN ({marks}) \
                     ORDER BY recorded_last DESC LIMIT {limit}"
                );
                let mut stmt = conn.prepare(&sql)?;
                let doubled: Vec<i64> = ids.iter().chain(ids.iter()).copied().collect();
                Ok(stmt.query_map(params_from_iter(doubled), |r| r.get(0))?.collect::<Result<_, _>>()?)
            })
            .await
    }

    async fn window_facts(&self, from: Day, to: Day) -> Result<Vec<i64>> {
        let (start, end, limit) = (from.start_ms(), to.end_ms(), self.cfg.window_facts as i64);
        self.store
            .read(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id FROM facts WHERE (occurred_from IS NOT NULL AND occurred_from <= ?2 AND occurred_to >= ?1) \
                     OR (occurred_from IS NULL AND recorded_first BETWEEN ?1 AND ?2) ORDER BY recorded_last DESC LIMIT ?3",
                )?;
                Ok(stmt.query_map(rusqlite::params![start, end, limit], |r| r.get(0))?.collect::<Result<_, _>>()?)
            })
            .await
    }

    /// Mean of the top three key/theme similarities per session (LiCoMemory's Ss).
    async fn session_scores(&self, q: &[f32]) -> Result<HashMap<i64, f64>> {
        let hits = self.store.knn(VectorKind::Key, q, 200, Filter::None).await?;
        if hits.is_empty() {
            return Ok(HashMap::new());
        }
        let ids: Vec<i64> = hits.iter().map(|h| h.owner).collect();
        let owner_session: HashMap<i64, i64> = self
            .store
            .read(move |conn| {
                let sql =
                    format!("SELECT id, session_id FROM summary_keys WHERE id IN ({})", vec!["?"; ids.len()].join(","));
                let mut stmt = conn.prepare(&sql)?;
                Ok(stmt.query_map(params_from_iter(&ids), |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?)
            })
            .await?;
        let mut per: HashMap<i64, Vec<f64>> = HashMap::new();
        for h in hits {
            if let Some(s) = owner_session.get(&h.owner) {
                per.entry(*s).or_default().push(h.score);
            }
        }
        Ok(per
            .into_iter()
            .map(|(s, mut v)| {
                v.sort_by(|a, b| b.total_cmp(a));
                v.truncate(3);
                (s, v.iter().sum::<f64>() / v.len() as f64)
            })
            .collect())
    }

    async fn exact_state_starts(&self, ids: &[i64]) -> Result<HashMap<i64, i64>> {
        let ids = ids.to_vec();
        self.store.read(move |conn| {
            let mut starts=HashMap::new();
            for batch in ids.chunks(400) {
                let sql=format!("SELECT next_fact_id,min(effective_at) FROM fact_updates WHERE kind='replace' AND effective_basis='explicit_day' AND next_fact_id IN ({}) GROUP BY next_fact_id",vec!["?";batch.len()].join(","));
                let mut stmt=conn.prepare(&sql)?;
                for row in stmt.query_map(params_from_iter(batch),|r|Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?)))? {
                    let (id,at)=row?;starts.insert(id,at);
                }
            }
            Ok(starts)
        }).await
    }

    async fn fact_rows(&self, ids: &[i64]) -> Result<HashMap<i64, FactRow>> {
        let ids = ids.to_vec();
        self.store
            .read(move |conn| {
                let mut out = HashMap::new();
                for batch in ids.chunks(400) {
                    let sql = format!(
                        "SELECT f.id, f.statement, f.status, f.occurred_from, f.recorded_last, \
                           (SELECT t.session_id FROM fact_evidence fe JOIN turns t ON t.id = fe.turn_id WHERE fe.fact_id = f.id ORDER BY t.id LIMIT 1), \
                           (SELECT s.ext_id FROM fact_evidence fe JOIN turns t ON t.id = fe.turn_id JOIN sessions s ON s.id = t.session_id \
                              WHERE fe.fact_id = f.id ORDER BY t.id LIMIT 1), \
                           (SELECT json_group_array(ext_id) FROM (SELECT t.ext_id FROM fact_evidence fe JOIN turns t ON t.id = fe.turn_id WHERE fe.fact_id = f.id ORDER BY t.id)), \
                           f.occurred_to, f.time_precision, f.valid_until, f.time_kind \
                         FROM facts f WHERE f.id IN ({})",
                        vec!["?"; batch.len()].join(",")
                    );
                    let mut stmt = conn.prepare(&sql)?;
                    let rows = stmt.query_map(params_from_iter(batch), |r| {
                        let evidence: String = r.get(7)?;
                        let evidence:Vec<String>=serde_json::from_str(&evidence).map_err(|e|rusqlite::Error::FromSqlConversionFailure(7,rusqlite::types::Type::Text,Box::new(e)))?;
                        Ok((
                            r.get::<_, i64>(0)?,
                            FactRow {
                                statement: r.get(1)?,
                                retracted:r.get::<_,String>(2)?=="retracted",
                                valid_until: r.get(10)?,
                                time_kind: r.get(11)?,
                                superseded: r.get::<_, String>(2)? != "active",
                                occurred: r.get(3)?,
                                occurred_to: r.get(8)?,
                                precision: r.get(9)?,
                                recorded: r.get(4)?,
                                session_id: r.get(5)?,
                                session: r.get::<_, Option<String>>(6)?.unwrap_or_default(),
                                evidence,
                            },
                        ))
                    })?;
                    for row in rows {
                        let (id, row) = row?;
                        out.insert(id, row);
                    }
                }
                Ok(out)
            })
            .await
    }

    async fn evidence_chunks(&self, fact_ids: &[i64]) -> Result<Vec<i64>> {
        let ids = fact_ids.to_vec();
        self.store
            .read(move |conn| {
                let mut out: Vec<i64> = Vec::new();
                let mut stmt = conn.prepare_cached(
                    "SELECT DISTINCT chunk_id FROM fact_evidence WHERE fact_id = ?1 AND chunk_id IS NOT NULL ORDER BY chunk_id",
                )?;
                for id in &ids {
                    for c in stmt.query_map([id], |r| r.get::<_, i64>(0))? {
                        let c = c?;
                        if !out.contains(&c) {
                            out.push(c);
                        }
                    }
                }
                Ok(out)
            })
            .await
    }

    async fn chunk_rows(&self, ids: &[i64]) -> Result<Vec<RetrievedChunk>> {
        let ids = ids.to_vec();
        self.store
            .read(move |conn| {
                let mut out = Vec::with_capacity(ids.len());
                for id in &ids {
                    let (session, ts, first, last): (String, i64, i64, i64) = conn.query_row(
                        "SELECT s.ext_id, coalesce(c.ts, 0), c.first_turn, c.last_turn FROM chunks c JOIN sessions s ON s.id = c.session_id \
                         WHERE c.id = ?1",
                        [id],
                        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                    )?;
                    let mut stmt = conn.prepare_cached(
                        "SELECT coalesce(ext_id, ''), speaker, text FROM turns WHERE id BETWEEN ?1 AND ?2 ORDER BY id",
                    )?;
                    let turns = stmt
                        .query_map([first, last], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                        .collect::<Result<Vec<_>, _>>()?;
                    out.push(RetrievedChunk { id: *id, session, day: Day::from_ms(ts), turns });
                }
                Ok(out)
            })
            .await
    }

    /// Query-only similarity over live theme vectors. Keyword vectors cannot crowd
    /// themes out of the bounded 200-result pool. Owner filters batch SQL bindings.
    async fn theme_rows(&self, q: &[f32]) -> Result<Vec<(i64, RetrievedSession)>> {
        let ids: Vec<i64> = self
            .store
            .read(|conn| {
                let mut stmt = conn.prepare("SELECT id FROM summary_keys WHERE live=1 AND kind='theme' ORDER BY id")?;
                Ok(stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?)
            })
            .await?;
        let hits = self.store.knn(VectorKind::Key, q, 200, Filter::Owners(ids)).await?;
        let ids: Vec<i64> = hits.iter().map(|h| h.owner).collect();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut rows: HashMap<i64, RetrievedSession> = self.store.read(move |conn| {
            let sql = format!("SELECT k.id,s.ext_id,coalesce(s.started_at,0),k.text FROM summary_keys k JOIN sessions s ON s.id=k.session_id WHERE k.live=1 AND k.kind='theme' AND k.id IN ({})", vec!["?"; ids.len()].join(","));
            let mut stmt = conn.prepare(&sql)?;
            Ok(stmt.query_map(params_from_iter(&ids), |r| {
                let session: String = r.get(1)?;
                let day = Day::from_ms(r.get(2)?);
                let text: String = r.get(3)?;
                // Same wrapper as stored session summaries, stripped by assemble.
                let summary = format!("Session {session} ({}): {text}", day.human());
                Ok((r.get(0)?, RetrievedSession {session, day, summary, score:0.0}))
            })?.collect::<Result<_, _>>()?)
        }).await?;
        Ok(hits
            .into_iter()
            .filter_map(|h| {
                rows.remove(&h.owner).map(|mut row| {
                    row.score = h.score;
                    (h.owner, row)
                })
            })
            .collect())
    }

    async fn session_rows(
        &self,
        scores: &HashMap<i64, f64>,
        fact_weight: &HashMap<String, f64>,
    ) -> Result<Vec<RetrievedSession>> {
        let scores = scores.clone();
        let fact_weight = fact_weight.clone();
        let take = self.cfg.summaries.max(60);
        self.store
            .read(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT s.id, s.ext_id, coalesce(s.started_at, 0), sm.text FROM sessions s JOIN summaries sm ON sm.session_id = s.id",
                )?;
                let mut rows: Vec<RetrievedSession> = stmt
                    .query_map([], |r| {
                        let id: i64 = r.get(0)?;
                        let ext: String = r.get(1)?;
                        let score = scores.get(&id).copied().unwrap_or(0.0) + fact_weight.get(&ext).copied().unwrap_or(0.0);
                        Ok(RetrievedSession { session: ext, day: Day::from_ms(r.get(2)?), summary: r.get(3)?, score })
                    })?
                    .collect::<Result<_, _>>()?;
                rows.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.session.cmp(&b.session)));
                rows.truncate(take);
                Ok(rows)
            })
            .await
    }
}

/// The context handed to the answering model (SPEC §9.6): facts in time order, with
/// their dates and sources, then the dialogue excerpts they came from.
fn assemble(
    at: Day,
    sessions: &[RetrievedSession],
    facts: &[RetrievedFact],
    chunks: &[RetrievedChunk],
    compact_excerpt_headers: bool,
) -> String {
    let mut out = format!("## Memory (as of {})\n", at.human());
    if !sessions.is_empty() {
        out.push_str("### Sessions\n");
        for s in sessions {
            let text = s.summary.split_once("): ").map(|(_, t)| t).unwrap_or(&s.summary);
            out.push_str(&format!("- {} [{}] {}\n", s.day.human(), s.session, text));
        }
    }
    if !facts.is_empty() {
        out.push_str("### Facts (chronological)\n");
        let mut ordered: Vec<&RetrievedFact> = facts.iter().collect();
        ordered.sort_by_key(|f| (f.day, f.id));
        for f in ordered {
            let when = match f.occurred {
                Some(interval) => interval.human(),
                None if f.dated => f.day.human(),
                None => format!("said {}", f.day.human()),
            };
            let note = match f.valid_until {
                Some(until) => format!(" (known superseded by {})", until.human()),
                None if f.superseded => " (no longer current)".into(),
                None => String::new(),
            };
            out.push_str(&format!("- {when}: {}{note} [{}]\n", f.statement, f.evidence_turns.join(", ")));
        }
    }
    if !chunks.is_empty() {
        out.push_str("### Excerpts\n");
        let mut ordered: Vec<&RetrievedChunk> = chunks.iter().collect();
        ordered.sort_by_key(|c| c.id);
        for c in ordered {
            if compact_excerpt_headers {
                out.push_str(&format!("({})\n", c.day.human()));
            } else {
                out.push_str(&format!("({}, session {})\n", c.day.human(), c.session));
            }
            for (id, speaker, text) in &c.turns {
                out.push_str(&format!("[{id}] {speaker}: {text}\n"));
            }
        }
    }
    out
}
