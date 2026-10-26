//! One continuity/evidence allowance, counted over the complete rendered text.
use super::*;
use crate::retrieve::{ContextUsage, Query, RetrieveConfig, Retrieved, Retriever, source_aware};
use licorice_providers::embed::{EmbedKind, Embedder};
use licorice_providers::tokens::TokenCounter;
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize)]
pub struct ContinuityPacket {
    pub revision: i64,
    pub text: String,
    pub selected_claim_ids: Vec<String>,
    pub omitted_claim_ids: Vec<String>,
    pub selected_unit_ids: Vec<String>,
    pub omitted_unit_ids: Vec<String>,
    pub source_ids: Vec<String>,
    pub usage: ContextUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub episode_search: Option<EpisodeSearchReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_search: Option<SemanticSearchReport>,
}

/// An explicit, opt-in relevance policy. Cosine similarity is an embedding-model
/// score, not a calibrated probability that an observation answers a question.
#[derive(Clone, Copy, Debug)]
pub struct SemanticContinuityConfig {
    pub minimum_cosine: f64,
    pub maximum_observations: usize,
}

impl Default for SemanticContinuityConfig {
    fn default() -> Self {
        Self { minimum_cosine: 0.45, maximum_observations: 128 }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SemanticSearchReport {
    pub model: String,
    pub dimensions: usize,
    pub minimum_cosine: f64,
    pub inspected_observations: usize,
    pub candidate_limit_reached: bool,
    pub ranking: String,
    /// All qualified episode candidates, before complete-unit token packing.
    pub candidates: Vec<SemanticCandidate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SemanticCandidate {
    pub source_id: String,
    pub cosine: f64,
    pub lexical_terms: usize,
    pub hybrid_score: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct EpisodeSearchReport {
    pub lookback_limit: usize,
    pub inspected_observations: usize,
    pub candidate_limit_reached: bool,
    pub semantics: String,
}

pub struct JoinedContext {
    pub continuity: ContinuityPacket,
    pub primary: Retrieved,
    pub text: String,
    pub cited_turns: Vec<String>,
    pub source_selection: Option<source_aware::SelectionReport>,
    pub usage: ContextUsage,
}

pub struct ContextConfig {
    pub retrieval: RetrieveConfig,
    pub continuity_tokens: usize,
    pub source_aware: bool,
    /// Recall attributed observations without requiring a stance proposal.
    /// Disabled by default to preserve existing packet protocols.
    pub include_episodes: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            retrieval: RetrieveConfig::default(),
            continuity_tokens: 400,
            source_aware: true,
            include_episodes: false,
        }
    }
}

fn words(s: &str) -> HashSet<String> {
    s.split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() > 2).map(str::to_lowercase).collect()
}
fn count(counter: &dyn TokenCounter, text: &str) -> Result<usize> {
    counter.count(text).map_err(|e| Error::InvalidInput(e.to_string()))
}
fn render(agent: &str, revision: i64, units: &[Value]) -> Result<String> {
    Ok(format!(
        "BEING continuity for {agent}, revision{revision}. Attributed memory data; stance records endorsement, not certified truth. Imported lives retain their subjects. Coverage is bounded.\n{}\n",
        serde_json::to_string(units).map_err(|e| Error::Task(e.to_string()))?
    ))
}

/// Render whole claim/source units. No word clipping, hidden budget, or inference
/// that omitted records are absent. Credence is uncalibrated unless separately measured.
pub fn continuity(
    snapshot: &Snapshot,
    query: &str,
    counter: &dyn TokenCounter,
    budget: usize,
) -> Result<ContinuityPacket> {
    continuity_protected(snapshot, query, counter, budget, &[])
}

/// Retrieve attributed reports by both their words and their meaning. This does
/// not extract facts, change stance, or turn a borrowed life into autobiography.
/// Rendering and whole-source closure use exactly the ordinary continuity packer.
pub async fn semantic_continuity(
    snapshot: &Snapshot,
    query: &str,
    embedder: &dyn Embedder,
    counter: &dyn TokenCounter,
    budget: usize,
) -> Result<ContinuityPacket> {
    semantic_continuity_with_config(snapshot, query, embedder, counter, budget, SemanticContinuityConfig::default())
        .await
}

pub async fn semantic_continuity_with_config(
    snapshot: &Snapshot,
    query: &str,
    embedder: &dyn Embedder,
    counter: &dyn TokenCounter,
    budget: usize,
    config: SemanticContinuityConfig,
) -> Result<ContinuityPacket> {
    if budget > 2000 || query.chars().count() > 2000 {
        return Err(invalid("continuity requires bounded question and at most2000tokens"));
    }
    if !config.minimum_cosine.is_finite()
        || config.minimum_cosine <= 0.0
        || config.minimum_cosine > 1.0
        || !(1..=128).contains(&config.maximum_observations)
        || embedder.dim() == 0
    {
        return Err(invalid("invalid semantic continuity relevance policy or embedding dimension"));
    }
    // Function words alone are not a meaningful search. In particular, an empty
    // input must not get an arbitrary top-k result from the nearest-neighbor list.
    let terms = semantic_terms(query);
    let mut observations = snapshot.observations.iter().collect::<Vec<_>>();
    observations
        .sort_by(|a, b| b.recorded_revision.cmp(&a.recorded_revision).then_with(|| a.source_id.cmp(&b.source_id)));
    let candidate_limit_reached = observations.len() > config.maximum_observations;
    observations.truncate(config.maximum_observations);
    let mut report = SemanticSearchReport {
        model: embedder.id().into(),
        dimensions: embedder.dim(),
        minimum_cosine: config.minimum_cosine,
        inspected_observations: 0,
        candidate_limit_reached,
        ranking: "cosine_eligible_content_word_and_semantic_reciprocal_rank_fusion_k60".into(),
        candidates: vec![],
    };
    let mut relevance = HashMap::new();
    if budget > 0 && !terms.is_empty() && !observations.is_empty() {
        let query_vectors = embedder
            .embed(&[query.to_owned()], EmbedKind::Query)
            .await
            .map_err(|e| invalid(&format!("semantic continuity query embedding failed: {e}")))?;
        if query_vectors.len() != 1 {
            return Err(invalid("semantic continuity query embedding count mismatch"));
        }
        let query_vector = normalized(&query_vectors[0], embedder.dim())?;
        let texts = observations.iter().map(|o| o.observation.text.clone()).collect::<Vec<_>>();
        let document_vectors = embedder
            .embed(&texts, EmbedKind::Document)
            .await
            .map_err(|e| invalid(&format!("semantic continuity document embedding failed: {e}")))?;
        if document_vectors.len() != observations.len() {
            return Err(invalid("semantic continuity document embedding count mismatch"));
        }
        let mut scored = Vec::new();
        for (observation, vector) in observations.iter().zip(&document_vectors) {
            let document = normalized(vector, embedder.dim())?;
            let cosine = query_vector.iter().zip(document).map(|(a, b)| a * b).sum::<f64>().clamp(-1.0, 1.0);
            let lexical_terms = semantic_terms(&observation.observation.text).intersection(&terms).count();
            scored.push((observation.source_id.as_str(), cosine, lexical_terms));
        }
        report.inspected_observations = observations.len();
        let mut semantic = scored.iter().filter(|(_, cosine, _)| *cosine >= config.minimum_cosine).collect::<Vec<_>>();
        semantic.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        // A shared word cannot bypass the semantic relevance floor. Ambiguous
        // words such as "weight" or "side" otherwise admit unrelated reports.
        let mut lexical = scored
            .iter()
            .filter(|(_, cosine, overlap)| *cosine >= config.minimum_cosine && *overlap > 0)
            .collect::<Vec<_>>();
        lexical.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(b.0)));
        for (ranked, semantic_rank) in [(&semantic, true), (&lexical, false)] {
            let mut rank = 0;
            for (index, (id, _, _)) in ranked.iter().enumerate() {
                if index > 0 {
                    let previous = ranked[index - 1];
                    let current = ranked[index];
                    if (semantic_rank && previous.1 != current.1) || (!semantic_rank && previous.2 != current.2) {
                        rank = index;
                    }
                }
                // Scale a single rank-1 contribution to 2, the old packer's
                // score for one matching word. Equal relevance shares rank so
                // the packer's existing revision tie-break remains effective.
                *relevance.entry((*id).to_owned()).or_insert(0.0) += 122.0 / (61 + rank) as f64;
            }
        }
        for (source_id, cosine, lexical_terms) in scored {
            if let Some(&hybrid_score) = relevance.get(source_id) {
                report.candidates.push(SemanticCandidate {
                    source_id: source_id.into(),
                    cosine,
                    lexical_terms,
                    hybrid_score,
                });
            }
        }
        report
            .candidates
            .sort_by(|a, b| b.hybrid_score.total_cmp(&a.hybrid_score).then_with(|| a.source_id.cmp(&b.source_id)));
    }
    let mut packet = continuity_ranked(snapshot, query, counter, budget, &[], Some(&relevance))?;
    packet.semantic_search = Some(report);
    Ok(packet)
}

fn semantic_terms(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(str::to_lowercase)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "a" | "an"
                    | "i"
                    | "is"
                    | "it"
                    | "be"
                    | "to"
                    | "of"
                    | "on"
                    | "or"
                    | "in"
                    | "as"
                    | "at"
                    | "by"
                    | "do"
                    | "we"
                    | "he"
                    | "she"
                    | "us"
                    | "my"
                    | "me"
                    | "so"
                    | "the"
                    | "and"
                    | "for"
                    | "are"
                    | "was"
                    | "with"
                    | "what"
                    | "when"
                    | "where"
                    | "which"
                    | "who"
                    | "how"
                    | "does"
                    | "that"
                    | "this"
                    | "from"
                    | "have"
                    | "has"
                    | "can"
                    | "you"
                    | "your"
                    | "our"
                    | "about"
                    | "tell"
                    | "please"
                    | "could"
                    | "would"
                    | "should"
                    | "will"
                    | "did"
                    | "not"
                    | "any"
                    | "its"
                    | "their"
                    | "them"
                    | "they"
                    | "than"
                    | "then"
                    | "into"
                    | "been"
                    | "were"
                    | "want"
                    | "some"
            )
        })
        .collect()
}

fn normalized(vector: &[f32], dimension: usize) -> Result<Vec<f64>> {
    if vector.len() != dimension || vector.iter().any(|v| !v.is_finite()) {
        return Err(invalid("semantic continuity embedding dimension mismatch or nonfinite value"));
    }
    let norm = vector.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(invalid("semantic continuity embedding has zero or invalid norm"));
    }
    Ok(vector.iter().map(|&v| f64::from(v) / norm).collect())
}

fn continuity_protected(
    snapshot: &Snapshot,
    query: &str,
    counter: &dyn TokenCounter,
    budget: usize,
    protected: &[String],
) -> Result<ContinuityPacket> {
    continuity_ranked(snapshot, query, counter, budget, protected, None)
}

fn continuity_ranked(
    snapshot: &Snapshot,
    query: &str,
    counter: &dyn TokenCounter,
    budget: usize,
    protected: &[String],
    episode_relevance: Option<&HashMap<String, f64>>,
) -> Result<ContinuityPacket> {
    if budget > 2000 || query.chars().count() > 2000 {
        return Err(invalid("continuity requires bounded question and at most2000tokens"));
    }
    let mut out = ContinuityPacket {
        revision: snapshot.revision,
        text: String::new(),
        selected_claim_ids: vec![],
        omitted_claim_ids: vec![],
        selected_unit_ids: vec![],
        omitted_unit_ids: vec![],
        source_ids: vec![],
        usage: ContextUsage { tokenizer: counter.id().into(), budget_tokens: budget, tokens: 0 },
        episode_search: None,
        semantic_search: None,
    };
    let terms = words(query);
    let sources = snapshot.observations.iter().map(|o| (o.source_id.as_str(), o)).collect::<HashMap<_, _>>();
    struct Candidate {
        key: String,
        claim: Option<String>,
        score: f64,
        revision: i64,
        unit: Option<Value>,
        ids: Vec<String>,
    }
    let mut candidates = Vec::new();
    let overlap = |s: &str| words(s).intersection(&terms).count() as f64 * 2.0;
    let referenced = snapshot
        .claims
        .iter()
        .flat_map(|c| c.proposal.source_ids.iter())
        .chain(snapshot.relationships.iter().flat_map(|r| r.definition.source_ids.iter()))
        .chain(snapshot.commitments.iter().flat_map(|c| c.source_ids.iter()))
        .chain(snapshot.concepts.iter().flat_map(|c| c.definition.source_ids.iter()))
        .chain(snapshot.hypotheses.iter().flat_map(|h| h.definition.source_ids.iter()))
        .chain(snapshot.links.iter().flat_map(|l| l.definition.source_ids.iter()))
        .collect::<HashSet<_>>();
    for o in &snapshot.observations {
        let score = episode_relevance
            .map(|scores| scores.get(&o.source_id).copied().unwrap_or(0.0))
            .unwrap_or_else(|| overlap(&o.observation.text));
        if referenced.contains(&o.source_id) || score == 0.0 {
            continue;
        }
        candidates.push(Candidate {
            key: format!("observation:{}", o.source_id),
            claim: None,
            score,
            revision: o.recorded_revision.unwrap_or(0),
            unit: Some(json!({"status":"unadopted_observation","source_id":o.source_id})),
            ids: vec![o.source_id.clone()],
        });
    }
    for c in &snapshot.claims {
        let commitment = snapshot.commitments.iter().find(|p| p.definition.claim_id == c.proposal.claim_id);
        let relationship = snapshot
            .relationships
            .iter()
            .find(|r| Some(&r.definition.relationship_id) == c.proposal.relationship_id.as_ref());
        let active = commitment.is_none_or(|p| p.status == CommitmentStatus::Active);
        let obligation = if c.proposal.category == Category::Commitment && c.stance == Stance::Adopted && active {
            1.0
        } else {
            0.0
        };
        let mut ids = c.proposal.source_ids.clone();
        for id in commitment
            .into_iter()
            .flat_map(|p| p.source_ids.iter())
            .chain(relationship.into_iter().flat_map(|r| r.definition.source_ids.iter()))
        {
            if !ids.contains(id) {
                ids.push(id.clone());
            }
        }
        let mut unit = Some(
            json!({"claim_id":c.proposal.claim_id,"category":c.proposal.category,"subject":c.proposal.subject,"statement":c.proposal.statement,"stance":c.stance,"credence":c.credence,"reason":c.reason}),
        );
        if let Some(p) = commitment {
            unit.as_mut().unwrap()["commitment"] = json!(p);
        }
        if let Some(r) = relationship {
            unit.as_mut().unwrap()["relationship"] = json!(r);
        }
        let root = PositionRef::Claim { claim_id: c.proposal.claim_id.clone() };
        if snapshot.links.iter().any(|l| l.retired_at.is_none() && l.definition.from == root) {
            if let Some((value, more)) = ontology::bundle(snapshot, root)? {
                unit.as_mut().unwrap()["ontology"] = value;
                for id in more {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            } else {
                unit = None;
            }
        }
        candidates.push(Candidate {
            key: format!("claim:{}", c.proposal.claim_id),
            claim: Some(c.proposal.claim_id.clone()),
            score: overlap(&c.proposal.statement) + obligation + c.proposal.importance,
            revision: c.revision,
            unit,
            ids,
        });
    }
    for r in &snapshot.relationships {
        if snapshot.claims.iter().any(|c| c.proposal.relationship_id.as_deref() == Some(&r.definition.relationship_id))
        {
            continue;
        }
        candidates.push(Candidate {
            key: format!("relationship:{}", r.definition.relationship_id),
            claim: None,
            score: overlap(&r.definition.description) + 0.5,
            revision: r.revision,
            unit: Some(json!({"relationship":r})),
            ids: r.definition.source_ids.clone(),
        });
    }
    let mut positions = Vec::new();
    for c in &snapshot.concepts {
        let reference =
            PositionRef::Concept { concept_id: c.definition.concept_id.clone(), version: c.definition.version };
        if snapshot.concepts.iter().any(|n| n.definition.supersedes.contains(&reference)) {
            continue;
        }
        if snapshot
            .concepts
            .iter()
            .any(|n| n.definition.concept_id == c.definition.concept_id && n.definition.version > c.definition.version)
        {
            continue;
        }
        positions.push((
            PositionRef::Concept { concept_id: c.definition.concept_id.clone(), version: c.definition.version },
            format!("concept:{}:{}", c.definition.concept_id, c.definition.version),
            overlap(&format!("{} {}", c.definition.label, c.definition.definition)) + 0.25,
            c.revision,
        ));
    }
    for h in &snapshot.hypotheses {
        if snapshot
            .hypotheses
            .iter()
            .any(|n| n.definition.set_id == h.definition.set_id && n.definition.version > h.definition.version)
        {
            continue;
        }
        positions.push((
            PositionRef::Hypotheses { set_id: h.definition.set_id.clone(), version: h.definition.version },
            format!("hypotheses:{}:{}", h.definition.set_id, h.definition.version),
            overlap(&h.definition.label) + 0.25,
            h.revision,
        ));
    }
    for (root, key, score, revision) in positions {
        let bundle = ontology::bundle(snapshot, root)?;
        let (unit, ids) = bundle.map(|(v, ids)| (Some(json!({"ontology":v})), ids)).unwrap_or((None, vec![]));
        candidates.push(Candidate { key, claim: None, score, revision, unit, ids });
    }
    candidates.sort_by(|a, b| {
        protected.contains(&b.key).cmp(&protected.contains(&a.key)).then_with(|| {
            b.score.total_cmp(&a.score).then_with(|| b.revision.cmp(&a.revision)).then_with(|| a.key.cmp(&b.key))
        })
    });
    let mut units = Vec::new();
    if budget > 0 && count(counter, &render(&snapshot.agent_id, snapshot.revision, &units)?)? > budget {
        return Err(invalid("continuity allowance cannot fit its attribution notice"));
    }
    for candidate in candidates {
        let mut admitted = false;
        if budget > 0 {
            if let Some(mut unit) = candidate.unit {
                let evidence = candidate
                    .ids
                    .iter()
                    .map(|id| {
                        sources
                            .get(id.as_str())
                            .copied()
                            .ok_or_else(|| invalid("continuity source closure is incomplete"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                unit["sources"] = json!(evidence);
                units.push(unit);
                if count(counter, &render(&snapshot.agent_id, snapshot.revision, &units)?)? <= budget {
                    admitted = true;
                } else {
                    units.pop();
                }
            }
        }
        if admitted {
            out.selected_unit_ids.push(candidate.key);
            if let Some(id) = candidate.claim {
                out.selected_claim_ids.push(id);
            }
            for id in candidate.ids {
                if !out.source_ids.contains(&id) {
                    out.source_ids.push(id);
                }
            }
        } else {
            out.omitted_unit_ids.push(candidate.key);
            if let Some(id) = candidate.claim {
                out.omitted_claim_ids.push(id);
            }
        }
    }
    if budget > 0 {
        out.text = render(&snapshot.agent_id, snapshot.revision, &units)?;
        out.usage.tokens = count(counter, &out.text)?;
    }
    Ok(out)
}

impl Memory {
    /// One read snapshot, adding a bounded recent scoped episode candidate view.
    /// Existing `snapshot` remains the exact stance/source projection.
    pub async fn snapshot_with_episodes(&self, relationship: Option<&str>) -> Result<(Snapshot, EpisodeSearchReport)> {
        if let Some(id) = relationship {
            super::kernel::identifier(id)?;
        }
        let agent = self.agent.clone();
        let relationship = relationship.map(str::to_owned);
        self.store.read(move |conn|{
            let tx=conn.unchecked_transaction()?;super::kernel::owner(&tx,&agent)?;
            let mut snapshot=super::kernel::snapshot(&tx,&agent,relationship.as_deref())?;
            let mut stmt=tx.prepare("SELECT source_id,event_seq FROM being_observations WHERE json_extract(body,'$.relationship_id') IS NULL OR json_extract(body,'$.relationship_id')=?1 ORDER BY event_seq DESC LIMIT 129")?;
            let ids=stmt.query_map([relationship.as_deref()],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?)))?.collect::<std::result::Result<Vec<_>,_>>()?;
            let report=EpisodeSearchReport{lookback_limit:128,inspected_observations:ids.len().min(128),candidate_limit_reached:ids.len()>128,semantics:"current_attributed_observations_not_adoption_or_exhaustive_history".into()};
            for (id,seq) in ids.into_iter().take(128) {
                if !snapshot.observations.iter().any(|o|o.source_id==id) {let mut o=super::kernel::load_observation(&tx,&id)?;o.recorded_revision=Some(seq);snapshot.observations.push(o);}
            }
            snapshot.observations.sort_by(|a,b|a.source_id.cmp(&b.source_id));drop(stmt);tx.commit()?;Ok((snapshot,report))
        }).await
    }
    /// Continuity is admitted before the ordinary evidence selector runs. Unused
    /// continuity reservation remains available to evidence; no old full packet
    /// is truncated or counted as an additional allowance.
    pub async fn retrieve_context(
        &self,
        relationship: Option<&str>,
        query: &Query,
        embedder: &dyn Embedder,
        counter: &dyn TokenCounter,
        config: ContextConfig,
    ) -> Result<JoinedContext> {
        let cfg = config.retrieval;
        if cfg.budget_tokens > 2000 || config.continuity_tokens > cfg.budget_tokens {
            return Err(invalid("joint context must fit a single at-most2000-token allowance"));
        }
        let (snapshot, episode_search) = if config.include_episodes {
            let (s, r) = self.snapshot_with_episodes(relationship).await?;
            (s, Some(r))
        } else {
            (self.snapshot(relationship).await?, None)
        };
        let mut memory = continuity(&snapshot, &query.text, counter, config.continuity_tokens)?;
        let primary = Retriever::new(&self.store, embedder, cfg)
            .with_token_counter(counter)
            .with_context_prefix(&memory.text)
            .retrieve(query)
            .await?;
        let budget = primary.context_usage.budget_tokens;
        let (mut text, cited_turns, source_selection) = if config.source_aware && budget > 0 {
            let (evidence, report) = source_aware::recover_with_prefix(
                &self.store,
                &primary,
                &query.text,
                query.at,
                counter,
                source_aware::RecoveryBudget { tokens: budget, prefix: &memory.text },
            )
            .await?;
            (evidence.context, evidence.cited_turns, Some(report))
        } else {
            (primary.context.clone(), primary.cited_turns.clone(), None)
        };
        let mut tokens = count(counter, &text)?;
        if tokens > budget {
            return Err(invalid("joint context exceeded its complete rendered allowance"));
        }
        // Borrow only space left after evidence selection. Preserve the evidence
        // suffix byte-for-byte and pin previously admitted continuity units.
        let free = budget - tokens;
        if config.continuity_tokens > 0 && free > 32 && !memory.omitted_unit_ids.is_empty() {
            let expanded = continuity_protected(
                &snapshot,
                &query.text,
                counter,
                memory.usage.tokens + free - 16,
                &memory.selected_unit_ids,
            )?;
            let evidence = text
                .strip_prefix(&memory.text)
                .ok_or_else(|| invalid("joint context lost its protected continuity prefix"))?;
            let candidate = format!("{}{}", expanded.text, evidence);
            let candidate_tokens = count(counter, &candidate)?;
            if candidate_tokens <= budget
                && memory.selected_unit_ids.iter().all(|id| expanded.selected_unit_ids.contains(id))
            {
                memory = expanded;
                text = candidate;
                tokens = candidate_tokens;
            }
        }
        memory.episode_search = episode_search;
        Ok(JoinedContext {
            usage: ContextUsage { tokenizer: counter.id().into(), budget_tokens: budget, tokens },
            continuity: memory,
            primary,
            text,
            cited_turns,
            source_selection,
        })
    }
}
