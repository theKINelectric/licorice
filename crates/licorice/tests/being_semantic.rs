//! Synthetic controls for the opt-in selector, not embedding-model quality claims.
use licorice::being::context::{
    SemanticContinuityConfig, continuity, semantic_continuity, semantic_continuity_with_config,
};
use licorice::being::{Assurance, Observation, Origin, Snapshot, Source, StoredObservation};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use std::sync::Mutex;

#[derive(Default)]
struct Synonyms {
    calls: Mutex<Vec<EmbedKind>>,
}

#[async_trait::async_trait]
impl Embedder for Synonyms {
    fn id(&self) -> &str {
        "synthetic-synonym-control"
    }
    fn dim(&self) -> usize {
        3
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.calls.lock().unwrap().push(kind);
        Ok(texts
            .iter()
            .map(|text| {
                if text.contains("canine") || text.contains("dog") {
                    // Deliberately unnormalized; the selector must compute cosine.
                    vec![7.0, 0.0, 0.0]
                } else if text.contains("plant") || text.contains("botanical") {
                    vec![0.0, 3.0, 0.0]
                } else {
                    vec![0.0, 0.0, 5.0]
                }
            })
            .collect())
    }
}

fn observation(id: &str, revision: i64, text: &str) -> StoredObservation {
    StoredObservation {
        source_id: id.into(),
        observation: Observation {
            origin: Origin::ImportedDocument,
            subject: "book-subject".into(),
            text: text.into(),
            relationship_id: None,
            occurred_at: Some(42),
            source: Source::Reported { reference: Some("book page 12".into()) },
        },
        assurance: Assurance::Reported,
        recorded_revision: Some(revision),
    }
}

fn snapshot(observations: Vec<StoredObservation>) -> Snapshot {
    Snapshot {
        agent_id: "agent-under-test".into(),
        revision: 10,
        observations,
        claims: vec![],
        relationships: vec![],
        commitments: vec![],
        concepts: vec![],
        hypotheses: vec![],
        links: vec![],
    }
}

#[tokio::test]
async fn synonym_recall_uses_whole_report_and_preserves_source_and_subject() {
    let source = observation("borrowed-life", 1, "Ash's dog enjoys fetching sticks.");
    let state = snapshot(vec![source.clone(), observation("garden", 2, "The plant thrives beside the window.")]);
    let embedder = Synonyms::default();
    let lexical = continuity(&state, "canine recreation", &Cl100kCounter, 2000).unwrap();
    assert!(lexical.source_ids.is_empty(), "control must have no overlapping words");
    let packet = semantic_continuity(&state, "canine recreation", &embedder, &Cl100kCounter, 2000).await.unwrap();
    assert_eq!(packet.source_ids, ["borrowed-life"]);
    assert_eq!(packet.selected_unit_ids, ["observation:borrowed-life"]);
    assert!(packet.selected_claim_ids.is_empty());
    let units: serde_json::Value = serde_json::from_str(packet.text.split_once('\n').unwrap().1).unwrap();
    assert_eq!(units[0]["status"], "unadopted_observation");
    assert_eq!(units[0]["sources"][0], serde_json::to_value(source).unwrap());
    assert_eq!(packet.usage.tokens, Cl100kCounter.count(&packet.text).unwrap());
    assert!(packet.usage.tokens <= 2000);
    assert_eq!(*embedder.calls.lock().unwrap(), [EmbedKind::Query, EmbedKind::Document]);
    let report = packet.semantic_search.unwrap();
    assert_eq!(report.candidates.len(), 1);
    assert_eq!(report.candidates[0].lexical_terms, 0);
    assert_eq!(report.candidates[0].cosine, 1.0);
    assert!(lexical.semantic_search.is_none());
    assert!(
        serde_json::to_value(lexical).unwrap().get("semantic_search").is_none(),
        "default packet schema is unchanged"
    );
}

#[tokio::test]
async fn unrelated_query_does_not_force_nearest_observation_into_context() {
    let state = snapshot(vec![observation("animal", 1, "A dog rested near the door.")]);
    let packet = semantic_continuity(&state, "botanical identification", &Synonyms::default(), &Cl100kCounter, 2000)
        .await
        .unwrap();
    assert!(packet.source_ids.is_empty());
    assert!(packet.selected_unit_ids.is_empty());
    assert!(packet.semantic_search.unwrap().candidates.is_empty());
}

#[tokio::test]
async fn misleading_lexical_overlap_cannot_bypass_the_semantic_floor() {
    let state = snapshot(vec![observation("lexical", 1, "The plant catalog includes the canine portrait.")]);
    // The fake provider assigns the document to the canine axis, the question to
    // the botanical axis; the shared word must not bypass semantic eligibility.
    let packet =
        semantic_continuity(&state, "botanical plant", &Synonyms::default(), &Cl100kCounter, 2000).await.unwrap();
    assert!(packet.source_ids.is_empty());
    assert!(packet.semantic_search.unwrap().candidates.is_empty());
}

#[tokio::test]
async fn empty_or_function_only_question_does_not_embed_or_select_episodes() {
    let state = snapshot(vec![observation("animal", 1, "The dog was there and this is its story.")]);
    let embedder = Synonyms::default();
    for query in ["", "  ... !!! ", "what was the and this"] {
        let packet = semantic_continuity(&state, query, &embedder, &Cl100kCounter, 2000).await.unwrap();
        assert!(packet.source_ids.is_empty());
        assert_eq!(packet.semantic_search.unwrap().inspected_observations, 0);
    }
    assert!(embedder.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn deterministic_whole_unit_packing_counts_the_complete_rendered_text() {
    let mut state = snapshot(
        (0..20)
            .map(|i| {
                observation(
                    &format!("dog-{i:02}"),
                    i,
                    &format!("The dog report {i}: {}", "evidence with attribution; ".repeat(25)),
                )
            })
            .collect(),
    );
    let first =
        semantic_continuity(&state, "canine recreation", &Synonyms::default(), &Cl100kCounter, 600).await.unwrap();
    state.observations.reverse();
    let second =
        semantic_continuity(&state, "canine recreation", &Synonyms::default(), &Cl100kCounter, 600).await.unwrap();
    assert_eq!(first.text, second.text);
    assert_eq!(first.source_ids, second.source_ids);
    assert!(!first.source_ids.is_empty());
    assert!(!first.omitted_unit_ids.is_empty());
    assert_eq!(first.usage.tokens, Cl100kCounter.count(&first.text).unwrap());
    assert!(first.usage.tokens <= 600);
    let units: serde_json::Value = serde_json::from_str(first.text.split_once('\n').unwrap().1).unwrap();
    for unit in units.as_array().unwrap() {
        let id = unit["source_id"].as_str().unwrap();
        let original = state.observations.iter().find(|o| o.source_id == id).unwrap();
        assert_eq!(unit["sources"][0], serde_json::to_value(original).unwrap());
    }
    assert_eq!(state.claims.len(), 0, "selection is read-only");
}

#[tokio::test]
async fn bounded_candidate_view_records_its_limit_and_prefers_recent_sources() {
    let state = snapshot((0..130).map(|i| observation(&format!("dog-{i:03}"), i, "The dog rests.")).collect());
    let config = SemanticContinuityConfig { minimum_cosine: 0.5, maximum_observations: 2 };
    let packet = semantic_continuity_with_config(&state, "canine", &Synonyms::default(), &Cl100kCounter, 2000, config)
        .await
        .unwrap();
    assert_eq!(packet.source_ids, ["dog-129", "dog-128"]);
    let report = packet.semantic_search.unwrap();
    assert_eq!(report.inspected_observations, 2);
    assert!(report.candidate_limit_reached);
}

struct Malformed {
    kind: &'static str,
    corrupt: EmbedKind,
}
#[async_trait::async_trait]
impl Embedder for Malformed {
    fn id(&self) -> &str {
        "malformed-control"
    }
    fn dim(&self) -> usize {
        3
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        if kind != self.corrupt {
            return Ok(vec![vec![1.0, 0.0, 0.0]; texts.len()]);
        }
        Ok(match self.kind {
            "count" => vec![],
            "dimension" => vec![vec![1.0]; texts.len()],
            "nan" => vec![vec![f32::NAN, 0.0, 0.0]; texts.len()],
            "infinity" => vec![vec![f32::INFINITY, 0.0, 0.0]; texts.len()],
            "zero" => vec![vec![0.0; 3]; texts.len()],
            _ => unreachable!(),
        })
    }
}

#[tokio::test]
async fn malformed_query_and_document_embeddings_are_refused_before_selection() {
    let state = snapshot(vec![observation("source", 1, "The dog rests.")]);
    for kind in ["count", "dimension", "nan", "infinity", "zero"] {
        for corrupt in [EmbedKind::Query, EmbedKind::Document] {
            let result =
                semantic_continuity(&state, "canine", &Malformed { kind, corrupt }, &Cl100kCounter, 2000).await;
            assert!(result.is_err(), "must reject {kind} in {corrupt:?}");
        }
    }
}

#[tokio::test]
async fn invalid_budget_question_or_relevance_policy_is_rejected_without_embedding() {
    let state = snapshot(vec![observation("source", 1, "The dog rests.")]);
    let embedder = Synonyms::default();
    assert!(semantic_continuity(&state, "canine", &embedder, &Cl100kCounter, 2001).await.is_err());
    assert!(semantic_continuity(&state, &"x".repeat(2001), &embedder, &Cl100kCounter, 2000).await.is_err());
    for minimum_cosine in [f64::NAN, -0.1, 0.0, 1.1] {
        let config = SemanticContinuityConfig { minimum_cosine, maximum_observations: 128 };
        assert!(
            semantic_continuity_with_config(&state, "canine", &embedder, &Cl100kCounter, 2000, config).await.is_err()
        );
    }
    for maximum_observations in [0, 129] {
        let config = SemanticContinuityConfig { minimum_cosine: 0.45, maximum_observations };
        assert!(
            semantic_continuity_with_config(&state, "canine", &embedder, &Cl100kCounter, 2000, config).await.is_err()
        );
    }
    assert!(embedder.calls.lock().unwrap().is_empty());
}
