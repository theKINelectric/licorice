use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever};
use licorice::{Store, StoreConfig};
use licorice_providers::chat::ChatError;
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::rerank::{Candidate, Ranked, Reranker};
use serde_json::json;
struct Flat;
#[async_trait::async_trait]
impl Embedder for Flat {
    fn id(&self) -> &str {
        "flat-graph-control"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1., 0., 0., 0.]; texts.len()])
    }
}
struct EndpointFirst;
#[async_trait::async_trait]
impl Reranker for EndpointFirst {
    fn id(&self) -> &str {
        "endpoint-first-control"
    }
    async fn rank(&self, _: &str, candidates: &[Candidate]) -> Result<Vec<Ranked>, ChatError> {
        Ok(candidates
            .iter()
            .map(|c| Ranked {
                id: c.id,
                score: if c.text.contains("Bergen") {
                    1.
                } else if c.text.contains("likes") {
                    0.9
                } else {
                    0.1
                },
            })
            .collect())
    }
}
async fn fixture(bridge: &str) -> (tempfile::TempDir, Store) {
    fixture_with(bridge, &Flat).await
}
async fn fixture_with(bridge: &str, embedder: &dyn Embedder) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let ing = Ingestor::new(&store, embedder, IngestConfig { merge_threshold: 1.1, ..IngestConfig::default() });
    ing.append_session(SessionInput {
        ext_id: "s".into(),
        started_at: 1683504000000,
        turns: vec![TurnInput {
            ext_id: "t1".into(),
            speaker: "Ada".into(),
            role: Role::User,
            text: "My sister Mira lives in Bergen. I like maple syrup.".into(),
        }],
    })
    .await
    .unwrap();
    let fact = |subject: &str, predicate: &str, object: &str, entity: bool, text: &str| json!({"subject":subject,"predicate":predicate,"object":object,"object_is_entity":entity,"statement":text,"negated":false,"salience":8,"time":{"kind":"unknown","expression":null},"evidence_turns":["t1"]});
    ing.apply_extraction("extract:s:0-0",&json!({"entities":[{"name":"Mira","kind":"person","aliases":[]}],"facts":[fact("Ada","sister","Mira",true,bridge),fact("Mira","lives in","Bergen",false,"Mira lives in Bergen."),fact("Ada","likes","maple syrup",false,"Ada likes maple syrup.")]})).await.unwrap();
    (dir, store)
}
#[tokio::test]
async fn connecting_path_survives_a_reranker_that_prefers_the_endpoint_and_a_distractor() {
    let (_dir, store) = fixture("Ada's sister is Mira.").await;
    let query = Query { text: "Where does Ada's sister live?".into(), at: 1683590400000 };
    let cfg = RetrieveConfig {
        graph_hops: 2,
        connected_evidence: false,
        final_facts: 2,
        chunks: 3,
        summaries: 0,
        ..RetrieveConfig::default()
    };
    let plain =
        Retriever::new(&store, &Flat, cfg.clone()).with_reranker(&EndpointFirst).retrieve(&query).await.unwrap();
    assert!(plain.facts.iter().any(|f| f.statement.contains("Bergen")));
    assert!(!plain.facts.iter().any(|f| f.statement.contains("sister")));
    let connected = Retriever::new(&store, &Flat, RetrieveConfig { connected_evidence: true, ..cfg })
        .with_reranker(&EndpointFirst)
        .retrieve(&query)
        .await
        .unwrap();
    assert_eq!(connected.facts.len(), 2);
    assert!(connected.facts.iter().any(|f| f.statement.contains("sister")));
    assert!(connected.facts.iter().any(|f| f.statement.contains("Bergen")));
    assert!(connected.graph_paths.iter().any(|p| p.facts.len() == 2));
    assert!(connected.context_usage.tokens <= 2000);
    for fact in &connected.facts {
        for source in &fact.evidence_turns {
            assert!(
                connected.chunks.iter().any(|c| c.turns.iter().any(|t| &t.0 == source)),
                "a retained path must retain its sources"
            );
        }
    }
    store.close().await.unwrap();
}
#[tokio::test]
async fn an_oversized_connecting_path_is_rejected_whole_and_later_evidence_can_fit() {
    let (_dir, store) = fixture(&format!("Ada's sister is Mira. {}", "detailed source words ".repeat(300))).await;
    let cfg = RetrieveConfig {
        budget_tokens: 180,
        graph_hops: 2,
        connected_evidence: true,
        final_facts: 2,
        chunks: 3,
        summaries: 0,
        ..RetrieveConfig::default()
    };
    let r = Retriever::new(&store, &Flat, cfg)
        .with_reranker(&EndpointFirst)
        .retrieve(&Query { text: "Where does Ada's sister live?".into(), at: 1683590400000 })
        .await
        .unwrap();
    assert!(
        !r.facts.iter().any(|f| f.statement.contains("Bergen")),
        "endpoint cannot escape a rejected connecting bundle"
    );
    assert!(r.facts.iter().any(|f| f.statement.contains("maple syrup")));
    assert!(r.context_usage.tokens <= 180);
    store.close().await.unwrap();
}

struct InvalidReranker;
#[async_trait::async_trait]
impl Reranker for InvalidReranker {
    fn id(&self) -> &str {
        "invalid-control"
    }
    async fn rank(&self, _: &str, _: &[Candidate]) -> Result<Vec<Ranked>, ChatError> {
        Ok(vec![Ranked { id: 999_999, score: 1. }])
    }
}
#[tokio::test]
async fn optional_rerank_failure_preserves_native_order_and_records_the_failure() {
    let (_dir, store) = fixture("Ada's sister is Mira.").await;
    let query = Query { text: "Where does Ada's sister live?".into(), at: 1683590400000 };
    let cfg = RetrieveConfig { rerank_fallback: true, ..Default::default() };
    let native = Retriever::new(&store, &Flat, cfg.clone()).retrieve(&query).await.unwrap();
    let fallback = Retriever::new(&store, &Flat, cfg).with_reranker(&InvalidReranker).retrieve(&query).await.unwrap();
    assert_eq!(fallback.context, native.context);
    assert!(fallback.rerank_error.is_some());
    assert!(fallback.reranking.is_none());
    assert!(
        Retriever::new(&store, &Flat, RetrieveConfig::default())
            .with_reranker(&InvalidReranker)
            .retrieve(&query)
            .await
            .is_err()
    );
    store.close().await.unwrap();
}

struct EndpointSemantic;
#[async_trait::async_trait]
impl Embedder for EndpointSemantic {
    fn id(&self) -> &str {
        "endpoint-semantic-control"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                if matches!(kind, EmbedKind::Query) || text.contains("Mira lives in Bergen") {
                    vec![1., 0., 0., 0.]
                } else {
                    vec![0., 1., 0., 0.]
                }
            })
            .collect())
    }
}

#[tokio::test]
async fn a_valid_bridge_outside_the_selection_pool_still_closes_the_selected_path() {
    let (_dir, store) = fixture_with("Ada's sister is Mira.", &EndpointSemantic).await;
    let cfg = RetrieveConfig {
        selection_candidates: 1,
        graph_hops: 2,
        connected_evidence: true,
        final_facts: 2,
        chunks: 3,
        summaries: 0,
        ..Default::default()
    };
    let r = Retriever::new(&store, &EndpointSemantic, cfg)
        .retrieve(&Query { text: "Where does Ada's sister live?".into(), at: 1683590400000 })
        .await
        .unwrap();
    assert!(
        r.facts.iter().any(|f| f.statement == "Mira lives in Bergen."),
        "a low-ranked supporting bridge must not discard the selected endpoint"
    );
    assert!(r.facts.iter().any(|f| f.statement == "Ada's sister is Mira."));
    assert_eq!(r.facts.len(), 2);
    assert_eq!(r.selection.candidates, 1, "support closure is separate from the candidate ranking pool");
    assert!(r.context_usage.tokens <= 2000);
    store.close().await.unwrap();
}

#[tokio::test]
async fn an_expired_bridge_cannot_be_restored_by_path_support_closure() {
    let (_dir, store) = fixture_with("Ada's sister is Mira.", &EndpointSemantic).await;
    store.write(|w|{w.conn().execute("UPDATE facts SET time_kind='ongoing',status='superseded',valid_until=1683504000000 WHERE statement='Ada''s sister is Mira.'",[])?;Ok(())}).await.unwrap();
    let cfg = RetrieveConfig {
        selection_candidates: 1,
        graph_hops: 2,
        connected_evidence: true,
        final_facts: 2,
        chunks: 3,
        summaries: 0,
        ..Default::default()
    };
    let r = Retriever::new(&store, &EndpointSemantic, cfg)
        .retrieve(&Query { text: "Where does Ada's sister live now?".into(), at: 1683590400000 })
        .await
        .unwrap();
    assert!(r.facts.iter().all(|f| f.statement != "Ada's sister is Mira."));
    assert!(
        r.facts.iter().all(|f| f.statement != "Mira lives in Bergen."),
        "an ineligible connecting state rejects its dependent bundle"
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn an_explicit_query_speaker_resolves_self_reference_without_rewriting_third_parties() {
    let (_dir, store) = fixture("Ada's sister is Mira.").await;
    let cfg:RetrieveConfig=serde_json::from_value(json!({"query_speaker":"Ada","graph_hops":2,"connected_evidence":true,"final_facts":2,"chunks":3,"summaries":0})).unwrap();
    let r = Retriever::new(&store, &Flat, cfg.clone())
        .with_reranker(&EndpointFirst)
        .retrieve(&Query { text: "Where does my sister live?".into(), at: 1683590400000 })
        .await
        .unwrap();
    assert!(
        r.linked_entities.iter().any(|name| name == "Ada"),
        "the explicitly declared speaker must anchor a first-person query"
    );
    assert!(r.facts.iter().any(|f| f.statement == "Ada's sister is Mira."));
    assert!(r.facts.iter().any(|f| f.statement == "Mira lives in Bergen."));
    assert!(r.context_usage.tokens <= 2000);
    let third = Retriever::new(&store, &Flat, cfg)
        .retrieve(&Query { text: "Where does Mira live?".into(), at: 1683590400000 })
        .await
        .unwrap();
    assert!(
        !third.linked_entities.iter().any(|name| name == "Ada"),
        "a declared speaker must not rewrite a third-party question"
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn missing_or_ambiguous_speaker_aliases_are_not_guessed_or_created() {
    let (_dir, store) = fixture("Ada's sister is Mira.").await;
    let before: i64 =
        store.read(|c| Ok(c.query_row("SELECT count(*) FROM entities", [], |r| r.get(0))?)).await.unwrap();
    let cfg: RetrieveConfig = serde_json::from_value(json!({"query_speaker":"Nobody","summaries":0})).unwrap();
    let query = Query { text: "Where does my sister live?".into(), at: 1683590400000 };
    let absent = Retriever::new(&store, &Flat, cfg).retrieve(&query).await.unwrap();
    assert!(!absent.linked_entities.iter().any(|name| name == "Nobody"));
    store
        .write(|w| {
            w.conn().execute(
                "INSERT INTO entity_aliases(alias,entity_id) SELECT 'ada',id FROM entities WHERE name='Mira'",
                [],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let cfg: RetrieveConfig = serde_json::from_value(json!({"query_speaker":"Ada","summaries":0})).unwrap();
    let ambiguous = Retriever::new(&store, &Flat, cfg).retrieve(&query).await.unwrap();
    assert!(
        ambiguous.linked_entities.is_empty(),
        "an implicit self reference cannot pick one of two known alias owners"
    );
    let after: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM entities", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(before, after);
    store.close().await.unwrap();
}
