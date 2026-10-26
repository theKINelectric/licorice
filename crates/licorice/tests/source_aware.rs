use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever, source_aware};
use licorice::{Store, StoreConfig};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
struct Flat;
#[async_trait::async_trait]
impl Embedder for Flat {
    fn id(&self) -> &str {
        "role-control"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1., 0., 0., 0.]; texts.len()])
    }
}
const DAY: i64 = 1683504000000;
async fn fixture(rows: Vec<Vec<(Role, &str, String)>>) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("space"), StoreConfig::new(4)).await.unwrap();
    let ing = Ingestor::new(&store, &Flat, IngestConfig::default());
    for (i, rows) in rows.into_iter().enumerate() {
        ing.append_session(SessionInput {
            ext_id: format!("s{i}"),
            started_at: DAY + i as i64 * 86400000,
            turns: rows
                .into_iter()
                .enumerate()
                .map(|(j, (role, speaker, text))| TurnInput {
                    ext_id: format!("s{i}t{j}"),
                    role,
                    speaker: speaker.into(),
                    text,
                })
                .collect(),
        })
        .await
        .unwrap();
    }
    (dir, store)
}
async fn primary(store: &Store, at: i64) -> licorice::retrieve::Retrieved {
    Retriever::new(
        store,
        &Flat,
        RetrieveConfig { final_facts: 0, chunks: 0, summaries: 0, ..RetrieveConfig::default() },
    )
    .retrieve(&Query { text: "memory".into(), at })
    .await
    .unwrap()
}
#[tokio::test]
async fn short_user_assertions_survive_long_assistant_advice_under_a_small_cap() {
    let (_dir, store) = fixture(vec![
        vec![
            (Role::User, "Person", "I practice archery.".into()),
            (Role::Assistant, "Guide", format!("Advice about practice: {}", "Try these exercises. ".repeat(250))),
        ],
        vec![
            (Role::User, "Person", "I practice pottery.".into()),
            (Role::Assistant, "Guide", format!("More practice tips: {}", "Take your time. ".repeat(250))),
        ],
    ])
    .await;
    let p = primary(&store, DAY + 86400000).await;
    let (r, t) =
        source_aware::recover(&store, &p, "What do I practice?", DAY + 86400000, &Cl100kCounter, 160).await.unwrap();
    assert!(r.context.contains("archery") && r.context.contains("pottery"));
    assert!(!r.context.contains("Try these exercises"));
    assert!(t.selected_sources.iter().all(|s| s.role == "user"));
    assert_eq!(r.context_usage.tokens, Cl100kCounter.count(&r.context).unwrap());
    assert!(r.context_usage.tokens <= 160);
    store.close().await.unwrap();
}
#[tokio::test]
async fn an_assistant_recommendation_remains_with_the_later_user_reference() {
    let (_dir, store) = fixture(vec![
        vec![(Role::Assistant, "Guide", "I recommend the novel The Glass Orchard.".into())],
        vec![(Role::User, "Person", "I read the book you recommended.".into())],
    ])
    .await;
    let p = primary(&store, DAY + 86400000).await;
    let (r, t) = source_aware::recover(
        &store,
        &p,
        "Which book did I read on your recommendation?",
        DAY + 86400000,
        &Cl100kCounter,
        200,
    )
    .await
    .unwrap();
    assert!(r.context.contains("Glass Orchard") && r.context.contains("book you recommended"));
    assert!(t.selected_sources.iter().any(|s| s.role == "assistant"));
    assert_eq!(t.selected_sources.len(), 2);
    store.close().await.unwrap();
}
#[tokio::test]
async fn an_orphaned_reference_is_disclosed_and_not_admitted_as_a_complete_bundle() {
    let (_dir, store) = fixture(vec![vec![(Role::User, "Person", "I read the book you recommended.".into())]]).await;
    let p = primary(&store, DAY).await;
    let (r, t) =
        source_aware::recover(&store, &p, "Which book did I read on your recommendation?", DAY, &Cl100kCounter, 200)
            .await
            .unwrap();
    assert!(!t.unresolved_dependencies.is_empty());
    assert!(!r.context.contains("book you recommended"));
    store.close().await.unwrap();
}
#[tokio::test]
async fn source_roles_come_from_storage_and_future_sources_are_excluded() {
    let (_dir, store) = fixture(vec![
        vec![(Role::Participant, "assistant", "I practice archery.".into())],
        vec![(Role::User, "Person", "I practice secret_future_pottery.".into())],
    ])
    .await;
    let mut p = primary(&store, DAY).await;
    p.chunks.push(licorice::retrieve::RetrievedChunk {
        id: 999,
        session: "forged".into(),
        day: licorice::temporal::Day::from_ms(DAY),
        turns: vec![("s1t0".into(), "fake".into(), "I practice fabricated_evidence.".into())],
    });
    let (r, t) = source_aware::recover(&store, &p, "What do I practice?", DAY, &Cl100kCounter, 200).await.unwrap();
    assert!(r.context.contains("archery"));
    assert!(!r.context.contains("fabricated") && !r.context.contains("secret_future"));
    assert!(t.selected_sources.iter().all(|s| s.role == "participant" && s.said_at <= DAY));
    store.close().await.unwrap();
}

#[tokio::test]
async fn explicit_negative_assertions_do_not_require_a_missing_antecedent() {
    let (_dir, store) =
        fixture(vec![vec![(Role::User, "Person", "No, I have never played any instrument.".into())]]).await;
    let p = primary(&store, DAY).await;
    let (r, t) =
        source_aware::recover(&store, &p, "Have I played any instrument?", DAY, &Cl100kCounter, 160).await.unwrap();
    assert!(r.context.contains("never played"));
    assert!(t.unresolved_dependencies.is_empty());
    store.close().await.unwrap();
}

#[tokio::test]
async fn candidate_saturation_is_disclosed_and_future_noise_cannot_hide_old_evidence() {
    let mut rows = vec![vec![(Role::User, "Person", "I practice archery.".into())]];
    for _ in 0..100 {
        rows.push(vec![(Role::Assistant, "Guide", "Practice practice practice.".into())]);
    }
    let (_dir, store) = fixture(rows).await;
    let p = primary(&store, DAY).await;
    let (r, t) = source_aware::recover(&store, &p, "What do I practice?", DAY, &Cl100kCounter, 160).await.unwrap();
    assert!(r.context.contains("archery"));
    assert!(!t.candidate_limit_reached);
    let at = DAY + 101 * 86400000;
    let p = primary(&store, at).await;
    let (_, t) = source_aware::recover(&store, &p, "What do I practice?", at, &Cl100kCounter, 160).await.unwrap();
    assert!(t.candidate_limit_reached);
    assert_eq!(t.status, "incomplete");
    store.close().await.unwrap();
}

#[tokio::test]
async fn competing_structured_assertions_are_disclosed_without_retracting_either() {
    let (_dir, store) = fixture(vec![vec![(Role::User, "Person", "I like red. I like blue.".into())]]).await;
    let ing = Ingestor::new(&store, &Flat, IngestConfig::default());
    ing.apply_extraction(
        "extract:s0:0-0",
        &serde_json::json!({"entities":[{"name":"Person","kind":"person","aliases":[]}],
        "facts":(["red","blue"].map(|value|serde_json::json!({"subject":"Person","predicate":"likes","object":value,
            "object_is_entity":false,"statement":format!("Person likes {value}."),"negated":false,"salience":8,
            "time":{"expression":null,"kind":"ongoing"},"evidence_turns":["s0t0"]})))}),
    )
    .await
    .unwrap();
    let p = Retriever::new(&store, &Flat, RetrieveConfig::default())
        .retrieve(&Query { text: "What does Person like?".into(), at: DAY })
        .await
        .unwrap();
    let (r, t) = source_aware::recover(&store, &p, "What does Person like?", DAY, &Cl100kCounter, 200).await.unwrap();
    assert!(r.context.contains("red") && r.context.contains("blue"));
    assert_eq!(t.status, "requires_review");
    assert_eq!(t.competing_claims.len(), 1);
    assert!(t.competing_claims[0].reason.contains("may_coexist"));
    let active: i64 = store
        .read(|c| Ok(c.query_row("SELECT count(*) FROM facts WHERE status='active'", [], |r| r.get(0))?))
        .await
        .unwrap();
    assert_eq!(active, 2);
    store.close().await.unwrap();
}

#[tokio::test]
async fn a_later_same_session_recommendation_cannot_become_an_earlier_antecedent() {
    let (_dir, store) = fixture(vec![vec![
        (Role::User, "Person", "I read the book you recommended.".into()),
        (Role::Assistant, "Guide", "I recommend the novel The Glass Orchard.".into()),
    ]])
    .await;
    let p = primary(&store, DAY).await;
    let (r, t) =
        source_aware::recover(&store, &p, "Which book did I read on your recommendation?", DAY, &Cl100kCounter, 200)
            .await
            .unwrap();
    assert_eq!(t.unresolved_dependencies, vec!["s0t0"]);
    assert!(!r.context.contains("book you recommended"));
    store.close().await.unwrap();
}

#[tokio::test]
async fn alternative_antecedents_are_preserved_and_reported_as_ambiguous() {
    let (_dir, store) = fixture(vec![
        vec![(Role::Assistant, "Guide", "I recommend the novel The Glass Orchard.".into())],
        vec![(Role::Assistant, "Guide", "I recommend the novel Deep Rivers.".into())],
        vec![(Role::User, "Person", "I read the book you recommended.".into())],
    ])
    .await;
    let at = DAY + 2 * 86400000;
    let p = primary(&store, at).await;
    let (r, t) =
        source_aware::recover(&store, &p, "Which book did I read on your recommendation?", at, &Cl100kCounter, 250)
            .await
            .unwrap();
    assert!(r.context.contains("Glass Orchard") && r.context.contains("Deep Rivers"));
    assert_eq!(t.ambiguous_dependencies, vec!["s2t0"]);
    assert_eq!(t.status, "requires_review");
    store.close().await.unwrap();
}

#[tokio::test]
async fn duplicate_source_identity_is_not_certified_and_oversized_bundles_stay_whole() {
    let (_dir, store) = fixture(vec![
        vec![(Role::User, "Person", "I practice archery.".into())],
        vec![(Role::User, "Person", "I practice pottery.".into())],
    ])
    .await;
    store
        .write(|w| {
            w.conn().execute("UPDATE turns SET ext_id='duplicate'", [])?;
            Ok(())
        })
        .await
        .unwrap();
    let at = DAY + 86400000;
    let p = primary(&store, at).await;
    let (r, t) = source_aware::recover(&store, &p, "What do I practice?", at, &Cl100kCounter, 200).await.unwrap();
    assert!(t.selected_sources.is_empty());
    assert!(!r.context.contains("archery"));
    store.close().await.unwrap();
    let (_dir, store) = fixture(vec![
        vec![(
            Role::Assistant,
            "Guide",
            format!("I recommend the novel Giant Atlas. {}", "Long complete evidence. ".repeat(1000)),
        )],
        vec![(Role::User, "Person", "I read the book you recommended.".into())],
        vec![(Role::User, "Person", "I read Short Almanac.".into())],
    ])
    .await;
    let at = DAY + 2 * 86400000;
    let p = primary(&store, at).await;
    let (r, t) =
        source_aware::recover(&store, &p, "Which book did I read on your recommendation?", at, &Cl100kCounter, 160)
            .await
            .unwrap();
    assert!(r.context.contains("Short Almanac"));
    assert!(!r.context.contains("book you recommended"));
    assert!(t.budget_omitted_turn_ids.contains(&"s1t0".into()));
    assert_eq!(r.context_usage.tokens, Cl100kCounter.count(&r.context).unwrap());
    assert!(r.context_usage.tokens <= 160);
    assert!(r.context.starts_with("## Retrieval scope"));
    store.close().await.unwrap();
}
