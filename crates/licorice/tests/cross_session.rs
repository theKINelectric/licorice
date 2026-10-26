use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever, cross_session};
use licorice::{Store, StoreConfig};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};

struct Flat;
#[async_trait::async_trait]
impl Embedder for Flat {
    fn id(&self) -> &str {
        "cross-session-control"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1., 0., 0., 0.]; texts.len()])
    }
}
const DAY: i64 = 1683504000000;
async fn fixture(sessions: Vec<Vec<(&str, String)>>) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory"), StoreConfig::new(4)).await.unwrap();
    let ing = Ingestor::new(&store, &Flat, IngestConfig::default());
    for (i, turns) in sessions.into_iter().enumerate() {
        ing.append_session(SessionInput {
            ext_id: format!("session{i}"),
            started_at: DAY + i as i64 * 86400000,
            turns: turns
                .into_iter()
                .enumerate()
                .map(|(j, (speaker, text))| TurnInput {
                    ext_id: format!("s{i}t{j}"),
                    speaker: speaker.into(),
                    role: Role::User,
                    text,
                })
                .collect(),
        })
        .await
        .unwrap();
    }
    store
        .write(|w| {
            w.conn()
                .execute("INSERT INTO entities(id,name,kind) VALUES(990,'Noa','person'),(991,'Omar','person')", [])?;
            w.conn().execute("INSERT INTO entity_aliases(alias,entity_id) VALUES('Noa',990),('Omar',991)", [])?;
            Ok(())
        })
        .await
        .unwrap();
    (dir, store)
}
async fn empty_primary(store: &Store, at: i64) -> licorice::retrieve::Retrieved {
    Retriever::new(
        store,
        &Flat,
        RetrieveConfig { final_facts: 0, chunks: 0, summaries: 0, ..RetrieveConfig::default() },
    )
    .retrieve(&Query { text: "Noa".into(), at })
    .await
    .unwrap()
}
#[tokio::test]
async fn carries_question_answer_neighbors_and_links_two_sessions() {
    let (_dir, store) = fixture(vec![
        vec![("Noa", "Which book would you recommend?".into()), ("Omar", "The Glass Orchard. I recommend it.".into())],
        vec![("Noa", "I am reading the book you recommended.".into())],
    ])
    .await;
    let at = DAY + 86400000;
    let primary = empty_primary(&store, at).await;
    let (found, trace) = cross_session::recover(
        &store,
        &primary,
        "Which book did Noa read from Omar's recommendation?",
        at,
        &Cl100kCounter,
        2000,
    )
    .await
    .unwrap();
    assert!(found.context.contains("Which book would you recommend?"));
    assert!(found.context.contains("The Glass Orchard"));
    assert!(found.context.contains("reading the book"));
    assert_eq!(trace.admitted_sessions.len(), 2);
    assert_eq!(found.context_usage.tokens, Cl100kCounter.count(&found.context).unwrap());
    assert!(found.context_usage.tokens <= 2000);
    store.close().await.unwrap();
}
#[tokio::test]
async fn list_recall_crosses_sessions_without_fetching_an_unrelated_person() {
    let (_dir, store) = fixture(vec![
        vec![("Noa", "I bought a telescope.".into())],
        vec![("Noa", "I purchased hiking boots.".into())],
        vec![("Omar", "I bought a kayak.".into())],
    ])
    .await;
    let at = DAY + 3 * 86400000;
    let p = empty_primary(&store, at).await;
    let (found, _) =
        cross_session::recover(&store, &p, "What has Noa bought?", at, &Cl100kCounter, 2000).await.unwrap();
    assert!(found.context.contains("telescope"));
    assert!(found.context.contains("hiking boots"));
    assert!(!found.context.contains("kayak"));
    store.close().await.unwrap();
}
#[tokio::test]
async fn future_turns_cannot_enter_seed_windows_or_primary_backfill() {
    let (_dir, store) = fixture(vec![
        vec![("Noa", "I bought a telescope.".into())],
        vec![("Noa", "I bought a secret future bicycle.".into())],
    ])
    .await;
    let mut p = empty_primary(&store, DAY).await;
    p.chunks.push(licorice::retrieve::RetrievedChunk {
        id: 999,
        session: "forged-old-date".into(),
        day: licorice::temporal::Day::from_ms(DAY),
        turns: vec![("s1t0".into(), "Noa".into(), "I bought a secret future bicycle.".into())],
    });
    let (found, _) =
        cross_session::recover(&store, &p, "What has Noa bought?", DAY, &Cl100kCounter, 2000).await.unwrap();
    assert!(found.context.contains("telescope"));
    assert!(!found.context.contains("future bicycle"));
    store.close().await.unwrap();
}
#[tokio::test]
async fn oversized_bundle_is_rejected_whole_and_later_unicode_bundle_fits() {
    let huge = format!("I recommend this book. {}", "unbroken evidence ".repeat(2000));
    let (_dir, store) = fixture(vec![
        vec![("Noa", huge), ("Omar", "Its title is a secret orphaned reply.".into())],
        vec![("Noa", "I recommend Café 日本語 🍌, a short book.".into())],
    ])
    .await;
    let at = DAY + 2 * 86400000;
    let p = empty_primary(&store, at).await;
    let (found, _) =
        cross_session::recover(&store, &p, "What book did Noa recommend?", at, &Cl100kCounter, 180).await.unwrap();
    assert!(found.context.contains("Café 日本語 🍌"));
    assert!(!found.context.contains("orphaned reply"));
    assert!(!found.context.contains("unbroken evidence"));
    assert_eq!(found.context_usage.tokens, Cl100kCounter.count(&found.context).unwrap());
    assert!(found.context_usage.tokens <= 180);
    store.close().await.unwrap();
}
#[tokio::test]
async fn overlaps_are_deduplicated_and_oversized_contract_is_rejected() {
    let (_dir, store) = fixture(vec![vec![
        ("Noa", "Tell me about the cats.".into()),
        ("Omar", "What are their names?".into()),
        ("Noa", "Juniper and Moss are my cats.".into()),
    ]])
    .await;
    let p = empty_primary(&store, DAY).await;
    let (found, _) =
        cross_session::recover(&store, &p, "What are Noa's pets' names?", DAY, &Cl100kCounter, 2000).await.unwrap();
    assert!(found.context.contains("Juniper and Moss"));
    assert_eq!(found.context.matches("[s0t2]").count(), 1);
    assert!(cross_session::recover(&store, &p, "Noa pets", DAY, &Cl100kCounter, 2001).await.is_err());
    store.close().await.unwrap();
}

#[tokio::test]
async fn complementary_noise_cannot_displace_a_primary_instrument_witness() {
    let mut sessions = vec![vec![("Noa", "I play the oboe.".into())]];
    for _ in 0..12 {
        sessions.push(vec![(
            "Noa",
            format!("I play musical instruments in a band concert. {}", "Music is relaxing. ".repeat(12)),
        )]);
    }
    let (_dir, store) = fixture(sessions).await;
    let at = DAY + 20 * 86400000;
    let mut p = empty_primary(&store, at).await;
    p.facts.push(licorice::retrieve::RetrievedFact {
        id: 9900,
        statement: "Noa plays the oboe.".into(),
        day: licorice::temporal::Day::from_ms(DAY),
        dated: false,
        occurred: None,
        superseded: false,
        valid_until: None,
        session: "session0".into(),
        evidence_turns: vec!["s0t0".into()],
        score: 1.0,
    });
    let (found, _) =
        cross_session::recover(&store, &p, "What instruments does Noa play?", at, &Cl100kCounter, 180).await.unwrap();
    assert!(found.context.contains("oboe"), "primary witness displaced: {}", found.context);
    assert!(found.context_usage.tokens <= 180);
    store.close().await.unwrap();
}

#[tokio::test]
async fn origin_question_finds_a_country_without_inventing_its_name() {
    let (_dir, store) = fixture(vec![
        vec![("Noa", "I moved four years ago.".into())],
        vec![("Noa", "This gift reminds me of my roots and my home country, Finland.".into())],
    ])
    .await;
    let at = DAY + 2 * 86400000;
    let p = empty_primary(&store, at).await;
    let (found, _) =
        cross_session::recover(&store, &p, "Where did Noa move from four years ago?", at, &Cl100kCounter, 2000)
            .await
            .unwrap();
    assert!(found.context.contains("Finland"));
    store.close().await.unwrap();
}

#[tokio::test]
async fn inflected_support_query_protects_all_short_primary_list_members() {
    let mut sessions = vec![vec![("Noa", "My family and my mentors support me.".into())]];
    for _ in 0..12 {
        sessions
            .push(vec![("Noa", format!("My friends support me. {}", "Their support helps every day. ".repeat(12)))]);
    }
    let (_dir, store) = fixture(sessions).await;
    let at = DAY + 20 * 86400000;
    let mut p = empty_primary(&store, at).await;
    for (id, statement) in [(9900, "Family support is important to Noa."), (9901, "Noa's mentors support her.")] {
        p.facts.push(licorice::retrieve::RetrievedFact {
            id,
            statement: statement.into(),
            day: licorice::temporal::Day::from_ms(DAY),
            dated: false,
            occurred: None,
            superseded: false,
            valid_until: None,
            session: "session0".into(),
            evidence_turns: vec!["s0t0".into()],
            score: 1.0,
        });
    }
    let (found, trace) =
        cross_session::recover(&store, &p, "Who supports Noa?", at, &Cl100kCounter, 180).await.unwrap();
    assert!(trace.protected_primary_fact_ids.contains(&9900), "{}", found.context);
    assert!(trace.protected_primary_fact_ids.contains(&9901), "{}", found.context);
    assert!(found.context.contains("Family") && found.context.contains("mentors"));
    assert!(found.context_usage.tokens <= 180);
    store.close().await.unwrap();
}

#[tokio::test]
async fn participation_query_collects_different_activity_verbs_across_sessions() {
    let (_dir, store) = fixture(vec![
        vec![("Noa", "I volunteered at the food bank.".into())],
        vec![("Noa", "I attended a neighborhood parade.".into())],
        vec![("Noa", "I organized a community concert.".into())],
        vec![("Noa", "I'm putting together a neighborhood exhibit next month.".into())],
        vec![("Omar", "I attended a chess tournament.".into())],
    ])
    .await;
    let at = DAY + 5 * 86400000;
    let p = empty_primary(&store, at).await;
    let (found, _) =
        cross_session::recover(&store, &p, "Which activities has Noa participated in?", at, &Cl100kCounter, 300)
            .await
            .unwrap();
    for member in ["food bank", "neighborhood parade", "community concert", "neighborhood exhibit next month"] {
        assert!(found.context.contains(member), "missing {member}: {}", found.context);
    }
    assert!(!found.context.contains("chess tournament"));
    store.close().await.unwrap();
}

#[tokio::test]
async fn recommendation_link_is_admitted_together_before_repeated_book_noise() {
    let mut sessions = vec![
        vec![("Noa", "Could you recommend a book?".into()), ("Omar", "I recommend The Glass Orchard.".into())],
        vec![("Noa", "I read the book you recommended.".into())],
    ];
    for _ in 0..20 {
        sessions.push(vec![(
            "Noa",
            format!("I read books and novels. Reading a title is relaxing. {}", "I read books every day. ".repeat(6)),
        )]);
    }
    let (_dir, store) = fixture(sessions).await;
    let at = DAY + 25 * 86400000;
    let p = empty_primary(&store, at).await;
    let (found, trace) =
        cross_session::recover(&store, &p, "Which book did Noa read on a recommendation?", at, &Cl100kCounter, 180)
            .await
            .unwrap();
    assert!(found.context.contains("The Glass Orchard"), "{}", found.context);
    assert!(found.context.contains("I read the book you recommended"), "{}", found.context);
    assert_eq!(trace.linked_windows, 2);
    assert!(found.context_usage.tokens <= 180);
    store.close().await.unwrap();
}
