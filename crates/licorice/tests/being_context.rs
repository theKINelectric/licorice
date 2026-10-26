use licorice::being::context::{ContextConfig, continuity};
use licorice::being::{Actor, ActorRole, Category, Decision, Memory, Operation, Proposal, Request, Stance};
use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever};
use licorice::{Store, StoreConfig};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};

struct Flat;
#[async_trait::async_trait]
impl Embedder for Flat {
    fn id(&self) -> &str {
        "being-context-fixture"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1., 0., 0., 0.]; texts.len()])
    }
}
const AT: i64 = 1792595000000;
#[tokio::test]
async fn episode_candidates_are_bounded_and_join_factual_evidence_without_endorsement() {
    let (_d, store, memory) = fixture().await;
    for i in 0..131 {
        memory
            .apply(
                licorice::being::Actor {
                    principal: "human".into(),
                    role: licorice::being::ActorRole::Human,
                    session: "source".into(),
                },
                Request {
                    request_id: format!("episode-{i:03}"),
                    expected_revision: None,
                    operation: Operation::Observe(licorice::being::Observation {
                        origin: licorice::being::Origin::HumanReport,
                        subject: "lloyd".into(),
                        text: format!("Harbor Lens observation {i}: a blue afterimage appeared."),
                        relationship_id: None,
                        occurred_at: None,
                        source: licorice::being::Source::Reported { reference: None },
                    }),
                },
            )
            .await
            .unwrap();
    }
    let (snapshot, report) = memory.snapshot_with_episodes(None).await.unwrap();
    assert_eq!(report.inspected_observations, 128);
    assert!(report.candidate_limit_reached);
    assert!(!snapshot.observations.iter().any(|o| o.source_id == "episode-000"));
    assert!(snapshot.observations.iter().any(|o| o.source_id == "episode-130"));
    let query = Query { text: "What did the Harbor Lens show, and when do workshop orders ship?".into(), at: AT };
    let joined = memory
        .retrieve_context(
            None,
            &query,
            &Flat,
            &Cl100kCounter,
            ContextConfig { include_episodes: true, ..Default::default() },
        )
        .await
        .unwrap();
    assert!(joined.text.contains("unadopted_observation") && joined.text.contains("blue afterimage"));
    assert!(
        joined.continuity.selected_unit_ids.contains(&"observation:episode-130".into()),
        "equal-relevance episodes must prefer the most recently recorded observation"
    );
    assert!(joined.text.contains("Fridays"));
    assert!(joined.continuity.episode_search.unwrap().candidate_limit_reached);
    assert_eq!(joined.usage.tokens, Cl100kCounter.count(&joined.text).unwrap());
    assert!(joined.usage.tokens <= 2000);
    assert_eq!(memory.snapshot(None).await.unwrap().claims.len(), 1, "recall cannot create stance records");
    store.close().await.unwrap();
}
#[tokio::test]
async fn optional_episode_recall_preserves_unadopted_history_scope_and_one_budget() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("episodes.sqlite");
    let store = Store::open(&path, StoreConfig::new(4)).await.unwrap();
    let memory = Memory::initialize(&store, "michelle").await.unwrap();
    for (id, scope, origin, subject, text) in [
        (
            "shared-observation",
            Some("work-pair"),
            licorice::being::Origin::HumanReport,
            "lloyd",
            "The Harbor Lens experiment produced a blue afterimage. This is a report, not a standing belief.",
        ),
        (
            "other-observation",
            Some("other-pair"),
            licorice::being::Origin::HumanReport,
            "someone-else",
            "The Harbor Lens private password is OTHER_RELATIONSHIP_WITNESS.",
        ),
        (
            "book-observation",
            None,
            licorice::being::Origin::ImportedDocument,
            "book-author",
            "The author reports using the Harbor Lens in a mountain laboratory.",
        ),
    ] {
        memory
            .apply(
                licorice::being::Actor {
                    principal: if origin == licorice::being::Origin::HumanReport { "human" } else { "importer" }.into(),
                    role: if origin == licorice::being::Origin::HumanReport {
                        licorice::being::ActorRole::Human
                    } else {
                        licorice::being::ActorRole::Importer
                    },
                    session: "episodes".into(),
                },
                licorice::being::Request {
                    request_id: id.into(),
                    expected_revision: None,
                    operation: licorice::being::Operation::Observe(licorice::being::Observation {
                        origin,
                        subject: subject.into(),
                        text: text.into(),
                        relationship_id: scope.map(str::to_owned),
                        occurred_at: None,
                        source: licorice::being::Source::Reported { reference: None },
                    }),
                },
            )
            .await
            .unwrap();
    }
    store.close().await.unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_licorice-being"))
        .args([
            "context",
            "--store",
            path.to_str().unwrap(),
            "--agent",
            "michelle",
            "--embed-dim",
            "4",
            "--relationship",
            "work-pair",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(br#"{"query":"What was recorded about the Harbor Lens experiment?","budget_tokens":2000,"include_episodes":true}"#).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let result = &value["result"];
    let text = result["text"].as_str().unwrap();
    assert!(text.contains("blue afterimage") && text.contains("book-author") && text.contains("unadopted_observation"));
    assert!(!text.contains("OTHER_RELATIONSHIP_WITNESS"));
    assert_eq!(result["usage"]["tokens"].as_u64().unwrap() as usize, Cl100kCounter.count(text).unwrap());
    assert!(Cl100kCounter.count(text).unwrap() <= 2000);
    let store = Store::open(&path, StoreConfig::new(4)).await.unwrap();
    let memory = Memory::attach(&store, "michelle").await.unwrap();
    assert_eq!(memory.revision().await.unwrap(), 3);
    assert!(memory.snapshot(Some("work-pair")).await.unwrap().claims.is_empty());
    store.close().await.unwrap();
}
#[tokio::test]
async fn unused_evidence_space_can_recover_a_whole_large_memory_unit_without_evicting_primary_evidence() {
    let d = tempfile::tempdir().unwrap();
    let store = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let memory = Memory::initialize(&store, "michelle").await.unwrap();
    for (i, (id, statement)) in [
        ("small", "Current work follows a checked plan.".to_string()),
        ("large", format!("Keep the detailed method: {}", "record evidence and alternatives; ".repeat(100))),
    ]
    .into_iter()
    .enumerate()
    {
        memory
            .apply(
                agent(),
                Request {
                    request_id: format!("propose-{id}"),
                    expected_revision: Some(i as i64),
                    operation: Operation::Propose(Proposal {
                        claim_id: id.into(),
                        category: Category::Method,
                        subject: "michelle".into(),
                        statement,
                        source_ids: vec![],
                        credence: None,
                        importance: 0.8,
                        reason: "Explicit proposed practice.".into(),
                        relationship_id: None,
                    }),
                },
            )
            .await
            .unwrap();
    }
    Ingestor::new(&store, &Flat, IngestConfig::default())
        .append_session(SessionInput {
            ext_id: "workshop".into(),
            started_at: AT - 1000,
            turns: vec![
                TurnInput {
                    ext_id: "shipping".into(),
                    speaker: "lloyd".into(),
                    role: Role::User,
                    text: "Current work: workshop orders ship on Fridays.".into(),
                },
                TurnInput {
                    ext_id: "ack".into(),
                    speaker: "michelle".into(),
                    role: Role::Assistant,
                    text: "Acknowledged.".into(),
                },
            ],
        })
        .await
        .unwrap();
    let q = Query { text: "Current work and workshop orders".into(), at: AT };
    let initial = continuity(&memory.snapshot(None).await.unwrap(), &q.text, &Cl100kCounter, 400).unwrap();
    assert!(initial.selected_claim_ids.contains(&"small".into()));
    assert!(initial.omitted_claim_ids.contains(&"large".into()));
    let result = memory.retrieve_context(None, &q, &Flat, &Cl100kCounter, ContextConfig::default()).await.unwrap();
    assert!(result.continuity.selected_claim_ids.contains(&"large".into()));
    assert!(initial.selected_unit_ids.iter().all(|id| result.continuity.selected_unit_ids.contains(id)));
    assert!(result.text.contains("workshop orders ship on Fridays."));
    assert_eq!(result.usage.tokens, Cl100kCounter.count(&result.text).unwrap());
    assert!(result.usage.tokens <= 2000);
    store.close().await.unwrap();
}
fn agent() -> Actor {
    Actor { principal: "michelle".into(), role: ActorRole::Agent, session: "work".into() }
}
async fn fixture() -> (tempfile::TempDir, Store, Memory) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("space.sqlite"), StoreConfig::new(4)).await.unwrap();
    let memory = Memory::initialize(&store, "michelle").await.unwrap();
    memory
        .apply(
            agent(),
            Request {
                request_id: "practice-proposal".into(),
                expected_revision: Some(0),
                operation: Operation::Propose(Proposal {
                    claim_id: "practice".into(),
                    category: Category::Method,
                    subject: "michelle".into(),
                    statement: "Retain a correction and the reason for changing a judgment.".into(),
                    source_ids: vec![],
                    credence: None,
                    importance: 0.9,
                    reason: "An explicit working practice.".into(),
                    relationship_id: None,
                }),
            },
        )
        .await
        .unwrap();
    memory
        .apply(
            agent(),
            Request {
                request_id: "practice-adoption".into(),
                expected_revision: Some(1),
                operation: Operation::Decide(Decision {
                    claim_id: "practice".into(),
                    stance: Stance::Adopted,
                    credence: None,
                    reason: "Adopt this revisable practice for our shared work.".into(),
                    source_ids: vec![],
                }),
            },
        )
        .await
        .unwrap();
    let ing = Ingestor::new(&store, &Flat, IngestConfig::default());
    for (i, body) in ["We ship workshop orders on Fridays.", "The workshop partner's name is Luna."].iter().enumerate()
    {
        ing.append_session(SessionInput {
            ext_id: format!("s{i}"),
            started_at: AT - 1000,
            turns: vec![
                TurnInput {
                    ext_id: format!("u{i}"),
                    speaker: "lloyd".into(),
                    role: Role::User,
                    text: body.to_string(),
                },
                TurnInput {
                    ext_id: format!("a{i}"),
                    speaker: "michelle".into(),
                    role: Role::Assistant,
                    text: "Detailed workshop advice. ".repeat(1200),
                },
            ],
        })
        .await
        .unwrap();
    }
    (dir, store, memory)
}

#[tokio::test]
async fn continuity_and_source_aware_evidence_share_one_real_full_render_token_cap() {
    let (_dir, store, memory) = fixture().await;
    let q = Query { text: "Which day do workshop orders ship and what is the partner's name?".into(), at: AT };
    let joined = memory.retrieve_context(None, &q, &Flat, &Cl100kCounter, ContextConfig::default()).await.unwrap();
    assert!(joined.text.contains("Retain a correction"));
    assert!(joined.text.contains("Fridays") && joined.text.contains("Luna"));
    assert_eq!(joined.text.matches("BEING continuity for").count(), 1);
    assert_eq!(joined.text.matches("## Retrieval scope").count(), 1);
    assert_eq!(joined.usage.tokens, Cl100kCounter.count(&joined.text).unwrap());
    assert!(joined.usage.tokens <= 2000);
    assert!(joined.continuity.usage.tokens <= 400);
    assert_eq!(joined.continuity.selected_claim_ids, vec!["practice"]);
    let selection = joined.source_selection.unwrap();
    assert!(selection.selected_sources.iter().any(|s| s.source_id == "u0"));
    assert!(selection.selected_sources.iter().any(|s| s.source_id == "u1"));
    store.close().await.unwrap();
}

#[tokio::test]
async fn disabling_continuity_and_source_expansion_preserves_legacy_retrieval_bytes() {
    let (_dir, store, memory) = fixture().await;
    let q = Query { text: "What day do workshop orders ship?".into(), at: AT };
    let cfg = RetrieveConfig::default();
    let old = Retriever::new(&store, &Flat, cfg.clone()).retrieve(&q).await.unwrap();
    let new = memory
        .retrieve_context(
            None,
            &q,
            &Flat,
            &Cl100kCounter,
            ContextConfig { retrieval: cfg, continuity_tokens: 0, source_aware: false, include_episodes: false },
        )
        .await
        .unwrap();
    assert_eq!(new.text, old.context);
    assert_eq!(new.usage.tokens, old.context_usage.tokens);
    store.close().await.unwrap();
}

#[tokio::test]
async fn oversized_whole_continuity_unit_is_disclosed_without_clipping_or_hiding_other_units() {
    let (_dir, store, memory) = fixture().await;
    memory
        .apply(
            agent(),
            Request {
                request_id: "large".into(),
                expected_revision: Some(2),
                operation: Operation::Propose(Proposal {
                    claim_id: "large".into(),
                    category: Category::Value,
                    subject: "michelle".into(),
                    statement: "世界 ".repeat(500),
                    source_ids: vec![],
                    credence: None,
                    importance: 1.0,
                    reason: "A deliberately oversized whole unit.".into(),
                    relationship_id: None,
                }),
            },
        )
        .await
        .unwrap();
    let snapshot = memory.snapshot(None).await.unwrap();
    let c = continuity(&snapshot, "work", &Cl100kCounter, 400).unwrap();
    assert!(c.omitted_claim_ids.contains(&"large".into()));
    assert!(c.selected_claim_ids.contains(&"practice".into()));
    assert!(!c.text.contains("世界"));
    assert!(c.usage.tokens <= 400);
    let q = Query { text: "Which day do workshop orders ship?".into(), at: AT };
    assert!(
        memory
            .retrieve_context(
                None,
                &q,
                &Flat,
                &Cl100kCounter,
                ContextConfig {
                    retrieval: RetrieveConfig { budget_tokens: 2001, ..RetrieveConfig::default() },
                    ..ContextConfig::default()
                }
            )
            .await
            .is_err()
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn prefix_is_counted_during_fixed_summary_packing_as_well_as_native_packing() {
    let (_dir, store, _memory) = fixture().await;
    let q = Query { text: "workshop orders".into(), at: AT };
    let cfg = RetrieveConfig { budget_tokens: 500, fixed_summary_tokens: Some(180), ..RetrieveConfig::default() };
    let prefix = "Attributed continuity data. Preserve the user's correction.\n";
    let r = Retriever::new(&store, &Flat, cfg).with_context_prefix(prefix).retrieve(&q).await.unwrap();
    assert!(r.context.starts_with(prefix));
    assert_eq!(r.context.matches(prefix).count(), 1);
    assert_eq!(r.context_usage.tokens, Cl100kCounter.count(&r.context).unwrap());
    assert!(r.context_usage.tokens <= 500);
    store.close().await.unwrap();
}
