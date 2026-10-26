use licorice::being::*;
use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::{Store, StoreConfig};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use rusqlite::{Connection, params};
use serde_json::json;
use sha2::{Digest, Sha256};

const AGENT: &str = "michelle";

#[test]
fn native_cli_refuses_foreign_sqlite_before_migration_and_creates_private_new_stores() {
    let dir = tempfile::tempdir().unwrap();
    let foreign = dir.path().join("foreign.sqlite");
    let c = Connection::open(&foreign).unwrap();
    c.execute_batch(
        "CREATE TABLE notebook(message TEXT); INSERT INTO notebook VALUES('retain this original database');",
    )
    .unwrap();
    drop(c);
    let before = std::fs::read(&foreign).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_licorice-being"))
        .args(["init", "--store", foreign.to_str().unwrap(), "--agent", AGENT, "--embed-dim", "4"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert_eq!(std::fs::read(&foreign).unwrap(), before);
    let native = dir.path().join("new.sqlite");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_licorice-being"))
        .args(["init", "--store", native.to_str().unwrap(), "--agent", AGENT, "--embed-dim", "4"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stdout));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(native.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }
}
fn actor(role: ActorRole) -> Actor {
    Actor {
        principal: match role {
            ActorRole::Agent => AGENT,
            ActorRole::Human => "lloyd",
            ActorRole::Harness => "claude",
            ActorRole::Importer => "sidecar-import",
        }
        .into(),
        role,
        session: "work-1".into(),
    }
}
fn req(id: &str, revision: Option<i64>, operation: Operation) -> Request {
    Request { request_id: id.into(), expected_revision: revision, operation }
}
fn proposal(id: &str, sources: &[&str]) -> Proposal {
    Proposal {
        claim_id: id.into(),
        category: Category::Method,
        subject: AGENT.into(),
        statement: "Record failed predictions before changing the explanation.".into(),
        source_ids: sources.iter().map(|s| s.to_string()).collect(),
        credence: Some(0.99),
        importance: 0.8,
        reason: "Consider a practice prompted by our work.".into(),
        relationship_id: None,
    }
}
fn observation(origin: Origin, subject: &str, body: &str) -> Observation {
    Observation {
        origin,
        subject: subject.into(),
        text: body.into(),
        relationship_id: None,
        occurred_at: None,
        source: Source::Reported { reference: None },
    }
}
async fn fixture() -> (tempfile::TempDir, Store, Memory) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(4)).await.unwrap();
    let mem = Memory::initialize(&store, AGENT).await.unwrap();
    (dir, store, mem)
}

#[tokio::test]
async fn explicit_adoption_veto_and_revision_survive_restart_without_erasing_sources() {
    let (dir, store, mem) = fixture().await;
    mem.apply(
        actor(ActorRole::Importer),
        req(
            "reading",
            None,
            Operation::Observe(observation(
                Origin::ImportedDocument,
                "author-f",
                "The author reports recording failures.",
            )),
        ),
    )
    .await
    .unwrap();
    let mut p = proposal("notebook", &["reading"]);
    p.category = Category::Belief;
    mem.apply(actor(ActorRole::Agent), req("proposal", Some(1), Operation::Propose(p))).await.unwrap();
    assert_eq!(mem.snapshot(None).await.unwrap().claims[0].stance, Stance::Considering);
    let d = Decision {
        claim_id: "notebook".into(),
        stance: Stance::Declined,
        credence: Some(0.9),
        reason: "Plausible, but not yet an adopted practice.".into(),
        source_ids: vec![],
    };
    mem.apply(actor(ActorRole::Agent), req("veto", Some(2), Operation::Decide(d))).await.unwrap();
    mem.apply(
        actor(ActorRole::Human),
        req(
            "response",
            None,
            Operation::Observe(observation(
                Origin::HumanReport,
                "lloyd",
                "A written failure record helped us distinguish the alternatives.",
            )),
        ),
    )
    .await
    .unwrap();
    let d = Decision {
        claim_id: "notebook".into(),
        stance: Stance::Adopted,
        credence: None,
        reason: "Adopt this revisable working practice after the observed outcome.".into(),
        source_ids: vec!["response".into()],
    };
    mem.apply(actor(ActorRole::Agent), req("adopt", Some(4), Operation::Decide(d))).await.unwrap();
    let history = mem.history().await.unwrap();
    assert!(matches!(&history[2].request.operation,Operation::Decide(d) if d.stance==Stance::Declined));
    assert_eq!(history[0].actor.role, ActorRole::Importer);
    store.close().await.unwrap();
    drop(mem);
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(4)).await.unwrap();
    let mem = Memory::attach(&store, AGENT).await.unwrap();
    let s = mem.snapshot(None).await.unwrap();
    assert_eq!(s.revision, 5);
    assert_eq!(s.claims[0].stance, Stance::Adopted);
    assert_eq!(s.claims[0].credence, None);
    assert_eq!(s.claims[0].proposal.source_ids, vec!["reading", "response"]);
    assert_eq!(s.observations.len(), 2);
    store.close().await.unwrap();
}

#[tokio::test]
async fn actor_and_payload_cannot_impersonate_another_agent_or_a_human_source() {
    let (_dir, store, mem) = fixture().await;
    let proposal = req("p", Some(0), Operation::Propose(proposal("practice", &[])));
    assert!(mem.apply(actor(ActorRole::Human), proposal.clone()).await.is_err());
    let mut stranger = actor(ActorRole::Agent);
    stranger.principal = "ash".into();
    assert!(mem.apply(stranger, proposal.clone()).await.is_err());
    let o =
        req("fake-human", None, Operation::Observe(observation(Origin::HumanReport, "lloyd", "Invented agreement.")));
    assert!(mem.apply(actor(ActorRole::Agent), o).await.is_err());
    assert!(Memory::initialize(&store, "ash").await.is_err());
    assert!(Memory::attach(&store, "ash").await.is_err());
    assert_eq!(mem.revision().await.unwrap(), 0);
    let mut forged = serde_json::to_value(proposal).unwrap();
    assert!(serde_json::from_value::<Request>(forged.clone()).is_ok());
    forged["actor"] = json!("lloyd");
    assert!(serde_json::from_value::<Request>(forged).is_err());
    store.close().await.unwrap();
}

#[tokio::test]
async fn nonfinite_native_credence_cannot_alias_a_cached_unknown_credence_receipt() {
    let (_dir, store, mem) = fixture().await;
    let mut p = proposal("practice", &[]);
    p.credence = None;
    mem.apply(actor(ActorRole::Agent), req("p", Some(0), Operation::Propose(p.clone()))).await.unwrap();
    p.credence = Some(f64::NAN);
    assert!(mem.apply(actor(ActorRole::Agent), req("p", Some(0), Operation::Propose(p))).await.is_err());
    assert_eq!(mem.revision().await.unwrap(), 1);
    store.close().await.unwrap();
}

#[tokio::test]
async fn replay_requires_the_same_principal_session_and_request_even_after_other_events() {
    let (_dir, store, mem) = fixture().await;
    let request = req("p", Some(0), Operation::Propose(proposal("practice", &[])));
    let first = mem.apply(actor(ActorRole::Agent), request.clone()).await.unwrap();
    mem.apply(
        actor(ActorRole::Harness),
        req("boundary", None, Operation::Boundary { turn: "turn-1".into(), phase: BoundaryPhase::Completed }),
    )
    .await
    .unwrap();
    let replay = mem.apply(actor(ActorRole::Agent), request.clone()).await.unwrap();
    assert!(replay.replayed);
    assert_eq!(replay.revision, first.revision);
    let mut changed = actor(ActorRole::Agent);
    changed.session = "other-session".into();
    assert!(mem.apply(changed, request.clone()).await.is_err());
    let mut changed = request;
    changed.expected_revision = Some(2);
    assert!(mem.apply(actor(ActorRole::Agent), changed).await.is_err());
    assert_eq!(mem.revision().await.unwrap(), 2);
    store.close().await.unwrap();
}

#[tokio::test]
async fn concurrent_stale_reviews_have_one_winner_and_no_lost_stance() {
    let (_dir, store, mem) = fixture().await;
    mem.apply(actor(ActorRole::Agent), req("p", Some(0), Operation::Propose(proposal("practice", &[])))).await.unwrap();
    let d = |stance| Decision {
        claim_id: "practice".into(),
        stance,
        credence: None,
        reason: "Review the same snapshot.".into(),
        source_ids: vec![],
    };
    let (a, b) = tokio::join!(
        mem.apply(actor(ActorRole::Agent), req("adopt", Some(1), Operation::Decide(d(Stance::Adopted)))),
        mem.apply(actor(ActorRole::Agent), req("decline", Some(1), Operation::Decide(d(Stance::Declined))))
    );
    assert_ne!(a.is_ok(), b.is_ok());
    assert_eq!(mem.history().await.unwrap().len(), 2);
    assert_eq!(
        mem.snapshot(None).await.unwrap().claims[0].stance,
        if a.is_ok() { Stance::Adopted } else { Stance::Declined }
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn projection_receipt_failure_rolls_back_the_event_and_claim_together() {
    let (_dir, store, mem) = fixture().await;
    mem.apply(
        actor(ActorRole::Human),
        req("source", None, Operation::Observe(observation(Origin::HumanReport, "lloyd", "Our agreement."))),
    )
    .await
    .unwrap();
    store.write(|tx| {tx.conn().execute_batch("CREATE TRIGGER fail_being_source BEFORE INSERT ON being_claim_sources BEGIN SELECT RAISE(ABORT,'simulated interruption'); END")?;Ok(())}).await.unwrap();
    assert!(
        mem.apply(
            actor(ActorRole::Agent),
            req("proposal", Some(1), Operation::Propose(proposal("practice", &["source"])))
        )
        .await
        .is_err()
    );
    assert_eq!(mem.revision().await.unwrap(), 1);
    assert!(mem.snapshot(None).await.unwrap().claims.is_empty());
    let count: i64 =
        store.read(|c| Ok(c.query_row("SELECT count(*) FROM being_claims", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(count, 0);
    store.close().await.unwrap();
}

#[tokio::test]
async fn scoped_evidence_cannot_become_global_or_leak_to_a_different_relationship() {
    let (_dir, store, mem) = fixture().await;
    let mut o = observation(Origin::HumanReport, "lloyd", "A private handoff agreement.");
    o.relationship_id = Some("lloyd-michelle".into());
    mem.apply(actor(ActorRole::Human), req("source", None, Operation::Observe(o))).await.unwrap();
    let mut p = proposal("agreement", &["source"]);
    p.category = Category::Relationship;
    assert!(
        mem.apply(actor(ActorRole::Agent), req("wrong-scope", Some(1), Operation::Propose(p.clone()))).await.is_err()
    );
    p.relationship_id = Some("lloyd-michelle".into());
    mem.apply(actor(ActorRole::Agent), req("right-scope", Some(1), Operation::Propose(p))).await.unwrap();
    assert!(mem.snapshot(None).await.unwrap().claims.is_empty());
    assert!(mem.snapshot(Some("other-person")).await.unwrap().observations.is_empty());
    assert_eq!(mem.snapshot(Some("lloyd-michelle")).await.unwrap().observations.len(), 1);
    store.close().await.unwrap();
}

#[tokio::test]
async fn reading_a_life_cannot_become_this_agents_autobiography() {
    let (_dir, store, mem) = fixture().await;
    mem.apply(
        actor(ActorRole::Importer),
        req(
            "reading",
            None,
            Operation::Observe(observation(Origin::ImportedDocument, "author-f", "An author's narrated experience.")),
        ),
    )
    .await
    .unwrap();
    let mut p = proposal("borrowed-life", &["reading"]);
    p.category = Category::Autobiographical;
    assert!(mem.apply(actor(ActorRole::Agent), req("borrowed", Some(1), Operation::Propose(p))).await.is_err());
    let mut p = proposal("wrong-self", &[]);
    p.category = Category::SelfDescription;
    p.subject = "other-agent".into();
    assert!(mem.apply(actor(ActorRole::Agent), req("wrong-self", Some(1), Operation::Propose(p))).await.is_err());
    mem.apply(
        actor(ActorRole::Agent),
        req("practice", Some(1), Operation::Propose(proposal("inspired-method", &["reading"]))),
    )
    .await
    .unwrap();
    assert_eq!(mem.snapshot(None).await.unwrap().observations[0].assurance, Assurance::Reported);
    store.close().await.unwrap();
}

struct Flat;
#[async_trait::async_trait]
impl Embedder for Flat {
    fn id(&self) -> &str {
        "being-fixture"
    }
    fn dim(&self) -> usize {
        4
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(vec![vec![1., 0., 0., 0.]; texts.len()])
    }
}

#[tokio::test]
async fn participation_requires_canonical_text_time_role_and_explicit_session_binding() {
    let (_dir, store, mem) = fixture().await;
    const AT: i64 = 1683504000000;
    Ingestor::new(&store, &Flat, IngestConfig::default())
        .append_session(SessionInput {
            ext_id: "native-session".into(),
            started_at: AT,
            turns: vec![
                TurnInput {
                    ext_id: "u1".into(),
                    speaker: "lloyd".into(),
                    role: Role::User,
                    text: "The last hypothesis failed.".into(),
                },
                TurnInput {
                    ext_id: "a1".into(),
                    speaker: AGENT.into(),
                    role: Role::Assistant,
                    text: "I will preserve that failed prediction.".into(),
                },
            ],
        })
        .await
        .unwrap();
    let mut o = observation(Origin::DirectInteraction, AGENT, "I will preserve that failed prediction.");
    o.occurred_at = Some(AT);
    o.source = Source::NativeTurn { session: "native-session".into(), turn: "a1".into() };
    assert!(mem.apply(actor(ActorRole::Harness), req("unbound", None, Operation::Observe(o.clone()))).await.is_err());
    mem.apply(
        actor(ActorRole::Harness),
        req("bind", None, Operation::BindSession { session: "native-session".into() }),
    )
    .await
    .unwrap();
    let mut forged = o.clone();
    forged.text = "A fabricated private experience.".into();
    assert!(mem.apply(actor(ActorRole::Harness), req("forged", None, Operation::Observe(forged))).await.is_err());
    let mut forged = o.clone();
    forged.occurred_at = Some(AT + 1);
    assert!(mem.apply(actor(ActorRole::Harness), req("wrong-time", None, Operation::Observe(forged))).await.is_err());
    let mut forged = o.clone();
    forged.origin = Origin::HumanReport;
    assert!(mem.apply(actor(ActorRole::Harness), req("wrong-role", None, Operation::Observe(forged))).await.is_err());
    mem.apply(actor(ActorRole::Harness), req("participation", None, Operation::Observe(o))).await.unwrap();
    let mut p = proposal("our-work", &["participation"]);
    p.category = Category::Autobiographical;
    mem.apply(actor(ActorRole::Agent), req("autobiography", Some(2), Operation::Propose(p))).await.unwrap();
    assert_eq!(mem.snapshot(None).await.unwrap().observations[0].assurance, Assurance::CanonicalTurn);
    store.close().await.unwrap();
}

#[tokio::test]
async fn host_tool_receipt_binds_exact_text_and_cannot_be_self_attested_by_agent() {
    let (_dir, store, mem) = fixture().await;
    let mut o = observation(Origin::ToolResult, AGENT, "The experiment returned a negative result.");
    o.source = Source::HostReceipt {
        receipt_id: "tool-1".into(),
        body_blake3: blake3::hash(o.text.as_bytes()).to_hex().to_string(),
    };
    assert!(mem.apply(actor(ActorRole::Agent), req("agent-fake", None, Operation::Observe(o.clone()))).await.is_err());
    let mut changed = o.clone();
    changed.text = "A changed result.".into();
    assert!(mem.apply(actor(ActorRole::Harness), req("bad-digest", None, Operation::Observe(changed))).await.is_err());
    mem.apply(actor(ActorRole::Harness), req("tool-observation", None, Operation::Observe(o))).await.unwrap();
    let mut p = proposal("experiment", &["tool-observation"]);
    p.category = Category::Autobiographical;
    mem.apply(actor(ActorRole::Agent), req("experience", Some(1), Operation::Propose(p))).await.unwrap();
    store.close().await.unwrap();
}

fn sidecar(path: &std::path::Path, broken: bool) -> Vec<String> {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch("PRAGMA application_id=1111837006; PRAGMA user_version=1; CREATE TABLE owner(agent TEXT); INSERT INTO owner VALUES('michelle'); CREATE TABLE events(seq INTEGER PRIMARY KEY,request_id TEXT,body TEXT,digest TEXT,recorded_at INTEGER);").unwrap();
    let events = [
        json!({"request_id":"legacy-reading","kind":"observe","payload":{"origin":"imported_document","subject":"author-f","text":"An attributed reading."}}),
        json!({"request_id":"legacy-proposal","kind":"propose","expected_revision":1,"payload":{"claim_id":"legacy-method","category":"method","subject":AGENT,"statement":"Retain failed predictions.","source_ids":["legacy-reading"],"credence":null,"importance":0.8,"reason":"A considered practice."}}),
        json!({"request_id":"legacy-decision","kind":"decide","expected_revision":2,"payload":{"claim_id":if broken {"missing"} else {"legacy-method"},"stance":"adopted","credence":null,"source_ids":[],"reason":"An explicit historical adoption."}}),
    ];
    let bodies = events.iter().map(|v| v.to_string()).collect::<Vec<_>>();
    for (i, (event, body)) in events.iter().zip(&bodies).enumerate() {
        conn.execute(
            "INSERT INTO events VALUES(?1,?2,?3,?4,?5)",
            params![
                i as i64 + 1,
                event["request_id"].as_str().unwrap(),
                body,
                format!("{:x}", Sha256::digest(body.as_bytes())),
                1000 + i as i64
            ],
        )
        .unwrap();
    }
    bodies
}

#[tokio::test]
async fn legacy_import_is_atomic_replayable_and_preserves_original_receipts_and_assurance() {
    let (dir, store, mem) = fixture().await;
    let path = dir.path().join("prototype.sqlite");
    let bodies = sidecar(&path, false);
    let r = mem.import_sidecar(actor(ActorRole::Importer), "prototype-v1", &path).await.unwrap();
    assert_eq!(r.events, 3);
    assert_eq!(r.final_revision, 3);
    let s = mem.snapshot(None).await.unwrap();
    assert_eq!(s.claims[0].stance, Stance::Adopted);
    assert_eq!(s.observations[0].assurance, Assurance::LegacyImport);
    let originals: Vec<String> = store
        .read(|c| {
            Ok(c.prepare("SELECT original_body FROM being_import_rows ORDER BY original_seq")?
                .query_map([], |r| r.get(0))?
                .collect::<std::result::Result<_, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(originals, bodies);
    assert_eq!(mem.history().await.unwrap()[0].recorded_at, 1000);
    mem.apply(
        actor(ActorRole::Harness),
        req("later", None, Operation::Boundary { turn: "t1".into(), phase: BoundaryPhase::Completed }),
    )
    .await
    .unwrap();
    assert!(mem.import_sidecar(actor(ActorRole::Importer), "prototype-v1", &path).await.unwrap().replayed);
    assert_eq!(mem.revision().await.unwrap(), 4);
    let source = Connection::open(&path).unwrap();
    source.execute("UPDATE events SET digest='changed' WHERE seq=1", []).unwrap();
    drop(source);
    assert!(mem.import_sidecar(actor(ActorRole::Importer), "other-import", &path).await.is_err());
    store.close().await.unwrap();
}

#[tokio::test]
async fn invalid_late_import_operation_rolls_back_earlier_native_events_and_projections() {
    let (dir, store, mem) = fixture().await;
    let path = dir.path().join("broken.sqlite");
    sidecar(&path, true);
    assert!(mem.import_sidecar(actor(ActorRole::Importer), "broken-import", &path).await.is_err());
    assert_eq!(mem.revision().await.unwrap(), 0);
    assert!(mem.snapshot(None).await.unwrap().claims.is_empty());
    let count: i64 =
        store.read(|c| Ok(c.query_row("SELECT count(*) FROM being_imports", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(count, 0);
    store.close().await.unwrap();
}

#[tokio::test]
async fn uncertified_legacy_autobiography_can_be_suspended_without_promoting_its_evidence() {
    let (dir, store, mem) = fixture().await;
    let path = dir.path().join("legacy-experience.sqlite");
    sidecar(&path, false);
    let conn = Connection::open(&path).unwrap();
    for seq in [1, 2] {
        let body: String = conn.query_row("SELECT body FROM events WHERE seq=?1", [seq], |r| r.get(0)).unwrap();
        let mut body: serde_json::Value = serde_json::from_str(&body).unwrap();
        if seq == 1 {
            body["payload"]["origin"] = json!("direct_interaction");
            body["payload"]["subject"] = json!(AGENT);
        } else {
            body["payload"]["category"] = json!("autobiographical");
        }
        let body = body.to_string();
        conn.execute(
            "UPDATE events SET body=?1,digest=?2 WHERE seq=?3",
            params![body, format!("{:x}", Sha256::digest(body.as_bytes())), seq],
        )
        .unwrap();
    }
    drop(conn);
    mem.import_sidecar(actor(ActorRole::Importer), "legacy-experience", &path).await.unwrap();
    let d = Decision {
        claim_id: "legacy-method".into(),
        stance: Stance::Suspended,
        credence: None,
        reason: "The imported record lacks a certified participation source.".into(),
        source_ids: vec![],
    };
    mem.apply(actor(ActorRole::Agent), req("suspend", Some(3), Operation::Decide(d))).await.unwrap();
    let s = mem.snapshot(None).await.unwrap();
    assert_eq!(s.claims[0].stance, Stance::Suspended);
    assert_eq!(s.observations[0].assurance, Assurance::LegacyImport);
    store.close().await.unwrap();
}
