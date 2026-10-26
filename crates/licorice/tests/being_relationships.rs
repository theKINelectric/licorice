use licorice::being::context::continuity;
use licorice::being::*;
use licorice::{Store, StoreConfig};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};

fn actor(role: ActorRole) -> Actor {
    Actor { principal: if role == ActorRole::Agent { "agent" } else { "human" }.into(), role, session: "s1".into() }
}
async fn write(m: &Memory, id: &str, op: Operation) -> licorice::Result<Receipt> {
    m.apply(
        actor(ActorRole::Agent),
        Request { request_id: id.into(), expected_revision: Some(m.revision().await.unwrap()), operation: op },
    )
    .await
}
async fn report(m: &Memory, id: &str, origin: Origin, subject: &str, scope: &str) -> Receipt {
    let role = if origin == Origin::AgentReflection { ActorRole::Agent } else { ActorRole::Human };
    m.apply(
        actor(role),
        Request {
            request_id: id.into(),
            expected_revision: None,
            operation: Operation::Observe(Observation {
                origin,
                subject: subject.into(),
                text: format!("Attributed outcome {id}"),
                relationship_id: Some(scope.into()),
                occurred_at: None,
                source: Source::Reported { reference: None },
            }),
        },
    )
    .await
    .unwrap()
}
async fn setup() -> (tempfile::TempDir, Store, Memory) {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::initialize(&s, "agent").await.unwrap();
    report(&m, "request", Origin::HumanReport, "human", "pair").await;
    write(
        &m,
        "relation",
        Operation::DefineRelationship(RelationshipDefinition {
            relationship_id: "pair".into(),
            version: 1,
            members: vec!["agent".into(), "human".into()],
            description: "Workshop collaborators, from this agent's perspective.".into(),
            boundaries: vec!["Do not infer reciprocal emotional states.".into()],
            source_ids: vec!["request".into()],
            reason: "Explicit relationship namespace.".into(),
        }),
    )
    .await
    .unwrap();
    write(
        &m,
        "promise",
        Operation::Propose(Proposal {
            claim_id: "delivery".into(),
            category: Category::Commitment,
            subject: "agent".into(),
            statement: "Deliver the revised workshop notes.".into(),
            source_ids: vec!["request".into()],
            credence: None,
            importance: 0.9,
            reason: "A requested undertaking.".into(),
            relationship_id: Some("pair".into()),
        }),
    )
    .await
    .unwrap();
    write(
        &m,
        "adoption",
        Operation::Decide(Decision {
            claim_id: "delivery".into(),
            stance: Stance::Adopted,
            credence: None,
            reason: "Undertake it.".into(),
            source_ids: vec![],
        }),
    )
    .await
    .unwrap();
    write(
        &m,
        "register",
        Operation::RegisterCommitment(CommitmentDefinition {
            claim_id: "delivery".into(),
            counterparty: "human".into(),
            next_action: "Write and verify the notes.".into(),
            due_at: Some(100),
            acknowledgement_source: Some("request".into()),
        }),
    )
    .await
    .unwrap();
    (d, s, m)
}
fn transition(status: CommitmentStatus, source: &str) -> Operation {
    Operation::TransitionCommitment(CommitmentTransition {
        claim_id: "delivery".into(),
        status,
        reason: "Explicit observed lifecycle change.".into(),
        source_ids: vec![source.into()],
    })
}

#[tokio::test]
async fn commitments_need_outcome_witness_and_survive_fresh_process_scope_and_budget() {
    let (d, s, m) = setup().await;
    report(&m, "thought", Origin::AgentReflection, "agent", "pair").await;
    let before = m.revision().await.unwrap();
    assert!(write(&m, "false-success", transition(CommitmentStatus::Fulfilled, "thought")).await.is_err());
    assert_eq!(m.revision().await.unwrap(), before);
    report(&m, "delivered", Origin::HumanReport, "human", "pair").await;
    let op = transition(CommitmentStatus::Fulfilled, "delivered");
    let req =
        Request { request_id: "success".into(), expected_revision: Some(m.revision().await.unwrap()), operation: op };
    m.apply(actor(ActorRole::Agent), req.clone()).await.unwrap();
    assert!(m.apply(actor(ActorRole::Agent), req).await.unwrap().replayed);
    assert!(m.snapshot(None).await.unwrap().commitments.is_empty());
    assert!(m.snapshot(Some("different-human")).await.unwrap().relationships.is_empty());
    s.close().await.unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::attach(&s, "agent").await.unwrap();
    let snap = m.snapshot(Some("pair")).await.unwrap();
    assert_eq!(snap.commitments[0].status, CommitmentStatus::Fulfilled);
    assert_eq!(snap.relationships[0].perspective, "agent");
    let packet = continuity(&snap, "delivery workshop notes", &Cl100kCounter, 2000).unwrap();
    assert!(packet.text.contains("fulfilled") && packet.text.contains("delivered"));
    assert_eq!(packet.usage.tokens, Cl100kCounter.count(&packet.text).unwrap());
    assert!(packet.usage.tokens <= 2000);
    s.close().await.unwrap();
}

#[tokio::test]
async fn release_reopen_and_relationship_revision_do_not_erase_history_or_change_counterparty() {
    let (_d, s, m) = setup().await;
    report(&m, "thought", Origin::AgentReflection, "agent", "pair").await;
    assert!(write(&m, "unilateral-release", transition(CommitmentStatus::Released, "thought")).await.is_err());
    report(&m, "other-release", Origin::HumanReport, "someone-else", "pair").await;
    assert!(write(&m, "other-release-attempt", transition(CommitmentStatus::Released, "other-release")).await.is_err());
    report(&m, "release", Origin::HumanReport, "human", "pair").await;
    write(&m, "release-it", transition(CommitmentStatus::Released, "release")).await.unwrap();
    assert!(write(&m, "terminal-rewrite", transition(CommitmentStatus::Fulfilled, "release")).await.is_err());
    write(&m, "reopen", transition(CommitmentStatus::Active, "release")).await.unwrap();
    let mut relation = m.snapshot(Some("pair")).await.unwrap().relationships[0].definition.clone();
    relation.version = 2;
    relation.description = "Revised working boundary.".into();
    relation.members[1] = "someone-else".into();
    assert!(write(&m, "replace-human", Operation::DefineRelationship(relation.clone())).await.is_err());
    relation.members[1] = "human".into();
    write(&m, "revise-boundary", Operation::DefineRelationship(relation)).await.unwrap();
    let history = m.history().await.unwrap();
    assert!(history.iter().any(
        |e| matches!(&e.request.operation,Operation::TransitionCommitment(t) if t.status==CommitmentStatus::Released)
    ));
    assert_eq!(m.snapshot(Some("pair")).await.unwrap().relationships[0].definition.version, 2);
    s.close().await.unwrap();
}
