use licorice::being::ontology::{ReviewLimits, changed_dependencies, impact};
use licorice::being::*;
use licorice::{Store, StoreConfig};

fn actor() -> Actor {
    Actor { principal: "agent".into(), role: ActorRole::Agent, session: "s".into() }
}
async fn write(m: &Memory, id: &str, op: Operation) -> licorice::Result<Receipt> {
    m.apply(
        actor(),
        Request { request_id: id.into(), expected_revision: Some(m.revision().await.unwrap()), operation: op },
    )
    .await
}
async fn fixture() -> (tempfile::TempDir, Store, Memory) {
    let d = tempfile::tempdir().unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::initialize(&s, "agent").await.unwrap();
    m.apply(
        actor(),
        Request {
            request_id: "reading".into(),
            expected_revision: None,
            operation: Operation::Observe(Observation {
                origin: Origin::ImportedDocument,
                subject: "author".into(),
                text: "Two competing explanations were proposed, with a possible mixed explanation.".into(),
                relationship_id: None,
                occurred_at: None,
                source: Source::Reported { reference: Some("synthetic:source".into()) },
            }),
        },
    )
    .await
    .unwrap();
    for id in ["a", "b", "c"] {
        write(
            &m,
            id,
            Operation::Propose(Proposal {
                claim_id: id.into(),
                category: Category::Belief,
                subject: "site".into(),
                statement: format!("Hypothesis {id} about site use."),
                source_ids: vec!["reading".into()],
                credence: None,
                importance: 0.5,
                reason: "An alternative, not an endorsement.".into(),
                relationship_id: None,
            }),
        )
        .await
        .unwrap();
    }
    (d, s, m)
}
fn cref(id: &str, v: u32) -> PositionRef {
    PositionRef::Concept { concept_id: id.into(), version: v }
}
fn claim(id: &str) -> PositionRef {
    PositionRef::Claim { claim_id: id.into() }
}
fn concept(id: &str, v: u32, supersedes: Vec<PositionRef>) -> Operation {
    Operation::DefineConcept(ConceptDefinition {
        concept_id: id.into(),
        version: v,
        label: id.into(),
        definition: format!("Meaning {id} version {v}."),
        relationship_id: None,
        supersedes,
        source_ids: vec!["reading".into()],
        reason: "Refine meaning without erasing previous use.".into(),
    })
}
fn link(id: &str, from: PositionRef, to: PositionRef) -> Operation {
    Operation::LinkPositions(PositionLinkDefinition {
        link_id: id.into(),
        from,
        to,
        relation: PositionRelation::DependsOn,
        relationship_id: None,
        source_ids: vec!["reading".into()],
        reason: "This interpretation depends on that position.".into(),
    })
}
#[tokio::test]
async fn split_merge_meaning_is_immutable_and_review_is_bounded_without_cascading_stance() {
    let (d, s, m) = fixture().await;
    write(&m, "meaning", concept("use", 1, vec![])).await.unwrap();
    write(&m, "usage", link("a-use", claim("a"), cref("use", 1))).await.unwrap();
    write(&m, "ba", link("ba", claim("b"), claim("a"))).await.unwrap();
    write(&m, "cb", link("cb", claim("c"), claim("b"))).await.unwrap();
    write(&m, "ac", link("ac", claim("a"), claim("c"))).await.unwrap();
    write(&m, "split-one", concept("ritual", 1, vec![cref("use", 1)])).await.unwrap();
    write(&m, "split-two", concept("domestic", 1, vec![cref("use", 1)])).await.unwrap();
    write(&m, "merge", concept("mixed", 1, vec![cref("ritual", 1), cref("domestic", 1)])).await.unwrap();
    assert!(write(&m, "rewrite", concept("use", 1, vec![])).await.is_err());
    assert!(write(&m, "skip-version", concept("use", 3, vec![cref("use", 1)])).await.is_err());
    s.close().await.unwrap();
    let s = Store::open(d.path().join("s.sqlite"), StoreConfig::new(4)).await.unwrap();
    let m = Memory::attach(&s, "agent").await.unwrap();
    let snap = m.snapshot(None).await.unwrap();
    assert_eq!(snap.concepts.len(), 4);
    assert_eq!(snap.concepts[3].definition.definition, "Meaning use version 1.");
    let roots = changed_dependencies(&snap);
    assert_eq!(roots, vec![cref("use", 1)]);
    let limited = impact(&snap, &roots, ReviewLimits { nodes: 2, edges: 1, depth: 1 }).unwrap();
    assert!(limited.saturated);
    assert_eq!(limited.affected, vec![claim("a")]);
    let complete = impact(&snap, &roots, ReviewLimits::default()).unwrap();
    assert!(!complete.saturated);
    assert_eq!(complete.affected.len(), 3);
    assert!(snap.claims.iter().all(|c| c.stance == Stance::Considering));
    let packet = licorice::being::context::continuity(
        &snap,
        "site use ritual domestic",
        &licorice_providers::tokens::Cl100kCounter,
        2000,
    )
    .unwrap();
    assert!(packet.text.contains("Meaning use version 1.") && packet.text.contains("Meaning ritual version 1."));
    assert!(packet.usage.tokens <= 2000);
    write(
        &m,
        "retire",
        Operation::RetireLink {
            link_id: "a-use".into(),
            reason: "Reviewed after concept split; do not silently substitute the new meaning.".into(),
        },
    )
    .await
    .unwrap();
    assert!(changed_dependencies(&m.snapshot(None).await.unwrap()).is_empty());
    assert!(
        m.history()
            .await
            .unwrap()
            .iter()
            .any(|e| matches!(&e.request.operation,Operation::LinkPositions(l) if l.link_id=="a-use"))
    );
    s.close().await.unwrap();
}
fn hypotheses(mode: HypothesisMode, weights: [Option<f64>; 2], unknown: Option<f64>) -> HypothesisDefinition {
    HypothesisDefinition {
        set_id: "explanations".into(),
        version: 1,
        label: "Site-use alternatives".into(),
        mode,
        alternatives: vec![
            Alternative { claim_id: "a".into(), weight: weights[0] },
            Alternative { claim_id: "b".into(), weight: weights[1] },
        ],
        unknown_weight: unknown,
        relationship_id: None,
        source_ids: vec!["reading".into()],
        reason: "Explicit uncalibrated estimates; preserve the unknown option.".into(),
    }
}
#[tokio::test]
async fn exclusive_unknown_mass_compatible_frames_and_nonfinite_replay_are_distinct() {
    let (_d, s, m) = fixture().await;
    let mut h = hypotheses(HypothesisMode::Exclusive, [Some(0.5), Some(0.3)], Some(0.2));
    let mut bad = h.clone();
    bad.unknown_weight = Some(0.3);
    assert!(write(&m, "bad-sum", Operation::DefineHypotheses(bad)).await.is_err());
    write(&m, "weights", Operation::DefineHypotheses(h.clone())).await.unwrap();
    h.version = 2;
    h.alternatives[0].weight = Some(0.4);
    h.alternatives[1].weight = Some(0.4);
    write(&m, "new-weights", Operation::DefineHypotheses(h)).await.unwrap();
    let compatible = hypotheses(HypothesisMode::Compatible, [None, Some(0.9)], None);
    let req = Request {
        request_id: "compatible".into(),
        expected_revision: Some(m.revision().await.unwrap()),
        operation: Operation::DefineHypotheses(HypothesisDefinition { set_id: "compatible".into(), ..compatible }),
    };
    m.apply(actor(), req.clone()).await.unwrap();
    let mut poisoned = req.clone();
    if let Operation::DefineHypotheses(h) = &mut poisoned.operation {
        h.alternatives[0].weight = Some(f64::NAN);
    }
    assert!(m.apply(actor(), poisoned).await.is_err());
    assert!(m.apply(actor(), req).await.unwrap().replayed);
    let snap = m.snapshot(None).await.unwrap();
    assert_eq!(snap.hypotheses.len(), 3);
    assert!(snap.hypotheses.iter().all(|h| h.calibration == "uncalibrated_explicit_estimate"));
    assert!(snap.claims.iter().all(|c| c.stance == Stance::Considering));
    let packet = licorice::being::context::continuity(
        &snap,
        "Site-use alternatives",
        &licorice_providers::tokens::Cl100kCounter,
        2000,
    )
    .unwrap();
    assert!(packet.text.contains("unknown_weight") && packet.text.contains("uncalibrated_explicit_estimate"));
    assert!(
        packet.text.contains("Hypothesis a about site use.") && packet.text.contains("Hypothesis b about site use.")
    );
    s.close().await.unwrap();
}
