use licorice_game::{being::*, hostile_hospitality as hh, *};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use std::task::Poll;

async fn game() -> GameMemory {
    GameMemory::new("test-world", "main", vec!["buddy".into(), "sarge".into()]).await.unwrap()
}

#[tokio::test]
async fn rosters_are_explicit_and_empty_lineage_forks_remain_restorable() {
    for roster in [
        vec![],
        vec!["buddy".into(), "buddy".into()],
        vec!["../escape".into()],
        (0..17).map(|i| format!("character-{i}")).collect(),
    ] {
        assert!(GameMemory::new("world", "main", roster).await.is_err());
    }
    let g = game().await;
    assert!(g.fork("main").await.is_err());
    let fork = g.fork("alternate").await.unwrap();
    let mut nested = fork.fork("nested").await.unwrap();
    let source = observe(&mut nested, "first", "First event after two empty forks.").await;
    let saved = nested.checkpoint().unwrap();
    let restored = GameMemory::restore(&saved).await.unwrap();
    assert_eq!(saved.to_json().unwrap(), restored.checkpoint().unwrap().to_json().unwrap());
    assert_eq!(nested.perceptions("buddy").unwrap().count(), 1);
    assert!(nested.perceptions("unknown").is_err());
    assert!(restored.recall("buddy", None, "First event", 2000, 1, 60).await.unwrap().source_ids.contains(&source));
    g.close().await.unwrap();
    fork.close().await.unwrap();
    nested.close().await.unwrap();
    restored.close().await.unwrap();
}
fn perception(id: &str, text: &str) -> Perception {
    Perception {
        event_id: id.into(),
        tick: 0,
        place: "garden".into(),
        subject: "buddy".into(),
        text: text.into(),
        channel: Channel::Witnessed,
        cue: None,
        relationship_id: None,
        fulfills: None,
    }
}
fn method(id: &str, source: &str) -> Operation {
    Operation::Propose(Proposal {
        claim_id: id.into(),
        category: Category::Method,
        subject: "buddy".into(),
        statement: "Isolate the live waterline before tending it.".into(),
        source_ids: vec![source.into()],
        credence: None,
        importance: 0.8,
        reason: "Consider the observed failure.".into(),
        relationship_id: None,
    })
}
fn decision(id: &str, stance: Stance) -> Operation {
    Operation::Decide(Decision {
        claim_id: id.into(),
        stance,
        credence: None,
        reason: "Explicit controller decision after reviewing the source.".into(),
        source_ids: vec![],
    })
}
async fn observe(g: &mut GameMemory, id: &str, text: &str) -> String {
    g.perceive("buddy", perception(id, text)).await.unwrap().source_id.unwrap()
}

#[tokio::test]
async fn perspectives_are_isolated_and_reports_are_not_adoption() {
    let mut g = game().await;
    let buddy = observe(&mut g, "bite-1", "I witnessed the supply hose being bitten.").await;
    let mut report = perception("radio-1", "Buddy reports that a plant bit a supply hose.");
    report.channel = Channel::Report { speaker: "buddy".into() };
    let sarge = g.perceive("sarge", report).await.unwrap().source_id.unwrap();
    assert_ne!(buddy, sarge);
    let b = g.snapshot("buddy", None).await.unwrap();
    let s = g.snapshot("sarge", None).await.unwrap();
    assert!(b.claims.is_empty() && s.claims.is_empty());
    assert_eq!(b.observations.len(), 1);
    assert!(b.observations[0].observation.text.contains("witnessed"));
    assert!(s.observations[0].observation.text.contains("reports"));
    assert!(g.deliberate("sarge", "foreign-source", 1, method("method", &buddy)).await.is_err());
    g.close().await.unwrap();
}

#[tokio::test]
async fn exact_event_replay_is_idempotent_and_conflicting_ids_are_atomic() {
    let mut g = game().await;
    let p = perception("one", "A particular event.");
    let receipt = g.perceive("buddy", p.clone()).await.unwrap();
    g.advance(300).unwrap();
    assert!(g.perceive("buddy", p).await.unwrap().replayed);
    let before = g.checkpoint().unwrap().to_json().unwrap();
    assert!(g.perceive("buddy", perception("one", "Changed history.")).await.is_err());
    assert_eq!(before, g.checkpoint().unwrap().to_json().unwrap());
    assert_eq!(g.revision("buddy").await.unwrap(), receipt.revision);
    assert_eq!(g.checkpoint().unwrap().operation_count(), 1);
    g.close().await.unwrap();
}

#[tokio::test]
async fn invalid_perceptions_never_replace_existing_history() {
    let mut g = game().await;
    let before = g.checkpoint().unwrap().to_json().unwrap();
    let mut future = perception("future", "Future event.");
    future.tick = 1;
    let mut forged = perception("report", "Someone claims a job is finished.");
    forged.channel = Channel::Report { speaker: "captain".into() };
    forged.fulfills = Some("promise".into());
    let mut empty = perception("empty", " ");
    empty.place.clear();
    let mut huge = perception("huge", "x");
    huge.text = "x".repeat(8193);
    for p in [future, forged, empty, huge] {
        assert!(g.perceive("buddy", p).await.is_err());
    }
    assert!(g.perceive("outsider", perception("one", "No owner.")).await.is_err());
    assert_eq!(before, g.checkpoint().unwrap().to_json().unwrap());
    assert!(g.advance(1).is_ok());
    assert!(g.advance(0).is_err());
    g.close().await.unwrap();
}

#[tokio::test]
async fn proposal_adoption_suspension_and_reconsideration_remain_explicit() {
    let mut g = game().await;
    let source = observe(&mut g, "bite", "A live waterline was severed.").await;
    g.deliberate("buddy", "consider", 1, method("isolate", &source)).await.unwrap();
    assert_eq!(g.snapshot("buddy", None).await.unwrap().claims[0].stance, Stance::Considering);
    assert!(g.deliberate("buddy", "stale", 1, decision("isolate", Stance::Adopted)).await.is_err());
    g.deliberate("buddy", "adopt", 2, decision("isolate", Stance::Adopted)).await.unwrap();
    let counter = observe(&mut g, "counter", "Isolation interrupted a safe emergency delivery.").await;
    let mut revised = match decision("isolate", Stance::Suspended) {
        Operation::Decide(d) => d,
        _ => unreachable!(),
    };
    revised.source_ids = vec![counter];
    revised.reason = "Reconsider the overly broad precaution after its counterexample.".into();
    g.deliberate("buddy", "suspend", 4, Operation::Decide(revised)).await.unwrap();
    let snapshot = g.snapshot("buddy", None).await.unwrap();
    assert_eq!(snapshot.claims[0].stance, Stance::Suspended);
    assert_eq!(snapshot.claims[0].proposal.source_ids.len(), 2);
    assert_eq!(g.history("buddy").await.unwrap().len(), 5);
    assert_eq!(g.revision("sarge").await.unwrap(), 0);
    let restored = GameMemory::restore(&g.checkpoint().unwrap()).await.unwrap();
    assert_eq!(
        serde_json::to_value(snapshot).unwrap(),
        serde_json::to_value(restored.snapshot("buddy", None).await.unwrap()).unwrap()
    );
    g.close().await.unwrap();
    restored.close().await.unwrap();
}

#[tokio::test]
async fn old_save_and_sibling_forks_do_not_adopt_abandoned_futures() {
    let mut g = game().await;
    observe(&mut g, "before", "Before the fork.").await;
    let old = g.checkpoint().unwrap();
    let mut sibling = g.fork("alternate").await.unwrap();
    g.advance(20).unwrap();
    sibling.advance(30).unwrap();
    observe(&mut g, "future", "Only the original future.").await;
    observe(&mut sibling, "future", "Only the alternate future.").await;
    assert_eq!(g.snapshot("buddy", None).await.unwrap().observations.len(), 2);
    assert_eq!(sibling.snapshot("buddy", None).await.unwrap().observations.len(), 2);
    let alternate = sibling.checkpoint().unwrap().to_json().unwrap();
    let reopened = GameMemory::restore(&Checkpoint::from_json(&alternate).unwrap()).await.unwrap();
    assert_eq!(alternate, reopened.checkpoint().unwrap().to_json().unwrap());
    g.restore_into(&old).await.unwrap();
    assert_eq!(g.tick(), 0);
    assert_eq!(g.snapshot("buddy", None).await.unwrap().observations.len(), 1);
    assert!(!g.recall("buddy", None, "future", 2000, 1, 180).await.unwrap().text.contains("original future"));
    g.close().await.unwrap();
    sibling.close().await.unwrap();
    reopened.close().await.unwrap();
}

#[tokio::test]
async fn corrupted_unknown_or_wrong_schema_save_is_refused_without_replacement() {
    let mut g = game().await;
    observe(&mut g, "keep", "Preserve this memory.").await;
    let text = g.checkpoint().unwrap().to_json().unwrap();
    let checkpoint = Checkpoint::from_json(&text).unwrap();
    assert!(checkpoint.check_world("test-world", "main", 0).is_ok());
    assert!(checkpoint.check_world("other-world", "main", 0).is_err());
    assert!(checkpoint.check_world("test-world", "sibling", 0).is_err());
    assert!(checkpoint.check_world("test-world", "main", 1).is_err());
    let mut bad: serde_json::Value = serde_json::from_str(&text).unwrap();
    bad["tick"] = 100.into();
    let unchecked: Checkpoint = serde_json::from_value(bad.clone()).unwrap();
    assert!(g.restore_into(&unchecked).await.is_err());
    bad["version"] = 999.into();
    assert!(Checkpoint::from_json(&bad.to_string()).is_err());
    bad["extra"] = true.into();
    assert!(Checkpoint::from_json(&bad.to_string()).is_err());
    assert!(Checkpoint::from_json(&" ".repeat(MAX_SAVE_BYTES + 1)).is_err());
    assert_eq!(text, g.checkpoint().unwrap().to_json().unwrap());
    g.close().await.unwrap();
}

#[tokio::test]
async fn callbacks_expire_on_policy_memory_world_restore_and_ticks() {
    let mut g = game().await;
    let stamp = g.recall("buddy", None, "current work", 2000, 7, 10).await.unwrap().stamp;
    assert!(g.validate_reply(&stamp, 7).await.is_ok());
    assert!(g.validate_reply(&stamp, 8).await.is_err());
    let mut other = game().await;
    assert!(other.validate_reply(&stamp, 7).await.is_err());
    let checkpoint = g.checkpoint().unwrap();
    g.restore_into(&checkpoint).await.unwrap();
    assert!(g.validate_reply(&stamp, 7).await.is_err());
    let stamp = g.recall("buddy", None, "work", 2000, 7, 10).await.unwrap().stamp;
    observe(&mut g, "new", "A new relevant occurrence.").await;
    assert!(g.validate_reply(&stamp, 7).await.is_err());
    let stamp = g.recall("buddy", None, "work", 2000, 7, 10).await.unwrap().stamp;
    g.advance(10).unwrap();
    assert!(g.validate_reply(&stamp, 7).await.is_ok());
    g.advance(11).unwrap();
    assert!(g.validate_reply(&stamp, 7).await.is_err());
    assert!(g.apply_reply(&stamp, 7, "late", decision("none", Stance::Adopted)).await.is_err());
    other.advance(u64::MAX).unwrap();
    assert!(other.recall("buddy", None, "work", 2000, 7, 1).await.is_err());
    g.close().await.unwrap();
    other.close().await.unwrap();
}

#[tokio::test]
async fn nonfinite_and_raw_observation_commands_cannot_alias_a_valid_replay() {
    let mut g = game().await;
    let source = observe(&mut g, "source", "A source.").await;
    let op = method("one", &source);
    g.deliberate("buddy", "request", 1, op.clone()).await.unwrap();
    let mut bad = match op {
        Operation::Propose(p) => p,
        _ => unreachable!(),
    };
    bad.credence = Some(f64::NAN);
    assert!(g.deliberate("buddy", "request", 1, Operation::Propose(bad)).await.is_err());
    let raw = Operation::Observe(Observation {
        origin: Origin::AgentReflection,
        subject: "buddy".into(),
        text: "Invent a receipt.".into(),
        relationship_id: None,
        occurred_at: None,
        source: Source::Reported { reference: None },
    });
    assert!(g.deliberate("buddy", "raw", 2, raw).await.is_err());
    assert_eq!(g.revision("buddy").await.unwrap(), 2);
    g.close().await.unwrap();
}

#[tokio::test]
async fn reported_participation_does_not_become_autobiography() {
    let mut g = game().await;
    let mut p = perception("rumour", "Someone says Buddy performed a rescue.");
    p.channel = Channel::Report { speaker: "captain".into() };
    let source = g.perceive("buddy", p).await.unwrap().source_id.unwrap();
    let mut proposal = match method("rescue", &source) {
        Operation::Propose(p) => p,
        _ => unreachable!(),
    };
    proposal.category = Category::Autobiographical;
    proposal.statement = "I performed the rescue.".into();
    assert!(g.deliberate("buddy", "autobiography", 1, Operation::Propose(proposal)).await.is_err());
    g.close().await.unwrap();
}

#[tokio::test]
async fn only_matching_host_outcomes_can_fulfill_a_structured_promise() {
    let mut g = game().await;
    let mut p = perception("order", "Captain asks Buddy to repair the supply line.");
    p.relationship_id = Some("crew-captain".into());
    let source = g.perceive("buddy", p).await.unwrap().source_id.unwrap();
    let relation = RelationshipDefinition {
        relationship_id: "crew-captain".into(),
        version: 1,
        members: vec!["buddy".into(), "captain".into()],
        description: "Crew work with the captain.".into(),
        boundaries: vec!["Actions require the game's authority.".into()],
        source_ids: vec![source.clone()],
        reason: "An actual request.".into(),
    };
    g.deliberate("buddy", "relationship", 1, Operation::DefineRelationship(relation)).await.unwrap();
    let mut p = match method("repair-promise", &source) {
        Operation::Propose(p) => p,
        _ => unreachable!(),
    };
    p.category = Category::Commitment;
    p.relationship_id = Some("crew-captain".into());
    p.statement = "Repair the supply line.".into();
    g.deliberate("buddy", "consider-promise", 2, Operation::Propose(p)).await.unwrap();
    g.deliberate("buddy", "adopt-promise", 3, decision("repair-promise", Stance::Adopted)).await.unwrap();
    g.deliberate(
        "buddy",
        "register",
        4,
        Operation::RegisterCommitment(CommitmentDefinition {
            claim_id: "repair-promise".into(),
            counterparty: "captain".into(),
            next_action: "Repair the line.".into(),
            due_at: None,
            acknowledgement_source: None,
        }),
    )
    .await
    .unwrap();
    let transition = |source: String| {
        Operation::TransitionCommitment(CommitmentTransition {
            claim_id: "repair-promise".into(),
            status: CommitmentStatus::Fulfilled,
            reason: "Check the actual outcome.".into(),
            source_ids: vec![source],
        })
    };
    assert!(g.deliberate("buddy", "unearned", 5, transition(source)).await.is_err());
    let wrong = observe(&mut g, "wrong", "A different line was repaired.").await;
    assert!(g.deliberate("buddy", "wrong-outcome", 6, transition(wrong)).await.is_err());
    let mut done = perception("done", "The game admitted the repair; flow was restored.");
    done.relationship_id = Some("crew-captain".into());
    done.fulfills = Some("repair-promise".into());
    let actual = g.perceive("buddy", done).await.unwrap().source_id.unwrap();
    g.deliberate("buddy", "earned", 7, transition(actual.clone())).await.unwrap();
    assert_eq!(
        g.snapshot("buddy", Some("crew-captain")).await.unwrap().commitments[0].status,
        CommitmentStatus::Fulfilled
    );
    g.advance(50).unwrap();
    g.deliberate(
        "buddy",
        "reopen",
        8,
        Operation::TransitionCommitment(CommitmentTransition {
            claim_id: "repair-promise".into(),
            status: CommitmentStatus::Active,
            reason: "A new break requires another repair.".into(),
            source_ids: vec![actual.clone()],
        }),
    )
    .await
    .unwrap();
    assert!(g.deliberate("buddy", "earned", 7, transition(actual.clone())).await.unwrap().replayed);
    assert!(g.deliberate("buddy", "reuse-old-outcome", 9, transition(actual)).await.is_err());
    let mut delayed = perception("old-delayed", "An old repair report delivered after the new break.");
    delayed.relationship_id = Some("crew-captain".into());
    delayed.fulfills = Some("repair-promise".into());
    assert!(g.perceive("buddy", delayed).await.is_err());
    let mut new_done = perception("new-done", "The new repair actually completed.");
    new_done.tick = 50;
    new_done.relationship_id = Some("crew-captain".into());
    new_done.fulfills = Some("repair-promise".into());
    let new_source = g.perceive("buddy", new_done).await.unwrap().source_id.unwrap();
    g.deliberate("buddy", "earned-again", 10, transition(new_source)).await.unwrap();
    g.close().await.unwrap();
}

#[tokio::test]
async fn outcome_receipts_require_a_currently_active_registered_commitment() {
    let mut g = game().await;
    let mut unearned = perception("unearned-proof", "A repair before anyone made this promise.");
    unearned.fulfills = Some("future-promise".into());
    assert!(g.perceive("buddy", unearned).await.is_err());
    assert_eq!(g.checkpoint().unwrap().operation_count(), 0);
    g.close().await.unwrap();
}

#[tokio::test]
async fn rendered_unicode_memory_stays_under_one_cap_and_protects_claim_sources() {
    let mut g = game().await;
    let old = observe(&mut g, "important", "The old supply line failed beside the fern.").await;
    g.deliberate("buddy", "propose", 1, method("precaution", &old)).await.unwrap();
    g.deliberate("buddy", "adopt", 2, decision("precaution", Stance::Adopted)).await.unwrap();
    for i in 0..130 {
        observe(
            &mut g,
            &format!("routine-{i}"),
            &format!("Routine garden 日本語 🍌 observation {i}. {}", "soil and water ".repeat(12)),
        )
        .await;
    }
    let packet = g.recall("buddy", None, "live waterline fern", 2000, 1, 60).await.unwrap();
    assert!(packet.tokens <= 2000);
    assert_eq!(packet.tokens, Cl100kCounter.count(&packet.text).unwrap());
    assert!(packet.episode_candidates_saturated);
    assert!(packet.source_ids.contains(&old));
    assert!(packet.text.contains("old supply line failed"));
    assert!(g.recall("buddy", None, "x", 2001, 1, 60).await.is_err());
    assert!(g.recall("buddy", None, "x", 0, 1, 60).await.is_err());
    assert!(g.recall("buddy", None, "x", 2000, 1, 0).await.is_err());
    g.close().await.unwrap();
}

#[tokio::test]
async fn scoped_episodes_are_not_shared_between_relationships() {
    let mut g = game().await;
    for (id, scope, text) in
        [("a", "captain-a", "Private incident with Alpha."), ("b", "captain-b", "Private incident with Beta.")]
    {
        let mut p = perception(id, text);
        p.relationship_id = Some(scope.into());
        g.perceive("buddy", p).await.unwrap();
    }
    let packet = g.recall("buddy", Some("captain-a"), "Private incident", 2000, 1, 60).await.unwrap();
    assert!(packet.text.contains("Alpha"));
    assert!(!packet.text.contains("Beta"));
    g.close().await.unwrap();
}

#[tokio::test]
async fn cancelling_an_inflight_write_refuses_cache_use_until_rebuild() {
    let mut g = game().await;
    let mut pending = Box::pin(g.perceive("buddy", perception("cancel", "An interrupted host write.")));
    let was_pending = std::future::poll_fn(|cx| Poll::Ready(matches!(pending.as_mut().poll(cx), Poll::Pending))).await;
    drop(pending);
    if was_pending {
        assert!(g.snapshot("buddy", None).await.is_err());
        assert!(g.hh_lessons("buddy").is_err());
        let saved = g.checkpoint().unwrap();
        assert_eq!(saved.operation_count(), 0);
        g.restore_into(&saved).await.unwrap();
        assert_eq!(g.revision("buddy").await.unwrap(), 0);
    } else {
        assert_eq!(g.checkpoint().unwrap().operation_count(), 1);
    }
    g.close().await.unwrap();
}

#[tokio::test]
async fn hh_compatibility_matches_lesson_presence_and_every_policy_combination() {
    for mask in 0..4 {
        let mut g = game().await;
        for (bit, actor, kind) in
            [(1, hh::BUDDY, hh::LessonKind::PowerInterrupted), (2, hh::SARGE, hh::LessonKind::PlantBitHose)]
        {
            if mask & bit != 0 {
                let incident = hh::Incident {
                    event_id: bit as u64,
                    tick: 0,
                    region: [0, 0],
                    subject: actor.into(),
                    text: "The original admitted incident.".into(),
                    kind,
                };
                g.perceive(actor, incident.witnessed().unwrap()).await.unwrap();
            }
        }
        for bits in 0..64 {
            let p = hh::Policy {
                caretaker: bits & 1 != 0,
                allow_bait: bits & 2 != 0,
                maintenance: bits & 4 != 0,
                irrigation_connected: bits & 8 != 0,
                hatch_open: bits & 16 != 0,
                gripped: bits & 32 != 0,
            };
            // Independent boolean reference table derived from the current
            // prepare_bait / secure_hatch predicates, not a running game.
            let baseline = hh::Behavior {
                prepare_bait: mask & 1 != 0 && bits & 3 == 3,
                secure_hatch: mask & 2 != 0 && bits & 61 == 29,
            };
            let result = hh::compare(&g, p, baseline).unwrap();
            assert!(result.matches, "mask={mask}, policy={bits}");
            assert_eq!(result.authoritative, baseline);
        }
        g.close().await.unwrap();
    }
}

#[tokio::test]
async fn hh_repeated_incidents_survive_reload_and_reports_do_not_create_legacy_lessons() {
    let mut g = game().await;
    for (event_id, tick) in [(50, 10), (60, 20)] {
        g.advance(tick).unwrap();
        let incident = hh::Incident {
            event_id,
            tick,
            region: [1, -1],
            subject: "buddy".into(),
            text: "Witnessed a particular outage.".into(),
            kind: hh::LessonKind::PowerInterrupted,
        };
        g.perceive("buddy", incident.witnessed().unwrap()).await.unwrap();
    }
    let mut report = perception("radio", "An outage report, not direct experience.");
    report.cue = Some(hh::LessonKind::PowerInterrupted.cue().into());
    report.channel = Channel::Report { speaker: "buddy".into() };
    g.perceive("sarge", report).await.unwrap();
    let lessons = g.hh_lessons("buddy").unwrap();
    assert_eq!(lessons[0].occurrences, 2);
    assert_eq!((lessons[0].first_event, lessons[0].last_event), (50, 60));
    assert!(g.hh_lessons("sarge").unwrap().is_empty());
    let restored = GameMemory::restore(&g.checkpoint().unwrap()).await.unwrap();
    assert_eq!(lessons, restored.hh_lessons("buddy").unwrap());
    let baseline = hh::Behavior { prepare_bait: false, secure_hatch: false };
    let mismatch = hh::compare(
        &g,
        hh::Policy {
            caretaker: true,
            allow_bait: true,
            maintenance: false,
            irrigation_connected: false,
            hatch_open: false,
            gripped: false,
        },
        baseline,
    )
    .unwrap();
    assert!(!mismatch.matches);
    assert_eq!(mismatch.authoritative, baseline);
    g.close().await.unwrap();
    restored.close().await.unwrap();
}
