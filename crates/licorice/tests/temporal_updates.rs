//! Explicit, source-backed updates must preserve temporal history and replay safety.
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
struct HashEmbedder;
#[async_trait::async_trait]
impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        "temporal-test"
    }
    fn dim(&self) -> usize {
        64
    }
    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.; 64];
                for word in text.split_whitespace() {
                    v[blake3::hash(word.as_bytes()).as_bytes()[0] as usize % 64] += 1.;
                }
                v
            })
            .collect())
    }
}
use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever};
use licorice::temporal::Day;
use licorice::{Store, StoreConfig};
use serde_json::{Value, json};

fn ms(month: u32, day: u32) -> i64 {
    Day::from_ymd(2023, month, day).unwrap().start_ms()
}
fn fact(value: &str, expression: Option<&str>, update: Option<(&str, &str)>, evidence: &str) -> Value {
    json!({"subject":"Ada","predicate":"lives in","object":value,"object_is_entity":false,
      "statement":format!("Ada lives in {value}."),"negated":false,"salience":8,
      "time":{"expression":expression,"kind":"ongoing","anchor_turn":evidence},
      "update":update.map(|(old,quote)|json!({"previous_object":old,"evidence_quote":quote})),"evidence_turns":[evidence]})
}
async fn session(ing: &Ingestor<'_>, id: &str, at: i64, text: &str, facts: Vec<Value>) {
    ing.append_session(SessionInput {
        ext_id: id.into(),
        started_at: at,
        turns: vec![TurnInput {
            ext_id: format!("{id}:1"),
            speaker: "Ada".into(),
            role: Role::User,
            text: text.into(),
        }],
    })
    .await
    .unwrap();
    ing.apply_extraction(&format!("extract:{id}:0-0"), &json!({"entities":[],"facts":facts})).await.unwrap();
}
#[tokio::test]
async fn explicit_replacement_preserves_history_and_late_mentions_do_not_restore_old_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
    let cfg = IngestConfig { temporal_updates: true, ..IngestConfig::default() };
    let ing = Ingestor::new(&store, &HashEmbedder, cfg);
    session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "old:1")]).await;
    session(
        &ing,
        "move",
        ms(3, 10),
        "I moved from Oslo to Bergen on 1 February 2023.",
        vec![fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "move:1")],
    )
    .await;
    session(
        &ing,
        "late",
        ms(4, 1),
        "Back in January 2023 I lived in Oslo.",
        vec![fact("Oslo", Some("January 2023"), None, "late:1")],
    )
    .await;
    let rows=store.read(|conn| {
      let mut s=conn.prepare("SELECT object_value,status,valid_until,transaction_created_at,transaction_invalidated_at FROM facts ORDER BY id")?;
      Ok(s.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?,r.get::<_,Option<i64>>(2)?,r.get::<_,Option<i64>>(3)?,r.get::<_,Option<i64>>(4)?)))?.collect::<Result<Vec<_>,_>>()?)
    }).await.unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!((&rows[0].1, rows[0].2), (&"superseded".into(), Some(ms(2, 1))));
    assert_eq!((&rows[2].1, rows[2].2), (&"superseded".into(), Some(ms(2, 1))));
    assert_eq!(rows[1].1, "active");
    assert!(rows.iter().all(|r| r.3.is_some()));
    assert!(rows[0].4.is_some());
    let r = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
        .retrieve(&Query { text: "Where does Ada live now?".into(), at: ms(4, 2) })
        .await
        .unwrap();
    assert_eq!(r.facts.iter().map(|f| f.statement.as_str()).collect::<Vec<_>>(), ["Ada lives in Bergen."]);
    assert_eq!(
        store
            .read(|c| Ok(c.query_row("SELECT count(*) FROM fact_evidence", [], |r| r.get::<_, i64>(0))?))
            .await
            .unwrap(),
        3
    );
    store.close().await.unwrap();
}
#[tokio::test]
async fn adding_a_second_value_without_explicit_change_does_not_invalidate_the_first() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..IngestConfig::default() });
    session(
        &ing,
        "one",
        ms(1, 1),
        "I divide my time between Oslo and Bergen.",
        vec![fact("Oslo", None, None, "one:1"), fact("Bergen", None, None, "one:1")],
    )
    .await;
    assert_eq!(
        store
            .read(|c| Ok(c.query_row("SELECT count(*) FROM facts WHERE status='active'", [], |r| r.get::<_, i64>(0))?))
            .await
            .unwrap(),
        2
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn an_explicit_change_date_takes_precedence_over_a_states_now_expression() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "old:1")]).await;
    let mut next = fact("Bergen", Some("now"), Some(("Oslo", "moved from Oslo to Bergen")), "move:1");
    next["update"]["effective_expression"] = json!("1 February 2023");
    session(
        &ing,
        "move",
        ms(3, 10),
        "I moved from Oslo to Bergen on 1 February 2023 and live in Bergen now.",
        vec![next],
    )
    .await;
    let rows = store
        .read(|c| {
            let mut s = c.prepare("SELECT object_value,occurred_from,valid_until FROM facts ORDER BY id")?;
            Ok(s.query_map([], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, Option<i64>>(1)?, r.get::<_, Option<i64>>(2)?))
            })?
            .collect::<Result<Vec<_>, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(
        rows[0].2,
        Some(ms(2, 1)),
        "the date a move took effect differs from the later assertion that it is current"
    );
    assert_eq!(rows[1].1, Some(ms(2, 1)));
    store.close().await.unwrap();
}

#[tokio::test]
async fn an_unsupported_effective_date_cannot_invalidate_a_supported_old_state() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "old:1")]).await;
    ing.append_session(SessionInput {
        ext_id: "move".into(),
        started_at: ms(3, 10),
        turns: vec![TurnInput {
            ext_id: "move:1".into(),
            speaker: "Ada".into(),
            role: Role::User,
            text: "I moved from Oslo to Bergen and live in Bergen now.".into(),
        }],
    })
    .await
    .unwrap();
    let mut next = fact("Bergen", Some("now"), Some(("Oslo", "moved from Oslo to Bergen")), "move:1");
    next["update"]["effective_expression"] = json!("1 February 2023");
    assert!(
        ing.apply_extraction("extract:move:0-0", &json!({"entities":[],"facts":[next]})).await.is_err(),
        "an invented effective date is not supported by an exact update quote alone"
    );
    let (facts, updates, active): (i64, i64, i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM facts", [], |r| r.get(0))?,
                c.query_row("SELECT count(*) FROM fact_updates", [], |r| r.get(0))?,
                c.query_row("SELECT count(*) FROM facts WHERE object_value='Oslo' AND status='active'", [], |r| {
                    r.get(0)
                })?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((facts, updates, active), (1, 0, 1));
    store.close().await.unwrap();
}

#[tokio::test]
async fn explicit_updates_match_named_entities_and_literal_values_across_import_orders() {
    for (late_history, old_is_entity) in [(false, true), (true, true), (false, false), (true, false)] {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
        let mut old = fact("Oslo", Some("January 2023"), None, "old:1");
        old["object_is_entity"] = json!(old_is_entity);
        let mut next = fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "move:1");
        next["object_is_entity"] = json!(!old_is_entity);
        if !late_history {
            session(&ing, "old", ms(1, 1), "I lived in Oslo in January 2023.", vec![old.clone()]).await;
        }
        session(&ing, "move", ms(3, 10), "I moved from Oslo to Bergen on 1 February 2023.", vec![next]).await;
        if late_history {
            session(&ing, "old", ms(1, 1), "I lived in Oslo in January 2023.", vec![old]).await;
        }
        let (status,until):(String,Option<i64>)=store.read(|c|Ok(c.query_row("SELECT f.status,f.valid_until FROM facts f LEFT JOIN entities e ON e.id=f.object_id WHERE coalesce(e.name,f.object_value)='Oslo'",[],|r|Ok((r.get(0)?,r.get(1)?)))?)).await.unwrap();
        assert_eq!(
            (status, until),
            ("superseded".into(), Some(ms(2, 1))),
            "a new literal object must not make the previous named-entity state invisible to the update, including late imports"
        );
        let current = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
            .retrieve(&Query { text: "Where does Ada live now?".into(), at: ms(4, 2) })
            .await
            .unwrap();
        assert!(current.facts.iter().all(|f| f.statement != "Ada lives in Oslo."));
        assert!(current.facts.iter().any(|f| f.statement == "Ada lives in Bergen."));
        store.close().await.unwrap();
    }
}
#[tokio::test]
async fn explicit_entity_alias_targets_require_one_owner_and_literals_keep_their_spelling() {
    for mode in ["unique_alias", "ambiguous_alias", "literal_case"] {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
        let mut old = fact("Oslo", None, None, "old:1");
        old["object_is_entity"] = json!(mode != "literal_case");
        session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![old]).await;
        if mode != "literal_case" {
            let ambiguous = mode == "ambiguous_alias";
            store
                .write(move |w| {
                    let oslo =
                        w.conn().query_row("SELECT id FROM entities WHERE name='Oslo'", [], |r| r.get::<_, i64>(0))?;
                    w.add_alias(oslo, "christiania")?;
                    if ambiguous {
                        w.conn().execute("INSERT INTO entities(name,kind) VALUES ('Other Christiania','place')", [])?;
                        let other = w.conn().last_insert_rowid();
                        w.add_alias(other, "christiania")?;
                    }
                    Ok(())
                })
                .await
                .unwrap();
        }
        let previous = if mode == "literal_case" { "OSLO" } else { "Christiania" };
        let quote = format!("moved from {previous} to Bergen");
        session(
            &ing,
            "move",
            ms(3, 10),
            &format!("I {quote} on 1 February 2023."),
            vec![fact("Bergen", Some("1 February 2023"), Some((previous, &quote)), "move:1")],
        )
        .await;
        let status: String = store
            .read(|c| {
                Ok(c.query_row("SELECT status FROM facts WHERE statement='Ada lives in Oslo.'", [], |r| r.get(0))?)
            })
            .await
            .unwrap();
        assert_eq!(status, if mode == "unique_alias" { "superseded" } else { "active" }, "{mode}");
        store.close().await.unwrap();
    }
}
#[tokio::test]
async fn correcting_a_claimed_move_removes_its_false_invalidation_of_the_previous_home() {
    for old_is_entity in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
        let mut old_fact = fact("Oslo", Some("1 January 2023"), None, "old:1");
        old_fact["object_is_entity"] = json!(old_is_entity);
        session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![old_fact]).await;
        session(
            &ing,
            "wrong_move",
            ms(2, 1),
            "I moved from Oslo to Bergen on 1 February 2023.",
            vec![fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "wrong_move:1")],
        )
        .await;
        let mut corrected =
            fact("Rome", Some("1 March 2023"), Some(("Bergen", "I never moved to Bergen")), "correction:1");
        corrected["update"]["kind"] = json!("correction");
        session(
            &ing,
            "correction",
            ms(3, 10),
            "Correction: I never moved to Bergen. I stayed in Oslo until I moved to Rome on 1 March 2023.",
            vec![corrected],
        )
        .await;
        let old: (String, Option<i64>) = store
            .read(|c| {
                Ok(c.query_row("SELECT status,valid_until FROM facts WHERE statement='Ada lives in Oslo.'", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?)
            })
            .await
            .unwrap();
        assert_eq!(
            old,
            ("active".into(), None),
            "a retracted move cannot continue to invalidate the source-backed prior state; no unstated new transition is inferred"
        );
        // A separate explicit source-backed old-home transition supplies the actual
        // end date; the now-retracted February move must not override it.
        session(
            &ing,
            "actual_move",
            ms(3, 11),
            "I moved from Oslo to Rome on 1 March 2023.",
            vec![fact("Rome", Some("1 March 2023"), Some(("Oslo", "moved from Oslo to Rome")), "actual_move:1")],
        )
        .await;
        let rows = store
        .read(|c| {
            let mut s = c.prepare("SELECT coalesce(f.object_value,e.name),f.status,f.valid_until FROM facts f LEFT JOIN entities e ON e.id=f.object_id ORDER BY f.id")?;
            Ok(s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<i64>>(2)?)))?
                .collect::<Result<Vec<_>, _>>()?)
        })
        .await
        .unwrap();
        assert!(rows.iter().any(|r| r == &("Bergen".into(), "retracted".into(), None)));
        assert!(rows.iter().any(|r| r == &("Oslo".into(), "superseded".into(), Some(ms(3, 1)))));
        let at = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
            .retrieve(&Query { text: "Where did Ada live in February 2023?".into(), at: ms(4, 1) })
            .await
            .unwrap();
        assert!(at.facts.iter().any(|f| f.statement == "Ada lives in Oslo."));
        assert!(at.facts.iter().all(|f| f.statement != "Ada lives in Bergen."));
        store.close().await.unwrap();
    }
}
#[tokio::test]
async fn temporal_residence_rejects_missing_places_and_instant_state_classification() {
    for invalid in ["missing_place", "instant_state"] {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
        ing.append_session(SessionInput {
            ext_id: "s".into(),
            started_at: ms(1, 1),
            turns: vec![TurnInput {
                ext_id: "s:1".into(),
                speaker: "Ada".into(),
                role: Role::User,
                text: "I live in Oslo.".into(),
            }],
        })
        .await
        .unwrap();
        let mut output = fact("Oslo", None, None, "s:1");
        if invalid == "missing_place" {
            output["object"] = json!(null);
        } else {
            output["time"]["kind"] = json!("instant");
        }
        assert!(
            ing.apply_extraction("extract:s:0-0", &json!({"entities":[],"facts":[output]})).await.is_err(),
            "{invalid} must trigger repair rather than creating an unusable residence state"
        );
        let counts: (i64, i64) = store
            .read(|c| {
                Ok((
                    c.query_row("SELECT count(*) FROM facts", [], |r| r.get(0))?,
                    c.query_row("SELECT count(*) FROM applied_outputs", [], |r| r.get(0))?,
                ))
            })
            .await
            .unwrap();
        assert_eq!(counts, (0, 0));
        store.close().await.unwrap();
    }
}
#[tokio::test]
async fn an_explicit_update_with_invalid_current_evidence_requires_repair_instead_of_silent_omission() {
    for invalid in ["evidence", "anchor"] {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
        ing.append_session(SessionInput {
            ext_id: "s".into(),
            started_at: ms(1, 1),
            turns: vec![
                TurnInput {
                    ext_id: "s:1".into(),
                    speaker: "Ada".into(),
                    role: Role::User,
                    text: "I lived in Oslo.".into(),
                },
                TurnInput {
                    ext_id: "s:2".into(),
                    speaker: "Ada".into(),
                    role: Role::User,
                    text: "I moved from Oslo to Rome.".into(),
                },
            ],
        })
        .await
        .unwrap();
        let mut output = fact("Rome", None, Some(("Oslo", "moved from Oslo to Rome")), "s:2");
        if invalid == "evidence" {
            output["evidence_turns"] = json!(["s:1"]);
        } else {
            output["time"]["anchor_turn"] = json!("s:1");
        }
        let error = ing
            .apply_extraction("extract:s:1-1", &json!({"entities":[],"facts":[output]}))
            .await
            .expect_err("an update cannot be silently omitted while returning an apparently complete application");
        assert!(error.to_string().contains("s:2"), "repair feedback must name the allowed current source: {error}");
        let count: i64 =
            store.read(|c| Ok(c.query_row("SELECT count(*) FROM applied_outputs", [], |r| r.get(0))?)).await.unwrap();
        assert_eq!(count, 0);
        store.close().await.unwrap();
    }
}
#[tokio::test]
async fn an_explicit_lifetime_denial_can_correct_an_ongoing_state_without_inventing_a_time_span() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "old", ms(1, 1), "I live in Bergen.", vec![fact("Bergen", None, None, "old:1")]).await;
    let mut correction = fact("Bergen", None, Some(("Bergen", "I never lived in Bergen")), "correction:1");
    correction["negated"] = json!(true);
    correction["statement"] = json!("Ada never lived in Bergen.");
    correction["time"]["kind"] = json!("unknown");
    correction["update"]["kind"] = json!("correction");
    session(&ing, "correction", ms(3, 10), "Correction: I never lived in Bergen.", vec![correction]).await;
    let rows = store
        .read(|c| {
            let mut s = c.prepare("SELECT negated,time_kind,status FROM facts ORDER BY id")?;
            Ok(s.query_map([], |r| Ok((r.get::<_, bool>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)))?
                .collect::<Result<Vec<_>, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(rows, vec![(false, "ongoing".into(), "retracted".into()), (true, "unknown".into(), "active".into())]);
    store.close().await.unwrap();
}
#[tokio::test]
async fn corrected_transition_history_is_consistent_across_all_source_import_orders() {
    let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for entities in [false, true] {
        for order in orders {
            let dir = tempfile::tempdir().unwrap();
            let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
            let ing =
                Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
            let mut old = fact("Oslo", Some("1 January 2023"), None, "old:1");
            old["object_is_entity"] = json!(entities);
            let mut wrong =
                fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "wrong:1");
            wrong["object_is_entity"] = json!(entities);
            let mut denial = fact("Bergen", None, Some(("Bergen", "I never lived in Bergen")), "correction:1");
            denial["statement"] = json!("Ada never lived in Bergen.");
            denial["negated"] = json!(true);
            denial["time"]["kind"] = json!("unknown");
            denial["update"]["kind"] = json!("correction");
            let actual = fact("Rome", Some("1 March 2023"), Some(("Oslo", "moved from Oslo to Rome")), "correction:1");
            for source in order {
                match source {
                    0 => session(&ing, "old", ms(1, 1), "I live in Oslo.", vec![old.clone()]).await,
                    1 => {
                        session(
                            &ing,
                            "wrong",
                            ms(2, 1),
                            "I moved from Oslo to Bergen on 1 February 2023.",
                            vec![wrong.clone()],
                        )
                        .await
                    }
                    _ => {
                        session(
                            &ing,
                            "correction",
                            ms(3, 10),
                            "Correction: I never lived in Bergen. I moved from Oslo to Rome on 1 March 2023.",
                            vec![denial.clone(), actual.clone()],
                        )
                        .await
                    }
                }
            }
            let (rows,turns,updates)=store.read(|c|{
                let mut s=c.prepare("SELECT coalesce(f.object_value,e.name),f.negated,f.status,f.valid_until FROM facts f LEFT JOIN entities e ON e.id=f.object_id ORDER BY 1,2")?;
                let rows=s.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,bool>(1)?,r.get::<_,String>(2)?,r.get::<_,Option<i64>>(3)?)))?.collect::<Result<Vec<_>,_>>()?;
                Ok((rows,c.query_row("SELECT count(*) FROM turns",[],|r|r.get::<_,i64>(0))?,c.query_row("SELECT count(*) FROM fact_updates",[],|r|r.get::<_,i64>(0))?))
            }).await.unwrap();
            assert_eq!(
                rows,
                vec![
                    ("Bergen".into(), false, "retracted".into(), None),
                    ("Bergen".into(), true, "active".into(), None),
                    ("Oslo".into(), false, "superseded".into(), Some(ms(3, 1))),
                    ("Rome".into(), false, "active".into(), None)
                ],
                "import order {order:?}, entity objects {entities}"
            );
            assert_eq!((turns, updates), (3, 3));
            store.close().await.unwrap();
        }
    }
}
#[tokio::test]
async fn message_timestamp_is_the_relative_date_anchor_and_replay_checks_it() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("memory.sqlite"), StoreConfig::new(64)).await.unwrap();
    let cfg = IngestConfig { temporal_updates: true, ..IngestConfig::default() };
    let ing = Ingestor::new(&store, &HashEmbedder, cfg);
    let input = SessionInput {
        ext_id: "multi".into(),
        started_at: ms(5, 1),
        turns: vec![
            TurnInput { ext_id: "m1".into(), speaker: "Ada".into(), role: Role::User, text: "First message.".into() },
            TurnInput {
                ext_id: "m2".into(),
                speaker: "Ada".into(),
                role: Role::User,
                text: "I moved to Rome yesterday.".into(),
            },
        ],
    };
    ing.append_timed_session(input.clone(), vec![ms(5, 1), ms(5, 10)]).await.unwrap();
    let request = ing.extraction_requests("multi").await.unwrap().remove(0);
    assert!(request.user.contains("2023-05-10"));
    let output = json!({"entities":[],"facts":[fact("Rome",Some("yesterday"),None,"m2")]});
    ing.apply_extraction("extract:multi:0-1", &output).await.unwrap();
    let date =
        store.read(|c| Ok(c.query_row("SELECT occurred_from FROM facts", [], |r| r.get::<_, i64>(0))?)).await.unwrap();
    assert_eq!(date, ms(5, 9));
    assert!(ing.apply_extraction("extract:multi:0-1", &output).await.unwrap().replayed);
    assert!(ing.append_timed_session(input, vec![ms(5, 1), ms(5, 11)]).await.is_err());
    store.close().await.unwrap();
}

#[tokio::test]
async fn returning_to_a_previous_value_starts_a_new_state_and_coarse_dates_stay_coarse() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "a", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "a:1")]).await;
    session(
        &ing,
        "b",
        ms(2, 2),
        "I moved from Oslo to Bergen on 1 February 2023.",
        vec![fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "b:1")],
    )
    .await;
    session(
        &ing,
        "c",
        ms(3, 2),
        "I moved from Bergen to Oslo on 1 March 2023.",
        vec![fact("Oslo", Some("1 March 2023"), Some(("Bergen", "moved from Bergen to Oslo")), "c:1")],
    )
    .await;
    let states = store
        .read(|c| {
            let mut s = c.prepare("SELECT object_value,status FROM facts ORDER BY id")?;
            Ok(s.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<Result<Vec<_>, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(
        states,
        vec![
            ("Oslo".into(), "superseded".into()),
            ("Bergen".into(), "superseded".into()),
            ("Oslo".into(), "active".into())
        ]
    );
    session(
        &ing,
        "d",
        ms(5, 10),
        "I moved from Oslo to Rome in 2023.",
        vec![fact("Rome", Some("2023"), Some(("Oslo", "moved from Oslo to Rome")), "d:1")],
    )
    .await;
    let (precision,effective,basis)=store.read(|c|Ok(c.query_row("SELECT f.time_precision,u.effective_at,u.effective_basis FROM fact_updates u JOIN facts f ON f.id=u.next_fact_id ORDER BY u.id DESC LIMIT 1",[],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,String>(2)?)))?)).await.unwrap();
    assert_eq!(precision, "year");
    assert_eq!(effective, ms(5, 10));
    assert_eq!(basis, "known_by_source_date");
    store.close().await.unwrap();
}
#[tokio::test]
async fn unsupported_replacement_quote_rolls_back_new_facts_and_the_application_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "a", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "a:1")]).await;
    ing.append_session(SessionInput {
        ext_id: "b".into(),
        started_at: ms(2, 1),
        turns: vec![TurnInput {
            ext_id: "b:1".into(),
            speaker: "Ada".into(),
            role: Role::User,
            text: "I visited Bergen.".into(),
        }],
    })
    .await
    .unwrap();
    let output = json!({"entities":[],"facts":[fact("Bergen",None,Some(("Oslo","moved from Oslo to Bergen")),"b:1")]});
    assert!(ing.apply_extraction("extract:b:0-0", &output).await.is_err());
    assert_eq!(
        store.read(|c| Ok(c.query_row("SELECT count(*) FROM facts", [], |r| r.get::<_, i64>(0))?)).await.unwrap(),
        1
    );
    assert_eq!(
        store
            .read(|c| Ok(c.query_row("SELECT count(*) FROM applied_outputs WHERE id='extract:b:0-0'", [], |r| r
                .get::<_, i64>(0))?))
            .await
            .unwrap(),
        0
    );
    store.close().await.unwrap();
}

#[tokio::test]
async fn a_correction_retracts_the_wrong_assertion_without_erasing_its_source() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "wrong", ms(1, 1), "I live in Paris.", vec![fact("Paris", None, None, "wrong:1")]).await;
    let mut corrected = fact("Rome", None, Some(("Paris", "never lived in Paris")), "fix:1");
    corrected["update"]["kind"] = json!("correction");
    session(&ing, "fix", ms(3, 1), "Correction: I never lived in Paris. I live in Rome.", vec![corrected]).await;
    let (status, source_count) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT status FROM facts WHERE object_value='Paris'", [], |r| r.get::<_, String>(0))?,
                c.query_row("SELECT count(*) FROM fact_evidence", [], |r| r.get::<_, i64>(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(status, "retracted");
    assert_eq!(source_count, 2);
    let r = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
        .retrieve(&Query { text: "Where does Ada live now?".into(), at: ms(4, 1) })
        .await
        .unwrap();
    assert!(r.facts.iter().all(|f| !f.statement.contains("Paris")));
    assert!(r.facts.iter().any(|f| f.statement.contains("Rome")));
    store.close().await.unwrap();
}

#[tokio::test]
async fn explicit_negation_and_restoration_preserve_polarity_and_state_history() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("s"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "yes", ms(1, 1), "I live in Paris.", vec![fact("Paris", None, None, "yes:1")]).await;
    let mut no = fact("Paris", None, Some(("Paris", "no longer live in Paris")), "no:1");
    no["negated"] = json!(true);
    no["statement"] = json!("Ada no longer lives in Paris.");
    session(&ing, "no", ms(2, 1), "I no longer live in Paris.", vec![no]).await;
    session(
        &ing,
        "again",
        ms(3, 1),
        "I live in Paris again.",
        vec![fact("Paris", None, Some(("Paris", "live in Paris again")), "again:1")],
    )
    .await;
    let states = store
        .read(|c| {
            let mut s = c.prepare("SELECT negated,status FROM facts ORDER BY id")?;
            Ok(s.query_map([], |r| Ok((r.get::<_, bool>(0)?, r.get::<_, String>(1)?)))?
                .collect::<Result<Vec<_>, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(states, vec![(false, "superseded".into()), (true, "superseded".into()), (false, "active".into())]);
    store.close().await.unwrap();
}

#[tokio::test]
async fn entity_aliases_preserve_negation_and_restoration_polarity() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("aliases.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    for (index, id, text, negative, previous) in [
        (0, "yes", "I live in New York City (NYC).", false, None),
        (1, "no", "I no longer live in NYC.", true, Some(("NYC", "no longer live in NYC"))),
        (2, "again", "I live in NYC again.", false, Some(("NYC", "live in NYC again"))),
    ] {
        ing.append_session(SessionInput {
            ext_id: id.into(),
            started_at: ms(index + 1, 1),
            turns: vec![TurnInput {
                ext_id: format!("{id}:1"),
                speaker: "Ada".into(),
                role: Role::User,
                text: text.into(),
            }],
        })
        .await
        .unwrap();
        let mut state = fact("New York City", None, previous, &format!("{id}:1"));
        state["object_is_entity"] = json!(true);
        state["negated"] = json!(negative);
        state["statement"] =
            json!(if negative { "Ada no longer lives in New York City." } else { "Ada lives in New York City." });
        let output = json!({"entities":[{"name":"New York City","kind":"place","aliases":["NYC"]}],"facts":[state]});
        if negative {
            // An ambiguous nickname must still fail atomically. Resolve the
            // ambiguity administratively, then retry the identical application.
            let other = store
                .write(|w| {
                    w.conn().execute("INSERT INTO entities(name,kind) VALUES ('Other NYC','place')", [])?;
                    let other = w.conn().last_insert_rowid();
                    w.add_alias(other, "nyc")?;
                    Ok(other)
                })
                .await
                .unwrap();
            assert!(ing.apply_extraction(&format!("extract:{id}:0-0"), &output).await.is_err());
            let count = store
                .read(|db| Ok(db.query_row("SELECT count(*) FROM facts", [], |r| r.get::<_, i64>(0))?))
                .await
                .unwrap();
            assert_eq!(count, 1, "ambiguous polarity update must leave the old fact intact");
            store
                .write(move |w| {
                    w.conn().execute("DELETE FROM entity_aliases WHERE entity_id=?1 AND alias='nyc'", [other])?;
                    Ok(())
                })
                .await
                .unwrap();
        }
        ing.apply_extraction(&format!("extract:{id}:0-0"), &output).await.unwrap();
    }
    let (states, evidence, updates) = store
        .read(|db| {
            let mut stmt = db.prepare("SELECT negated,status,object_id FROM facts ORDER BY id")?;
            let states = stmt
                .query_map([], |r| Ok((r.get::<_, bool>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok((
                states,
                db.query_row("SELECT count(*) FROM fact_evidence", [], |r| r.get::<_, i64>(0))?,
                db.query_row("SELECT count(*) FROM fact_updates", [], |r| r.get::<_, i64>(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(
        states.iter().map(|(n, s, _)| (*n, s.as_str())).collect::<Vec<_>>(),
        vec![(false, "superseded"), (true, "superseded"), (false, "active")]
    );
    assert!(states.iter().all(|(_, _, id)| *id == states[0].2));
    assert_eq!((evidence, updates), (3, 2));
    store.close().await.unwrap();
}

#[derive(Default)]
struct CapturedStateCandidates(std::sync::Mutex<Vec<Vec<String>>>);
#[async_trait::async_trait]
impl licorice_providers::rerank::Reranker for CapturedStateCandidates {
    fn id(&self) -> &str {
        "historical-state-candidate-control"
    }
    async fn rank(
        &self,
        _: &str,
        candidates: &[licorice_providers::rerank::Candidate],
    ) -> Result<Vec<licorice_providers::rerank::Ranked>, licorice_providers::chat::ChatError> {
        self.0.lock().unwrap().push(candidates.iter().map(|c| c.text.clone()).collect());
        Ok(candidates
            .iter()
            .map(|c| licorice_providers::rerank::Ranked {
                id: c.id,
                score: if c.text.contains("Rome") { 1. } else { 0.1 },
            })
            .collect())
    }
}
#[tokio::test]
async fn historical_state_bounds_exclude_disjoint_transitions_before_scoring_and_keep_comparison_queries_complete() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("history.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "jan", ms(1, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "jan:1")]).await;
    session(
        &ing,
        "feb",
        ms(2, 1),
        "I moved from Oslo to Bergen on 1 February 2023.",
        vec![fact("Bergen", Some("1 February 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "feb:1")],
    )
    .await;
    session(
        &ing,
        "mar",
        ms(3, 1),
        "I moved from Bergen to Rome on 1 March 2023.",
        vec![fact("Rome", Some("1 March 2023"), Some(("Bergen", "moved from Bergen to Rome")), "mar:1")],
    )
    .await;
    let scorer = CapturedStateCandidates::default();
    let cfg = RetrieveConfig { chunks: 0, summaries: 0, filter_state_window: true, ..Default::default() };
    for (month, city) in [("January", "Oslo"), ("February", "Bergen"), ("March", "Rome")] {
        let query = Query { text: format!("Where was Ada living on 15 {month} 2023?"), at: ms(4, 1) };
        let result =
            Retriever::new(&store, &HashEmbedder, cfg.clone()).with_reranker(&scorer).retrieve(&query).await.unwrap();
        assert_eq!(result.facts.len(), 1, "only a state that can overlap the requested window may reach the scorer");
        assert!(result.facts[0].statement.contains(city));
        let candidates = scorer.0.lock().unwrap().last().unwrap().clone();
        assert_eq!(candidates.len(), 1);
        assert!(candidates[0].starts_with(&format!("Ada lives in {city}.")));
        assert!(candidates[0].contains("[event date:"), "existing date annotations remain available to the scorer");
    }
    let compared = Retriever::new(&store, &HashEmbedder, cfg)
        .retrieve(&Query { text: "Which homes did Ada live in before 15 March 2023?".into(), at: ms(4, 1) })
        .await
        .unwrap();
    assert_eq!(compared.facts.len(), 3, "ordering questions need evidence across the date anchor");
    store.close().await.unwrap();
}

#[tokio::test]
async fn state_window_filter_keeps_unknown_coarse_and_event_time_uncertainty() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("uncertain.sqlite"), StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig { temporal_updates: true, ..Default::default() });
    session(&ing, "unknown", ms(3, 1), "I live in Oslo.", vec![fact("Oslo", None, None, "unknown:1")]).await;
    // Being stated on a day is not proof that the state began on that day.
    session(
        &ing,
        "stated",
        ms(3, 2),
        "I also live in Rome on 1 March 2023.",
        vec![fact("Rome", Some("1 March 2023"), None, "stated:1")],
    )
    .await;
    session(
        &ing,
        "coarse",
        ms(4, 1),
        "I moved from Oslo to Bergen in March 2023.",
        vec![fact("Bergen", Some("March 2023"), Some(("Oslo", "moved from Oslo to Bergen")), "coarse:1")],
    )
    .await;
    let mut event = fact("Lisbon", Some("15 December 2022"), None, "event:1");
    event["predicate"] = json!("visited");
    event["statement"] = json!("Ada visited Lisbon.");
    event["time"]["kind"] = json!("instant");
    session(&ing, "event", ms(4, 2), "I visited Lisbon on 15 December 2022.", vec![event]).await;
    let query = Query { text: "Where was Ada living on 15 January 2023?".into(), at: ms(4, 3) };
    let cfg = RetrieveConfig { filter_state_window: true, chunks: 0, summaries: 0, ..Default::default() };
    let result = Retriever::new(&store, &HashEmbedder, cfg).retrieve(&query).await.unwrap();
    assert_eq!(
        result.facts.len(),
        4,
        "unknown starts, coarse replacement dates and point events must not gain invented exact state bounds"
    );
    assert!(!RetrieveConfig::default().filter_state_window, "the unvalidated retrieval option stays off by default");
    store.close().await.unwrap();
}

struct BiasedActivityEmbedder;
#[async_trait::async_trait]
impl Embedder for BiasedActivityEmbedder {
    fn id(&self) -> &str {
        "activity-context-control"
    }
    fn dim(&self) -> usize {
        2
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                if matches!(kind, EmbedKind::Document) && text.contains("swam") { vec![0.01, 1.] } else { vec![1., 0.] }
            })
            .collect())
    }
}
#[derive(Default)]
struct ClockActivityModel(std::sync::Mutex<Vec<licorice_providers::chat::ModelRequest>>);
#[async_trait::async_trait]
impl licorice_providers::chat::ChatModel for ClockActivityModel {
    fn id(&self) -> &str {
        "clock-activity-control"
    }
    async fn complete(
        &self,
        request: &licorice_providers::chat::ModelRequest,
    ) -> Result<Value, licorice_providers::chat::ChatError> {
        self.0.lock().unwrap().push(request.clone());
        let payload: Value = serde_json::from_str(&request.user).unwrap();
        let swimming = payload["context"]["reference_date_utc"] == "2023-01-03";
        let scores: serde_json::Map<String, Value> = payload["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let selected = c["text"].as_str().unwrap().contains(if swimming { "swam" } else { "ran" });
                (c["id"].as_i64().unwrap().to_string(), json!(if selected { 1. } else { 0. }))
            })
            .collect();
        Ok(json!({"scores":scores}))
    }
}
#[tokio::test]
async fn relative_query_clock_and_unique_speaker_reach_scorer_with_same_candidate_pool() {
    use std::sync::Arc;
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path().join("clock.sqlite"), StoreConfig::new(2)).await.unwrap();
    let ing = Ingestor::new(&store, &BiasedActivityEmbedder, IngestConfig::default());
    for (day, verb) in [(1, "ran"), (2, "swam")] {
        let id = format!("activity{day}");
        let mut event = fact(verb, Some(&format!("{day} January 2023")), None, &format!("{id}:1"));
        event["predicate"] = json!("did");
        event["statement"] = json!(format!("Ada {verb}."));
        event["time"]["kind"] = json!("instant");
        session(&ing, &id, ms(1, day), &format!("I {verb}."), vec![event]).await;
    }
    let model = Arc::new(ClockActivityModel::default());
    let scorer = licorice_providers::rerank::ChatReranker(model.clone());
    let cfg = RetrieveConfig {
        query_speaker: Some("Ada".into()),
        w_fact: 1.,
        w_session: 0.,
        alpha: 1.,
        final_facts: 1,
        chunks: 0,
        summaries: 0,
        ..Default::default()
    };
    for (day, verb) in [(2, "ran"), (3, "swam")] {
        let result = Retriever::new(&store, &BiasedActivityEmbedder, cfg.clone())
            .with_reranker(&scorer)
            .retrieve(&Query { text: "What did I do yesterday?".into(), at: ms(1, day) })
            .await
            .unwrap();
        assert_eq!(
            result.facts[0].statement,
            format!("Ada {verb}."),
            "the same relative question must be anchored to its own reference day"
        );
    }
    {
        let calls = model.0.lock().unwrap();
        let a: Value = serde_json::from_str(&calls[0].user).unwrap();
        let b: Value = serde_json::from_str(&calls[1].user).unwrap();
        assert_eq!(a["candidates"], b["candidates"], "control holds ordered candidate inputs fixed");
        assert_eq!(a["context"]["speaker"], "Ada");
        assert_eq!(b["context"]["speaker"], "Ada");
        assert_ne!(calls[0].id, calls[1].id, "different clock contexts cannot share scorer cache identity");
    }
    for (speaker, text) in [(Some("missing"), "What did I do yesterday?"), (Some("Ada"), "What did Ada do yesterday?")]
    {
        let cfg = RetrieveConfig { query_speaker: speaker.map(str::to_owned), ..cfg.clone() };
        Retriever::new(&store, &BiasedActivityEmbedder, cfg)
            .with_reranker(&scorer)
            .retrieve(&Query { text: text.into(), at: ms(1, 3) })
            .await
            .unwrap();
        let payload: Value = serde_json::from_str(&model.0.lock().unwrap().last().unwrap().user).unwrap();
        assert!(payload["context"]["speaker"].is_null(), "no guessed first-person actor");
    }
    store
        .write(|w| {
            w.conn().execute("INSERT INTO entities(name,kind) VALUES ('Other Ada','person')", [])?;
            let other = w.conn().last_insert_rowid();
            w.add_alias(other, "ada")?;
            Ok(())
        })
        .await
        .unwrap();
    for limit in [0, 8] {
        let cfg = RetrieveConfig { linked_entities: limit, ..cfg.clone() };
        Retriever::new(&store, &BiasedActivityEmbedder, cfg)
            .with_reranker(&scorer)
            .retrieve(&Query { text: "What did I do yesterday?".into(), at: ms(1, 3) })
            .await
            .unwrap();
        let payload: Value = serde_json::from_str(&model.0.lock().unwrap().last().unwrap().user).unwrap();
        assert!(payload["context"]["speaker"].is_null(), "disabled or ambiguous linking cannot resolve the actor");
    }
    store.close().await.unwrap();
}
