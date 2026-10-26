//! Ingestion end to end with a deterministic embedder and hand-written model output.

mod common;

use common::space;
use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Intent, Query, RetrieveConfig, Retriever};
use licorice::temporal::Day;
use licorice::{Filter, Index, LexicalKind, Store, StoreConfig, VectorKind};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use serde_json::json;

/// Bag-of-words hashing embedder: deterministic, and identical texts embed identically.
struct HashEmbedder;

#[tokio::test]
async fn focused_themes_retain_whole_relevant_content_from_an_oversized_session() {
    use licorice::retrieve::{SelectionMode, SummaryMode};
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo and Rome.").await;
    ing.apply_summary(
        "summary:D1",
        &json!({"keys":["Ada"],"themes":[
            {"title":"Accounting","summary":"Accounting books. ".repeat(300)},
            {"title":"Travel","summary":"Ada visited Oslo and Rome."}
        ]}),
    )
    .await
    .unwrap();
    let mut prefixes = Vec::new();
    for selection in [SelectionMode::Relevance, SelectionMode::Mmr] {
        let cfg = RetrieveConfig {
            budget_tokens: 200,
            fixed_summary_tokens: Some(80),
            summary_mode: SummaryMode::Themes,
            selection,
            final_facts: 0,
            chunks: 0,
            ..RetrieveConfig::default()
        };
        let r = Retriever::new(&store, &HashEmbedder, cfg)
            .retrieve(&Query { text: "Where did Ada travel?".into(), at: may_8() })
            .await
            .unwrap();
        assert!(r.context.contains("Travel: Ada visited Oslo and Rome."), "{}", r.context);
        assert!(!r.context.contains("Accounting books"));
        assert!(r.context_usage.tokens <= 80);
        prefixes.push(r.context);
    }
    assert_eq!(prefixes[0], prefixes[1]);
    store.close().await.unwrap();
}

#[tokio::test]
async fn focused_theme_caps_exclude_retired_versions_and_require_explicit_allowance() {
    use licorice::retrieve::SummaryMode;
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    for session in ["D1", "D2", "D3"] {
        identity_session(&ing, session, "Ada travels.").await;
        ing.apply_summary(
            &format!("summary:{session}"),
            &json!({"keys":[],"themes":[
                {"title":"Travel","summary":"Ada travels to Oslo."},
                {"title":"Food","summary":"Ada travels for food."},
                {"title":"Music","summary":"Ada travels for music."},
                {"title":"Retired","summary":"Ada travels RETIRED_MARKER"}
            ]}),
        )
        .await
        .unwrap();
    }
    // Fixture of an older retired theme alongside current themes.
    store
        .write(|w| {
            let ids: Vec<i64> = {
                let mut stmt = w.conn().prepare("SELECT id FROM summary_keys WHERE text LIKE '%RETIRED_MARKER%'")?;
                stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?
            };
            for id in ids {
                w.conn().execute("UPDATE summary_keys SET live=0 WHERE id=?1", [id])?;
                w.retire_vectors(VectorKind::Key, id)?;
            }
            Ok(())
        })
        .await
        .unwrap();
    let query = Query { text: "Ada travels".into(), at: may_8() };
    let cfg = RetrieveConfig { summary_mode: SummaryMode::Themes, ..RetrieveConfig::default() };
    assert!(matches!(
        Retriever::new(&store, &HashEmbedder, cfg.clone()).retrieve(&query).await,
        Err(licorice::Error::InvalidInput(_))
    ));
    let cfg = RetrieveConfig { fixed_summary_tokens: Some(600), final_facts: 0, chunks: 0, ..cfg };
    let a = Retriever::new(&store, &HashEmbedder, cfg.clone()).retrieve(&query).await.unwrap();
    let b = Retriever::new(&store, &HashEmbedder, cfg).retrieve(&query).await.unwrap();
    assert_eq!(a.context, b.context);
    assert!(!a.context.contains("RETIRED_MARKER"));
    assert_eq!(a.sessions.len(), 2);
    assert_eq!(a.fixed_summary.unwrap().theme_ids.len(), 4);
    for session in a.sessions {
        assert_eq!(
            ["Travel:", "Food:", "Music:"].into_iter().filter(|title| session.summary.contains(title)).count(),
            2
        );
    }
    store.close().await.unwrap();
}

#[tokio::test]
async fn fixed_summaries_survive_fact_pressure_and_match_across_selectors() {
    use licorice::retrieve::SelectionMode;
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo.").await;
    ing.apply_extraction(
        "extract:D1:0-0",
        &json!({"entities":[],"facts":[fact(&"Ada visited Oslo. ".repeat(35),false,None,"unknown","D1:1")]}),
    )
    .await
    .unwrap();
    ing.apply_summary(
        "summary:D1",
        &json!({"keys":["Ada Oslo"],"themes":[{
            "title":"Travel", "summary":format!("SUMMARY_SUPPORT {}", "Ada enjoys travel. ".repeat(12))
        }]}),
    )
    .await
    .unwrap();
    let query = Query { text: "Where did Ada travel?".into(), at: may_8() };
    let mut prefixes = Vec::new();
    for selection in [SelectionMode::Relevance, SelectionMode::Mmr] {
        let cfg = RetrieveConfig {
            budget_tokens: 180,
            fixed_summary_tokens: Some(110),
            selection,
            ..RetrieveConfig::default()
        };
        let r = Retriever::new(&store, &HashEmbedder, cfg).retrieve(&query).await.unwrap();
        assert!(r.context.contains("SUMMARY_SUPPORT"), "summary must be reserved before facts: {}", r.context);
        assert!(r.context_usage.tokens <= 180);
        assert_eq!(r.sessions.len(), 1);
        let trace = r.fixed_summary.as_ref().unwrap();
        assert_eq!(trace.budget_tokens, 110);
        assert!(trace.tokens <= 110);
        assert_eq!(trace.session_ids, ["D1"]);
        prefixes.push(r.context.split("### Facts").next().unwrap().split("### Excerpts").next().unwrap().to_owned());
    }
    assert_eq!(prefixes[0], prefixes[1], "fixed summary text/order must not depend on fact selection");
    for invalid_budget in [0, 181] {
        let invalid = RetrieveConfig {
            budget_tokens: 180,
            fixed_summary_tokens: Some(invalid_budget),
            ..RetrieveConfig::default()
        };
        assert!(matches!(
            Retriever::new(&store, &HashEmbedder, invalid).retrieve(&query).await,
            Err(licorice::Error::InvalidInput(_))
        ));
    }
    store.close().await.unwrap();
}

struct RejectEmbeddings;

#[async_trait::async_trait]
impl Embedder for RejectEmbeddings {
    fn id(&self) -> &str {
        "test/hash"
    }
    fn dim(&self) -> usize {
        64
    }
    async fn embed(&self, _: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Err(EmbedError::Embed("replay must not embed".into()))
    }
}

#[async_trait::async_trait]
impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        "test/hash"
    }

    fn dim(&self) -> usize {
        64
    }

    async fn embed(&self, texts: &[String], _: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.01f32; 64];
                for w in t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
                    let h = w.bytes().fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
                    v[(h % 64) as usize] += 1.0;
                }
                v
            })
            .collect())
    }
}

fn turn(id: &str, speaker: &str, text: &str) -> TurnInput {
    TurnInput { ext_id: id.into(), speaker: speaker.into(), role: Role::Participant, text: text.into() }
}

fn may_8() -> i64 {
    Day::from_ymd(2023, 5, 8).unwrap().start_ms() + 13 * 3_600_000
}

#[tokio::test(flavor = "multi_thread")]
async fn windows_entities_facts_and_summaries() {
    let (_dir, path) = space();
    let store = Store::open(&path, StoreConfig::new(64)).await.unwrap();
    let embedder = HashEmbedder;
    let ing = Ingestor::new(&store, &embedder, IngestConfig { window_turns: 4, ..IngestConfig::default() });

    let turns = vec![
        turn("D1:1", "Caroline", "Hey Mel! Good to see you!"),
        turn("D1:2", "Melanie", "Hey Caroline! I'm swamped with the kids & work."),
        turn("D1:3", "Caroline", "I went to a LGBTQ support group yesterday and it was so powerful."),
        turn("D1:4", "Melanie", "Wow, that's cool, Caroline!"),
        turn("D1:5", "Caroline", "I'm keen on counseling as a career."),
        turn("D1:6", "Melanie", "I painted a sunrise last year. [shares a photo: a painting of a sunrise over a lake]"),
    ];
    store.write(|_| Ok(())).await.unwrap();
    ing.append_session(SessionInput { ext_id: "D1".into(), started_at: may_8(), turns }).await.unwrap();

    // Raw dialogue is searchable right after append (SPEC I9).
    let raw = store.lexical(LexicalKind::Chunk, "support group", 3).await.unwrap();
    assert!(!raw.is_empty());

    let requests = ing.extraction_requests("D1").await.unwrap();
    assert_eq!(requests.len(), 2, "6 turns in windows of 4");
    assert_eq!(requests[0].id, "extract:D1:0-3");
    assert!(requests[0].user.contains("[D1:3] Caroline: I went to a LGBTQ support group"));
    assert!(requests[1].user.contains("Earlier turns (context only):\n[D1:3]"), "two context turns");
    assert!(requests[0].user.contains("Session date: 8 May 2023"));

    let output = json!({
        "entities": [
            {"name": "Caroline", "kind": "person", "aliases": ["Caro"]},
            {"name": "LGBTQ support group", "kind": "event", "aliases": []}
        ],
        "facts": [
            {"subject": "Caroline", "predicate": "attended", "object": "LGBTQ support group", "object_is_entity": true,
             "statement": "Caroline attended an LGBTQ support group.", "negated": false, "salience": 8,
             "time": {"expression": "yesterday", "kind": "instant"}, "evidence_turns": ["D1:3"]},
            {"subject": "Caroline", "predicate": "attended", "object": "LGBTQ support group", "object_is_entity": true,
             "statement": "Caroline went to the LGBTQ support group.", "negated": false, "salience": 7,
             "time": {"expression": null, "kind": "unknown"}, "evidence_turns": ["D1:3"]},
            {"subject": "Melanie", "predicate": "is busy with", "object": "kids and work", "object_is_entity": false,
             "statement": "Melanie is swamped with her kids and work.", "negated": false, "salience": 5,
             "time": {"expression": null, "kind": "ongoing"}, "evidence_turns": ["D1:2"]},
            {"subject": "Melanie", "predicate": "said", "object": null, "object_is_entity": false,
             "statement": "This cites a turn outside the window.", "negated": false, "salience": 1,
             "time": {"expression": null, "kind": "unknown"}, "evidence_turns": ["D1:6"]}
        ]
    });
    let report = ing.apply_extraction(&requests[0].id, &output).await.unwrap();
    assert_eq!(report.facts_created, 2, "{report:?}");
    assert_eq!(report.facts_merged, 1, "same subject, predicate and object: {report:?}");
    assert_eq!(report.facts_dropped, 1, "evidence outside the window: {report:?}");
    assert_eq!(report.facts_dated, 1);
    assert_eq!(report.entities_created, 1, "only the support group is new: {report:?}");

    let (entities, aliases, dated): (i64, i64, (i64, String)) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM entities", [], |r| r.get(0))?,
                c.query_row(
                    "SELECT count(*) FROM entity_aliases a JOIN entities e ON e.id = a.entity_id WHERE e.name = 'Caroline'",
                    [],
                    |r| r.get(0),
                )?,
                c.query_row(
                    "SELECT occurred_from, time_precision FROM facts WHERE statement LIKE 'Caroline attended%'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(entities, 3, "two pinned speakers and the support group");
    assert_eq!(aliases, 2, "caroline plus the extracted alias caro");
    assert_eq!(Day::from_ms(dated.0), Day::from_ymd(2023, 5, 7).unwrap(), "yesterday from 8 May");
    assert_eq!(dated.1, "day");

    // Facts are findable lexically and by vector, with evidence and mentions merged.
    let hits = store.lexical(LexicalKind::Fact, "support group", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    let (mentions, evidence): (i64, i64) = store
        .read(move |c| {
            Ok((
                c.query_row("SELECT mentions FROM facts WHERE id = ?1", [hits[0].owner], |r| r.get(0))?,
                c.query_row("SELECT count(*) FROM fact_evidence WHERE fact_id = ?1", [hits[0].owner], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((mentions, evidence), (1, 1), "a same-source paraphrase is not another mention");
    let q = embedder.embed(&["Melanie swamped kids work".into()], EmbedKind::Query).await.unwrap();
    let near = store.knn(VectorKind::Fact, &q[0], 1, Filter::None).await.unwrap();
    assert_eq!(near.len(), 1);

    // Summaries become keys, themes and a session vector.
    let summary = ing.summary_request("D1").await.unwrap();
    assert_eq!(summary.id, "summary:D1");
    ing.apply_summary(
        &summary.id,
        &json!({"keys": ["8 May 2023", "LGBTQ support group", "counseling"],
                "themes": [{"title": "Support group", "summary": "Caroline attended an LGBTQ support group the day before."}]}),
    )
    .await
    .unwrap();
    let (keys, text): (i64, String) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM summary_keys WHERE live = 1", [], |r| r.get(0))?,
                c.query_row("SELECT text FROM summaries", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(keys, 4, "three keys and one theme");
    assert!(text.starts_with("Session D1 (8 May 2023): Support group:"), "{text}");

    // Retrieval: the dated fact, its evidence turn and the session summary reach the context.
    let retriever = Retriever::new(&store, &embedder, RetrieveConfig::default());
    let at = Day::from_ymd(2023, 10, 1).unwrap().start_ms();
    let r = retriever
        .retrieve(&Query { text: "When did Caroline go to the LGBTQ support group?".into(), at })
        .await
        .unwrap();
    assert_eq!(r.intent, Intent::When);
    assert!(r.linked_entities.iter().any(|e| e == "Caroline"), "{:?}", r.linked_entities);
    assert!(r.context.contains("- 7 May 2023: Caroline attended an LGBTQ support group. [D1:3]"), "{}", r.context);
    assert!(r.context.contains("[D1:3] Caroline: I went to a LGBTQ support group yesterday"), "{}", r.context);
    assert!(r.context.contains("### Sessions\n- 8 May 2023 [D1] Support group:"), "{}", r.context);
    assert!(r.cited_turns.contains(&"D1:3".to_owned()));
    let req = licorice::answer::request("answer:1", "When did Caroline go to the LGBTQ support group?", &r);
    assert!(req.user.starts_with("## Memory (as of 1 October 2023)"), "{}", req.user);
    assert!(req.user.contains("\nQuestion: When did Caroline go to the LGBTQ support group?\n"));
    assert_eq!((req.task.as_str(), req.schema.is_none()), ("answer", true));
    store.close().await.unwrap();
}

fn fact(statement: &str, negated: bool, expression: Option<&str>, kind: &str, evidence: &str) -> serde_json::Value {
    json!({"subject":"Ada", "predicate":"visited", "object":"Oslo", "object_is_entity":false,
        "statement":statement, "negated":negated, "salience":7,
        "time":{"expression":expression,"kind":kind},"evidence_turns":[evidence]})
}

async fn identity_session(ing: &Ingestor<'_>, id: &str, text: &str) {
    ing.append_session(SessionInput {
        ext_id: id.into(),
        started_at: may_8(),
        turns: vec![turn(&format!("{id}:1"), "Ada", text)],
    })
    .await
    .unwrap();
}

async fn identity_snapshot(store: &Store) -> Vec<i64> {
    store
        .read(|c| {
            let sqls = [
                "SELECT count(*) FROM facts",
                "SELECT coalesce(sum(mentions),0) FROM facts",
                "SELECT count(*) FROM fact_evidence",
                "SELECT count(*) FROM entities",
                "SELECT count(*) FROM entity_aliases",
                "SELECT count(*) FROM vec_fact",
                "SELECT count(*) FROM vec_entity",
                "SELECT count(*) FROM vec_key",
                "SELECT count(*) FROM vec_summary",
                "SELECT coalesce(sum(version),0) FROM summaries",
            ];
            sqls.iter().map(|sql| c.query_row(sql, [], |r| r.get(0)).map_err(Into::into)).collect()
        })
        .await
        .unwrap()
}

#[tokio::test]
async fn identity_preserves_negation_and_distinct_source_occurrences() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    for (id, negative, time) in [
        ("D1", false, Some("yesterday")),
        ("D2", true, Some("yesterday")),
        ("D3", false, Some("yesterday")),
        ("D4", false, None),
        ("D5", false, None),
    ] {
        identity_session(&ing, id, "Ada visited Oslo.").await;
        let output =
            json!({"entities":[],"facts":[fact("Ada visited Oslo.",negative,time,"instant",&format!("{id}:1"))]});
        ing.apply_extraction(&format!("extract:{id}:0-0"), &output).await.unwrap();
    }
    let count: i64 = store.read(|c| Ok(c.query_row("SELECT count(*) FROM facts", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(count, 5, "negation, same-day distinct evidence, and undated events must survive");
    let r = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
        .retrieve(&Query { text: "How many times did Ada visit Oslo?".into(), at: may_8() })
        .await
        .unwrap();
    assert_eq!(r.facts.len(), 5, "statement-text dedup must not erase separate occurrences");
    store.close().await.unwrap();
}

#[tokio::test]
async fn identical_extraction_and_summary_replay_has_no_storage_effects() {
    let (_dir, path) = space();
    let store = Store::open(&path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo yesterday.").await;
    let output = json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,Some("yesterday"),"instant","D1:1")]});
    let summary = json!({"keys":["Oslo"],"themes":[{"title":"Visit","summary":"Ada visited Oslo."}]});
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    ing.apply_summary("summary:D1", &summary).await.unwrap();
    let before = identity_snapshot(&store).await;
    store.close().await.unwrap();
    let store = Store::open(&path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &RejectEmbeddings, IngestConfig::default());
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    ing.apply_summary("summary:D1", &summary).await.unwrap();
    assert_eq!(identity_snapshot(&store).await, before, "a retry after lost acknowledgment must be a no-op");
    store.close().await.unwrap();
}

#[tokio::test]
async fn changed_output_under_applied_id_is_a_conflict() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo.").await;
    let mut output = json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,None,"unknown","D1:1")]});
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    let before = identity_snapshot(&store).await;
    output["facts"][0]["negated"] = json!(true);
    assert!(ing.apply_extraction("extract:D1:0-0", &output).await.is_err());
    assert_eq!(identity_snapshot(&store).await, before);
    store.close().await.unwrap();
}

#[tokio::test]
async fn identical_session_replay_succeeds_and_changed_source_is_refused() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    let mut session =
        SessionInput { ext_id: "D1".into(), started_at: may_8(), turns: vec![turn("D1:1", "Ada", "I live in Oslo.")] };
    let first = ing.append_session(session.clone()).await.unwrap();
    assert_eq!(ing.append_session(session.clone()).await.unwrap(), first);
    session.turns[0].text = "I live in Rome.".into();
    assert!(ing.append_session(session).await.is_err());
    store.close().await.unwrap();
}

#[tokio::test]
async fn ongoing_mentions_merge_but_incompatible_dates_do_not() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    for (id, time, kind) in [("D1", None, "ongoing"), ("D2", None, "ongoing"), ("D3", Some("yesterday"), "instant")] {
        identity_session(&ing, id, "Ada visits Oslo.").await;
        let mut first = fact("Ada visits Oslo.", false, time, kind, &format!("{id}:1"));
        if id == "D3" {
            let second = fact("Ada visits Oslo.", false, Some("today"), kind, "D3:1");
            ing.apply_extraction("extract:D3:0-0", &json!({"entities":[],"facts":[first,second]})).await.unwrap();
        } else {
            first["predicate"] = json!("visits");
            ing.apply_extraction(&format!("extract:{id}:0-0"), &json!({"entities":[],"facts":[first]})).await.unwrap();
        }
    }
    let (facts, mentions): (i64, i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT count(*) FROM facts", [], |r| r.get(0))?,
                c.query_row("SELECT mentions FROM facts WHERE time_kind='ongoing'", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!((facts, mentions), (3, 2));
    store.close().await.unwrap();
}

struct BarrierEmbedder(tokio::sync::Barrier);

#[async_trait::async_trait]
impl Embedder for BarrierEmbedder {
    fn id(&self) -> &str {
        "barrier/hash"
    }
    fn dim(&self) -> usize {
        64
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0.wait().await;
        HashEmbedder.embed(texts, kind).await
    }
}

#[tokio::test]
async fn concurrent_applications_commit_once_and_resolve_racing_entities() {
    for same_request in [true, false] {
        let (_dir, path) = space();
        let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
        identity_session(&ing, "D1", "Ada visited Oslo.").await;
        identity_session(&ing, "D2", "Ada visited Oslo.").await;
        let embedder = BarrierEmbedder(tokio::sync::Barrier::new(2));
        let ing = Ingestor::new(&store, &embedder, IngestConfig::default());
        let mut first = fact("Ada visited Oslo.", false, None, "unknown", "D1:1");
        first["object_is_entity"] = json!(true);
        let mut second = first.clone();
        if !same_request {
            second["evidence_turns"] = json!(["D2:1"]);
        }
        let a = json!({"entities":[],"facts":[first]});
        let b = json!({"entities":[],"facts":[second]});
        let (left, right) = tokio::join!(
            ing.apply_extraction("extract:D1:0-0", &a),
            ing.apply_extraction(if same_request { "extract:D1:0-0" } else { "extract:D2:0-0" }, &b)
        );
        let (left, right) = (left.unwrap(), right.unwrap());
        assert_eq!(left.replayed as usize + right.replayed as usize, usize::from(same_request));
        let (entities, facts, receipts): (i64, i64, i64) = store
            .read(|c| {
                Ok((
                    c.query_row("SELECT count(*) FROM entities WHERE name='Oslo'", [], |r| r.get(0))?,
                    c.query_row("SELECT count(*) FROM facts", [], |r| r.get(0))?,
                    c.query_row("SELECT count(*) FROM applied_outputs", [], |r| r.get(0))?,
                ))
            })
            .await
            .unwrap();
        assert_eq!(
            (entities, facts, receipts),
            (1, if same_request { 1 } else { 2 }, if same_request { 1 } else { 2 })
        );
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn receipt_failure_rolls_back_facts_indexes_and_new_entities() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo.").await;
    let before = identity_snapshot(&store).await;
    store.write(|w| {w.conn().execute_batch("CREATE TRIGGER receipt_fault BEFORE INSERT ON applied_outputs BEGIN SELECT RAISE(ABORT,'injected receipt failure'); END;")?;Ok(())}).await.unwrap();
    let mut f = fact("Ada visited Oslo.", false, None, "instant", "D1:1");
    f["object_is_entity"] = json!(true);
    let output = json!({"entities":[],"facts":[f]});
    assert!(ing.apply_extraction("extract:D1:0-0", &output).await.is_err());
    assert_eq!(identity_snapshot(&store).await, before);
    assert!(store.lexical(LexicalKind::Fact, "Oslo", 10).await.unwrap().is_empty());
    store
        .write(|w| {
            w.conn().execute_batch("DROP TRIGGER receipt_fault")?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(ing.apply_extraction("extract:D1:0-0", &output).await.unwrap().facts_created, 1);
    let count: i64 =
        store.read(|c| Ok(c.query_row("SELECT count(*) FROM applied_outputs", [], |r| r.get(0))?)).await.unwrap();
    assert_eq!(count, 1);
    store.close().await.unwrap();
}

#[tokio::test]
async fn v1_migration_preserves_rows_and_refuses_unverifiable_legacy_replay() {
    let (_dir, path) = space();
    {
        let c = licorice::rusqlite::Connection::open(&path).unwrap();
        c.execute_batch(include_str!("../migrations/0001_init.sql")).unwrap();
        c.execute_batch("PRAGMA user_version=1; INSERT INTO meta VALUES ('embed_dim','64');
            INSERT INTO sessions(id,ext_id,started_at) VALUES(1,'D1',0);
            INSERT INTO turns(id,session_id,seq,speaker,role,text,ext_id) VALUES(1,1,0,'Ada','participant','Visited Oslo','D1:1');
            INSERT INTO entities(id,name,kind) VALUES(1,'Ada','person');
            INSERT INTO facts(id,subject_id,predicate,object_value,statement,recorded_first,recorded_last,dedup_key)
              VALUES(1,1,'visited','Oslo','Visited Oslo',0,0,'legacy');
            INSERT INTO fact_evidence(fact_id,turn_id) VALUES(1,1);").unwrap();
    }
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let version: i64 = store
        .read(|c| Ok(c.query_row("SELECT identity_version FROM facts WHERE id=1", [], |r| r.get(0))?))
        .await
        .unwrap();
    assert_eq!(version, 1);
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    let result = ing
        .apply_extraction(
            "extract:D1:0-0",
            &json!({"entities":[],"facts":[fact("Visited Oslo",false,None,"unknown","D1:1")]}),
        )
        .await;
    assert!(matches!(result, Err(licorice::Error::LegacyApplication { .. })));
    store.close().await.unwrap();
}

#[tokio::test]
async fn changed_source_and_changed_summary_conflict_without_partial_application() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo.").await;
    let output = json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,None,"unknown","D1:1")]});
    let summary = json!({"keys":["Oslo"],"themes":[]});
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    ing.apply_summary("summary:D1", &summary).await.unwrap();
    let before = identity_snapshot(&store).await;
    assert!(matches!(
        ing.apply_summary("summary:D1", &json!({"keys":["Rome"],"themes":[]})).await,
        Err(licorice::Error::ApplicationConflict { .. })
    ));
    store
        .write(|w| {
            w.conn().execute("UPDATE turns SET text='Changed source'", [])?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(matches!(
        ing.apply_extraction("extract:D1:0-0", &output).await,
        Err(licorice::Error::ApplicationConflict { .. })
    ));
    assert_eq!(identity_snapshot(&store).await, before);
    store.close().await.unwrap();
}

#[tokio::test]
async fn default_context_budget_skips_an_oversized_turn_but_keeps_later_evidence() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    ing.append_session(SessionInput {
        ext_id: "D1".into(),
        started_at: may_8(),
        turns: vec![
            turn("D1:1", "Ada", &"banana ".repeat(10_000)),
            turn("D1:2", "Ada", "My favorite fruit is banana."),
        ],
    })
    .await
    .unwrap();
    let cfg = RetrieveConfig { final_facts: 0, summaries: 0, chunks: 2, ..RetrieveConfig::default() };
    let r = Retriever::new(&store, &HashEmbedder, cfg)
        .retrieve(&Query { text: "What fruit does Ada like?".into(), at: may_8() })
        .await
        .unwrap();
    use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
    assert_eq!(r.context_usage.tokens, Cl100kCounter.count(&r.context).unwrap());
    assert!(r.context_usage.tokens <= 2000, "the default context must be bounded");
    assert_eq!(r.context_usage.budget_tokens, 2000);
    assert_eq!(r.budget_omissions.turn_ids, vec!["D1:1"]);
    assert!(!r.cited_turns.contains(&"D1:1".into()), "a dropped turn cannot count as retrieved evidence");
    assert!(r.cited_turns.contains(&"D1:2".into()), "skip a long turn without losing a later short answer");
    store.close().await.unwrap();
}

#[tokio::test]
async fn unresolved_temporal_refinement_is_retained_for_later_identity_checks() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo during two different festivals.").await;
    let facts = [None, Some("during the sapphire festival"), Some("during the amber festival")]
        .map(|time| fact("Ada visited Oslo.", false, time, "instant", "D1:1"));
    let report = ing.apply_extraction("extract:D1:0-0", &json!({"entities":[],"facts":facts})).await.unwrap();
    assert_eq!(report.facts_created, 2, "a refined unresolved qualifier must distinguish a later event");
    assert_eq!(report.facts_merged, 1);
    let qualifiers: Vec<String> = store
        .read(|c| {
            let mut stmt = c.prepare("SELECT time_text FROM facts ORDER BY id")?;
            Ok(stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?)
        })
        .await
        .unwrap();
    assert_eq!(qualifiers, vec!["during the sapphire festival", "during the amber festival"]);
    store.close().await.unwrap();
}

#[tokio::test]
async fn replay_hashes_include_exact_source_metadata() {
    for change in
        ["UPDATE sessions SET started_at=started_at+1", "UPDATE turns SET role='assistant'", "UPDATE turns SET ts=ts+1"]
    {
        let (_dir, path) = space();
        let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
        let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
        identity_session(&ing, "D1", "Ada visited Oslo.").await;
        let output = json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,None,"instant","D1:1")]});
        let summary = json!({"keys":["Ada"],"themes":[]});
        ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
        ing.apply_summary("summary:D1", &summary).await.unwrap();
        store
            .write(move |w| {
                w.conn().execute(change, [])?;
                Ok(())
            })
            .await
            .unwrap();
        assert!(
            matches!(
                ing.apply_extraction("extract:D1:0-0", &output).await,
                Err(licorice::Error::ApplicationConflict { .. })
            ),
            "{change}"
        );
        assert!(
            matches!(ing.apply_summary("summary:D1", &summary).await, Err(licorice::Error::ApplicationConflict { .. })),
            "{change}"
        );
        store.close().await.unwrap();
    }
}

#[tokio::test]
async fn time_window_count_questions_keep_occurrences_when_mmr_is_requested() {
    use licorice::retrieve::SelectionMode;
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    for session in ["D1", "D2"] {
        identity_session(&ing, session, "Ada visited Oslo.").await;
        ing.apply_extraction(&format!("extract:{session}:0-0"),&json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,Some("yesterday"),"instant",&format!("{session}:1"))]})).await.unwrap();
    }
    let cfg = RetrieveConfig { selection: SelectionMode::Mmr, ..RetrieveConfig::default() };
    let r = Retriever::new(&store, &HashEmbedder, cfg)
        .retrieve(&Query { text: "How many times did Ada visit Oslo in May 2023?".into(), at: may_8() })
        .await
        .unwrap();
    assert!(matches!(r.intent, Intent::Window(_)));
    assert_eq!(r.selection.requested, SelectionMode::Mmr);
    assert_eq!(r.selection.effective, SelectionMode::Relevance, "a time window must not mask count/list shape");
    assert_eq!(r.facts.len(), 2);
    store.close().await.unwrap();
}

#[tokio::test]
async fn packed_citations_preserve_source_ids_and_use_the_supplied_counter() {
    use licorice_providers::tokens::{CountError, TokenCounter};
    struct ScalarCounter;
    impl TokenCounter for ScalarCounter {
        fn id(&self) -> &str {
            "test/unicode-scalars"
        }
        fn count(&self, text: &str) -> Result<usize, CountError> {
            Ok(text.chars().count())
        }
    }
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    ing.append_session(SessionInput {
        ext_id: "D1".into(),
        started_at: may_8(),
        turns: vec![turn("D1|1", "Ada", "Ada visited Oslo. café 🍌")],
    })
    .await
    .unwrap();
    ing.apply_extraction(
        "extract:D1:0-0",
        &json!({"entities":[],"facts":[fact("Ada visited Oslo. café 🍌",false,None,"instant","D1|1")]}),
    )
    .await
    .unwrap();
    for budget_tokens in [120, 0] {
        let cfg = RetrieveConfig { budget_tokens, chunks: 0, summaries: 0, ..RetrieveConfig::default() };
        let r = Retriever::new(&store, &HashEmbedder, cfg)
            .with_token_counter(&ScalarCounter)
            .retrieve(&Query { text: "Where did Ada visit?".into(), at: may_8() })
            .await
            .unwrap();
        assert_eq!(r.context_usage.tokenizer, "test/unicode-scalars");
        assert_eq!(r.context_usage.tokens, r.context.chars().count());
        assert!(r.context_usage.tokens <= budget_tokens);
        if budget_tokens > 0 {
            assert_eq!(r.cited_turns, vec!["D1|1"], "IDs must not be split on an arbitrary delimiter");
            assert!(r.context.contains("[D1|1]"));
        } else {
            assert!(r.context.is_empty());
            assert!(r.facts.is_empty());
            assert!(r.cited_turns.is_empty());
        }
    }
    store.close().await.unwrap();
}

#[tokio::test]
async fn vague_event_times_reach_the_context_at_their_own_precision() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    identity_session(&ing, "D1", "Ada visited Oslo last summer.").await;
    ing.apply_extraction(
        "extract:D1:0-0",
        &json!({"entities":[],"facts":[fact("Ada visited Oslo.",false,Some("last summer"),"instant","D1:1")]}),
    )
    .await
    .unwrap();
    let r = Retriever::new(&store, &HashEmbedder, RetrieveConfig::default())
        .retrieve(&Query { text: "When did Ada visit Oslo?".into(), at: may_8() })
        .await
        .unwrap();
    assert!(r.context.contains("- summer 2022: Ada visited Oslo. [D1:1]"), "{}", r.context);
    assert!(!r.context.contains("1 June 2022"), "a season must not read as its first day: {}", r.context);
    let occurred = r.facts[0].occurred.expect("dated");
    assert_eq!((occurred.from, occurred.to), (Day::from_ymd(2022, 6, 1).unwrap(), Day::from_ymd(2022, 8, 31).unwrap()));
    store.close().await.unwrap();
}

fn ongoing(subject: &str, predicate: &str, object: &str, statement: &str, evidence: &str) -> serde_json::Value {
    json!({"subject":subject, "predicate":predicate, "object":object, "object_is_entity":true,
        "statement":statement, "negated":false, "salience":5,
        "time":{"expression":null,"kind":"ongoing"},"evidence_turns":[evidence]})
}

async fn one_turn_session(ing: &Ingestor<'_>, id: &str, speaker: &str, text: &str) {
    ing.append_session(SessionInput {
        ext_id: id.into(),
        started_at: may_8(),
        turns: vec![turn(&format!("{id}:1"), speaker, text)],
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn names_differing_only_in_symbols_stay_separate_entities() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    one_turn_session(&ing, "D1", "Ada", "I write C++ at work and C# at home.").await;
    let output = json!({
        "entities": [{"name":"C++","kind":"concept","aliases":[]}, {"name":"C#","kind":"concept","aliases":[]}],
        "facts": [ongoing("Ada","uses","C++","Ada uses C++.","D1:1"), ongoing("Ada","uses","C#","Ada uses C#.","D1:1")]
    });
    let report = ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    assert_eq!((report.facts_created, report.facts_merged), (2, 0), "{report:?}");
    let objects: i64 = store
        .read(|c| Ok(c.query_row("SELECT count(DISTINCT object_id) FROM facts", [], |r| r.get(0))?))
        .await
        .unwrap();
    assert_eq!(objects, 2, "C++ and C# are different languages");
    store.close().await.unwrap();
}

#[tokio::test]
async fn a_new_entity_cannot_claim_an_existing_speakers_name_as_an_alias() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    one_turn_session(&ing, "D1", "Melanie", "I painted a lake.").await;
    let output = json!({
        "entities": [{"name":"Mel","kind":"person","aliases":["Melanie"]}],
        "facts": [{"subject":"Melanie", "predicate":"painted", "object":"a lake", "object_is_entity":false,
            "statement":"Melanie painted a lake.", "negated":false, "salience":5,
            "time":{"expression":null,"kind":"unknown"},"evidence_turns":["D1:1"]}]
    });
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    let (subject, owners): ((String, i64), i64) = store
        .read(|c| {
            Ok((
                c.query_row("SELECT e.name, e.pinned FROM facts f JOIN entities e ON e.id = f.subject_id", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })?,
                c.query_row("SELECT count(*) FROM entity_aliases WHERE alias = 'melanie'", [], |r| r.get(0))?,
            ))
        })
        .await
        .unwrap();
    assert_eq!(subject, ("Melanie".to_owned(), 1), "the fact belongs to the pinned speaker");
    assert_eq!(owners, 1, "\"melanie\" still names one entity");
    store.close().await.unwrap();
}

#[tokio::test]
async fn a_speaker_is_pinned_even_when_a_place_shares_their_name() {
    let (_dir, path) = space();
    let store = Store::open(path, StoreConfig::new(64)).await.unwrap();
    let ing = Ingestor::new(&store, &HashEmbedder, IngestConfig::default());
    one_turn_session(&ing, "D1", "Ada", "I flew to Jordan.").await;
    let output = json!({
        "entities": [{"name":"Jordan","kind":"location","aliases":[]}],
        "facts": [ongoing("Ada","has visited","Jordan","Ada has visited Jordan.","D1:1")]
    });
    ing.apply_extraction("extract:D1:0-0", &output).await.unwrap();
    one_turn_session(&ing, "D2", "Jordan", "Hi Ada!").await;
    let kinds: Vec<(String, i64)> = store
        .read(|c| {
            let mut stmt = c.prepare("SELECT kind, pinned FROM entities WHERE name = 'Jordan' ORDER BY id")?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<Result<_, _>>()?;
            Ok(rows)
        })
        .await
        .unwrap();
    assert_eq!(kinds, [("location".to_owned(), 0), ("person".to_owned(), 1)]);
    store.close().await.unwrap();
}
