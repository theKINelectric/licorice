//! Contract tests for the subprocess judge against `fixtures/fake-coffeecat`, a shell
//! stand-in for `coffeecat ask --jsonl`. The ignored tests run the installed native
//! Coffee Cat and its model.

use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use licorice_providers::judge::{Judge, JudgeError, Judgment, Point, ProcessJudge, ProcessJudgeConfig, Question};
use licorice_providers::tokens::{Cl100kCounter, TokenCounter};
use serde_json::{Value, json};

const SECOND: Duration = Duration::from_secs(1);

fn fake() -> ProcessJudgeConfig {
    ProcessJudgeConfig {
        program: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fake-coffeecat").into(),
        ..ProcessJudgeConfig::coffee_cat()
    }
}

fn question() -> IndexMap<String, Question> {
    IndexMap::from([("q".to_owned(), Question::noul("Is it so?"))])
}

async fn ask(judge: &ProcessJudge, state: Value, deadline: Duration) -> Result<Judgment, JudgeError> {
    judge.ask(Point::J1, &state, &question(), deadline).await
}

fn noul(judgment: &Judgment) -> f64 {
    judgment.answers["q"].as_noul().unwrap()
}

/// A fresh file for the fake to append raw requests to.
fn record_path(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("{name}-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path.to_str().unwrap().to_owned()
}

fn recorded(path: &str) -> Vec<Value> {
    match std::fs::read_to_string(path) {
        Ok(text) => text.lines().map(|l| serde_json::from_str(l).unwrap()).collect(),
        Err(_) => Vec::new(),
    }
}

#[tokio::test]
async fn one_process_serves_successive_calls() {
    let judge = ProcessJudge::new(fake()).unwrap();
    assert!(judge.is_loopback());
    assert!(judge.id().ends_with("fake-coffeecat ask --jsonl --context 4096"), "{}", judge.id());
    let first = ask(&judge, json!({"p": 0.25}), SECOND).await.unwrap();
    assert_eq!(noul(&first), 0.25);
    // The reported model is the judgment's model; the fake reports its pid and arguments.
    assert!(first.model.starts_with("fake-"), "{}", first.model);
    assert!(first.model.ends_with(" ask --jsonl --context 4096"), "{}", first.model);
    assert_eq!(first.usage.input_tokens, Some(7));
    let second = ask(&judge, json!({"p": 0.75}), SECOND).await.unwrap();
    assert_eq!(noul(&second), 0.75);
    assert_eq!(second.model, first.model, "the process is kept between calls");
}

#[tokio::test]
async fn the_process_starts_on_first_use() {
    let mut cfg = fake();
    cfg.program = "/nonexistent/coffeecat".into();
    let judge = ProcessJudge::new(cfg).unwrap();
    let err = ask(&judge, json!({}), SECOND).await.unwrap_err();
    assert!(matches!(err, JudgeError::Startup(_)), "{err}");
    assert!(matches!(judge.warm_up().await, Err(JudgeError::Startup(_))));
}

#[tokio::test]
async fn context_and_extra_arguments_are_passed() {
    let mut cfg = fake();
    cfg.context = 8192;
    cfg.extra_args = vec!["--gpu-layers".into(), "0".into()];
    let judge = ProcessJudge::new(cfg).unwrap();
    judge.warm_up().await.unwrap();
    let judgment = ask(&judge, json!({}), SECOND).await.unwrap();
    assert!(judgment.model.ends_with(" ask --jsonl --context 8192 --gpu-layers 0"), "{}", judgment.model);
}

#[tokio::test]
async fn sends_only_fields_the_cli_accepts() {
    let path = record_path("fields");
    let judge = ProcessJudge::new(fake()).unwrap();
    let questions = IndexMap::from([(
        "pick".to_owned(),
        Question::choice("Which one?", [("a", None), ("b", Some("the second".to_owned()))]),
    )]);
    let err = judge.ask(Point::J2, &json!({"path": path}), &questions, SECOND).await.unwrap_err();
    // The fake answers only `q`, so decoding `pick` fails; the request is what matters.
    assert!(matches!(err, JudgeError::Decode(_)), "{err}");
    let sent = recorded(&path);
    assert_eq!(sent.len(), 1);
    let keys: Vec<&str> = sent[0].as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys.len(), 4, "{keys:?}");
    for key in ["id", "state", "questions", "deadline_ms"] {
        assert!(keys.contains(&key), "{keys:?}");
    }
    let deadline_ms = sent[0]["deadline_ms"].as_u64().unwrap();
    assert!((100..=1000).contains(&deadline_ms), "{deadline_ms}");
    assert_eq!(sent[0]["questions"]["pick"]["criteria"], json!({"a": "a", "b": "the second"}));
}

#[tokio::test]
async fn concurrent_callers_each_get_their_own_answer() {
    let judge = Arc::new(ProcessJudge::new(fake()).unwrap());
    let mut tasks = tokio::task::JoinSet::new();
    for i in 1..=8 {
        let judge = Arc::clone(&judge);
        let p = f64::from(i) / 10.0;
        tasks.spawn(async move { (p, ask(&judge, json!({"p": p}), 5 * SECOND).await.unwrap()) });
    }
    let mut models = Vec::new();
    while let Some(done) = tasks.join_next().await {
        let (p, judgment) = done.unwrap();
        assert_eq!(noul(&judgment), p);
        models.push(judgment.model);
    }
    models.dedup();
    assert_eq!(models.len(), 1, "one process: {models:?}");
}

#[tokio::test]
async fn one_request_is_in_flight_at_a_time() {
    let path = record_path("in-flight");
    let judge = Arc::new(ProcessJudge::new(fake()).unwrap());
    let slow = {
        let judge = Arc::clone(&judge);
        let path = path.clone();
        tokio::spawn(async move { ask(&judge, json!({"mode": "sleep", "p": 0.9, "path": path}), 3 * SECOND).await })
    };
    tokio::time::sleep(Duration::from_millis(100)).await;
    // This call gives up while the slow one holds the process, so it is never sent.
    let waiting = ask(&judge, json!({"p": 0.1, "path": path}), Duration::from_millis(200)).await.unwrap_err();
    assert!(matches!(waiting, JudgeError::Timeout(_)), "{waiting}");
    assert_eq!(noul(&slow.await.unwrap().unwrap()), 0.9);
    assert_eq!(recorded(&path).len(), 1);
}

#[tokio::test]
async fn a_late_answer_is_discarded_rather_than_handed_to_the_next_caller() {
    let judge = ProcessJudge::new(fake()).unwrap();
    let before = ask(&judge, json!({"p": 0.2}), SECOND).await.unwrap();
    let started = Instant::now();
    let err = ask(&judge, json!({"mode": "sleep", "p": 0.9}), Duration::from_millis(200)).await.unwrap_err();
    assert!(matches!(err, JudgeError::Timeout(_)), "{err}");
    assert!(started.elapsed() < Duration::from_millis(400));
    // The abandoned request is still answered (0.9) while this one waits behind it.
    let after = ask(&judge, json!({"p": 0.3}), 3 * SECOND).await.unwrap();
    assert_eq!(noul(&after), 0.3);
    assert_eq!(after.model, before.model, "a timeout keeps the process");
    let next = ask(&judge, json!({"p": 0.4}), SECOND).await.unwrap();
    assert_eq!(noul(&next), 0.4);
}

#[tokio::test]
async fn a_crash_fails_that_call_and_the_next_call_respawns() {
    let judge = ProcessJudge::new(fake()).unwrap();
    let before = ask(&judge, json!({"p": 0.2}), SECOND).await.unwrap();
    let err = ask(&judge, json!({"mode": "crash"}), SECOND).await.unwrap_err();
    match &err {
        JudgeError::Transport(message) => assert!(message.contains("exit status: 3"), "{message}"),
        other => panic!("expected a transport error, got {other}"),
    }
    let after = ask(&judge, json!({"p": 0.4}), SECOND).await.unwrap();
    assert_eq!(noul(&after), 0.4);
    assert_ne!(after.model, before.model, "a new process");
}

#[tokio::test]
async fn coffee_cat_error_codes_map_to_judge_errors() {
    let judge = ProcessJudge::new(fake()).unwrap();
    let error = |code: &str| json!({"mode": "error", "code": code});
    let before = ask(&judge, json!({}), SECOND).await.unwrap();
    // Each variant has its own message prefix.
    for (code, expected) in [
        ("deadline_exceeded", "no answer within"),
        ("bounded_limit_exceeded", "invalid request: bounded_limit_exceeded"),
        ("invalid_input", "invalid request: invalid_input"),
        ("inference_error", "judge failed: inference_error"),
    ] {
        let err = ask(&judge, error(code), SECOND).await.unwrap_err();
        assert!(err.to_string().starts_with(expected), "{code}: {err}");
    }
    // A record Coffee Cat cannot attribute (`id: null`) is matched by its line.
    let err = ask(&judge, json!({"mode": "anonymous"}), SECOND).await.unwrap_err();
    assert!(matches!(err, JudgeError::Invalid(_)), "{err}");
    assert_eq!(ask(&judge, json!({}), SECOND).await.unwrap().model, before.model, "refusals keep the process");

    let err = ask(&judge, error("startup_error"), SECOND).await.unwrap_err();
    assert!(matches!(err, JudgeError::Startup(_)), "{err}");
    assert_ne!(ask(&judge, json!({}), SECOND).await.unwrap().model, before.model, "a startup error respawns");
}

#[tokio::test]
async fn a_deadline_below_coffee_cats_minimum_is_refused_without_sending() {
    let path = record_path("short");
    let judge = ProcessJudge::new(fake()).unwrap();
    let err = ask(&judge, json!({"path": path}), Duration::from_millis(50)).await.unwrap_err();
    assert!(matches!(err, JudgeError::Timeout(_)), "{err}");
    assert!(recorded(&path).is_empty());
}

#[test]
fn invalid_configuration_is_refused() {
    for context in [0, 1000] {
        let cfg = ProcessJudgeConfig { context, ..ProcessJudgeConfig::coffee_cat() };
        assert!(matches!(ProcessJudge::new(cfg), Err(JudgeError::Invalid(_))), "{context}");
    }
    let cfg = ProcessJudgeConfig { program: "".into(), ..ProcessJudgeConfig::coffee_cat() };
    assert!(matches!(ProcessJudge::new(cfg), Err(JudgeError::Invalid(_))));
}

/// The three J10 cases from the 2026-10-19 smoke test, as Licorice sent them.
fn smoke_cases() -> Vec<(String, Value, IndexMap<String, Question>)> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/eval/judge-smoke-2026-10-19.json");
    let smoke: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    smoke["attempts"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["engine"] == "coffee-cat" && a["attempt"] == "diagnostic_5s")
        .map(|a| {
            let request = &a["logical_request"];
            let questions = serde_json::from_value(request["questions"].clone()).unwrap();
            (a["case"].as_str().unwrap().to_owned(), request["state"].clone(), questions)
        })
        .collect()
}

#[tokio::test]
#[ignore = "needs the installed coffeecat and its pinned model"]
async fn native_coffee_cat_matches_the_measured_j10_scores() {
    // Native Coffee Cat 0.4 measurements (sufficient, contradicted).
    let expected = [("supported", 0.9993, 0.7805), ("missing", 0.8117, 0.0903), ("contradicted", 0.4633, 0.1543)];
    let judge = ProcessJudge::new(ProcessJudgeConfig::coffee_cat()).unwrap();
    let started = Instant::now();
    judge.warm_up().await.unwrap();
    eprintln!("warm-up (spawn + model load + probe): {:?}", started.elapsed());
    let cases = smoke_cases();
    assert_eq!(cases.len(), 3);
    for (case, state, questions) in cases {
        let judgment = judge.ask(Point::J10, &state, &questions, 5 * SECOND).await.unwrap();
        let (_, sufficient, contradicted) = expected.iter().find(|(c, ..)| *c == case).unwrap();
        let got =
            (judgment.answers["sufficient"].as_noul().unwrap(), judgment.answers["contradicted"].as_noul().unwrap());
        eprintln!("{case}: {got:?} in {:?} ({:?} tokens)", judgment.latency, judgment.usage.input_tokens);
        assert_eq!(judgment.model, "bonsai-1.7b-gguf");
        assert!((got.0 - sufficient).abs() <= 0.005, "{case} sufficient {got:?}");
        assert!((got.1 - contradicted).abs() <= 0.005, "{case} contradicted {got:?}");
    }
}

/// About `tokens` cl100k tokens of memory rendered the way Licorice packs facts.
fn full_memory(tokens: usize) -> String {
    let (people, places) =
        (["Ada", "Bo", "Caro", "Dev", "Esme"], ["Oslo", "Lisbon", "Kyoto", "Lima", "Accra", "Hanoi"]);
    let doings =
        ["visited", "planned a trip to", "recommended a café in", "took a pottery class in", "met a friend in"];
    let mut memory = String::from("## Memory (as of 8 May 2023)\n### Facts (chronological)\n");
    for i in 0.. {
        let line = format!(
            "- said {} {} 2023: {} {} {} with {} and described the {} in detail. [S{}:{}]\n",
            1 + i % 28,
            ["March", "April", "May"][i % 3],
            people[i % 5],
            doings[(i / 5) % 5],
            places[(i / 3) % 6],
            people[(i + 2) % 5],
            ["weather", "food", "museums", "trains", "markets"][(i / 7) % 5],
            1 + i / 12,
            1 + i % 12
        );
        if Cl100kCounter.count(&(memory.clone() + &line)).unwrap() > tokens {
            break;
        }
        memory.push_str(&line);
    }
    memory
}

#[tokio::test]
#[ignore = "needs the installed coffeecat and its pinned model"]
async fn native_coffee_cat_latency() {
    let judge = ProcessJudge::new(ProcessJudgeConfig::coffee_cat()).unwrap();
    let started = Instant::now();
    judge.warm_up().await.unwrap();
    eprintln!("warm-up: {:?}", started.elapsed());
    let (_, _, questions) = smoke_cases().remove(0);
    let question = "Which cities did Ada visit in April?";
    // This synthetic prose costs Bonsai about 1.1 tokens per cl100k token. Real
    // 2,000-token LoCoMo memory (dates, names, citations) reaches about 3,540 per
    // question, which 3,100 cl100k tokens of it matches.
    let states = [
        ("small", json!({"question": "Where does Ada live?", "memory": "Ada lives in Oslo. [S1:1]"})),
        ("2,000 cl100k", json!({"question": question, "memory": full_memory(2000)})),
        ("3,100 cl100k", json!({"question": question, "memory": full_memory(3100)})),
    ];
    for (label, state) in states {
        let mut latencies = Vec::new();
        let mut tokens = None;
        for _ in 0..5 {
            let judgment = judge.ask(Point::J10, &state, &questions, 10 * SECOND).await.unwrap();
            latencies.push(judgment.latency.as_secs_f64() * 1000.0);
            tokens = judgment.usage.input_tokens;
        }
        latencies.sort_by(f64::total_cmp);
        // Two questions, each reading the whole state.
        eprintln!("{label}: {tokens:?} Bonsai tokens read; ms {latencies:.1?}");
    }
}
