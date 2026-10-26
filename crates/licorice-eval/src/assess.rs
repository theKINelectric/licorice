//! Judge the exact saved answer contexts. Gold labels stay in the evaluator.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use licorice::assess::{
    AssessConfig, AssessmentStatus, EvidenceAssessor, allow_remote_judgments, remote_judgments_allowed,
};
use licorice::{Store, StoreConfig, answer, rusqlite};
use licorice_providers::judge::{HttpJudge, HttpJudgeConfig, Judge, ProcessJudge, ProcessJudgeConfig};
use serde_json::json;

use crate::batch::Res;
use crate::{Args, SPACE, answer_id, batch, load_run, locomo};

fn file_credential(path: &Path, name: &str) -> Res<Option<String>> {
    // Iterator parsing does not mutate process environment (important once Tokio
    // threads exist). Never include a parser error's raw line in an error message.
    let entries = dotenvy::from_path_iter(path)
        .map_err(|_| format!("cannot read env file {} (values omitted)", path.display()))?;
    let mut found = None;
    for entry in entries {
        let (key, value) = entry.map_err(|_| format!("cannot parse env file {} (values omitted)", path.display()))?;
        if key == name {
            found = Some(value);
        }
    }
    Ok(found)
}

fn credential(a: &Args, name: &str) -> Res<Option<String>> {
    if let Ok(value) = std::env::var(name) {
        return Ok(Some(value));
    }
    let path = Path::new(a.get("env-file").unwrap_or(".env"));
    if !path.exists() && a.get("env-file").is_none() {
        return Ok(None);
    }
    file_credential(path, name)
}

pub(super) fn required_credential(a: &Args, name: &str) -> Res<String> {
    credential(a, name)?
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{name} is not set in the environment or env file").into())
}

pub(super) fn judge(a: &Args) -> Res<(Box<dyn Judge>, AssessConfig)> {
    let engine = a.require("engine")?;
    let (local, http) = (["judge-command", "judge-context"], ["judge-url", "judge-model"]);
    let unused = if engine == "coffee-cat-local" { http } else { local };
    if let Some(flag) = unused.iter().find(|flag| a.get(flag).is_some()) {
        return Err(format!("--{flag} does not apply to --engine {engine}").into());
    }
    let judge: Box<dyn Judge> = if engine == "coffee-cat-local" {
        let mut cfg = ProcessJudgeConfig::coffee_cat();
        if let Some(program) = a.get("judge-command") {
            cfg.program = program.into();
        }
        cfg.context = u32::try_from(a.num("judge-context", cfg.context as usize)?)?;
        Box::new(ProcessJudge::new(cfg)?)
    } else {
        let mut cfg = match engine {
            "coffee-cat" => {
                let mut cfg = HttpJudgeConfig::coffee_cat();
                cfg.api_key = credential(a, "COFFEECAT_TOKEN")?;
                cfg
            }
            "jev" => HttpJudgeConfig::jev_with_key(required_credential(a, "TYPESAFE_API_KEY")?)?,
            "openrouter" => HttpJudgeConfig::openrouter_with_key(required_credential(a, "OPENROUTER_API_KEY")?)?,
            "mercury" => HttpJudgeConfig::mercury_with_key(required_credential(a, "MERCURY_API_KEY")?)?,
            engine => {
                return Err(format!(
                    "unknown engine {engine}; choose coffee-cat, coffee-cat-local, jev, openrouter or mercury"
                )
                .into());
            }
        };
        if let Some(url) = a.get("judge-url") {
            cfg.base_url = url.to_owned();
        }
        if let Some(model) = a.get("judge-model") {
            cfg.model = Some(model.to_owned());
        }
        Box::new(HttpJudge::new(cfg)?)
    };
    let deadline = a.num("deadline-ms", 1000)?;
    if deadline == 0 {
        return Err("--deadline-ms must be positive".into());
    }
    Ok((judge, AssessConfig { deadline: Duration::from_millis(deadline as u64), ..AssessConfig::default() }))
}

/// Starts a local engine before the first deadline-bound judgment. A no-op for HTTP.
pub(super) async fn warm_up(judge: &dyn Judge) -> Res<()> {
    judge.warm_up().await.map_err(|e| format!("{} is not ready: {e}", judge.id()))?;
    Ok(())
}

pub(super) async fn permission(store: &Store, judge: &dyn Judge, a: &Args) -> Res<()> {
    let (allow, deny) = (a.switches.contains("allow-hosted"), a.switches.contains("deny-hosted"));
    if allow && deny {
        return Err("choose only one of --allow-hosted and --deny-hosted".into());
    }
    if allow || deny {
        allow_remote_judgments(store, allow).await?;
    }
    if !judge.is_loopback() && !remote_judgments_allowed(store).await? {
        return Err("this space has not enabled remote judgments; use --allow-hosted to opt in for this space".into());
    }
    Ok(())
}

/// Read only dimension metadata before opening the store. Assessment never needs
/// WordLlama assets or a new query embedding.
async fn existing_store(path: &Path) -> Res<Store> {
    let dim: String = {
        let conn = rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        conn.query_row("SELECT value FROM meta WHERE key = 'embed_dim'", [], |r| r.get(0))?
    };
    Ok(Store::open(path, StoreConfig::new(dim.parse()?)).await?)
}

pub async fn run(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let conv = locomo::load(&cfg.data, &cfg.conversation)?;
    let requests: HashMap<_, _> =
        batch::read_requests(&run, "answer")?.into_iter().map(|r| (r.id.clone(), r)).collect();
    // Validate every context before making any request or changing permissions.
    let mut contexts = Vec::new();
    for qa in &conv.qa {
        let id = answer_id(qa.index);
        let request = requests.get(&id).ok_or_else(|| format!("missing answer request {id}; run retrieve first"))?;
        let context = answer::context_from_request(request, &qa.question)
            .ok_or_else(|| format!("{id}: cannot recover the exact saved memory; regenerate with retrieve"))?;
        contexts.push((qa, context));
    }
    let (judge, assess_cfg) = judge(a)?;
    let output =
        a.get("out").map(PathBuf::from).unwrap_or_else(|| run.join("assessments").join(a.get("engine").unwrap()));
    if output.exists() {
        return Err(format!(
            "{} already exists; use a new --out directory to preserve the earlier experiment",
            output.display()
        )
        .into());
    }
    let store = existing_store(&run.join(SPACE)).await?;
    permission(&store, judge.as_ref(), a).await?;
    warm_up(judge.as_ref()).await?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    let manifest_path = output.join("manifest.json");
    let mut manifest = json!({
        "engine": a.get("engine"), "endpoint": judge.id(), "requested_model": judge.requested_model(),
        "deadline_ms": assess_cfg.deadline.as_millis() as u64, "input": batch::requests_path(&run, "answer"),
        "questions": contexts.len(), "completed": 0, "judged": 0, "unavailable": 0, "complete": false,
        "behavior": "observational; answers and retrieval are unchanged"
    });
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    let mut out = OpenOptions::new().create_new(true).write(true).open(output.join("results.jsonl"))?;
    let assessor = EvidenceAssessor::new(&store, judge.as_ref(), assess_cfg)?;
    let (mut judged, mut unavailable) = (0, 0);
    for (qa, context) in contexts {
        let assessment = assessor.assess(&qa.question, context).await?;
        if assessment.status == AssessmentStatus::Judged {
            judged += 1;
        } else {
            unavailable += 1;
        }
        // Metadata is attached AFTER assessment and never enters the model state.
        writeln!(out, "{}", json!({"qa": qa.index, "category": qa.category.id(), "assessment": assessment}))?;
        out.flush()?;
        manifest["completed"] = json!(judged + unavailable);
        manifest["judged"] = json!(judged);
        manifest["unavailable"] = json!(unavailable);
        fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    }
    manifest["complete"] = json!(true);
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    store.close().await?;
    println!("{judged} judged; {unavailable} unavailable. Results: {}", output.display());
    Ok(unavailable == 0)
}

/// Synthetic only. Uses the full policy, transport and persistence path, with no
/// private memory, dataset or embedding-model assets required.
pub async fn probe(a: &Args) -> Res<bool> {
    let (judge, cfg) = judge(a)?;
    let store = Store::open(a.require("space")?, StoreConfig::new(2)).await?;
    permission(&store, judge.as_ref(), a).await?;
    warm_up(judge.as_ref()).await?;
    let assessor = EvidenceAssessor::new(&store, judge.as_ref(), cfg)?;
    let mut success = true;
    for (case, question, context) in [
        ("supported", "Where does Ada live?", "Ada lives in Oslo. [S1:1]"),
        ("missing", "What is Ada's favorite color?", "Ada lives in Oslo. [S1:1]"),
        ("contradicted", "When did Ada move to Rome?", "Ada has never lived in Rome. [S1:2]"),
    ] {
        let assessment = assessor.assess(question, context).await?;
        success &= assessment.status == AssessmentStatus::Judged;
        println!("{}", json!({"case": case, "assessment": assessment}));
    }
    store.close().await?;
    Ok(success)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::{Router, routing::post};
    use licorice::retrieve::{Intent, Retrieved};

    use super::*;
    use crate::RunConfig;

    async fn server() -> (String, Arc<Mutex<Vec<serde_json::Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let app = Router::new().route(
            "/api/v1/decisions",
            post(move |body: axum::Json<serde_json::Value>| {
                recorded.lock().unwrap().push(body.0);
                async {
                    axum::Json(json!({"model": "coffee-cat-mock", "answers": {
                        "sufficient": {"type": "noul", "noul": 0.8},
                        "contradicted": {"type": "noul", "noul": 0.1}
                    }}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, seen)
    }

    async fn fixture(run: &Path) -> Vec<u8> {
        let data = run.join("dataset.json");
        fs::write(
            &data,
            json!([{
                "sample_id": "conv-test", "conversation": {
                    "speaker_a": "Ada", "speaker_b": "Bo",
                    "session_1_date_time": "9:00 am on 2 June, 2023",
                    "session_1": [{"speaker": "Ada", "dia_id": "D1:1", "text": "I live in Oslo."}]
                },
                "qa": [{"question": "Where does Ada live?", "answer": "GOLD_MUST_STAY_PRIVATE",
                    "category": 4, "evidence": ["D1:1"]}]
            }])
            .to_string(),
        )
        .unwrap();
        let cfg = RunConfig {
            embedding_cache: None,
            data,
            conversation: "conv-test".into(),
            models: run.join("absent-model-assets"),
            embedder: "synthetic/not-loaded".into(),
            window_turns: 8,
        };
        fs::write(run.join("run.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
        let store = Store::open(run.join(SPACE), StoreConfig::new(2)).await.unwrap();
        store.close().await.unwrap();
        let retrieved = Retrieved {
            intent: Intent::None,
            linked_entities: vec![],
            facts: vec![],
            chunks: vec![],
            sessions: vec![],
            context: "Ada lives in Oslo. [D1:1]".into(),
            cited_turns: vec!["D1:1".into()],
            context_usage: Default::default(),
            budget_omissions: Default::default(),
            selection: Default::default(),
            fixed_summary: None,
            graph_paths: vec![],
            reranking: None,
            rerank_error: None,
        };
        let request = answer::request("answer:q000", "Where does Ada live?", &retrieved);
        batch::write_requests(run, "answer", &[request]).unwrap();
        fs::read(batch::requests_path(run, "answer")).unwrap()
    }

    #[tokio::test]
    async fn saved_context_assessment_needs_no_embeddings_and_does_not_send_gold() {
        let dir = tempfile::tempdir().unwrap();
        let original = fixture(dir.path()).await;
        let (url, seen) = server().await;
        let args = Args::parse(
            ["--run", dir.path().to_str().unwrap(), "--engine", "coffee-cat", "--judge-url", &url]
                .into_iter()
                .map(str::to_owned),
        );
        assert!(run(&args).await.unwrap());
        let output = dir.path().join("assessments/coffee-cat");
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["complete"], true);
        assert_eq!(manifest["judged"], 1);
        assert_eq!(manifest["unavailable"], 0);
        {
            let seen = seen.lock().unwrap();
            assert_eq!(
                seen[0]["state"],
                json!({"question": "Where does Ada live?", "memory": "Ada lives in Oslo. [D1:1]"})
            );
            assert!(!seen[0].to_string().contains("GOLD_MUST_STAY_PRIVATE"));
        }
        assert_eq!(fs::read(batch::requests_path(dir.path(), "answer")).unwrap(), original);
        assert!(run(&args).await.unwrap_err().to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn malformed_saved_context_fails_before_creating_experiment_or_granting_permission() {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path()).await;
        let mut requests = batch::read_requests(dir.path(), "answer").unwrap();
        requests[0].user.push_str(" changed prompt");
        batch::write_requests(dir.path(), "answer", &requests).unwrap();
        let args = Args::parse(
            ["--run", dir.path().to_str().unwrap(), "--engine", "coffee-cat", "--allow-hosted"]
                .into_iter()
                .map(str::to_owned),
        );
        assert!(run(&args).await.unwrap_err().to_string().contains("exact saved memory"));
        assert!(!dir.path().join("assessments").exists());
        let store = existing_store(&dir.path().join(SPACE)).await.unwrap();
        assert!(!remote_judgments_allowed(&store).await.unwrap());
        store.close().await.unwrap();
    }

    #[tokio::test]
    async fn synthetic_probe_uses_the_full_assessment_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.licorice");
        let (url, seen) = server().await;
        let args = Args::parse(
            ["--space", path.to_str().unwrap(), "--engine", "coffee-cat", "--judge-url", &url]
                .into_iter()
                .map(str::to_owned),
        );
        assert!(probe(&args).await.unwrap());
        assert_eq!(seen.lock().unwrap().len(), 3);
        let store = existing_store(&path).await.unwrap();
        let count: i64 = store
            .read(|c| Ok(c.query_row("SELECT count(*) FROM judgments WHERE point = 'J10'", [], |r| r.get(0))?))
            .await
            .unwrap();
        assert_eq!(count, 3);
        store.close().await.unwrap();
    }

    const FAKE_COFFEECAT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../licorice-providers/tests/fixtures/fake-coffeecat");

    #[tokio::test]
    async fn synthetic_probe_runs_on_a_local_judge_process() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("probe.licorice");
        let args = Args::parse(
            ["--space", path.to_str().unwrap(), "--engine", "coffee-cat-local", "--judge-command", FAKE_COFFEECAT]
                .into_iter()
                .map(str::to_owned),
        );
        // Loopback: no --allow-hosted needed.
        assert!(probe(&args).await.unwrap());
        let store = existing_store(&path).await.unwrap();
        let models: Vec<String> = store
            .read(|c| {
                let mut stmt = c.prepare("SELECT model FROM judgments WHERE point = 'J10' ORDER BY id")?;
                Ok(stmt.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?)
            })
            .await
            .unwrap();
        store.close().await.unwrap();
        assert_eq!(models.len(), 3);
        assert!(models[0].starts_with("fake-") && models[0].ends_with(" ask --jsonl --context 4096"), "{models:?}");
        assert!(models.iter().all(|m| *m == models[0]), "one process served every case: {models:?}");
    }

    #[test]
    fn judge_flags_must_fit_the_engine() {
        let error = |raw: &[&str]| match judge(&Args::parse(raw.iter().map(|s| (*s).to_owned()))) {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        assert!(error(&["--engine", "coffee-cat-local", "--judge-url", "http://x"]).contains("--judge-url"));
        assert!(error(&["--engine", "coffee-cat", "--judge-command", "coffeecat"]).contains("--judge-command"));
        assert!(error(&["--engine", "coffee-cat-local", "--judge-context", "4000"]).contains("multiple of 256"));
        let (local, _) = judge(&Args::parse(
            ["--engine", "coffee-cat-local", "--judge-context", "8192"].into_iter().map(str::to_owned),
        ))
        .unwrap();
        assert!(local.is_loopback());
        assert_eq!(local.id(), "coffeecat ask --jsonl --context 8192");
    }

    #[test]
    fn permission_flags_are_switches_even_before_positional_arguments() {
        let args =
            Args::parse(["--allow-hosted", "positional", "--engine", "coffee-cat"].into_iter().map(str::to_owned));
        assert!(args.switches.contains("allow-hosted"));
        assert_eq!(args.positional, ["positional"]);
        assert_eq!(args.get("engine"), Some("coffee-cat"));
    }

    #[test]
    fn env_file_credentials_are_parsed_without_process_mutation_or_secret_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.env");
        fs::write(
            &path,
            "export LICORICE_ENV_TEST_KEY='synthetic-secret'\nOPENROUTER_API_KEY=synthetic-openrouter-key\n",
        )
        .unwrap();
        assert_eq!(file_credential(&path, "OPENROUTER_API_KEY").unwrap().as_deref(), Some("synthetic-openrouter-key"));
        assert!(std::env::var_os("LICORICE_ENV_TEST_KEY").is_none());
        fs::write(&path, "OPENROUTER_API_KEY='do-not-echo-this-secret\n").unwrap();
        let error = file_credential(&path, "OPENROUTER_API_KEY").unwrap_err().to_string();
        assert!(!error.contains("do-not-echo-this-secret"));
        assert!(error.contains("values omitted"));
    }
}
