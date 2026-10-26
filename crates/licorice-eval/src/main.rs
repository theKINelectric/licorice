//! `licorice-eval`: runs Licorice on a LoCoMo conversation in steps. Every model call is
//! written to a request file and answered from a response file ([`batch`]), so the same
//! run can be driven by a live model, by replayed outputs, or by a stand-in.
//!
//! ```text
//! prepare         ingest the conversation; write extraction and summary requests
//! apply           apply extraction and summary outputs to the space
//! retrieve        retrieve for every question; write answer requests (no gold answers)
//! grade-requests  pair answers with gold answers; write grading requests
//! report          accuracy, token F1 and evidence recall by category
//! check           validate one batch's response file
//! batches         re-split a task's requests into batches of another size
//! ```

mod assess;
mod batch;
mod budget_probe;
mod cached_embed;
mod chat_execute;
mod evidence_search;
mod execute;
mod grade;
mod locomo;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use licorice::ingest::{IngestConfig, Ingestor};
use licorice::retrieve::{Intent, Query, RetrieveConfig, Retriever, SelectionMode, SummaryMode};
use licorice::temporal::Day;
use licorice::{Store, StoreConfig, answer};
use licorice_providers::embed::{Embedder, WordLlama};
use licorice_providers::tokens::{Cl100kCounter, HfCounter, TokenCounter};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use batch::Res;
use locomo::{Category, Conversation};

const USAGE: &str = "usage:
  licorice-eval prepare --run DIR --data locomo10.json [--conv conv-26] [--models models/wordllama]
                        [--window 8] [--extract-batch 6] [--summary-batch 10]
  licorice-eval apply --run DIR [--partial]
  licorice-eval execute --run DIR --task extract|summarize|answer|grade --ledger FILE --budget-usd N
                        [--model claude-sonnet-4-6] [--env-file FILE] [--limit N]
  licorice-eval retrieve --run DIR [--answer-batch 25] [--recency-weight W]
                         [--budget-tokens 2000] [--selection relevance|mmr] [--mmr-lambda 0.7]
                         [--selection-candidates 60] [--context-tokenizer tokenizer.json] [--fixed-summary-tokens 600] [--summary-mode sessions|themes]
  licorice-eval evidence-search --run DIR --input JSON --out JSON
                         [--cross-session | --source-aware] (opt-in bounded source selection)
                         (annotation-free as-of snapshot; <=2 subqueries; 2000-token memory)
  licorice-eval grade-requests --run DIR [--grade-batch 50] [--rubric legacy|source-v2] [--partial]
  licorice-eval grade-controls --run DIR
  licorice-eval report --run DIR
  licorice-eval check --run DIR BATCH
  licorice-eval batches --run DIR --task TASK --size N
  licorice-eval assess --run DIR --engine coffee-cat|coffee-cat-local|jev|openrouter|mercury [--out DIR]
                      [--deadline-ms 1000] [--judge-url URL] [--judge-model MODEL]
                      [--judge-command coffeecat] [--judge-context 4096] (coffee-cat-local only)
                      [--allow-hosted | --deny-hosted] [--env-file FILE (default .env)]
  licorice-eval judge-probe --space FILE --engine ENGINE [same judge options]
  licorice-eval budget-probe --out DIR [--engine ENGINE] [same judge options]";

const SPACE: &str = "space.licorice";

/// What `prepare` fixed for the rest of the run.
#[derive(Serialize, Deserialize)]
struct RunConfig {
    data: PathBuf,
    conversation: String,
    models: PathBuf,
    embedder: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_cache: Option<PathBuf>,
    window_turns: usize,
}

impl RunConfig {
    fn ingest(&self) -> IngestConfig {
        IngestConfig { window_turns: self.window_turns, ..IngestConfig::default() }
    }
}

struct Args {
    flags: HashMap<String, String>,
    switches: HashSet<String>,
    positional: Vec<String>,
}

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Args {
        let raw: Vec<String> = raw.collect();
        let mut args = Args { flags: HashMap::new(), switches: HashSet::new(), positional: Vec::new() };
        let mut i = 0;
        while i < raw.len() {
            match raw[i].strip_prefix("--") {
                Some(name)
                    if raw.get(i + 1).is_some_and(|v| !v.starts_with("--"))
                        && !matches!(
                            name,
                            "partial"
                                | "allow-hosted"
                                | "deny-hosted"
                                | "connected-evidence"
                                | "raw-only"
                                | "state-window"
                                | "compact-excerpt-headers"
                                | "cross-session"
                                | "source-aware"
                        ) =>
                {
                    args.flags.insert(name.to_owned(), raw[i + 1].clone());
                    i += 1;
                }
                Some(name) => {
                    args.switches.insert(name.to_owned());
                }
                None => args.positional.push(raw[i].clone()),
            }
            i += 1;
        }
        args
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(String::as_str)
    }

    fn require(&self, name: &str) -> Res<&str> {
        Ok(self.get(name).ok_or_else(|| format!("missing --{name}\n{USAGE}"))?)
    }

    fn num(&self, name: &str, default: usize) -> Res<usize> {
        match self.get(name) {
            Some(v) => Ok(v.parse().map_err(|_| format!("--{name}: not a number: {v}"))?),
            None => Ok(default),
        }
    }

    fn run_dir(&self) -> Res<PathBuf> {
        Ok(PathBuf::from(self.require("run")?))
    }
}

fn main() -> ExitCode {
    let mut raw = std::env::args().skip(1);
    let Some(command) = raw.next() else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    let args = Args::parse(raw);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    let result = rt.block_on(async {
        match command.as_str() {
            "prepare" => prepare(&args).await,
            "apply" => apply(&args).await,
            "execute" => execute::run(&args).await,
            "chat-execute" => chat_execute::run(&args).await,
            "retrieve" => retrieve(&args).await,
            "evidence-search" => evidence_search::run(&args).await,
            "grade-requests" => grade_requests(&args).await,
            "grade-controls" => grade_controls(&args),
            "report" => report(&args).await,
            "check" => check(&args),
            "batches" => rebatch(&args),
            "assess" => assess::run(&args).await,
            "judge-probe" => assess::probe(&args).await,
            "budget-probe" => budget_probe::run(&args).await,
            _ => Err(USAGE.into()),
        }
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn load_run(run: &Path) -> Res<RunConfig> {
    let path = run.join("run.json");
    Ok(serde_json::from_str(&fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?)?)
}

async fn open(run: &Path, cfg: &RunConfig) -> Res<(Store, Box<dyn Embedder>)> {
    use licorice::rusqlite::OptionalExtension;
    let embedder: Box<dyn Embedder> = match &cfg.embedding_cache {
        Some(path) => Box::new(cached_embed::CachedEmbedder::load(path)?),
        None => Box::new(WordLlama::load(&cfg.models)?),
    };
    if cfg.embedder != embedder.id() {
        return Err("declared run embedder differs from the loaded model".into());
    }
    let store = Store::open(run.join(SPACE), StoreConfig::new(embedder.dim())).await?;
    let recorded: Option<String> = store
        .read(|c| {
            Ok(c.query_row("SELECT value FROM meta WHERE key='eval.embedder_id'", [], |r| r.get(0)).optional()?)
        })
        .await?;
    if recorded.as_ref().is_some_and(|id| id != embedder.id()) || (cfg.embedding_cache.is_some() && recorded.is_none())
    {
        store.close().await?;
        return Err(
            "stored vectors do not declare the requested offline embedding model; reindex a separate snapshot first"
                .into(),
        );
    }
    Ok((store, embedder))
}

fn write_jsonl(path: &Path, rows: &[Value]) -> Res<()> {
    let mut out = String::new();
    for r in rows {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

fn read_jsonl(path: &Path) -> Res<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?)
}

fn print_batches(task: &str, names: &[String]) {
    println!("{task}: {} batches ({} … {})", names.len(), names[0], names[names.len() - 1]);
}

async fn prepare(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    if run.join(SPACE).exists() {
        return Err(format!("{} already holds a space; use a new run directory", run.display()).into());
    }
    fs::create_dir_all(&run)?;
    let data = PathBuf::from(a.require("data")?).canonicalize()?;
    let conv = locomo::load(&data, a.get("conv").unwrap_or("conv-26"))?;
    let models = PathBuf::from(a.get("models").unwrap_or("models/wordllama")).canonicalize()?;
    let embedder = WordLlama::load(&models)?;
    let cfg = RunConfig {
        data,
        conversation: conv.id.clone(),
        models,
        embedder: embedder.id().to_owned(),
        embedding_cache: None,
        window_turns: a.num("window", IngestConfig::default().window_turns)?,
    };
    fs::write(run.join("run.json"), serde_json::to_string_pretty(&cfg)?)?;

    let store = Store::open(run.join(SPACE), StoreConfig::new(embedder.dim())).await?;
    let ing = Ingestor::new(&store, &embedder, cfg.ingest());
    let (mut extract, mut summaries) = (Vec::new(), Vec::new());
    let turns: usize = conv.sessions.iter().map(|s| s.turns.len()).sum();
    for session in conv.sessions.iter().cloned() {
        let ext = session.ext_id.clone();
        ing.append_session(session).await?;
        extract.extend(ing.extraction_requests(&ext).await?);
        summaries.push(ing.summary_request(&ext).await?);
    }
    store.close().await?;
    batch::write_requests(&run, "extract", &extract)?;
    batch::write_requests(&run, "summarize", &summaries)?;
    println!(
        "{} ({}): {} sessions, {turns} turns, {} questions",
        conv.id,
        conv.speakers.join(" and "),
        conv.sessions.len(),
        conv.qa.len()
    );
    print_batches("extract", &batch::write_batches(&run, "extract", &extract, a.num("extract-batch", 6)?)?);
    print_batches("summarize", &batch::write_batches(&run, "summarize", &summaries, a.num("summary-batch", 10)?)?);
    Ok(true)
}

async fn apply(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let (mut todo, mut missing) = (Vec::new(), 0);
    for task in ["extract", "summarize"] {
        let (outputs, problems) = batch::collect(&run, task)?;
        for p in &problems {
            eprintln!("warning: {p}");
        }
        for r in batch::read_requests(&run, task)? {
            match outputs.get(&r.id) {
                Some(out) => todo.push((task, r.id, out.clone())),
                None => missing += 1,
            }
        }
    }
    if missing > 0 && !a.switches.contains("partial") {
        return Err(format!("{missing} requests have no valid output yet; pass --partial to apply the rest").into());
    }
    let (store, embedder) = open(&run, &cfg).await?;
    let ing = Ingestor::new(&store, embedder.as_ref(), cfg.ingest());
    let mut facts = 0;
    for (task, id, output) in &todo {
        if *task == "extract" {
            let r = ing.apply_extraction(id, output).await?;
            if !r.replayed {
                facts += r.facts_created;
            }
        } else {
            ing.apply_summary(id, output).await?;
        }
    }
    write_apply_projection(&store, &run).await?;
    store.close().await?;
    println!("applied {} outputs ({facts} new facts); {missing} still missing", todo.len());
    Ok(true)
}

async fn write_apply_projection(store: &Store, run: &Path) -> Res<()> {
    // These are projections for inspection/reporting, never the replay authority.
    // A crash after a DB commit can regenerate them without double application.
    let rows = store
        .read(|conn| {
            let mut stmt = conn.prepare("SELECT id,report FROM applied_outputs ORDER BY id")?;
            let mut rows = Vec::new();
            for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))? {
                let (id, report) = row?;
                let mut report: Value =
                    serde_json::from_str(&report).map_err(|e| licorice::Error::Task(e.to_string()))?;
                report["id"] = json!(id);
                rows.push(report);
            }
            Ok(rows)
        })
        .await?;
    write_jsonl(&run.join("apply.jsonl"), &rows)?;
    fs::write(
        run.join("applied.txt"),
        rows.iter().filter_map(|r| r["id"].as_str()).collect::<Vec<_>>().join("\n") + "\n",
    )?;
    Ok(())
}

fn intent_name(intent: &Intent) -> String {
    match intent {
        Intent::None => "none".into(),
        Intent::Window(i) => format!("window {}..{}", i.from.iso(), i.to.iso()),
        Intent::Current => "current".into(),
        Intent::When => "when".into(),
        Intent::Ordering => "ordering".into(),
    }
}

fn answer_id(index: usize) -> String {
    format!("answer:q{index:03}")
}

async fn retrieve(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let conv = locomo::load(&cfg.data, &cfg.conversation)?;
    let (store, embedder) = open(&run, &cfg).await?;
    let mut rcfg = RetrieveConfig::default();
    rcfg.budget_tokens = a.num("budget-tokens", rcfg.budget_tokens)?;
    rcfg.selection_candidates = a.num("selection-candidates", rcfg.selection_candidates)?;
    rcfg.graph_hops = a.num("graph-hops", 0)?;
    rcfg.rerank_candidates = a.num("rerank-candidates", rcfg.rerank_candidates)?;
    rcfg.connected_evidence = a.switches.contains("connected-evidence");
    rcfg.filter_state_window = a.switches.contains("state-window");
    rcfg.compact_excerpt_headers = a.switches.contains("compact-excerpt-headers");
    rcfg.query_speaker = a.get("query-speaker").map(str::to_owned);
    if a.switches.contains("raw-only") {
        rcfg.final_facts = 0;
        rcfg.summaries = 0;
        rcfg.chunks = 100;
        rcfg.chunk_candidates = 100;
    }
    rcfg.chunks = a.num("max-excerpts", rcfg.chunks)?;
    if rcfg.chunks > 100 {
        return Err("--max-excerpts must be in 0..=100".into());
    }
    let reranker = if a.get("rerank") == Some("gpt-4o-mini") {
        let key = assess::required_credential(a, "OPENROUTER_API_KEY")?;
        let model = licorice_providers::chat_http::BudgetChat::open(
            key,
            Path::new(a.require("ledger")?),
            a.require("budget-usd")?.parse()?,
        )?;
        rcfg.rerank_fallback = true;
        let scorer: std::sync::Arc<dyn licorice_providers::rerank::Reranker> =
            std::sync::Arc::new(licorice_providers::rerank::ChatReranker(std::sync::Arc::new(model)));
        Some(if rcfg.rerank_candidates > 24 {
            std::sync::Arc::new(licorice_providers::rerank::BatchedReranker(scorer))
                as std::sync::Arc<dyn licorice_providers::rerank::Reranker>
        } else {
            scorer
        })
    } else if a.get("rerank").is_some() {
        return Err("--rerank must be gpt-4o-mini".into());
    } else {
        None
    };
    let source_map: HashMap<String, String> = a
        .get("source-map")
        .map(|path| -> Res<_> { Ok(serde_json::from_str(&fs::read_to_string(path)?)?) })
        .transpose()?
        .unwrap_or_default();
    rcfg.fixed_summary_tokens = a.get("fixed-summary-tokens").map(str::parse).transpose()?;
    rcfg.summary_mode = match a.get("summary-mode").unwrap_or("sessions") {
        "sessions" => SummaryMode::Sessions,
        "themes" => SummaryMode::Themes,
        _ => return Err("--summary-mode must be sessions or themes".into()),
    };
    rcfg.selection = match a.get("selection").unwrap_or("relevance") {
        "relevance" => SelectionMode::Relevance,
        "mmr" => SelectionMode::Mmr,
        _ => return Err("--selection must be relevance or mmr".into()),
    };
    if let Some(v) = a.get("mmr-lambda") {
        rcfg.mmr_lambda = v.parse().map_err(|_| "--mmr-lambda must be a number")?;
    }
    if let Some(w) = a.get("recency-weight") {
        rcfg.recency_weight = w.parse().map_err(|_| format!("--recency-weight: not a number: {w}"))?;
    }
    let hf_counter = a.get("context-tokenizer").map(HfCounter::from_file).transpose()?;
    let counter: &dyn TokenCounter = hf_counter.as_ref().map(|c| c as &dyn TokenCounter).unwrap_or(&Cl100kCounter);
    let mut recorded_cfg = serde_json::to_value(&rcfg)?;
    recorded_cfg["context_tokenizer"] = json!(counter.id());
    fs::write(run.join("retrieve.json"), serde_json::to_string_pretty(&recorded_cfg)?)?;
    let mut retriever = Retriever::new(&store, embedder.as_ref(), rcfg).with_token_counter(counter);
    if let Some(reranker) = &reranker {
        retriever = retriever.with_reranker(reranker.as_ref());
    }
    let canonical = |id: &str| source_map.get(id).map(String::as_str).unwrap_or(id).to_owned();
    let at = conv.question_time();
    let (mut rows, mut requests) = (Vec::new(), Vec::new());
    for qa in &conv.qa {
        let started = Instant::now();
        let r = retriever.retrieve(&Query { text: qa.question.clone(), at }).await?;
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let evidence: Vec<&String> = qa.evidence.iter().filter(|e| conv.has_turn(e)).collect();
        let in_facts: HashSet<&str> =
            r.facts.iter().flat_map(|f| f.evidence_turns.iter().map(String::as_str)).collect();
        let in_excerpts: HashSet<&str> = r.chunks.iter().flat_map(|c| c.turns.iter().map(|t| t.0.as_str())).collect();
        let recall = |found: &dyn Fn(&str) -> bool| -> Value {
            if evidence.is_empty() {
                Value::Null
            } else {
                json!(evidence.iter().filter(|e| found(e)).count() as f64 / evidence.len() as f64)
            }
        };
        rows.push(json!({
            "qa": qa.index,
            "category": qa.category.id(),
            "intent": intent_name(&r.intent),
            "linked": r.linked_entities,
            "facts": r.facts.len(),
            "chunks": r.chunks.len(),
            "context_chars": r.context.len(),
            "context_usage":r.context_usage,
            "budget_omissions":r.budget_omissions,
            "selection":r.selection,
            "fixed_summary":r.fixed_summary,
            "evidence": evidence,
            "invalid_evidence": qa.evidence.len() - evidence.len(),
            "recall": recall(&|t| in_facts.iter().chain(in_excerpts.iter()).any(|id|canonical(id)==t)),
            "fact_recall": recall(&|t| in_facts.iter().any(|id|canonical(id)==t)),
            "excerpt_recall": recall(&|t| in_excerpts.iter().any(|id|canonical(id)==t)),
            "cited_turns":r.cited_turns.iter().map(|id|canonical(id)).collect::<Vec<_>>(),
            "graph_paths":r.graph_paths,
            "reranking":r.reranking,"rerank_error":r.rerank_error,
            "ms": ms,
        }));
        requests.push(answer::request(answer_id(qa.index), &qa.question, &r));
    }
    store.close().await?;
    write_jsonl(&run.join("retrieval.jsonl"), &rows)?;
    batch::write_requests(&run, "answer", &requests)?;
    println!("retrieved for {} questions at {}", rows.len(), Day::from_ms(at).human());
    print_batches("answer", &batch::write_batches(&run, "answer", &requests, a.num("answer-batch", 25)?)?);
    Ok(true)
}

async fn grade_requests(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let conv = locomo::load(&cfg.data, &cfg.conversation)?;
    let source_v2 = match a.get("rubric").unwrap_or("legacy") {
        "legacy" => false,
        "source-v2" => true,
        _ => return Err("--rubric must be legacy or source-v2".into()),
    };
    let task = if source_v2 { "adjudicate" } else { "grade" };
    let source_extra: HashMap<String, Vec<String>> = a
        .get("source-extra")
        .map(|path| -> Res<_> { Ok(serde_json::from_str(&fs::read_to_string(path)?)?) })
        .transpose()?
        .unwrap_or_default();
    let source_budget = a.num("source-budget-tokens", 3000)?;
    let (answers, problems) = batch::collect(&run, "answer")?;
    for p in &problems {
        eprintln!("warning: {p}");
    }
    let mut requests = Vec::new();
    let mut missing = 0;
    for qa in &conv.qa {
        match answers.get(&answer_id(qa.index)).and_then(Value::as_str) {
            Some(response) => requests.push(if source_v2 {
                grade::source_request(
                    qa,
                    response,
                    &if let Some(extra) = source_extra.get(&qa.index.to_string()) {
                        grade::source_packet_with_extra(&conv, qa, source_budget, extra)?
                    } else {
                        grade::source_packet(&conv, qa, source_budget)?
                    },
                )
            } else {
                grade::request(qa, response)
            }),
            None => missing += 1,
        }
    }
    batch::write_requests(&run, task, &requests)?;
    println!("{} grading requests; {missing} questions have no answer yet", requests.len());
    print_batches(task, &batch::write_batches(&run, task, &requests, a.num("grade-batch", 50)?)?);
    Ok(missing == 0 || (source_v2 && a.switches.contains("partial")))
}

fn grade_controls(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let controls = grade::controls();
    let expected: Vec<Value> =
        controls.iter().map(|(r, v, s)| json!({"id":r.id,"verdict":v,"gold_status":s})).collect();
    let requests: Vec<_> = controls.into_iter().map(|(r, _, _)| r).collect();
    batch::write_requests(&run, "adjudicate", &requests)?;
    fs::write(run.join("expected.json"), serde_json::to_string_pretty(&expected)?)?;
    Ok(true)
}

#[derive(Default)]
struct Tally {
    graded: usize,
    correct: usize,
    f1: Vec<f64>,
    recall: Vec<f64>,
    full_recall: usize,
    wrong_no_evidence: usize,
    wrong_some_evidence: usize,
    context_chars: Vec<f64>,
    context_tokens: Vec<f64>,
}

impl Tally {
    fn add(&mut self, correct: bool, f1: Option<f64>, retrieval: Option<&Value>) {
        self.graded += 1;
        self.correct += correct as usize;
        self.f1.extend(f1);
        if let Some(r) = retrieval {
            self.context_chars.extend(r["context_chars"].as_f64());
            self.context_tokens.extend(r["context_usage"]["tokens"].as_f64());
            if let Some(recall) = r["recall"].as_f64() {
                self.recall.push(recall);
                self.full_recall += (recall >= 1.0) as usize;
                if !correct {
                    if recall == 0.0 { self.wrong_no_evidence += 1 } else { self.wrong_some_evidence += 1 }
                }
            }
        }
    }

    fn json(&self) -> Value {
        let (lo, hi) = grade::wilson(self.correct, self.graded);
        json!({
            "graded": self.graded,
            "correct": self.correct,
            "accuracy": ratio(self.correct, self.graded),
            "accuracy_ci95": [lo, hi],
            "token_f1": mean(&self.f1),
            "evidence_recall": mean(&self.recall),
            "full_evidence_share": ratio(self.full_recall, self.recall.len()),
            "wrong_with_no_evidence": self.wrong_no_evidence,
            "wrong_with_some_evidence": self.wrong_some_evidence,
            "context_tokens_est": mean(&self.context_chars).map(|c| c / 4.0),
            "context_tokens": mean(&self.context_tokens),
            "context_tokens_measured_rows":self.context_tokens.len(),
        })
    }
}

fn ratio(k: usize, n: usize) -> Option<f64> {
    (n > 0).then(|| k as f64 / n as f64)
}

fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

fn percentile(v: &[f64], p: f64) -> Option<f64> {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    (!v.is_empty()).then(|| v[((v.len() as f64 - 1.0) * p).round() as usize])
}

async fn space_stats(run: &Path, cfg: &RunConfig) -> Res<Value> {
    let (store, _) = open(run, cfg).await?;
    let stats = store
        .read(|c| {
            let n = |sql: &str| c.query_row(sql, [], |r| r.get::<_, i64>(0));
            Ok(json!({
                "sessions": n("SELECT count(*) FROM sessions")?,
                "turns": n("SELECT count(*) FROM turns")?,
                "chunks": n("SELECT count(*) FROM chunks")?,
                "entities": n("SELECT count(*) FROM entities WHERE merged_into IS NULL")?,
                "aliases": n("SELECT count(*) FROM entity_aliases")?,
                "facts": n("SELECT count(*) FROM facts")?,
                "facts_dated": n("SELECT count(*) FROM facts WHERE occurred_from IS NOT NULL")?,
                "summaries": n("SELECT count(*) FROM summaries")?,
            }))
        })
        .await?;
    store.close().await?;
    Ok(stats)
}

fn fmt_pct(v: &Value) -> String {
    v.as_f64().map(|x| format!("{:.1}", x * 100.0)).unwrap_or_else(|| "–".into())
}

async fn report(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let conv = locomo::load(&cfg.data, &cfg.conversation)?;
    let (answers, _) = batch::collect(&run, "answer")?;
    let (grades, problems) = batch::collect(&run, "grade")?;
    for p in &problems {
        eprintln!("warning: {p}");
    }
    let retrieval: HashMap<u64, Value> =
        read_jsonl(&run.join("retrieval.jsonl"))?.into_iter().filter_map(|r| Some((r["qa"].as_u64()?, r))).collect();

    let mut by_category: BTreeMap<Category, Tally> = BTreeMap::new();
    let (mut answerable, mut all) = (Tally::default(), Tally::default());
    let mut questions = Vec::new();
    for qa in &conv.qa {
        let Some(g) = grades.get(&grade::request_id(qa)) else { continue };
        let correct = grade::is_correct(g);
        let response = answers.get(&answer_id(qa.index)).and_then(Value::as_str).unwrap_or_default();
        let f1 = match (qa.category, &qa.answer) {
            (Category::Adversarial, _) | (_, None) => None,
            (_, Some(gold)) => Some(grade::token_f1(response, gold)),
        };
        let r = retrieval.get(&(qa.index as u64));
        by_category.entry(qa.category).or_default().add(correct, f1, r);
        if qa.category != Category::Adversarial {
            answerable.add(correct, f1, r);
        }
        all.add(correct, f1, r);
        questions.push(json!({"qa": qa.index, "category": qa.category.id(), "correct": correct, "token_f1": f1,
                              "recall": r.map(|r| r["recall"].clone())}));
    }
    let latencies: Vec<f64> = retrieval.values().filter_map(|r| r["ms"].as_f64()).collect();
    let applied = read_jsonl(&run.join("apply.jsonl"))?;
    let sum = |k: &str| applied.iter().filter_map(|r| r[k].as_i64()).sum::<i64>();
    let stats = space_stats(&run, &cfg).await?;
    let complete = all.graded == conv.qa.len() && answers.len() == conv.qa.len();
    let result = json!({
        "conversation": conv.id,
        "embedder": cfg.embedder,
        "window_turns": cfg.window_turns,
        // Written by `retrieve`; runs from before it existed used CogniRank at full weight.
        "retrieve": fs::read_to_string(run.join("retrieve.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok())
            .unwrap_or_else(|| json!({"recency_weight": 1.0})),
        "questions": conv.qa.len(),
        "graded": all.graded,
        "complete":complete,
        "missing_answers":conv.qa.len().saturating_sub(answers.len()),
        "missing_grades":conv.qa.len().saturating_sub(all.graded),
        "answerable": answerable.json(),
        "adversarial": by_category.get(&Category::Adversarial).map(Tally::json),
        "all": all.json(),
        "categories": by_category.iter().map(|(c, t)| (c.name().to_owned(), t.json())).collect::<BTreeMap<_, _>>(),
        "retrieval_ms": {"p50": percentile(&latencies, 0.5), "p95": percentile(&latencies, 0.95), "max": percentile(&latencies, 1.0)},
        "ingest": {
            "space": stats,
            "facts_merged": sum("facts_merged"),
            "facts_dropped": sum("facts_dropped"),
            "entities_linked": sum("entities_linked"),
        },
        "per_question": questions,
    });
    fs::write(run.join("report.json"), serde_json::to_string_pretty(&result)?)?;
    let md = render_report(&conv, &result, &by_category);
    fs::write(run.join("report.md"), &md)?;
    print!("{md}");
    Ok(complete)
}

fn render_report(conv: &Conversation, r: &Value, by_category: &BTreeMap<Category, Tally>) -> String {
    let mut md = format!(
        "# LoCoMo {}: Licorice retrieval\n\n\
         {} questions, {} graded. Embedder `{}`; extraction windows of {} turns; recency weight {}.\n\n\
         | Category | n | Accuracy % (95% CI) | Token F1 | Evidence recall % | All evidence % | Wrong, no evidence | Wrong, some evidence | Context tokens |\n\
         |---|---|---|---|---|---|---|---|---|\n",
        conv.id,
        r["questions"],
        r["graded"],
        r["embedder"].as_str().unwrap_or_default(),
        r["window_turns"],
        r["retrieve"]["recency_weight"]
    );
    if !r["retrieve"]["budget_tokens"].is_null() {
        // Keep legacy reports readable while identifying actual-budget runs.
        let details = format!(
            "Memory budget: {} tokens; tokenizer `{}`; selection `{}`.\n\n",
            r["retrieve"]["budget_tokens"],
            r["retrieve"]["context_tokenizer"].as_str().unwrap_or("unknown"),
            r["retrieve"]["selection"].as_str().unwrap_or("unknown")
        );
        let table = md.find("| Category").unwrap_or(md.len());
        md.insert_str(table, &details);
    }
    let row = |name: &str, t: &Value| {
        let ci = &t["accuracy_ci95"];
        format!(
            "| {name} | {} | {} ({}–{}) | {} | {} | {} | {} | {} | {} |\n",
            t["graded"],
            fmt_pct(&t["accuracy"]),
            fmt_pct(&ci[0]),
            fmt_pct(&ci[1]),
            t["token_f1"].as_f64().map(|f| format!("{f:.3}")).unwrap_or_else(|| "–".into()),
            fmt_pct(&t["evidence_recall"]),
            fmt_pct(&t["full_evidence_share"]),
            t["wrong_with_no_evidence"],
            t["wrong_with_some_evidence"],
            t["context_tokens"]
                .as_f64()
                .map(|c| format!("{c:.0}"))
                .or_else(|| t["context_tokens_est"].as_f64().map(|c| format!("~{c:.0}")))
                .unwrap_or_else(|| "–".into()),
        )
    };
    for (c, t) in by_category {
        md.push_str(&row(&format!("{} ({})", c.name(), c.id()), &t.json()));
    }
    md.push_str(&row("**categories 1–4**", &r["answerable"]));
    md.push_str(&row("all five", &r["all"]));
    let ms = &r["retrieval_ms"];
    let ing = &r["ingest"];
    let space = &ing["space"];
    md.push_str(&format!(
        "\nRetrieval latency (ms, includes query embedding): p50 {:.1}, p95 {:.1}, max {:.1}.\n\n\
         Space: {} sessions, {} turns, {} chunks, {} entities ({} aliases), {} facts ({} with a resolved date), {} summaries. \
         Extraction merged {} duplicate facts, dropped {} and linked {} entity mentions to existing entities.\n",
        ms["p50"].as_f64().unwrap_or(0.0),
        ms["p95"].as_f64().unwrap_or(0.0),
        ms["max"].as_f64().unwrap_or(0.0),
        space["sessions"],
        space["turns"],
        space["chunks"],
        space["entities"],
        space["aliases"],
        space["facts"],
        space["facts_dated"],
        space["summaries"],
        ing["facts_merged"],
        ing["facts_dropped"],
        ing["entities_linked"],
    ));
    md
}

fn check(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let name = a.positional.first().ok_or_else(|| format!("missing BATCH\n{USAGE}"))?;
    let name = name.trim_end_matches(".jsonl").trim_end_matches(".md").rsplit('/').next().unwrap_or(name);
    let problems = batch::check(&run, name)?;
    if problems.is_empty() {
        println!("ok: {name}");
        return Ok(true);
    }
    for p in &problems {
        println!("{p}");
    }
    println!("{} problems in {name}", problems.len());
    Ok(false)
}

fn rebatch(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let task = a.require("task")?;
    let requests = batch::read_requests(&run, task)?;
    print_batches(task, &batch::write_batches(&run, task, &requests, a.num("size", 10)?)?);
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_session_switch_does_not_consume_the_next_flag_value() {
        let a = Args::parse(["--cross-session", "--run", "example"].into_iter().map(str::to_owned));
        assert!(a.switches.contains("cross-session"));
        assert_eq!(a.get("run"), Some("example"));
    }
    #[tokio::test]
    async fn offline_embeddings_require_the_same_declared_model_in_run_cache_and_store() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("embeddings.json");
        fs::write(&cache, json!({"model_id":"fixture","dimensions":2,"rows":[]}).to_string()).unwrap();
        let mut cfg = RunConfig {
            data: dir.path().join("unused"),
            conversation: "fixture".into(),
            models: dir.path().join("absent"),
            embedder: "fixture".into(),
            embedding_cache: Some(cache),
            window_turns: 1,
        };
        let store = Store::open(dir.path().join(SPACE), StoreConfig::new(2)).await.unwrap();
        store.close().await.unwrap();
        assert!(
            open(dir.path(), &cfg).await.is_err(),
            "an unmarked existing vector space must not accept cached queries from another model"
        );
        for id in ["other-model", "fixture"] {
            let store = Store::open(dir.path().join(SPACE), StoreConfig::new(2)).await.unwrap();
            let id = id.to_owned();
            let matching = id == "fixture";
            store
                .write(move |w| {
                    w.conn().execute("INSERT OR REPLACE INTO meta(key,value) VALUES('eval.embedder_id',?1)", [id])?;
                    Ok(())
                })
                .await
                .unwrap();
            store.close().await.unwrap();
            let result = open(dir.path(), &cfg).await;
            if matching {
                let (store, embedder) = result.unwrap();
                assert_eq!(embedder.id(), "fixture");
                store.close().await.unwrap();
            } else {
                assert!(result.is_err(), "same dimension does not make model identities interchangeable");
            }
        }
        cfg.embedder = "wrong-run-declaration".into();
        assert!(open(dir.path(), &cfg).await.is_err());
    }

    #[tokio::test]
    async fn application_logs_recover_from_committed_receipts() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join(SPACE), StoreConfig::new(2)).await.unwrap();
        store.write(|w|{
            w.conn().execute("INSERT INTO sessions(ext_id,started_at) VALUES ('S1',0)",[])?;
            w.conn().execute("INSERT INTO applied_outputs(id,kind,session_id,input_hash,output_hash,report,created_at) VALUES ('extract:S1:0-0','extract',1,zeroblob(32),zeroblob(32),'{\"facts_created\":2}',0)",[])?;
            Ok(())
        }).await.unwrap();
        // Simulate a crash before logs were written, then a stale/partial log.
        write_apply_projection(&store, dir.path()).await.unwrap();
        let expected = fs::read(dir.path().join("apply.jsonl")).unwrap();
        fs::write(dir.path().join("apply.jsonl"), "partial").unwrap();
        fs::write(dir.path().join("applied.txt"), "nonexistent-id\n").unwrap();
        write_apply_projection(&store, dir.path()).await.unwrap();
        assert_eq!(fs::read(dir.path().join("apply.jsonl")).unwrap(), expected);
        assert_eq!(fs::read_to_string(dir.path().join("applied.txt")).unwrap(), "extract:S1:0-0\n");
        let rows = read_jsonl(&dir.path().join("apply.jsonl")).unwrap();
        assert_eq!(rows[0]["facts_created"], 2);
        store.close().await.unwrap();
    }

    #[test]
    fn report_keeps_measured_tokens_separate_from_legacy_estimates() {
        let mut tally = Tally::default();
        tally.add(true, None, Some(&json!({"context_chars":100,"context_usage":{"tokens":42}})));
        assert_eq!(tally.json()["context_tokens"], 42.0);
        assert_eq!(tally.json()["context_tokens_est"], 25.0);
        let mut legacy = Tally::default();
        legacy.add(true, None, Some(&json!({"context_chars":100})));
        assert!(legacy.json()["context_tokens"].is_null());
        assert_eq!(legacy.json()["context_tokens_measured_rows"], 0);
    }
}
