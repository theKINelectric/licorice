//! A reproducible synthetic packing control. Deliberate embedding geometry isolates
//! the selection mechanism; this is not an embedding-quality or held-out benchmark.

use std::fs;

use licorice::assess::{AssessmentStatus, EvidenceAssessor};
use licorice::ingest::{IngestConfig, Ingestor, Role, SessionInput, TurnInput};
use licorice::retrieve::{Query, RetrieveConfig, Retriever, SelectionMode};
use licorice::temporal::Day;
use licorice::{Store, StoreConfig};
use licorice_providers::embed::{EmbedError, EmbedKind, Embedder};
use serde_json::{Value, json};

use crate::{Args, assess, batch::Res};

struct SyntheticEmbedder;

#[async_trait::async_trait]
impl Embedder for SyntheticEmbedder {
    fn id(&self) -> &str {
        "synthetic/city-selection-v1"
    }
    fn dim(&self) -> usize {
        2
    }
    async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                if matches!(kind, EmbedKind::Query) {
                    vec![1.0, 0.0]
                } else if text.contains("Lisbon") {
                    vec![0.85, -0.5267827]
                } else {
                    vec![0.9, 0.4358899]
                }
            })
            .collect())
    }
}

async fn populate(store: &Store, at: i64) -> Res<()> {
    let ing = Ingestor::new(store, &SyntheticEmbedder, IngestConfig::default());
    let statements = ["Ada visited Oslo.", "Ada took a sightseeing trip to Oslo.", "Ada visited Lisbon."];
    ing.append_session(SessionInput {
        ext_id: "S1".into(),
        started_at: at,
        turns: statements
            .iter()
            .enumerate()
            .map(|(i, text)| TurnInput {
                ext_id: format!("S1:{}", i + 1),
                speaker: "Ada".into(),
                role: Role::Participant,
                text: (*text).into(),
            })
            .collect(),
    })
    .await?;
    ing.apply_extraction(
        "extract:S1:0-2",
        &json!({
            "entities":[{"name":"Ada","kind":"person","aliases":[]}],
            "facts":statements.iter().enumerate().map(|(i,text)|json!({
                "subject":"Ada","predicate":if i==1 {"took a sightseeing trip to"}else{"visited"},
                "object":if i==2 {"Lisbon"}else{"Oslo"},"object_is_entity":false,
                "statement":text,"negated":false,"salience":8,
                "time":{"expression":null,"kind":"instant"},"evidence_turns":[format!("S1:{}",i+1)]
            })).collect::<Vec<_>>()
        }),
    )
    .await?;
    Ok(())
}

pub async fn run(a: &Args) -> Res<bool> {
    let out = std::path::PathBuf::from(a.require("out")?);
    if out.exists() {
        return Err(format!("{} already exists; use a new --out directory", out.display()).into());
    }
    let judge = if a.get("engine").is_some() { Some(assess::judge(a)?) } else { None };
    fs::create_dir_all(&out)?;
    let store = Store::open(out.join("space.licorice"), StoreConfig::new(2)).await?;
    let at = Day::from_ymd(2023, 5, 8).unwrap().start_ms();
    populate(&store, at).await?;
    if let Some((judge, _)) = &judge {
        assess::permission(&store, judge.as_ref(), a).await?;
        assess::warm_up(judge.as_ref()).await?;
    }
    let assessor =
        judge.as_ref().map(|(judge, cfg)| EvidenceAssessor::new(&store, judge.as_ref(), cfg.clone())).transpose()?;
    let cfg = RetrieveConfig {
        chunks: 0,
        summaries: 0,
        w_session: 0.0,
        w_fact: 1.0,
        alpha: 1.0,
        recency_weight: 0.0,
        ..RetrieveConfig::default()
    };
    let list = Query { text: "Which two cities did Ada visit?".into(), at };
    // Choose one reproducible budget that fits either pair but cannot fit all three.
    // Preliminary two-item runs are local only. The compared runs below are governed
    // by this common token budget, with the normal twenty-fact cap.
    let mut budget = 0;
    for selection in [SelectionMode::Relevance, SelectionMode::Mmr] {
        let r = Retriever::new(&store, &SyntheticEmbedder, RetrieveConfig { selection, final_facts: 2, ..cfg.clone() })
            .retrieve(&list)
            .await?;
        budget = budget.max(r.context_usage.tokens);
    }
    let mut results = Vec::new();
    let mut success = true;
    for (case, query, budget_tokens) in [
        ("city_list", list, budget),
        ("occurrence_count", Query { text: "How many visits did Ada make?".into(), at }, 200),
    ] {
        for selection in [SelectionMode::Relevance, SelectionMode::Mmr] {
            let config = RetrieveConfig { selection, budget_tokens, ..cfg.clone() };
            let r = Retriever::new(&store, &SyntheticEmbedder, config.clone()).retrieve(&query).await?;
            let assessment = if let Some(assessor) = &assessor {
                let assessment = assessor.assess(&query.text, &r.context).await?;
                success &= assessment.status == AssessmentStatus::Judged;
                Some(assessment)
            } else {
                None
            };
            // Synthetic truth and coverage are attached AFTER assessment. Neither
            // truth nor metrics enter Jev's question/memory state.
            let cities: [&str; 2] = ["Oslo", "Lisbon"];
            let coverage = cities.iter().filter(|city| r.facts.iter().any(|f| f.statement.contains(**city))).count();
            results.push(json!({"case":case,"question":query.text,"config":config,
                "context":r.context,"context_usage":r.context_usage,"selection":r.selection,
                "budget_omissions":r.budget_omissions,"fact_ids":r.facts.iter().map(|f|f.id).collect::<Vec<_>>(),
                "cited_turns":r.cited_turns,"synthetic_city_coverage":coverage as f64/2.0,"assessment":assessment}));
            fs::write(out.join("results.json"), serde_json::to_string_pretty(&results)?)?;
        }
    }
    let audits: Vec<Value> = store
        .read(|conn| {
            let mut stmt = conn.prepare("SELECT request,answers FROM judgments ORDER BY id")?;
            let raw = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
                .collect::<Result<Vec<_>, _>>()?;
            raw.into_iter().map(|(request,answers)|Ok(json!({
            "request":serde_json::from_str::<Value>(&request).map_err(|e|licorice::Error::Task(e.to_string()))?,
            "response":serde_json::from_str::<Value>(&answers).map_err(|e|licorice::Error::Task(e.to_string()))?,
        }))).collect::<licorice::Result<Vec<_>>>()
        })
        .await?;
    fs::write(out.join("judgments.json"), serde_json::to_string_pretty(&audits)?)?;
    store.close().await?;
    let judged = results.iter().filter(|r| r["assessment"]["status"] == "judged").count();
    println!(
        "{}",
        json!({"results":out.join("results.json"),"conditions":results.len(),"attempted_judgments":audits.len(),"judged":judged,"success":success})
    );
    Ok(success)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn joined_ingestion_selection_and_budget_control() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("probe");
        let args = Args::parse(["--out".to_owned(), out.to_string_lossy().into_owned()].into_iter());
        assert!(run(&args).await.unwrap());
        let rows: Vec<Value> = serde_json::from_str(&fs::read_to_string(out.join("results.json")).unwrap()).unwrap();
        assert_eq!(rows.len(), 4);
        for row in &rows {
            assert!(
                row["context_usage"]["tokens"].as_u64().unwrap() <= row["config"]["budget_tokens"].as_u64().unwrap()
            );
            assert!(row["assessment"].is_null());
        }
        assert_eq!(rows[0]["config"]["budget_tokens"], rows[1]["config"]["budget_tokens"]);
        assert_eq!(rows[0]["fact_ids"], json!([1, 2]));
        assert_eq!(rows[1]["fact_ids"], json!([1, 3]));
        assert_eq!(rows[0]["synthetic_city_coverage"], 0.5);
        assert_eq!(rows[1]["synthetic_city_coverage"], 1.0);
        assert_eq!(rows[2]["fact_ids"], rows[3]["fact_ids"]);
        assert_eq!(rows[3]["fact_ids"].as_array().unwrap().len(), 3);
        assert_eq!(rows[3]["selection"]["effective"], "relevance");
        assert!(run(&args).await.is_err(), "preserve prior experiments");
    }
}
