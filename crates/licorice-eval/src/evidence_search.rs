//! Annotation-free native retrieval for bounded model-generated subqueries.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    question: String,
    at: i64,
    queries: Vec<String>,
}

pub async fn run(a: &Args) -> Res<bool> {
    if a.switches.contains("source-aware") && a.switches.contains("cross-session") {
        return Err("choose only one of --source-aware and --cross-session".into());
    }
    let run = a.run_dir()?;
    let cfg = load_run(&run)?;
    let inputs: Vec<Input> = serde_json::from_str(&fs::read_to_string(a.require("input")?)?)?;
    let mut ids = HashSet::new();
    for row in &inputs {
        if row.id.is_empty()
            || !ids.insert(&row.id)
            || row.question.trim().is_empty()
            || row.question.chars().count() > 2000
            || row.queries.len() > 2
            || row.queries.iter().any(|q| q.trim().is_empty() || q.chars().count() > 240)
        {
            return Err("invalid bounded evidence-search input".into());
        }
    }
    let (store, embedder) = open(&run, &cfg).await?;
    // This command accepts an immutable as-of snapshot, not a future-containing
    // space: temporal eligibility must not be approximated by relevance scoring.
    let latest: Option<i64> =
        store.read(|c| Ok(c.query_row("SELECT MAX(started_at) FROM sessions", [], |r| r.get(0))?)).await?;
    if inputs.iter().any(|r| latest.is_some_and(|t| t > r.at)) {
        store.close().await?;
        return Err("evidence-search snapshot contains sessions after the question clock".into());
    }
    let primary = Retriever::new(&store, embedder.as_ref(), RetrieveConfig::default());
    let raw = Retriever::new(
        &store,
        embedder.as_ref(),
        RetrieveConfig {
            final_facts: 0,
            summaries: 0,
            chunks: 100,
            chunk_candidates: 100,
            ..RetrieveConfig::default()
        },
    );
    let mut rows = Vec::new();
    for row in inputs {
        let first = primary.retrieve(&Query { text: row.question.clone(), at: row.at }).await?;
        if a.switches.contains("source-aware") {
            let (merged, assessment) =
                licorice::retrieve::source_aware::recover(&store, &first, &row.question, row.at, &Cl100kCounter, 2000)
                    .await?;
            rows.push(json!({"id":row.id,"question":row.question,"at":row.at,
                "merged":merged,"assessment":assessment}));
            continue;
        }
        if a.switches.contains("cross-session") {
            let (merged, trace) =
                licorice::retrieve::cross_session::recover(&store, &first, &row.question, row.at, &Cl100kCounter, 2000)
                    .await?;
            rows.push(json!({"id":row.id,"question":row.question,"at":row.at,
                "merged":merged,"recovery":trace}));
            continue;
        }
        let mut followups = Vec::new();
        for text in &row.queries {
            followups.push(raw.retrieve(&Query { text: text.clone(), at: row.at }).await?);
        }
        let merged = licorice::retrieve::complementary::merge(
            &first,
            &followups,
            Day::from_ms(row.at),
            &Cl100kCounter,
            2000,
            1400,
        )?;
        rows.push(json!({"id":row.id,"question":row.question,"at":row.at,"queries":row.queries,
            "primary_context":first.context,"primary_cited_turns":first.cited_turns,
            "followup_cited_turns":followups.iter().map(|r| &r.cited_turns).collect::<Vec<_>>(),
            "merged":merged}));
    }
    store.close().await?;
    fs::write(a.require("out")?, serde_json::to_string_pretty(&rows)?)?;
    Ok(true)
}
