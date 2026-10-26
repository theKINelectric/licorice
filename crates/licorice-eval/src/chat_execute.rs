//! OpenRouter executor for controlled comparisons; shares the new durable ledger.
use crate::batch::Res;
use crate::{Args, assess, batch};
use licorice_providers::chat::{ChatModel, ModelResponse};
use licorice_providers::chat_http::BudgetChat;
use serde_json::json;
use std::fs;
use std::sync::Arc;
pub async fn run(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let task = a.require("task")?;
    if !matches!(task, "answer" | "extract" | "summarize" | "grade") {
        return Err("chat-execute supports answer/extract/summarize/grade".into());
    }
    let key = assess::required_credential(a, "OPENROUTER_API_KEY")?;
    let model =
        Arc::new(BudgetChat::open(key, std::path::Path::new(a.require("ledger")?), a.require("budget-usd")?.parse()?)?);
    let requests = batch::read_requests(&run, task)?;
    let concurrency = a.num("concurrency", 8)?.clamp(1, 8);
    let path = run.join(format!("responses/{task}-live.jsonl"));
    fs::create_dir_all(path.parent().unwrap())?;
    let mut pending = tokio::task::JoinSet::new();
    let mut outputs = std::collections::BTreeMap::new();
    let mut failure = None;
    let total = requests.len();
    for (index, request) in requests.into_iter().enumerate() {
        let model = model.clone();
        pending.spawn(async move {
            let output = model.complete(&request).await;
            (index, request.id, output)
        });
        if pending.len() >= concurrency {
            let (index, id, output) = pending.join_next().await.unwrap()?;
            match output {
                Ok(output) => {
                    outputs.insert(index, ModelResponse { id, output });
                }
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
            write(&path, &outputs)?;
            if outputs.len() % 25 == 0 {
                println!("{}", json!({"complete":outputs.len(),"total":total,"accounted_usd":model_accounted(a)?}));
            }
        }
    }
    while let Some(done) = pending.join_next().await {
        let (index, id, output) = done?;
        match output {
            Ok(output) => {
                outputs.insert(index, ModelResponse { id, output });
            }
            Err(e) => {
                if failure.is_none() {
                    failure = Some(e);
                }
            }
        }
        write(&path, &outputs)?;
    }
    let complete = failure.is_none() && outputs.len() == total;
    fs::write(
        run.join(format!("chat-execute-{task}.json")),
        serde_json::to_string_pretty(
            &json!({"complete":complete,"model":model.id(),"expected":total,"completed":outputs.len(),"accounted_usd":model.accounted()?}),
        )?,
    )?;
    if let Some(error) = failure {
        return Err(error.into());
    }
    Ok(complete)
}
fn model_accounted(a: &Args) -> Res<f64> {
    let conn = licorice::rusqlite::Connection::open(a.require("ledger")?)?;
    Ok(conn.query_row("SELECT coalesce(sum(coalesce(cost,reservation)),0) FROM calls", [], |r| r.get(0))?)
}
fn write(path: &std::path::Path, outputs: &std::collections::BTreeMap<usize, ModelResponse>) -> Res<()> {
    let text = outputs.values().map(serde_json::to_string).collect::<Result<Vec<_>, _>>()?.join("\n") + "\n";
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, text)?;
    fs::rename(tmp, path)?;
    Ok(())
}
