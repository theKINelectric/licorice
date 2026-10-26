//! Live execution of the existing replay files. A durable ledger reserves the full
//! call allowance before transmission, retains unknown costs, and caches exact inputs.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use licorice::rusqlite::{self, Connection, OptionalExtension, params};
use licorice_providers::chat::{ModelRequest, ModelResponse, validate};
use serde_json::{Value, json};

use crate::{Args, assess, batch, batch::Res};

const PRICE_SOURCE: &str = "https://platform.claude.com/docs/en/about-claude/pricing";

fn rates(model: &str) -> Res<(f64, f64)> {
    match model {
        "claude-sonnet-4-6" => Ok((3.0, 15.0)),
        "claude-haiku-4-5-20251001" => Ok((1.0, 5.0)),
        _ => Err("unqualified execution model; choose claude-sonnet-4-6 or claude-haiku-4-5-20251001".into()),
    }
}

/// Anthropic accepts the structural schema, with numerical/length constraints
/// retained as descriptions; validation below still enforces the original schema.
fn provider_schema(schema: &Value) -> Value {
    let mut value = schema.clone();
    match &mut value {
        Value::Object(map) => {
            if map.contains_key("type") {
                let mut constraints = Vec::new();
                for key in ["minimum", "maximum", "maxItems"] {
                    if let Some(v) = map.remove(key) {
                        constraints.push(format!("{key}: {v}"));
                    }
                }
                if !constraints.is_empty() {
                    let previous = map.get("description").and_then(Value::as_str).unwrap_or("");
                    map.insert("description".into(), json!(format!("{previous} {}", constraints.join(", ")).trim()));
                }
            }
            for v in map.values_mut() {
                *v = provider_schema(v);
            }
        }
        Value::Array(a) => {
            for v in a {
                *v = provider_schema(v);
            }
        }
        _ => {}
    }
    value
}

fn body(request: &ModelRequest, model: &str) -> Value {
    let mut body = json!({"model":model,"max_tokens":request.max_tokens,"thinking":{"type":"disabled"},
        "system":request.system,"messages":[{"role":"user","content":request.user}]});
    if let Some(schema) = &request.schema {
        body["output_config"] = json!({"format":{"type":"json_schema","schema":provider_schema(schema)}});
    }
    body
}

fn temperature(value: Option<&str>) -> Res<Option<f64>> {
    value
        .map(|s| {
            s.parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                .ok_or_else(|| "--temperature must be finite and in 0..=1".into())
        })
        .transpose()
}

fn ledger(path: &Path, limit: f64) -> Res<Connection> {
    if !limit.is_finite() || limit <= 0.0 {
        return Err("--budget-usd must be finite and positive".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1),ceiling REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY,hash TEXT NOT NULL,request TEXT NOT NULL,
            reservation REAL NOT NULL,cost REAL,raw TEXT,output TEXT,error TEXT,created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS calls_hash ON calls(hash);")?;
    let old: Option<f64> = conn.query_row("SELECT ceiling FROM budget WHERE id=1", [], |r| r.get(0)).optional()?;
    // A lower per-command limit may stage the shared, cumulative budget. Raising
    // the declared limit is explicit in the command and remains in the receipt.
    if old.is_none() {
        conn.execute("INSERT INTO budget(id,ceiling) VALUES(1,?1)", [limit])?;
    }
    Ok(conn)
}

fn charged(conn: &Connection) -> Res<f64> {
    Ok(conn.query_row("SELECT coalesce(sum(coalesce(cost,reservation)),0) FROM calls", [], |r| r.get(0))?)
}

fn reserve(conn: &mut Connection, hash: &str, request: &Value, reservation: f64, limit: f64) -> Res<i64> {
    if !reservation.is_finite() || reservation < 0.0 {
        return Err("invalid reservation".into());
    }
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    if charged(&tx)? + reservation > limit {
        return Err(format!("spend limit: next reservation {reservation:.6} would exceed USD {limit:.2}").into());
    }
    tx.execute(
        "INSERT INTO calls(hash,request,reservation) VALUES(?1,?2,?3)",
        params![hash, request.to_string(), reservation],
    )?;
    let id = tx.last_insert_rowid();
    tx.commit()?;
    Ok(id)
}

fn cached(conn: &Connection, hash: &str) -> Res<Option<Value>> {
    let output: Option<String> = conn
        .query_row(
            "SELECT output FROM calls WHERE hash=?1 AND output IS NOT NULL ORDER BY id DESC LIMIT 1",
            [hash],
            |r| r.get(0),
        )
        .optional()?;
    output.map(|s| serde_json::from_str(&s).map_err(Into::into)).transpose()
}

fn parsed(request: &ModelRequest, raw: &Value) -> Res<Value> {
    if raw["stop_reason"] != "end_turn" {
        return Err(format!("generation did not finish: {}", raw["stop_reason"]).into());
    }
    let text = raw["content"]
        .as_array()
        .ok_or("missing content")?
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<String>();
    if let Some(schema) = &request.schema {
        let value: Value = serde_json::from_str(&text)?;
        validate(schema, &value).map_err(|e| format!("invalid output: {}", e.join("; ")))?;
        Ok(value)
    } else if text.trim().is_empty() {
        Err("empty model answer".into())
    } else {
        Ok(json!(text))
    }
}

fn cost(raw: &Value, input: f64, output: f64) -> Option<f64> {
    let usage = &raw["usage"];
    let i = usage["input_tokens"].as_u64()? as f64;
    let o = usage["output_tokens"].as_u64()? as f64;
    let writes = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0) as f64;
    let reads = usage["cache_read_input_tokens"].as_u64().unwrap_or(0) as f64;
    Some((i * input + o * output + writes * input * 1.25 + reads * input * 0.1) / 1e6)
}

async fn post(client: &reqwest::Client, url: &str, key: &str, body: &Value) -> Res<Value> {
    for attempt in 0..4 {
        let response = client
            .post(url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(body)
            .send()
            .await?;
        if response.status().as_u16() == 429 && attempt < 3 {
            let seconds = response
                .headers()
                .get("retry-after")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|s| s.is_finite())
                .unwrap_or(20.0)
                .clamp(0.01, 30.0);
            eprintln!("rate limited; retry {} after {seconds:.2} seconds (same allowance reserved)", attempt + 1);
            tokio::time::sleep(Duration::from_secs_f64(seconds)).await;
            continue;
        }
        if !response.status().is_success() {
            return Err(format!("Anthropic returned HTTP {}", response.status()).into());
        }
        return Ok(response.json().await?);
    }
    unreachable!("four bounded attempts return a response or error")
}

fn write_outputs(path: &Path, responses: &[ModelResponse]) -> Res<()> {
    let mut text = String::new();
    for response in responses {
        text.push_str(&serde_json::to_string(response)?);
        text.push('\n');
    }
    let temp = path.with_extension("jsonl.tmp");
    fs::write(&temp, text)?;
    fs::rename(temp, path)?;
    Ok(())
}

struct Completed {
    index: usize,
    id: i64,
    request: ModelRequest,
    reservation: f64,
    raw: Result<Value, String>,
    ms: u64,
}

fn finish(
    conn: &Connection,
    done: Completed,
    input_rate: f64,
    output_rate: f64,
    outputs: &mut [Option<ModelResponse>],
) -> Res<()> {
    let raw = match done.raw {
        Ok(raw) => raw,
        Err(error) => {
            let known_rejection =
                ["HTTP 400", "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 429"].iter().any(|s| error.contains(s));
            conn.execute(
                "UPDATE calls SET error=?1,cost=?2 WHERE id=?3",
                params![error, if known_rejection { Some(0.0) } else { None }, done.id],
            )?;
            return Err(error.into());
        }
    };
    let actual = cost(&raw, input_rate, output_rate);
    let parsed = parsed(&done.request, &raw);
    conn.execute(
        "UPDATE calls SET raw=?1,cost=?2,output=?3,error=?4 WHERE id=?5",
        params![
            raw.to_string(),
            actual,
            parsed.as_ref().ok().map(Value::to_string),
            parsed.as_ref().err().map(ToString::to_string),
            done.id
        ],
    )?;
    println!(
        "{}",
        json!({"request":done.request.id,"model":raw["model"],"cost_usd":actual,"accounted_usd":charged(conn)?,"latency_ms":done.ms})
    );
    if actual.is_none() {
        return Err("missing usage; full reservation retained".into());
    }
    if actual.is_some_and(|c| c > done.reservation) {
        return Err("usage exceeded conservative reservation; stopping for budget review".into());
    }
    outputs[done.index] = Some(ModelResponse { id: done.request.id, output: parsed? });
    Ok(())
}

async fn drain(
    pending: &mut tokio::task::JoinSet<Completed>,
    conn: &Connection,
    input: f64,
    output: f64,
    outputs: &mut [Option<ModelResponse>],
    path: &Path,
) -> Res<usize> {
    let mut error = None;
    let mut completed = 0;
    while let Some(done) = pending.join_next().await {
        let result = done.map_err(|e| e.into()).and_then(|done| finish(conn, done, input, output, outputs));
        if let Err(e) = result {
            if error.is_none() {
                error = Some(e);
            }
        }
        completed += 1;
        write_outputs(path, &outputs.iter().flatten().cloned().collect::<Vec<_>>())?;
    }
    if let Some(error) = error { Err(error) } else { Ok(completed) }
}

pub async fn run(a: &Args) -> Res<bool> {
    let run = a.run_dir()?;
    let task = a.require("task")?;
    if !["extract", "summarize", "answer", "grade"].contains(&task) {
        return Err("unknown execution task".into());
    }
    let model = a.get("model").unwrap_or("claude-sonnet-4-6");
    let (input_rate, output_rate) = rates(model)?;
    let temperature = temperature(a.get("temperature"))?;
    let limit: f64 = a.require("budget-usd")?.parse()?;
    let ledger_path = PathBuf::from(a.require("ledger")?);
    let mut conn = ledger(&ledger_path, limit)?;
    // An OS-backed SQLite lock prevents two executors from issuing the same calls.
    // It is released automatically on exit/crash, unlike a PID-file lease.
    let mut lock = Connection::open(ledger_path.with_extension("execution-lock.sqlite"))?;
    let _exclusive = lock.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    let key = assess::required_credential(a, "ANTHROPIC_API_KEY")?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(90))
        .build()?;
    let requests = batch::read_requests(&run, task)?;
    fs::write(
        run.join(format!("execute-{task}.json")),
        serde_json::to_string_pretty(
            &json!({"model":model,"task":task,"expected":requests.len(),"complete":false,"budget_usd":limit,"ledger":ledger_path}),
        )?,
    )?;
    let take = a.num("limit", requests.len())?;
    let path = run.join("responses").join(format!("{task}-live.jsonl"));
    fs::create_dir_all(path.parent().unwrap())?;
    let requests: Vec<_> = requests.into_iter().take(take).collect();
    let expected = batch::read_requests(&run, task)?.len();
    let concurrency = a.num("concurrency", 2)?.clamp(1, 8);
    let mut pending = tokio::task::JoinSet::new();
    let mut outputs: Vec<Option<ModelResponse>> = vec![None; requests.len()];
    let (mut hits, mut calls) = (0, 0);
    for (index, request) in requests.iter().enumerate() {
        if pending.len() >= concurrency {
            if let Err(error) =
                finish(&conn, pending.join_next().await.unwrap()?, input_rate, output_rate, &mut outputs)
            {
                let _ = drain(&mut pending, &conn, input_rate, output_rate, &mut outputs, &path).await;
                return Err(error);
            }
            calls += 1;
            write_outputs(&path, &outputs.iter().flatten().cloned().collect::<Vec<_>>())?;
        }
        let mut body = body(request, model);
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        let identity = json!({"version":1,"task":task,"body":body,"input_rate":input_rate,"output_rate":output_rate});
        let hash = blake3::hash(identity.to_string().as_bytes()).to_hex().to_string();
        if let Some(value) = cached(&conn, &hash)? {
            hits += 1;
            outputs[index] = Some(ModelResponse { id: request.id.clone(), output: value });
            write_outputs(&path, &outputs.iter().flatten().cloned().collect::<Vec<_>>())?;
            continue;
        }
        let mut count_body = body.clone();
        count_body.as_object_mut().unwrap().remove("max_tokens");
        count_body.as_object_mut().unwrap().remove("temperature");
        let counted = match post(&client, "https://api.anthropic.com/v1/messages/count_tokens", &key, &count_body).await
        {
            Ok(counted) => counted,
            Err(error) => {
                let _ = drain(&mut pending, &conn, input_rate, output_rate, &mut outputs, &path).await;
                return Err(error);
            }
        };
        let tokens = counted["input_tokens"].as_u64().ok_or("token counter returned no input count")?;
        let reservation =
            ((tokens + 512) as f64 * input_rate * 1.25 + f64::from(request.max_tokens) * output_rate) / 1e6;
        let id = match reserve(&mut conn, &hash, &identity, reservation, limit) {
            Ok(id) => id,
            Err(error) => {
                let _ = drain(&mut pending, &conn, input_rate, output_rate, &mut outputs, &path).await;
                return Err(error);
            }
        };
        let (client, key, request) = (client.clone(), key.clone(), request.clone());
        pending.spawn(async move {
            let started = Instant::now();
            let raw =
                post(&client, "https://api.anthropic.com/v1/messages", &key, &body).await.map_err(|e| e.to_string());
            Completed { index, id, request, reservation, raw, ms: started.elapsed().as_millis() as u64 }
        });
    }
    calls += drain(&mut pending, &conn, input_rate, output_rate, &mut outputs, &path).await?;
    let responses: Vec<_> = outputs.into_iter().flatten().collect();
    let summary = json!({"model":model,"task":task,"requests":responses.len(),"expected":expected,"complete":responses.len()==expected,
        "calls":calls,"cache_hits":hits,"accounted_usd":charged(&conn)?,"budget_usd":limit,"ledger":ledger_path,"temperature":temperature,
        "input_usd_per_mtok":input_rate,"output_usd_per_mtok":output_rate,"price_source":PRICE_SOURCE});
    fs::write(run.join(format!("execute-{task}.json")), serde_json::to_string_pretty(&summary)?)?;
    println!("{summary}");
    Ok(responses.len() == expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temperature_override_is_optional_and_rejects_invalid_sampling_controls() {
        assert_eq!(temperature(None).unwrap(), None);
        assert_eq!(temperature(Some("0")).unwrap(), Some(0.0));
        for value in ["NaN", "inf", "-0.1", "1.1", "invalid"] {
            assert!(temperature(Some(value)).is_err());
        }
    }
    fn request() -> ModelRequest {
        ModelRequest {
            id: "a".into(),
            task: "grade".into(),
            system: "Return JSON".into(),
            user: "grade this".into(),
            schema: Some(
                json!({"type":"object","additionalProperties":false,"required":["n"],"properties":{"n":{"type":"integer","minimum":1,"maximum":10}}}),
            ),
            max_tokens: 200,
        }
    }
    #[test]
    fn unknown_charges_survive_restart_and_cannot_overrun_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spend.sqlite");
        let mut conn = ledger(&path, 1.0).unwrap();
        reserve(&mut conn, "h", &json!({"request":"synthetic"}), 0.75, 1.0).unwrap();
        drop(conn);
        let mut conn = ledger(&path, 1.0).unwrap();
        assert_eq!(charged(&conn).unwrap(), 0.75);
        assert!(reserve(&mut conn, "other", &json!({}), 0.3, 1.0).is_err());
        assert!(cached(&conn, "h").unwrap().is_none());
        conn.execute("UPDATE calls SET cost=0.2,output='{}'", []).unwrap();
        assert_eq!(charged(&conn).unwrap(), 0.2);
        assert_eq!(cached(&conn, "h").unwrap(), Some(json!({})));
        assert!(cached(&conn, "changed").unwrap().is_none());
    }
    #[test]
    fn schemas_are_relaxed_only_for_transport_and_outputs_are_still_validated() {
        let r = request();
        let sent = body(&r, "claude-sonnet-4-6");
        assert!(sent["output_config"]["format"]["schema"]["properties"]["n"]["minimum"].is_null());
        let raw = |n, stop| json!({"stop_reason":stop,"content":[{"type":"text","text":format!("{{\"n\":{n}}}")}]});
        assert!(parsed(&r, &raw(0, "end_turn")).is_err());
        assert!(parsed(&r, &raw(5, "max_tokens")).is_err());
        assert_eq!(parsed(&r, &raw(5, "end_turn")).unwrap(), json!({"n":5}));
        assert_eq!(r.schema.unwrap()["properties"]["n"]["minimum"], 1);
    }
    #[test]
    fn actual_cost_counts_cached_and_generated_tokens() {
        let raw = json!({"usage":{"input_tokens":100,"output_tokens":20,"cache_creation_input_tokens":40,"cache_read_input_tokens":50}});
        assert!((cost(&raw, 3.0, 15.0).unwrap() - 0.000765).abs() < 1e-12);
        assert!(cost(&json!({}), 3.0, 15.0).is_none());
    }
    #[tokio::test]
    async fn http_contract_sends_only_the_request_and_does_not_echo_error_bodies() {
        use axum::{
            Json, Router,
            http::{HeaderMap, StatusCode},
            routing::post as route,
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/ok",
                route(|headers: HeaderMap, Json(body): Json<Value>| async move {
                    assert_eq!(headers["x-api-key"], "synthetic-key");
                    assert_eq!(headers["anthropic-version"], "2023-06-01");
                    assert_eq!(body["model"], "claude-sonnet-4-6");
                    Json(json!({"input_tokens":123}))
                }),
            )
            .route(
                "/bad",
                route(|| async { (StatusCode::UNAUTHORIZED, "synthetic-key must never appear in an error") }),
            );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        assert_eq!(
            post(&client, &format!("http://{address}/ok"), "synthetic-key", &body(&request(), "claude-sonnet-4-6"))
                .await
                .unwrap()["input_tokens"],
            123
        );
        let error =
            post(&client, &format!("http://{address}/bad"), "synthetic-key", &json!({})).await.unwrap_err().to_string();
        assert!(error.contains("401"));
        assert!(!error.contains("synthetic-key"));
        server.abort();
    }

    #[test]
    fn committed_results_rehydrate_after_missing_output_and_failures_keep_unknown_charges() {
        let dir = tempfile::tempdir().unwrap();
        let mut conn = ledger(&dir.path().join("spend.sqlite"), 1.0).unwrap();
        let id = reserve(&mut conn, "exact-input", &json!({}), 0.1, 1.0).unwrap();
        let raw = json!({"model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[{"type":"text","text":"{\"n\":5}"}],"usage":{"input_tokens":100,"output_tokens":20}});
        let mut outputs = vec![None];
        finish(
            &conn,
            Completed { index: 0, id, request: request(), reservation: 0.1, raw: Ok(raw), ms: 1 },
            3.0,
            15.0,
            &mut outputs,
        )
        .unwrap();
        assert_eq!(cached(&conn, "exact-input").unwrap(), Some(json!({"n":5})));
        assert!(charged(&conn).unwrap() < 0.001);
        let id = reserve(&mut conn, "timeout", &json!({}), 0.1, 1.0).unwrap();
        assert!(
            finish(
                &conn,
                Completed { index: 0, id, request: request(), reservation: 0.1, raw: Err("timeout".into()), ms: 1 },
                3.0,
                15.0,
                &mut outputs
            )
            .is_err()
        );
        assert!(charged(&conn).unwrap() > 0.1);
        assert!(cached(&conn, "timeout").unwrap().is_none());
    }
}
