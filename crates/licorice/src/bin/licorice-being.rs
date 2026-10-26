//! Native local BEING transport. Actor metadata comes from explicit transport
//! flags, never from an event body. Local filesystem access is the trust boundary.
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::path::PathBuf;

use licorice::being::PositionRef;
use licorice::being::context::{ContextConfig, continuity};
use licorice::being::ontology::{ReviewLimits, changed_dependencies, impact};
use licorice::being::reflection::ReflectionInput;
use licorice::being::{Actor, ActorRole, Memory, Request};
use licorice::retrieve::Query;
use licorice::{Store, StoreConfig};
use licorice_providers::chat::ModelResponse;
use licorice_providers::embed::WordLlama;
use licorice_providers::tokens::Cl100kCounter;
use serde::Deserialize;
use serde_json::{Value, json};

const USAGE: &str = "licorice-being init|apply|snapshot|context|history|impact|reflect-plan|reflect-complete|import-sidecar --store ABS --agent ID [--embed-dim 256]\napply/import/reflect commands require --principal ID --role agent|human|harness|importer --session ID\nsnapshot/context/impact accept --relationship ID; context optionally uses --models ABS for local evidence\nimport-sidecar requires --source ABS --import-id ID; apply/context/impact/reflect commands read bounded typed JSON on stdin.";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextInput {
    query: String,
    at: Option<i64>,
    budget_tokens: Option<usize>,
    continuity_tokens: Option<usize>,
    #[serde(default)]
    include_episodes: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImpactInput {
    roots: Option<Vec<PositionRef>>,
    limits: Option<ReviewLimits>,
}

fn args() -> Result<(String, BTreeMap<String, String>), String> {
    let mut raw = std::env::args().skip(1);
    let command = raw.next().ok_or(USAGE)?;
    let mut flags = BTreeMap::new();
    while let Some(key) = raw.next() {
        let name = key.strip_prefix("--").ok_or("expected an explicit named flag")?;
        if !matches!(
            name,
            "store"
                | "agent"
                | "embed-dim"
                | "principal"
                | "role"
                | "session"
                | "relationship"
                | "source"
                | "import-id"
                | "models"
        ) {
            return Err("unknown transport flag".into());
        }
        let value = raw.next().ok_or("flag is missing its value")?;
        if flags.insert(name.to_owned(), value).is_some() {
            return Err("duplicate transport flag".into());
        }
    }
    Ok((command, flags))
}
fn required<'a>(flags: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    flags.get(key).map(String::as_str).ok_or_else(|| format!("missing --{key}"))
}
fn actor(flags: &BTreeMap<String, String>) -> Result<Actor, String> {
    let role = match required(flags, "role")? {
        "agent" => ActorRole::Agent,
        "human" => ActorRole::Human,
        "harness" => ActorRole::Harness,
        "importer" => ActorRole::Importer,
        _ => return Err("unknown actor role".into()),
    };
    Ok(Actor { principal: required(flags, "principal")?.into(), role, session: required(flags, "session")?.into() })
}

fn check_existing_store(path: &std::path::Path) -> Result<(), String> {
    if !path.exists() || path.metadata().map_err(|_| "cannot inspect store")?.len() == 0 {
        return Ok(());
    }
    let conn = rusqlite::Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|_| "not a native Licorice store")?;
    let names: i64 = conn
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN ('meta','sessions','turns','facts')",
            [],
            |r| r.get(0),
        )
        .map_err(|_| "cannot inspect native schema")?;
    if names != 4 {
        return Err("foreign/prototype database refused; use explicit sidecar import into a native store".into());
    }
    let valid: bool = conn
        .query_row("SELECT EXISTS(SELECT 1 FROM meta WHERE key='embed_dim')", [], |r| r.get(0))
        .map_err(|_| "invalid native metadata")?;
    if !valid {
        return Err("uninitialized or foreign native metadata".into());
    }
    Ok(())
}

async fn run() -> Result<Value, String> {
    let (command, flags) = args()?;
    if !matches!(
        command.as_str(),
        "init"
            | "apply"
            | "snapshot"
            | "context"
            | "history"
            | "import-sidecar"
            | "impact"
            | "reflect-plan"
            | "reflect-complete"
    ) {
        return Err(USAGE.into());
    }
    let path = PathBuf::from(required(&flags, "store")?);
    let agent = required(&flags, "agent")?;
    if !path.is_absolute() || (command != "init" && !path.is_file()) {
        return Err("use an absolute initialized native store".into());
    }
    let dim = flags
        .get("embed-dim")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid embedding dimension")?
        .unwrap_or(256);
    if !(1..=65536).contains(&dim) {
        return Err("invalid embedding dimension".into());
    }
    let body = if matches!(command.as_str(), "apply" | "context" | "impact" | "reflect-plan" | "reflect-complete") {
        let mut body = String::new();
        io::stdin().take(65537).read_to_string(&mut body).map_err(|_| "invalid request input")?;
        if body.len() > 65536 {
            return Err("request exceeds native admission limit".into());
        }
        body
    } else {
        String::new()
    };
    let input = if command == "apply" {
        Some(serde_json::from_str::<Request>(&body).map_err(|_| "invalid typed native request")?)
    } else {
        None
    };
    let context_input = if command == "context" {
        let input = if body.trim().is_empty() {
            ContextInput {
                query: "current commitments and shared work".into(),
                at: None,
                budget_tokens: None,
                continuity_tokens: None,
                include_episodes: false,
            }
        } else {
            serde_json::from_str::<ContextInput>(&body).map_err(|_| "invalid typed context request")?
        };
        let budget = input.budget_tokens.unwrap_or(2000);
        if !(64..=2000).contains(&budget)
            || input.query.trim().is_empty()
            || input.query.chars().count() > 2000
            || input.continuity_tokens.is_some_and(|n| n > budget)
        {
            return Err("context request must have a bounded query and a single64..2000-token allowance".into());
        }
        Some(input)
    } else {
        None
    };
    // Validate required actor flags before opening or migrating the target.
    let author = if matches!(command.as_str(), "apply" | "import-sidecar" | "reflect-plan" | "reflect-complete") {
        Some(actor(&flags)?)
    } else {
        None
    };
    if command == "init" {
        let parent = path.parent().ok_or("store has no parent")?;
        std::fs::create_dir_all(parent).map_err(|_| "cannot create store parent")?;
        let mut create = std::fs::OpenOptions::new();
        create.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            create.mode(0o600);
        }
        match create.open(&path) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err("cannot create native store".into()),
        }
    }
    check_existing_store(&path)?;
    let store = Store::open(&path, StoreConfig::new(dim)).await.map_err(|e| e.to_string())?;
    let result = async {
        let mem = if command == "init" {
            Memory::initialize(&store, agent).await
        } else {
            Memory::attach(&store, agent).await
        }
        .map_err(|e| e.to_string())?;
        match command.as_str() {
            "init" => {
                let schema: i64 = store
                    .read(|conn| Ok(conn.pragma_query_value(None, "user_version", |r| r.get(0))?))
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({"agent_id":agent,"schema":schema}))
            }
            "reflect-plan"=>serde_json::to_value(mem.plan_reflection(author.expect("actor flags checked"),serde_json::from_str::<ReflectionInput>(&body).map_err(|_|"invalid reflection input")?).await.map_err(|e|e.to_string())?).map_err(|e|e.to_string()),
            "reflect-complete"=>serde_json::to_value(mem.complete_reflection(author.expect("actor flags checked"),serde_json::from_str::<ModelResponse>(&body).map_err(|_|"invalid reflection response")?).await.map_err(|e|e.to_string())?).map_err(|e|e.to_string()),
            "apply" => serde_json::to_value(
                mem.apply(author.expect("actor flags checked"), input.expect("request checked"))
                    .await
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string()),
            "snapshot" => serde_json::to_value(
                mem.snapshot(flags.get("relationship").map(String::as_str)).await.map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string()),
            "impact"=>{
                let input=if body.trim().is_empty(){ImpactInput{roots:None,limits:None}}else{serde_json::from_str::<ImpactInput>(&body).map_err(|_|"invalid impact request")?};
                let s=mem.snapshot(flags.get("relationship").map(String::as_str)).await.map_err(|e|e.to_string())?;
                let roots=input.roots.unwrap_or_else(||changed_dependencies(&s));
                if roots.is_empty(){Ok(json!({"roots":[],"affected":[],"saturated":false,"conclusion":"No changed dependency in this selected snapshot."}))}
                else{serde_json::to_value(impact(&s,&roots,input.limits.unwrap_or_default()).map_err(|e|e.to_string())?).map_err(|e|e.to_string())}
            }
            "context" => {
                let input=context_input.expect("context request checked");
                let budget=input.budget_tokens.unwrap_or(2000);
                let relationship=flags.get("relationship").map(String::as_str);
                if let Some(models)=flags.get("models") {
                    if !PathBuf::from(models).is_absolute() { return Err("models path must be absolute".into()); }
                    let embedder=WordLlama::load(&PathBuf::from(models)).map_err(|e|e.to_string())?;
                    let at=input.at.unwrap_or_else(||std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64);
                    let config=ContextConfig{retrieval:licorice::retrieve::RetrieveConfig{budget_tokens:budget,..Default::default()},continuity_tokens:input.continuity_tokens.unwrap_or(400).min(budget),source_aware:true,include_episodes:input.include_episodes};
                    let joined=mem.retrieve_context(relationship,&Query{text:input.query,at},&embedder,&Cl100kCounter,config).await.map_err(|e|e.to_string())?;
                    Ok(json!({"text":joined.text,"usage":joined.usage,"continuity":joined.continuity,"cited_turns":joined.cited_turns,"source_selection":joined.source_selection,"evidence_mode":"source_aware"}))
                }else{
                    let (snapshot,episode_search)=if input.include_episodes {let (s,r)=mem.snapshot_with_episodes(relationship).await.map_err(|e|e.to_string())?;(s,Some(r))}else{(mem.snapshot(relationship).await.map_err(|e|e.to_string())?,None)};
                    let mut memory=continuity(&snapshot,&input.query,&Cl100kCounter,budget).map_err(|e|e.to_string())?;
                    memory.episode_search=episode_search;
                    Ok(json!({"text":memory.text,"usage":memory.usage,"continuity":memory,"cited_turns":[],"evidence_mode":"continuity_only"}))
                }
            },
            "history" => {
                serde_json::to_value(mem.history().await.map_err(|e| e.to_string())?).map_err(|e| e.to_string())
            }
            "import-sidecar" => {
                let source = PathBuf::from(required(&flags, "source")?);
                if !source.is_absolute() {
                    return Err("sidecar source path must be absolute".into());
                }
                serde_json::to_value(
                    mem.import_sidecar(author.expect("actor flags checked"), required(&flags, "import-id")?, source)
                        .await
                        .map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())
            }
            _ => unreachable!(),
        }
    }
    .await;
    store.close().await.map_err(|e| e.to_string())?;
    result
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    match run().await {
        Ok(result) => println!("{}", json!({"ok":true,"result":result})),
        Err(error) => {
            println!("{}", json!({"ok":false,"error":error}));
            std::process::exit(1);
        }
    }
}
