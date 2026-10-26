//! Copy-on-write publication keeps failed remote phases invisible to Search.
use super::*;
use licorice::ingest::{IngestConfig, Ingestor};
use licorice_providers::chat::{ModelRequest, validate};
use serde_json::Value;

#[derive(Debug)]
enum ModelFailure {
    OutputLimit,
    Backend,
}
impl From<ApiError> for ModelFailure {
    fn from(_: ApiError) -> Self {
        Self::Backend
    }
}
impl From<ModelFailure> for ApiError {
    fn from(_: ModelFailure) -> Self {
        Self::unavailable()
    }
}

async fn reject_phase(store: &Store, id: &str, output: &Value, reason: &str) -> ApiResult<()> {
    let (id, output, reason) = (id.to_owned(), output.to_string(), reason.to_owned());
    store
        .write(move |w| {
            w.conn().execute(
                "INSERT OR IGNORE INTO aml_rejections(request_id,output,reason) VALUES(?1,?2,?3)",
                params![id, output, reason],
            )?;
            Ok(())
        })
        .await
        .map_err(|_| ApiError::unavailable())
}

const SOURCE_BINDING: &str = "Source binding contract: evidence_turns and time.anchor_turn must use the current turn ID allowed by the schema. For an update, output evidence_source with that current turn ID. Native code copies the complete immutable source message into evidence_quote; do not generate evidence_quote or copy a phrase from an instruction example. The earlier literal-quote format applies only to native application after this conversion.";

fn bind_request_source(request: &mut ModelRequest, id: &str) {
    let fact = &mut request.schema.as_mut().expect("native extraction schema")["properties"]["facts"]["items"];
    fact["properties"]["time"]["properties"]["anchor_turn"]["enum"] = json!([id]);
    fact["properties"]["evidence_turns"]["items"]["enum"] = json!([id]);
    let update = &mut fact["properties"]["update"]["anyOf"][1];
    update["properties"].as_object_mut().unwrap().remove("evidence_quote");
    update["properties"]["evidence_source"] = json!({"type":"string","enum":[id]});
    for field in update["required"].as_array_mut().unwrap() {
        if *field == "evidence_quote" {
            *field = json!("evidence_source");
        }
    }
    request.user.push_str(&format!("\n{SOURCE_BINDING}\nCurrent source ID: {id}\n"));
}

// Identity guards must agree with the alias rules used by native storage.
// Otherwise punctuation/articles can bypass the actor-kind check and create
// a concept and person with the same stored alias.
fn canonical_actor(name: &str) -> Option<&'static str> {
    match licorice::ingest::normalize(name).as_str() {
        "user" => Some("User"),
        "assistant" => Some("Assistant"),
        _ => None,
    }
}

fn bind_output_source(mut output: Value, id: &str, text: &str, role: MessageRole) -> Result<Value, licorice::Error> {
    if matches!(role, MessageRole::User | MessageRole::Assistant) {
        let mut actors = std::collections::BTreeSet::new();
        for fact in output["facts"].as_array().into_iter().flatten() {
            for name in fact["subject"].as_str().into_iter().chain(if fact["object_is_entity"] == true {
                fact["object"].as_str()
            } else {
                None
            }) {
                if let Some(actor) = canonical_actor(name) {
                    actors.insert(actor);
                }
            }
        }
        let entities = output["entities"].as_array_mut().expect("validated entity list");
        for actor in actors {
            if !entities.iter().any(|e| e["name"].as_str().is_some_and(|n| canonical_actor(n) == Some(actor))) {
                // These are declared namespace actors, not inferred third parties.
                // Core's generic fallback kind must not create a second actor.
                entities.push(json!({"name":actor,"kind":"person","aliases":[]}));
            }
        }
    }
    for fact in output["facts"].as_array_mut().into_iter().flatten() {
        if let Some(update) = fact.get_mut("update").and_then(Value::as_object_mut) {
            if update.contains_key("evidence_quote")
                || update.get("evidence_source").and_then(Value::as_str) != Some(id)
            {
                return Err(licorice::Error::InvalidInput(format!(
                    "update evidence_source must select the current source ID {id}; native code supplies its verbatim quote"
                )));
            }
            update.remove("evidence_source");
            update.insert("evidence_quote".into(), json!(text));
        }
    }
    Ok(output)
}

impl Adapter {
    pub(super) fn profile(&self) -> String {
        match &self.0.model {
            Some(model) => format!("aml-textual/temporal-facts-v12-output-capacity:{}", model.id()),
            None => PROFILE.into(),
        }
    }
    pub(super) async fn model_add(&self, request: AddRequest) -> ApiResult<Value> {
        let main = self.path(&request.user_id);
        let pending = main.with_extension("pending.sqlite3");
        let payload = serde_json::to_string(&request).unwrap();
        if main.exists() {
            let store = self.store(&request.user_id).await?;
            let id = request.request_id.clone();
            let old: Option<String> = store
                .read(move |c| {
                    Ok(c.query_row("SELECT payload FROM aml_requests WHERE request_id=?1", [&id], |r| r.get(0))
                        .optional()?)
                })
                .await
                .map_err(|_| ApiError::unavailable())?;
            store.close().await.map_err(|_| ApiError::unavailable())?;
            if let Some(old) = old {
                if old != payload {
                    return Err(ApiError(
                        StatusCode::CONFLICT,
                        "request_id already exists with different content or session",
                    ));
                }
                return Ok(success(&request));
            }
        }
        if !pending.exists() && main.exists() {
            let source = main.clone();
            let target = pending.clone();
            tokio::task::spawn_blocking(move || -> licorice::rusqlite::Result<()> {
                let conn = Connection::open(source)?;
                conn.backup("main", target, None)
            })
            .await
            .map_err(|_| ApiError::unavailable())?
            .map_err(|_| ApiError::unavailable())?;
        }
        let store = self.store_at(pending.clone()).await?;
        let check = payload.clone();
        let prepared=store.write(move |w| {
            w.conn().execute_batch("CREATE TABLE IF NOT EXISTS aml_pending(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS aml_models(hash TEXT PRIMARY KEY,request TEXT NOT NULL,output TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS aml_rejections(request_id TEXT PRIMARY KEY,output TEXT NOT NULL,reason TEXT NOT NULL);")?;
            let old:Option<String>=w.conn().query_row("SELECT payload FROM aml_pending WHERE id=1",[],|r|r.get(0)).optional()?;
            if old.as_ref().is_some_and(|old|old!=&check){return Ok(false);}
            w.conn().execute("INSERT OR IGNORE INTO aml_pending(id,payload) VALUES(1,?1)",[check])?;Ok(true)
        }).await;
        if !matches!(prepared, Ok(true)) {
            let _ = store.close().await;
            return match prepared {
                Ok(false) => {
                    Err(ApiError(StatusCode::CONFLICT, "an unfinished Add must be retried with its original payload"))
                }
                _ => Err(ApiError::unavailable()),
            };
        }
        let work = async {
            self.append(&store, &request).await?;
            self.model_phases(&store, &request).await?;
            store
                .write(|w| {
                    w.conn().execute("DELETE FROM aml_pending", [])?;
                    Ok(())
                })
                .await
                .map_err(|_| ApiError::unavailable())?;
            Ok::<_, ApiError>(())
        }
        .await;
        let closed = store.close().await;
        work?;
        closed.map_err(|_| ApiError::unavailable())?;
        // Every native connection is closed. Checkpoint before atomic publication;
        // restart sees either the preceding version or the entire completed Add.
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let conn = Connection::open(&pending).map_err(|_| "open staging checkpoint")?;
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);").map_err(|_| "checkpoint staging")?;
            drop(conn);
            std::fs::rename(&pending, &main).map_err(|_| "publish completed Add")?;
            Ok(())
        })
        .await
        .map_err(|_| ApiError::unavailable())?
        .map_err(|_| ApiError::unavailable())?;
        Ok(success(&request))
    }
    async fn cached_model(&self, store: &Store, request: &ModelRequest) -> Result<Value, ModelFailure> {
        let encoded = serde_json::to_string(request).unwrap();
        let hash = blake3::hash(encoded.as_bytes()).to_hex().to_string();
        let lookup = hash.clone();
        let cached: Option<String> = store
            .read(move |c| {
                Ok(c.query_row("SELECT output FROM aml_models WHERE hash=?1", [&lookup], |r| r.get(0)).optional()?)
            })
            .await
            .map_err(|_| ApiError::unavailable())?;
        if let Some(cached) = cached {
            return serde_json::from_str(&cached).map_err(|_| ModelFailure::Backend);
        }
        let output = self
            .0
            .model
            .as_ref()
            .unwrap()
            .complete(request)
            .await
            .map_err(|e| if e.is_output_limit() { ModelFailure::OutputLimit } else { ModelFailure::Backend })?;
        if let Some(schema) = &request.schema {
            validate(schema, &output).map_err(|_| ApiError::unavailable())?;
        }
        let saved = output.to_string();
        store
            .write(move |w| {
                w.conn().execute(
                    "INSERT INTO aml_models(hash,request,output) VALUES(?1,?2,?3)",
                    params![hash, encoded, saved],
                )?;
                Ok(())
            })
            .await
            .map_err(|_| ApiError::unavailable())?;
        Ok(output)
    }
    async fn model_phases(&self, store: &Store, input: &AddRequest) -> ApiResult<()> {
        let session =
            blake3::hash(&serde_json::to_vec(&(&input.user_id, &input.request_id)).unwrap()).to_hex().to_string();
        let cfg = IngestConfig {
            window_turns: 1,
            context_turns: 0,
            chunk_turns: 1,
            merge_threshold: 1.,
            temporal_updates: true,
        };
        let ing = Ingestor::new(store, self.0.embedder.as_ref(), cfg);
        let requests = ing.extraction_requests(&session).await.map_err(|_| ApiError::unavailable())?;
        let mut prepared = Vec::new();
        for (index, mut request) in requests.into_iter().enumerate() {
            if applied(store, &request.id).await? {
                continue;
            }
            let source_session = session.clone();
            let external = input.session_id.clone();
            let (prefix,current_id)=store.read(move |c| {
                let (turn,current_id):(i64,String)=c.query_row("SELECT t.id,t.ext_id FROM turns t JOIN sessions s ON s.id=t.session_id WHERE s.ext_id=?1 AND t.seq=?2",params![source_session,index as i64],|r|Ok((r.get(0)?,r.get(1)?)))?;
                let mut s=c.prepare("SELECT t.ext_id,t.ts,t.speaker,t.text FROM turns t JOIN sessions s ON s.id=t.session_id WHERE json_extract(s.meta,'$.session_id')=?1 AND t.id<?2 ORDER BY t.id DESC LIMIT 4")?;
                let mut rows=s.query_map(params![external,turn],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,String>(2)?,r.get::<_,String>(3)?)))?.collect::<Result<Vec<_>,_>>()?;
                rows.reverse();Ok((rows.into_iter().map(|(id,ts,speaker,text)|format!("[{id}] [UTC date {}] {speaker}: {text}",Day::from_ms(ts).iso())).collect::<Vec<_>>().join("\n"),current_id))
            }).await.map_err(|_|ApiError::unavailable())?;
            if !prefix.is_empty() {
                request.user = format!(
                    "Earlier source turns (context only; do not extract these again):\n{prefix}\n\n{}",
                    request.user
                );
            }
            let role = input.messages[index].role;
            let context = if matches!(role, MessageRole::User | MessageRole::Assistant) {
                let identity = "Role identity instruction (not source evidence): user messages in this namespace belong to the same user; assistant messages belong to the assistant. Use the canonical names User and Assistant for their first-person self references, and User for an assistant's second-person reference to the user. If the user supplies their own name, add it as an alias of User. Keep explicitly named third parties distinct. Do not name a person merely 'the speaker', 'speaker', 'I', 'me', or 'myself'; resolve the source author or use a specific third-party description. Do not put pronouns or generic speaker labels in person aliases. If speaker means a physical device or other non-person entity, declare its entity kind explicitly.";
                request.user = format!("{identity}\n\n{}", request.user);
                format!("{identity}\n\n{prefix}")
            } else {
                prefix
            };
            bind_request_source(&mut request, &current_id);
            let context = format!("{context}\n{SOURCE_BINDING}");
            prepared.push((request, context, role, current_id, input.messages[index].content.clone()));
        }
        // Requests depend only on immutable raw context. Prefetch in bounded groups,
        // then apply in source order; drain every started call before returning.
        for group in prepared.chunks(8) {
            let mut pending = tokio::task::JoinSet::new();
            for (request, _, _, _, _) in group {
                let adapter = self.clone();
                let store = store.clone();
                let request = request.clone();
                pending.spawn(async move {
                    let result = adapter.cached_model(&store, &request).await.map(|_| ());
                    (request.id, result)
                });
            }
            let mut failed = false;
            let mut limited = std::collections::HashSet::new();
            while let Some(done) = pending.join_next().await {
                match done {
                    Ok((id, Err(ModelFailure::OutputLimit))) => {
                        limited.insert(id);
                    }
                    Ok((_, Ok(()))) => {}
                    _ => failed = true,
                }
            }
            if failed {
                return Err(ApiError::unavailable());
            }
            for (request, prefix, role, current_id, current_text) in group {
                let original = request.clone();
                let mut request = original.clone();
                for attempt in 0..3 {
                    let completion = if attempt == 0 && limited.contains(&original.id) {
                        Err(ModelFailure::OutputLimit)
                    } else {
                        self.cached_model(store, &request).await
                    };
                    let output = match completion {
                        Ok(output) => output,
                        Err(ModelFailure::OutputLimit) if attempt < 2 && request.max_tokens < 8192 => {
                            request.max_tokens = 8192;
                            request.id = format!("{}:capacity:8192", request.id);
                            continue;
                        }
                        Err(ModelFailure::OutputLimit) => {
                            reject_phase(
                                store,
                                &original.id,
                                &json!({"request_id":request.id,"max_tokens":request.max_tokens}),
                                "output capacity exhausted; raw source retained",
                            )
                            .await?;
                            ing.apply_extraction_with_context(&original.id, &json!({"entities":[],"facts":[]}), prefix)
                                .await
                                .map_err(|_| ApiError::unavailable())?;
                            break;
                        }
                        Err(ModelFailure::Backend) => return Err(ApiError::unavailable()),
                    };
                    let applied = match validate_role_identity(&output, *role)
                        .map_err(licorice::Error::InvalidInput)
                        .and_then(|()| bind_output_source(output.clone(), current_id, current_text, *role))
                    {
                        Ok(bound) => ing.apply_extraction_with_context(&original.id, &bound, prefix).await,
                        Err(reason) => Err(reason),
                    };
                    match applied {
                        Ok(_) => break,
                        Err(licorice::Error::InvalidInput(reason)) if attempt < 2 => {
                            let capacity = request.max_tokens;
                            request = original.clone();
                            request.max_tokens = capacity;
                            request.id = format!("{}:repair:{}", original.id, attempt + 1);
                            request.user.push_str(&format!("\nThe previous output failed source validation: {reason}. Fix it using only the supplied source and role identity instruction: time.anchor_turn must be a cited current turn; update.evidence_source must select the current ID allowed by the schema; effective_expression must copy its literal source date; replacements require an ongoing new state, while corrections retract an explicitly erroneous assertion. Previous output:\n{output}"));
                        }
                        Err(licorice::Error::InvalidInput(reason)) => {
                            // Retain unusable outputs for audit, preserve raw memory,
                            // and never publish an unsupported derived assertion.
                            let id = original.id.clone();
                            let rejected = output.to_string();
                            let reason = format!("source validation failed after two repairs: {reason}");
                            store.write(move |w|{w.conn().execute("INSERT OR REPLACE INTO aml_rejections(request_id,output,reason) VALUES(?1,?2,?3)",params![id,rejected,reason])?;Ok(())}).await.map_err(|_|ApiError::unavailable())?;
                            ing.apply_extraction_with_context(&original.id, &json!({"entities":[],"facts":[]}), prefix)
                                .await
                                .map_err(|_| ApiError::unavailable())?;
                            break;
                        }
                        Err(_) => return Err(ApiError::unavailable()),
                    }
                }
            }
        }
        let original = ing.summary_request(&session).await.map_err(|_| ApiError::unavailable())?;
        if !applied(store, &original.id).await? {
            let mut request = original.clone();
            let output = match self.cached_model(store, &request).await {
                Ok(output) => output,
                Err(ModelFailure::OutputLimit) => {
                    request.id = format!("{}:capacity:8192", original.id);
                    request.max_tokens = 8192;
                    match self.cached_model(store, &request).await {
                        Ok(output) => output,
                        Err(ModelFailure::OutputLimit) => {
                            reject_phase(
                                store,
                                &original.id,
                                &json!({"request_id":request.id,"max_tokens":8192}),
                                "summary output capacity exhausted; raw source and facts retained",
                            )
                            .await?;
                            json!({"keys":[],"themes":[]})
                        }
                        Err(ModelFailure::Backend) => return Err(ApiError::unavailable()),
                    }
                }
                Err(ModelFailure::Backend) => return Err(ApiError::unavailable()),
            };
            ing.apply_summary(&original.id, &output).await.map_err(|_| ApiError::unavailable())?;
        }
        Ok(())
    }
}
fn success(request: &AddRequest) -> Value {
    json!({"success":true,"request_id":request.request_id,"user_id":request.user_id,"session_id":request.session_id})
}
/// Reject unresolved deictic person labels rather than silently creating a new
/// person on each Add. This never rewrites a typed physical speaker into a user.
fn validate_role_identity(output: &Value, role: MessageRole) -> Result<(), String> {
    if !matches!(role, MessageRole::User | MessageRole::Assistant) {
        return Ok(());
    }
    let normalize = licorice::ingest::normalize;
    let proxy = |name: &str| {
        matches!(normalize(name).as_str(), "speaker" | "i" | "me" | "myself" | "person speaking" | "unnamed speaker")
    };
    let typed_nonperson: std::collections::HashSet<_> = output["entities"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entity| {
            let name = entity["name"].as_str()?;
            let kind = entity["kind"].as_str()?.to_ascii_lowercase();
            (!matches!(kind.as_str(), "person" | "concept" | "speaker" | "unknown" | "other")).then(|| normalize(name))
        })
        .collect();
    for entity in output["entities"].as_array().into_iter().flatten() {
        let name = entity["name"].as_str().unwrap_or_default();
        if canonical_actor(name).is_some() && entity["kind"] != "person" {
            return Err("canonical namespace actors User and Assistant must have kind person; keep third parties and typed physical objects under their own distinct names".into());
        }
        if !typed_nonperson.contains(&normalize(name))
            && entity["aliases"].as_array().into_iter().flatten().filter_map(Value::as_str).any(proxy)
        {
            return Err("ambiguous person alias: omit generic speaker labels and first-person pronouns; retain actual names as aliases of User/Assistant".into());
        }
    }
    for fact in output["facts"].as_array().into_iter().flatten() {
        for name in fact["subject"].as_str().into_iter().chain(if fact["object_is_entity"] == true {
            fact["object"].as_str()
        } else {
            None
        }) {
            if proxy(name) && !typed_nonperson.contains(&normalize(name)) {
                return Err("ambiguous unnamed speaker identity: use User/Assistant for the source author or a specific third-party reference; declare physical objects with their non-person entity kind".into());
            }
        }
    }
    Ok(())
}
async fn applied(store: &Store, id: &str) -> ApiResult<bool> {
    let id = id.to_owned();
    store
        .read(
            move |c| Ok(c.query_row("SELECT EXISTS(SELECT 1 FROM applied_outputs WHERE id=?1)", [&id], |r| r.get(0))?),
        )
        .await
        .map_err(|_| ApiError::unavailable())
}
