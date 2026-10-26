//! Request and response files of a run, and the batch files a stand-in model works from.
//!
//! A run directory holds `requests/<task>.jsonl` (one [`ModelRequest`] per line). For
//! each task the requests are split into `batches/<task>-NN.md`, a readable rendering
//! for whoever plays the model, with its ids in `batches/<task>-NN.ids`. Outputs come
//! back as `responses/<task>-NN.jsonl`: one `{"id": …, "output": …}` per request.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use licorice_providers::chat::{ModelRequest, ModelResponse, validate};
use serde_json::Value;

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub fn requests_path(run: &Path, task: &str) -> PathBuf {
    run.join("requests").join(format!("{task}.jsonl"))
}

pub fn write_requests(run: &Path, task: &str, requests: &[ModelRequest]) -> Res<()> {
    let path = requests_path(run, task);
    fs::create_dir_all(path.parent().unwrap())?;
    let mut out = String::new();
    for r in requests {
        out.push_str(&serde_json::to_string(r)?);
        out.push('\n');
    }
    fs::write(path, out)?;
    Ok(())
}

pub fn read_requests(run: &Path, task: &str) -> Res<Vec<ModelRequest>> {
    let path = requests_path(run, task);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<Result<_, _>>()?)
}

/// Splits `requests` into batches of `size` and writes them; returns the batch names.
pub fn write_batches(run: &Path, task: &str, requests: &[ModelRequest], size: usize) -> Res<Vec<String>> {
    let dir = run.join("batches");
    fs::create_dir_all(&dir)?;
    fs::create_dir_all(run.join("responses"))?;
    for entry in fs::read_dir(&dir)? {
        let path = entry?.path();
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        if stem
            .strip_prefix(task)
            .and_then(|r| r.strip_prefix('-'))
            .is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit()))
        {
            fs::remove_file(path)?;
        }
    }
    let mut names = Vec::new();
    for (i, batch) in requests.chunks(size.max(1)).enumerate() {
        let name = format!("{task}-{:02}", i + 1);
        fs::write(dir.join(format!("{name}.md")), render(&name, batch))?;
        let ids: Vec<&str> = batch.iter().map(|r| r.id.as_str()).collect();
        fs::write(dir.join(format!("{name}.ids")), ids.join("\n") + "\n")?;
        names.push(name);
    }
    Ok(names)
}

fn fence(lang: &str, body: &str) -> String {
    format!("````{lang}\n{}\n````\n", body.trim_end())
}

fn render(name: &str, batch: &[ModelRequest]) -> String {
    let first = &batch[0];
    let shared = batch.iter().all(|r| r.system == first.system && r.schema == first.schema);
    let output = match &first.schema {
        Some(_) => "a JSON object that matches the output schema",
        None => "a JSON string holding your answer",
    };
    let mut out = format!(
        "# Batch {name}: {} {} requests\n\n\
         Act as the model for each request below: follow its system prompt and its instructions, and \
         produce the output it asks for. Write the outputs to `responses/{name}.jsonl` in the run \
         directory, one line per request, in request order:\n\n\
         {{\"id\": \"<request id>\", \"output\": <output>}}\n\n\
         where <output> is {output}. Answer every request.\n\n",
        batch.len(),
        first.task
    );
    if shared {
        out.push_str("## System prompt (all requests)\n\n");
        out.push_str(&fence("text", &first.system));
        if let Some(schema) = &first.schema {
            out.push_str("\n## Output schema (all requests)\n\n");
            out.push_str(&fence("json", &serde_json::to_string_pretty(schema).unwrap()));
        }
        out.push('\n');
    }
    for (i, r) in batch.iter().enumerate() {
        out.push_str(&format!("## Request {} of {}: id `{}`\n\n", i + 1, batch.len(), r.id));
        if !shared {
            out.push_str("System prompt:\n\n");
            out.push_str(&fence("text", &r.system));
            if let Some(schema) = &r.schema {
                out.push_str("Output schema:\n\n");
                out.push_str(&fence("json", &serde_json::to_string_pretty(schema).unwrap()));
            }
        }
        out.push_str(&fence("text", &r.user));
        out.push('\n');
    }
    out
}

/// Reads a response file: JSON Lines, a JSON array, or JSON objects one after another.
pub fn read_responses(path: &Path) -> Res<Vec<ModelResponse>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for value in serde_json::Deserializer::from_str(&text).into_iter::<Value>() {
        let value = value.map_err(|e| format!("{}: {e}", path.display()))?;
        match value {
            Value::Array(items) => {
                for item in items {
                    out.push(serde_json::from_value(item)?);
                }
            }
            item => out.push(serde_json::from_value(item)?),
        }
    }
    Ok(out)
}

/// A schema output sometimes arrives as a string holding JSON; parse it.
fn normalized(request: &ModelRequest, output: Value) -> Value {
    match (&request.schema, output) {
        (Some(_), Value::String(s)) => serde_json::from_str(s.trim()).unwrap_or(Value::String(s)),
        (_, output) => output,
    }
}

fn check_output(request: &ModelRequest, output: &Value) -> Vec<String> {
    match &request.schema {
        Some(schema) => validate(schema, output).err().unwrap_or_default(),
        None => match output.as_str() {
            Some(s) if !s.trim().is_empty() => Vec::new(),
            _ => vec!["$: expected a non-empty JSON string".into()],
        },
    }
}

/// Checks `responses/<batch>.jsonl` against the batch: every id answered once, every
/// output valid. Returns the problems found.
pub fn check(run: &Path, batch: &str) -> Res<Vec<String>> {
    let (task, _) = batch.rsplit_once('-').ok_or_else(|| format!("not a batch name: {batch}"))?;
    let ids_path = run.join("batches").join(format!("{batch}.ids"));
    let ids: Vec<String> = fs::read_to_string(&ids_path)
        .map_err(|e| format!("{}: {e}", ids_path.display()))?
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let requests: HashMap<String, ModelRequest> =
        read_requests(run, task)?.into_iter().map(|r| (r.id.clone(), r)).collect();
    let responses = match read_responses(&run.join("responses").join(format!("{batch}.jsonl"))) {
        Ok(r) => r,
        Err(e) => return Ok(vec![e.to_string()]),
    };
    let mut problems = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for r in &responses {
        if !ids.contains(&r.id) {
            problems.push(format!("{}: not in this batch", r.id));
            continue;
        }
        *seen.entry(r.id.as_str()).or_default() += 1;
        let request = &requests[&r.id];
        for e in check_output(request, &normalized(request, r.output.clone())) {
            problems.push(format!("{}: {e}", r.id));
        }
    }
    for id in &ids {
        match seen.get(id.as_str()) {
            None => problems.push(format!("{id}: no output")),
            Some(n) if *n > 1 => problems.push(format!("{id}: {n} outputs")),
            _ => {}
        }
    }
    Ok(problems)
}

/// Every valid output for `task`, by request id, from all of its response files.
/// Invalid or duplicate outputs are reported and left out.
pub fn collect(run: &Path, task: &str) -> Res<(HashMap<String, Value>, Vec<String>)> {
    let requests: HashMap<String, ModelRequest> =
        read_requests(run, task)?.into_iter().map(|r| (r.id.clone(), r)).collect();
    let mut files: Vec<PathBuf> = fs::read_dir(run.join("responses"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| e == "jsonl")
                && p.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with(&format!("{task}-")))
        })
        .collect();
    files.sort();
    let mut out = HashMap::new();
    let mut problems = Vec::new();
    for file in files {
        let responses = match read_responses(&file) {
            Ok(r) => r,
            Err(e) => {
                problems.push(e.to_string());
                continue;
            }
        };
        for r in responses {
            let Some(request) = requests.get(&r.id) else {
                problems.push(format!("{}: {}: unknown request", file.display(), r.id));
                continue;
            };
            let output = normalized(request, r.output);
            let errors = check_output(request, &output);
            if !errors.is_empty() {
                problems.push(format!("{}: {}: {}", file.display(), r.id, errors.join("; ")));
            } else if out.insert(r.id.clone(), output).is_some() {
                problems.push(format!("{}: {}: answered twice; kept the later output", file.display(), r.id));
            }
        }
    }
    Ok((out, problems))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(id: &str, schema: Option<Value>) -> ModelRequest {
        ModelRequest {
            id: id.into(),
            task: "t".into(),
            system: "sys".into(),
            user: format!("prompt for {id}"),
            schema,
            max_tokens: 10,
        }
    }

    #[test]
    fn batches_check_and_collect() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path();
        let schema = json!({"type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}}});
        let requests: Vec<ModelRequest> = (1..=3).map(|i| request(&format!("t:{i}"), Some(schema.clone()))).collect();
        write_requests(run, "t", &requests).unwrap();
        assert_eq!(read_requests(run, "t").unwrap(), requests);
        assert_eq!(write_batches(run, "t", &requests, 2).unwrap(), ["t-01", "t-02"]);
        let md = fs::read_to_string(run.join("batches/t-01.md")).unwrap();
        assert!(md.contains("## Output schema (all requests)") && md.contains("prompt for t:2") && !md.contains("t:3"));

        // JSON Lines and pretty-printed objects are both accepted; a JSON string holding
        // the object is parsed.
        fs::write(
            run.join("responses/t-01.jsonl"),
            "{\"id\": \"t:1\", \"output\": {\"x\": 1}}\n{\n  \"id\": \"t:2\",\n  \"output\": \"{\\\"x\\\": 2}\"\n}\n",
        )
        .unwrap();
        assert!(check(run, "t-01").unwrap().is_empty());
        fs::write(
            run.join("responses/t-02.jsonl"),
            "{\"id\": \"t:3\", \"output\": {\"x\": \"3\"}}\n{\"id\": \"t:1\", \"output\": {\"x\": 1}}\n",
        )
        .unwrap();
        let problems = check(run, "t-02").unwrap();
        assert_eq!(problems, ["t:3: $.x: expected integer, got string", "t:1: not in this batch"]);

        let (outputs, problems) = collect(run, "t").unwrap();
        assert_eq!(outputs.len(), 2, "t:3 is invalid");
        assert_eq!(outputs["t:2"], json!({"x": 2}));
        assert_eq!(problems.len(), 2, "{problems:?}");

        // Re-batching replaces the old batch files.
        assert_eq!(write_batches(run, "t", &requests, 3).unwrap(), ["t-01"]);
        assert!(!run.join("batches/t-02.md").exists());
    }
}
