//! Model requests and responses as plain data (SPEC §10.1).
//!
//! Licorice renders every model call (extraction, summaries, answers, grading) as a
//! [`ModelRequest`]. A live provider sends it to a model; the replay path writes it
//! to a JSONL file, has something else answer it, and reads the [`ModelResponse`]s
//! back. Both see the same prompts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One model call, fully rendered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Unique within a run; responses are matched by it.
    pub id: String,
    /// What the call is for: `extract`, `summarize`, `answer`, `grade`.
    pub task: String,
    pub system: String,
    pub user: String,
    /// JSON Schema the output must satisfy; `None` for free text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    pub max_tokens: u32,
}

/// A model's answer to a [`ModelRequest`]: parsed JSON when the request had a schema,
/// otherwise a JSON string.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub id: String,
    pub output: Value,
}

/// Checks `value` against the subset of JSON Schema that Licorice's schemas use: `type`
/// (one or a list), `enum`, `required`, `properties`, `additionalProperties: false`,
/// `items`, `maxItems`, `minimum` and `maximum`. Returns every violation, each with its
/// JSON path.
pub fn validate(schema: &Value, value: &Value) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    check(schema, value, "$", &mut errors);
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

fn check(schema: &Value, value: &Value, path: &str, errors: &mut Vec<String>) {
    if let Some(options) = schema.get("anyOf").and_then(Value::as_array) {
        if !options.iter().any(|option| {
            let mut branch = Vec::new();
            check(option, value, path, &mut branch);
            branch.is_empty()
        }) {
            errors.push(format!("{path}: no anyOf branch matched"));
            return;
        }
    }
    if let Some(types) = schema.get("type") {
        let allowed: Vec<&str> = match types {
            Value::String(t) => vec![t.as_str()],
            Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !allowed.iter().any(|t| has_type(value, t)) {
            errors.push(format!("{path}: expected {}, got {}", allowed.join(" or "), type_name(value)));
            return;
        }
    }
    if let Some(Value::Array(options)) = schema.get("enum") {
        if !options.contains(value) {
            errors.push(format!("{path}: {value} is not one of {}", Value::Array(options.clone())));
        }
    }
    if let Some(n) = value.as_f64() {
        if let Some(min) = schema.get("minimum").and_then(Value::as_f64) {
            if n < min {
                errors.push(format!("{path}: {n} is below the minimum {min}"));
            }
        }
        if let Some(max) = schema.get("maximum").and_then(Value::as_f64) {
            if n > max {
                errors.push(format!("{path}: {n} is above the maximum {max}"));
            }
        }
    }
    if let Value::Object(map) = value {
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(Value::Array(required)) = schema.get("required") {
            for key in required.iter().filter_map(Value::as_str) {
                if !map.contains_key(key) {
                    errors.push(format!("{path}: missing \"{key}\""));
                }
            }
        }
        for (key, v) in map {
            match properties.and_then(|p| p.get(key)) {
                Some(sub) => check(sub, v, &format!("{path}.{key}"), errors),
                None if schema.get("additionalProperties") == Some(&Value::Bool(false)) => {
                    errors.push(format!("{path}: unexpected \"{key}\""))
                }
                None => {}
            }
        }
    }
    if let Value::Array(items) = value {
        if let Some(max) = schema.get("maxItems").and_then(Value::as_u64) {
            if items.len() as u64 > max {
                errors.push(format!("{path}: {} items, at most {max} allowed", items.len()));
            }
        }
        if let Some(sub) = schema.get("items") {
            for (i, v) in items.iter().enumerate() {
                check(sub, v, &format!("{path}[{i}]"), errors);
            }
        }
    }
}

fn has_type(value: &Value, t: &str) -> bool {
    match t {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|f| f.fract() == 0.0),
        _ => false,
    }
}

fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate;

    #[test]
    fn validates_the_schema_subset() {
        let schema = json!({
            "type": "object", "additionalProperties": false, "required": ["a", "b"],
            "properties": {
                "a": {"type": "array", "maxItems": 2, "items": {"type": "integer", "minimum": 1, "maximum": 10}},
                "b": {"type": ["string", "null"], "enum": ["x", null]}
            }
        });
        assert!(validate(&schema, &json!({"a": [1, 10], "b": null})).is_ok());
        let errors = validate(&schema, &json!({"a": [0, 11, 3], "b": "y", "c": 1})).unwrap_err();
        assert_eq!(errors.len(), 5, "{errors:?}");
        assert!(errors.iter().any(|e| e == "$.a: 3 items, at most 2 allowed"), "{errors:?}");
        assert!(errors.iter().any(|e| e == "$: unexpected \"c\""), "{errors:?}");
        assert_eq!(validate(&schema, &json!({"a": "1"})).unwrap_err().len(), 2, "wrong type and missing b");
    }
}

/// Injectable model execution; requests and parsed outputs remain replayable data.
#[async_trait::async_trait]
pub trait ChatModel: Send + Sync {
    fn id(&self) -> &str;
    async fn complete(&self, request: &ModelRequest) -> Result<Value, ChatError>;
}

#[derive(Debug, thiserror::Error)]
#[error("model execution: {0}")]
pub struct ChatError(pub String);
impl ChatError {
    pub fn output_limit() -> Self {
        Self("model output token limit reached".into())
    }
    pub fn is_output_limit(&self) -> bool {
        self.0 == "model output token limit reached"
    }
}
