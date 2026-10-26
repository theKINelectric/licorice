//! Coffee Cat's native CLI as a long-lived local judge.
//!
//! `coffeecat ask --jsonl` loads its pinned Bonsai GGUF on the first valid request, then
//! answers one JSON request per stdin line with one flushed JSON line on stdout, in
//! order. The process starts on first use and is kept for later calls; one request is
//! in flight at a time. A request abandoned at its deadline is still answered, so
//! responses are matched by request id and late ones are discarded.

use std::io;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde_json::{Map, Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use super::http::{coffee_cat_question, decode};
use super::{Judge, JudgeError, Judgment, Point, Question};

/// Coffee Cat refuses request lines above 1 MiB, not counting the newline.
const MAX_REQUEST_BYTES: usize = 1 << 20;

/// How to run the judge process.
#[derive(Clone, Debug)]
pub struct ProcessJudgeConfig {
    /// The `coffeecat` executable; a bare name is looked up on `PATH`.
    pub program: PathBuf,
    /// KV context in tokens (`--context`, a multiple of 256). Coffee Cat's 2,048 default
    /// refuses a 2,000-token Licorice memory, which needs about 3,540 Bonsai tokens.
    pub context: u32,
    /// Passed after `ask --jsonl --context N`, for example `--gpu-layers 0`.
    pub extra_args: Vec<String>,
    /// Deadline for [`Judge::warm_up`], which pays for process start and model load.
    pub startup_deadline: Duration,
}

impl ProcessJudgeConfig {
    /// `coffeecat` on `PATH` (Coffee Cat native 0.4) with a 4,096-token context.
    pub fn coffee_cat() -> ProcessJudgeConfig {
        ProcessJudgeConfig {
            program: "coffeecat".into(),
            context: 4096,
            extra_args: Vec::new(),
            startup_deadline: Duration::from_secs(60),
        }
    }
}

pub struct ProcessJudge {
    cfg: ProcessJudgeConfig,
    id: String,
    next_request: AtomicU64,
    /// Held for a whole exchange, so one request is in flight.
    worker: Mutex<Option<Worker>>,
}

/// One running process. Dropping it kills the process.
struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Bytes of the response line being read. Kept here so a read cancelled by a
    /// deadline resumes where it stopped.
    partial: Vec<u8>,
    /// Requests written; Coffee Cat echoes each one's one-based `line`.
    sent: u64,
    /// Whether the process has written anything yet.
    spoke: bool,
    /// A write cancelled midway may have left half a request in the pipe.
    torn: bool,
}

impl ProcessJudge {
    /// Validates the configuration. The process starts on first use.
    pub fn new(cfg: ProcessJudgeConfig) -> Result<ProcessJudge, JudgeError> {
        if cfg.program.as_os_str().is_empty() {
            return Err(JudgeError::Invalid("judge program is empty".into()));
        }
        if cfg.context == 0 || cfg.context % 256 != 0 {
            return Err(JudgeError::Invalid(format!("context {} is not a positive multiple of 256", cfg.context)));
        }
        let mut id = format!("{} ask --jsonl --context {}", cfg.program.display(), cfg.context);
        for arg in &cfg.extra_args {
            id.push(' ');
            id.push_str(arg);
        }
        Ok(ProcessJudge { cfg, id, next_request: AtomicU64::new(0), worker: Mutex::new(None) })
    }

    fn spawn(&self) -> Result<Worker, JudgeError> {
        let mut child = Command::new(&self.cfg.program)
            .args(["ask", "--jsonl", "--context"])
            .arg(self.cfg.context.to_string())
            .args(&self.cfg.extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| JudgeError::Startup(format!("cannot run {}: {e}", self.cfg.program.display())))?;
        let stdin = child.stdin.take().expect("stdin is piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
        tracing::debug!(judge = %self.id, pid = child.id(), "judge process started");
        Ok(Worker { child, stdin, stdout, partial: Vec::new(), sent: 0, spoke: false, torn: false })
    }

    /// The running process, or a new one if there is none, it has exited, or a
    /// cancelled write tore its input.
    fn live<'a>(&self, slot: &'a mut Option<Worker>) -> Result<&'a mut Worker, JudgeError> {
        let usable = slot.as_mut().is_some_and(|w| !w.torn && matches!(w.child.try_wait(), Ok(None)));
        if !usable {
            *slot = None;
            *slot = Some(self.spawn()?);
        }
        Ok(slot.as_mut().expect("worker was just ensured"))
    }

    /// Writes one request line and returns its answer. Late answers to earlier,
    /// abandoned requests are read and discarded on the way.
    async fn exchange(
        &self,
        slot: &mut Option<Worker>,
        id: &str,
        line: &[u8],
        deadline: Duration,
    ) -> Result<Value, JudgeError> {
        // A process that died while idle may only show it when written to; the request
        // never reached it, so one fresh process may retry it.
        for attempt in 1.. {
            let worker = self.live(slot)?;
            worker.torn = true;
            match write_line(&mut worker.stdin, line).await {
                Ok(()) => {
                    worker.torn = false;
                    worker.sent += 1;
                    break;
                }
                Err(e) => {
                    *slot = None;
                    if attempt == 2 {
                        return Err(JudgeError::Transport(format!("writing to {}: {e}", self.cfg.program.display())));
                    }
                }
            }
        }
        let worker = slot.as_mut().expect("the request was written");
        let line_no = worker.sent;
        loop {
            let raw = match next_line(worker).await {
                Ok(Some(raw)) => raw,
                Ok(None) => {
                    let worker = slot.take().expect("worker is present");
                    let spoke = worker.spoke;
                    let message = format!(
                        "{} exited before answering ({})",
                        self.cfg.program.display(),
                        exit_status(worker).await
                    );
                    // Silence from a new process means it never started (bad arguments, wrong program).
                    return Err(if spoke { JudgeError::Transport(message) } else { JudgeError::Startup(message) });
                }
                Err(e) => {
                    *slot = None;
                    return Err(JudgeError::Transport(format!("reading from {}: {e}", self.cfg.program.display())));
                }
            };
            worker.spoke = true;
            let value: Value = match serde_json::from_slice(&raw) {
                Ok(value) => value,
                Err(e) => {
                    tracing::warn!(judge = %self.id, error = %e, "skipping a judge output line that is not JSON");
                    continue;
                }
            };
            let line = value.get("line").and_then(Value::as_u64);
            // Records Coffee Cat could not parse carry `id: null`; their `line` still
            // locates them.
            let ours = match value.get("id") {
                Some(Value::String(echoed)) => echoed == id,
                Some(Value::Null) | None => line == Some(line_no),
                Some(_) => false,
            };
            if !ours {
                if let Some(ahead) = line.filter(|l| *l > line_no) {
                    *slot = None;
                    return Err(JudgeError::Decode(format!("judge answered line {ahead}, but {line_no} were sent")));
                }
                tracing::debug!(judge = %self.id, id = %value["id"], "discarding a late answer");
                continue;
            }
            if value.get("kind").and_then(Value::as_str) == Some("answer") {
                return Ok(value);
            }
            let error = value.get("error");
            let code = error.and_then(|e| e.get("code")).and_then(Value::as_str).unwrap_or("unknown");
            let message = error.and_then(|e| e.get("message")).and_then(Value::as_str).unwrap_or("");
            if code == "startup_error" {
                // The process exits after reporting it; the next call starts afresh.
                *slot = None;
            }
            return Err(refusal(code, message, deadline));
        }
    }

    async fn judge(
        &self,
        state: &Value,
        questions: &IndexMap<String, Question>,
        deadline: Duration,
    ) -> Result<Judgment, JudgeError> {
        if questions.is_empty() {
            return Err(JudgeError::Invalid("at least one question is required".into()));
        }
        let started = Instant::now();
        let mut qs = Map::new();
        for (name, q) in questions {
            let mut value = serde_json::to_value(q).map_err(|e| JudgeError::Invalid(e.to_string()))?;
            coffee_cat_question(&mut value);
            qs.insert(name.clone(), value);
        }
        let id = format!("licorice-{}", self.next_request.fetch_add(1, Ordering::Relaxed));
        let work = async {
            let mut slot = self.worker.lock().await;
            // Coffee Cat refuses deadlines below 100 ms; never extend the caller's.
            let remaining = deadline.saturating_sub(started.elapsed());
            if remaining < Duration::from_millis(100) {
                return Err(JudgeError::Timeout(deadline));
            }
            let request = json!({
                "id": id, "state": state, "questions": qs,
                "deadline_ms": remaining.as_millis().min(120_000) as u64,
            });
            let mut line = serde_json::to_vec(&request).map_err(|e| JudgeError::Invalid(e.to_string()))?;
            if line.len() > MAX_REQUEST_BYTES {
                return Err(JudgeError::Invalid(format!(
                    "request is {} bytes; Coffee Cat accepts at most {MAX_REQUEST_BYTES}",
                    line.len()
                )));
            }
            line.push(b'\n');
            self.exchange(&mut slot, &id, &line, deadline).await
        };
        let body = tokio::time::timeout(deadline, work).await.map_err(|_| JudgeError::Timeout(deadline))??;
        let asked: Vec<(&String, &Question)> = questions.iter().collect();
        let (model, answers, mut usage) = decode(&body, &asked)?;
        // Every prompt and candidate token is read; none is generated.
        usage.input_tokens = usage.input_tokens.or_else(|| body.get("usage")?.get("tokens")?.as_u64());
        Ok(Judgment { model, answers, usage, latency: started.elapsed() })
    }
}

#[async_trait::async_trait]
impl Judge for ProcessJudge {
    fn id(&self) -> &str {
        &self.id
    }

    fn is_loopback(&self) -> bool {
        true
    }

    async fn warm_up(&self) -> Result<(), JudgeError> {
        let started = Instant::now();
        let probe = IndexMap::from([("ready".to_owned(), Question::noul("Is this engine ready?"))]);
        self.judge(&json!({}), &probe, self.cfg.startup_deadline).await?;
        tracing::debug!(judge = %self.id, latency_ms = started.elapsed().as_millis() as u64, "judge ready");
        Ok(())
    }

    async fn ask(
        &self,
        point: Point,
        state: &Value,
        questions: &IndexMap<String, Question>,
        deadline: Duration,
    ) -> Result<Judgment, JudgeError> {
        let judgment = self.judge(state, questions, deadline).await?;
        tracing::debug!(
            %point, model = %judgment.model, questions = questions.len(),
            latency_ms = judgment.latency.as_millis() as u64, "judgment"
        );
        Ok(judgment)
    }
}

/// Maps a Coffee Cat error code (README, "Refusals") onto a judge error.
fn refusal(code: &str, message: &str, deadline: Duration) -> JudgeError {
    match code {
        "deadline_exceeded" => JudgeError::Timeout(deadline),
        "startup_error" => JudgeError::Startup(message.to_owned()),
        "invalid_input" | "bounded_limit_exceeded" | "unknown_model" => {
            JudgeError::Invalid(format!("{code}: {message}"))
        }
        _ => JudgeError::Engine(format!("{code}: {message}")),
    }
}

async fn write_line(stdin: &mut ChildStdin, line: &[u8]) -> io::Result<()> {
    stdin.write_all(line).await?;
    stdin.flush().await
}

/// The next nonblank output line without its newline, or `None` at end of output.
async fn next_line(worker: &mut Worker) -> io::Result<Option<Vec<u8>>> {
    loop {
        let read = worker.stdout.read_until(b'\n', &mut worker.partial).await?;
        if worker.partial.last() == Some(&b'\n') {
            let mut line = std::mem::take(&mut worker.partial);
            line.pop();
            if !line.iter().all(u8::is_ascii_whitespace) {
                return Ok(Some(line));
            }
        } else if read == 0 {
            return Ok(None);
        }
    }
}

/// Why a process that closed its output stopped, for the error message.
async fn exit_status(mut worker: Worker) -> String {
    match tokio::time::timeout(Duration::from_millis(500), worker.child.wait()).await {
        Ok(Ok(status)) => status.to_string(),
        Ok(Err(e)) => e.to_string(),
        Err(_) => "closed its output but kept running; killed".into(),
    }
}
