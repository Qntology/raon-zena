//! Client for premium-discount-app's standard I/O protocol (`app/src/commands.rs::serve_lines`):
//! one JSON object per line, `{"id", "cmd", "args"}` in and `{"id", "ok": true, "result"}` or
//! `{"id", "ok": false, "error": {"kind", "message"}}` out. The app answers requests one at a time, in order.
//!
//! Same shape as `zena_avatar_mcp::protocol::McpClient`: a reader thread routes replies to the waiting call by
//! id, stderr is forwarded prefixed, and a dead child fails every waiting and later call.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

/// Why a command did not return a result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallError {
    /// The app answered with an error: `kind` is its `CommandError.kind` (`offline`, `not_ready`, `invalid_input` …).
    Command { kind: String, message: String },
    /// The app could not be reached, died, or did not answer in time.
    Transport(String),
}

impl CallError {
    pub fn kind(&self) -> &str {
        match self {
            CallError::Command { kind, .. } => kind,
            CallError::Transport(_) => "unavailable",
        }
    }
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Command { kind, message } => write!(f, "{kind}: {message}"),
            CallError::Transport(m) => write!(f, "unavailable: {m}"),
        }
    }
}

impl std::error::Error for CallError {}

type Pending = Arc<Mutex<HashMap<u64, Sender<Value>>>>;

/// Decrements without going below zero.
fn release(n: &AtomicUsize) {
    let mut cur = n.load(Ordering::SeqCst);
    while cur > 0 {
        match n.compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return,
            Err(actual) => cur = actual,
        }
    }
}

/// A connection to one premium-discount-app process (or any reader / writer pair speaking the protocol).
pub struct LineClient {
    child: Option<Child>,
    writer: Arc<Mutex<Box<dyn Write + Send>>>,
    pending: Pending,
    alive: Arc<AtomicBool>,
    /// Requests written and not answered yet, including ones whose caller gave up waiting: the app is still
    /// working on them, so it must not be stopped.
    in_flight: Arc<AtomicUsize>,
    next_id: u64,
    label: String,
}

fn forward_stderr(label: String, stderr: impl Read + Send + 'static) {
    let _ = std::thread::Builder::new().name(format!("{label}-stderr")).spawn(move || {
        for line in BufReader::new(stderr).lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => eprintln!("[{label}] {l}"),
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });
}

impl LineClient {
    /// Spawns `program args…` with `envs` added to the inherited environment.
    pub fn spawn(program: &Path, args: &[String], envs: &[(String, String)]) -> Result<LineClient, CallError> {
        let mut cmd = Command::new(program);
        cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        for (k, v) in envs {
            cmd.env(k, v);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| CallError::Transport(format!("spawning {}: {e}", program.display())))?;
        let label = program
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "premium-discount-app".into());
        if let Some(stderr) = child.stderr.take() {
            forward_stderr(label.clone(), stderr);
        }
        let stdin = child.stdin.take().ok_or_else(|| CallError::Transport("child stdin".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| CallError::Transport("child stdout".into()))?;
        let mut client = LineClient::from_io(&label, stdout, stdin)?;
        client.child = Some(child);
        Ok(client)
    }

    /// Speaks the protocol over an existing pair (tests, or a process started elsewhere).
    pub fn from_io(label: &str, reader: impl Read + Send + 'static, writer: impl Write + Send + 'static) -> Result<LineClient, CallError> {
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let in_flight = Arc::new(AtomicUsize::new(0));
        {
            let (pending, alive, in_flight, label) = (pending.clone(), alive.clone(), in_flight.clone(), label.to_string());
            std::thread::Builder::new()
                .name(format!("{label}-stdout"))
                .spawn(move || {
                    for line in BufReader::new(reader).lines() {
                        let Ok(line) = line else { break };
                        if line.trim().is_empty() {
                            continue;
                        }
                        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                            eprintln!("[{label}] non-JSON output: {line}");
                            continue;
                        };
                        let Some(id) = msg.get("id").and_then(Value::as_u64) else {
                            // `id: null` answers a line the app could not parse; nothing waits for it.
                            eprintln!("[{label}] reply without a request id: {line}");
                            continue;
                        };
                        release(&in_flight);
                        if let Some(tx) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                            let _ = tx.send(msg);
                        }
                    }
                    alive.store(false, Ordering::SeqCst);
                    in_flight.store(0, Ordering::SeqCst);
                    if let Ok(mut p) = pending.lock() {
                        p.clear();
                    }
                })
                .map_err(|e| CallError::Transport(format!("reader thread: {e}")))?;
        }
        Ok(LineClient {
            child: None,
            writer: Arc::new(Mutex::new(Box::new(writer))),
            pending,
            alive,
            in_flight,
            next_id: 1,
            label: label.to_string(),
        })
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    /// Requests sent whose reply has not arrived (the app is busy with them).
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Sends `cmd` with `args` and waits up to `timeout` for its reply.
    pub fn call(&mut self, cmd: &str, args: Value, timeout: Duration) -> Result<Value, CallError> {
        if !self.is_alive() {
            return Err(CallError::Transport(format!("{} exited", self.label)));
        }
        let id = self.next_id;
        self.next_id += 1;
        let (tx, rx) = mpsc::channel();
        self.pending
            .lock()
            .map_err(|_| CallError::Transport("pending lock poisoned".into()))?
            .insert(id, tx);
        let mut line = json!({"id": id, "cmd": cmd, "args": args}).to_string();
        line.push('\n');
        // Counted before writing: the reply can arrive before `write_all` returns.
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        let sent = self
            .writer
            .lock()
            .map_err(|_| std::io::Error::other("writer lock poisoned"))
            .and_then(|mut w| w.write_all(line.as_bytes()).and_then(|_| w.flush()));
        if let Err(e) = sent {
            self.forget(id);
            release(&self.in_flight);
            return Err(CallError::Transport(format!("{}: writing {cmd}: {e}", self.label)));
        }
        let msg = match rx.recv_timeout(timeout) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => {
                self.forget(id);
                return Err(CallError::Transport(format!("{}: {cmd} timed out after {timeout:?}", self.label)));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err(CallError::Transport(format!("{} closed the connection during {cmd}", self.label)))
            }
        };
        if msg.get("ok").and_then(Value::as_bool) == Some(true) {
            return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
        }
        let err = msg.get("error");
        let field = |k: &str| err.and_then(|e| e.get(k)).and_then(Value::as_str).unwrap_or_default().to_string();
        let kind = field("kind");
        Err(CallError::Command {
            kind: if kind.is_empty() { "unknown".into() } else { kind },
            message: field("message"),
        })
    }

    fn forget(&self, id: u64) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(&id);
        }
    }
}

impl Drop for LineClient {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake app on OS pipes: answers `echo` with its args, `fail` with an error, `slow` after a delay and
    /// `quit` by hanging up.
    fn fake() -> LineClient {
        let (req_r, req_w) = std::io::pipe().unwrap();
        let (rep_r, mut rep_w) = std::io::pipe().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(req_r).lines() {
                let req: Value = serde_json::from_str(&line.unwrap()).unwrap();
                let id = req["id"].clone();
                let reply = match req["cmd"].as_str().unwrap() {
                    "echo" => json!({"id": id, "ok": true, "result": req["args"]}),
                    "fail" => json!({"id": id, "ok": false, "error": {"kind": "offline", "message": "no network"}}),
                    "slow" => {
                        std::thread::sleep(Duration::from_millis(300));
                        json!({"id": id, "ok": true, "result": "late"})
                    }
                    _ => return,
                };
                writeln!(rep_w, "not json").unwrap();
                writeln!(rep_w, "{reply}").unwrap();
            }
        });
        LineClient::from_io("fake", rep_r, req_w).unwrap()
    }

    #[test]
    fn routes_replies_and_errors() {
        let mut c = fake();
        assert_eq!(c.call("echo", json!({"a": 1}), Duration::from_secs(5)).unwrap(), json!({"a": 1}));
        let e = c.call("fail", Value::Null, Duration::from_secs(5)).unwrap_err();
        assert_eq!(
            e,
            CallError::Command {
                kind: "offline".into(),
                message: "no network".into()
            }
        );
        assert_eq!(e.kind(), "offline");
        assert_eq!(c.in_flight(), 0);
    }

    #[test]
    fn timeouts_keep_the_request_in_flight_until_it_is_answered() {
        let mut c = fake();
        let e = c.call("slow", Value::Null, Duration::from_millis(20)).unwrap_err();
        assert!(matches!(e, CallError::Transport(ref m) if m.contains("timed out")), "{e}");
        assert_eq!(c.in_flight(), 1, "the app is still working on it");
        // The next call waits behind it (the app answers in order) and the late reply is dropped.
        assert_eq!(c.call("echo", json!(2), Duration::from_secs(5)).unwrap(), json!(2));
        assert_eq!(c.in_flight(), 0);
    }

    #[test]
    fn a_dead_app_fails_cleanly() {
        let mut c = fake();
        let e = c.call("quit", Value::Null, Duration::from_secs(5)).unwrap_err();
        assert_eq!(e.kind(), "unavailable");
        assert!(!c.is_alive());
        assert!(c
            .call("echo", json!(1), Duration::from_secs(1))
            .unwrap_err()
            .to_string()
            .contains("exited"));
    }

    #[test]
    fn spawning_a_missing_program_fails_cleanly() {
        let e = LineClient::spawn(Path::new("definitely-not-premium-discount-app-xyz"), &[], &[])
            .err()
            .unwrap();
        assert!(e.to_string().contains("spawning"), "{e}");
    }
}
