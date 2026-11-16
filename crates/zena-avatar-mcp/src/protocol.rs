//! Minimal MCP (Model Context Protocol) over stdio: JSON-RPC 2.0, one JSON message per line.
//!
//! * [`McpClient`] spawns a server process, performs the `initialize` → `notifications/initialized`
//!   handshake (protocol `2025-06-18`) and calls tools. A reader thread routes responses to the waiting call
//!   by id, answers server `ping`s and rejects other server→client requests (no sampling / roots support).
//! * [`serve`] runs a server loop for a [`ToolHandler`] (`initialize`, `ping`, `tools/list`, `tools/call`;
//!   notifications are accepted silently). Handler errors and panics become `isError: true` tool results.
//!
//! Synchronous threads only — no async runtime.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

pub const PROTOCOL_VERSION: &str = "2025-06-18";
/// Older protocol revisions a client may ask for; the server echoes a supported one.
const SUPPORTED_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

type Pending = Arc<Mutex<HashMap<u64, Sender<Value>>>>;

/// A connected MCP server child process.
pub struct McpClient {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    pending: Pending,
    alive: Arc<AtomicBool>,
    next_id: u64,
    server_name: String,
    label: String,
}

fn write_line(w: &Mutex<ChildStdin>, v: &Value) -> Result<()> {
    let mut line = serde_json::to_string(v)?;
    line.push('\n');
    let mut w = w.lock().map_err(|_| anyhow!("stdin lock poisoned"))?;
    w.write_all(line.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Joins the `text` items of a tool result's `content`.
pub fn result_text(result: &Value) -> String {
    result
        .get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|c| c.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// The structured payload of a tool result: `structuredContent`, else the first text item parsed as JSON,
/// else the text as a JSON string.
pub fn result_json(result: &Value) -> Value {
    if let Some(s) = result.get("structuredContent") {
        return s.clone();
    }
    let text = result_text(result);
    serde_json::from_str(&text).unwrap_or(Value::String(text))
}

impl McpClient {
    /// Spawns `program args…` with piped stdin/stdout (stderr is forwarded line by line to ours, prefixed) and
    /// runs the MCP handshake.
    pub fn spawn(program: &Path, args: &[String]) -> Result<McpClient> {
        let mut cmd = Command::new(program);
        cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd.spawn().with_context(|| format!("spawning {}", program.display()))?;
        let label = program
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "mcp".into());
        let stdin = Arc::new(Mutex::new(child.stdin.take().context("child stdin")?));
        let stdout = child.stdout.take().context("child stdout")?;
        if let Some(stderr) = child.stderr.take() {
            let label = label.clone();
            std::thread::Builder::new().name(format!("{label}-stderr")).spawn(move || {
                for line in BufReader::new(stderr).lines() {
                    match line {
                        Ok(l) if !l.trim().is_empty() => eprintln!("[{label}] {l}"),
                        Ok(_) => {}
                        Err(_) => break,
                    }
                }
            })?;
        }
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        {
            let (pending, alive, stdin, label) = (pending.clone(), alive.clone(), stdin.clone(), label.clone());
            std::thread::Builder::new().name(format!("{label}-stdout")).spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if line.trim().is_empty() {
                        continue;
                    }
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                        eprintln!("[{label}] non-JSON output: {line}");
                        continue;
                    };
                    let id = msg.get("id").cloned();
                    match (msg.get("method").and_then(Value::as_str), id) {
                        // Server → client request.
                        (Some(method), Some(id)) if !id.is_null() => {
                            let reply = if method == "ping" {
                                json!({"jsonrpc": "2.0", "id": id, "result": {}})
                            } else {
                                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("client does not support {method}")}})
                            };
                            let _ = write_line(&stdin, &reply);
                        }
                        // Notification (logging, progress, list_changed …): ignored.
                        (Some(_), _) => {}
                        // Response.
                        (None, Some(id)) => {
                            if let Some(id) = id.as_u64() {
                                if let Some(tx) = pending.lock().ok().and_then(|mut p| p.remove(&id)) {
                                    let _ = tx.send(msg);
                                }
                            }
                        }
                        (None, None) => {}
                    }
                }
                alive.store(false, Ordering::SeqCst);
                // Dropping the senders wakes every waiting call with a disconnect.
                if let Ok(mut p) = pending.lock() {
                    p.clear();
                }
            })?;
        }
        let mut client = McpClient {
            child,
            stdin,
            pending,
            alive,
            next_id: 1,
            server_name: String::new(),
            label,
        };
        let init = client.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "raon-zena", "version": env!("CARGO_PKG_VERSION")},
            }),
            Duration::from_secs(20),
        )?;
        client.server_name = init
            .pointer("/serverInfo/name")
            .and_then(Value::as_str)
            .unwrap_or(&client.label)
            .to_string();
        write_line(&client.stdin, &json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))?;
        Ok(client)
    }

    /// The server's `serverInfo.name` from `initialize`.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Sends a JSON-RPC request and waits for its `result` (JSON-RPC errors become `Err`).
    pub fn request(&mut self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        if !self.alive.load(Ordering::SeqCst) {
            bail!("{}: server exited", self.label);
        }
        let id = self.next_id;
        self.next_id += 1;
        let (tx, rx) = mpsc::channel();
        self.pending.lock().map_err(|_| anyhow!("pending lock poisoned"))?.insert(id, tx);
        let sent = write_line(
            &self.stdin,
            &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
        );
        if let Err(e) = sent {
            self.forget(id);
            return Err(e.context(format!("{}: writing {method}", self.label)));
        }
        let msg = match rx.recv_timeout(timeout) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => {
                self.forget(id);
                bail!("{}: {method} timed out after {:?}", self.label, timeout);
            }
            Err(RecvTimeoutError::Disconnected) => bail!("{}: server closed the connection during {method}", self.label),
        };
        if let Some(err) = msg.get("error") {
            bail!(
                "{}: {method} failed: {} ({})",
                self.label,
                err.get("message").and_then(Value::as_str).unwrap_or("unknown error"),
                err.get("code").cloned().unwrap_or(Value::Null)
            );
        }
        Ok(msg.get("result").cloned().unwrap_or(Value::Null))
    }

    fn forget(&self, id: u64) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(&id);
        }
    }

    /// `tools/call`; returns the whole `result`. `result.isError == true` becomes `Err` with its text content.
    pub fn call_tool(&mut self, name: &str, args: Value, timeout: Duration) -> Result<Value> {
        let result = self.request("tools/call", json!({"name": name, "arguments": args}), timeout)?;
        if result.get("isError").and_then(Value::as_bool) == Some(true) {
            let text = result_text(&result);
            bail!("{name}: {}", if text.is_empty() { "tool error".to_string() } else { text });
        }
        Ok(result)
    }

    /// Names of every tool (`tools/list`, following `nextCursor`).
    pub fn list_tools(&mut self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..64 {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let r = self.request("tools/list", params, Duration::from_secs(20))?;
            for t in r.get("tools").and_then(Value::as_array).into_iter().flatten() {
                if let Some(n) = t.get("name").and_then(Value::as_str) {
                    names.push(n.to_string());
                }
            }
            cursor = r.get("nextCursor").and_then(Value::as_str).map(str::to_string);
            if cursor.is_none() {
                break;
            }
        }
        Ok(names)
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Server-side tool provider for [`serve`].
pub trait ToolHandler {
    /// Tool descriptors: `{"name", "description", "inputSchema"}`.
    fn tools(&self) -> Vec<Value>;
    fn call(&mut self, name: &str, args: &Value) -> anyhow::Result<Value>;
}

fn rpc_error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Wraps a handler value as an MCP tool result (`structuredContent` must be an object, so other values are
/// wrapped as `{"value": …}`).
pub fn tool_result(value: &Value) -> Value {
    let structured = if value.is_object() {
        value.clone()
    } else {
        json!({"value": value})
    };
    json!({
        "content": [{"type": "text", "text": value.to_string()}],
        "structuredContent": structured,
        "isError": false,
    })
}

fn handle(handler: &mut dyn ToolHandler, msg: &Value, server_name: &str) -> Option<Value> {
    let method = msg.get("method").and_then(Value::as_str);
    let id = msg.get("id").cloned();
    let Some(method) = method else {
        // A response to a request we never send, or an invalid message.
        return match id {
            Some(id) if msg.get("result").is_none() && msg.get("error").is_none() => Some(rpc_error(id, -32600, "invalid request")),
            _ => None,
        };
    };
    let id = match id {
        Some(id) if !id.is_null() => id,
        _ => return None, // notification
    };
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOL_VERSION);
            let version = if SUPPORTED_VERSIONS.contains(&asked) {
                asked
            } else {
                PROTOCOL_VERSION
            };
            json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": server_name, "version": env!("CARGO_PKG_VERSION")},
            })
        }
        "ping" => json!({}),
        "tools/list" => json!({"tools": handler.tools()}),
        "tools/call" => {
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return Some(rpc_error(id, -32602, "tools/call needs a string `name`"));
            };
            let args = match params.get("arguments") {
                None | Some(Value::Null) => json!({}),
                Some(a @ Value::Object(_)) => a.clone(),
                Some(_) => return Some(rpc_error(id, -32602, "`arguments` must be an object")),
            };
            let known = handler.tools().iter().any(|t| t.get("name").and_then(Value::as_str) == Some(name));
            if !known {
                return Some(rpc_error(id, -32602, &format!("unknown tool: {name}")));
            }
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.call(name, &args)));
            match outcome {
                Ok(Ok(v)) => tool_result(&v),
                Ok(Err(e)) => json!({"content": [{"type": "text", "text": format!("{e:#}")}], "isError": true}),
                Err(_) => json!({"content": [{"type": "text", "text": format!("{name} panicked")}], "isError": true}),
            }
        }
        _ => return Some(rpc_error(id, -32601, &format!("method not found: {method}"))),
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// Serves MCP on newline-delimited JSON until `input` ends.
pub fn serve(handler: &mut dyn ToolHandler, input: impl BufRead, mut output: impl Write, server_name: &str) -> Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(Value::Object(m)) => handle(handler, &Value::Object(m), server_name),
            Ok(_) => Some(rpc_error(Value::Null, -32600, "invalid request (batches are not supported)")),
            Err(e) => Some(rpc_error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(r) = reply {
            serde_json::to_writer(&mut output, &r)?;
            output.write_all(b"\n")?;
            output.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Echo;
    impl ToolHandler for Echo {
        fn tools(&self) -> Vec<Value> {
            vec![
                json!({"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}),
                json!({"name": "fail", "description": "always fails", "inputSchema": {"type": "object"}}),
                json!({"name": "boom", "description": "panics", "inputSchema": {"type": "object"}}),
            ]
        }
        fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
            match name {
                "echo" => Ok(json!({"echo": args})),
                "fail" => bail!("nope"),
                _ => panic!("boom"),
            }
        }
    }

    fn run(lines: &[Value]) -> Vec<Value> {
        let input: String = lines.iter().map(|l| format!("{l}\n")).collect::<String>() + "\nnot json\n";
        let mut out = Vec::new();
        serve(&mut Echo, Cursor::new(input), &mut out, "test-server").unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn serve_speaks_mcp() {
        let r = run(&[
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"x": 1}}}),
            json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "fail"}}),
            json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "missing"}}),
            json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"arguments": {}}}),
            json!({"jsonrpc": "2.0", "id": 7, "method": "resources/list"}),
            json!({"jsonrpc": "2.0", "id": "p", "method": "ping"}),
            json!({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "boom"}}),
        ]);
        assert_eq!(
            r.len(),
            10,
            "the notification gets no reply, the garbage line gets a parse error: {r:?}"
        );
        assert_eq!(r[0]["result"]["serverInfo"]["name"], "test-server");
        assert_eq!(r[0]["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(r[1]["result"]["tools"].as_array().unwrap().len(), 3);
        assert_eq!(r[2]["result"]["isError"], false);
        assert_eq!(r[2]["result"]["structuredContent"]["echo"]["x"], 1);
        assert_eq!(result_json(&r[2]["result"])["echo"]["x"], 1);
        let text: Value = serde_json::from_str(r[2]["result"]["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(text["echo"]["x"], 1);
        assert_eq!(r[3]["result"]["isError"], true);
        assert_eq!(result_text(&r[3]["result"]), "nope");
        assert_eq!(r[4]["error"]["code"], -32602);
        assert_eq!(r[5]["error"]["code"], -32602);
        assert_eq!(r[6]["error"]["code"], -32601);
        assert_eq!(r[7]["id"], "p");
        assert_eq!(r[7]["result"], json!({}));
        assert_eq!(r[8]["result"]["isError"], true, "a panicking handler is contained");
        assert_eq!(r[9]["error"]["code"], -32700);
    }

    #[test]
    fn spawning_a_missing_program_fails_cleanly() {
        let err = McpClient::spawn(Path::new("definitely-not-a-real-mcp-server-xyz"), &[])
            .err()
            .unwrap();
        assert!(format!("{err:#}").contains("spawning"));
    }
}
