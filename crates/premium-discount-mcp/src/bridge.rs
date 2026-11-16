//! The premium-discount-app process behind the MCP tools.
//!
//! * Started on the first command that needs it, not with the server: the app loads embeddinggemma-2 while it
//!   starts (and the model is the only one it holds), so an idle Raon Zena keeps neither the model nor LanceDB
//!   in memory.
//! * Started again on the next command after it died.
//! * Stopped after `idle` without commands, unless it is still working on a request or a model download is
//!   running (stopping it would cancel the download). Its outbox is on disk, so nothing is lost: the app
//!   rebuilds the queue from the `queued` / `pending` rows when it starts again.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::lines::{CallError, LineClient};

/// What the tools need from premium-discount-app: one command at a time.
pub trait Backend {
    fn call(&mut self, cmd: &str, args: Value) -> Result<Value, CallError>;
    /// Process state for `pd_model_manage status` (`Null` when there is no process).
    fn state(&self) -> Value {
        Value::Null
    }
}

/// How long a command may take. The first command after a start also waits for the model to load.
pub fn timeout_for(cmd: &str) -> Duration {
    match cmd {
        "cmd_model_status" | "cmd_model_download" | "cmd_model_download_cancel" => Duration::from_secs(30),
        _ => Duration::from_secs(180),
    }
}

/// premium-discount-app's settings (`app/src/main.rs`), passed as `PD_*` environment variables.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub program: PathBuf,
    pub model_dir: PathBuf,
    pub data_dir: PathBuf,
    pub server_url: String,
    /// `text` (default here: Raon Zena only sends text) | `auto`.
    pub modalities: String,
    /// `q8_0` (default here: ~300 MB resident instead of ~660 MB F32, cos 0.9999 to the reference) | `auto` | `off`.
    pub quant: String,
    /// Stop the app after this long without commands (`None`: keep it).
    pub idle: Option<Duration>,
}

impl AppConfig {
    pub fn env(&self) -> Vec<(String, String)> {
        vec![
            ("PD_MODEL_DIR".into(), self.model_dir.display().to_string()),
            ("PD_DATA_DIR".into(), self.data_dir.display().to_string()),
            ("PD_SERVER_URL".into(), self.server_url.clone()),
            ("PD_MODALITIES".into(), self.modalities.clone()),
            ("PD_QUANT".into(), self.quant.clone()),
        ]
    }
}

struct Slot {
    client: Option<LineClient>,
    last_used: Instant,
    started: u64,
}

/// [`Backend`] over a premium-discount-app child process.
pub struct ProcessBackend {
    config: AppConfig,
    slot: Arc<Mutex<Slot>>,
}

fn download_running(status: &Value) -> bool {
    status.pointer("/download/state").and_then(Value::as_str) == Some("running")
}

impl ProcessBackend {
    pub fn new(config: AppConfig) -> ProcessBackend {
        let slot = Arc::new(Mutex::new(Slot {
            client: None,
            last_used: Instant::now(),
            started: 0,
        }));
        if let Some(idle) = config.idle {
            let weak = Arc::downgrade(&slot);
            let tick = (idle / 4).clamp(Duration::from_millis(50), Duration::from_secs(5));
            let _ = std::thread::Builder::new().name("pd-app-idle".into()).spawn(move || loop {
                std::thread::sleep(tick);
                let Some(slot) = weak.upgrade() else { break };
                // A command holds the lock for its whole call: skip this round instead of waiting for it.
                let Ok(mut guard) = slot.try_lock() else { continue };
                let Slot { client, last_used, .. } = &mut *guard;
                let Some(c) = client.as_mut() else { continue };
                if last_used.elapsed() < idle || c.in_flight() > 0 {
                    continue;
                }
                if c.is_alive() {
                    match c.call("cmd_model_status", Value::Null, Duration::from_secs(5)) {
                        Ok(st) if download_running(&st) => {
                            *last_used = Instant::now();
                            continue;
                        }
                        // Busy past the 5 s (in_flight > 0 now): try again later.
                        Err(CallError::Transport(_)) if c.in_flight() > 0 => continue,
                        _ => {}
                    }
                }
                eprintln!("premium-discount-mcp: premium-discount-app idle for {:?}, stopping it", idle);
                *client = None;
            });
        }
        ProcessBackend { config, slot }
    }

    pub fn config(&self) -> &AppConfig {
        &self.config
    }
}

impl Backend for ProcessBackend {
    /// Whether the app is running, how often it was started and how long it has been idle.
    fn state(&self) -> Value {
        match self.slot.lock() {
            Ok(s) => json!({
                "running": s.client.as_ref().is_some_and(LineClient::is_alive),
                "starts": s.started,
                "idle_ms": s.last_used.elapsed().as_millis() as u64,
            }),
            Err(_) => json!({"running": false}),
        }
    }

    fn call(&mut self, cmd: &str, args: Value) -> Result<Value, CallError> {
        let mut s = self.slot.lock().map_err(|_| CallError::Transport("app slot poisoned".into()))?;
        s.last_used = Instant::now();
        if s.client.as_ref().is_some_and(|c| !c.is_alive()) {
            eprintln!("premium-discount-mcp: premium-discount-app exited; starting it again");
            s.client = None;
        }
        if s.client.is_none() {
            for dir in [&self.config.model_dir, &self.config.data_dir] {
                std::fs::create_dir_all(dir).map_err(|e| CallError::Transport(format!("creating {}: {e}", dir.display())))?;
            }
            s.client = Some(LineClient::spawn(&self.config.program, &[], &self.config.env())?);
            s.started += 1;
        }
        let client = s.client.as_mut().expect("client just started");
        let out = client.call(cmd, args, timeout_for(cmd));
        s.last_used = Instant::now();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_program_is_unavailable_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("zena-pd-bridge-test-{}", std::process::id()));
        let mut b = ProcessBackend::new(AppConfig {
            program: PathBuf::from("definitely-not-premium-discount-app-xyz"),
            model_dir: dir.join("model"),
            data_dir: dir.join("data"),
            server_url: "http://127.0.0.1:9".into(),
            modalities: "text".into(),
            quant: "q8_0".into(),
            idle: Some(Duration::from_millis(100)),
        });
        let e = b.call("cmd_model_status", Value::Null).unwrap_err();
        assert_eq!(e.kind(), "unavailable");
        assert_eq!(b.state()["running"], false);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn download_state_is_read_from_model_status() {
        assert!(download_running(&json!({"download": {"state": "running"}})));
        assert!(!download_running(&json!({"download": {"state": "done"}})));
        assert!(!download_running(&json!({})));
    }
}
