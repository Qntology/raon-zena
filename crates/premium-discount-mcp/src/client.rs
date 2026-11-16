//! The app side: a typed [`PdSyncClient`] over `zena_avatar_mcp::protocol::McpClient`, and [`SyncLoop`], the
//! coalesced background sync of premium-discount's `app/ui/coalesce.ts` + `main.ts`:
//!
//! * a request while idle runs at once;
//! * requests while a run is in progress are not dropped: they share exactly one rerun after it, and the rerun
//!   flushes if any of them asked to;
//! * while any row is `queued` or `pending`, a run with flush is due every `interval` (10 s in the app).
//!
//! The drop handler, opening the history, the "check now" button and the timer can all call
//! [`SyncLoop::request`]; the server sees one `pd_flush_and_sync` at a time. Offline, the flush stops (the
//! outbox keeps its rows) and the local rows still come back, so the history keeps working.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::{json, Value};
use zena_avatar_mcp::protocol::{result_json, McpClient};

use crate::server::error_kind;

/// Longer than the server's own wait for premium-discount-app (180 s), so its error arrives first.
pub const TOOL_TIMEOUT: Duration = Duration::from_secs(200);
/// `main.ts`: the refresh period while something waits to be sent or checked.
pub const SYNC_INTERVAL: Duration = Duration::from_secs(10);

pub struct PdSyncClient {
    mcp: McpClient,
}

impl PdSyncClient {
    /// Spawns `premium-discount-mcp args…` and runs the MCP handshake.
    pub fn spawn(program: &Path, args: &[String]) -> Result<PdSyncClient> {
        Ok(PdSyncClient {
            mcp: McpClient::spawn(program, args)?,
        })
    }

    pub fn from_client(mcp: McpClient) -> PdSyncClient {
        PdSyncClient { mcp }
    }

    /// Calls `tool` and returns its structured result. A tool error is `Err("<tool>: <kind>: <message>")`.
    pub fn call(&mut self, tool: &str, args: Value) -> Result<Value> {
        Ok(result_json(&self.mcp.call_tool(tool, args, TOOL_TIMEOUT)?))
    }

    pub fn enqueue(&mut self, node: Value) -> Result<Value> {
        self.call("pd_enqueue_node", node)
    }

    pub fn flush_and_sync(&mut self, since_ms: i64, flush: bool) -> Result<Value> {
        self.call("pd_flush_and_sync", json!({"since_ms": since_ms, "flush": flush}))
    }

    pub fn search_merged(&mut self, query: Value) -> Result<Value> {
        self.call("pd_search_merged", query)
    }
}

/// One sync run: flush (when asked) and read the rows changed after `since_ms`.
pub trait SyncRunner: Send + 'static {
    fn run(&mut self, since_ms: i64, flush: bool) -> Result<Value>;
}

impl SyncRunner for PdSyncClient {
    fn run(&mut self, since_ms: i64, flush: bool) -> Result<Value> {
        self.flush_and_sync(since_ms, flush)
    }
}

/// Shares the client with the app's other calls (enqueue, search).
impl<R: SyncRunner> SyncRunner for Arc<Mutex<R>> {
    fn run(&mut self, since_ms: i64, flush: bool) -> Result<Value> {
        self.lock()
            .map_err(|_| anyhow::anyhow!("sync client lock poisoned"))?
            .run(since_ms, flush)
    }
}

/// What the overlay shows: the local rows (by `request_id`, like the web UI's Dexie cache) and the last run.
#[derive(Debug, Clone, Default)]
pub struct SyncSnapshot {
    pub rows: BTreeMap<String, Value>,
    pub since_ms: i64,
    /// The last flush summary (`sent`, `remaining`, `stopped`, `stored`, `still_pending` …).
    pub last_flush: Option<Value>,
    pub last_error: Option<String>,
    /// The last run could not reach the server (or premium-discount-app).
    pub offline: bool,
    pub runs: u64,
}

impl SyncSnapshot {
    pub fn count(&self, state: &str) -> usize {
        self.rows
            .values()
            .filter(|r| r.get("state").and_then(Value::as_str) == Some(state))
            .count()
    }

    /// Something waits to be sent (`queued`) or checked (`pending`).
    pub fn has_work(&self) -> bool {
        self.count("queued") + self.count("pending") > 0
    }

    /// The Stage 0 status dot: the least settled state present.
    pub fn status_dot(&self) -> Option<&'static str> {
        ["queued", "pending", "stored"].into_iter().find(|s| self.count(s) > 0)
    }

    fn apply(&mut self, result: Result<Value>) {
        self.runs += 1;
        match result {
            Ok(v) => {
                for row in v.get("rows").and_then(Value::as_array).into_iter().flatten() {
                    if let Some(id) = row.get("request_id").and_then(Value::as_str) {
                        self.rows.insert(id.to_string(), row.clone());
                    }
                }
                if let Some(next) = v.get("next_since_ms").and_then(Value::as_i64) {
                    self.since_ms = self.since_ms.max(next);
                }
                let flush = v.get("flush").filter(|f| !f.is_null()).cloned();
                self.offline = flush
                    .as_ref()
                    .is_some_and(|f| f.get("stopped").is_some_and(|s| !s.is_null()) || f.get("sync_stopped").is_some_and(|s| !s.is_null()));
                if flush.is_some() {
                    self.last_flush = flush;
                }
                self.last_error = None;
            }
            Err(e) => {
                let message = format!("{e:#}");
                // `pd_flush_and_sync: <kind>: …` from the server, or a transport failure of the MCP client itself.
                let kind = message.split_once(": ").and_then(|(_, rest)| error_kind(rest));
                self.offline = matches!(kind, Some("offline" | "unavailable") | None);
                self.last_error = Some(message);
            }
        }
    }
}

#[derive(Default)]
struct LoopState {
    requested: bool,
    flush: bool,
    running: bool,
    stop: bool,
    snapshot: SyncSnapshot,
}

struct Shared {
    state: Mutex<LoopState>,
    cv: Condvar,
}

/// Background sync with coalesced requests (see the module docs).
pub struct SyncLoop {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

type OnUpdate = Box<dyn Fn(&SyncSnapshot) + Send>;

impl SyncLoop {
    /// Starts the loop. Nothing runs until the first [`request`](Self::request); afterwards the timer keeps
    /// running while [`SyncSnapshot::has_work`]. `on_update` is called on the loop's thread after every run.
    pub fn start(mut runner: impl SyncRunner, interval: Duration, on_update: Option<OnUpdate>) -> SyncLoop {
        let shared = Arc::new(Shared {
            state: Mutex::new(LoopState::default()),
            cv: Condvar::new(),
        });
        let worker = shared.clone();
        let thread = std::thread::Builder::new()
            .name("pd-sync".into())
            .spawn(move || loop {
                let (since, flush) = {
                    let Ok(mut st) = worker.state.lock() else { return };
                    let mut due: Option<Instant> = None;
                    loop {
                        if st.stop {
                            return;
                        }
                        if st.requested {
                            break;
                        }
                        if st.snapshot.has_work() {
                            let deadline = *due.get_or_insert_with(|| Instant::now() + interval);
                            let now = Instant::now();
                            if now >= deadline {
                                st.requested = true;
                                st.flush = true;
                                break;
                            }
                            st = match worker.cv.wait_timeout(st, deadline - now) {
                                Ok((g, _)) => g,
                                Err(_) => return,
                            };
                        } else {
                            due = None;
                            st = match worker.cv.wait(st) {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                        }
                    }
                    st.requested = false;
                    st.running = true;
                    (st.snapshot.since_ms, std::mem::take(&mut st.flush))
                };
                let result = runner.run(since, flush);
                let snapshot = {
                    let Ok(mut st) = worker.state.lock() else { return };
                    st.snapshot.apply(result);
                    st.running = false;
                    worker.cv.notify_all();
                    st.snapshot.clone()
                };
                if let Some(f) = &on_update {
                    f(&snapshot);
                }
            })
            .expect("spawning the pd-sync thread");
        SyncLoop {
            shared,
            thread: Some(thread),
        }
    }

    /// Asks for a run (coalesced). `flush` sends queued evaluations and checks pending ones first.
    pub fn request(&self, flush: bool) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.requested = true;
            st.flush |= flush;
            self.shared.cv.notify_all();
        }
    }

    pub fn snapshot(&self) -> SyncSnapshot {
        self.shared.state.lock().map(|s| s.snapshot.clone()).unwrap_or_default()
    }

    /// Waits until no run is in progress or requested (true), or `timeout` passes (false).
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let Ok(mut st) = self.shared.state.lock() else { return false };
        while st.requested || st.running {
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            st = match self.shared.cv.wait_timeout(st, deadline - now) {
                Ok((g, _)) => g,
                Err(_) => return false,
            };
        }
        true
    }
}

impl Drop for SyncLoop {
    fn drop(&mut self) {
        if let Ok(mut st) = self.shared.state.lock() {
            st.stop = true;
            self.shared.cv.notify_all();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Records every run; each run waits for a go signal so a test can pile up requests during it.
    struct Scripted {
        log: Arc<Mutex<Vec<(i64, bool)>>>,
        go: mpsc::Receiver<Result<Value, String>>,
    }

    impl SyncRunner for Scripted {
        fn run(&mut self, since_ms: i64, flush: bool) -> Result<Value> {
            self.log.lock().unwrap().push((since_ms, flush));
            match self.go.recv() {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(anyhow::anyhow!(e)),
                Err(_) => Err(anyhow::anyhow!("test over")),
            }
        }
    }

    fn rows(states: &[(&str, &str, i64)]) -> Value {
        json!({
            "rows": states.iter().map(|(id, st, at)| json!({"request_id": id, "state": st, "updated_at": at})).collect::<Vec<_>>(),
            "next_since_ms": states.iter().map(|s| s.2).max().unwrap_or(0),
            "flush": {"sent": 0, "remaining": 0, "stopped": null, "still_pending": 0, "sync_stopped": null},
        })
    }

    type RunLog = Arc<Mutex<Vec<(i64, bool)>>>;

    fn started(interval: Duration) -> (SyncLoop, RunLog, mpsc::Sender<Result<Value, String>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = mpsc::channel();
        let l = SyncLoop::start(Scripted { log: log.clone(), go: rx }, interval, None);
        (l, log, tx)
    }

    fn wait_for_runs(log: &Mutex<Vec<(i64, bool)>>, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while log.lock().unwrap().len() < n {
            assert!(Instant::now() < deadline, "expected {n} runs, saw {:?}", log.lock().unwrap());
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn requests_during_a_run_share_one_rerun_with_flush_ored() {
        let (l, log, go) = started(Duration::from_secs(3600));
        l.request(false);
        wait_for_runs(&log, 1);
        // Three calls while the first run is in progress, one of them with flush.
        l.request(false);
        l.request(true);
        l.request(false);
        go.send(Ok(rows(&[("a", "stored", 5)]))).unwrap();
        wait_for_runs(&log, 2);
        go.send(Ok(rows(&[("b", "stored", 9)]))).unwrap();
        assert!(l.wait_idle(Duration::from_secs(5)));
        assert_eq!(
            *log.lock().unwrap(),
            vec![(0, false), (5, true)],
            "one rerun, flushing, from the new since"
        );
        let snap = l.snapshot();
        assert_eq!((snap.runs, snap.since_ms, snap.rows.len()), (2, 9, 2));
        assert!(!snap.has_work());
        assert_eq!(snap.status_dot(), Some("stored"));
    }

    #[test]
    fn the_timer_runs_while_rows_wait_and_stops_after() {
        let (l, log, go) = started(Duration::from_millis(30));
        l.request(true);
        wait_for_runs(&log, 1);
        go.send(Ok(rows(&[("a", "queued", 1)]))).unwrap();
        // has_work: the timer fires a flushing run by itself.
        wait_for_runs(&log, 2);
        assert_eq!(log.lock().unwrap()[1], (1, true));
        go.send(Ok(rows(&[("a", "pending", 2)]))).unwrap();
        wait_for_runs(&log, 3);
        go.send(Ok(rows(&[("a", "stored", 3)]))).unwrap();
        assert!(l.wait_idle(Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(120));
        assert_eq!(log.lock().unwrap().len(), 3, "nothing left: no more timer runs");
        assert_eq!(l.snapshot().count("stored"), 1);
    }

    #[test]
    fn errors_keep_the_rows_and_mark_offline() {
        let (l, log, go) = started(Duration::from_secs(3600));
        l.request(true);
        wait_for_runs(&log, 1);
        go.send(Ok(rows(&[("a", "queued", 1)]))).unwrap();
        assert!(l.wait_idle(Duration::from_secs(5)));
        l.request(true);
        wait_for_runs(&log, 2);
        go.send(Err(
            "pd_flush_and_sync: unavailable: spawning premium-discount-app: not found".into()
        ))
        .unwrap();
        assert!(l.wait_idle(Duration::from_secs(5)));
        let snap = l.snapshot();
        assert!(snap.offline);
        assert_eq!(snap.rows.len(), 1, "the cache survives a failed run");
        assert!(snap.last_error.as_deref().unwrap().contains("unavailable"));
        assert!(snap.has_work());

        let mut s = SyncSnapshot::default();
        s.apply(Err(anyhow::anyhow!("pd_flush_and_sync: invalid_input: bad since_ms")));
        assert!(!s.offline, "an argument error is not a connection problem");
        let mut flush = rows(&[]);
        flush["flush"]["stopped"] = json!("offline: no network");
        s.apply(Ok(flush));
        assert!(s.offline);
    }
}
