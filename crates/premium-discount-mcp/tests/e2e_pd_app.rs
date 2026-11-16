//! End to end with the real premium-discount-app (LanceDB outbox, embeddinggemma-2) against
//! premium-discount's fake server (`app/tests/e2e/mock_server.py`). Opt-in: it needs the app binary, the model
//! and a running fake server, so it returns early unless all three are given:
//!
//! ```text
//! PD_E2E_APP=…/premium-discount-app.exe  PD_E2E_MODEL_DIR=…/model  PD_E2E_SERVER=http://127.0.0.1:8787
//! PD_E2E_LOG=…/requests.jsonl (the fake server's request log, optional)
//! cargo test -p premium-discount-mcp --test e2e_pd_app -- --nocapture
//! ```
//!
//! Scenario: offline drop → queued parent + child (`parent_local`) → local-only search with remote_error →
//! back online with moderation pending → one coalesced flush sends both, the child's `ref` is the parent's server
//! id → the 10 s timer (500 ms here) re-checks the `pending` rows until they are `stored`, then stops → a guest
//! vote is stored but not counted, the verdict stays `unrated` → after QR login a vote counts (evidence 1) →
//! search carries the verdict, a `rated` filter returns only judged evaluations.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use premium_discount_mcp::{PdSyncClient, SyncLoop};
use serde_json::{json, Value};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn spawn(app: &str, model: &str, data: &Path, server: &str) -> PdSyncClient {
    let args: Vec<String> = [
        "--pd-app",
        app,
        "--model-dir",
        model,
        "--data-dir",
        data.to_str().unwrap(),
        "--server-url",
        server,
        "--idle-secs",
        "0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    PdSyncClient::spawn(Path::new(env!("CARGO_BIN_EXE_premium-discount-mcp")), &args).unwrap()
}

/// `POST /__control` on the fake server (`done: false` keeps evaluations `pending`).
fn control(server: &str, body: Value) {
    ureq::post(&format!("{server}/__control")).send_string(&body.to_string()).unwrap();
}

fn wait_until(mut ok: impl FnMut() -> bool, log: &Mutex<Vec<(u64, usize, usize, usize)>>) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !ok() {
        assert!(
            Instant::now() < deadline,
            "timed out; runs (runs, queued, pending, stored): {:?}",
            log.lock().unwrap()
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn offline_drop_then_coalesced_flush_against_the_real_app() {
    let (Some(app), Some(model), Some(server)) = (env("PD_E2E_APP"), env("PD_E2E_MODEL_DIR"), env("PD_E2E_SERVER")) else {
        eprintln!("skipped: set PD_E2E_APP, PD_E2E_MODEL_DIR and PD_E2E_SERVER to run");
        return;
    };
    let data: PathBuf = std::env::temp_dir().join(format!("zena-pd-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let t0 = Instant::now();

    // 1. Offline (nothing listens on port 9): both drops stay in the outbox, the child chained by parent_local.
    let mut off = spawn(&app, &model, &data, "http://127.0.0.1:9");
    let st = off.call("pd_model_manage", json!({"action": "status"})).unwrap();
    eprintln!("model: installed={} app={}", st["installed"], st["app"]);
    let parent = off
        .enqueue(json!({
            "text": "SurfaceError::Outdated 는 창 크기를 바꾼 직후 surface 를 다시 configure 해서 해결",
            "cc": "app://Code.exe/coding",
            "to_key": "raon-zena — overlay.rs",
            "bcc_key": "wgpu error: SurfaceError::Outdated",
        }))
        .unwrap();
    eprintln!("parent (offline): {parent} [{:?}]", t0.elapsed());
    assert_eq!(parent["status"], "queued");
    assert_eq!(parent["cc"], "code.exe");
    let parent_local = parent["local_id"].as_str().unwrap().to_string();
    let child = off
        .enqueue(json!({
            "text": "같은 오류가 최소화 뒤 복귀할 때도 나온다: 크기가 0 이면 configure 를 건너뛴다",
            "cc": "app://Code.exe/coding",
            "parent_local": parent_local,
        }))
        .unwrap();
    assert_eq!(child["status"], "queued", "{child}");
    let found = off
        .search_merged(json!({"query": "SurfaceError Outdated 해결", "cc": "app://Code.exe"}))
        .unwrap();
    eprintln!("offline search: {found}");
    assert!(found["remote_error"].is_string(), "offline: remote_error is set");
    let first = &found["matches"][0];
    assert_eq!((first["source"].as_str(), first["state"].as_str()), (Some("local"), Some("queued")));
    assert!(first["text"].as_str().unwrap().contains("SurfaceError"));
    let flushed = off.flush_and_sync(0, true).unwrap();
    assert!(flushed["flush"]["stopped"].is_string(), "{flushed}");
    assert_eq!(flushed["counts"]["queued"], 2);
    drop(off);

    // 2. Online, the fake server's moderation not done yet: the SyncLoop sends both (requests made before the loop
    //    wakes merge into one flushing run), the rows wait as `pending`, and the timer alone checks them again
    //    until the server says `stored`.
    control(&server, json!({"done": false}));
    let client = Arc::new(Mutex::new(spawn(&app, &model, &data, &server)));
    let updates = Arc::new(Mutex::new(Vec::new()));
    let seen = updates.clone();
    let sync = SyncLoop::start(
        client.clone(),
        Duration::from_millis(500),
        Some(Box::new(move |s| {
            seen.lock()
                .unwrap()
                .push((s.runs, s.count("queued"), s.count("pending"), s.count("stored")))
        })),
    );
    sync.request(false);
    sync.request(true);
    sync.request(false);
    wait_until(|| sync.snapshot().count("pending") == 2, &updates);
    let runs_pending = sync.snapshot().runs;
    wait_until(|| sync.snapshot().runs >= runs_pending + 2, &updates);
    assert_eq!(sync.snapshot().count("pending"), 2, "still under review");
    control(&server, json!({"done": true}));
    wait_until(|| sync.snapshot().count("stored") == 2, &updates);
    let settled = sync.snapshot().runs;
    std::thread::sleep(Duration::from_millis(1500));
    assert_eq!(sync.snapshot().runs, settled, "nothing left: the timer stopped");
    assert!(sync.wait_idle(Duration::from_secs(30)));
    let snap = sync.snapshot();
    eprintln!("sync runs: {:?} [{:?}]", updates.lock().unwrap(), t0.elapsed());
    assert!(!snap.offline && snap.last_error.is_none(), "{:?}", snap.last_error);
    let rows: Vec<&Value> = snap.rows.values().collect();
    let parent_row = rows.iter().find(|r| r["data"]["local_id"] == parent_local.as_str()).unwrap();
    let child_row = rows.iter().find(|r| r["data"]["parent_local"] == parent_local.as_str()).unwrap();
    assert_eq!(
        child_row["ref"], parent_row["id"],
        "the child's parent_local became the parent's server id"
    );
    assert_eq!(parent_row["bcc"].as_str().unwrap().len(), 42);
    drop(sync);

    let mut c = client.lock().unwrap();
    let found = c
        .search_merged(json!({"query": "SurfaceError Outdated 해결", "cc": "code.exe"}))
        .unwrap();
    eprintln!("online search: {found}");
    assert!(found["remote_error"].is_null());
    assert_eq!(found["matches"][0]["source"], "both");
    assert_eq!(found["matches"][0]["verdict"]["tier"], "unrated", "no votes yet");

    // 3. Votes: a guest vote is stored but carries no weight; after QR login the same vote counts.
    let parent_id = parent_row["id"].as_str().unwrap().to_string();
    let child_id = child_row["id"].as_str().unwrap().to_string();
    let guest = c.call("pd_vote", json!({"id": parent_id, "value": 1})).unwrap();
    assert_eq!(
        (guest["status"].as_str(), guest["counted"].as_bool()),
        (Some("sent"), Some(false)),
        "{guest}"
    );
    let v = c
        .call("pd_verdicts", json!({"ids": [parent_id, child_id], "refresh": true}))
        .unwrap();
    eprintln!("verdicts (guest vote): {v}");
    assert_eq!(v["verdicts"][0]["tier"], "unrated");
    assert_eq!(v["verdicts"][0]["evidence"], 0.0);
    assert_eq!(v["verdicts"][0]["reasons"][0], "low_evidence");

    let login = c.call("pd_auth_session", json!({"action": "start"})).unwrap();
    assert!(login["mailto"].as_str().unwrap().starts_with("mailto:"));
    assert_eq!(c.call("pd_auth_session", json!({"action": "poll"})).unwrap()["state"], "pending");
    control(&server, json!({"verified": true}));
    assert_eq!(c.call("pd_auth_session", json!({"action": "poll"})).unwrap()["state"], "signed_in");
    let signed = c.call("pd_vote", json!({"id": child_id, "value": -1})).unwrap();
    assert_eq!(signed["counted"], true, "{signed}");
    let v = c.call("pd_verdicts", json!({"ids": [child_id], "refresh": true})).unwrap();
    eprintln!("verdicts (signed-in vote): {v}");
    assert_eq!(v["verdicts"][0]["down"], 1);
    assert!((v["verdicts"][0]["evidence"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert_eq!(v["verdicts"][0]["tier"], "unrated", "one vote is below min_evidence 2");
    // The cache answers without the server for 10 minutes.
    let cached = c.call("pd_verdicts", json!({"ids": [child_id]})).unwrap();
    assert_eq!(
        (cached["stale"].as_bool(), cached["verdicts"][0]["down"].as_u64()),
        (Some(false), Some(1))
    );

    let found = c
        .search_merged(json!({"query": "SurfaceError Outdated 해결", "cc": "code.exe"}))
        .unwrap();
    let mine = found["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == parent_id.as_str())
        .unwrap();
    assert_eq!(mine["verdict"]["tier"], "unrated");
    let rated = c
        .search_merged(json!({"query": "SurfaceError Outdated 해결", "tier": "rated"}))
        .unwrap();
    assert!(rated["matches"].as_array().unwrap().is_empty(), "nothing is judged yet: {rated}");

    if let Some(log) = env("PD_E2E_LOG") {
        let sent: Vec<Value> = std::fs::read_to_string(log)
            .unwrap()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .filter(|v: &Value| {
                v["cc"] == "code.exe"
                    && v["text"]
                        .as_str()
                        .is_some_and(|t| t.contains("SurfaceError") || t.contains("최소화"))
            })
            .collect();
        eprintln!("server saw: {sent:?}");
        let p = sent
            .iter()
            .rev()
            .find(|v| v["text"].as_str().unwrap().contains("SurfaceError"))
            .unwrap();
        let ch = sent.iter().rev().find(|v| v["text"].as_str().unwrap().contains("최소화")).unwrap();
        assert_eq!(ch["ref"], p["id"]);
        assert_eq!(p["vector_len"], 768);
    }
    drop(c);
    drop(client);
    let _ = std::fs::remove_dir_all(&data);
}
