//! The tools against a scripted premium-discount-app, and the real binary over MCP.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use premium_discount_mcp::address::{content_address, hash_id};
use premium_discount_mcp::{error_kind, Backend, CallError, PdSyncServer, TOOLS};
use serde_json::{json, Value};
use zena_avatar_mcp::protocol::{result_json, McpClient, ToolHandler};

const ID: &str = "0190f0a0-0000-7000-8000-0000000000aa";

/// premium-discount-app stand-in: records `(cmd, args)` and answers from a script.
#[derive(Clone, Default)]
struct Scripted {
    calls: Arc<Mutex<Vec<(String, Value)>>>,
    replies: Arc<Mutex<VecDeque<Result<Value, CallError>>>>,
}

impl Scripted {
    fn reply(&self, r: Result<Value, CallError>) {
        self.replies.lock().unwrap().push_back(r);
    }
    fn calls(&self) -> Vec<(String, Value)> {
        self.calls.lock().unwrap().clone()
    }
}

impl Backend for Scripted {
    fn call(&mut self, cmd: &str, args: Value) -> Result<Value, CallError> {
        self.calls.lock().unwrap().push((cmd.to_string(), args));
        self.replies.lock().unwrap().pop_front().unwrap_or(Ok(Value::Null))
    }
}

fn server() -> (PdSyncServer<Scripted>, Scripted) {
    let app = Scripted::default();
    (PdSyncServer::new(app.clone(), "http://127.0.0.1:9"), app)
}

#[test]
fn enqueue_maps_the_drop_envelope_onto_submit() {
    let (mut s, app) = server();
    app.reply(Ok(json!({"status": "queued", "local_id": "local-7"})));
    let out = s
        .call(
            "pd_enqueue_node",
            &json!({
                "text": "  보스 패턴: 빙결 회피가 우세  ",
                "cc": "app://EldenRing.exe/boss_fight",
                "to_key": "ELDEN RING™",
                "bcc_key": "Frost  breath\n— dodge left",
                "parent_local": "local-6",
            }),
        )
        .unwrap();
    let (cmd, args) = &app.calls()[0];
    assert_eq!(cmd, "cmd_submit_evaluation");
    assert_eq!(args["cc"], "eldenring.exe");
    assert_eq!(args["to"], hash_id("ELDEN RING™").unwrap());
    assert_eq!(args["bcc"], content_address("Frost breath — dodge left").unwrap());
    assert_eq!(args["local_parent"], "local-6");
    assert_eq!(args["ref"], Value::Null);
    assert_eq!(out["status"], "queued");
    assert_eq!(out["local_id"], "local-7");
    assert_eq!((out["cc_kind"].as_str(), out["scene"].as_str()), (Some("app"), Some("boss-fight")));

    // Rows come back as premium-discount-app wrote them; pending votes keep the timer running.
    app.reply(Ok(
        json!({"sent": 1, "remaining": 0, "still_pending": 0, "votes_remaining": 1, "stopped": null}),
    ));
    app.reply(Ok(json!([
        {"request_id": "r1", "id": ID, "state": "stored", "updated_at": 42, "data": {"local_id": "local-7"}},
        {"request_id": "r2", "id": "", "state": "queued", "updated_at": 40, "data": {"local_id": "local-8"}},
    ])));
    let synced = s.call("pd_flush_and_sync", &json!({"since_ms": 10})).unwrap();
    let calls = app.calls();
    assert_eq!(calls[1].0, "cmd_flush_outbox");
    assert_eq!(calls[2], ("cmd_sync_cache".to_string(), json!({"since": 10})));
    assert_eq!(synced["next_since_ms"], 42);
    assert_eq!(synced["has_work"], true, "a queued vote is work too");
    assert_eq!(synced["counts"], json!({"queued": 1, "stored": 1}));
}

#[test]
fn search_passes_the_tier_filter_and_returns_server_verdicts() {
    let (mut s, app) = server();
    app.reply(Ok(json!({"matches": [
        {"id": ID, "score": 0.9, "source": "remote", "verdict": {"tier": "premium", "grade": 2, "rank": 0.97, "evidence": 3.0}, "my_vote": 1},
    ], "remote_error": null})));
    let found = s
        .call(
            "pd_search_merged",
            &json!({"query": "빙결", "cc": "https://www.eldenring.exe/", "tier": "premium"}),
        )
        .unwrap();
    let args = &app.calls()[0].1;
    assert_eq!(
        (args["cc"].as_str(), args["tier"].as_str(), args["top_k"].as_u64()),
        (Some("eldenring.exe"), Some("premium"), Some(10))
    );
    assert_eq!(found["matches"][0]["verdict"]["tier"], "premium");
    assert_eq!(found["matches"][0]["my_vote"], 1);
    assert_eq!(found["cc_kind"], "web");

    let e = format!(
        "{:#}",
        s.call("pd_search_merged", &json!({"query": "q", "tier": "gold"})).unwrap_err()
    );
    assert_eq!(error_kind(&e), Some("invalid_input"));
}

#[test]
fn votes_and_verdicts_map_onto_the_app_commands() {
    let (mut s, app) = server();
    app.reply(Ok(json!({"status": "sent", "request_id": "r", "counted": true})));
    let out = s.call("pd_vote", &json!({"id": ID, "value": -1})).unwrap();
    assert_eq!(
        app.calls()[0],
        ("cmd_vote".to_string(), json!({"id": ID, "value": -1, "kind": "vote"}))
    );
    assert_eq!(out["status"], "sent");

    app.reply(Ok(
        json!({"verdicts": [{"id": ID, "tier": "discount", "grade": -1}], "missing": [], "stale": false, "remote_error": null}),
    ));
    let out = s.call("pd_verdicts", &json!({"ids": [ID], "refresh": true})).unwrap();
    assert_eq!(app.calls()[1], ("cmd_verdicts".to_string(), json!({"ids": [ID], "refresh": true})));
    assert_eq!(out["verdicts"][0]["tier"], "discount");

    let n = app.calls().len();
    for (tool, args) in [
        ("pd_vote", json!({"id": ID})),
        ("pd_vote", json!({"id": ID, "value": 2})),
        ("pd_vote", json!({"id": ID, "value": 1, "kind": "like"})),
        ("pd_vote", json!({"value": 1})),
        ("pd_verdicts", json!({})),
        ("pd_verdicts", json!({"ids": []})),
        ("pd_verdicts", json!({"ids": [1, 2]})),
    ] {
        let e = format!("{:#}", s.call(tool, &args).unwrap_err());
        assert_eq!(error_kind(&e), Some("invalid_input"), "{tool} {args}: {e}");
    }
    assert_eq!(app.calls().len(), n, "nothing refused here reaches the app");

    // The app's own refusal keeps its kind (an evaluation the server does not have).
    app.reply(Err(CallError::Command {
        kind: "rejected".into(),
        message: "server returned 404: evaluation not found".into(),
    }));
    let e = format!("{:#}", s.call("pd_vote", &json!({"id": ID, "value": 1})).unwrap_err());
    assert_eq!(error_kind(&e), Some("rejected"));
}

#[test]
fn defaults_and_refusals() {
    let (mut s, app) = server();
    app.reply(Ok(
        json!({"status": "saved", "id": ID, "root": ID, "ref": null, "from": "0x00000000000000000000000000000000000000ab"}),
    ));
    s.call(
        "pd_enqueue_node",
        &json!({"text": "SurfaceError: Outdated", "cc": "store.steampowered.com"}),
    )
    .unwrap();
    let args = &app.calls()[0].1;
    assert_eq!(
        args["bcc"],
        content_address("SurfaceError: Outdated").unwrap(),
        "bcc defaults to the text's address"
    );
    assert_eq!(args["to"], Value::Null);

    app.reply(Ok(json!({"status": "blocked"})));
    s.call("pd_enqueue_node", &json!({"text": "x", "cc": "a.com", "bcc": "none"}))
        .unwrap();
    assert_eq!(app.calls()[1].1["bcc"], Value::Null);

    let n = app.calls().len();
    for (tool, args) in [
        ("pd_enqueue_node", json!({"text": "x", "cc": "app://"})),
        ("pd_enqueue_node", json!({"text": "x", "cc": "localhost"})),
        (
            "pd_enqueue_node",
            json!({"text": "x", "cc": "a.com", "ref": ID, "parent_local": "local-1"}),
        ),
        ("pd_enqueue_node", json!({"text": "x", "cc": "a.com", "to": "0x12"})),
        ("pd_enqueue_node", json!({"cc": "a.com"})),
        ("pd_graph_inspect", json!({"node_id": "nope"})),
        ("pd_auth_session", json!({"action": "hack"})),
        ("pd_model_manage", json!({"action": "download", "variant": "huge"})),
    ] {
        let e = format!("{:#}", s.call(tool, &args).unwrap_err());
        assert_eq!(error_kind(&e), Some("invalid_input"), "{tool} {args}: {e}");
    }
    assert_eq!(app.calls().len(), n, "nothing refused here reaches the app");

    app.reply(Err(CallError::Command {
        kind: "not_ready".into(),
        message: "model missing".into(),
    }));
    let e = format!("{:#}", s.call("pd_search_merged", &json!({"query": "q"})).unwrap_err());
    assert_eq!(e, "not_ready: model missing");
    app.reply(Err(CallError::Transport("spawning premium-discount-app: not found".into())));
    let e = format!("{:#}", s.call("pd_flush_and_sync", &json!({})).unwrap_err());
    assert_eq!(error_kind(&e), Some("unavailable"));
}

#[test]
fn session_and_model_actions() {
    let (mut s, app) = server();
    for (tool, action, cmd) in [
        ("pd_auth_session", "status", "cmd_session"),
        ("pd_auth_session", "start", "cmd_login_start"),
        ("pd_auth_session", "poll", "cmd_login_status"),
        ("pd_auth_session", "logout", "cmd_logout"),
        ("pd_model_manage", "download", "cmd_model_download"),
        ("pd_model_manage", "cancel", "cmd_model_download_cancel"),
        ("pd_model_manage", "status", "cmd_model_status"),
    ] {
        app.reply(Ok(json!({"state": "x"})));
        s.call(tool, &json!({"action": action, "verify": true})).unwrap();
        assert_eq!(app.calls().last().unwrap().0, cmd);
    }
    let calls = app.calls();
    assert_eq!(calls[0].1, json!({"verify": true}));
    assert_eq!(calls[4].1, json!({"variant": "text"}));
    let (mut s, app) = server();
    app.reply(Ok(json!({"installed": "none"})));
    let st = s.call("pd_model_manage", &json!({"action": "status"})).unwrap();
    assert_eq!(st["installed"], "none");
    assert_eq!(st["app"], Value::Null, "the scripted backend has no process");
}

/// One-shot HTTP server answering `GET /graph…`.
fn graph_server(body: Value) -> (String, std::thread::JoinHandle<String>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let body = body.to_string();
    let h = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut r = BufReader::new(sock.try_clone().unwrap());
        let mut first = String::new();
        r.read_line(&mut first).unwrap();
        loop {
            let mut l = String::new();
            if r.read_line(&mut l).unwrap() == 0 || l == "\r\n" {
                break;
            }
        }
        write!(
            sock,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        first
    });
    (base, h)
}

#[test]
fn the_binary_speaks_mcp_without_premium_discount_app() {
    let home = std::env::temp_dir().join(format!("premium-discount-mcp-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&home);
    let graph = json!({
        "nodes": [
            {"id": ID, "type": "evaluation", "label": "평가", "cc": "code.exe", "root": ID, "tier": "premium", "grade": 1},
            {"id": "author:0123456789ab", "type": "author", "label": "작성자 012345"},
        ],
        "edges": [{"source": "author:0123456789ab", "target": ID, "type": "authored"}],
        "summary": {"premium": 1, "neutral": 0, "discount": 0, "unrated": 0},
    });
    let (base, http) = graph_server(graph);
    let args: Vec<String> = [
        "--pd-app",
        "definitely-not-premium-discount-app-xyz",
        "--home",
        home.to_str().unwrap(),
        "--server-url",
        &base,
        "--idle-secs",
        "1",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut c = McpClient::spawn(Path::new(env!("CARGO_BIN_EXE_premium-discount-mcp")), &args).unwrap();
    assert_eq!(c.server_name(), "premium-discount-mcp");
    assert_eq!(c.list_tools().unwrap(), TOOLS.to_vec());

    // No premium-discount-app: a tool error with kind `unavailable`, and the server keeps running.
    let e = format!(
        "{:#}",
        c.call_tool("pd_vote", json!({"id": ID, "value": 1}), Duration::from_secs(30))
            .unwrap_err()
    );
    assert!(e.starts_with("pd_vote: unavailable: spawning"), "{e}");
    let e = format!(
        "{:#}",
        c.call_tool("pd_enqueue_node", json!({"text": "x", "cc": "app://"}), Duration::from_secs(30))
            .unwrap_err()
    );
    assert!(e.contains("invalid_input"), "{e}");

    // The graph does not need the app.
    let g = result_json(
        &c.call_tool("pd_graph_inspect", json!({"cc": "app://Code.exe/coding"}), Duration::from_secs(30))
            .unwrap(),
    );
    assert_eq!(g["path"], "/graph?cc=code.exe");
    assert_eq!(g["summary"]["authors"], 1);
    assert_eq!(g["summary"]["tiers"]["premium"], 1);
    assert_eq!(g["summary"]["threads"][0]["root"], ID);
    assert!(http.join().unwrap().starts_with("GET /graph?cc=code.exe "));
    drop(c);
    let _ = std::fs::remove_dir_all(&home);
}
