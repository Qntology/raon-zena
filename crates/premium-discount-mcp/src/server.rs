//! The MCP tools: premium-discount-app's 16 line commands regrouped for Raon Zena.
//!
//! | tool | premium-discount-app command(s) |
//! |---|---|
//! | `pd_enqueue_node` | `cmd_submit_evaluation` (local filter → embedding → LanceDB row → send) |
//! | `pd_flush_and_sync` | `cmd_flush_outbox` (evaluations and votes) + `cmd_sync_cache` |
//! | `pd_search_merged` | `cmd_search_merged` (with the server's verdict and my vote on each match) |
//! | `pd_vote` | `cmd_vote` |
//! | `pd_verdicts` | `cmd_verdicts` |
//! | `pd_graph_inspect` | none: `GET /graph`, `/graph/local`, `/graph/similar` on the server |
//! | `pd_auth_session` | `cmd_session`, `cmd_login_start`, `cmd_login_status`, `cmd_logout` |
//! | `pd_model_manage` | `cmd_model_status`, `cmd_model_download`, `cmd_model_download_cancel` |
//!
//! The schema is premium-discount's: argument and result field names are its own, and recommendations
//! (`premium` / `discount`) are the server's verdicts (rule: premium-discount `contracts/verdict_v1.json`).
//! This layer only adds what Raon Zena's places need: `cc` from `app://exe/scene` and URLs
//! ([`crate::envelope`]), `to` / `bcc` from keys with the web client's `hashId` ([`crate::address`]), and the graph.
//! Tool errors read `<kind>: <message>`, with premium-discount-app's error kinds (`offline`, `not_ready`,
//! `invalid_input`, `rejected`, `queue_full` …) and `unavailable` when the app cannot be started or reached.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use zena_avatar_mcp::protocol::ToolHandler;

use crate::address::{content_address, hash_id, normalize_address};
use crate::bridge::Backend;
use crate::envelope::{normalize_cc, Cc};
use crate::graph::{query_of, summarize, GraphClient};
use crate::lines::CallError;

pub const SERVER_NAME: &str = "premium-discount-mcp";

pub const TOOLS: [&str; 8] = [
    "pd_enqueue_node",
    "pd_flush_and_sync",
    "pd_search_merged",
    "pd_vote",
    "pd_verdicts",
    "pd_graph_inspect",
    "pd_auth_session",
    "pd_model_manage",
];

/// Error kinds a tool error starts with (premium-discount-app's `CommandError.kind`, plus `unavailable`).
pub const ERROR_KINDS: [&str; 15] = [
    "invalid_input",
    "embed",
    "filter",
    "offline",
    "http",
    "bad_response",
    "server",
    "rejected",
    "queue_full",
    "storage",
    "not_ready",
    "unknown_command",
    "internal",
    "unavailable",
    "unknown",
];

/// The kind of a tool error message (`<kind>: <message>`).
pub fn error_kind(message: &str) -> Option<&str> {
    let (kind, _) = message.split_once(": ")?;
    ERROR_KINDS.contains(&kind).then_some(kind)
}

fn has_kind(message: &str) -> bool {
    error_kind(message).is_some()
}

/// Search tier filters (premium-discount `TIER_FILTERS`).
pub const TIER_FILTERS: [&str; 3] = ["premium", "discount", "rated"];

pub struct PdSyncServer<B: Backend> {
    backend: B,
    graph: GraphClient,
}

fn opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
}

fn req_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    opt_str(args, key).with_context(|| format!("missing `{key}`"))
}

fn call(backend: &mut impl Backend, cmd: &str, args: Value) -> Result<Value> {
    backend.call(cmd, args).map_err(|e: CallError| anyhow!("{e}"))
}

/// `address` given directly, else `hash_id(key)`.
fn address_or_key(args: &Value, field: &str, key_field: &str) -> Result<Option<String>> {
    match (opt_str(args, field), opt_str(args, key_field)) {
        (Some(_), Some(_)) => bail!("give `{field}` or `{key_field}`, not both"),
        (Some(a), None) => Ok(Some(normalize_address(field, a)?)),
        (None, Some(k)) => Ok(Some(hash_id(k)?)),
        (None, None) => Ok(None),
    }
}

fn cc_fields(cc: &Cc) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("cc".into(), Value::from(cc.cc.clone()));
    m.insert("cc_kind".into(), Value::from(cc.kind.name()));
    if let Some(s) = &cc.scene {
        m.insert("scene".into(), Value::from(s.clone()));
    }
    m
}

fn merge(mut base: Value, extra: Map<String, Value>) -> Value {
    if let Some(obj) = base.as_object_mut() {
        obj.extend(extra);
    }
    base
}

impl<B: Backend> PdSyncServer<B> {
    pub fn new(backend: B, server_url: &str) -> PdSyncServer<B> {
        PdSyncServer {
            backend,
            graph: GraphClient::new(server_url),
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn enqueue(&mut self, args: &Value) -> Result<Value> {
        let text = req_str(args, "text")?;
        let cc = normalize_cc(req_str(args, "cc")?)?;
        let to = address_or_key(args, "to", "to_key")?;
        let bcc = match (opt_str(args, "bcc"), opt_str(args, "bcc_key")) {
            (Some("none"), None) => None,
            (Some(_), Some(_)) => bail!("give `bcc` or `bcc_key`, not both"),
            (Some(a), None) => Some(normalize_address("bcc", a)?),
            (None, Some(k)) => Some(content_address(k)?),
            (None, None) => Some(content_address(text)?),
        };
        let from = opt_str(args, "from").map(|a| normalize_address("from", a)).transpose()?;
        let (ref_, parent) = (opt_str(args, "ref"), opt_str(args, "parent_local"));
        if ref_.is_some() && parent.is_some() {
            bail!("invalid_input: give `ref` (parent on the server) or `parent_local` (parent in the outbox), not both");
        }
        let result = call(
            &mut self.backend,
            "cmd_submit_evaluation",
            json!({"text": text, "cc": cc.cc, "from": from, "to": to, "bcc": bcc, "ref": ref_, "local_parent": parent}),
        )?;
        let mut extra = cc_fields(&cc);
        extra.insert("to".into(), json!(to));
        extra.insert("bcc".into(), json!(bcc));
        Ok(merge(result, extra))
    }

    fn flush_and_sync(&mut self, args: &Value) -> Result<Value> {
        let since = args.get("since_ms").and_then(Value::as_i64).unwrap_or(0).max(0);
        let flush = args.get("flush").and_then(Value::as_bool).unwrap_or(true);
        let summary = if flush {
            Some(call(&mut self.backend, "cmd_flush_outbox", Value::Null)?)
        } else {
            None
        };
        let rows = match call(&mut self.backend, "cmd_sync_cache", json!({"since": since}))? {
            Value::Array(r) => r,
            other => bail!("bad_response: cmd_sync_cache returned {other}"),
        };
        let next = rows
            .iter()
            .filter_map(|r| r.get("updated_at").and_then(Value::as_i64))
            .max()
            .unwrap_or(since)
            .max(since);
        let mut counts: Map<String, Value> = Map::new();
        for r in &rows {
            let state = r.get("state").and_then(Value::as_str).unwrap_or("unknown").to_string();
            let n = counts.get(&state).and_then(Value::as_u64).unwrap_or(0) + 1;
            counts.insert(state, Value::from(n));
        }
        let has_work = summary.as_ref().map(|s| {
            let n = |k: &str| s.get(k).and_then(Value::as_u64).unwrap_or(0);
            n("remaining") + n("still_pending") + n("votes_remaining") > 0
        });
        Ok(json!({"flush": summary, "rows": rows, "next_since_ms": next, "counts": counts, "has_work": has_work}))
    }

    fn search(&mut self, args: &Value) -> Result<Value> {
        let query = req_str(args, "query")?;
        let cc = opt_str(args, "cc").map(normalize_cc).transpose()?;
        let to = address_or_key(args, "to", "to_key")?;
        let top_k = args.get("top_k").and_then(Value::as_u64).unwrap_or(10);
        let tier = opt_str(args, "tier");
        if let Some(t) = tier {
            if !TIER_FILTERS.contains(&t) {
                bail!("tier must be premium, discount or rated (got {t})");
            }
        }
        let found = call(
            &mut self.backend,
            "cmd_search_merged",
            json!({
                "query": query,
                "top_k": top_k,
                "cc": cc.as_ref().map(|c| c.cc.clone()),
                "to": to,
                "root": opt_str(args, "root"),
                "ref": opt_str(args, "ref"),
                "tier": tier,
            }),
        )?;
        Ok(merge(found, cc.as_ref().map(cc_fields).unwrap_or_default()))
    }

    fn vote(&mut self, args: &Value) -> Result<Value> {
        let id = req_str(args, "id")?;
        let value = args.get("value").and_then(Value::as_i64).context("missing `value` (1, -1 or 0)")?;
        if !matches!(value, -1..=1) {
            bail!("value must be 1, -1 or 0 (got {value})");
        }
        let kind = opt_str(args, "kind").unwrap_or("vote");
        if !matches!(kind, "vote" | "vote_outcome") {
            bail!("kind must be vote or vote_outcome (got {kind})");
        }
        call(&mut self.backend, "cmd_vote", json!({"id": id, "value": value, "kind": kind}))
    }

    fn verdicts(&mut self, args: &Value) -> Result<Value> {
        let ids: Vec<&str> = args
            .get("ids")
            .and_then(Value::as_array)
            .context("missing `ids` (an array of evaluation ids)")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .context("`ids` must hold strings")
            })
            .collect::<Result<_>>()?;
        if ids.is_empty() {
            bail!("`ids` is empty");
        }
        let refresh = args.get("refresh").and_then(Value::as_bool).unwrap_or(false);
        call(&mut self.backend, "cmd_verdicts", json!({"ids": ids, "refresh": refresh}))
    }

    fn graph(&mut self, args: &Value) -> Result<Value> {
        let cc = opt_str(args, "cc").map(normalize_cc).transpose()?;
        let q = query_of(
            cc.as_ref().map(|c| c.cc.clone()),
            opt_str(args, "node_id").map(str::to_string),
            opt_str(args, "mode"),
            args.get("top_k").and_then(Value::as_u64),
        )?;
        let graph = self.graph.get(&q).map_err(|e| anyhow!("{e}"))?;
        let summary = summarize(&graph);
        Ok(json!({"path": q.path(), "summary": summary, "graph": graph}))
    }

    fn auth(&mut self, args: &Value) -> Result<Value> {
        match req_str(args, "action")? {
            "status" => {
                let verify = args.get("verify").and_then(Value::as_bool).unwrap_or(false);
                call(&mut self.backend, "cmd_session", json!({"verify": verify}))
            }
            "start" => call(&mut self.backend, "cmd_login_start", Value::Null),
            "poll" => call(&mut self.backend, "cmd_login_status", Value::Null),
            "logout" => call(&mut self.backend, "cmd_logout", Value::Null),
            other => bail!("action must be status, start, poll or logout (got {other})"),
        }
    }

    fn model(&mut self, args: &Value) -> Result<Value> {
        match req_str(args, "action")? {
            "status" => {
                let status = call(&mut self.backend, "cmd_model_status", Value::Null)?;
                Ok(merge(status, Map::from_iter([("app".to_string(), self.backend.state())])))
            }
            "download" => {
                let variant = opt_str(args, "variant").unwrap_or("text");
                if !matches!(variant, "text" | "full") {
                    bail!("variant must be text or full (got {variant})");
                }
                call(&mut self.backend, "cmd_model_download", json!({"variant": variant}))
            }
            "cancel" => call(&mut self.backend, "cmd_model_download_cancel", Value::Null),
            other => bail!("action must be status, download or cancel (got {other})"),
        }
    }
}

fn s(desc: &str) -> Value {
    json!({"type": "string", "description": desc})
}

impl<B: Backend> ToolHandler for PdSyncServer<B> {
    fn tools(&self) -> Vec<Value> {
        let cc = s("Domain (store.steampowered.com), web URL (host is used) or desktop program app://<exe>[/<scene>] (→ <exe>.exe; the scene is returned, not sent)");
        let address = |what: &str| s(&format!("{what}: 0x + 40 hex"));
        vec![
            json!({
                "name": "pd_enqueue_node",
                "description": "Record one evaluation in the local premium-discount outbox (LanceDB, original text kept) and send it when online. Profanity is blocked locally (status blocked: nothing stored). Returns status saved | accepted (server checks it) | queued (offline or behind earlier items) with id or local_id.",
                "inputSchema": {"type": "object", "properties": {
                    "text": s("Text to store (at most 8192 bytes): the OCR / translation summary"),
                    "cc": cc,
                    "to": address("Target address"),
                    "to_key": s("Target key (window title, Relay Key slug); hashed to `to` with hashId"),
                    "bcc": address("Content address, or \"none\" to send none"),
                    "bcc_key": s("Content to address (the region text read at the drop); default: `text`. Normalized (NFC, whitespace) then hashId"),
                    "ref": s("Parent evaluation UUID on the server"),
                    "parent_local": s("Parent's outbox id (local-N) while it has no server id; resolved to `ref` when it is sent"),
                    "from": address("Author address (the server prefers the signed-in session)"),
                }, "required": ["text", "cc"]},
            }),
            json!({
                "name": "pd_flush_and_sync",
                "description": "Send queued evaluations and votes, check pending evaluations (stored / rejected / missing), then return local rows changed after since_ms (original text and state). Offline: flush.stopped / votes_stopped say why, rows still come back.",
                "inputSchema": {"type": "object", "properties": {
                    "since_ms": {"type": "integer", "minimum": 0, "default": 0, "description": "Last next_since_ms (0: every row)"},
                    "flush": {"type": "boolean", "default": true, "description": "Send and check before reading rows"},
                }},
            }),
            json!({
                "name": "pd_search_merged",
                "description": "One embedding, two searches: my local outbox (original text) and the server's index (other users' threads), merged by server id (source local | remote | both). Server matches carry the global verdict (tier premium | neutral | discount | unrated, grade -2..2) and my_vote. Offline: local matches only, with remote_error. A root / ref / to / tier filter returns server matches only.",
                "inputSchema": {"type": "object", "properties": {
                    "query": s("Search text (a question, or the text read at a drop)"),
                    "cc": cc,
                    "to": address("Target filter"),
                    "to_key": s("Target key, hashed to `to`"),
                    "root": s("Thread root UUID filter"),
                    "ref": s("Parent UUID filter"),
                    "tier": {"type": "string", "enum": TIER_FILTERS, "description": "Only recommended (premium), not recommended (discount) or any judged (rated) evaluations"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                }, "required": ["query"]},
            }),
            json!({
                "name": "pd_vote",
                "description": "Recommend (1) or not recommend (-1) another user's evaluation, or withdraw (0). Written to the local vote outbox first and sent at once (status sent with counted, or queued offline). Only signed-in votes on other people's evaluations count; guest and own votes are stored but carry no weight.",
                "inputSchema": {"type": "object", "properties": {
                    "id": s("Evaluation UUID on the server"),
                    "value": {"type": "integer", "enum": [1, -1, 0]},
                    "kind": {"type": "string", "enum": ["vote", "vote_outcome"], "default": "vote", "description": "vote_outcome: a vote confirmed by an outcome (send only with the user's consent)"},
                }, "required": ["id", "value"]},
            }),
            json!({
                "name": "pd_verdicts",
                "description": "Global verdicts of evaluations (PD-Verdict: tier, grade -2..2, rank, value, evidence, reason codes). Served from the local cache when it is under 10 minutes old, else asked from the server; offline: the cache with stale=true and never-seen ids in missing.",
                "inputSchema": {"type": "object", "properties": {
                    "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 200},
                    "refresh": {"type": "boolean", "default": false},
                }, "required": ["ids"]},
            }),
            json!({
                "name": "pd_graph_inspect",
                "description": "Read the server graph: a domain's evaluations, anonymous authors (author:<12 hex>) and reply threads (mode domain), the thread around one evaluation (local) or its nearest evaluations (similar). Evaluation nodes carry the server's tier and grade; the summary counts them.",
                "inputSchema": {"type": "object", "properties": {
                    "cc": cc,
                    "node_id": s("Evaluation UUID (modes local, similar)"),
                    "mode": {"type": "string", "enum": ["domain", "local", "similar"], "description": "Default: local with node_id, else domain"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 49, "default": 10, "description": "similar only"},
                }},
            }),
            json!({
                "name": "pd_auth_session",
                "description": "QR e-mail login: status (verify asks the server), start (returns the address and mailto: link to show as a QR), poll (call every 3 s until signed_in or expired, 15 min), logout. The session is saved by premium-discount-app (session.json, owner-only), never returned.",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["status", "start", "poll", "logout"]},
                    "verify": {"type": "boolean", "default": false},
                }, "required": ["action"]},
            }),
            json!({
                "name": "pd_model_manage",
                "description": "embeddinggemma-2 for premium-discount-app: status (installed variant, download progress, whether the app process is running), download (text: 574 MB text-only; full: 1.52 GB; resumable, SHA-256 checked) or cancel.",
                "inputSchema": {"type": "object", "properties": {
                    "action": {"type": "string", "enum": ["status", "download", "cancel"]},
                    "variant": {"type": "string", "enum": ["text", "full"], "default": "text"},
                }, "required": ["action"]},
            }),
        ]
    }

    fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
        let out = match name {
            "pd_enqueue_node" => self.enqueue(args),
            "pd_flush_and_sync" => self.flush_and_sync(args),
            "pd_search_merged" => self.search(args),
            "pd_vote" => self.vote(args),
            "pd_verdicts" => self.verdicts(args),
            "pd_graph_inspect" => self.graph(args),
            "pd_auth_session" => self.auth(args),
            "pd_model_manage" => self.model(args),
            other => bail!("unknown_command: unknown tool {other}"),
        };
        // Arguments this layer refused carry no kind yet.
        out.map_err(|e| {
            if has_kind(&format!("{e}")) {
                e
            } else {
                anyhow!("invalid_input: {e:#}")
            }
        })
    }
}
