//! premium.discount's public graph (`www/src/graph.ts`), read directly: premium-discount-app has no graph
//! command.
//!
//! * `GET /graph?cc=` — a domain's evaluations, their authors and reply edges
//! * `GET /graph/local?id=` — the thread around one evaluation
//! * `GET /graph/similar?id=&topK=` — its nearest evaluations (`similar` edges with a score)
//!
//! Each returns `{nodes: [{id, type: evaluation | domain | author, label, cc?, root?, created_at?}],
//! edges: [{source, target, type: belongs | authored | reply | similar, score?}]}` (at most 200 rows).
//! Authors are only `author:<12 hex>` keys (HMAC of the address, computed when the graph is read). Evaluation
//! nodes carry the server's verdict (`tier`, `grade`); older servers send none, which counts as `unrated`.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{bail, Result};
use serde_json::{json, Value};

use crate::lines::CallError;

pub struct GraphClient {
    base: String,
    agent: ureq::Agent,
}

/// Which graph to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphQuery {
    Domain { cc: String },
    Local { id: String },
    Similar { id: String, top_k: u32 },
}

fn query_escape(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

impl GraphQuery {
    pub fn path(&self) -> String {
        match self {
            GraphQuery::Domain { cc } => format!("/graph?cc={}", query_escape(cc)),
            GraphQuery::Local { id } => format!("/graph/local?id={}", query_escape(id)),
            GraphQuery::Similar { id, top_k } => format!("/graph/similar?id={}&topK={top_k}", query_escape(id)),
        }
    }
}

impl GraphClient {
    pub fn new(base_url: &str) -> GraphClient {
        let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(15)).build();
        GraphClient {
            base: base_url.trim_end_matches('/').to_string(),
            agent,
        }
    }

    pub fn get(&self, q: &GraphQuery) -> Result<Value, CallError> {
        let url = format!("{}{}", self.base, q.path());
        match self.agent.get(&url).call() {
            Ok(resp) => {
                let body = resp
                    .into_string()
                    .map_err(|e| CallError::Transport(format!("reading {url}: {e}")))?;
                serde_json::from_str(&body).map_err(|e| CallError::Command {
                    kind: "bad_response".into(),
                    message: format!("{url}: {e}"),
                })
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                let message = serde_json::from_str::<Value>(&body)
                    .ok()
                    .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or(body);
                Err(CallError::Command {
                    kind: "server".into(),
                    message: format!("{code}: {message}"),
                })
            }
            Err(e) => Err(CallError::Command {
                kind: "offline".into(),
                message: e.to_string(),
            }),
        }
    }
}

fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.char_indices().all(|(i, c)| match i {
            8 | 13 | 18 | 23 => c == '-',
            _ => c.is_ascii_hexdigit(),
        })
}

/// Checks a graph request the way the server does (`topK` 1..=49, ids are UUIDs).
pub fn query_of(cc: Option<String>, id: Option<String>, mode: Option<&str>, top_k: Option<u64>) -> Result<GraphQuery> {
    let mode = mode.unwrap_or(if id.is_some() { "local" } else { "domain" });
    let id = id.map(|i| i.trim().to_ascii_lowercase());
    match mode {
        "domain" => match cc {
            Some(cc) => Ok(GraphQuery::Domain { cc }),
            None => bail!("mode domain needs `cc`"),
        },
        "local" | "similar" => {
            let Some(id) = id else { bail!("mode {mode} needs `node_id`") };
            if !is_uuid(&id) {
                bail!("node_id must be an evaluation UUID");
            }
            if mode == "local" {
                Ok(GraphQuery::Local { id })
            } else {
                let top_k = top_k.unwrap_or(10);
                if !(1..=49).contains(&top_k) {
                    bail!("top_k must be between 1 and 49 for similar");
                }
                Ok(GraphQuery::Similar { id, top_k: top_k as u32 })
            }
        }
        other => bail!("mode must be domain, local or similar (got {other})"),
    }
}

/// Counts, thread shape and the tier distribution of a graph.
pub fn summarize(graph: &Value) -> Value {
    let empty = Vec::new();
    let nodes = graph.get("nodes").and_then(Value::as_array).unwrap_or(&empty);
    let edges = graph.get("edges").and_then(Value::as_array).unwrap_or(&empty);
    let str_of = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
    let mut node_types: BTreeMap<String, usize> = BTreeMap::new();
    let mut threads: BTreeMap<String, usize> = BTreeMap::new();
    let mut tiers: BTreeMap<String, usize> = ["premium", "neutral", "discount", "unrated"]
        .into_iter()
        .map(|t| (t.to_string(), 0))
        .collect();
    for n in nodes {
        let ty = str_of(n, "type").unwrap_or_default();
        *node_types.entry(ty.clone()).or_default() += 1;
        if ty != "evaluation" {
            continue;
        }
        let id = str_of(n, "id");
        let root = str_of(n, "root").or_else(|| id.clone()).unwrap_or_default();
        *threads.entry(root).or_default() += 1;
        *tiers.entry(str_of(n, "tier").unwrap_or_else(|| "unrated".into())).or_default() += 1;
    }
    let mut edge_types: BTreeMap<String, usize> = BTreeMap::new();
    for e in edges {
        *edge_types.entry(str_of(e, "type").unwrap_or_default()).or_default() += 1;
    }
    let mut thread_list: Vec<(String, usize)> = threads.into_iter().collect();
    thread_list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let authors = node_types.get("author").copied().unwrap_or(0);
    json!({
        "nodes": node_types,
        "edges": edge_types,
        "authors": authors,
        "threads": thread_list.iter().map(|(root, size)| json!({"root": root, "evaluations": size})).collect::<Vec<_>>(),
        "tiers": tiers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    const A: &str = "0190f0a0-0000-7000-8000-0000000000aa";
    const B: &str = "0190f0a0-0000-7000-8000-0000000000bb";

    fn graph() -> Value {
        json!({
            "nodes": [
                {"id": "domain:code.exe", "type": "domain", "label": "code.exe"},
                {"id": A, "type": "evaluation", "label": "평가", "cc": "code.exe", "root": A, "tier": "premium", "grade": 2},
                {"id": B, "type": "evaluation", "label": "답글", "cc": "code.exe", "root": A},
                {"id": "author:0123456789ab", "type": "author", "label": "작성자 012345"},
            ],
            "edges": [
                {"source": A, "target": "domain:code.exe", "type": "belongs"},
                {"source": B, "target": A, "type": "reply"},
                {"source": "author:0123456789ab", "target": A, "type": "authored"},
            ],
        })
    }

    #[test]
    fn summary_counts_threads_and_tiers() {
        let s = summarize(&graph());
        assert_eq!(s["nodes"]["evaluation"], 2);
        assert_eq!(s["edges"]["reply"], 1);
        assert_eq!(s["authors"], 1);
        assert_eq!(s["threads"][0], json!({"root": A, "evaluations": 2}));
        assert_eq!(s["tiers"]["premium"], 1);
        assert_eq!(s["tiers"]["unrated"], 1, "a node without a tier (older server) is unrated");
        assert_eq!(s["tiers"]["discount"], 0);
    }

    #[test]
    fn queries_are_checked_like_the_server() {
        assert_eq!(
            query_of(Some("code.exe".into()), None, None, None).unwrap().path(),
            "/graph?cc=code.exe"
        );
        assert_eq!(
            query_of(None, Some(A.to_uppercase()), Some("similar"), Some(5)).unwrap().path(),
            format!("/graph/similar?id={A}&topK=5")
        );
        assert!(query_of(None, Some(A.into()), Some("similar"), Some(50)).is_err());
        assert!(query_of(None, Some("nope".into()), None, None).is_err());
        assert!(query_of(None, None, None, None).is_err());
    }

    /// One-shot HTTP server on localhost answering `status` with `body`.
    fn serve_once(status: &str, body: &str) -> (String, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (status, body) = (status.to_string(), body.to_string());
        let handle = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(sock.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            loop {
                let mut l = String::new();
                if reader.read_line(&mut l).unwrap() == 0 || l == "\r\n" {
                    break;
                }
            }
            write!(
                sock,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            request_line
        });
        (base, handle)
    }

    #[test]
    fn reads_the_graph_over_http() {
        let (base, server) = serve_once("200 OK", &graph().to_string());
        let g = GraphClient::new(&base).get(&GraphQuery::Domain { cc: "code.exe".into() }).unwrap();
        assert_eq!(g["nodes"].as_array().unwrap().len(), 4);
        assert!(server.join().unwrap().starts_with("GET /graph?cc=code.exe "));

        let (base, server) = serve_once("429 Too Many Requests", r#"{"error":"rate limited"}"#);
        let e = GraphClient::new(&base).get(&GraphQuery::Local { id: A.into() }).unwrap_err();
        assert_eq!(e.to_string(), "server: 429: rate limited");
        server.join().unwrap();

        let e = GraphClient::new("http://127.0.0.1:9")
            .get(&GraphQuery::Local { id: A.into() })
            .unwrap_err();
        assert_eq!(e.kind(), "offline");
    }
}
