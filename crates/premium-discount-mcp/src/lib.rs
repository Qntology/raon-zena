//! premium-discount-mcp: premium.discount sync, votes and verdicts for Raon Zena over MCP.
//!
//! premium-discount defines the schema: evaluations, votes, verdicts (rule: its `contracts/verdict_v1.json`) and
//! every field name. This crate exposes them as MCP tools and maps Raon Zena's places onto them.
//!
//! ```text
//! raon-zena ──McpClient (stdio JSON-RPC 2.0)──▶ premium-discount-mcp ──serve_lines (stdio JSON lines)──▶ premium-discount-app
//!   SyncLoop (coalesced, 10 s while work)          8 pd_* tools, cc / hashId,         LanceDB outbox (original text, votes),
//!                                                  GET /graph                         verdict cache, embeddinggemma-2, QR login
//!                                                         │                                    │
//!                                                         └──────────── HTTPS ─────────────────┴──▶ premium.discount
//! ```
//!
//! premium-discount-app is used as a process, not linked: it pins LanceDB 0.31 / candle 0.9 while this workspace
//! is on LanceDB 0.41 / the candle 0.11 fork, so linking it would build both trees. Its line protocol
//! (`serve_lines`) already exposes every command; this crate wraps it in the MCP framing of
//! [`zena_avatar_mcp::protocol`], the same one Track A and Track B use.
//!
//! * [`server`] — the tools ([`server::PdSyncServer`])
//! * [`bridge`] — the premium-discount-app process (lazy start, restart, idle stop)
//! * [`lines`] — its line protocol client
//! * [`envelope`] / [`address`] — `cc`, and `to` / `bcc` addresses (`hashId`)
//! * [`graph`] — `GET /graph*`
//! * [`client`] — the app side: [`client::PdSyncClient`], [`client::SyncLoop`]

pub mod address;
pub mod bridge;
pub mod client;
pub mod envelope;
pub mod graph;
pub mod lines;
pub mod server;

pub use bridge::{AppConfig, Backend, ProcessBackend};
pub use client::{PdSyncClient, SyncLoop, SyncRunner, SyncSnapshot, SYNC_INTERVAL};
pub use lines::{CallError, LineClient};
pub use server::{error_kind, PdSyncServer, SERVER_NAME, TIER_FILTERS, TOOLS};

/// premium.discount.
pub const DEFAULT_SERVER_URL: &str = "https://premium.discount";

/// `%LOCALAPPDATA%\RaonZena\premium-discount` on Windows, `$XDG_DATA_HOME/raon-zena/premium-discount`
/// (or `~/.local/share/...`) elsewhere. Holds `model/` (embeddinggemma-2) and `data/` (outbox, votes, verdict
/// cache, session.json).
pub fn default_home() -> std::path::PathBuf {
    use std::path::PathBuf;
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("RaonZena").join("premium-discount");
        }
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("raon-zena").join("premium-discount")
}

/// premium-discount-app: `$PD_APP`, else next to this executable, else on `PATH`.
pub fn find_pd_app() -> std::path::PathBuf {
    use std::path::PathBuf;
    let name = format!("premium-discount-app{}", std::env::consts::EXE_SUFFIX);
    if let Some(p) = std::env::var_os("PD_APP").filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    if let Some(p) = std::env::current_exe()
        .ok()
        .map(|e| e.with_file_name(&name))
        .filter(|p| p.is_file())
    {
        return p;
    }
    PathBuf::from(name)
}

/// `premium-discount-mcp` next to this executable, if it was built (how the app finds the server).
pub fn sibling_server() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let p = exe.with_file_name(format!("premium-discount-mcp{}", std::env::consts::EXE_SUFFIX));
    p.is_file().then_some(p)
}
