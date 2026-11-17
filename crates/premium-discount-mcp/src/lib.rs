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

/// The file in `--home` that may name premium-discount-app: `{"pd_app": "C:\\...\\premium-discount-app.exe"}`.
pub const PD_APP_CONFIG: &str = "pd-app.json";

/// The installer's product name (premium-discount `app/src-tauri/tauri.conf.json` `productName`), which the
/// Tauri installers use as the install folder.
pub const PD_PRODUCT_NAME: &str = "premium.discount";

/// premium-discount-app (the headless core that speaks `serve_lines`), in order:
/// 1. `$PD_APP`
/// 2. next to this executable (a Raon Zena build that ships both)
/// 3. `"pd_app"` in `<home>/pd-app.json` (written by hand or by an installer)
/// 4. the standalone desktop install: `%LOCALAPPDATA%\premium.discount\` (NSIS per-user, the default),
///    `%ProgramFiles%\premium.discount\` (per-machine), `/Applications/premium.discount.app/Contents/MacOS/`
/// 5. the bare name, so `PATH` decides.
///
/// Steps 3 and 4 only return a file that exists. The desktop installer carries the core only when premium-discount
/// bundles it as a sidecar (`bundle.externalBin`); without it step 4 finds nothing and step 5 applies.
pub fn find_pd_app() -> std::path::PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
    find_pd_app_with(&env, std::env::current_exe().ok().as_deref(), &default_home())
}

/// [`find_pd_app`] with the environment, this executable and `home` passed in (tests, `--home`).
pub fn find_pd_app_with(
    env: &dyn Fn(&str) -> Option<std::ffi::OsString>,
    exe: Option<&std::path::Path>,
    home: &std::path::Path,
) -> std::path::PathBuf {
    use std::path::PathBuf;
    let name = format!("premium-discount-app{}", std::env::consts::EXE_SUFFIX);
    if let Some(p) = env("PD_APP") {
        return PathBuf::from(p);
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(e) = exe {
        candidates.push(e.with_file_name(&name));
    }
    if let Some(p) = std::fs::read_to_string(home.join(PD_APP_CONFIG))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("pd_app").and_then(|p| p.as_str()).map(str::to_string))
        .filter(|p| !p.trim().is_empty())
    {
        candidates.push(PathBuf::from(p.trim()));
    }
    for root in ["LOCALAPPDATA", "ProgramFiles", "ProgramW6432"] {
        if let Some(d) = env(root) {
            candidates.push(PathBuf::from(d).join(PD_PRODUCT_NAME).join(&name));
        }
    }
    if cfg!(target_os = "macos") {
        candidates.push(PathBuf::from(format!("/Applications/{PD_PRODUCT_NAME}.app/Contents/MacOS")).join(&name));
    }
    candidates.into_iter().find(|p| p.is_file()).unwrap_or_else(|| PathBuf::from(name))
}

/// `premium-discount-mcp` next to this executable, if it was built (how the app finds the server).
pub fn sibling_server() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let p = exe.with_file_name(format!("premium-discount-mcp{}", std::env::consts::EXE_SUFFIX));
    p.is_file().then_some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pd-mcp-find-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    #[test]
    fn find_pd_app_order() {
        let root = temp("order");
        let name = format!("premium-discount-app{}", std::env::consts::EXE_SUFFIX);
        let home = root.join("home");
        let local = root.join("local");
        let vars = |extra: Vec<(&'static str, OsString)>| {
            move |k: &str| extra.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone())
        };
        let exe = root.join("zena").join("premium-discount-mcp.exe");
        let base = vars(vec![("LOCALAPPDATA", local.clone().into_os_string())]);

        // Nothing installed: the bare name (PATH decides).
        assert_eq!(find_pd_app_with(&base, Some(&exe), &home), PathBuf::from(&name));

        // The standalone desktop install under %LOCALAPPDATA%\premium.discount.
        let installed = local.join(PD_PRODUCT_NAME).join(&name);
        touch(&installed);
        assert_eq!(find_pd_app_with(&base, Some(&exe), &home), installed);

        // <home>/pd-app.json wins over the install folder; a missing file there is skipped.
        std::fs::create_dir_all(&home).unwrap();
        let named = root.join("custom").join(&name);
        let cfg = serde_json::json!({ "pd_app": named.to_string_lossy() }).to_string();
        std::fs::write(home.join(PD_APP_CONFIG), &cfg).unwrap();
        assert_eq!(find_pd_app_with(&base, Some(&exe), &home), installed);
        touch(&named);
        assert_eq!(find_pd_app_with(&base, Some(&exe), &home), named);

        // Next to this executable wins over both.
        let sibling = exe.with_file_name(&name);
        touch(&sibling);
        assert_eq!(find_pd_app_with(&base, Some(&exe), &home), sibling);

        // $PD_APP wins over everything, even when the file does not exist yet.
        let forced = vars(vec![("PD_APP", OsString::from("/opt/pd/app")), ("LOCALAPPDATA", local.into_os_string())]);
        assert_eq!(find_pd_app_with(&forced, Some(&exe), &home), PathBuf::from("/opt/pd/app"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
