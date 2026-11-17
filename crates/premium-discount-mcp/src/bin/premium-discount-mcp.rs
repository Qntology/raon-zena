//! `premium-discount-mcp [--pd-app PATH] [--home DIR] [--model-dir DIR] [--data-dir DIR] [--server-url URL]
//! [--modalities text|auto] [--quant q8_0|auto|off] [--idle-secs N]` — premium.discount sync, votes and verdicts
//! for Raon Zena as an MCP server over stdio (newline-delimited JSON-RPC 2.0).
//!
//! Tools: `pd_enqueue_node`, `pd_flush_and_sync`, `pd_search_merged`, `pd_vote`, `pd_verdicts`,
//! `pd_graph_inspect`, `pd_auth_session`, `pd_model_manage`. premium-discount-app is started on the first tool
//! that needs it and stopped after `--idle-secs` (default 300, 0 = never) without calls.
//!
//! Defaults: `--pd-app` from `$PD_APP`, next to this executable, `<home>/pd-app.json`, the premium.discount desktop
//! install folder or `PATH` ([`premium_discount_mcp::find_pd_app`]); `--home` is
//! `%LOCALAPPDATA%\RaonZena\premium-discount` (`model/`, `data/`); `--model-dir` / `--data-dir` /
//! `--server-url` also read `PD_MODEL_DIR` / `PD_DATA_DIR` / `PD_SERVER_URL`.

use std::path::PathBuf;
use std::time::Duration;

use premium_discount_mcp::{default_home, find_pd_app_with, AppConfig, PdSyncServer, ProcessBackend, DEFAULT_SERVER_URL, SERVER_NAME};
use zena_avatar_mcp::protocol::serve;

const USAGE: &str = "usage: premium-discount-mcp [--pd-app PATH] [--home DIR] [--model-dir DIR] [--data-dir DIR] \
                     [--server-url URL] [--modalities text|auto] [--quant q8_0|auto|off] [--idle-secs N]";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

fn main() {
    let mut pd_app: Option<PathBuf> = None;
    let mut home: Option<PathBuf> = None;
    let mut model_dir = env_path("PD_MODEL_DIR");
    let mut data_dir = env_path("PD_DATA_DIR");
    let mut server_url = std::env::var("PD_SERVER_URL").ok().filter(|s| !s.trim().is_empty());
    let mut modalities = "text".to_string();
    let mut quant = "q8_0".to_string();
    let mut idle_secs: u64 = 300;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |flag: &str| {
            args.next().unwrap_or_else(|| {
                eprintln!("premium-discount-mcp: {flag} needs a value\n{USAGE}");
                std::process::exit(2);
            })
        };
        match a.as_str() {
            "--pd-app" => pd_app = Some(value("--pd-app").into()),
            "--home" => home = Some(value("--home").into()),
            "--model-dir" => model_dir = Some(value("--model-dir").into()),
            "--data-dir" => data_dir = Some(value("--data-dir").into()),
            "--server-url" => server_url = Some(value("--server-url")),
            "--modalities" => modalities = value("--modalities"),
            "--quant" => quant = value("--quant"),
            "--idle-secs" => {
                let v = value("--idle-secs");
                idle_secs = v.parse().unwrap_or_else(|_| {
                    eprintln!("premium-discount-mcp: --idle-secs {v} is not a number");
                    std::process::exit(2);
                });
            }
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return;
            }
            other => eprintln!("premium-discount-mcp: ignoring unknown argument {other}"),
        }
    }
    if !matches!(modalities.as_str(), "text" | "auto") || !matches!(quant.as_str(), "q8_0" | "auto" | "off") {
        eprintln!("premium-discount-mcp: --modalities is text|auto and --quant is q8_0|auto|off\n{USAGE}");
        std::process::exit(2);
    }
    let home = home.unwrap_or_else(default_home);
    let config = AppConfig {
        program: pd_app.unwrap_or_else(|| {
            let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty());
            find_pd_app_with(&env, std::env::current_exe().ok().as_deref(), &home)
        }),
        model_dir: model_dir.unwrap_or_else(|| home.join("model")),
        data_dir: data_dir.unwrap_or_else(|| home.join("data")),
        server_url: server_url.unwrap_or_else(|| DEFAULT_SERVER_URL.to_string()),
        modalities,
        quant,
        idle: (idle_secs > 0).then(|| Duration::from_secs(idle_secs)),
    };
    eprintln!(
        "premium-discount-mcp: app {} · model {} · data {} · server {}",
        config.program.display(),
        config.model_dir.display(),
        config.data_dir.display(),
        config.server_url
    );
    let server_url = config.server_url.clone();
    let mut server = PdSyncServer::new(ProcessBackend::new(config), &server_url);
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    if let Err(e) = serve(&mut server, stdin, stdout, SERVER_NAME) {
        eprintln!("premium-discount-mcp: {e:#}");
        std::process::exit(1);
    }
}
