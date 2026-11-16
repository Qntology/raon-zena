//! CPU timings and memory of the granite embedder (prints only; nothing is asserted about speed).
//!
//! `CARGO_TARGET_DIR=target/agent-granite cargo test -p zena-nlp --no-default-features --release --test granite_bench -- --nocapture`
//!
//! A test binary of its own, with a single test, so that the process memory delta is the model's alone.
//! Model folder: `RAON_GRANITE_DIR`, else `%LOCALAPPDATA%\logis-center\models\granite-embedding-97m-multilingual-r2`.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use candle_core::Device;
use zena_nlp::granite::{Granite, MAX_TOKENS};

fn model_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("RAON_GRANITE_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
            .join("logis-center")
            .join("models")
            .join("granite-embedding-97m-multilingual-r2"),
    };
    ["config.json", "model.safetensors", "tokenizer.json"]
        .iter()
        .all(|f| dir.join(f).is_file())
        .then_some(dir)
}

/// (resident set, private bytes) of this process in MiB, if the platform makes that easy.
fn memory_mib() -> Option<(f64, f64)> {
    const MIB: f64 = 1024.0 * 1024.0;
    if cfg!(windows) {
        let script = format!(
            "$p = Get-Process -Id {}; \"$($p.WorkingSet64) $($p.PrivateMemorySize64)\"",
            std::process::id()
        );
        let out = std::process::Command::new("powershell")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout);
        let mut it = text.split_whitespace().map(|n| n.parse::<f64>().ok());
        Some((it.next()?? / MIB, it.next()?? / MIB))
    } else {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        let kib = |key: &str| -> Option<f64> {
            let line = status.lines().find(|l| l.starts_with(key))?;
            line.split_whitespace().nth(1)?.parse::<f64>().ok()
        };
        Some((kib("VmRSS:")? / 1024.0, kib("VmData:")? / 1024.0))
    }
}

/// Median of `runs` timings of `f` (after one warm-up call).
fn median(runs: usize, mut f: impl FnMut()) -> Duration {
    f();
    let mut t: Vec<Duration> = (0..runs)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed()
        })
        .collect();
    t.sort();
    t[runs / 2]
}

#[test]
fn cpu_timings() {
    let Some(dir) = model_dir() else {
        eprintln!("[granite-bench] model folder not found (set RAON_GRANITE_DIR); skipping");
        return;
    };
    let before = memory_mib();
    let started = Instant::now();
    let m = Granite::load(&dir, &Device::Cpu).expect("load granite");
    let load = started.elapsed();
    let after = memory_mib();
    eprintln!("[granite-bench] load {load:.2?} (tokenizer + F32 weights) from {}", dir.display());
    if let (Some((rss0, priv0)), Some((rss1, priv1))) = (before, after) {
        eprintln!(
            "[granite-bench] memory: RSS {rss0:.0} → {rss1:.0} MiB (+{:.0}), private {priv0:.0} → {priv1:.0} MiB (+{:.0})",
            rss1 - rss0,
            priv1 - priv0
        );
    }

    let one = "보스 패턴 어떻게 피해?";
    let t_one = median(30, || {
        m.embed(one).unwrap();
    });
    eprintln!(
        "[granite-bench] embed 1 short sentence ({} tokens): {t_one:.2?} (median of 30)",
        m.token_count(one).unwrap()
    );

    let spans = [
        "또 죽었어",
        "와 잡았다 고마워",
        "보스 패턴 어떻게 피해?",
        "로봇 모드로 변신해 줘",
        "안경 써 봐",
        "인벤토리 열어 줘",
        "보스한테 졌어",
        "tell me the boss pattern",
        "wgpu SurfaceError Outdated",
        "このボス強すぎる",
        "설정 메뉴 열어 줘",
        "볼륨 좀 낮춰",
        "다시 해 볼까?",
        "open the map",
        "체력 얼마나 남았어?",
        "?",
    ];
    let max_len = spans.iter().map(|s| m.token_count(s).unwrap()).max().unwrap();
    let t_batch = median(10, || {
        m.embed_batch(&spans).unwrap();
    });
    let t_seq = median(5, || {
        for s in spans {
            m.embed(s).unwrap();
        }
    });
    eprintln!(
        "[granite-bench] 16 short spans (≤ {max_len} tokens): embed_batch {t_batch:.2?} ({:.2?}/span), one-by-one embed {t_seq:.2?} (median of 10 / 5)",
        t_batch / 16,
    );

    let long =
        "보스는 체력이 절반 아래로 떨어지면 패턴이 바뀌고 불기둥이 올라온다. The renderer reconfigures the surface on resize. ".repeat(40);
    let t_long = median(5, || {
        m.embed(&long).unwrap();
    });
    eprintln!(
        "[granite-bench] embed 1 long input ({} tokens, truncated to {MAX_TOKENS}): {t_long:.2?} (median of 5)",
        m.token_count(&long).unwrap()
    );

    if let Some((rss, private)) = memory_mib() {
        eprintln!("[granite-bench] memory after the runs: RSS {rss:.0} MiB, private {private:.0} MiB");
    }
}
