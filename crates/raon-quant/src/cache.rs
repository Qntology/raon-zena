//! Runtime block cache for load-time pyramid quantization — the scheme Qntology/logis-center
//! uses for its SigLIP2 towers (`models/siglip2/gguf_quant.rs`) and Qwen3.5 runtime GGUFs:
//!
//! ```text
//! <model dir>/runtime-<preset>[-hg]-<fnv32(key)>/
//!     blocks.gguf    the quantized blocks of every matrix the plan quantizes (original names)
//!     report.json    per-tensor bytes / relative error of the build
//!     READY          the recipe key, written last
//! ```
//!
//! The key covers everything the blocks depend on: cache format, encoder version, preset and
//! its full configuration, the imatrix content (FNV-64), whether host-gathered tables are
//! included, and every source shard's name, size and modification time. When `READY` matches,
//! the blocks are only mapped; otherwise they are built once from the original safetensors and
//! older caches of the same flavour (preset, with `-hg` when host-gathered tables are included —
//! the GPU variant) are removed, so CPU and GPU caches coexist. If the model directory is not writable the
//! user cache directory is used (`%LOCALAPPDATA%` / `$XDG_CACHE_HOME` / `~/.cache` + `raon-zena`).

use crate::imatrix::read_imatrix;
use crate::pyramid::{convert_with, n_layers_from_config, nano_cache_plan, PyramidConfig, Report};
use crate::safetensors::SafeTensors;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

pub const RECIPE: &str = "raon-nano-blockcache/1";
/// Bump whenever the encoder's output can change.
pub const ENCODER: &str = "kquant-wls-closedform/1";
pub const BLOCKS_FILE: &str = "blocks.gguf";
pub const READY_FILE: &str = "READY";

pub struct CacheSpec<'a> {
    pub model_dir: &'a Path,
    pub preset: &'a str,
    pub cfg: &'a PyramidConfig,
    pub imatrix: Option<&'a Path>,
    /// Include host-gathered tables (Q8_0 `embed_tokens` for a GPU full head).
    pub host_gather: bool,
    /// Explicit cache root; `None` = the model directory, falling back to the user cache.
    pub root: Option<&'a Path>,
}

#[derive(Debug)]
pub enum Prepared {
    Hit {
        blocks: PathBuf,
    },
    Built {
        blocks: PathBuf,
        seconds: f32,
        report: Box<Report>,
    },
}

impl Prepared {
    pub fn blocks(&self) -> &Path {
        match self {
            Prepared::Hit { blocks } | Prepared::Built { blocks, .. } => blocks,
        }
    }
}

pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn source_fingerprint(model_dir: &Path) -> Result<String> {
    let st = SafeTensors::open_model_dir(model_dir, &[])?;
    let mut parts = Vec::new();
    for p in &st.paths {
        let m = std::fs::metadata(p).with_context(|| format!("stat {}", p.display()))?;
        let mtime = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        parts.push(format!(
            "{}:{}:{}",
            p.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            m.len(),
            mtime
        ));
    }
    Ok(parts.join(","))
}

/// The recipe key written to `READY`.
pub fn recipe_key(spec: &CacheSpec) -> Result<String> {
    let imat = match spec.imatrix {
        Some(p) => format!(
            "{:016x}",
            fnv64(&std::fs::read(p).with_context(|| format!("read {}", p.display()))?)
        ),
        None => "none".into(),
    };
    Ok(format!(
        "{RECIPE}|{ENCODER}|preset={}|cfg={:?}|imatrix={imat}|host_gather={}|src={}",
        spec.preset,
        spec.cfg,
        spec.host_gather,
        source_fingerprint(spec.model_dir)?
    ))
}

fn flavour(spec: &CacheSpec) -> String {
    if spec.host_gather {
        format!("{}-hg", spec.preset)
    } else {
        spec.preset.to_string()
    }
}

fn dir_name(flavour: &str, key: &str) -> String {
    format!("runtime-{flavour}-{:08x}", fnv64(key.as_bytes()) as u32)
}

fn user_cache_root(model_dir: &Path) -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    let model = model_dir.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    Some(base.join("raon-zena").join(model))
}

fn ready(dir: &Path, key: &str) -> bool {
    let mark = std::fs::read_to_string(dir.join(READY_FILE)).unwrap_or_default();
    mark.trim() == key && dir.join(BLOCKS_FILE).is_file()
}

/// Remove other caches of exactly this flavour (`runtime-<flavour>-<8 hex>`) that this module wrote.
fn remove_stale(root: &Path, keep: &Path, flavour: &str) {
    let prefix = format!("runtime-{flavour}-");
    let Ok(entries) = std::fs::read_dir(root) else { return };
    for e in entries.flatten() {
        let p = e.path();
        let ours = std::fs::read_to_string(p.join(READY_FILE))
            .map(|r| r.starts_with(RECIPE))
            .unwrap_or(false);
        let name = e.file_name();
        let same_flavour = name
            .to_str()
            .and_then(|n| n.strip_prefix(&prefix))
            .is_some_and(|h| h.len() == 8 && h.chars().all(|c| c.is_ascii_hexdigit()));
        if p != keep && ours && same_flavour && std::fs::remove_dir_all(&p).is_ok() {
            eprintln!("[quant-cache] removed stale cache {}", p.display());
        }
    }
}

static BUILD_LOCK: Mutex<()> = Mutex::new(());

/// Map an existing cache or build it once from the original safetensors.
pub fn prepare(spec: &CacheSpec) -> Result<Prepared> {
    let key = recipe_key(spec)?;
    let flavour = flavour(spec);
    let name = dir_name(&flavour, &key);
    let roots: Vec<PathBuf> = match spec.root {
        Some(r) => vec![r.to_path_buf()],
        None => std::iter::once(spec.model_dir.to_path_buf())
            .chain(user_cache_root(spec.model_dir))
            .collect(),
    };
    for r in &roots {
        let dir = r.join(&name);
        if ready(&dir, &key) {
            return Ok(Prepared::Hit {
                blocks: dir.join(BLOCKS_FILE),
            });
        }
    }
    let _guard = BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // First writable root wins.
    let mut last_err = None;
    for r in &roots {
        let dir = r.join(&name);
        if ready(&dir, &key) {
            return Ok(Prepared::Hit {
                blocks: dir.join(BLOCKS_FILE),
            });
        }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            last_err = Some(anyhow::anyhow!("{}: {e}", dir.display()));
            continue;
        }
        match build(spec, &dir, &key) {
            Ok((seconds, report)) => {
                remove_stale(r, &dir, &flavour);
                return Ok(Prepared::Built {
                    blocks: dir.join(BLOCKS_FILE),
                    seconds,
                    report: Box::new(report),
                });
            }
            Err(e) => {
                let _ = std::fs::remove_dir_all(&dir);
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no cache directory available")))
}

fn build(spec: &CacheSpec, dir: &Path, key: &str) -> Result<(f32, Report)> {
    let t0 = Instant::now();
    let _ = std::fs::remove_file(dir.join(READY_FILE));
    let st = SafeTensors::open_model_dir(spec.model_dir, &[])?;
    let plan = nano_cache_plan(&st, spec.cfg, n_layers_from_config(spec.model_dir)?, spec.host_gather)?;
    let imatrix = spec.imatrix.map(read_imatrix).transpose()?;
    let total = plan.items.len();
    eprintln!(
        "[quant-cache] building {} ({total} tensors, preset '{}') from the original safetensors",
        dir.display(),
        spec.preset
    );
    let step = (total / 10).max(1);
    let report = convert_with(
        &st,
        &plan,
        spec.cfg,
        &dir.join(BLOCKS_FILE),
        vec![("raon.cache.key".into(), crate::gguf::Kv::Str(key.to_string()))],
        imatrix.as_ref(),
        &|k, n, _, _, _| {
            if (k + 1) % step == 0 || k + 1 == n {
                eprintln!("[quant-cache]   {}/{n} tensors, {:.0}s", k + 1, t0.elapsed().as_secs_f32());
            }
        },
    )?;
    write_atomic(&dir.join("report.json"), serde_json::to_string_pretty(&report)?.as_bytes())?;
    // READY last: a crash before this line leaves no valid cache behind.
    write_atomic(&dir.join(READY_FILE), key.as_bytes())?;
    Ok((t0.elapsed().as_secs_f32(), report))
}

/// Writes `bytes` to a unique sibling temp file, `sync_all()`s it, and atomically replaces `target`.
pub fn write_atomic(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let mut tmp_os = target.as_os_str().to_os_string();
    tmp_os.push(format!(".tmp.{}.{seq}", std::process::id()));
    let tmp = PathBuf::from(tmp_os);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    match std::fs::rename(&tmp, target) {
        Ok(()) => Ok(()),
        Err(_) if target.exists() => {
            let mut bak_os = target.as_os_str().to_os_string();
            bak_os.push(format!(".bak.{}.{seq}", std::process::id()));
            let bak = PathBuf::from(bak_os);
            let _ = std::fs::remove_file(&bak);
            std::fs::rename(target, &bak)?;
            match std::fs::rename(&tmp, target) {
                Ok(()) => {
                    let _ = std::fs::remove_file(&bak);
                    Ok(())
                }
                Err(e) => {
                    let _ = std::fs::rename(&bak, target);
                    let _ = std::fs::remove_file(&tmp);
                    Err(e)
                }
            }
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{write_safetensors, Dtype};

    fn bf16_bytes(n: usize, seed: u32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                half::bf16::from_f32((((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 1000.0 - 0.5) * 0.05)
                    .to_le_bytes()
            })
            .collect()
    }

    /// A miniature Raon-1-nano: 4 layers (1 MLA), hidden 256.
    fn tiny_model(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = serde_json::json!({"num_hidden_layers": 4, "rda_config": {"rda_layers": [1, 2, 3], "mla_layers": [4]}});
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut t: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = vec![(
            "model.embed_tokens.weight".into(),
            Dtype::BF16,
            vec![64, 256],
            bf16_bytes(64 * 256, 1),
        )];
        for l in 0..4 {
            for (n, s) in [
                ("mlp.gate_proj.weight", [512, 256]),
                ("mlp.down_proj.weight", [256, 512]),
                ("self_attn.q_proj.weight", [256, 256]),
            ] {
                t.push((
                    format!("model.layers.{l}.{n}"),
                    Dtype::BF16,
                    s.to_vec(),
                    bf16_bytes(s[0] * s[1], l as u32 + 7),
                ));
            }
            t.push((
                format!("model.layers.{l}.input_layernorm.weight"),
                Dtype::BF16,
                vec![256],
                bf16_bytes(256, 3),
            ));
        }
        let refs: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)> = t.iter().map(|(n, d, s, b)| (n.as_str(), *d, s.clone(), b.clone())).collect();
        write_safetensors(&dir.join("model.safetensors"), &refs).unwrap();
    }

    #[test]
    fn builds_once_hits_after_and_rebuilds_on_recipe_change() {
        let dir = std::env::temp_dir().join(format!("raonq-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        tiny_model(&dir);
        let cfg = PyramidConfig::preset("prd").unwrap();
        let spec = CacheSpec {
            model_dir: &dir,
            preset: "prd",
            cfg: &cfg,
            imatrix: None,
            host_gather: false,
            root: None,
        };
        let first = prepare(&spec).unwrap();
        let blocks = first.blocks().to_path_buf();
        let Prepared::Built { report, .. } = &first else {
            panic!("expected a build")
        };
        // 4 layers × 3 matrices; norms (T1) and the gathered embedding table are not cached.
        assert_eq!(report.items.len(), 12);
        let g = crate::gguf::GgufFile::open(&blocks).unwrap();
        assert!(g.tensors.contains_key("model.layers.2.mlp.gate_proj.weight"));
        assert!(!g.tensors.contains_key("model.embed_tokens.weight"));
        assert_eq!(g.kv_str("general.architecture"), Some("raon1_nano_blockcache"));
        let mtime = std::fs::metadata(&blocks).unwrap().modified().unwrap();
        assert!(matches!(prepare(&spec).unwrap(), Prepared::Hit { .. }));
        assert_eq!(
            std::fs::metadata(&blocks).unwrap().modified().unwrap(),
            mtime,
            "a hit must not rewrite"
        );
        // The GPU flavour (host-gathered tables included) is a separate cache; the CPU one stays.
        let spec_hg = CacheSpec { host_gather: true, ..spec };
        let gpu = prepare(&spec_hg).unwrap();
        assert_ne!(gpu.blocks(), blocks);
        assert!(gpu
            .blocks()
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("runtime-prd-hg-"));
        assert!(crate::gguf::GgufFile::open(gpu.blocks())
            .unwrap()
            .tensors
            .contains_key("model.embed_tokens.weight"));
        assert!(blocks.exists());
        // A recipe change of the same flavour replaces the older cache.
        let cfg2 = PyramidConfig {
            edge_front: 1,
            ..cfg.clone()
        };
        let spec3 = CacheSpec { cfg: &cfg2, ..spec };
        let third = prepare(&spec3).unwrap();
        assert!(matches!(third, Prepared::Built { .. }));
        assert!(!blocks.exists(), "stale cache of the same flavour is removed");
        assert!(gpu.blocks().exists(), "other flavours are kept");
        // Touching the source invalidates.
        let st = dir.join("model.safetensors");
        let f = std::fs::OpenOptions::new().append(true).open(&st).unwrap();
        f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(5))
            .unwrap();
        assert!(matches!(prepare(&spec_hg).unwrap(), Prepared::Built { .. }));
        std::fs::remove_dir_all(dir).ok();
    }
}
