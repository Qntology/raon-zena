//! `raon-quant` CLI.
//!
//! ```text
//! raon-quant verify
//! raon-quant nano   --model-dir hf/KRAFTON/Raon-1-nano --out out/raon-1-nano.pyramid.gguf
//! raon-quant vision --model-dir hf/KRAFTON/Raon-VisionEncoder --out out/raon-ve-vision.pyramid.gguf
//! raon-quant check  out/raon-1-nano.pyramid.gguf
//! ```

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use raon_quant::gguf::{GgmlType, GgufFile};
use raon_quant::kernels::{dequant_row, dot_row, matvec, verify_kernels};
use raon_quant::lmhead::{exact_argmax, ColdGuard};
use raon_quant::pyramid::{self, parse_format, Plan, PyramidConfig};
use raon_quant::safetensors::SafeTensors;
use raon_quant::vocab::{kept_ids_from_tokenizer, VocabSlice};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Single-pass 4-tier pyramid quantizer for KRAFTON Raon models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the fused-kernel self-test (max |error| <= 1e-3 against an F64 reference).
    Verify,
    /// Quantize KRAFTON/Raon-1-nano (+ MTP head) into one GGUF.
    Nano {
        #[command(flatten)]
        common: Common,
        /// MTP weights (default: <model-dir>/mtp/model.safetensors if present).
        #[arg(long)]
        mtp: Option<PathBuf>,
        /// tokenizer.json for kept_ids (default: <model-dir>/tokenizer.json).
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Keep every vocabulary row resident (disable kept_ids slicing).
        #[arg(long)]
        no_slice: bool,
        /// Keep only the N most frequent (lowest-id) script-filtered text tokens resident.
        /// 0 = no cap. Default sizes `lm_head.kept` so the resident set fits ~890 MB.
        #[arg(long, default_value_t = 24576)]
        keep_top: usize,
        /// Clusters for the cold-row exactness guard.
        #[arg(long, default_value_t = 1024)]
        guard_clusters: usize,
    },
    /// Quantize the vision tower of KRAFTON/Raon-VisionEncoder (text tower excluded).
    Vision {
        #[command(flatten)]
        common: Common,
    },
    /// Open a produced GGUF, decode every tensor's first row, and exercise the sliced head.
    Check { gguf: PathBuf },
}

#[derive(Args)]
struct Common {
    /// `prd`: the PRD v3.0 pyramid as written (T4 Q4_K for MLP and the PLE table).
    /// `zena`: the measured quality recipe for Raon-1-nano — MLP Q5_K, the 6 MLPs after the front
    /// edge Q6_K, PLE table Q8_0 (disk-gathered, no VRAM cost); use with `--imatrix`.
    /// Explicit --tier4 / --ple-format / --mlp-sensitive-front override the preset.
    #[arg(long, default_value = "prd")]
    preset: String,
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    out: PathBuf,
    /// Write the JSON report here (default: <out>.report.json).
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, default_value_t = 2)]
    edge_front: usize,
    #[arg(long, default_value_t = 2)]
    edge_back: usize,
    #[arg(long, default_value = "q8_0")]
    tier2: String,
    #[arg(long, default_value = "q6_k")]
    tier3: String,
    #[arg(long, default_value = "q4_k")]
    tier4: String,
    /// Per-layer-embedding table format (default: same as --tier4). Disk-gathered: no VRAM cost.
    #[arg(long)]
    ple_format: Option<String>,
    /// MLP layers right after the front edge kept at --tier3 (sensitivity override).
    #[arg(long, default_value_t = 0)]
    mlp_sensitive_front: usize,
    /// llama.cpp legacy imatrix.dat keyed by the original tensor names.
    #[arg(long)]
    imatrix: Option<PathBuf>,
    /// Print the plan and estimated sizes without writing.
    #[arg(long)]
    dry_run: bool,
}

impl Common {
    fn config(&self) -> Result<PyramidConfig> {
        // Explicit flags override the preset only when they differ from the CLI defaults.
        let base = PyramidConfig::preset(&self.preset)?;
        Ok(PyramidConfig {
            edge_front: self.edge_front,
            edge_back: self.edge_back,
            tier2: if self.tier2 == "q8_0" {
                base.tier2
            } else {
                parse_format(&self.tier2)?
            },
            tier3: if self.tier3 == "q6_k" {
                base.tier3
            } else {
                parse_format(&self.tier3)?
            },
            tier4: if self.tier4 == "q4_k" {
                base.tier4
            } else {
                parse_format(&self.tier4)?
            },
            ple: match &self.ple_format {
                Some(f) => Some(parse_format(f)?),
                None => base.ple,
            },
            mlp_sensitive_front: if self.mlp_sensitive_front == 0 {
                base.mlp_sensitive_front
            } else {
                self.mlp_sensitive_front
            },
            ..base
        })
    }
}

fn kernel_gate() -> Result<()> {
    let checks = verify_kernels(1e-3);
    for c in &checks {
        eprintln!(
            "kernel self-test {:<5} max|err| = {:.3e} {}",
            c.ty,
            c.max_abs_err,
            if c.ok { "ok" } else { "FAIL" }
        );
    }
    if checks.iter().any(|c| !c.ok) {
        bail!("kernel self-test failed; refusing to continue");
    }
    Ok(())
}

fn summarize(plan: &Plan) {
    use std::collections::BTreeMap;
    let mut t: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for it in &plan.items {
        let bytes = raon_quant::gguf::TensorDecl {
            name: String::new(),
            ty: it.ty,
            shape: it.stored.clone(),
        }
        .nbytes();
        let e = t.entry(format!("{:?}/{:?}/{}", it.tier, it.residency, it.ty.name())).or_default();
        e.0 += 1;
        e.1 += it.logical.iter().product::<usize>();
        e.2 += bytes;
    }
    eprintln!(
        "{:<40} {:>7} {:>14} {:>10} {:>6}",
        "tier/residency/type", "tensors", "params", "MiB", "bpw"
    );
    for (k, (n, p, b)) in t {
        eprintln!(
            "{k:<40} {n:>7} {p:>14} {:>10.1} {:>6.2}",
            b as f64 / 1048576.0,
            b as f64 * 8.0 / p.max(1) as f64
        );
    }
    for a in &plan.aux {
        eprintln!("aux {:<36} {:?} {:?}", a.name, a.ty, a.shape);
    }
    for (n, why) in plan.excluded.iter().take(3) {
        eprintln!("excluded {n}: {why}");
    }
    if plan.excluded.len() > 3 {
        eprintln!("… {} excluded tensors in total", plan.excluded.len());
    }
}

fn run(st: &SafeTensors, plan: Plan, common: &Common, cfg: &PyramidConfig, extra: Vec<(String, raon_quant::gguf::Kv)>) -> Result<()> {
    summarize(&plan);
    if common.dry_run {
        return Ok(());
    }
    kernel_gate()?;
    let imat = common.imatrix.as_ref().map(|p| raon_quant::imatrix::read_imatrix(p)).transpose()?;
    if let Some(parent) = common.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let report = pyramid::convert(st, &plan, cfg, &common.out, extra, imat.as_ref())?;
    let rp = common.report.clone().unwrap_or_else(|| common.out.with_extension("report.json"));
    std::fs::write(&rp, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "\nwrote {} ({:.1} MiB) in {:.0}s; report {}",
        report.out,
        report.file_bytes as f64 / 1048576.0,
        report.seconds,
        rp.display()
    );
    for (k, t) in &report.by_tier {
        eprintln!(
            "  {k:<12} {:>4} tensors {:>13} params {:>9.1} MiB {:>5.2} bpw",
            t.tensors,
            t.params,
            t.bytes as f64 / 1048576.0,
            t.bpw
        );
    }
    for (k, t) in &report.by_residency {
        eprintln!(
            "  {k:<12} {:>4} tensors {:>13} params {:>9.1} MiB",
            t.tensors,
            t.params,
            t.bytes as f64 / 1048576.0
        );
    }
    Ok(())
}

fn check(path: &std::path::Path) -> Result<()> {
    let g = GgufFile::open(path)?;
    eprintln!(
        "{}: {} tensors, {} KVs, {:.1} MiB, arch={}",
        path.display(),
        g.tensors.len(),
        g.kvs.len(),
        g.file_len() as f64 / 1048576.0,
        g.kv_str("general.architecture").unwrap_or("?")
    );
    let mut bad = 0;
    for name in &g.order {
        let t = g.tensor(name)?;
        let (_, cols) = t.rows_cols();
        let mut row = vec![0f32; cols];
        dequant_row(t.ty, g.row_bytes(name, 0)?, &mut row);
        if row.iter().any(|v| !v.is_finite()) {
            eprintln!("  non-finite values in {name}");
            bad += 1;
        }
    }
    if bad > 0 {
        bail!("{bad} tensors decode to non-finite values");
    }
    eprintln!("  every tensor's first row decodes to finite values");
    if g.tensors.contains_key("lm_head.kept") {
        let kept: Vec<u32> = g.i32_values("raon.vocab.kept_ids")?.into_iter().map(|x| x as u32).collect();
        let head = g.tensor("lm_head.kept")?;
        let (k, d) = head.rows_cols();
        let guard = ColdGuard {
            dim: d,
            centers: g.f32_values("lm_head.cold.centers")?,
            radius: g.f32_values("lm_head.cold.radius")?,
            members: g.i32_values("lm_head.cold.members")?.into_iter().map(|x| x as u32).collect(),
            offsets: g.i32_values("lm_head.cold.offsets")?.into_iter().map(|x| x as u32).collect(),
        };
        let embed = g.tensor("model.embed_tokens.weight")?;
        if embed.ty != GgmlType::Q8_0 || head.ty != GgmlType::Q8_0 {
            eprintln!("  (head types {:?}/{:?})", head.ty, embed.ty);
        }
        // Use real embedding rows of kept tokens as probe hidden states; compare against a
        // brute-force arg-max over the full on-disk table.
        let probe_ids = [kept[kept.len() / 7], kept[kept.len() / 2], kept[kept.len() - 100]];
        for &pid in &probe_ids {
            let mut h = vec![0f32; d];
            dequant_row(embed.ty, g.row_bytes("model.embed_tokens.weight", pid as usize)?, &mut h);
            let mut logits = vec![0f32; k];
            matvec(head.ty, g.tensor_bytes("lm_head.kept")?, k, d, &h, &mut logits);
            let cold = |id: u32| dot_row(embed.ty, g.row_bytes("model.embed_tokens.weight", id as usize).unwrap(), &h);
            let t = std::time::Instant::now();
            let (id, l, st) = exact_argmax(&logits, &kept, &guard, &h, cold);
            let dt = t.elapsed();
            let v = embed.rows_cols().0;
            let mut full = vec![0f32; v];
            matvec(embed.ty, g.tensor_bytes("model.embed_tokens.weight")?, v, d, &h, &mut full);
            let (fid, fl) = full
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
            eprintln!(
                "  probe token {pid}: sliced+guard → {id} ({l:.4}), full → {fid} ({fl:.4}); refined {}/{} clusters, {} rows, {:?}",
                st.clusters_refined, st.clusters, st.rows_refined, dt
            );
            if id as usize != fid {
                bail!("sliced head disagrees with the full head");
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Verify => kernel_gate(),
        Cmd::Check { gguf } => check(&gguf),
        Cmd::Nano {
            common,
            mtp,
            tokenizer,
            no_slice,
            keep_top,
            guard_clusters,
        } => {
            let cfg = PyramidConfig {
                guard_clusters,
                ..common.config()?
            };
            let mtp = mtp.or_else(|| Some(common.model_dir.join("mtp/model.safetensors")).filter(|p| p.exists()));
            let st = SafeTensors::open_model_dir(&common.model_dir, &mtp.into_iter().collect::<Vec<_>>())?;
            let n_layers = pyramid::n_layers_from_config(&common.model_dir)?;
            let vocab_size = st.info("model.embed_tokens.weight")?.shape[0];
            let vocab: VocabSlice = if no_slice {
                VocabSlice::full(vocab_size)
            } else {
                kept_ids_from_tokenizer(
                    &tokenizer.unwrap_or_else(|| common.model_dir.join("tokenizer.json")),
                    vocab_size,
                    (keep_top > 0).then_some(keep_top),
                )?
            };
            eprintln!("vocab slice: {:?}", vocab.stats);
            let plan = pyramid::plan_raon1_nano(&st, &cfg, &vocab, n_layers)?;
            let extra = pyramid::source_config_kv(&common.model_dir).into_iter().collect();
            run(&st, plan, &common, &cfg, extra)
        }
        Cmd::Vision { common } => {
            let cfg = common.config()?;
            let st = SafeTensors::open_model_dir(&common.model_dir, &[]).context("open vision encoder")?;
            let plan = pyramid::plan_raon_ve(&st, &cfg)?;
            let extra = pyramid::source_config_kv(&common.model_dir).into_iter().collect();
            run(&st, plan, &common, &cfg, extra)
        }
    }
}
