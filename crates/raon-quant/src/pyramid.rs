//! 4-tier pyramid plan + single-pass converter (PRD §3.1–§3.4).
//!
//! | tier | default | Raon-1-nano                                                        | Raon-VisionEncoder (vision tower only)            |
//! |------|---------|--------------------------------------------------------------------|---------------------------------------------------|
//! | T1   | F32/BF16| every norm, scalar/1-D, Raon-mHC mixers, RDA gates (`A_log`,`dt_bias`, `b_proj`, `rda_w_proj`, `f_a/f_b_proj`) and short convs | norms, biases, `patch_embed`, `pos_embed`, `attn_pool.latent`, logit scale/bias |
//! | T2   | Q8_0    | first/last edge layers, MTP layer, PLE `embedding_projection`, `lm_head.kept`, `embed_tokens` (disk) | edge blocks, MAP head (`attn_pool.*`)             |
//! | T3   | Q6_K    | RDA/MLA attention projections, PLE input gates                     | `attn.qkv`, `attn.proj`                           |
//! | T4   | Q4_K    | MLP `gate/up/down`, per-layer embedding table (kept + cold rows, disk) | `mlp.fc1`, `mlp.fc2`                          |
//!
//! Every tensor is read once from the original BF16/F32 safetensors through `mmap` and written
//! once; nothing is ever re-quantized from an already-quantized file. Widths that are not a
//! multiple of the block are zero-padded (SO400M: 1152 → 1280 for K-quants; the MLP hidden
//! 4304 is padded on fc1's rows *and* fc2's columns so `gelu(0)=0` keeps the product exact).

use crate::gguf::{GgmlType, GgufWriter, Kv, TensorDecl};
use crate::kernels::dequant_row;
use crate::kquant::{encode, Format};
use crate::lmhead::{build_cold_guard, ColdGuard};
use crate::safetensors::{Dtype, SafeTensors};
use crate::vocab::VocabSlice;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Tier {
    T1Spine,
    T2Gate,
    T3Attention,
    T4Bulk,
    Aux,
}

/// Where the runtime keeps a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Residency {
    /// Uploaded to VRAM (or kept hot in RAM for CPU inference).
    Device,
    /// Never loaded as a whole: rows are gathered on demand straight from the mmap.
    HostGather,
}

#[derive(Debug, Clone)]
pub enum RowSel {
    All,
    Ids(Arc<Vec<u32>>),
}

#[derive(Debug, Clone, Copy)]
pub enum Store {
    /// Copy the original bytes (lossless; dtype must match).
    Raw,
    /// Decode to F32 (lossless for BF16/F16 sources), with optional zero padding.
    F32,
    Quant(Format),
}

#[derive(Debug, Clone)]
pub struct PlanItem {
    pub out: String,
    pub src: String,
    pub rows: RowSel,
    pub tier: Tier,
    pub store: Store,
    pub ty: GgmlType,
    pub residency: Residency,
    /// Logical (unpadded) shape after row selection.
    pub logical: Vec<usize>,
    /// Stored (padded) shape.
    pub stored: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct AuxItem {
    pub name: String,
    pub ty: GgmlType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PyramidConfig {
    pub edge_front: usize,
    pub edge_back: usize,
    pub tier2: Format,
    pub tier3: Format,
    pub tier4: Format,
    /// Format of the per-layer-embedding table (disk-gathered, so it costs no VRAM).
    /// `None` = `tier4`.
    pub ple: Option<Format>,
    /// Extra sensitive MLP layers right after the front edge that use `tier3` instead of
    /// `tier4` (Raon-1-nano's early MLPs dominate the Q4_K error; see docs/INFERENCE.md).
    pub mlp_sensitive_front: usize,
    pub guard_clusters: usize,
    pub guard_iters: usize,
}

impl PyramidConfig {
    /// Named recipes.
    /// * `prd`  — the PRD v3.0 pyramid as written (T4 Q4_K for MLP and the PLE table).
    /// * `zena` — the measured quality recipe for Raon-1-nano: MLP Q5_K, the 6 MLPs after the
    ///   front edge Q6_K, PLE table Q8_0 (disk-gathered). Use with an imatrix.
    /// * `q8`   — every quantized tensor Q8_0 (uniform highest-precision quantized baseline).
    pub fn preset(name: &str) -> Result<Self> {
        let d = Self::default();
        Ok(match name {
            "prd" => d,
            "zena" => Self {
                tier4: Format::Q5K,
                ple: Some(Format::Q8_0),
                mlp_sensitive_front: 6,
                ..d
            },
            "q8" => Self {
                tier3: Format::Q8_0,
                tier4: Format::Q8_0,
                ple: Some(Format::Q8_0),
                ..d
            },
            _ => bail!("unknown preset {name} (prd, zena, q8)"),
        })
    }
}

/// The part of the Raon-1-nano plan that a load-time block cache stores: every matrix quantized
/// in place (original names, no vocabulary slicing). `host_gather` also keeps the Q8_0
/// `embed_tokens` (needed when the full head runs on a GPU). Norms, gates and the other T1
/// tensors are not cached — they are read from the originals.
pub fn nano_cache_plan(st: &SafeTensors, cfg: &PyramidConfig, n_layers: usize, host_gather: bool) -> Result<Plan> {
    let vocab = VocabSlice::full(st.info("model.embed_tokens.weight")?.shape[0]);
    let mut plan = plan_raon1_nano(st, cfg, &vocab, n_layers)?;
    plan.items
        .retain(|it| it.out == it.src && matches!(it.store, Store::Quant(_)) && (host_gather || it.residency == Residency::Device));
    plan.aux.clear();
    plan.excluded.clear();
    plan.arch = "raon1_nano_blockcache";
    plan.kvs = vec![
        ("general.architecture".into(), Kv::Str("raon1_nano_blockcache".into())),
        (
            "general.name".into(),
            Kv::Str("Raon-1-nano load-time block cache (Raon Zena)".into()),
        ),
    ];
    Ok(plan)
}

/// Load-time view of the Raon-1-nano plan: original tensor name → quantized format, for every
/// matrix the pyramid quantizes in place (no vocabulary slicing; the gathered tables stay in
/// their original dtype). Used by `raon-infer` to quantize straight from safetensors at load.
pub fn nano_formats(st: &SafeTensors, cfg: &PyramidConfig, n_layers: usize) -> Result<HashMap<String, (Tier, Format, Residency)>> {
    let vocab = VocabSlice::full(st.info("model.embed_tokens.weight")?.shape[0]);
    let plan = plan_raon1_nano(st, cfg, &vocab, n_layers)?;
    Ok(plan
        .items
        .into_iter()
        .filter(|it| it.out == it.src)
        .filter_map(|it| match it.store {
            Store::Quant(f) => Some((it.src, (it.tier, f, it.residency))),
            _ => None,
        })
        .collect())
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            edge_front: 2,
            edge_back: 2,
            tier2: Format::Q8_0,
            tier3: Format::Q6K,
            tier4: Format::Q4K,
            ple: None,
            mlp_sensitive_front: 0,
            guard_clusters: 512,
            guard_iters: 4,
        }
    }
}

pub struct Plan {
    pub arch: &'static str,
    pub items: Vec<PlanItem>,
    pub aux: Vec<AuxItem>,
    pub excluded: Vec<(String, String)>,
    pub kvs: Vec<(String, Kv)>,
}

fn fmt_name(f: Format) -> &'static str {
    f.ggml_type().name()
}

pub fn parse_format(s: &str) -> Result<Format> {
    Ok(match s.to_ascii_lowercase().replace('_', "").as_str() {
        "q80" | "q8" => Format::Q8_0,
        "q6k" | "q6" => Format::Q6K,
        "q5k" | "q5" => Format::Q5K,
        "q4k" | "q4" => Format::Q4K,
        "f16" => Format::F16,
        "f32" => Format::F32,
        _ => bail!("unknown format {s} (q8_0, q6_k, q5_k, q4_k, f16, f32)"),
    })
}

fn pad_to(n: usize, b: usize) -> usize {
    n.div_ceil(b) * b
}

fn layer_of(name: &str, marker: &str) -> Option<usize> {
    let i = name.find(marker)? + marker.len();
    name[i..].split('.').next()?.parse().ok()
}

fn spine(st: &SafeTensors, name: &str, pad_len: Option<usize>) -> Result<PlanItem> {
    let info = st.info(name)?;
    let n = info.numel();
    let (store, ty, stored, logical) = if info.shape.len() <= 1 || pad_len.is_some() {
        let l = pad_len.unwrap_or(n);
        (Store::F32, GgmlType::F32, vec![l], vec![n])
    } else {
        let ty = match info.dtype {
            Dtype::BF16 => GgmlType::BF16,
            Dtype::F16 => GgmlType::F16,
            Dtype::F32 => GgmlType::F32,
        };
        (Store::Raw, ty, info.shape.clone(), info.shape.clone())
    };
    Ok(PlanItem {
        out: name.into(),
        src: name.into(),
        rows: RowSel::All,
        tier: Tier::T1Spine,
        store,
        ty,
        residency: Residency::Device,
        logical,
        stored,
    })
}

#[allow(clippy::too_many_arguments)]
fn matrix(
    st: &SafeTensors,
    src: &str,
    out: &str,
    rows: RowSel,
    tier: Tier,
    f: Format,
    res: Residency,
    pad_rows: Option<usize>,
    pad_cols: Option<usize>,
) -> Result<PlanItem> {
    let info = st.info(src)?;
    if info.shape.len() != 2 {
        bail!("{src}: expected a matrix, got {:?}", info.shape);
    }
    let (r, c) = (info.shape[0], info.shape[1]);
    let lr = match &rows {
        RowSel::All => r,
        RowSel::Ids(ids) => ids.len(),
    };
    let sc = pad_to(pad_cols.unwrap_or(c).max(c), f.block_size());
    let sr = pad_rows.unwrap_or(lr).max(lr);
    Ok(PlanItem {
        out: out.into(),
        src: src.into(),
        rows,
        tier,
        store: Store::Quant(f),
        ty: f.ggml_type(),
        residency: res,
        logical: vec![lr, c],
        stored: vec![sr, sc],
    })
}

fn natural_key(name: &str) -> (usize, String) {
    (
        layer_of(name, ".layers.")
            .or_else(|| layer_of(name, ".blocks."))
            .map(|l| l + 1)
            .unwrap_or(0),
        name.to_string(),
    )
}

fn finish(mut items: Vec<PlanItem>) -> Vec<PlanItem> {
    items.sort_by_key(|a| (a.residency, natural_key(&a.out)));
    items
}

fn i32_bytes(v: impl Iterator<Item = i32>) -> Vec<u8> {
    v.flat_map(|x| x.to_le_bytes()).collect()
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------
// Raon-1-nano
// ---------------------------------------------------------------------------------------------

fn nano_spine(name: &str) -> bool {
    name.contains("raon_mhc.")
        || name.ends_with("_conv1d.weight")
        || name.ends_with(".A_log")
        || name.ends_with(".dt_bias")
        || name.ends_with(".self_attn.b_proj.weight")
        || name.ends_with(".rda_w_proj.weight")
        || name.ends_with(".f_a_proj.weight")
        || name.ends_with(".f_b_proj.weight")
}

fn nano_attention(name: &str) -> bool {
    const P: [&str; 9] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "g_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    ];
    name.contains(".self_attn.") && P.iter().any(|p| name.ends_with(&format!(".{p}.weight")))
}

/// Quantize the cold rows exactly as the main pass will (same encoder, same rows) and cluster
/// their *decoded* values, so the bound holds for the bytes the runtime actually multiplies.
fn cold_guard(st: &SafeTensors, src: &str, cold: &[u32], f: Format, cfg: &PyramidConfig) -> Result<ColdGuard> {
    let cols = st.info(src)?.rows_cols().1;
    let ty = f.ggml_type();
    let rb = ty.row_bytes(cols);
    let mut rows = Vec::with_capacity(cold.len() * cols);
    for chunk in cold.chunks(4096) {
        let mut buf = Vec::with_capacity(chunk.len() * cols);
        st.gather_f32(src, chunk, &mut buf)?;
        let packed = encode(&buf, cols, f, None).map_err(anyhow::Error::msg)?;
        let base = rows.len();
        rows.resize(base + chunk.len() * cols, 0.0);
        for (i, out) in rows[base..].chunks_exact_mut(cols).enumerate() {
            dequant_row(ty, &packed[i * rb..(i + 1) * rb], out);
        }
    }
    Ok(build_cold_guard(&rows, cold, cols, cfg.guard_clusters, cfg.guard_iters))
}

pub fn plan_raon1_nano(st: &SafeTensors, cfg: &PyramidConfig, vocab: &VocabSlice, n_layers: usize) -> Result<Plan> {
    let mut items = Vec::new();
    let mut excluded = Vec::new();
    let mut aux = Vec::new();
    let kept = Arc::new(vocab.kept.clone());
    let cold = Arc::new(vocab.cold.clone());
    let embed = "model.embed_tokens.weight";
    // Tied head: use embed_tokens unless a separate lm_head differs from it.
    let mut head_src = embed.to_string();
    if st.tensors.contains_key("lm_head.weight") {
        let rows = st.info(embed)?.rows_cols().0;
        let tied = [0usize, 1, rows / 3, rows / 2, rows - 1]
            .iter()
            .all(|&r| st.row_bytes(embed, r).ok() == st.row_bytes("lm_head.weight", r).ok());
        if tied {
            excluded.push((
                "lm_head.weight".into(),
                "byte-identical to model.embed_tokens.weight (tie_word_embeddings)".into(),
            ));
        } else {
            head_src = "lm_head.weight".into();
            items.push(matrix(
                st,
                "lm_head.weight",
                "lm_head.weight",
                RowSel::All,
                Tier::T2Gate,
                cfg.tier2,
                Residency::HostGather,
                None,
                None,
            )?);
        }
    }
    for (name, info) in &st.tensors {
        let name = name.as_str();
        if name == "lm_head.weight" {
            continue;
        }
        if name == embed {
            items.push(matrix(
                st,
                embed,
                embed,
                RowSel::All,
                Tier::T2Gate,
                cfg.tier2,
                Residency::HostGather,
                None,
                None,
            )?);
            continue;
        }
        if name == "model.per_layer_inputs.table" {
            items.push(matrix(
                st,
                name,
                "model.per_layer_inputs.table.kept",
                RowSel::Ids(kept.clone()),
                Tier::T4Bulk,
                cfg.ple.unwrap_or(cfg.tier4),
                Residency::HostGather,
                None,
                None,
            )?);
            if !cold.is_empty() {
                items.push(matrix(
                    st,
                    name,
                    "model.per_layer_inputs.table.cold",
                    RowSel::Ids(cold.clone()),
                    Tier::T4Bulk,
                    cfg.ple.unwrap_or(cfg.tier4),
                    Residency::HostGather,
                    None,
                    None,
                )?);
            }
            continue;
        }
        if info.shape.len() <= 1 || nano_spine(name) {
            items.push(spine(st, name, None)?);
            continue;
        }
        let layer = layer_of(name, ".layers.");
        let edge = match layer {
            Some(l) if l >= n_layers => true, // MTP head (model.layers.24.*)
            Some(l) => l < cfg.edge_front || l + cfg.edge_back >= n_layers,
            None => true,
        };
        let (tier, f) = if edge || name.ends_with("per_layer_inputs.embedding_projection") || name.contains(".eh_proj.") {
            (Tier::T2Gate, cfg.tier2)
        } else if nano_attention(name) || name.contains(".per_layer_input_gate.") {
            (Tier::T3Attention, cfg.tier3)
        } else if name.contains(".mlp.") {
            let sensitive = layer.is_some_and(|l| l < cfg.edge_front + cfg.mlp_sensitive_front);
            if sensitive {
                (Tier::T3Attention, cfg.tier3)
            } else {
                (Tier::T4Bulk, cfg.tier4)
            }
        } else {
            (Tier::T2Gate, cfg.tier2)
        };
        items.push(matrix(st, name, name, RowSel::All, tier, f, Residency::Device, None, None)?);
    }
    // Resident sliced head + exactness guard.
    items.push(matrix(
        st,
        &head_src,
        "lm_head.kept",
        RowSel::Ids(kept.clone()),
        Tier::T2Gate,
        cfg.tier2,
        Residency::Device,
        None,
        None,
    )?);
    aux.push(AuxItem {
        name: "raon.vocab.kept_ids".into(),
        ty: GgmlType::I32,
        shape: vec![kept.len()],
        bytes: i32_bytes(kept.iter().map(|&x| x as i32)),
    });
    if !cold.is_empty() {
        let t = Instant::now();
        let g = cold_guard(st, &head_src, &cold, cfg.tier2, cfg)?;
        eprintln!(
            "  cold-row guard: {} rows → {} clusters in {:.1}s",
            cold.len(),
            g.radius.len(),
            t.elapsed().as_secs_f32()
        );
        let c = g.radius.len();
        aux.push(AuxItem {
            name: "lm_head.cold.centers".into(),
            ty: GgmlType::F32,
            shape: vec![c, g.dim],
            bytes: f32_bytes(&g.centers),
        });
        aux.push(AuxItem {
            name: "lm_head.cold.radius".into(),
            ty: GgmlType::F32,
            shape: vec![c],
            bytes: f32_bytes(&g.radius),
        });
        aux.push(AuxItem {
            name: "lm_head.cold.members".into(),
            ty: GgmlType::I32,
            shape: vec![g.members.len()],
            bytes: i32_bytes(g.members.iter().map(|&x| x as i32)),
        });
        aux.push(AuxItem {
            name: "lm_head.cold.offsets".into(),
            ty: GgmlType::I32,
            shape: vec![c + 1],
            bytes: i32_bytes(g.offsets.iter().map(|&x| x as i32)),
        });
    }
    let kvs = vec![
        ("general.architecture".into(), Kv::Str("raon1_nano".into())),
        ("general.name".into(), Kv::Str("KRAFTON/Raon-1-nano (Raon Zena pyramid)".into())),
        ("raon.vocab.size".into(), Kv::U32(vocab.vocab_size as u32)),
        ("raon.vocab.kept".into(), Kv::U32(vocab.kept.len() as u32)),
        ("raon.lm_head.source".into(), Kv::Str(head_src)),
    ];
    Ok(Plan {
        arch: "raon1_nano",
        items: finish(items),
        aux,
        excluded,
        kvs,
    })
}

// ---------------------------------------------------------------------------------------------
// Raon-VisionEncoder (vision tower only)
// ---------------------------------------------------------------------------------------------

/// The vision tower of Raon-VisionEncoder (timm names) or of `google/siglip2-so400m-patch16-naflex` (Hugging Face
/// names: `vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj`, `vision_model.head.*`, `text_model.*`) —
/// the same network, so the same tiers: separate q / k / v projections are T3 like the fused `qkv`, the packed
/// MAP-head `in_proj` is T2 like `attn_pool.q` / `kv`, and `position_embedding` stays in the spine like `pos_embed`.
pub fn plan_raon_ve(st: &SafeTensors, cfg: &PyramidConfig) -> Result<Plan> {
    let hf = st.tensors.keys().any(|n| n.starts_with("vision_model."));
    let marker = if hf { ".layers." } else { ".blocks." };
    let n_blocks = st
        .tensors
        .keys()
        .filter_map(|n| layer_of(n, marker))
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut items = Vec::new();
    let mut excluded = Vec::new();
    let edge = |n: &str| match layer_of(n, marker) {
        Some(l) => l < cfg.edge_front || l + cfg.edge_back >= n_blocks,
        None => true, // attn_pool / head (MAP head) = output boundary
    };
    // The MLP hidden width must agree between fc1 rows and fc2 columns.
    let hidden_pad = |n: &str| -> Result<usize> {
        let fc2 = n.replace("fc1.weight", "fc2.weight").replace("fc1.bias", "fc2.weight");
        let f = if edge(n) { cfg.tier2 } else { cfg.tier4 };
        Ok(pad_to(st.info(&fc2)?.shape[1], f.block_size()))
    };
    for (name, info) in &st.tensors {
        let name = name.as_str();
        if name.starts_with("text.") || name.starts_with("text_model.") {
            excluded.push((
                name.to_string(),
                "contrastive text tower (0.71B) excluded; vision path only".to_string(),
            ));
            continue;
        }
        if name.ends_with("mlp.fc1.bias") {
            items.push(spine(st, name, Some(hidden_pad(name)?))?);
            continue;
        }
        if info.shape.len() != 2
            || name.contains("patch_embed")
            || name.contains("pos_embed")
            || name.contains("position_embedding")
        {
            items.push(spine(st, name, None)?);
            continue;
        }
        let e = edge(name);
        let (tier, f) = if e {
            (Tier::T2Gate, cfg.tier2)
        } else if name.contains(".attn.") || name.contains(".self_attn.") {
            (Tier::T3Attention, cfg.tier3)
        } else {
            (Tier::T4Bulk, cfg.tier4)
        };
        let item = if name.ends_with("mlp.fc1.weight") {
            matrix(
                st,
                name,
                name,
                RowSel::All,
                tier,
                f,
                Residency::Device,
                Some(hidden_pad(name)?),
                None,
            )?
        } else {
            matrix(st, name, name, RowSel::All, tier, f, Residency::Device, None, None)?
        };
        items.push(item);
    }
    let kvs = vec![
        ("general.architecture".into(), Kv::Str("raon_ve_vision".into())),
        (
            "general.name".into(),
            Kv::Str(if hf {
                "google/siglip2-so400m-patch16-naflex vision tower (Raon Zena pyramid)".into()
            } else {
                "KRAFTON/Raon-VisionEncoder vision tower (Raon Zena pyramid)".into()
            }),
        ),
        ("raon_ve.block_count".into(), Kv::U32(n_blocks as u32)),
        ("raon_ve.patch_size".into(), Kv::U32(16)),
        ("raon_ve.embedding_length".into(), Kv::U32(1152)),
        ("raon_ve.naflex.default_max_patches".into(), Kv::U32(256)),
    ];
    Ok(Plan {
        arch: "raon_ve_vision",
        items: finish(items),
        aux: vec![],
        excluded,
        kvs,
    })
}

// ---------------------------------------------------------------------------------------------
// Converter
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ItemReport {
    pub name: String,
    pub src: String,
    pub tier: Tier,
    pub ty: &'static str,
    pub residency: Residency,
    pub logical: Vec<usize>,
    pub stored: Vec<usize>,
    pub params: usize,
    pub bytes: usize,
    /// ‖x̂ − x‖ / ‖x‖ over the logical values (0 for lossless tiers).
    pub rel_err: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Totals {
    pub tensors: usize,
    pub params: usize,
    pub bytes: usize,
    pub bpw: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub arch: &'static str,
    pub out: String,
    pub file_bytes: u64,
    pub seconds: f32,
    pub config: BTreeMap<String, String>,
    pub by_tier: BTreeMap<String, Totals>,
    pub by_residency: BTreeMap<String, Totals>,
    pub excluded: Vec<(String, String)>,
    pub excluded_params: usize,
    pub items: Vec<ItemReport>,
}

fn add(t: &mut Totals, params: usize, bytes: usize) {
    t.tensors += 1;
    t.params += params;
    t.bytes += bytes;
    t.bpw = t.bytes as f32 * 8.0 / t.params.max(1) as f32;
}

fn rows_of(it: &PlanItem, total: usize) -> Vec<u32> {
    match &it.rows {
        RowSel::All => (0..total as u32).collect(),
        RowSel::Ids(ids) => ids.to_vec(),
    }
}

pub fn convert(
    st: &SafeTensors,
    plan: &Plan,
    cfg: &PyramidConfig,
    out: &Path,
    extra_kvs: Vec<(String, Kv)>,
    imatrix: Option<&HashMap<String, Vec<f32>>>,
) -> Result<Report> {
    convert_with(st, plan, cfg, out, extra_kvs, imatrix, &|k, n, it, rep, secs| {
        eprintln!(
            "[{:>4}/{}] {:<62} {:?} {:>5} {:>12?} rel_err={:.5} {:.1}s",
            k + 1,
            n,
            it.out,
            it.tier,
            it.ty.name(),
            it.stored,
            rep.rel_err,
            secs
        );
    })
}

/// Progress callback of [`convert_with`]: (index, total, item, its report, seconds).
pub type OnItem<'a> = &'a (dyn Fn(usize, usize, &PlanItem, &ItemReport, f32) + Sync);

/// [`convert`] with a caller-supplied progress callback (the runtime block cache logs sparsely).
pub fn convert_with(
    st: &SafeTensors,
    plan: &Plan,
    cfg: &PyramidConfig,
    out: &Path,
    extra_kvs: Vec<(String, Kv)>,
    imatrix: Option<&HashMap<String, Vec<f32>>>,
    on_item: OnItem,
) -> Result<Report> {
    let t0 = Instant::now();
    let mut config = BTreeMap::new();
    config.insert("recipe".into(), "pyramid-4tier/1".into());
    config.insert("edge_front".into(), cfg.edge_front.to_string());
    config.insert("edge_back".into(), cfg.edge_back.to_string());
    config.insert("tier2".into(), fmt_name(cfg.tier2).into());
    config.insert("tier3".into(), fmt_name(cfg.tier3).into());
    config.insert("tier4".into(), fmt_name(cfg.tier4).into());
    config.insert("ple".into(), fmt_name(cfg.ple.unwrap_or(cfg.tier4)).into());
    config.insert("mlp_sensitive_front".into(), cfg.mlp_sensitive_front.to_string());
    config.insert("imatrix".into(), imatrix.is_some().to_string());
    let mut meta = serde_json::Map::new();
    for it in &plan.items {
        meta.insert(
            it.out.clone(),
            serde_json::json!({"tier": it.tier, "residency": it.residency, "logical": it.logical, "src": it.src}),
        );
    }
    for a in &plan.aux {
        meta.insert(
            a.name.clone(),
            serde_json::json!({"tier": Tier::Aux, "residency": Residency::HostGather, "logical": a.shape}),
        );
    }
    let mut kvs: Vec<(String, Kv)> = plan.kvs.clone();
    kvs.push(("general.quantized_by".into(), Kv::Str("raon-quant (Raon Zena)".into())));
    kvs.push(("raon.quant.config".into(), Kv::Str(serde_json::to_string(&config)?)));
    kvs.push(("raon.tensor_meta".into(), Kv::Str(serde_json::Value::Object(meta).to_string())));
    kvs.extend(extra_kvs);
    let mut decls: Vec<TensorDecl> = plan
        .items
        .iter()
        .map(|it| TensorDecl {
            name: it.out.clone(),
            ty: it.ty,
            shape: it.stored.clone(),
        })
        .collect();
    decls.extend(plan.aux.iter().map(|a| TensorDecl {
        name: a.name.clone(),
        ty: a.ty,
        shape: a.shape.clone(),
    }));
    let mut w = GgufWriter::create(out, &kvs, decls)?;
    let mut reports = Vec::new();
    for (k, it) in plan.items.iter().enumerate() {
        let t = Instant::now();
        let rep = write_item(st, it, &mut w, imatrix)?;
        on_item(k, plan.items.len(), it, &rep, t.elapsed().as_secs_f32());
        reports.push(rep);
    }
    for a in &plan.aux {
        w.write(&a.bytes)?;
        let params = a.shape.iter().product();
        reports.push(ItemReport {
            name: a.name.clone(),
            src: String::new(),
            tier: Tier::Aux,
            ty: a.ty.name(),
            residency: Residency::HostGather,
            logical: a.shape.clone(),
            stored: a.shape.clone(),
            params,
            bytes: a.bytes.len(),
            rel_err: 0.0,
        });
    }
    let path = w.finish()?;
    let mut by_tier = BTreeMap::new();
    let mut by_res = BTreeMap::new();
    for r in &reports {
        add(by_tier.entry(format!("{:?}", r.tier)).or_default(), r.params, r.bytes);
        if r.tier != Tier::Aux {
            add(by_res.entry(format!("{:?}", r.residency)).or_default(), r.params, r.bytes);
        }
    }
    let excluded_params = plan.excluded.iter().filter_map(|(n, _)| st.info(n).ok()).map(|i| i.numel()).sum();
    Ok(Report {
        arch: plan.arch,
        out: path.display().to_string(),
        file_bytes: std::fs::metadata(&path)?.len(),
        seconds: t0.elapsed().as_secs_f32(),
        config,
        by_tier,
        by_residency: by_res,
        excluded: plan.excluded.clone(),
        excluded_params,
        items: reports,
    })
}

fn write_item(st: &SafeTensors, it: &PlanItem, w: &mut GgufWriter, imatrix: Option<&HashMap<String, Vec<f32>>>) -> Result<ItemReport> {
    let info = st.info(&it.src)?;
    let params: usize = it.logical.iter().product();
    let bytes = TensorDecl {
        name: String::new(),
        ty: it.ty,
        shape: it.stored.clone(),
    }
    .nbytes();
    let mut rel_err = 0f32;
    match it.store {
        Store::Raw => {
            let want = match it.ty {
                GgmlType::BF16 => Dtype::BF16,
                GgmlType::F16 => Dtype::F16,
                GgmlType::F32 => Dtype::F32,
                t => bail!("{}: raw copy into {t:?}", it.src),
            };
            if info.dtype != want {
                bail!("{}: raw copy dtype mismatch", it.src);
            }
            for c in st.bytes(&it.src)?.chunks(64 << 20) {
                w.write(c)?;
            }
        }
        Store::F32 => {
            let mut v = st.all_f32(&it.src)?;
            v.resize(it.stored.iter().product(), 0.0);
            w.write(&f32_bytes(&v))?;
        }
        Store::Quant(f) => {
            let (lr, lc) = (it.logical[0], it.logical[1]);
            let (sr, sc) = (it.stored[0], it.stored[1]);
            let total_rows = info.rows_cols().0;
            let ids = rows_of(it, total_rows);
            debug_assert_eq!(ids.len(), lr);
            let imat: Option<Vec<f32>> = imatrix.and_then(|m| m.get(&it.src)).and_then(|v| {
                if v.len() == lc {
                    let mut p = v.clone();
                    p.resize(sc, 1.0);
                    Some(p)
                } else {
                    eprintln!("  warning: imatrix for {} has {} values, expected {lc}; ignored", it.src, v.len());
                    None
                }
            });
            let rb = it.ty.row_bytes(sc);
            let chunk_rows = ((16usize << 20) / sc).max(1);
            let (mut num, mut den) = (0f64, 0f64);
            let mut dq = vec![0f32; sc];
            for chunk in ids.chunks(chunk_rows) {
                let mut raw = Vec::with_capacity(chunk.len() * lc);
                match &it.rows {
                    RowSel::All => st.rows_f32(&it.src, chunk[0] as usize, chunk[0] as usize + chunk.len(), &mut raw)?,
                    RowSel::Ids(_) => st.gather_f32(&it.src, chunk, &mut raw)?,
                }
                let buf = if sc == lc {
                    raw
                } else {
                    let mut p = vec![0f32; chunk.len() * sc];
                    for (dst, src) in p.chunks_exact_mut(sc).zip(raw.chunks_exact(lc)) {
                        dst[..lc].copy_from_slice(src);
                    }
                    p
                };
                let packed = encode(&buf, sc, f, imat.as_deref()).map_err(|e| anyhow::anyhow!("{}: {e}", it.src))?;
                w.write(&packed)?;
                for (r, x) in buf.chunks_exact(sc).enumerate() {
                    dequant_row(it.ty, &packed[r * rb..(r + 1) * rb], &mut dq);
                    for (a, b) in dq[..lc].iter().zip(&x[..lc]) {
                        num += ((a - b) as f64).powi(2);
                        den += (*b as f64).powi(2);
                    }
                }
            }
            if sr > lr {
                let zeros = vec![0f32; (sr - lr) * sc];
                w.write(&encode(&zeros, sc, f, None).map_err(anyhow::Error::msg)?)?;
            }
            rel_err = (num / den.max(1e-30)).sqrt() as f32;
        }
    }
    Ok(ItemReport {
        name: it.out.clone(),
        src: it.src.clone(),
        tier: it.tier,
        ty: it.ty.name(),
        residency: it.residency,
        logical: it.logical.clone(),
        stored: it.stored.clone(),
        params,
        bytes,
        rel_err,
    })
}

/// Read `config.json` next to the weights (stored verbatim in the GGUF for the runtime).
pub fn source_config_kv(dir: &Path) -> Option<(String, Kv)> {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .map(|s| ("raon.source_config".to_string(), Kv::Str(s)))
}

pub fn n_layers_from_config(dir: &Path) -> Result<usize> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).context("config.json")?)?;
    v["num_hidden_layers"].as_u64().map(|x| x as usize).context("num_hidden_layers")
}
