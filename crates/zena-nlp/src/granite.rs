//! `ibm-granite/granite-embedding-97m-multilingual-r2` sentence embeddings on candle.
//!
//! The model is a 12-layer ModernBERT encoder (hidden 384, 12 heads of 32, GLU MLP of 1536, bias-free LayerNorm,
//! no linear biases) with a 180k-token multilingual byte-level BPE vocabulary. Its official sentence-transformers
//! pipeline (the Hub's `modules.json`: Transformer → Pooling with `pooling_mode_cls_token` → Normalize; `config.json`
//! also says `classifier_pooling: "cls"`) is the L2-normalised final hidden state of the first token, the
//! tokenizer's `<|startoftext|>` [CLS]. [`Granite`] computes exactly that in F32; against Hugging Face transformers
//! (float32, CPU) every reference sentence has cosine ≥ 0.9999 (`tests/granite_reference.rs`, fixture written by
//! `scripts/granite_reference.py`).
//!
//! # Forward pass
//!
//! A port of the `ForwardMode::Exact` path of Qntology/logis-center (`commerce/src-tauri/src/models/embedding.rs`),
//! which mirrors transformers' `ModernBertModel`:
//!
//! * embeddings: token embedding, then LayerNorm;
//! * each layer: `x += Wo(attention(attn_norm(x)))`, then `x += Wo(act(a) * b)` where `[a | b] = Wi(mlp_norm(x))`;
//!   layer 0 has no `attn_norm` (identity), as in transformers;
//! * attention: layers with `index % global_attn_every_n_layers == 0` are global and use RoPE with
//!   `global_rope_theta`; the others are local — a token sees only tokens at most `local_attention / 2` positions away
//!   — and use `local_rope_theta` (both tables are F32, like transformers);
//! * `final_norm`, then the [CLS] row, L2-normalised.
//!
//! # Truncation
//!
//! Input longer than [`MAX_TOKENS`] (512 tokens, [CLS] and [SEP] included) is cut from the right before [SEP] is
//! appended — the tokenizer's own right / longest-first truncation, i.e. sentence-transformers with
//! `max_seq_length = 512`. The checkpoint accepts far longer input (`max_position_embeddings` 32768), but attention
//! time and memory grow with the square of the length, Raon Zena embeds short spans, and 512 is logis-center's
//! forward length too (its Exact CPU vectors for inputs of at most 512 tokens are the same computation). Callers that
//! need a whole long document should split it themselves; [`Granite::token_count`] tells when truncation happens.
//!
//! # Batching
//!
//! [`Granite::embed_batch`] sorts the inputs by token count, packs neighbours into forward passes of at most
//! `BATCH_TOKENS` padded tokens, right-pads with the tokenizer's pad id and masks padded keys. Positions start at 0
//! in every row, so the real tokens see exactly what they see in a single-text pass and the vectors equal
//! [`Granite::embed`]'s up to float summation order.

use std::fmt;
use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Embedding space tag stored next to every vector / SDS ledger made with this model.
pub const SPACE_TAG: &str = "granite-97m-r2-cls-v1";
/// Embedding dimension (the model's `hidden_size`).
pub const DIM: usize = 384;
/// Longest input the model sees, in tokens, [CLS] and [SEP] included; longer input is truncated from the right.
pub const MAX_TOKENS: usize = 512;
/// Most padded tokens (rows × longest row) in one batched forward pass. At 512 tokens that is 8 rows, whose
/// attention scores take 8 × 12 × 512 × 512 × 4 bytes = 100 MB.
const BATCH_TOKENS: usize = 4096;
/// Additive attention mask for keys a query must not see. Finite (transformers uses the F32 minimum) so that two
/// masks can be added and a padded query row whose whole window is padding stays finite; `exp` of it is exactly 0.
const MASKED: f32 = -1e30;

/// The `config.json` fields this forward pass reads. Defaults are transformers' `ModernBertConfig` defaults.
#[derive(Debug, Clone, Deserialize)]
struct Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "Config::default_norm_eps")]
    norm_eps: f64,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default = "Config::default_activation")]
    hidden_activation: String,
    #[serde(default = "Config::default_global_rope_theta")]
    global_rope_theta: f64,
    #[serde(default = "Config::default_local_rope_theta")]
    local_rope_theta: f64,
    #[serde(default = "Config::default_local_attention")]
    local_attention: usize,
    #[serde(default = "Config::default_global_attn_every_n_layers")]
    global_attn_every_n_layers: usize,
    pad_token_id: Option<u32>,
}

impl Config {
    fn default_norm_eps() -> f64 {
        1e-5
    }
    fn default_activation() -> String {
        "gelu".into()
    }
    fn default_global_rope_theta() -> f64 {
        160_000.0
    }
    fn default_local_rope_theta() -> f64 {
        10_000.0
    }
    fn default_local_attention() -> usize {
        128
    }
    fn default_global_attn_every_n_layers() -> usize {
        3
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size == DIM,
            "granite: hidden_size {} (expected {DIM})",
            self.hidden_size
        );
        ensure!(
            self.num_attention_heads > 0
                && self.hidden_size.is_multiple_of(self.num_attention_heads)
                && (self.hidden_size / self.num_attention_heads).is_multiple_of(2),
            "granite: hidden_size {} does not split into {} even-sized heads",
            self.hidden_size,
            self.num_attention_heads
        );
        ensure!(
            !self.norm_bias && !self.attention_bias && !self.mlp_bias,
            "granite: biased norms or linear layers are not supported (norm_bias {}, attention_bias {}, mlp_bias {})",
            self.norm_bias,
            self.attention_bias,
            self.mlp_bias
        );
        ensure!(self.global_attn_every_n_layers > 0, "granite: global_attn_every_n_layers is 0");
        Activation::parse(&self.hidden_activation)?;
        Ok(())
    }
}

/// The MLP gate activation (`hidden_activation`; granite uses `silu`).
#[derive(Debug, Clone, Copy)]
enum Activation {
    Silu,
    /// transformers' `gelu`, the exact erf form.
    Gelu,
}

impl Activation {
    fn parse(name: &str) -> Result<Self> {
        match name {
            "silu" | "swish" => Ok(Self::Silu),
            "gelu" => Ok(Self::Gelu),
            other => bail!("granite: unsupported hidden_activation {other:?}"),
        }
    }

    fn apply(self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Silu => x.silu(),
            Self::Gelu => x.gelu_erf(),
        }
    }
}

/// LayerNorm without bias. Two-pass mean / variance, like PyTorch (not the fused E[x²] − E[x]² kernel).
struct LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centred = x.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(D::Minus1)?;
        centred.broadcast_div(&(var + self.eps)?.sqrt()?)?.broadcast_mul(&self.weight)
    }
}

/// RoPE tables for positions `0..MAX_TOKENS`, `[MAX_TOKENS, head_dim / 2]` each (rotate-half layout).
struct Rope {
    cos: Tensor,
    sin: Tensor,
    /// candle-nn's fused kernel exists on CPU, CUDA, Metal and ROCm; elsewhere the same formula runs as tensor ops.
    fused: bool,
}

impl Rope {
    fn new(head_dim: usize, theta: f64, device: &Device) -> candle_core::Result<Self> {
        let (cos, sin) = rope_tables(head_dim, theta, MAX_TOKENS);
        let half = head_dim / 2;
        Ok(Self {
            cos: Tensor::from_vec(cos, (MAX_TOKENS, half), device)?,
            sin: Tensor::from_vec(sin, (MAX_TOKENS, half), device)?,
            fused: device.is_cpu() || device.is_cuda() || device.is_metal() || device.is_rocm(),
        })
    }

    /// `x` is `[batch, heads, len, head_dim]` and contiguous.
    fn apply(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let len = x.dim(2)?;
        let cos = self.cos.narrow(0, 0, len)?;
        let sin = self.sin.narrow(0, 0, len)?;
        if self.fused {
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope_slow(x, &cos, &sin)
        }
    }
}

/// cos / sin of `position × inv_freq`, row-major `[len, head_dim / 2]`. Like transformers, `inv_freq` and the angle
/// are F32 (`1 / θ^(2i / head_dim)`, then one F32 product); cos / sin of that F32 angle are rounded from F64.
fn rope_tables(head_dim: usize, theta: f64, len: usize) -> (Vec<f32>, Vec<f32>) {
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / (theta as f32).powf(i as f32 / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(len * inv_freq.len());
    let mut sin = Vec::with_capacity(len * inv_freq.len());
    for pos in 0..len {
        for f in &inv_freq {
            let angle = f64::from(pos as f32 * f);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// Additive sliding-window mask `[1, 1, len, len]`: 0 where `|i − j| ≤ half`, [`MASKED`] elsewhere.
/// `None` when the window covers the whole sequence.
fn window_mask(len: usize, half: usize, device: &Device) -> candle_core::Result<Option<Tensor>> {
    if len <= half + 1 {
        return Ok(None);
    }
    let mask: Vec<f32> = (0..len)
        .flat_map(|i| (0..len).map(move |j| if i.abs_diff(j) > half { MASKED } else { 0.0 }))
        .collect();
    Tensor::from_vec(mask, (1, 1, len, len), device).map(Some)
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            wqkv: candle_nn::linear_no_bias(h, 3 * h, vb.pp("Wqkv"))?,
            wo: candle_nn::linear_no_bias(h, h, vb.pp("Wo"))?,
            heads: cfg.num_attention_heads,
            head_dim: h / cfg.num_attention_heads,
        })
    }

    /// `x` is `[batch, len, hidden]`; `mask` is additive and broadcasts to `[batch, heads, len, len]`.
    fn forward(&self, x: &Tensor, rope: &Rope, mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (b, len, hidden) = x.dims3()?;
        let qkv = self.wqkv.forward(x)?.reshape((b, len, 3, self.heads, self.head_dim))?;
        // [batch, len, heads, head_dim] → [batch, heads, len, head_dim]
        let part = |i: usize| qkv.i((.., .., i))?.transpose(1, 2)?.contiguous();
        let q = rope.apply(&part(0)?)?;
        let k = rope.apply(&part(1)?)?;
        let v = part(2)?;
        let scores = (q.matmul(&k.t()?)? * (self.head_dim as f64).powf(-0.5))?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let y = probs.matmul(&v)?.transpose(1, 2)?.reshape((b, len, hidden))?;
        self.wo.forward(&y)
    }
}

struct Mlp {
    wi: Linear,
    wo: Linear,
    act: Activation,
    inner: usize,
}

impl Mlp {
    fn load(cfg: &Config, act: Activation, vb: VarBuilder) -> candle_core::Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            wi: candle_nn::linear_no_bias(h, 2 * i, vb.pp("Wi"))?,
            wo: candle_nn::linear_no_bias(i, h, vb.pp("Wo"))?,
            act,
            inner: i,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let wi = self.wi.forward(x)?;
        // transformers: `input, gate = Wi(x).chunk(2)`; `Wo(act(input) * gate)`
        let input = wi.narrow(D::Minus1, 0, self.inner)?;
        let gate = wi.narrow(D::Minus1, self.inner, self.inner)?;
        self.wo.forward(&(self.act.apply(&input)? * gate)?)
    }
}

struct Layer {
    /// `None` for layer 0 (transformers uses `nn.Identity` there).
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
    global: bool,
}

struct Encoder {
    tok_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    global_rope: Rope,
    local_rope: Rope,
    window_half: usize,
}

impl Encoder {
    fn load(cfg: &Config, act: Activation, vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        let (h, eps) = (cfg.hidden_size, cfg.norm_eps);
        let head_dim = h / cfg.num_attention_heads;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb = vb.pp(format!("layers.{i}"));
            layers.push(Layer {
                attn_norm: if i == 0 {
                    None
                } else {
                    Some(LayerNorm::load(h, eps, vb.pp("attn_norm"))?)
                },
                attn: Attention::load(cfg, vb.pp("attn"))?,
                mlp_norm: LayerNorm::load(h, eps, vb.pp("mlp_norm"))?,
                mlp: Mlp::load(cfg, act, vb.pp("mlp"))?,
                global: i % cfg.global_attn_every_n_layers == 0,
            });
        }
        Ok(Self {
            tok_embeddings: candle_nn::embedding(cfg.vocab_size, h, vb.pp("embeddings.tok_embeddings"))?,
            embeddings_norm: LayerNorm::load(h, eps, vb.pp("embeddings.norm"))?,
            layers,
            final_norm: LayerNorm::load(h, eps, vb.pp("final_norm"))?,
            global_rope: Rope::new(head_dim, cfg.global_rope_theta, device)?,
            local_rope: Rope::new(head_dim, cfg.local_rope_theta, device)?,
            window_half: cfg.local_attention / 2,
        })
    }

    /// `ids` is `[batch, len]`; `pad_mask` (additive, `[batch, 1, 1, len]`) hides padded keys.
    /// Returns the final-normed [CLS] rows, `[batch, hidden]`.
    fn forward_cls(&self, ids: &Tensor, pad_mask: Option<&Tensor>) -> candle_core::Result<Tensor> {
        let (_, len) = ids.dims2()?;
        let mut x = self.embeddings_norm.forward(&self.tok_embeddings.forward(ids)?)?;
        let local_mask = match (pad_mask, window_mask(len, self.window_half, ids.device())?) {
            (Some(p), Some(w)) => Some(p.broadcast_add(&w)?),
            (None, w) => w,
            (p, None) => p.cloned(),
        };
        for layer in &self.layers {
            let (rope, mask) = if layer.global {
                (&self.global_rope, pad_mask)
            } else {
                (&self.local_rope, local_mask.as_ref())
            };
            let h = match &layer.attn_norm {
                Some(norm) => norm.forward(&x)?,
                None => x.clone(),
            };
            x = (&x + layer.attn.forward(&h, rope, mask)?)?;
            x = (&x + layer.mlp.forward(&layer.mlp_norm.forward(&x)?)?)?;
        }
        // LayerNorm works per token, so normalising only the pooled row is exact.
        self.final_norm.forward(&x.i((.., 0, ..))?)
    }
}

/// granite-embedding-97m-multilingual-r2: L2-normalised CLS sentence embeddings, [`DIM`] floats, F32 on any candle
/// device. `Send + Sync`; `embed` / `embed_batch` take `&self`.
pub struct Granite {
    tokenizer: Tokenizer,
    encoder: Encoder,
    device: Device,
    pad_id: u32,
}

impl fmt::Debug for Granite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Granite")
            .field("space_tag", &SPACE_TAG)
            .field("dim", &DIM)
            .field("layers", &self.encoder.layers.len())
            .field("max_tokens", &MAX_TOKENS)
            .field("device", &self.device)
            .finish()
    }
}

impl Granite {
    /// `dir` has config.json, model.safetensors, tokenizer.json. Weights are loaded in F32 on `device`.
    pub fn load(dir: &Path, device: &Device) -> Result<Granite> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let tokenizer_path = dir.join("tokenizer.json");
        let cfg: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).with_context(|| format!("granite: reading {}", config_path.display()))?,
        )
        .with_context(|| format!("granite: parsing {}", config_path.display()))?;
        cfg.validate()?;
        let act = Activation::parse(&cfg.hidden_activation)?;

        let mut tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("granite: loading {}: {e}", tokenizer_path.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| anyhow!("granite: setting truncation: {e}"))?;
        let pad_id = cfg
            .pad_token_id
            .or_else(|| tokenizer.get_padding().map(|p| p.pad_id))
            .context("granite: neither config.json nor tokenizer.json has a pad token id")?;
        // tokenizer.json carries batch padding; embed_ids pads batches itself.
        tokenizer.with_padding(None);

        ensure!(weights_path.is_file(), "granite: {} not found", weights_path.display());
        // SAFETY: the safetensors file is memory-mapped only while the weights are read and converted to owned F32
        // tensors below; nothing writes to the model folder meanwhile.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, device) }
            .with_context(|| format!("granite: mapping {}", weights_path.display()))?;
        let encoder = Encoder::load(&cfg, act, vb, device).with_context(|| format!("granite: loading weights from {}", dir.display()))?;
        Ok(Granite {
            tokenizer,
            encoder,
            device: device.clone(),
            pad_id,
        })
    }

    pub fn dim(&self) -> usize {
        DIM
    }

    pub fn space_tag(&self) -> &'static str {
        SPACE_TAG
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Tokens the model sees for `text`: [CLS] and [SEP] included, at most [`MAX_TOKENS`] (so a result equal to
    /// `MAX_TOKENS` may mean the text was truncated).
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text)?.len())
    }

    /// L2-normalised CLS embedding of `text`, [`DIM`] floats.
    ///
    /// Input longer than [`MAX_TOKENS`] tokens (512, [CLS] and [SEP] included) is truncated from the right — see the
    /// module documentation. The empty string is embedded as `[CLS] [SEP]`, as transformers does.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let ids = self.encode(text)?;
        let mut v = self.embed_ids(&[&ids])?;
        Ok(v.pop().expect("one row in, one vector out"))
    }

    /// The vectors [`Granite::embed`] gives for each text, in input order, computed in padded batches.
    ///
    /// Texts are grouped by token count into forward passes of at most 4096 padded tokens; padding is right-sided
    /// and masked, so each vector equals the single-text one up to float summation order (cosine ≥ 0.9999,
    /// `tests/granite_reference.rs`). An empty slice gives an empty result.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let ids = texts.iter().map(|t| self.encode(t)).collect::<Result<Vec<_>>>()?;
        let lens: Vec<usize> = ids.iter().map(Vec::len).collect();
        let mut out = vec![Vec::new(); texts.len()];
        for group in batch_groups(&lens, BATCH_TOKENS) {
            let rows: Vec<&[u32]> = group.iter().map(|&i| ids[i].as_slice()).collect();
            for (&i, v) in group.iter().zip(self.embed_ids(&rows)?) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("granite: tokenizing failed: {e}"))?;
        ensure!(
            !enc.get_ids().is_empty(),
            "granite: the tokenizer produced no tokens (no [CLS] template?)"
        );
        Ok(enc.get_ids().to_vec())
    }

    /// One forward pass over `rows` (right-padded to the longest), returning their normalised CLS vectors.
    fn embed_ids(&self, rows: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        let (b, len) = (rows.len(), rows.iter().map(|r| r.len()).max().unwrap_or(0));
        ensure!(b > 0 && len > 0, "granite: empty forward pass");
        let mut ids = Vec::with_capacity(b * len);
        let mut mask = Vec::with_capacity(b * len);
        for r in rows {
            ids.extend_from_slice(r);
            ids.resize(ids.len() + len - r.len(), self.pad_id);
            mask.resize(mask.len() + r.len(), 0f32);
            mask.resize(mask.len() + len - r.len(), MASKED);
        }
        let ids = Tensor::from_vec(ids, (b, len), &self.device)?;
        let pad_mask = if rows.iter().any(|r| r.len() < len) {
            Some(Tensor::from_vec(mask, (b, 1, 1, len), &self.device)?)
        } else {
            None
        };
        let cls = self.encoder.forward_cls(&ids, pad_mask.as_ref())?;
        let cls = cls.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        Ok(cls.into_iter().map(l2_normalize).collect())
    }
}

/// `v / max(‖v‖, 1e-12)` (sentence-transformers' `Normalize`), so a zero vector stays zero instead of NaN.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt().max(1e-12) as f32;
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// Splits row indices into forward passes: rows sorted by length (stable), neighbours packed while
/// `rows × longest row ≤ budget`; a row longer than the budget still gets a pass of its own.
fn batch_groups(lens: &[usize], budget: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let mut groups = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for i in order {
        // Sorted ascending: `lens[i]` would be the longest row of the grown group.
        if !current.is_empty() && (current.len() + 1) * lens[i] > budget {
            groups.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granite_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Granite>();
    }

    #[test]
    fn batch_groups_cover_every_row_once_within_budget() {
        let lens = [9, 512, 3, 9, 130, 2, 512, 600, 64];
        let groups = batch_groups(&lens, 1024);
        let mut seen: Vec<usize> = groups.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..lens.len()).collect::<Vec<_>>());
        for g in &groups {
            let longest = g.iter().map(|&i| lens[i]).max().unwrap();
            assert!(g.len() == 1 || g.len() * longest <= 1024, "{g:?}");
            assert!(g.windows(2).all(|w| lens[w[0]] <= lens[w[1]]), "rows are sorted by length: {g:?}");
        }
        assert_eq!(groups.last().unwrap(), &vec![7], "an over-budget row gets its own pass");
        assert!(batch_groups(&[], 1024).is_empty());
        assert_eq!(batch_groups(&[5; 16], 4096), vec![(0..16).collect::<Vec<_>>()]);
    }

    #[test]
    fn window_mask_hides_tokens_beyond_half_the_window() {
        assert!(window_mask(65, 64, &Device::Cpu).unwrap().is_none(), "distance ≤ 64 everywhere");
        let m = window_mask(70, 64, &Device::Cpu).unwrap().unwrap();
        assert_eq!(m.dims(), &[1, 1, 70, 70]);
        let m = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let at = |i: usize, j: usize| m[i * 70 + j];
        assert_eq!(at(0, 64), 0.0);
        assert_eq!(at(0, 65), MASKED);
        assert_eq!(at(69, 5), 0.0);
        assert_eq!(at(69, 4), MASKED);
        assert_eq!(at(30, 30), 0.0);
    }

    #[test]
    fn rope_tables_start_at_identity_and_follow_the_formula() {
        let (cos, sin) = rope_tables(32, 160_000.0, 4);
        assert_eq!(cos.len(), 4 * 16);
        assert!(cos[..16].iter().all(|&c| c == 1.0) && sin[..16].iter().all(|&s| s == 0.0));
        // position 3, frequency index 1: θ^(−2/32)
        let angle = 3.0 * 160_000f64.powf(-2.0 / 32.0);
        assert!((f64::from(cos[3 * 16 + 1]) - angle.cos()).abs() < 1e-6);
        assert!((f64::from(sin[3 * 16 + 1]) - angle.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_tensor_op_fallback_matches_the_fused_kernel() {
        let rope = Rope::new(32, 160_000.0, &Device::Cpu).unwrap();
        assert!(rope.fused);
        let x = Tensor::randn(0f32, 1.0, (2, 3, 70, 32), &Device::Cpu).unwrap();
        let fused = rope.apply(&x).unwrap();
        let slow = Rope { fused: false, ..rope }.apply(&x).unwrap();
        let diff = (fused - slow).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap();
        assert!(diff.to_scalar::<f32>().unwrap() < 1e-6);
    }

    #[test]
    fn l2_normalize_gives_unit_vectors_and_keeps_zero() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-7 && (v[1] - 0.8).abs() < 1e-7);
        assert_eq!(l2_normalize(vec![0.0; 4]), vec![0.0; 4]);
    }

    #[test]
    fn config_reads_the_granite_fields_and_rejects_what_the_port_does_not_do() {
        let json = r#"{"vocab_size": 180000, "hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 12,
            "num_attention_heads": 12, "norm_eps": 1e-05, "norm_bias": false, "attention_bias": false, "mlp_bias": false,
            "hidden_activation": "silu", "global_rope_theta": 150000.0, "local_rope_theta": 160000.0,
            "local_attention": 128, "global_attn_every_n_layers": 3, "pad_token_id": 179935}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            (cfg.global_rope_theta, cfg.local_rope_theta, cfg.local_attention),
            (150_000.0, 160_000.0, 128)
        );
        assert_eq!(cfg.pad_token_id, Some(179_935));
        let biased: Config = serde_json::from_str(&json.replace(r#""mlp_bias": false"#, r#""mlp_bias": true"#)).unwrap();
        assert!(biased.validate().is_err());
        let wide: Config = serde_json::from_str(&json.replace(r#""hidden_size": 384"#, r#""hidden_size": 768"#)).unwrap();
        assert!(wide.validate().is_err());
    }
}
