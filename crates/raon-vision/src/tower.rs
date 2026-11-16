//! Raon-VisionEncoder's vision tower on candle: the timm `vit_so400m_patch16_siglip_256` trunk (27 pre-norm blocks,
//! fused qkv, GELU-tanh MLP) and its MAP head (`attn_pool`: one latent query), in the released NaFlex 1-D mode — each
//! 16×16 patch through the patch projection as a linear map, the 16×16 position table bilinearly resized to the
//! patch grid (`align_corners=False`, no antialias, as `_setup_1d_forward` does).
//!
//! Besides the dense pass it has the sparse passes of noto-sandbox's `noto-ml` (`VisionTower::forward_sparse /
//! extend_sparse / merge_sparse`): only some cells of the grid go through the encoder, each with its own cell's position
//! embedding (the holes are absent, not blacked out); cells opened later attend to every token computed so far through
//! the keys and values the earlier pass kept. With every cell in grid order the sparse pass *is* the dense pass.

use anyhow::{bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::LayerNorm;
use raon_quant::kquant::{encode, Format};
use raon_quant::safetensors::SafeTensors;
use std::collections::HashMap;
use std::sync::Mutex;
use zena_sensor::naflex::PatchGrid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub mlp: usize,
    pub patch: usize,
    /// Side of the learned position table (16 → 256 positions).
    pub side: usize,
    pub eps: f64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            hidden: 1152,
            heads: 16,
            layers: 27,
            mlp: 4304,
            patch: 16,
            side: 16,
            eps: 1e-6,
        }
    }
}

/// How the large matrices are kept on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightFormat {
    F32,
    F16,
    BF16,
    /// ggml Q8_0 blocks (the MLP width is padded 4304 → 4320 with zero rows / columns, which add nothing).
    Q8,
}

impl std::str::FromStr for WeightFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "f32" => WeightFormat::F32,
            "f16" => WeightFormat::F16,
            "bf16" => WeightFormat::BF16,
            "q8" | "q8_0" => WeightFormat::Q8,
            _ => bail!("unknown weight format '{s}' (f32, f16, bf16, q8)"),
        })
    }
}

impl WeightFormat {
    /// The activation dtype.
    pub fn dtype(self) -> DType {
        match self {
            WeightFormat::F16 => DType::F16,
            WeightFormat::BF16 => DType::BF16,
            WeightFormat::F32 | WeightFormat::Q8 => DType::F32,
        }
    }
}

/// `y = x·Wᵀ (+ b)` with W dense or Q8_0.
pub(crate) struct Proj {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl Proj {
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = self.inner.forward(&x.contiguous()?)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Reads weights from the checkpoint (mmap, f32) into the chosen format.
pub(crate) struct Loader<'a> {
    pub st: &'a SafeTensors,
    pub device: Device,
    pub format: WeightFormat,
}

impl Loader<'_> {
    pub(crate) fn dtype(&self) -> DType {
        self.format.dtype()
    }

    pub(crate) fn tensor(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let data = self.st.all_f32(name).with_context(|| name.to_string())?;
        if data.len() != shape.iter().product::<usize>() {
            bail!("{name}: {} values for shape {shape:?}", data.len());
        }
        Ok(Tensor::from_vec(data, shape, &self.device)?.to_dtype(self.dtype())?)
    }

    pub(crate) fn norm(&self, prefix: &str, d: usize, eps: f64) -> Result<LayerNorm> {
        Ok(LayerNorm::new(
            self.tensor(&format!("{prefix}.weight"), &[d])?,
            self.tensor(&format!("{prefix}.bias"), &[d])?,
            eps,
        ))
    }

    /// A projection `[out, in]` (+ bias `[out]`), zero-padded to `[out_pad, in_pad]`; Q8_0 when the format says so
    /// and `in_pad` is a whole number of 32-wide blocks.
    pub(crate) fn proj(&self, weight: &str, bias: Option<&str>, out_pad: usize, in_pad: usize) -> Result<Proj> {
        let info = self.st.info(weight)?;
        let (out, inp) = match info.shape.len() {
            2 => (info.shape[0], info.shape[1]),
            4 => (info.shape[0], info.shape[1..].iter().product()), // the patch conv as a linear map
            _ => bail!("{weight}: unexpected shape {:?}", info.shape),
        };
        let mut w = self.st.all_f32(weight)?;
        if (out_pad, in_pad) != (out, inp) {
            let mut p = vec![0f32; out_pad * in_pad];
            for r in 0..out {
                p[r * in_pad..r * in_pad + inp].copy_from_slice(&w[r * inp..(r + 1) * inp]);
            }
            w = p;
        }
        let inner = if self.format == WeightFormat::Q8 && in_pad.is_multiple_of(32) {
            let bytes = encode(&w, in_pad, Format::Q8_0, None).map_err(|e| anyhow::anyhow!("{weight}: {e}"))?;
            QMatMul::from_qtensor(qtensor_from_ggml(GgmlDType::Q8_0, &bytes, vec![out_pad, in_pad], &self.device)?)?
        } else {
            QMatMul::Tensor(Tensor::from_vec(w, (out_pad, in_pad), &self.device)?.to_dtype(self.dtype())?)
        };
        let bias = match bias {
            Some(b) => {
                let mut v = self.st.all_f32(b)?;
                v.resize(out_pad, 0.0);
                Some(Tensor::from_vec(v, out_pad, &self.device)?.to_dtype(self.dtype())?)
            }
            None => None,
        };
        Ok(Proj { inner, bias })
    }
}

/// The MLP width the format stores (Q8_0 blocks are 32 wide).
pub(crate) fn mlp_width(mlp: usize, format: WeightFormat) -> usize {
    if format == WeightFormat::Q8 {
        mlp.div_ceil(32) * 32
    } else {
        mlp
    }
}

pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor, hd: usize) -> candle_core::Result<Tensor> {
    let dtype = q.dtype();
    let scores = (q.matmul(&k.t()?)? * (hd as f64).powf(-0.5))?;
    let probs = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
    probs.matmul(v)
}

/// A layer's own keys and values `(1, heads, n, head_dim)`, for later passes to attend to.
#[derive(Clone, Debug)]
pub struct LayerKv {
    pub k: Tensor,
    pub v: Tensor,
}

impl LayerKv {
    fn select(&self, rows: &Tensor) -> candle_core::Result<LayerKv> {
        Ok(LayerKv {
            k: self.k.index_select(rows, 2)?,
            v: self.v.index_select(rows, 2)?,
        })
    }
}

struct Block {
    norm1: LayerNorm,
    qkv: Proj,
    proj: Proj,
    norm2: LayerNorm,
    fc1: Proj,
    fc2: Proj,
    heads: usize,
    hd: usize,
}

impl Block {
    fn load(l: &Loader<'_>, i: usize, cfg: &VisionConfig) -> Result<Block> {
        let p = format!("visual.trunk.blocks.{i}");
        let (d, m) = (cfg.hidden, mlp_width(cfg.mlp, l.format));
        Ok(Block {
            norm1: l.norm(&format!("{p}.norm1"), d, cfg.eps)?,
            qkv: l.proj(&format!("{p}.attn.qkv.weight"), Some(&format!("{p}.attn.qkv.bias")), 3 * d, d)?,
            proj: l.proj(&format!("{p}.attn.proj.weight"), Some(&format!("{p}.attn.proj.bias")), d, d)?,
            norm2: l.norm(&format!("{p}.norm2"), d, cfg.eps)?,
            fc1: l.proj(&format!("{p}.mlp.fc1.weight"), Some(&format!("{p}.mlp.fc1.bias")), m, d)?,
            fc2: l.proj(&format!("{p}.mlp.fc2.weight"), Some(&format!("{p}.mlp.fc2.bias")), d, m)?,
            heads: cfg.heads,
            hd: d / cfg.heads,
        })
    }

    /// `x + attn(norm1 x)`, then `+ mlp(norm2 ·)`; the attention also sees `ctx`'s keys and values.
    fn forward_kv(&self, x: &Tensor, ctx: Option<&LayerKv>) -> candle_core::Result<(Tensor, LayerKv)> {
        let (b, n, d) = x.dims3()?;
        let h = self.norm1.forward(x)?;
        let qkv = self
            .qkv
            .forward(&h)?
            .reshape((b, n, 3, self.heads, self.hd))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (qkv.get(0)?.contiguous()?, qkv.get(1)?.contiguous()?, qkv.get(2)?.contiguous()?);
        let (ka, va) = match ctx {
            Some(c) => (Tensor::cat(&[&c.k, &k], 2)?, Tensor::cat(&[&c.v, &v], 2)?),
            None => (k.clone(), v.clone()),
        };
        let o = attend(&q, &ka, &va, self.hd)?.transpose(1, 2)?.reshape((b, n, d))?;
        let x = (x + self.proj.forward(&o)?)?;
        let m = self.fc2.forward(&self.fc1.forward(&self.norm2.forward(&x)?)?.gelu()?)?;
        Ok(((x + m)?, LayerKv { k, v }))
    }
}

/// timm `AttentionPoolLatent`: one latent query attends over the tokens, then `proj`, then `+ mlp(norm ·)`.
struct MapHead {
    latent: Tensor,
    q: Proj,
    kv: Proj,
    proj: Proj,
    norm: LayerNorm,
    fc1: Proj,
    fc2: Proj,
    heads: usize,
    hd: usize,
}

impl MapHead {
    fn load(l: &Loader<'_>, cfg: &VisionConfig) -> Result<MapHead> {
        let p = "visual.trunk.attn_pool";
        let (d, m) = (cfg.hidden, mlp_width(cfg.mlp, l.format));
        Ok(MapHead {
            latent: l.tensor(&format!("{p}.latent"), &[1, 1, d])?,
            q: l.proj(&format!("{p}.q.weight"), Some(&format!("{p}.q.bias")), d, d)?,
            kv: l.proj(&format!("{p}.kv.weight"), Some(&format!("{p}.kv.bias")), 2 * d, d)?,
            proj: l.proj(&format!("{p}.proj.weight"), Some(&format!("{p}.proj.bias")), d, d)?,
            norm: l.norm(&format!("{p}.norm"), d, cfg.eps)?,
            fc1: l.proj(&format!("{p}.mlp.fc1.weight"), Some(&format!("{p}.mlp.fc1.bias")), m, d)?,
            fc2: l.proj(&format!("{p}.mlp.fc2.weight"), Some(&format!("{p}.mlp.fc2.bias")), d, m)?,
            heads: cfg.heads,
            hd: d / cfg.heads,
        })
    }

    fn residual_mlp(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        x + self.fc2.forward(&self.fc1.forward(&self.norm.forward(x)?)?.gelu()?)?
    }

    /// The pooled embedding `(b, d)` of tokens `(b, n, d)`.
    fn pooled(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let lat = self.latent.broadcast_as((b, 1, d))?.contiguous()?;
        let q = self
            .q
            .forward(&lat)?
            .reshape((b, 1, self.heads, self.hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let kv = self
            .kv
            .forward(x)?
            .reshape((b, n, 2, self.heads, self.hd))?
            .permute((2, 0, 3, 1, 4))?;
        let (k, v) = (kv.get(0)?.contiguous()?, kv.get(1)?.contiguous()?);
        let o = attend(&q, &k, &v, self.hd)?.transpose(1, 2)?.reshape((b, 1, d))?;
        self.residual_mlp(&self.proj.forward(&o)?)?.reshape((b, d))
    }

    /// Per-token features in the pooled space (the value path, as if each token attended only to itself; MaskCLIP).
    fn dense(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let d = x.dim(D::Minus1)?;
        let v = self.kv.forward(x)?.narrow(D::Minus1, d, d)?;
        self.residual_mlp(&self.proj.forward(&v)?)
    }
}

/// An encoder pass over some cells of a patch grid: the cells, their tokens after the final norm `(1, n, d)`, and
/// every layer's keys and values for a later pass to attend to.
#[derive(Clone, Debug)]
pub struct SparsePass {
    pub cells: Vec<u32>,
    pub tokens: Tensor,
    kv: Vec<LayerKv>,
}

pub struct VisionTower {
    patch: Proj,
    pos: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    head: MapHead,
    pub cfg: VisionConfig,
    pub format: WeightFormat,
    device: Device,
    pos_cache: Mutex<HashMap<(usize, usize), Tensor>>,
}

/// PyTorch's bilinear weights for `src → dst` with `align_corners=False` and no antialias: `(dst, src)`, two taps a row.
pub fn bilinear_weights(src: usize, dst: usize) -> Vec<f32> {
    let scale = src as f64 / dst as f64;
    let mut w = vec![0f32; dst * src];
    for i in 0..dst {
        let x = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (x.floor() as usize).min(src - 1);
        let i1 = (i0 + 1).min(src - 1);
        let l1 = (x - i0 as f64) as f32;
        w[i * src + i0] += 1.0 - l1;
        w[i * src + i1] += l1;
    }
    w
}

impl VisionTower {
    pub fn load(st: &SafeTensors, device: &Device, format: WeightFormat) -> Result<VisionTower> {
        let cfg = VisionConfig::default();
        let l = Loader {
            st,
            device: device.clone(),
            format,
        };
        let d = cfg.hidden;
        let pd = 3 * cfg.patch * cfg.patch;
        let blocks = (0..cfg.layers).map(|i| Block::load(&l, i, &cfg)).collect::<Result<Vec<_>>>()?;
        Ok(VisionTower {
            patch: l.proj(
                "visual.trunk.patch_embed.proj.weight",
                Some("visual.trunk.patch_embed.proj.bias"),
                d,
                pd,
            )?,
            pos: Tensor::from_vec(st.all_f32("visual.trunk.pos_embed")?, (cfg.side * cfg.side, d), device)?,
            blocks,
            norm: l.norm("visual.trunk.norm", d, cfg.eps)?,
            head: MapHead::load(&l, &cfg)?,
            cfg,
            format,
            device: device.clone(),
            pos_cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The position table resized to `gh × gw` (f32 resample, then the activation dtype), cached per grid.
    fn positions(&self, gh: usize, gw: usize) -> Result<Tensor> {
        if let Some(t) = self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).get(&(gh, gw)) {
            return Ok(t.clone());
        }
        let (s, d) = (self.cfg.side, self.cfg.hidden);
        let t = if (gh, gw) == (s, s) {
            self.pos.clone()
        } else {
            let wy = Tensor::from_vec(bilinear_weights(s, gh), (gh, s), &self.device)?;
            let wx = Tensor::from_vec(bilinear_weights(s, gw), (gw, s), &self.device)?;
            // rows: (gh, s) · (s, s·d) → (gh, s, d); columns: (gw, s) · (s, d) per row → (gh, gw, d)
            let by_rows = wy.matmul(&self.pos.reshape((s, s * d))?)?.reshape((gh, s, d))?;
            wx.broadcast_left(gh)?.contiguous()?.matmul(&by_rows)?.reshape((gh * gw, d))?
        }
        .to_dtype(self.format.dtype())?;
        self.pos_cache.lock().unwrap_or_else(|p| p.into_inner()).insert((gh, gw), t.clone());
        Ok(t)
    }

    /// The patches of `cells` through the patch projection, each with its own cell's position: `(1, n, d)`.
    fn cell_tokens(&self, grid: &PatchGrid, cells: &[u32]) -> Result<Tensor> {
        let n = grid.cells();
        let mut flat = Vec::with_capacity(cells.len() * PatchGrid::DIM);
        for &c in cells {
            if c as usize >= n {
                bail!("cell {c} is outside the {}x{} grid", grid.grid_h, grid.grid_w);
            }
            flat.extend_from_slice(grid.patch(c as usize));
        }
        let patches = Tensor::from_vec(flat, (1, cells.len(), PatchGrid::DIM), &self.device)?.to_dtype(self.format.dtype())?;
        let rows = Tensor::from_slice(cells, cells.len(), &self.device)?;
        let pos = self.positions(grid.grid_h, grid.grid_w)?.index_select(&rows, 0)?.unsqueeze(0)?;
        Ok(self.patch.forward(&patches)?.broadcast_add(&pos)?)
    }

    /// The encoder over `cells` only: they attend to each other and nothing else.
    pub fn forward_sparse(&self, grid: &PatchGrid, cells: &[u32]) -> Result<SparsePass> {
        let mut x = self.cell_tokens(grid, cells)?;
        let mut kv = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let (y, layer_kv) = b.forward_kv(&x, None)?;
            x = y;
            kv.push(layer_kv);
        }
        Ok(SparsePass {
            cells: cells.to_vec(),
            tokens: self.norm.forward(&x)?,
            kv,
        })
    }

    /// Every cell in grid order: the dense pass.
    pub fn forward_full(&self, grid: &PatchGrid) -> Result<SparsePass> {
        let cells: Vec<u32> = (0..grid.cells() as u32).collect();
        self.forward_sparse(grid, &cells)
    }

    /// Opens `cells` on top of `first`: at every layer they attend to each other and to `first`'s tokens (keys and
    /// values as `first` computed them). A cell of `first` listed again is computed anew and leaves the context;
    /// `first`'s other tokens keep their features (the attention only runs one way).
    pub fn extend_sparse(&self, grid: &PatchGrid, first: &SparsePass, cells: &[u32]) -> Result<SparsePass> {
        let again: std::collections::HashSet<u32> = cells.iter().copied().collect();
        let keep: Vec<u32> = first
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !again.contains(c))
            .map(|(i, _)| i as u32)
            .collect();
        let context: Option<Vec<LayerKv>> = if keep.len() == first.cells.len() {
            None
        } else if keep.is_empty() {
            Some(Vec::new())
        } else {
            let rows = Tensor::from_slice(&keep, keep.len(), &self.device)?;
            Some(first.kv.iter().map(|kv| kv.select(&rows)).collect::<candle_core::Result<_>>()?)
        };
        let mut x = self.cell_tokens(grid, cells)?;
        let mut kv = Vec::with_capacity(self.blocks.len());
        for (i, b) in self.blocks.iter().enumerate() {
            let ctx = match &context {
                None => Some(&first.kv[i]),
                Some(v) => v.get(i),
            };
            let (y, layer_kv) = b.forward_kv(&x, ctx)?;
            x = y;
            kv.push(layer_kv);
        }
        Ok(SparsePass {
            cells: cells.to_vec(),
            tokens: self.norm.forward(&x)?,
            kv,
        })
    }

    /// `first` with `second` (an `extend_sparse` of it) folded in, so a further opening attends to every cell so far.
    pub fn merge_sparse(&self, first: &SparsePass, second: &SparsePass) -> Result<SparsePass> {
        let again: std::collections::HashSet<u32> = second.cells.iter().copied().collect();
        let keep: Vec<u32> = first
            .cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !again.contains(c))
            .map(|(i, _)| i as u32)
            .collect();
        let mut cells: Vec<u32> = keep.iter().map(|&i| first.cells[i as usize]).collect();
        cells.extend_from_slice(&second.cells);
        if keep.is_empty() {
            return Ok(SparsePass {
                cells,
                tokens: second.tokens.clone(),
                kv: second.kv.clone(),
            });
        }
        let rows = Tensor::from_slice(&keep, keep.len(), &self.device)?;
        let tokens = Tensor::cat(&[&first.tokens.index_select(&rows, 1)?, &second.tokens], 1)?;
        let kv = first
            .kv
            .iter()
            .zip(&second.kv)
            .map(|(a, b)| {
                let a = a.select(&rows)?;
                Ok(LayerKv {
                    k: Tensor::cat(&[&a.k, &b.k], 2)?,
                    v: Tensor::cat(&[&a.v, &b.v], 2)?,
                })
            })
            .collect::<candle_core::Result<_>>()?;
        Ok(SparsePass { cells, tokens, kv })
    }

    /// The MAP head's pooled embedding of a pass's tokens (unit length).
    pub fn pooled(&self, pass: &SparsePass) -> Result<Vec<f32>> {
        let v: Vec<f32> = self.head.pooled(&pass.tokens)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(unit(v))
    }

    /// Per-cell unit features of a pass's tokens, in `pass.cells` order.
    pub fn dense(&self, pass: &SparsePass) -> Result<Vec<Vec<f32>>> {
        let m: Vec<Vec<f32>> = self.head.dense(&pass.tokens)?.to_dtype(DType::F32)?.squeeze(0)?.to_vec2()?;
        Ok(m.into_iter().map(unit).collect())
    }
}

pub fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_weights_match_pytorch_align_corners_false() {
        // Upsampling 2 → 4: [1, .75/.25, .25/.75, 1] (edges clamp).
        let w = bilinear_weights(2, 4);
        assert_eq!(w, vec![1.0, 0.0, 0.75, 0.25, 0.25, 0.75, 0.0, 1.0]);
        // Same size is the identity; every row sums to one.
        let w = bilinear_weights(16, 16);
        assert!((0..16).all(|i| w[i * 16 + i] == 1.0));
        let w = bilinear_weights(16, 9);
        assert!(w.chunks(16).all(|r| (r.iter().sum::<f32>() - 1.0).abs() < 1e-6));
    }
}
