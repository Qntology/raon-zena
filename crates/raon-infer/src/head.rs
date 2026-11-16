//! Output head (tied to `model.embed_tokens.weight`).
//!
//! Modes:
//! * [`HeadMode::Full`] (default, exact): every vocabulary row is scored. On the CPU the fused
//!   kernels read the mmap'd Q8_0 blocks directly; on a GPU the table is uploaded as a quantized
//!   `QMatMul` (+404 MiB VRAM for the pyramid GGUF).
//! * [`HeadMode::Kept`] (approximate): only the resident `lm_head.kept` rows (`kept_ids`) — the
//!   PRD's sliced head; tokens outside the kept set can never be produced.
//! * [`HeadMode::Guarded`] (exact): kept rows + the cold-row cluster bound of `raon-quant`. Measured
//!   on real hidden states the bound prunes almost nothing (≈ 99.9 % of cold rows re-scored), so it
//!   is slower than `Full`; kept for experiments.

use crate::weights::{Linear, Source, Weights};
use anyhow::{bail, Context, Result};
use candle_core::Tensor;
use raon_quant::gguf::GgmlType;
use raon_quant::kernels::{dot_row, matvec};
use raon_quant::lmhead::{exact_argmax, ArgmaxStats, ColdGuard};

const EMBED: &str = "model.embed_tokens.weight";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadMode {
    Full,
    Kept,
    Guarded,
}

impl std::str::FromStr for HeadMode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "full" | "exact" => HeadMode::Full,
            "kept" | "sliced" => HeadMode::Kept,
            "guarded" | "guard" => HeadMode::Guarded,
            _ => bail!("unknown head mode {s} (full, kept, guarded)"),
        })
    }
}

pub struct Head {
    pub mode: HeadMode,
    /// GPU copy of the scored rows (full table or kept rows); `None` on the CPU.
    dev: Option<Linear>,
    kept_ids: Option<Vec<u32>>,
    guard: Option<ColdGuard>,
}

fn table<'a>(w: &'a Weights, name: &str) -> Result<(GgmlType, &'a [u8], usize, usize)> {
    Ok(match &w.src {
        Source::Gguf(g) => {
            let t = g.tensor(name)?;
            (t.ty, g.tensor_bytes(name)?, t.shape[0], t.shape[1])
        }
        Source::SafeTensors(s) => {
            let i = s.info(name)?;
            let ty = match i.dtype {
                raon_quant::safetensors::Dtype::BF16 => GgmlType::BF16,
                raon_quant::safetensors::Dtype::F16 => GgmlType::F16,
                raon_quant::safetensors::Dtype::F32 => GgmlType::F32,
            };
            (ty, s.bytes(name)?, i.shape[0], i.shape[1])
        }
    })
}

impl Head {
    pub fn load(w: &Weights, mode: HeadMode) -> Result<Self> {
        let sliced = w.kept_ids.is_some() && w.has("lm_head.kept");
        let mode = if !sliced && mode != HeadMode::Full {
            eprintln!("[raon-infer] no sliced head in these weights; using the full head");
            HeadMode::Full
        } else {
            mode
        };
        let gpu = !w.device.is_cpu();
        let dev = match mode {
            HeadMode::Full if gpu => Some(w.linear(EMBED)?),
            HeadMode::Kept | HeadMode::Guarded if gpu => Some(w.linear("lm_head.kept")?),
            _ => None,
        };
        let guard = if mode == HeadMode::Guarded {
            let g = w.gguf().context("guarded head needs a GGUF")?;
            Some(ColdGuard {
                dim: g.tensor("lm_head.kept")?.shape[1],
                centers: g.f32_values("lm_head.cold.centers")?,
                radius: g.f32_values("lm_head.cold.radius")?,
                members: g.i32_values("lm_head.cold.members")?.into_iter().map(|x| x as u32).collect(),
                offsets: g.i32_values("lm_head.cold.offsets")?.into_iter().map(|x| x as u32).collect(),
            })
        } else {
            None
        };
        Ok(Self {
            mode,
            dev,
            kept_ids: w.kept_ids.clone(),
            guard,
        })
    }

    /// Logits of every vocabulary row on the host (validation).
    pub fn full_logits(&self, w: &Weights, h: &[f32]) -> Result<Vec<f32>> {
        let (ty, bytes, v, c) = table(w, EMBED)?;
        let mut y = vec![0f32; v];
        matvec(ty, bytes, v, c, h, &mut y);
        Ok(y)
    }

    fn scored(&self, w: &Weights, h: &Tensor, hv: &[f32]) -> Result<Vec<f32>> {
        if let Some(lin) = &self.dev {
            return Ok(lin.forward(&h.reshape((1, hv.len()))?)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let name = if self.mode == HeadMode::Full { EMBED } else { "lm_head.kept" };
        let (ty, bytes, v, c) = table(w, name)?;
        let mut y = vec![0f32; v];
        matvec(ty, bytes, v, c, hv, &mut y);
        Ok(y)
    }

    /// Greedy token for one hidden state `h [C]` → (token id, logit, guard stats).
    pub fn argmax(&self, w: &Weights, h: &Tensor) -> Result<(u32, f32, ArgmaxStats)> {
        let hv = h.flatten_all()?.to_vec1::<f32>()?;
        let y = self.scored(w, h, &hv)?;
        let (i, l) = y
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| if x > a.1 { (i, x) } else { a });
        match self.mode {
            HeadMode::Full => Ok((i as u32, l, ArgmaxStats::default())),
            HeadMode::Kept => Ok((self.kept_ids.as_ref().unwrap()[i], l, ArgmaxStats::default())),
            HeadMode::Guarded => {
                let g = w.gguf().context("guarded head needs a GGUF")?;
                let ty = g.tensor(EMBED)?.ty;
                let cold = |id: u32| dot_row(ty, g.row_bytes(EMBED, id as usize).unwrap(), &hv);
                Ok(exact_argmax(
                    &y,
                    self.kept_ids.as_ref().unwrap(),
                    self.guard.as_ref().unwrap(),
                    &hv,
                    cold,
                ))
            }
        }
    }
}
