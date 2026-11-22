//! Output head (tied to `model.embed_tokens.weight`).
//!
//! Modes:
//! * [`HeadMode::Full`] (default, exact): every vocabulary row is scored.
//!   - GPU: uses Pack A/C `candle_core::lm_head::fused_linear_argmax_cs` (Mode A dense: 0 bytes
//!     logits materialization) or `logits_argmax_cs` (Mode B quantized: 8-byte D2H instead of 1 MiB).
//!   - CPU: zero-allocation parallel tiled argmax over mmap'd blocks (no 1.04 MiB logits Vec).
//! * [`HeadMode::Kept`] (approximate): only the resident `lm_head.kept` rows (`kept_ids`).
//! * [`HeadMode::Guarded`] (exact): kept top-1 + cold-row bound refinement (`ColdGuard`).

use crate::device::{accel_op_ok, AccelOp};
use crate::weights::{Linear, Source, Weights};
use anyhow::{bail, Context, Result};
use candle_core::quantized::QMatMul;
use candle_core::Tensor;
use raon_quant::gguf::GgmlType;
use raon_quant::kernels::{dot_row, matvec};
use raon_quant::lmhead::{exact_argmax, ArgmaxStats, ColdGuard};
use rayon::prelude::*;

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

#[inline]
fn better_pair(a: (usize, f32), b: (usize, f32)) -> (usize, f32) {
    if a.1 > b.1 || (a.1 == b.1 && a.0 < b.0) {
        a
    } else {
        b
    }
}

/// Zero-allocation parallel tiled argmax over packed host rows `[rows, cols]` with deterministic
/// smallest-index tie-breaking.
fn cpu_argmax_table(ty: GgmlType, bytes: &[u8], rows: usize, cols: usize, hv: &[f32]) -> (usize, f32) {
    if rows == 0 {
        return (0, f32::NEG_INFINITY);
    }
    #[cfg(target_endian = "little")]
    if ty == GgmlType::F32 {
        // SAFETY: `align_to` only yields `mid` when properly 4-byte aligned.
        let (head, mid, tail) = unsafe { bytes.align_to::<f32>() };
        if head.is_empty() && tail.is_empty() && mid.len() == rows * cols {
            if let Some(best) = raon_cpu_kernels::dot_argmax_rows_f32(hv, mid, rows, cols) {
                return best;
            }
        }
    }
    let rb = ty.row_bytes(cols);
    let chunk = (rows / rayon::current_num_threads().max(1)).clamp(64, 2048);
    let n_chunks = rows.div_ceil(chunk);
    (0..n_chunks)
        .into_par_iter()
        .map(|c| {
            let r0 = c * chunk;
            let r1 = (r0 + chunk).min(rows);
            let mut best = (usize::MAX, f32::NEG_INFINITY);
            for r in r0..r1 {
                let l = dot_row(ty, &bytes[r * rb..(r + 1) * rb], hv);
                best = better_pair((r, l), best);
            }
            best
        })
        .reduce(|| (usize::MAX, f32::NEG_INFINITY), better_pair)
}

/// On-device argmax over `lin` for `h_row [M, C]`:
/// - Mode A (`QMatMul::Tensor`): `fused_linear_argmax_cs` (0-byte logits materialization).
/// - Quantized Mode A (CUDA / ROCm + `QMatMul::QTensor` + `Q8_0|Q4_0|Q4K|Q5K|Q6K`):
///   `fused_quant_argmax_cs` (0-byte `[M, V]` logits materialization, automatically chunking `M > 8`).
/// - Mode B (Vulkan / other `QTensor` / `TensorF16`): `lin.forward` + `logits_argmax_cs` (8 bytes D2H).
fn gpu_argmax_linear_batch(lin: &Linear, h_row: &Tensor, need_val: bool) -> Result<Vec<(usize, f32)>> {
    use candle_core::quantized::GgmlDType;
    let dev = h_row.device();
    let m = h_row.dim(0)?;
    if accel_op_ok(dev, AccelOp::LmHeadArgmax) {
        let out = match lin.inner() {
            QMatMul::Tensor(weight) => {
                candle_core::lm_head::fused_linear_argmax_cs(h_row, weight, None, None, None)
            }
            QMatMul::QTensor(qweight)
                if (crate::device::is_rocm(dev) || dev.is_cuda())
                    && matches!(
                        qweight.dtype(),
                        GgmlDType::Q8_0 | GgmlDType::Q4_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
                    )
                    && h_row.dim(1).unwrap_or(0).is_multiple_of(qweight.dtype().block_size()) =>
            {
                candle_core::lm_head::fused_quant_argmax_cs(h_row, qweight, None, None, None)
                    .or_else(|_| {
                        let logits = lin.forward(h_row)?;
                        candle_core::lm_head::logits_argmax_cs(&logits, None, None, None)
                    })
            }
            _ => {
                let logits = lin.forward(h_row)?;
                candle_core::lm_head::logits_argmax_cs(&logits, None, None, None)
            }
        };
        if let Ok(out) = out {
            let (ids, _) = out.download_ids_and_mask()?;
            let vals = if need_val {
                match out.max_vals {
                    Some(mv) => mv.to_device(&candle_core::Device::Cpu)?.to_vec1::<f32>()?,
                    None => vec![0.0; ids.len()],
                }
            } else {
                vec![0.0; ids.len()]
            };
            return Ok(ids
                .into_iter()
                .zip(vals)
                .map(|(id, v)| (id as usize, v))
                .collect());
        }
    }
    let logits = lin.forward(h_row)?.to_device(&candle_core::Device::Cpu)?;
    let vocab = logits.dim(1)?;
    let flat = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(m);
    for row in flat.chunks_exact(vocab) {
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        for (i, &x) in row.iter().enumerate() {
            best = better_pair((i, x), best);
        }
        out.push(best);
    }
    Ok(out)
}

fn gpu_argmax_linear(lin: &Linear, h_row: &Tensor, need_val: bool) -> Result<(usize, f32)> {
    Ok(gpu_argmax_linear_batch(lin, h_row, need_val)?[0])
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

    /// Greedy token for one hidden state `h [C]` (or `[1, C]`) → (token id, logit, guard stats).
    pub fn argmax(&self, w: &Weights, h: &Tensor) -> Result<(u32, f32, ArgmaxStats)> {
        self.argmax_inner(w, h, true)
    }

    /// Greedy token ID for one hidden state `h [C]` (or `[1, C]`), skipping the extra `max_vals`
    /// D2H sync in `HeadMode::Full`/`HeadMode::Kept` when only the winning token ID is needed.
    pub fn argmax_id(&self, w: &Weights, h: &Tensor) -> Result<(u32, ArgmaxStats)> {
        let (id, _, stats) = self.argmax_inner(w, h, self.mode == HeadMode::Guarded)?;
        Ok((id, stats))
    }

    fn argmax_inner(&self, w: &Weights, h: &Tensor, need_val: bool) -> Result<(u32, f32, ArgmaxStats)> {
        let c = h.elem_count();
        let h_row = h.reshape((1, c))?;
        let (best_idx, best_logit, hv_opt) = if let Some(lin) = &self.dev {
            let (i, l) = gpu_argmax_linear(lin, &h_row, need_val)?;
            (i, l, None)
        } else {
            let hv = h.flatten_all()?.to_vec1::<f32>()?;
            let name = if self.mode == HeadMode::Full { EMBED } else { "lm_head.kept" };
            let (ty, bytes, v, cols) = table(w, name)?;
            let (i, l) = cpu_argmax_table(ty, bytes, v, cols, &hv);
            (i, l, Some(hv))
        };
        match self.mode {
            HeadMode::Full => Ok((best_idx as u32, best_logit, ArgmaxStats::default())),
            HeadMode::Kept => Ok((self.kept_ids.as_ref().unwrap()[best_idx], best_logit, ArgmaxStats::default())),
            HeadMode::Guarded => {
                let hv = match hv_opt {
                    Some(v) => v,
                    None => h.flatten_all()?.to_vec1::<f32>()?,
                };
                let g = w.gguf().context("guarded head needs a GGUF")?;
                let ty = g.tensor(EMBED)?.ty;
                let best_kept_id = self.kept_ids.as_ref().unwrap()[best_idx];
                let cold = |id: u32| dot_row(ty, g.row_bytes(EMBED, id as usize).unwrap(), &hv);
                Ok(exact_argmax(
                    &[best_logit],
                    &[best_kept_id],
                    self.guard.as_ref().unwrap(),
                    &hv,
                    cold,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{drain_gpu_pool, select_device, Backend};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::Device;

    fn det(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn cpu_zero_alloc_argmax_matches_matvec_and_tie_breaks_smallest_id() {
        let (rows, cols) = (512usize, 64usize);
        let mut w = det(rows * cols, 77);
        // Make row 100 and row 200 identical winners to verify smallest-ID tie-break.
        let h = det(cols, 88);
        for j in 0..cols {
            w[100 * cols + j] = h[j] * 4.0;
            w[200 * cols + j] = h[j] * 4.0;
        }
        let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (idx, val) = cpu_argmax_table(GgmlType::F32, &bytes, rows, cols, &h);
        assert_eq!(idx, 100, "must tie-break to smallest row index");
        let mut ref_y = vec![0.0f32; rows];
        matvec(GgmlType::F32, &bytes, rows, cols, &h, &mut ref_y);
        assert!((val - ref_y[100]).abs() < 1e-4, "{val} vs {}", ref_y[100]);
    }

    #[test]
    fn lm_head_mode_a_and_mode_b_match_reference_on_active_device() -> Result<()> {
        let dev = select_device(Backend::Auto, 0)?;
        let (rows, cols) = (256usize, 256usize);
        let w = Tensor::from_vec(det(rows * cols, 91), (rows, cols), &Device::Cpu)?;
        let h = Tensor::from_vec(det(cols, 92), (1, cols), &Device::Cpu)?;
        let ref_logits = h.matmul(&w.t()?)?.squeeze(0)?.to_vec1::<f32>()?;
        let (want_id, want_val) = ref_logits
            .iter()
            .enumerate()
            .fold((usize::MAX, f32::NEG_INFINITY), |a, (i, &x)| better_pair((i, x), a));

        // Mode A: dense Tensor -> fused_linear_argmax_cs
        let lin_dense = Linear::from_qmatmul(QMatMul::Tensor(w.to_device(&dev)?), true, "test.dense", rows, cols);
        let (got_id, got_val) = gpu_argmax_linear(&lin_dense, &h.to_device(&dev)?, true)?;
        assert_eq!(got_id, want_id);
        assert!((got_val - want_val).abs() < 1e-3, "{got_val} vs {want_val}");

        // Quantized Mode A (CUDA/ROCm fused_quant_argmax_cs, M=12 > 8 chunking) / Mode B (Vulkan GpuQuant + logits_argmax_cs)
        let w_dev = w.to_device(&dev)?;
        let h_batch = Tensor::from_vec(det(12 * cols, 93), (12, cols), &Device::Cpu)?;
        let h_batch_dev = h_batch.to_device(&dev)?;
        for dt in [
            GgmlDType::Q8_0,
            GgmlDType::Q4_0,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let exact = crate::device::exact_matmul_native(&dev);
            let qt_cpu = QTensor::quantize(&w, dt)?;
            let q_ref_flat = if exact || dev.is_vulkan() {
                qt_cpu.forward_exact(&h_batch)?
            } else {
                candle_core::Module::forward(&QMatMul::from_qtensor(qt_cpu)?, &h_batch)?
            }
            .flatten_all()?
            .to_vec1::<f32>()?;
            let qt = QTensor::quantize(&w_dev, dt)?;
            let lin_q = Linear::from_qmatmul(
                QMatMul::from_qtensor(qt)?,
                exact,
                "test.quant",
                rows,
                cols,
            );
            let got = gpu_argmax_linear_batch(&lin_q, &h_batch_dev, true)?;
            assert_eq!(got.len(), 12);
            for (r, &(q_id, q_val)) in got.iter().enumerate() {
                let (q_want_id, q_want_val) = q_ref_flat[r * rows..(r + 1) * rows]
                    .iter()
                    .enumerate()
                    .fold((usize::MAX, f32::NEG_INFINITY), |a, (i, &x)| better_pair((i, x), a));
                assert_eq!(q_id, q_want_id, "{dt:?} row={r} argmax mismatch");
                assert!((q_val - q_want_val).abs() < 0.1, "{dt:?} row={r}: {q_val} vs {q_want_val}");
            }
        }

        if !dev.is_cpu() {
            drain_gpu_pool(&dev);
        }
        Ok(())
    }
}
