//! Device selection, following logis-center's `utils/device_utils.rs::new_gpu_device`: the first
//! backend compiled in that opens wins (CUDA → ROCm → Vulkan → Metal), otherwise the CPU.

use anyhow::{bail, Result};
use candle_core::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Cuda,
    Rocm,
    Vulkan,
    Metal,
}

impl std::str::FromStr for Backend {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" | "gpu" => Backend::Auto,
            "cpu" => Backend::Cpu,
            "cuda" => Backend::Cuda,
            "rocm" | "hip" => Backend::Rocm,
            "vulkan" | "vk" => Backend::Vulkan,
            "metal" => Backend::Metal,
            _ => bail!("unknown device {s} (auto, cpu, cuda, rocm, vulkan, metal)"),
        })
    }
}

/// Backends this binary was built with.
pub fn compiled_backends() -> Vec<&'static str> {
    let mut v = vec!["cpu"];
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "rocm") {
        v.push("rocm");
    }
    if cfg!(feature = "vulkan") {
        v.push("vulkan");
    }
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    v
}

fn try_gpu(b: Backend, ordinal: usize) -> Result<Option<Device>> {
    let _ = ordinal;
    Ok(match b {
        #[cfg(feature = "cuda")]
        Backend::Cuda => Some(Device::new_cuda(ordinal)?),
        #[cfg(feature = "rocm")]
        Backend::Rocm => Some(Device::new_rocm(ordinal)?),
        #[cfg(feature = "vulkan")]
        Backend::Vulkan => Some(Device::new_vulkan(ordinal)?),
        #[cfg(feature = "metal")]
        Backend::Metal => Some(Device::new_metal(ordinal)?),
        Backend::Cpu => Some(Device::Cpu),
        _ => None,
    })
}

pub fn select_device(b: Backend, ordinal: usize) -> Result<Device> {
    if b != Backend::Auto {
        return match try_gpu(b, ordinal)? {
            Some(d) => Ok(d),
            None => bail!("{b:?} support is not compiled in (built with: {})", compiled_backends().join(", ")),
        };
    }
    for cand in [Backend::Cuda, Backend::Rocm, Backend::Vulkan, Backend::Metal] {
        if let Ok(Some(d)) = try_gpu(cand, ordinal) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}

pub fn describe(d: &Device) -> String {
    match d {
        Device::Cpu => format!("cpu ({} threads)", rayon::current_num_threads()),
        Device::Cuda(_) => "cuda".into(),
        Device::Metal(_) => "metal".into(),
        #[allow(unreachable_patterns)]
        _ => format!("{:?}", d.location()),
    }
}

/// Whether `QMatMul::forward_exact` is cheap on this device (native exact kernels). On the CPU,
/// Vulkan and Metal it dequantizes the whole weight on every call, so the k-quant `forward` path
/// (activations quantized to Q8_K) is used there instead — the same policy logis-center applies.
pub fn exact_matmul_native(d: &Device) -> bool {
    d.is_cuda() || is_rocm(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{Module, Tensor};

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
    }

    /// With a GPU backend compiled in (`--features cuda` …) and a device present, dense and quantized kernels run
    /// on it and agree with the CPU; without one the test only checks that the CPU is chosen.
    #[test]
    fn the_gpu_backend_runs_and_matches_the_cpu() -> Result<()> {
        let dev = select_device(Backend::Auto, 0)?;
        eprintln!("device: {} (compiled: {})", describe(&dev), compiled_backends().join(", "));
        if dev.is_cpu() {
            return Ok(());
        }
        let a = Tensor::randn(0f32, 1.0, (8, 256), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 0.1, (512, 256), &Device::Cpu)?;
        let cpu = candle_nn::ops::softmax_last_dim(&a.matmul(&w.t()?)?)?;
        let gpu = candle_nn::ops::softmax_last_dim(&a.to_device(&dev)?.matmul(&w.to_device(&dev)?.t()?)?)?;
        let d = max_abs_diff(&cpu, &gpu.to_device(&Device::Cpu)?)?;
        assert!(d < 1e-4, "dense matmul + softmax differ by {d}");
        // Q8_0 weights through QMatMul (the block format the pyramid's T2 tier uses).
        let q_cpu = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q8_0)?)?;
        let q_gpu = QMatMul::from_qtensor(QTensor::quantize(&w.to_device(&dev)?, GgmlDType::Q8_0)?)?;
        let y_cpu = q_cpu.forward(&a)?;
        let y_gpu = q_gpu.forward(&a.to_device(&dev)?)?.to_device(&Device::Cpu)?;
        let scale = y_cpu.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let d = max_abs_diff(&y_cpu, &y_gpu)?;
        assert!(d <= 0.05 * scale.max(1e-3), "Q8_0 matmul differs by {d} (scale {scale})");
        Ok(())
    }
}

#[allow(unused_variables)]
pub fn is_rocm(d: &Device) -> bool {
    #[cfg(feature = "rocm")]
    {
        d.is_rocm()
    }
    #[cfg(not(feature = "rocm"))]
    {
        false
    }
}
