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
