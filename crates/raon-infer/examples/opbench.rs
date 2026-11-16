//! Micro-benchmark of single-token linears (CPU k-quant QMatMul vs raon-quant fused kernels).
use candle_core::{Device, Tensor};
use raon_infer::weights::Weights;
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let w = Weights::open_gguf(a[1].as_ref(), Device::Cpu, false)?;
    let g = w.gguf().unwrap();
    for name in [
        "model.layers.5.mlp.gate_proj.weight",
        "model.layers.5.mlp.down_proj.weight",
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.5.self_attn.f_b_proj.weight",
        "model.layers.5.mlp_raon_mhc.phi_pre",
    ] {
        let shape = w.shape(name)?;
        let (o, i) = (shape[0], shape[1]);
        let x = Tensor::randn(0f32, 1.0, (1, i), &Device::Cpu)?;
        let lin = w.linear(name)?;
        lin.forward(&x)?;
        let t = Instant::now();
        for _ in 0..20 {
            lin.forward(&x)?;
        }
        let candle_ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
        let t = g.tensor(name)?;
        let xv = x.flatten_all()?.to_vec1::<f32>()?;
        let mut y = vec![0f32; o];
        let tt = Instant::now();
        for _ in 0..20 {
            raon_quant::kernels::matvec(t.ty, g.tensor_bytes(name)?, o, i, &xv, &mut y);
        }
        println!(
            "{name:<45} {:?} {:>5} candle {candle_ms:.2} ms | raon fused {:.2} ms",
            t.ty,
            format!("{o}x{i}"),
            tt.elapsed().as_secs_f64() * 1e3 / 20.0
        );
    }
    Ok(())
}
