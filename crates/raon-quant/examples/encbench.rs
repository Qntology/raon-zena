//! Encoder throughput per format (Melem/s) on a 6144x1536 matrix.
use raon_quant::kquant::{encode, Format};
use std::time::Instant;
fn main() {
    let (r, c) = (6144usize, 1536usize);
    let mut s = 1u64;
    let x: Vec<f32> = (0..r * c)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.04
        })
        .collect();
    for f in [Format::Q8_0, Format::Q6K, Format::Q5K, Format::Q4K] {
        let t = Instant::now();
        let b = encode(&x, c, f, None).unwrap();
        let dt = t.elapsed().as_secs_f64();
        println!(
            "{f:?}: {:.1} Melem/s ({} bytes, {:.2}s, threads {})",
            (r * c) as f64 / dt / 1e6,
            b.len(),
            dt,
            rayon::current_num_threads()
        );
    }
}
