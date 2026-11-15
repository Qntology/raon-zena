//! Phase timing: forward vs head, per decode token.
use raon_infer::generate::Generator;
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dev = raon_infer::device::select_device(
        a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(raon_infer::device::Backend::Cpu),
        0,
    )?;
    let g = Generator::load_gguf(a[1].as_ref(), a[2].as_ref(), dev, None, raon_infer::head::HeadMode::Guarded)?;
    let ids = g.encode("<|im_start|>user\n안녕<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n")?;
    let mut st = g.model.new_state();
    let t = Instant::now();
    let h = g.model.forward(&ids, &mut st)?;
    println!("prefill {} tok: {:.1} ms", ids.len(), t.elapsed().as_secs_f64() * 1e3);
    let last = h.narrow(0, ids.len() - 1, 1)?;
    raon_infer::model::profile::ON.store(true, std::sync::atomic::Ordering::Relaxed);
    raon_infer::model::profile::take();
    for i in 0..5 {
        let t = Instant::now();
        let h = g.model.forward(&[1000 + i], &mut st)?;
        let f = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let (_, _, s) = g.head.argmax(&g.model.w, &h)?;
        let hd = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let hv = h.flatten_all()?.to_vec1::<f32>()?;
        let _ = g.head.full_logits(&g.model.w, &hv)?;
        println!(
            "  phases: {:?}",
            raon_infer::model::profile::take()
                .iter()
                .map(|(n, v)| format!("{n}={v:.1}"))
                .collect::<Vec<_>>()
        );
        println!(
            "decode: forward {f:.1} ms, sliced+guard head {hd:.1} ms (refined {} rows), full-vocab head {:.1} ms",
            s.rows_refined,
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let _ = last;
    Ok(())
}
