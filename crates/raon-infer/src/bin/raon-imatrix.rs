//! `raon-imatrix`: collect an importance matrix for `raon-quant --imatrix` by running a
//! calibration corpus through the ORIGINAL Raon-1-nano weights.
//!
//! ```text
//! raon-imatrix --safetensors models/KRAFTON/Raon-1-nano --calib scripts/calibration/raon_calib.txt \
//!              --out out/raon-1-nano.imatrix.dat [--chunk 512]
//! ```
//! Paragraphs (blank-line separated) are packed into chunks of at most `--chunk` tokens.

use anyhow::{bail, Result};
use raon_infer::generate::Generator;
use raon_infer::weights::imatrix;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

fn main() -> Result<()> {
    let (mut st, mut calib, mut out, mut chunk, mut dev) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>, 512usize, "cpu".to_string());
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_default();
        match k.as_str() {
            "--safetensors" => st = Some(v.into()),
            "--calib" => calib = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--chunk" => chunk = v.parse()?,
            "--device" => dev = v,
            _ => bail!("unknown argument {k}"),
        }
    }
    let (Some(st), Some(calib), Some(out)) = (st, calib, out) else {
        bail!("--safetensors DIR --calib FILE --out FILE required")
    };
    let device = raon_infer::device::select_device(dev.parse()?, 0)?;
    let g = Generator::load_safetensors(&st, &st.join("tokenizer.json"), device)?;
    let text = std::fs::read_to_string(&calib)?;
    let mut chunks: Vec<Vec<u32>> = vec![vec![]];
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        let ids = g.encode(&format!("{para}\n"))?;
        if chunks.last().unwrap().len() + ids.len() > chunk && !chunks.last().unwrap().is_empty() {
            chunks.push(vec![]);
        }
        chunks.last_mut().unwrap().extend(ids.into_iter().take(chunk));
    }
    let total: usize = chunks.iter().map(Vec::len).sum();
    eprintln!("[raon-imatrix] {} chunks, {total} tokens", chunks.len());
    imatrix::ON.store(true, Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    for (i, c) in chunks.iter().enumerate() {
        let mut s = g.model.new_state();
        g.model.forward(c, &mut s)?;
        eprintln!(
            "[raon-imatrix] chunk {}/{} ({} tokens) {:.0}s",
            i + 1,
            chunks.len(),
            c.len(),
            t0.elapsed().as_secs_f32()
        );
    }
    let n = imatrix::write(&out)?;
    eprintln!("[raon-imatrix] wrote {n} entries to {}", out.display());
    raon_infer::device::rocm_exit_now_if_used(0);
    Ok(())
}
