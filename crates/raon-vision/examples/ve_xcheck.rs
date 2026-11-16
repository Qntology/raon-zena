//! Cross-checks the candle Raon-VisionEncoder against the official PyTorch model (`scripts/ve_reference.py`).
//!
//! ```text
//! cargo run --release -p raon-vision --example ve_xcheck -- MODEL_DIR TOKENIZER_JSON REF_DIR [f32|f16|bf16|q8] [checker|fovea|checker-x|fovea-x ...]
//! ```
//! For each reference screen and budget: the official processor's patches through our tower (embedding cosine),
//! our own PIL-bicubic patchify (patch difference and embedding cosine), and — with patterns given — the perforated
//! passes against the full pass (embedding cosine, top-1 agreement on the reference prompts, encoder time).

use anyhow::{Context, Result};
use raon_vision::perforate::PerforatedCfg;
use raon_vision::text::{SiglipTokenizer, TextTower};
use raon_vision::tower::{cosine, WeightFormat};
use std::path::PathBuf;
use std::time::Instant;
use zena_sensor::naflex::{patchify_frame, PatchGrid};

fn read_f32(p: &PathBuf) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| p.display().to_string())?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        anyhow::bail!("usage: ve_xcheck MODEL_DIR TOKENIZER_JSON REF_DIR [format] [patterns...]");
    }
    let (model, tokp, refd) = (PathBuf::from(&a[0]), PathBuf::from(&a[1]), PathBuf::from(&a[2]));
    let format: WeightFormat = a.get(3).map(|s| s.parse()).transpose()?.unwrap_or(WeightFormat::F32);
    let patterns: Vec<PerforatedCfg> = a.iter().skip(4).filter_map(|s| PerforatedCfg::parse(s)).collect();
    let dev = raon_infer::device::select_device(raon_infer::device::Backend::Auto, 0)?;
    let t = Instant::now();
    let (tower, scale, bias) = raon_vision::load_vision(&model, &dev, format)?;
    eprintln!(
        "vision tower ({format:?}) loaded in {:.1}s on {}",
        t.elapsed().as_secs_f32(),
        raon_infer::device::describe(&dev)
    );

    // Text: token ids and embeddings.
    let texts: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refd.join("texts.json"))?)?;
    let prompts: Vec<String> = texts["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let tok = SiglipTokenizer::load(&tokp)?;
    let mut ids_ok = 0;
    for (i, p) in prompts.iter().enumerate() {
        let want: Vec<u32> = texts["ids"][i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if tok.encode(p)? == want {
            ids_ok += 1;
        } else {
            eprintln!("  ids differ for {p:?}: ours {:?}", &tok.encode(p)?[..12]);
        }
    }
    println!("tokenizer: {ids_ok}/{} prompts give the official ids", prompts.len());
    let t = Instant::now();
    let text = TextTower::load(&raon_vision::checkpoint(&model), &dev, format)?;
    let ours = text.embed(&tok, &prompts, 8)?;
    let mut worst = 1f32;
    let ref_text: Vec<Vec<f32>> = (0..prompts.len())
        .map(|i| {
            texts["embeddings"][i]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();
    for (o, r) in ours.iter().zip(&ref_text) {
        worst = worst.min(cosine(o, r));
    }
    println!(
        "text tower: worst cosine to the official embeddings {worst:.6} ({:.1}s); logit scale {scale:.3} bias {bias:.3}",
        t.elapsed().as_secs_f32()
    );
    drop(text);

    let images: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refd.join("images.json"))?)?;
    let mut names: Vec<&String> = images.as_object().unwrap().keys().collect();
    names.sort();
    let mut agree = vec![0usize; patterns.len()];
    let mut total = 0usize;
    for name in names {
        let (w, h) = (
            images[name]["w"].as_u64().unwrap() as usize,
            images[name]["h"].as_u64().unwrap() as usize,
        );
        let rgb = std::fs::read(refd.join(format!("{name}.rgb")))?;
        for budget in [256usize, 144] {
            let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refd.join(format!("{name}.{budget}.json")))?)?;
            let grid = r["grid"].as_array().unwrap();
            let (gh, gw) = (grid[0].as_u64().unwrap() as usize, grid[1].as_u64().unwrap() as usize);
            let want: Vec<f32> = r["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let official = PatchGrid {
                grid_h: gh,
                grid_w: gw,
                patches: read_f32(&refd.join(format!("{name}.{budget}.patches.f32")))?,
            };
            let t = Instant::now();
            let full = tower.forward_full(&official)?;
            let ms = t.elapsed().as_secs_f32() * 1000.0;
            let e = tower.pooled(&full)?;
            let mine = patchify_frame(&rgb, w, h, budget);
            let pd = mine
                .patches
                .iter()
                .zip(&official.patches)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let e2 = tower.pooled(&tower.forward_full(&mine)?)?;
            println!(
                "{name:>8} {budget} ({gh}x{gw}): official patches → cos {:.6} [{ms:.0} ms]; our patchify (max |Δ| {pd:.3}) → cos {:.6}",
                cosine(&e, &want),
                cosine(&e2, &want)
            );
            let top = |v: &[f32]| {
                ref_text
                    .iter()
                    .enumerate()
                    .max_by(|a, b| cosine(v, a.1).total_cmp(&cosine(v, b.1)))
                    .map(|(i, _)| i)
                    .unwrap()
            };
            total += 1;
            for (k, p) in patterns.iter().enumerate() {
                let cfg = raon_vision::ScreenConfig {
                    max_patches: budget,
                    perforation: Some(*p),
                    seed_labels: 3,
                };
                let bank = raon_vision::AnchorBank::from_vectors(&prompts, &ref_text, scale, bias);
                let t = Instant::now();
                let (pass, rep) = raon_vision::screen::encode_pass(&tower, &cfg, &official, &bank)?;
                let pms = t.elapsed().as_secs_f32() * 1000.0;
                let pe = tower.pooled(&pass)?;
                let same = top(&pe) == top(&e);
                agree[k] += same as usize;
                println!(
                    "         {:<10} first {:>3}/{} seeds {} opened {:>2} tokens {:>3}: cos to full {:.4}, top-1 {} [{pms:.0} ms, {:+.0}%]",
                    p.tag(),
                    rep.first,
                    rep.cells,
                    rep.seeds,
                    rep.opened,
                    rep.tokens_run,
                    cosine(&pe, &e),
                    if same { "same" } else { "DIFFERENT" },
                    (pms / ms - 1.0) * 100.0
                );
            }
        }
    }
    for (k, p) in patterns.iter().enumerate() {
        println!("{}: top-1 agrees with the full pass on {}/{} screens", p.tag(), agree[k], total);
    }
    Ok(())
}
