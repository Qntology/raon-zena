//! # raon-vision
//!
//! KRAFTON **Raon-VisionEncoder** (LocCa ViT-SO400M-16-SigLIP2, NaFlex) on candle, for Zena's periodic screen watcher.
//!
//! * [`tower`]     – the vision tower and MAP head, dense pass plus noto-sandbox's sparse passes
//!   (`forward_sparse` / `extend_sparse` / `merge_sparse`).
//! * [`perforate`] – checkerboard / fovea first passes and the minesweeper opening (port of noto-sandbox).
//! * [`text`]      – the text tower and SigLIP2 tokenizer, used when the dictionary changes to compute anchors.
//! * [`bank`]      – zero-shot screen domains from the dictionary's `vision` nodes and their anchors.
//! * [`screen`]    – one screen reading: encode, score, locate.

pub mod bank;
pub mod perforate;
pub mod screen;
pub mod text;
pub mod tower;

use anyhow::{Context, Result};
use candle_core::Device;
use raon_quant::safetensors::SafeTensors;
use std::path::{Path, PathBuf};

pub use bank::{AnchorBank, VisionScore};
pub use perforate::{ExpandCfg, PerfPattern, PerfReport, PerforatedCfg};
pub use screen::{ScreenConfig, ScreenEncoder, ScreenReading};
pub use tower::{VisionTower, WeightFormat};

/// The checkpoint file of a Raon-VisionEncoder directory (or the file itself).
pub fn checkpoint(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("model.safetensors")
    } else {
        path.to_path_buf()
    }
}

/// A name for the checkpoint that changes when the weights change (anchors computed by another checkpoint are stale).
pub fn fingerprint(path: &Path) -> String {
    let f = checkpoint(path);
    let size = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
    format!("raon-ve:{size}")
}

/// The vision tower and SigLIP's calibration `(exp(logit_scale), logit_bias)`.
pub fn load_vision(path: &Path, device: &Device, format: WeightFormat) -> Result<(VisionTower, f32, f32)> {
    let f = checkpoint(path);
    let st = SafeTensors::open(&[f.clone()]).with_context(|| f.display().to_string())?;
    let tower = VisionTower::load(&st, device, format)?;
    let scale = st.all_f32("logit_scale")?[0].exp();
    let bias = st.all_f32("logit_bias")?[0];
    Ok((tower, scale, bias))
}
