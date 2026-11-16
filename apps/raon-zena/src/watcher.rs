//! The periodic screen watcher: every few seconds the pipeline hands it one downscaled frame; it skips the frame when
//! the screen has not changed, otherwise runs the Raon-VisionEncoder vision tower (full or perforated pass) and scores
//! the dictionary's screen domains. Only the winning domain's key, its score and one gaze cell leave this thread.
//!
//! Privacy: frames live in memory only — never written, never logged — and are zeroed when dropped; the patches are
//! dropped after the encoder ran; nothing about the pixels is kept but a 32 × 18 luma thumbnail for change detection.
//! A screen the dictionary marks `privacy` (login, banking, payment, personal chat) gets no comment, and the watcher
//! stops looking for a while.

use raon_vision::{Acceptance, AnchorBank, ScreenConfig, ScreenEncoder, WeightFormat};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use zena_lexicon::{Lexicon, VisionKind};
use zena_sensor::naflex::patchify_frame;

/// One frame for the watcher (RGB8). Zeroed when dropped.
pub struct ScreenFrame {
    pub rgb: Vec<u8>,
    pub w: usize,
    pub h: usize,
}

impl ScreenFrame {
    /// Overwrites the pixels (volatile writes, so the zeroing is not optimised away before the free).
    fn wipe(&mut self) {
        for b in self.rgb.iter_mut() {
            // SAFETY: a valid, exclusive reference to an initialised byte.
            unsafe { std::ptr::write_volatile(b, 0) };
        }
    }
}

impl Drop for ScreenFrame {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// What the watcher saw.
#[derive(Clone, Debug)]
pub struct ScreenObservation {
    pub key: String,
    pub kind: VisionKind,
    /// The dictionary node whose expression Zena plays.
    pub link: String,
    pub score: f32,
    /// Cosine of the domain's best anchor.
    pub cos: f32,
    /// Centre of the domain's hottest cell, 0..1 screen coordinates.
    pub peak: Option<(f32, f32)>,
    pub cooldown_s: u32,
    pub encode_ms: f32,
    pub tokens: usize,
    pub cells: usize,
}

/// A dictionary (re)load for the watcher: the vision nodes and their anchors.
pub struct AnchorsUpdate {
    pub lexicon: Lexicon,
    pub anchors: HashMap<String, Vec<f32>>,
}

pub struct WatcherConfig {
    pub model: PathBuf,
    pub backend: raon_infer::device::Backend,
    pub format: Option<WeightFormat>,
    pub screen: ScreenConfig,
    /// Perforated pass only when the encoder runs on the CPU (noto's finding: on a GPU it saves little).
    pub perforate_on_cpu_only: bool,
    /// Mean absolute luma change (0–255) of the thumbnail under which a frame counts as unchanged.
    pub change_min: f32,
    /// When a reading counts (best cosine per kind, lead over the runner-up).
    pub accept: Acceptance,
    pub privacy_backoff: Duration,
}

impl WatcherConfig {
    pub fn new(model: PathBuf) -> Self {
        WatcherConfig {
            model,
            backend: raon_infer::device::Backend::Auto,
            format: None,
            screen: ScreenConfig::default(),
            perforate_on_cpu_only: true,
            change_min: 3.0,
            accept: Acceptance::default(),
            privacy_backoff: Duration::from_secs(60),
        }
    }
}

/// 32 × 18 mean-luma thumbnail.
fn thumbnail(f: &ScreenFrame) -> Vec<f32> {
    let (tw, th) = (32usize, 18usize);
    let mut out = vec![0f32; tw * th];
    let mut cnt = vec![0u32; tw * th];
    for y in 0..f.h {
        let ty = y * th / f.h.max(1);
        for x in 0..f.w {
            let tx = x * tw / f.w.max(1);
            let p = &f.rgb[(y * f.w + x) * 3..(y * f.w + x) * 3 + 3];
            out[ty * tw + tx] += 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
            cnt[ty * tw + tx] += 1;
        }
    }
    out.iter_mut().zip(&cnt).for_each(|(v, c)| *v /= (*c).max(1) as f32);
    out
}

fn changed(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len().max(1) as f32
}

/// Starts the watcher thread. It loads the vision tower first (and keeps it: it is the always-on part, ~0.4 B
/// parameters), then waits for frames and dictionary updates.
pub fn spawn(cfg: WatcherConfig, frames: Receiver<ScreenFrame>, updates: Receiver<AnchorsUpdate>, out: Sender<ScreenObservation>) {
    std::thread::Builder::new()
        .name("zena-screen".into())
        .spawn(move || {
            let dev = match raon_infer::device::select_device(cfg.backend, 0) {
                Ok(d) => d,
                Err(e) => return eprintln!("[screen] no device: {e:#}"),
            };
            let gpu = !dev.is_cpu();
            let format = cfg.format.unwrap_or(if gpu { WeightFormat::F16 } else { WeightFormat::F32 });
            let mut screen = cfg.screen;
            if gpu && cfg.perforate_on_cpu_only {
                screen.perforation = None;
            }
            let t = Instant::now();
            let (tower, scale, bias) = match raon_vision::load_vision(&cfg.model, &dev, format) {
                Ok(x) => x,
                Err(e) => return eprintln!("[screen] vision tower not loaded ({}): {e:#}", cfg.model.display()),
            };
            eprintln!(
                "[screen] Raon-VisionEncoder vision tower ({format:?}) on {} in {:.1}s; {} patches, {} pass",
                raon_infer::device::describe(&dev),
                t.elapsed().as_secs_f32(),
                screen.max_patches,
                screen.perforation.map(|p| p.tag()).unwrap_or_else(|| "full".into())
            );
            let enc = ScreenEncoder::new(tower, screen);
            let mut bank = AnchorBank::empty(scale, bias);
            let mut last_thumb: Option<Vec<f32>> = None;
            let mut paused_until: Option<Instant> = None;
            let mut warned = false;
            loop {
                for u in updates.try_iter() {
                    bank = AnchorBank::build(&u.lexicon, &u.anchors, scale, bias);
                    last_thumb = None;
                    eprintln!("[screen] {} screen domains anchored", bank.domains.len());
                }
                let frame = match frames.recv_timeout(Duration::from_millis(500)) {
                    Ok(f) => f,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(_) => return,
                };
                if bank.is_empty() {
                    if !warned {
                        eprintln!("[screen] no anchors yet: run `zena-lexicon anchors` once to let Zena read the screen");
                        warned = true;
                    }
                    continue;
                }
                if paused_until.is_some_and(|t| Instant::now() < t) {
                    continue;
                }
                let thumb = thumbnail(&frame);
                if last_thumb.as_ref().is_some_and(|l| changed(l, &thumb) < cfg.change_min) {
                    continue;
                }
                let grid = patchify_frame(&frame.rgb, frame.w, frame.h, enc.cfg.max_patches);
                drop(frame);
                let reading = match enc.read(&grid, &bank) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[screen] encoder failed: {e:#}");
                        continue;
                    }
                };
                drop(grid);
                last_thumb = Some(thumb);
                let Some(top) = cfg.accept.accept(&reading.scores) else { continue };
                if top.kind == VisionKind::Privacy {
                    paused_until = Some(Instant::now() + cfg.privacy_backoff);
                }
                {
                    let _ = out.send(ScreenObservation {
                        key: top.key.clone(),
                        kind: top.kind,
                        link: top.link.clone(),
                        score: top.score,
                        cos: top.cos,
                        peak: reading.peak,
                        cooldown_s: top.cooldown_s,
                        encode_ms: reading.encode_ms,
                        tokens: reading.report.tokens_run,
                        cells: reading.report.cells,
                    });
                }
            }
        })
        .expect("spawn screen watcher");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_are_zeroed_when_dropped_and_change_is_measured_on_a_thumbnail() {
        let f = ScreenFrame {
            rgb: vec![200; 64 * 36 * 3],
            w: 64,
            h: 36,
        };
        let a = thumbnail(&f);
        assert_eq!(a.len(), 32 * 18);
        assert!((a[0] - 200.0).abs() < 1e-3);
        let mut g = ScreenFrame {
            rgb: vec![200; 64 * 36 * 3],
            w: 64,
            h: 36,
        };
        assert_eq!(changed(&a, &thumbnail(&g)), 0.0);
        g.rgb[..64 * 18 * 3].iter_mut().for_each(|b| *b = 0);
        assert!(changed(&a, &thumbnail(&g)) > 50.0);
        let mut f = f;
        f.wipe();
        assert!(f.rgb.iter().all(|&b| b == 0), "what Drop does before the buffer is freed");
    }
}
