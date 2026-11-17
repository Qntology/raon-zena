//! NaFlex patchification for the SigLIP2 So400m/16 NaFlex encoders.
//!
//! Port of the released processors: Raon-VisionEncoder's (`modeling_raonve.py::RaonVEProcessor` and
//! `raon_vision_encoder/transform.py::get_image_size_for_max_num_patches`: PIL bicubic, CLIP mean/std) and
//! transformers' `Siglip2ImageProcessor` for `google/siglip2-so400m-patch16-naflex` (bilinear, mean = std = 0.5,
//! i.e. `pixel / 127.5 − 1`). Both resize while keeping aspect ratio so `ceil(h/16)*ceil(w/16) <= max_patches`
//! (the same search) and cut 16×16 patches. Patches are flattened here as `[C, ph, pw]` (channel-major) for both;
//! a Hugging Face checkpoint's patch projection is reordered to match when it is loaded (`raon_vision::layout`).
//! [`Preprocess`] holds what differs.

pub const PATCH: usize = 16;
pub const DEFAULT_MAX_PATCHES: usize = 256;
pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const STD: [f32; 3] = [0.268_629_5, 0.261_302_6, 0.275_777_1];

/// The resampling filter of the processor (PIL's, with the kernel stretched when shrinking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resample {
    /// `PIL.Image.BICUBIC` (3): Raon-VisionEncoder.
    Bicubic,
    /// `PIL.Image.BILINEAR` (2): `Siglip2ImageProcessor`.
    Bilinear,
}

impl Resample {
    /// PIL's numbering, as `preprocessor_config.json` writes it (`"resample": 2`).
    pub fn from_pil(code: u64) -> Option<Resample> {
        match code {
            2 => Some(Resample::Bilinear),
            3 => Some(Resample::Bicubic),
            _ => None,
        }
    }
}

/// What a checkpoint's image processor does to pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preprocess {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub resample: Resample,
}

impl Preprocess {
    /// Raon-VisionEncoder: PIL bicubic, CLIP mean / std.
    pub const RAON_VE: Preprocess = Preprocess {
        mean: MEAN,
        std: STD,
        resample: Resample::Bicubic,
    };
    /// `google/siglip2-*` (`Siglip2ImageProcessor`): bilinear, mean = std = 0.5.
    pub const SIGLIP2: Preprocess = Preprocess {
        mean: [0.5; 3],
        std: [0.5; 3],
        resample: Resample::Bilinear,
    };
}

/// Binary search identical to the Python reference (same eps and bounds).
pub fn image_size_for_max_patches(h: usize, w: usize, patch: usize, max_patches: usize) -> (usize, usize) {
    let (mut lo, mut hi) = (1e-6f64, 100.0f64);
    let size = |s: f64| {
        let th = patch.max(((h as f64 * s / patch as f64).ceil() as usize) * patch);
        let tw = patch.max(((w as f64 * s / patch as f64).ceil() as usize) * patch);
        (th, tw)
    };
    while hi - lo >= 1e-5 {
        let mid = (lo + hi) / 2.0;
        let (th, tw) = size(mid);
        if (th / patch) * (tw / patch) <= max_patches {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    size(lo)
}

/// PIL's bicubic kernel (`a = -0.5`), support 2.
fn bicubic(x: f64) -> f64 {
    let a = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

/// PIL's bilinear (triangle) kernel, support 1.
fn triangle(x: f64) -> f64 {
    (1.0 - x.abs()).max(0.0)
}

/// Per output pixel: the first input index and the normalised weights (PIL `precompute_coeffs`: the kernel is
/// stretched by the scale when shrinking, which is what makes PIL's resize antialiased).
fn coeffs(in_size: usize, out_size: usize, resample: Resample) -> Vec<(usize, Vec<f64>)> {
    let (kernel, base_support): (fn(f64) -> f64, f64) = match resample {
        Resample::Bicubic => (bicubic, 2.0),
        Resample::Bilinear => (triangle, 1.0),
    };
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = base_support * filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + support + 0.5).floor() as usize).min(in_size);
            let mut w: Vec<f64> = (xmin..xmax).map(|x| kernel((x as f64 - center + 0.5) / filterscale)).collect();
            let sum: f64 = w.iter().sum();
            if sum != 0.0 {
                w.iter_mut().for_each(|v| *v /= sum);
            }
            (xmin, w)
        })
        .collect()
}

/// Resizes interleaved 8-bit pixels (`ch` channels) like `PIL.Image.resize(..., BICUBIC)`: horizontal pass, round
/// and clip to 8 bits, vertical pass (the order and the intermediate rounding PIL uses).
pub fn resize_bicubic(src: &[u8], w: usize, h: usize, ch: usize, tw: usize, th: usize) -> Vec<u8> {
    resize(src, w, h, ch, tw, th, Resample::Bicubic)
}

/// [`resize_bicubic`] with the filter chosen (`PIL.Image.resize(..., BILINEAR)` for [`Resample::Bilinear`]).
pub fn resize(src: &[u8], w: usize, h: usize, ch: usize, tw: usize, th: usize, resample: Resample) -> Vec<u8> {
    let clip = |v: f64| v.round().clamp(0.0, 255.0) as u8;
    let horizontal = if tw == w {
        src[..w * h * ch].to_vec()
    } else {
        let cx = coeffs(w, tw, resample);
        let mut out = vec![0u8; tw * h * ch];
        for y in 0..h {
            for (ox, (x0, ws)) in cx.iter().enumerate() {
                for c in 0..ch {
                    let v: f64 = ws
                        .iter()
                        .enumerate()
                        .map(|(k, wt)| wt * src[(y * w + x0 + k) * ch + c] as f64)
                        .sum();
                    out[(y * tw + ox) * ch + c] = clip(v);
                }
            }
        }
        out
    };
    if th == h {
        return horizontal;
    }
    let cy = coeffs(h, th, resample);
    let mut out = vec![0u8; tw * th * ch];
    for (oy, (y0, ws)) in cy.iter().enumerate() {
        for x in 0..tw {
            for c in 0..ch {
                let v: f64 = ws
                    .iter()
                    .enumerate()
                    .map(|(k, wt)| wt * horizontal[((y0 + k) * tw + x) * ch + c] as f64)
                    .sum();
                out[(oy * tw + x) * ch + c] = clip(v);
            }
        }
    }
    out
}

/// The patches of a whole image on its NaFlex grid, without padding: `patches[cell * 768..]` is cell `cell`
/// (row-major), channel-major inside the patch like the released processor.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchGrid {
    pub grid_h: usize,
    pub grid_w: usize,
    pub patches: Vec<f32>,
}

impl PatchGrid {
    pub const DIM: usize = 3 * PATCH * PATCH;
    pub fn cells(&self) -> usize {
        self.grid_h * self.grid_w
    }
    pub fn patch(&self, cell: usize) -> &[f32] {
        &self.patches[cell * Self::DIM..(cell + 1) * Self::DIM]
    }
}

/// `RaonVEProcessor` on an RGB8 image: PIL-bicubic resize to the largest grid within `max_patches`, CLIP mean/std,
/// `[C, ph, pw]` patches.
pub fn patchify_frame(rgb: &[u8], w: usize, h: usize, max_patches: usize) -> PatchGrid {
    patchify_frame_with(rgb, w, h, max_patches, &Preprocess::RAON_VE)
}

/// [`patchify_frame`] with the checkpoint's own processor settings ([`Preprocess::SIGLIP2`] for google's).
pub fn patchify_frame_with(rgb: &[u8], w: usize, h: usize, max_patches: usize, pre: &Preprocess) -> PatchGrid {
    let (th, tw) = image_size_for_max_patches(h, w, PATCH, max_patches);
    let img = resize(rgb, w, h, 3, tw, th, pre.resample);
    let (gh, gw) = (th / PATCH, tw / PATCH);
    let mut patches = vec![0f32; gh * gw * PatchGrid::DIM];
    for py in 0..gh {
        for px in 0..gw {
            let base = (py * gw + px) * PatchGrid::DIM;
            for c in 0..3 {
                for yy in 0..PATCH {
                    for xx in 0..PATCH {
                        let v = img[((py * PATCH + yy) * tw + px * PATCH + xx) * 3 + c] as f32 / 255.0;
                        patches[base + c * PATCH * PATCH + yy * PATCH + xx] = (v - pre.mean[c]) / pre.std[c];
                    }
                }
            }
        }
    }
    PatchGrid {
        grid_h: gh,
        grid_w: gw,
        patches,
    }
}

/// BGRA8 (screen capture) to RGB8.
pub fn bgra_to_rgb(bgra: &[u8]) -> Vec<u8> {
    bgra.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0]]).collect()
}

pub struct NaFlexInput {
    /// `[max_patches, 3*16*16]`, row-major.
    pub patches: Vec<f32>,
    /// `true` for real patches, `false` for padding.
    pub valid: Vec<bool>,
    /// `(grid_h, grid_w)`.
    pub spatial_shape: (usize, usize),
}

/// Crop `(x0,y0,x1,y1)` out of an RGB8 image and patchify it (bilinear resize, half-pixel centres).
pub fn patchify_rgb(rgb: &[u8], img_w: usize, img_h: usize, crop: (usize, usize, usize, usize), max_patches: usize) -> NaFlexInput {
    let (x0, y0, x1, y1) = crop;
    let (cw, ch) = ((x1 - x0).max(1), (y1 - y0).max(1));
    let (th, tw) = image_size_for_max_patches(ch, cw, PATCH, max_patches);
    let (gh, gw) = (th / PATCH, tw / PATCH);
    let mut resized = vec![0f32; 3 * th * tw];
    let sx = cw as f32 / tw as f32;
    let sy = ch as f32 / th as f32;
    for oy in 0..th {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).clamp(0.0, (ch - 1) as f32);
        let (iy, wy) = (fy.floor() as usize, fy.fract());
        let iy1 = (iy + 1).min(ch - 1);
        for ox in 0..tw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).clamp(0.0, (cw - 1) as f32);
            let (ix, wx) = (fx.floor() as usize, fx.fract());
            let ix1 = (ix + 1).min(cw - 1);
            for c in 0..3 {
                let px = |x: usize, y: usize| rgb[((y0 + y).min(img_h - 1) * img_w + (x0 + x).min(img_w - 1)) * 3 + c] as f32 / 255.0;
                let top = px(ix, iy) * (1.0 - wx) + px(ix1, iy) * wx;
                let bot = px(ix, iy1) * (1.0 - wx) + px(ix1, iy1) * wx;
                let v = top * (1.0 - wy) + bot * wy;
                resized[c * th * tw + oy * tw + ox] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    let n = gh * gw;
    let dim = 3 * PATCH * PATCH;
    let mut patches = vec![0f32; max_patches * dim];
    // t.reshape(3, gh, ps, gw, ps).permute(1, 3, 0, 2, 4).reshape(n, 3*ps*ps)
    for py in 0..gh {
        for px in 0..gw {
            let p = py * gw + px;
            let mut k = 0;
            for c in 0..3 {
                for yy in 0..PATCH {
                    for xx in 0..PATCH {
                        patches[p * dim + k] = resized[c * th * tw + (py * PATCH + yy) * tw + px * PATCH + xx];
                        k += 1;
                    }
                }
            }
        }
    }
    let valid = (0..max_patches).map(|i| i < n).collect();
    NaFlexInput {
        patches,
        valid,
        spatial_shape: (gh, gw),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_respect_budget_and_aspect() {
        assert_eq!(image_size_for_max_patches(256, 256, 16, 256), (256, 256));
        let (h, w) = image_size_for_max_patches(100, 400, 16, 256);
        assert!((h / 16) * (w / 16) <= 256);
        assert!(w > 3 * h);
        let (h, w) = image_size_for_max_patches(1080, 1920, 16, 256);
        assert!((h / 16) * (w / 16) <= 256 && (h / 16) * (w / 16) >= 230, "{h}x{w}");
    }

    #[test]
    fn bicubic_is_identity_at_scale_one_and_averages_when_shrinking() {
        let src: Vec<u8> = (0..4 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        assert_eq!(resize_bicubic(&src, 4, 3, 3, 4, 3), src);
        // A 1-pixel checkerboard shrunk 4x is flat grey (the stretched kernel averages, no aliasing).
        let (w, h) = (64, 64);
        let cb: Vec<u8> = (0..w * h).map(|i| if (i % w + i / w) % 2 == 0 { 255 } else { 0 }).collect();
        let small = resize_bicubic(&cb, w, h, 1, 16, 16);
        assert!(small[1..15].iter().all(|&v| (120..=135).contains(&v)), "{:?}", &small[..16]);
    }

    #[test]
    fn frames_patchify_without_padding() {
        let (w, h) = (480, 270);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let g = patchify_frame(&rgb, w, h, 144);
        assert!(g.cells() <= 144 && g.cells() >= 120, "{}x{}", g.grid_h, g.grid_w);
        assert_eq!(g.patches.len(), g.cells() * PatchGrid::DIM);
        assert!(g.grid_w > g.grid_h);
        assert_eq!(bgra_to_rgb(&[1, 2, 3, 255, 4, 5, 6, 255]), vec![3, 2, 1, 6, 5, 4]);
    }

    #[test]
    fn patch_layout_is_channel_major() {
        // 32x16 image: left half red, right half blue → 2 patches.
        let (w, h) = (32, 16);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                if x < 16 {
                    rgb[i] = 255
                } else {
                    rgb[i + 2] = 255
                }
            }
        }
        let inp = patchify_rgb(&rgb, w, h, (0, 0, w, h), 4);
        assert_eq!(inp.spatial_shape, (1, 2));
        assert_eq!(inp.valid, vec![true, true, false, false]);
        let red0 = inp.patches[0];
        assert!((red0 - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-5);
        let blue_in_p1 = inp.patches[768 + 2 * 256];
        assert!((blue_in_p1 - (1.0 - MEAN[2]) / STD[2]).abs() < 1e-5);
        assert!(inp.patches[2 * 768..].iter().all(|&v| v == 0.0));
    }
}
