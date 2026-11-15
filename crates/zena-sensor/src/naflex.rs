//! NaFlex patchification for `Raon-VisionEncoder` (`LocCa ViT-SO400M-16-SigLIP2`).
//!
//! Port of the released processor (`modeling_raonve.py::RaonVEProcessor` and
//! `raon_vision_encoder/transform.py::get_image_size_for_max_num_patches`): resize while keeping
//! aspect ratio so `ceil(h/16)*ceil(w/16) <= max_patches`, normalise with the CLIP mean/std,
//! and flatten each 16×16 patch as `[C, ph, pw]` (channel-major), zero-padding to `max_patches`.

pub const PATCH: usize = 16;
pub const DEFAULT_MAX_PATCHES: usize = 256;
pub const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const STD: [f32; 3] = [0.268_629_5, 0.261_302_6, 0.275_777_1];

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
