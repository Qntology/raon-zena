//! # zena-sensor
//!
//! Stage 1 of Raon Zena: a *virtual event camera* over the captured game frame.
//!
//! * [`EVENT_DIFF_WGSL`] – the GPU compute shader (runs on the single shared `wgpu` device).
//! * [`LogLumSensor`]    – a CPU reference of the exact same algorithm (tests, CPU fallback).
//! * [`extract_rois`]    – tile clustering into regions of interest with polar statistics.
//! * [`RoiTracker`]      – frame-to-frame association and Inward/Outward flow classification.
//! * [`naflex`]          – NaFlex patch extraction for `Raon-VisionEncoder` (variable aspect ratio).

pub mod naflex;

use zena_core::geometry::{classify_flow, BBox, FlowDirection, FlowThresholds, Polar, ScreenGeometry};

pub const EVENT_DIFF_WGSL: &str = include_str!("event_diff.wgsl");
pub const TILE: usize = 32;
const EPS: f32 = 1.0 / 255.0;

/// Per-tile accumulators (same layout as the shader's storage buffers).
#[derive(Debug, Clone, Default)]
pub struct TileStats {
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub count: Vec<u32>,
    /// Σ|ΔL| (f32 here; the shader stores milli-units in u32).
    pub energy: Vec<f32>,
    /// Number of positive-polarity (brightening) events.
    pub on: Vec<u32>,
}

/// CPU reference sensor holding the previous log-luminance frame.
pub struct LogLumSensor {
    pub width: usize,
    pub height: usize,
    /// Contrast threshold `C` on |Δ log I|.
    pub threshold: f32,
    prev: Option<Vec<f32>>,
}

#[inline]
pub fn log_luma(r: u8, g: u8, b: u8) -> f32 {
    let y = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
    (y + EPS).ln()
}

impl LogLumSensor {
    pub fn new(width: usize, height: usize, threshold: f32) -> Self {
        Self {
            width,
            height,
            threshold,
            prev: None,
        }
    }

    /// Process one BGRA8 frame (`DXGI_FORMAT_B8G8R8A8_UNORM`, tightly packed). The first frame
    /// only primes the reference and yields no events.
    pub fn process_bgra(&mut self, bgra: &[u8]) -> TileStats {
        assert_eq!(bgra.len(), self.width * self.height * 4, "frame size mismatch");
        let tx = self.width.div_ceil(TILE);
        let ty = self.height.div_ceil(TILE);
        let mut st = TileStats {
            tiles_x: tx,
            tiles_y: ty,
            count: vec![0; tx * ty],
            energy: vec![0.0; tx * ty],
            on: vec![0; tx * ty],
        };
        let first = self.prev.is_none();
        let prev = self.prev.get_or_insert_with(|| vec![0.0; self.width * self.height]);
        for y in 0..self.height {
            for x in 0..self.width {
                let i = y * self.width + x;
                let p = &bgra[i * 4..i * 4 + 4];
                let l = log_luma(p[2], p[1], p[0]);
                let d = l - prev[i];
                prev[i] = l;
                if !first && d.abs() >= self.threshold {
                    let t = (y / TILE) * tx + x / TILE;
                    st.count[t] += 1;
                    st.energy[t] += d.abs().min(16.0);
                    if d > 0.0 {
                        st.on[t] += 1;
                    }
                }
            }
        }
        st
    }
}

/// A connected cluster of active tiles.
#[derive(Debug, Clone)]
pub struct Roi {
    pub bbox: BBox,
    /// Energy-weighted centroid in pixels.
    pub centroid: (f32, f32),
    pub polar: Polar,
    /// Energy-weighted mean normalised radius (used for burst expansion/contraction).
    pub mean_r: f32,
    pub events: u32,
    pub energy: f32,
    /// Fraction of brightening events.
    pub on_ratio: f32,
}

/// Cluster tiles with at least `min_events` events (8-connectivity) into ROIs, largest energy first.
pub fn extract_rois(st: &TileStats, screen: &ScreenGeometry, min_events: u32) -> Vec<Roi> {
    let (tx, ty) = (st.tiles_x, st.tiles_y);
    let active = |t: usize| st.count[t] >= min_events;
    let mut seen = vec![false; tx * ty];
    let mut out = Vec::new();
    for start in 0..tx * ty {
        if seen[start] || !active(start) {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        let (mut e, mut n, mut on, mut cx, mut cy, mut rr) = (0f32, 0u32, 0u32, 0f32, 0f32, 0f32);
        while let Some(t) = stack.pop() {
            let (x, y) = (t % tx, t / tx);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
            let w = st.energy[t];
            let (px, py) = ((x * TILE + TILE / 2) as f32, (y * TILE + TILE / 2) as f32);
            e += w;
            n += st.count[t];
            on += st.on[t];
            cx += w * px;
            cy += w * py;
            rr += w * screen.to_polar(px, py).r;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= tx as i32 || ny >= ty as i32 {
                        continue;
                    }
                    let nt = ny as usize * tx + nx as usize;
                    if !seen[nt] && active(nt) {
                        seen[nt] = true;
                        stack.push(nt);
                    }
                }
            }
        }
        let w = e.max(f32::MIN_POSITIVE);
        let centroid = (cx / w, cy / w);
        out.push(Roi {
            bbox: BBox::new(
                (x0 * TILE) as f32,
                (y0 * TILE) as f32,
                (((x1 + 1) * TILE) as f32).min(screen.width),
                (((y1 + 1) * TILE) as f32).min(screen.height),
            ),
            centroid,
            polar: screen.to_polar(centroid.0, centroid.1),
            mean_r: rr / w,
            events: n,
            energy: e,
            on_ratio: if n > 0 { on as f32 / n as f32 } else { 0.0 },
        });
    }
    out.sort_by(|a, b| b.energy.total_cmp(&a.energy));
    out
}

/// A tracked ROI with its classified flow.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub roi: Roi,
    pub flow: FlowDirection,
    pub radial_velocity: f32,
    pub age: u32,
    last_ms: u64,
}

/// Greedy nearest-centroid tracker.
pub struct RoiTracker {
    pub max_match_px: f32,
    pub thresholds: FlowThresholds,
    /// Tracks not seen for this long are dropped.
    pub ttl_ms: u64,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Default for RoiTracker {
    fn default() -> Self {
        Self {
            max_match_px: 160.0,
            thresholds: FlowThresholds::default(),
            ttl_ms: 400,
            tracks: Vec::new(),
            next_id: 1,
        }
    }
}

impl RoiTracker {
    pub fn update(&mut self, rois: Vec<Roi>, now_ms: u64) -> &[Track] {
        let mut used = vec![false; self.tracks.len()];
        let mut fresh = Vec::new();
        for roi in rois {
            let best = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(i, _)| !used[*i])
                .map(|(i, t)| {
                    let d = ((t.roi.centroid.0 - roi.centroid.0).powi(2) + (t.roi.centroid.1 - roi.centroid.1).powi(2)).sqrt();
                    (i, d)
                })
                .filter(|(_, d)| *d <= self.max_match_px)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((i, _)) => {
                    used[i] = true;
                    let t = &mut self.tracks[i];
                    let dt = (now_ms.saturating_sub(t.last_ms)) as f32 / 1000.0;
                    let (flow, vr) = classify_flow(t.roi.polar, roi.polar, dt, self.thresholds);
                    t.flow = flow;
                    t.radial_velocity = vr;
                    t.roi = roi;
                    t.age += 1;
                    t.last_ms = now_ms;
                }
                None => {
                    fresh.push(Track {
                        id: self.next_id,
                        roi,
                        flow: FlowDirection::Static,
                        radial_velocity: 0.0,
                        age: 0,
                        last_ms: now_ms,
                    });
                    self.next_id += 1;
                }
            }
        }
        let ttl = self.ttl_ms;
        self.tracks.retain(|t| now_ms.saturating_sub(t.last_ms) <= ttl);
        self.tracks.extend(fresh);
        &self.tracks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, sq: Option<(usize, usize, usize)>) -> Vec<u8> {
        let mut f = vec![20u8; w * h * 4];
        if let Some((sx, sy, s)) = sq {
            for y in sy..(sy + s).min(h) {
                for x in sx..(sx + s).min(w) {
                    let i = (y * w + x) * 4;
                    f[i..i + 3].copy_from_slice(&[230, 230, 230]);
                }
            }
        }
        f
    }

    #[test]
    fn first_frame_primes_only_and_static_scene_is_silent() {
        let (w, h) = (256, 192);
        let mut s = LogLumSensor::new(w, h, 0.15);
        assert!(s.process_bgra(&frame(w, h, None)).count.iter().all(|&c| c == 0));
        assert!(s.process_bgra(&frame(w, h, None)).count.iter().all(|&c| c == 0));
    }

    #[test]
    fn approaching_object_is_classified_inward() {
        let (w, h) = (640, 480);
        let geo = ScreenGeometry::new(w as u32, h as u32);
        let mut s = LogLumSensor::new(w, h, 0.15);
        let mut tr = RoiTracker::default();
        s.process_bgra(&frame(w, h, None));
        // A bright square moves from the top-right corner toward the centre.
        let path = [(540, 20), (480, 70), (420, 120), (370, 160)];
        let mut last_flow = FlowDirection::Static;
        for (k, &(x, y)) in path.iter().enumerate() {
            let st = s.process_bgra(&frame(w, h, Some((x, y, 48))));
            let rois = extract_rois(&st, &geo, 8);
            assert!(!rois.is_empty());
            let tracks = tr.update(rois, (k as u64 + 1) * 100);
            if k > 0 {
                last_flow = tracks.iter().max_by(|a, b| a.roi.energy.total_cmp(&b.roi.energy)).unwrap().flow;
            }
        }
        assert_eq!(last_flow, FlowDirection::Inward);
    }

    #[test]
    fn roi_bbox_covers_change() {
        let (w, h) = (320, 240);
        let geo = ScreenGeometry::new(w as u32, h as u32);
        let mut s = LogLumSensor::new(w, h, 0.15);
        s.process_bgra(&frame(w, h, None));
        let st = s.process_bgra(&frame(w, h, Some((100, 60, 40))));
        let r = &extract_rois(&st, &geo, 4)[0];
        assert!(
            r.bbox.x0 <= 100.0 && r.bbox.x1 >= 140.0 && r.bbox.y0 <= 60.0 && r.bbox.y1 >= 100.0,
            "{:?}",
            r.bbox
        );
        assert!(r.on_ratio > 0.99);
    }

    #[test]
    fn shader_source_is_embedded() {
        assert!(EVENT_DIFF_WGSL.contains("@compute"));
        assert!(EVENT_DIFF_WGSL.contains("const TILE: u32 = 32u;"));
    }
}

#[cfg(test)]
mod shader_validation {
    #[test]
    fn wgsl_parses_and_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(super::EVENT_DIFF_WGSL).expect("WGSL parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .expect("WGSL validation");
        assert_eq!(module.entry_points[0].workgroup_size, [16, 16, 1]);
    }
}
