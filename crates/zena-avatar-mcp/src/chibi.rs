//! Track B-1: SD (chibi) emote stamps composed from layered PNG parts.
//!
//! A kit is a canvas size, an ordered slot list (`hair_back, body, face, eyes, mouth, hair_front, hands, fx`)
//! and per slot a set of named variant layers. An emote picks one variant per slot; [`ChibiKit::compose`]
//! stacks them with a per-emote animation (bounce / squash, hand wave, heart pulse, tear drip, floating fx)
//! and [`ChibiKit::atlas`] lays frames out as a horizontal strip.
//!
//! [`ChibiKit::builtin`] draws SD Zena procedurally with anti-aliased signed-distance shapes (purple twin-tail
//! hair, violet eyes, white / violet outfit). [`ChibiKit::load`] reads an artist kit from `kit.json`:
//!
//! ```json
//! { "size": [w, h], "slots": ["hair_back", ...],
//!   "layers": { "<slot>": { "<variant>": {"png": "file.png", "offset": [x, y]} } },
//!   "emotes": { "CHIBI_CHEER": {"eyes": "happy", "mouth": "open", "hands": "up", "fx": "sparkle"}, "NEUTRAL": {...} } }
//! ```
//!
//! Anything the kit leaves out falls back to the built-in variants (drawn at the kit size). PNGs are straight
//! (non-premultiplied) alpha.

use std::collections::HashMap;
use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use image::{Rgba, RgbaImage};
use serde_json::Value;
use zena_core::tools::CharacterEmote;

/// Default draw order.
pub const DEFAULT_SLOTS: [&str; 8] = ["hair_back", "body", "face", "eyes", "mouth", "hair_front", "hands", "fx"];
/// Built-in stamp size.
pub const DEFAULT_SIZE: (u32, u32) = (160, 160);
/// The built-in kit is designed on a 160-unit square.
const DESIGN: f32 = 160.0;

// ---------------------------------------------------------------------------------------------------------
// Signed-distance shapes (design units, y down)
// ---------------------------------------------------------------------------------------------------------

type Col = [f32; 4];

#[derive(Clone)]
struct Sdf {
    f: Rc<dyn Fn(f32, f32) -> f32>,
    bb: [f32; 4],
}

const FAR: [f32; 4] = [-1e5, -1e5, 1e5, 1e5];

impl Sdf {
    fn new(bb: [f32; 4], f: impl Fn(f32, f32) -> f32 + 'static) -> Sdf {
        Sdf { f: Rc::new(f), bb }
    }
    fn eval(&self, x: f32, y: f32) -> f32 {
        (self.f)(x, y)
    }
    fn union(self, o: Sdf) -> Sdf {
        let bb = [
            self.bb[0].min(o.bb[0]),
            self.bb[1].min(o.bb[1]),
            self.bb[2].max(o.bb[2]),
            self.bb[3].max(o.bb[3]),
        ];
        Sdf::new(bb, move |x, y| self.eval(x, y).min(o.eval(x, y)))
    }
    fn smooth_union(self, o: Sdf, k: f32) -> Sdf {
        let bb = [
            self.bb[0].min(o.bb[0]),
            self.bb[1].min(o.bb[1]),
            self.bb[2].max(o.bb[2]),
            self.bb[3].max(o.bb[3]),
        ];
        Sdf::new(bb, move |x, y| {
            let (a, b) = (self.eval(x, y), o.eval(x, y));
            let h = (k - (a - b).abs()).max(0.0) / k;
            a.min(b) - h * h * k * 0.25
        })
    }
    fn inter(self, o: Sdf) -> Sdf {
        let bb = [
            self.bb[0].max(o.bb[0]),
            self.bb[1].max(o.bb[1]),
            self.bb[2].min(o.bb[2]),
            self.bb[3].min(o.bb[3]),
        ];
        Sdf::new(bb, move |x, y| self.eval(x, y).max(o.eval(x, y)))
    }
    fn minus(self, o: Sdf) -> Sdf {
        let bb = self.bb;
        Sdf::new(bb, move |x, y| self.eval(x, y).max(-o.eval(x, y)))
    }
    fn grow(self, r: f32) -> Sdf {
        let bb = [self.bb[0] - r, self.bb[1] - r, self.bb[2] + r, self.bb[3] + r];
        Sdf::new(bb, move |x, y| self.eval(x, y) - r)
    }
}

fn circle(cx: f32, cy: f32, r: f32) -> Sdf {
    Sdf::new([cx - r, cy - r, cx + r, cy + r], move |x, y| {
        ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r
    })
}

fn ellipse_rot(cx: f32, cy: f32, rx: f32, ry: f32, angle: f32) -> Sdf {
    let m = rx.max(ry);
    let (s, c) = angle.sin_cos();
    Sdf::new([cx - m, cy - m, cx + m, cy + m], move |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let (px, py) = (c * dx + s * dy, -s * dx + c * dy);
        let k0 = ((px / rx).powi(2) + (py / ry).powi(2)).sqrt();
        let k1 = ((px / (rx * rx)).powi(2) + (py / (ry * ry)).powi(2)).sqrt();
        if k1 < 1e-6 {
            -rx.min(ry)
        } else {
            k0 * (k0 - 1.0) / k1
        }
    })
}

fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> Sdf {
    ellipse_rot(cx, cy, rx, ry, 0.0)
}

fn rbox(cx: f32, cy: f32, hw: f32, hh: f32, r: f32) -> Sdf {
    Sdf::new([cx - hw, cy - hh, cx + hw, cy + hh], move |x, y| {
        let qx = (x - cx).abs() - hw + r;
        let qy = (y - cy).abs() - hh + r;
        (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - r
    })
}

fn seg_dist(x: f32, y: f32, a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let (pax, pay, bax, bay) = (x - a.0, y - a.1, b.0 - a.0, b.1 - a.1);
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay).max(1e-9)).clamp(0.0, 1.0);
    (((pax - bax * h).powi(2) + (pay - bay * h).powi(2)).sqrt(), h)
}

/// Tapered capsule from `a` (radius `ra`) to `b` (radius `rb`).
fn capsule(a: (f32, f32), ra: f32, b: (f32, f32), rb: f32) -> Sdf {
    let m = ra.max(rb);
    let bb = [a.0.min(b.0) - m, a.1.min(b.1) - m, a.0.max(b.0) + m, a.1.max(b.1) + m];
    Sdf::new(bb, move |x, y| {
        let (d, h) = seg_dist(x, y, a, b);
        d - (ra + (rb - ra) * h)
    })
}

fn polyline(pts: Vec<(f32, f32)>, r: f32) -> Sdf {
    let bb = pts.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| {
        [b[0].min(p.0), b[1].min(p.1), b[2].max(p.0), b[3].max(p.1)]
    });
    Sdf::new([bb[0] - r, bb[1] - r, bb[2] + r, bb[3] + r], move |x, y| {
        pts.windows(2).map(|w| seg_dist(x, y, w[0], w[1]).0).fold(f32::MAX, f32::min) - r
    })
}

fn polygon(pts: Vec<(f32, f32)>) -> Sdf {
    let bb = pts.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |b, p| {
        [b[0].min(p.0), b[1].min(p.1), b[2].max(p.0), b[3].max(p.1)]
    });
    Sdf::new(bb, move |x, y| {
        let n = pts.len();
        let mut d = f32::MAX;
        let mut s = 1.0;
        let mut j = n - 1;
        for i in 0..n {
            let (vi, vj) = (pts[i], pts[j]);
            let (ex, ey) = (vj.0 - vi.0, vj.1 - vi.1);
            let (wx, wy) = (x - vi.0, y - vi.1);
            let h = ((wx * ex + wy * ey) / (ex * ex + ey * ey).max(1e-12)).clamp(0.0, 1.0);
            let (bx, by) = (wx - ex * h, wy - ey * h);
            d = d.min(bx * bx + by * by);
            let c = (y >= vi.1, y < vj.1, ex * wy > ey * wx);
            if (c.0 && c.1 && c.2) || (!c.0 && !c.1 && !c.2) {
                s = -s;
            }
            j = i;
        }
        s * d.sqrt()
    })
}

/// Inside where `(p - (px, py)) · n < 0` (`n` is normalised).
fn half_plane(px: f32, py: f32, nx: f32, ny: f32) -> Sdf {
    let l = (nx * nx + ny * ny).sqrt().max(1e-9);
    let (nx, ny) = (nx / l, ny / l);
    Sdf::new(FAR, move |x, y| (x - px) * nx + (y - py) * ny)
}

/// Points on a circle arc; angles in degrees, screen convention (y down, 270° = up).
fn arc_pts(cx: f32, cy: f32, r: f32, a0: f32, a1: f32, n: usize) -> Vec<(f32, f32)> {
    (0..=n)
        .map(|i| {
            let a = (a0 + (a1 - a0) * i as f32 / n as f32).to_radians();
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect()
}

fn star(cx: f32, cy: f32, r_out: f32, r_in: f32) -> Sdf {
    polygon(
        (0..10)
            .map(|i| {
                let a = -PI / 2.0 + PI * i as f32 / 5.0;
                let r = if i % 2 == 0 { r_out } else { r_in };
                (cx + r * a.cos(), cy + r * a.sin())
            })
            .collect(),
    )
}

fn sparkle(cx: f32, cy: f32, r: f32) -> Sdf {
    polygon(
        crate::forge::sparkle_polygon(r, r * 0.2, 4, 48)
            .iter()
            .map(|p| (cx + p[0], cy - p[1]))
            .collect(),
    )
}

fn heart(cx: f32, cy: f32, width: f32) -> Sdf {
    polygon(
        crate::forge::heart_polygon(width, 64)
            .iter()
            .map(|p| (cx + p[0], cy - p[1]))
            .collect(),
    )
}

fn teardrop(x: f32, y: f32, r: f32) -> Sdf {
    circle(x, y, r).smooth_union(
        polygon(vec![(x - r * 0.92, y - r * 0.35), (x + r * 0.92, y - r * 0.35), (x, y - r * 2.5)]),
        r * 0.6,
    )
}

// ---------------------------------------------------------------------------------------------------------
// Painter
// ---------------------------------------------------------------------------------------------------------

/// Premultiplied RGBA canvas mapping design units to pixels (`px = origin + design * scale`).
struct Painter {
    w: usize,
    h: usize,
    px: Vec<Col>,
    s: f32,
    ox: f32,
    oy: f32,
}

fn over(dst: &mut Col, c: Col, cov: f32) {
    let a = c[3] * cov;
    if a <= 0.0 {
        return;
    }
    for k in 0..3 {
        dst[k] = c[k] * a + dst[k] * (1.0 - a);
    }
    dst[3] = a + dst[3] * (1.0 - a);
}

impl Painter {
    fn new(w: u32, h: u32) -> Painter {
        let s = w.min(h) as f32 / DESIGN;
        Painter {
            w: w as usize,
            h: h as usize,
            px: vec![[0.0; 4]; (w * h) as usize],
            s,
            ox: (w as f32 - DESIGN * s) * 0.5,
            oy: (h as f32 - DESIGN * s) * 0.5,
        }
    }

    /// Fills `sdf` with `fill(x, y)` and optionally an outline `(colour, width)` around it.
    fn draw(&mut self, sdf: &Sdf, fill: &dyn Fn(f32, f32) -> Col, line: Option<(Col, f32)>) {
        let lw = line.map_or(0.0, |l| l.1);
        let pad = lw + 2.0 / self.s;
        let bb = sdf.bb;
        let x0 = ((self.ox + (bb[0] - pad) * self.s).floor().max(0.0)) as usize;
        let y0 = ((self.oy + (bb[1] - pad) * self.s).floor().max(0.0)) as usize;
        let x1 = ((self.ox + (bb[2] + pad) * self.s).ceil().max(0.0) as usize).min(self.w);
        let y1 = ((self.oy + (bb[3] + pad) * self.s).ceil().max(0.0) as usize).min(self.h);
        for y in y0..y1 {
            for x in x0..x1 {
                let u = (x as f32 + 0.5 - self.ox) / self.s;
                let v = (y as f32 + 0.5 - self.oy) / self.s;
                let d = sdf.eval(u, v);
                let px = &mut self.px[y * self.w + x];
                if let Some((lc, w)) = line {
                    over(px, lc, (0.5 - (d - w) * self.s).clamp(0.0, 1.0));
                }
                let cov = (0.5 - d * self.s).clamp(0.0, 1.0);
                if cov > 0.0 {
                    over(px, fill(u, v), cov);
                }
            }
        }
    }

    fn fill(&mut self, sdf: &Sdf, c: Col) {
        self.draw(sdf, &|_, _| c, None);
    }

    fn outlined(&mut self, sdf: &Sdf, c: Col, line: Col, w: f32) {
        self.draw(sdf, &|_, _| c, Some((line, w)));
    }

    fn image(&self) -> RgbaImage {
        let mut img = RgbaImage::new(self.w as u32, self.h as u32);
        for (i, p) in self.px.iter().enumerate() {
            img.put_pixel((i % self.w) as u32, (i / self.w) as u32, straight(*p));
        }
        img
    }
}

fn straight(p: Col) -> Rgba<u8> {
    let a = p[3].clamp(0.0, 1.0);
    if a <= 1e-6 {
        return Rgba([0, 0, 0, 0]);
    }
    let q = |v: f32| ((v / a).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    Rgba([q(p[0]), q(p[1]), q(p[2]), (a * 255.0 + 0.5) as u8])
}

fn lerp(a: Col, b: Col, t: f32) -> Col {
    let t = t.clamp(0.0, 1.0);
    std::array::from_fn(|k| a[k] + (b[k] - a[k]) * t)
}

// ---------------------------------------------------------------------------------------------------------
// Built-in SD Zena
// ---------------------------------------------------------------------------------------------------------

const LINE: Col = [0.22, 0.13, 0.24, 1.0];
const HAIR: Col = [0.30, 0.22, 0.45, 1.0];
const HAIR_LIGHT: Col = [0.43, 0.33, 0.62, 1.0];
const HAIR_SHINE: Col = [0.66, 0.56, 0.86, 0.85];
const SKIN: Col = [1.0, 0.89, 0.82, 1.0];
const SKIN_SHADE: Col = [0.95, 0.77, 0.72, 1.0];
const EYE_DARK: Col = [0.18, 0.09, 0.32, 1.0];
const EYE_LIGHT: Col = [0.72, 0.56, 0.96, 1.0];
const WHITE: Col = [1.0, 1.0, 1.0, 1.0];
const OUTFIT_WHITE: Col = [0.97, 0.96, 1.0, 1.0];
const VIOLET: Col = [0.42, 0.25, 0.70, 1.0];
const VIOLET_DARK: Col = [0.30, 0.17, 0.52, 1.0];
const BLUSH: Col = [1.0, 0.5, 0.6, 0.45];
const MOUTH: Col = [0.55, 0.13, 0.22, 1.0];
const MOUTH_LINE: Col = [0.42, 0.13, 0.18, 1.0];
const PINK: Col = [1.0, 0.38, 0.6, 1.0];
const PINK_LINE: Col = [0.55, 0.1, 0.25, 1.0];
const WATER: Col = [0.58, 0.84, 1.0, 0.92];
const WATER_LINE: Col = [0.25, 0.45, 0.75, 1.0];
const GOLD: Col = [1.0, 0.9, 0.38, 1.0];
const GOLD_LINE: Col = [0.82, 0.52, 0.08, 1.0];

const EYES: [(f32, f32, f32); 2] = [(66.0, 69.0, -1.0), (94.0, 69.0, 1.0)];

fn hair_fill(x: f32, y: f32) -> Col {
    let _ = x;
    lerp(HAIR_LIGHT, HAIR, (y - 22.0) / 70.0)
}

fn draw_hair_back(p: &mut Painter) {
    let mass = circle(80.0, 58.0, 39.0).smooth_union(rbox(80.0, 77.0, 35.0, 17.0, 12.0), 6.0);
    p.draw(&mass, &hair_fill, Some((LINE, 1.4)));
    for (side, ang) in [(-1.0f32, 0.285f32), (1.0, -0.285)] {
        let tail = ellipse_rot(80.0 + side * 49.0, 80.0, 10.5, 27.0, ang)
            .smooth_union(capsule((80.0 + side * 52.0, 100.0), 8.0, (80.0 + side * 60.0, 119.0), 2.2), 5.0);
        p.draw(&tail, &hair_fill, Some((LINE, 1.4)));
        p.fill(
            &capsule((80.0 + side * 47.0, 70.0), 1.3, (80.0 + side * 53.0, 100.0), 0.8),
            HAIR_SHINE,
        );
        p.outlined(&circle(80.0 + side * 39.0, 52.0, 5.5), VIOLET, LINE, 1.2);
        p.fill(&circle(80.0 + side * 39.0 - 1.6, 50.4, 1.6), [1.0, 1.0, 1.0, 0.8]);
    }
}

fn draw_body(p: &mut Painter) {
    for x in [73.0f32, 87.0] {
        p.outlined(&rbox(x, 137.0, 3.6, 9.0, 3.0), SKIN, LINE, 1.1);
        p.outlined(&ellipse(x + (x - 80.0) * 0.15, 147.0, 6.5, 3.8), VIOLET_DARK, LINE, 1.1);
    }
    p.outlined(&rbox(80.0, 94.0, 4.5, 5.5, 2.0), SKIN_SHADE, LINE, 1.0);
    let skirt = polygon(vec![(65.0, 116.0), (95.0, 116.0), (106.0, 132.0), (54.0, 132.0)]).grow(2.0);
    p.outlined(&skirt.clone(), VIOLET, LINE, 1.3);
    p.fill(
        &skirt
            .inter(half_plane(0.0, 128.0, 0.0, -1.0))
            .inter(half_plane(0.0, 131.5, 0.0, 1.0)),
        OUTFIT_WHITE,
    );
    for (a, b) in [
        ((72.0, 119.0), (68.0, 131.0)),
        ((80.0, 119.0), (80.0, 131.0)),
        ((88.0, 119.0), (92.0, 131.0)),
    ] {
        p.fill(&capsule(a, 0.5, b, 0.5), VIOLET_DARK);
    }
    let torso = polygon(vec![(68.0, 99.0), (92.0, 99.0), (97.0, 116.0), (63.0, 116.0)]).grow(2.5);
    p.outlined(&torso, OUTFIT_WHITE, LINE, 1.3);
    let collar = polygon(vec![(67.0, 97.0), (93.0, 97.0), (80.0, 109.0)]).grow(1.0);
    p.outlined(&collar, VIOLET, LINE, 1.1);
    p.fill(
        &capsule((69.0, 98.5), 0.6, (80.0, 107.0), 0.6).union(capsule((91.0, 98.5), 0.6, (80.0, 107.0), 0.6)),
        OUTFIT_WHITE,
    );
    let bow = polygon(vec![(80.0, 107.0), (71.0, 102.5), (72.0, 111.5)]).union(polygon(vec![(80.0, 107.0), (89.0, 102.5), (88.0, 111.5)]));
    p.outlined(&bow, [0.95, 0.42, 0.62, 1.0], LINE, 1.0);
    p.outlined(&circle(80.0, 107.0, 2.3), [0.85, 0.3, 0.5, 1.0], LINE, 0.9);
}

fn draw_face(p: &mut Painter) {
    let face = ellipse(80.0, 62.0, 31.0, 28.0).smooth_union(ellipse(80.0, 75.0, 25.0, 18.0), 8.0);
    p.draw(&face, &|_, y| lerp(SKIN_SHADE, SKIN, (y - 38.0) / 18.0), Some((LINE, 1.4)));
    for x in [60.0f32, 100.0] {
        p.fill(&ellipse(x, 78.0, 6.5, 3.2), BLUSH);
        for k in 0..3 {
            let dx = x - 3.0 + 3.0 * k as f32;
            p.fill(&capsule((dx + 0.8, 76.6), 0.35, (dx - 0.8, 79.2), 0.35), [0.95, 0.4, 0.5, 0.6]);
        }
    }
}

/// The normal anime eye; `clip` (if any) cuts it (angry lids).
fn eye_normal(p: &mut Painter, cx: f32, cy: f32, side: f32, clip: Option<Sdf>) {
    let c = |s: Sdf| match &clip {
        Some(k) => s.inter(k.clone()),
        None => s,
    };
    p.outlined(&c(ellipse(cx, cy, 8.0, 10.5)), WHITE, LINE, 0.9);
    p.draw(
        &c(ellipse(cx, cy + 0.8, 6.6, 9.3)),
        &|_, y| lerp(EYE_DARK, EYE_LIGHT, (y - (cy - 8.0)) / 16.0),
        None,
    );
    p.fill(
        &c(ellipse(cx, cy + 0.8, 6.6, 9.3).minus(ellipse(cx, cy + 0.8, 5.2, 7.9))),
        [0.2, 0.1, 0.36, 0.55],
    );
    p.fill(&c(ellipse(cx, cy + 1.6, 3.2, 4.6)), [0.12, 0.06, 0.22, 1.0]);
    p.fill(&c(circle(cx - 2.4, cy - 3.6, 2.6)), WHITE);
    p.fill(&c(circle(cx + 2.6, cy + 4.2, 1.2)), [1.0, 1.0, 1.0, 0.9]);
    if clip.is_none() {
        p.fill(&polyline(arc_pts(cx, cy + 3.0, 11.0, 203.0, 337.0, 16), 1.5), LINE);
        let outer = cx + side * 9.6;
        p.fill(&capsule((outer, cy - 5.0), 1.2, (outer + side * 3.2, cy - 8.0), 0.5), LINE);
    }
}

fn draw_eyes(p: &mut Painter, variant: &str) {
    for (cx, cy, side) in EYES {
        match variant {
            "happy" => p.fill(&polyline(arc_pts(cx, cy + 4.0, 7.5, 200.0, 340.0, 14), 1.6), LINE),
            "star" => {
                p.outlined(&star(cx, cy + 1.0, 10.5, 4.6), [1.0, 0.85, 0.3, 1.0], [0.6, 0.35, 0.1, 1.0], 1.0);
                p.fill(&circle(cx - 2.0, cy - 1.0, 1.7), WHITE);
            }
            "heart" => {
                p.outlined(&heart(cx, cy + 1.0, 19.0), [1.0, 0.32, 0.52, 1.0], PINK_LINE, 1.0);
                p.fill(&circle(cx - 4.0, cy - 2.5, 1.9), WHITE);
            }
            "spiral" => {
                p.outlined(&ellipse(cx, cy, 8.5, 10.0), WHITE, LINE, 0.9);
                let turns = 2.6;
                let pts: Vec<(f32, f32)> = (0..=80)
                    .map(|i| {
                        let t = i as f32 / 80.0;
                        let a = side * t * turns * TAU;
                        let r = 0.6 + 7.2 * t;
                        (cx + r * a.cos(), cy + r * 1.15 * a.sin())
                    })
                    .collect();
                p.fill(&polyline(pts, 0.85), LINE);
            }
            "teary" => {
                eye_normal(p, cx, cy, side, None);
                p.fill(
                    &ellipse(cx, cy, 8.0, 10.5).inter(half_plane(0.0, cy + 2.5, 0.0, -1.0)),
                    [0.62, 0.86, 1.0, 0.7],
                );
                p.fill(&circle(cx + 2.5, cy - 4.5, 1.4), WHITE);
                p.fill(&polyline(arc_pts(cx, cy + 1.0, 10.0, 40.0, 140.0, 10), 0.7), WATER_LINE);
            }
            "squeeze" => {
                let (o, i) = (cx + side * 5.5, cx - side * 5.0);
                p.fill(&polyline(vec![(o, cy - 6.0), (i, cy), (o, cy + 6.0)], 1.7), LINE);
            }
            "angry" => {
                // Lid slanting down toward the nose; the eye is cut below it.
                let (outer, inner) = ((cx + side * 10.0, cy - 10.0), (cx - side * 9.0, cy - 3.0));
                let (dx, dy) = (inner.0 - outer.0, inner.1 - outer.1);
                // Normal pointing up, away from the kept (lower) side.
                let n = if dx > 0.0 { (dy, -dx) } else { (-dy, dx) };
                let lid = half_plane(outer.0, outer.1, n.0, n.1);
                eye_normal(p, cx, cy, side, Some(lid));
                p.fill(&capsule(outer, 1.6, inner, 1.3), LINE);
            }
            _ => eye_normal(p, cx, cy, side, None),
        }
    }
}

fn draw_mouth(p: &mut Painter, variant: &str) {
    match variant {
        "open" => {
            let m = ellipse(80.0, 82.5, 6.5, 6.5).inter(half_plane(0.0, 81.5, 0.0, -1.0));
            p.outlined(&m, MOUTH, LINE, 1.0);
            p.fill(&ellipse(80.0, 87.3, 4.0, 2.4).inter(m), [1.0, 0.55, 0.6, 1.0]);
        }
        "wavy" => {
            let pts: Vec<(f32, f32)> = (0..=24)
                .map(|i| {
                    let x = 72.0 + 16.0 * i as f32 / 24.0;
                    (x, 84.0 + 1.4 * ((x - 72.0) / 16.0 * 3.0 * PI).sin())
                })
                .collect();
            p.fill(&polyline(pts, 0.9), MOUTH_LINE);
        }
        "frown" => p.fill(&polyline(arc_pts(80.0, 89.5, 5.5, 210.0, 330.0, 12), 1.0), MOUTH_LINE),
        "o" => p.outlined(&ellipse(80.0, 84.0, 3.0, 3.6), MOUTH, LINE, 0.9),
        _ => p.fill(&polyline(arc_pts(80.0, 79.5, 5.5, 25.0, 155.0, 12), 1.0), MOUTH_LINE),
    }
}

fn arm(p: &mut Painter, shoulder: (f32, f32), hand: (f32, f32), hand_r: f32) {
    p.outlined(&capsule(shoulder, 5.0, hand, 4.4), OUTFIT_WHITE, LINE, 1.2);
    p.outlined(
        &circle(hand.0 + (hand.0 - shoulder.0) * 0.12, hand.1 + (hand.1 - shoulder.1) * 0.12, hand_r),
        SKIN,
        LINE,
        1.1,
    );
}

fn draw_hands(p: &mut Painter, variant: &str) {
    match variant {
        "none" => {}
        "up" => {
            arm(p, (65.0, 100.0), (53.0, 83.0), 4.8);
            arm(p, (95.0, 100.0), (107.0, 83.0), 4.8);
        }
        "heart" => {
            arm(p, (65.0, 100.0), (69.0, 112.0), 4.3);
            arm(p, (95.0, 100.0), (91.0, 112.0), 4.3);
            p.outlined(&heart(80.0, 113.0, 26.0), PINK, PINK_LINE, 1.2);
            p.fill(&ellipse(73.5, 108.0, 2.6, 1.6), [1.0, 1.0, 1.0, 0.8]);
            for x in [67.5f32, 92.5] {
                p.outlined(&circle(x, 114.0, 4.2), SKIN, LINE, 1.1);
            }
        }
        "point" => {
            arm(p, (65.0, 100.0), (59.0, 114.0), 4.3);
            p.outlined(&capsule((95.0, 100.0), 5.0, (108.0, 87.0), 4.4), OUTFIT_WHITE, LINE, 1.2);
            let finger = capsule((112.0, 82.0), 1.8, (114.5, 72.0), 1.6);
            p.outlined(&finger.union(circle(111.0, 84.0, 4.5)), SKIN, LINE, 1.1);
        }
        _ => {
            arm(p, (65.0, 100.0), (59.0, 114.0), 4.3);
            arm(p, (95.0, 100.0), (101.0, 114.0), 4.3);
        }
    }
}

fn draw_hair_front(p: &mut Painter) {
    let cap = ellipse(80.0, 55.0, 35.0, 31.0).inter(half_plane(0.0, 45.0, 0.0, 1.0));
    let mut bangs = cap;
    for (tx, ty, hw) in [
        (56.0f32, 60.0f32, 8.0f32),
        (67.0, 57.0, 8.5),
        (80.0, 61.0, 9.0),
        (93.0, 57.0, 8.5),
        (104.0, 60.0, 8.0),
    ] {
        let lock = polygon(vec![(tx - hw, 38.0), (tx + hw, 38.0), (tx + (tx - 80.0) * 0.08, ty)]).grow(0.8);
        bangs = bangs.smooth_union(lock, 3.0);
    }
    let bangs = bangs.inter(circle(80.0, 58.0, 38.6));
    p.draw(&bangs, &hair_fill, Some((LINE, 1.4)));
    for side in [-1.0f32, 1.0] {
        let lock = capsule((80.0 + side * 30.0, 44.0), 6.0, (80.0 + side * 35.0, 93.0), 1.8);
        p.draw(&lock, &hair_fill, Some((LINE, 1.3)));
    }
    p.fill(&polyline(arc_pts(80.0, 60.0, 27.0, 218.0, 322.0, 18), 1.6), HAIR_SHINE);
    let ahoge = capsule((80.0, 23.0), 2.4, (83.0, 15.0), 1.8)
        .smooth_union(capsule((83.0, 15.0), 1.8, (89.0, 10.0), 1.2), 1.5)
        .smooth_union(capsule((89.0, 10.0), 1.2, (94.0, 11.5), 0.7), 1.0);
    p.draw(&ahoge, &hair_fill, Some((LINE, 1.2)));
}

fn note(p: &mut Painter, x: f32, y: f32, s: f32) {
    let head = ellipse_rot(x - 4.0 * s, y + 8.0 * s, 4.6 * s, 3.4 * s, -0.4);
    let stem = capsule((x, y + 7.0 * s), 1.0 * s, (x, y - 12.0 * s), 1.0 * s);
    let flag = polyline(
        vec![
            (x, y - 12.0 * s),
            (x + 5.0 * s, y - 9.0 * s),
            (x + 7.0 * s, y - 4.0 * s),
            (x + 4.0 * s, y),
        ],
        1.2 * s,
    );
    p.outlined(&head.union(stem).union(flag), [0.35, 0.3, 0.8, 1.0], [1.0, 1.0, 1.0, 0.95], 1.0);
}

fn draw_fx(p: &mut Painter, variant: &str) {
    match variant {
        "vein" => {
            let (cx, cy) = (122.0f32, 30.0f32);
            for (sx, sy) in [(1.0f32, 1.0f32), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
                let ax: f32 = if sx > 0.0 { 180.0 } else { 0.0 };
                let ay: f32 = if sy > 0.0 { 270.0 } else { 90.0 };
                let delta = (ay - ax + 540.0).rem_euclid(360.0) - 180.0;
                let dir = delta.signum();
                let pts = arc_pts(cx + sx * 8.0, cy + sy * 8.0, 6.5, ax - dir * 14.0, ax + delta + dir * 14.0, 14);
                p.outlined(&polyline(pts, 1.8), [0.92, 0.12, 0.18, 1.0], [0.45, 0.04, 0.08, 1.0], 0.9);
            }
        }
        "sweat" => {
            for (x, y, r) in [(118.0f32, 44.0f32, 4.6f32), (129.0, 60.0, 3.6), (110.0, 30.0, 2.8)] {
                p.outlined(&teardrop(x, y, r), WATER, WATER_LINE, 0.9);
                p.fill(&circle(x - r * 0.35, y - r * 0.1, r * 0.3), WHITE);
            }
        }
        "sparkle" => {
            for (x, y, r) in [
                (124.0f32, 24.0f32, 10.0f32),
                (35.0, 30.0, 6.5),
                (134.0, 52.0, 5.0),
                (27.0, 58.0, 4.0),
            ] {
                p.outlined(&sparkle(x, y, r), GOLD, GOLD_LINE, 0.8);
            }
        }
        "exclaim" => {
            let mark = capsule((124.0, 13.0), 4.2, (124.0, 30.0), 2.3).union(circle(124.0, 38.5, 3.1));
            p.outlined(&mark, [0.95, 0.2, 0.15, 1.0], [0.45, 0.05, 0.05, 1.0], 1.0);
        }
        "question" => {
            let mut pts = arc_pts(124.0, 21.0, 6.5, 200.0, 420.0, 24);
            pts.extend([(124.0, 29.5), (124.0, 32.0)]);
            let mark = polyline(pts, 2.0).union(circle(124.0, 39.5, 2.6));
            p.outlined(&mark, [0.45, 0.38, 0.95, 1.0], [0.2, 0.15, 0.5, 1.0], 0.9);
        }
        "tears" => {
            for side in [-1.0f32, 1.0] {
                let stream = capsule((80.0 + side * 20.0, 77.0), 1.8, (80.0 + side * 25.0, 104.0), 3.2).union(circle(
                    80.0 + side * 26.0,
                    112.0,
                    2.4,
                ));
                p.outlined(&stream, WATER, WATER_LINE, 0.7);
            }
        }
        "notes" => {
            note(p, 124.0, 30.0, 1.0);
            note(p, 38.0, 38.0, 0.72);
        }
        "puff" => {
            let cloud = circle(110.0, 88.0, 5.0)
                .smooth_union(circle(116.0, 84.0, 6.0), 3.0)
                .smooth_union(circle(123.0, 87.0, 5.0), 3.0)
                .smooth_union(circle(117.0, 91.0, 4.5), 3.0);
            p.outlined(&cloud, [0.97, 0.98, 1.0, 0.95], [0.6, 0.65, 0.8, 1.0], 0.9);
            p.outlined(
                &circle(129.0, 80.0, 2.0).union(circle(133.0, 76.0, 1.3)),
                [0.97, 0.98, 1.0, 0.95],
                [0.6, 0.65, 0.8, 1.0],
                0.7,
            );
        }
        "bulb" => {
            p.fill(&circle(124.0, 22.0, 13.0), [1.0, 0.95, 0.5, 0.3]);
            for k in 0..5 {
                let a = (200.0 + 35.0 * k as f32).to_radians();
                p.fill(
                    &capsule(
                        (124.0 + 12.0 * a.cos(), 21.0 + 12.0 * a.sin()),
                        0.9,
                        (124.0 + 16.0 * a.cos(), 21.0 + 16.0 * a.sin()),
                        0.7,
                    ),
                    GOLD_LINE,
                );
            }
            p.outlined(&circle(124.0, 21.0, 8.0), [1.0, 0.9, 0.4, 1.0], [0.6, 0.45, 0.1, 1.0], 1.0);
            p.fill(
                &polyline(vec![(121.5, 24.0), (123.0, 19.5), (124.5, 24.0), (126.0, 19.5)], 0.5),
                [0.85, 0.55, 0.1, 1.0],
            );
            p.fill(&circle(121.0, 18.0, 1.8), [1.0, 1.0, 1.0, 0.85]);
            p.outlined(&rbox(124.0, 31.5, 4.2, 3.5, 1.2), [0.68, 0.68, 0.75, 1.0], LINE, 0.9);
            p.fill(&capsule((120.5, 30.5), 0.4, (127.5, 30.5), 0.4), [0.4, 0.4, 0.48, 1.0]);
        }
        "hearts" => {
            for (x, y, w) in [(122.0f32, 28.0f32, 15.0f32), (34.0, 36.0, 10.0), (132.0, 52.0, 8.0)] {
                p.outlined(&heart(x, y, w), PINK, PINK_LINE, 0.9);
            }
        }
        "arrow" => {
            let a = polygon(vec![
                (104.0, 25.0),
                (122.0, 25.0),
                (122.0, 19.0),
                (137.0, 28.0),
                (122.0, 37.0),
                (122.0, 31.0),
                (104.0, 31.0),
            ]);
            p.outlined(&a, [1.0, 0.42, 0.12, 1.0], [0.55, 0.15, 0.05, 1.0], 1.2);
        }
        "dizzy" => {
            let ring = ellipse(80.0, 20.0, 30.0, 7.0);
            let ring = Sdf::new([48.0, 11.0, 112.0, 29.0], move |x, y| ring.eval(x, y).abs() - 0.6);
            p.fill(&ring, [0.5, 0.45, 0.75, 0.6]);
            for (x, y) in [(52.0f32, 21.0f32), (106.0, 18.0), (80.0, 27.0)] {
                p.outlined(&star(x, y, 4.8, 2.1), [1.0, 0.88, 0.3, 1.0], [0.7, 0.45, 0.1, 1.0], 0.7);
            }
        }
        _ => {}
    }
}

/// Variant names the built-in kit draws per slot.
fn builtin_variants(slot: &str) -> &'static [&'static str] {
    match slot {
        "eyes" => &["normal", "happy", "star", "heart", "spiral", "teary", "squeeze", "angry"],
        "mouth" => &["smile", "open", "wavy", "frown", "o"],
        "hands" => &["down", "up", "heart", "point", "none"],
        "fx" => &[
            "none", "vein", "sweat", "sparkle", "exclaim", "question", "tears", "notes", "puff", "bulb", "hearts", "arrow", "dizzy",
        ],
        "hair_back" | "body" | "face" | "hair_front" => &["default"],
        _ => &[],
    }
}

fn draw_builtin(slot: &str, variant: &str, w: u32, h: u32) -> RgbaImage {
    let mut p = Painter::new(w, h);
    match slot {
        "hair_back" => draw_hair_back(&mut p),
        "body" => draw_body(&mut p),
        "face" => draw_face(&mut p),
        "eyes" => draw_eyes(&mut p, variant),
        "mouth" => draw_mouth(&mut p, variant),
        "hair_front" => draw_hair_front(&mut p),
        "hands" => draw_hands(&mut p, variant),
        "fx" => draw_fx(&mut p, variant),
        _ => {}
    }
    p.image()
}

/// Built-in `(eyes, mouth, hands, fx)` for an emote (`None` = neutral).
fn builtin_choice(emote: Option<CharacterEmote>) -> [&'static str; 4] {
    use CharacterEmote as E;
    match emote {
        None => ["normal", "smile", "down", "none"],
        Some(E::ChibiCheer) => ["happy", "open", "up", "sparkle"],
        Some(E::ChibiHeart) => ["happy", "smile", "heart", "hearts"],
        Some(E::ComicAnger3d) => ["angry", "frown", "down", "vein"],
        Some(E::SweatPanic) => ["squeeze", "wavy", "down", "sweat"],
        Some(E::SparkleIdea) => ["normal", "open", "point", "bulb"],
        Some(E::Exclaim3d) => ["normal", "o", "down", "exclaim"],
        Some(E::Question) => ["normal", "wavy", "down", "question"],
        Some(E::ThreatArrow3d) => ["normal", "open", "point", "arrow"],
        Some(E::Tears) => ["teary", "frown", "down", "tears"],
        Some(E::StarEyes) => ["star", "open", "up", "sparkle"],
        Some(E::HeartEyes) => ["heart", "smile", "down", "hearts"],
        Some(E::DizzySpiral) => ["spiral", "wavy", "down", "dizzy"],
        Some(E::ReliefSigh) => ["happy", "o", "down", "puff"],
        Some(E::MusicNote) => ["happy", "smile", "down", "notes"],
    }
}

// ---------------------------------------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------------------------------------

/// 2D affine `x' = a·x + c·y + e`, `y' = b·x + d·y + f` (pixel space).
#[derive(Debug, Clone, Copy)]
struct Affine([f32; 6]);

impl Affine {
    const ID: Affine = Affine([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    fn apply(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.0;
        (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
    }
    /// `self` first, then `o`.
    fn then(self, o: Affine) -> Affine {
        let (a, b) = (self.0, o.0);
        Affine([
            b[0] * a[0] + b[2] * a[1],
            b[1] * a[0] + b[3] * a[1],
            b[0] * a[2] + b[2] * a[3],
            b[1] * a[2] + b[3] * a[3],
            b[0] * a[4] + b[2] * a[5] + b[4],
            b[1] * a[4] + b[3] * a[5] + b[5],
        ])
    }
    fn translate(x: f32, y: f32) -> Affine {
        Affine([1.0, 0.0, 0.0, 1.0, x, y])
    }
    fn scale_about(px: f32, py: f32, sx: f32, sy: f32) -> Affine {
        Affine([sx, 0.0, 0.0, sy, px - sx * px, py - sy * py])
    }
    fn rotate_about(px: f32, py: f32, a: f32) -> Affine {
        let (s, c) = a.sin_cos();
        Affine::translate(-px, -py)
            .then(Affine([c, s, -s, c, 0.0, 0.0]))
            .then(Affine::translate(px, py))
    }
    fn invert(&self) -> Affine {
        let m = &self.0;
        let det = m[0] * m[3] - m[1] * m[2];
        if det.abs() < 1e-12 {
            return Affine::ID;
        }
        let (a, b, c, d) = (m[3] / det, -m[1] / det, -m[2] / det, m[0] / det);
        Affine([a, b, c, d, -(a * m[4] + c * m[5]), -(b * m[4] + d * m[5])])
    }
}

/// One layer image placed at `offset` on the kit canvas.
#[derive(Clone)]
struct Layer {
    img: RgbaImage,
    offset: (i32, i32),
    /// Opaque bounds in canvas pixels `[x0, y0, x1, y1]` (empty layers have x0 >= x1).
    bbox: [f32; 4],
    /// Alpha-weighted centre in canvas pixels.
    centroid: (f32, f32),
}

impl Layer {
    fn new(img: RgbaImage, offset: (i32, i32)) -> Layer {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        let (mut sx, mut sy, mut sa) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y, p) in img.enumerate_pixels() {
            if p[3] > 0 {
                let (fx, fy) = ((x as i32 + offset.0) as f32, (y as i32 + offset.1) as f32);
                x0 = x0.min(fx);
                y0 = y0.min(fy);
                x1 = x1.max(fx + 1.0);
                y1 = y1.max(fy + 1.0);
                let a = p[3] as f64;
                sx += (fx as f64 + 0.5) * a;
                sy += (fy as f64 + 0.5) * a;
                sa += a;
            }
        }
        let centroid = if sa > 0.0 {
            ((sx / sa) as f32, (sy / sa) as f32)
        } else {
            (0.0, 0.0)
        };
        Layer {
            img,
            offset,
            bbox: [x0, y0, x1, y1],
            centroid,
        }
    }

    fn is_empty(&self) -> bool {
        self.bbox[0] >= self.bbox[2]
    }

    fn texel(&self, x: i64, y: i64) -> Col {
        if x < 0 || y < 0 || x >= self.img.width() as i64 || y >= self.img.height() as i64 {
            return [0.0; 4];
        }
        let p = self.img.get_pixel(x as u32, y as u32);
        let a = p[3] as f32 / 255.0;
        [p[0] as f32 / 255.0 * a, p[1] as f32 / 255.0 * a, p[2] as f32 / 255.0 * a, a]
    }

    /// Bilinear premultiplied sample at layer-pixel coordinates (texel centres at +0.5).
    fn sample(&self, u: f32, v: f32) -> Col {
        let (x, y) = (u - 0.5, v - 0.5);
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        let t = [
            self.texel(xi, yi),
            self.texel(xi + 1, yi),
            self.texel(xi, yi + 1),
            self.texel(xi + 1, yi + 1),
        ];
        std::array::from_fn(|k| (t[0][k] * (1.0 - fx) + t[1][k] * fx) * (1.0 - fy) + (t[2][k] * (1.0 - fx) + t[3][k] * fx) * fy)
    }
}

// ---------------------------------------------------------------------------------------------------------
// Kit
// ---------------------------------------------------------------------------------------------------------

/// A layered SD stamp kit (built-in or loaded from `kit.json`).
pub struct ChibiKit {
    size: (u32, u32),
    slots: Vec<String>,
    layers: HashMap<String, HashMap<String, Layer>>,
    /// Per-emote slot → variant overrides from `kit.json` (`None` key = `NEUTRAL`).
    emotes: HashMap<Option<CharacterEmote>, HashMap<String, String>>,
    /// Design-unit → pixel scale and origin (the built-in layout and animation pivots).
    scale: f32,
    origin: (f32, f32),
}

impl ChibiKit {
    /// The procedural SD Zena kit at 160×160.
    pub fn builtin() -> ChibiKit {
        ChibiKit::builtin_sized(DEFAULT_SIZE.0, DEFAULT_SIZE.1)
    }

    /// The procedural kit drawn at any canvas size (centred, scaled to the shorter side).
    pub fn builtin_sized(width: u32, height: u32) -> ChibiKit {
        let (w, h) = (width.max(8), height.max(8));
        let mut layers: HashMap<String, HashMap<String, Layer>> = HashMap::new();
        for slot in DEFAULT_SLOTS {
            let variants = layers.entry(slot.to_string()).or_default();
            for v in builtin_variants(slot) {
                variants.insert(v.to_string(), Layer::new(draw_builtin(slot, v, w, h), (0, 0)));
            }
        }
        let s = w.min(h) as f32 / DESIGN;
        ChibiKit {
            size: (w, h),
            slots: DEFAULT_SLOTS.iter().map(|s| s.to_string()).collect(),
            layers,
            emotes: HashMap::new(),
            scale: s,
            origin: ((w as f32 - DESIGN * s) * 0.5, (h as f32 - DESIGN * s) * 0.5),
        }
    }

    /// Loads `<dir>/kit.json` (see the module docs); missing slots / variants / emotes use the built-in ones.
    pub fn load(dir: &Path) -> Result<ChibiKit> {
        let path = dir.join("kit.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let v: Value = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let dim = |i: usize| {
            v.get("size")
                .and_then(|s| s.get(i))
                .and_then(Value::as_u64)
                .map(|x| x.clamp(8, 4096) as u32)
        };
        let mut kit = ChibiKit::builtin_sized(dim(0).unwrap_or(DEFAULT_SIZE.0), dim(1).unwrap_or(DEFAULT_SIZE.1));
        if let Some(slots) = v.get("slots").and_then(Value::as_array) {
            let s: Vec<String> = slots.iter().filter_map(Value::as_str).map(str::to_string).collect();
            if !s.is_empty() {
                kit.slots = s;
            }
        }
        if let Some(layers) = v.get("layers").and_then(Value::as_object) {
            for (slot, variants) in layers {
                for (variant, spec) in variants.as_object().into_iter().flatten() {
                    let png = spec
                        .get("png")
                        .and_then(Value::as_str)
                        .with_context(|| format!("layer {slot}/{variant} has no png"))?;
                    let img = image::open(dir.join(png)).with_context(|| format!("loading {png}"))?.to_rgba8();
                    let off = |i: usize| spec.get("offset").and_then(|o| o.get(i)).and_then(Value::as_f64).unwrap_or(0.0) as i32;
                    kit.layers
                        .entry(slot.clone())
                        .or_default()
                        .insert(variant.clone(), Layer::new(img, (off(0), off(1))));
                }
            }
        }
        if let Some(emotes) = v.get("emotes").and_then(Value::as_object) {
            for (name, map) in emotes {
                let key = if name.eq_ignore_ascii_case("NEUTRAL") {
                    None
                } else if let Some(e) = CharacterEmote::parse(name) {
                    Some(e)
                } else {
                    eprintln!("[chibi] {}: unknown emote {name}", path.display());
                    continue;
                };
                let m: HashMap<String, String> = map
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter_map(|(k, x)| x.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect();
                kit.emotes.insert(key, m);
            }
        }
        Ok(kit)
    }

    /// Canvas (frame) size.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Draw order.
    pub fn slots(&self) -> &[String] {
        &self.slots
    }

    /// Variant names available for a slot (sorted).
    pub fn variants(&self, slot: &str) -> Vec<String> {
        let mut v: Vec<String> = self.layers.get(slot).map(|m| m.keys().cloned().collect()).unwrap_or_default();
        v.sort();
        v
    }

    /// `(slot, variant)` actually drawn for an emote (`None` = neutral), in draw order.
    pub fn choice(&self, emote: Option<CharacterEmote>) -> Vec<(String, String)> {
        let [eyes, mouth, hands, fx] = builtin_choice(emote);
        let custom = self.emotes.get(&emote);
        let has = |slot: &str, v: &str| self.layers.get(slot).is_some_and(|m| m.contains_key(v));
        let mut out = Vec::new();
        for slot in &self.slots {
            let builtin = match slot.as_str() {
                "eyes" => eyes,
                "mouth" => mouth,
                "hands" => hands,
                "fx" => fx,
                _ => "default",
            };
            let picked = custom
                .and_then(|c| c.get(slot))
                .filter(|v| has(slot, v.as_str()))
                .cloned()
                .or_else(|| has(slot, builtin).then(|| builtin.to_string()))
                .or_else(|| has(slot, "default").then(|| "default".to_string()));
            if let Some(v) = picked {
                out.push((slot.clone(), v));
            }
        }
        out
    }

    fn px(&self, x: f32, y: f32) -> (f32, f32) {
        (self.origin.0 + x * self.scale, self.origin.1 + y * self.scale)
    }

    /// Whole-character motion for an emote at phase `ph` (radians).
    fn body_motion(&self, emote: Option<CharacterEmote>, ph: f32) -> Affine {
        use CharacterEmote as E;
        let s = self.scale;
        let feet = self.px(80.0, 152.0);
        let up = |dy: f32| Affine::translate(0.0, -dy * s);
        let rot = |a: f32| Affine::rotate_about(feet.0, feet.1, a);
        match emote {
            None => Affine::ID,
            Some(E::ChibiCheer) => {
                let hgt = ph.sin().abs();
                Affine::scale_about(feet.0, feet.1, 1.04 - 0.05 * hgt, 0.94 + 0.08 * hgt).then(up(6.0 * hgt))
            }
            Some(E::ChibiHeart | E::HeartEyes) => up(2.0 * ph.sin()),
            Some(E::ComicAnger3d) => Affine::translate(1.5 * s * (5.0 * ph).sin(), 0.0),
            Some(E::SweatPanic) => Affine::translate(1.2 * s * (7.0 * ph).sin(), 0.0),
            Some(E::SparkleIdea) => up(3.0 * ph.sin().abs()),
            Some(E::Exclaim3d) => up(5.0 * ph.sin().max(0.0)),
            Some(E::Question) => rot(0.05 * ph.sin()),
            Some(E::ThreatArrow3d) => rot(-0.03 * ph.sin()),
            Some(E::Tears) => up(1.2 * ph.sin().abs()),
            Some(E::StarEyes) => up(5.0 * ph.sin().abs()),
            Some(E::DizzySpiral) => rot(0.08 * ph.sin()),
            Some(E::ReliefSigh) => Affine::scale_about(feet.0, feet.1, 1.0, 1.0 + 0.03 * ph.sin()),
            Some(E::MusicNote) => rot(0.06 * ph.sin()),
        }
    }

    /// Extra per-slot motion (before the body motion).
    fn slot_motion(&self, emote: Option<CharacterEmote>, slot: &str, layer: &Layer, ph: f32) -> Affine {
        use CharacterEmote as E;
        let s = self.scale;
        let (cx, cy) = layer.centroid;
        let tr = |x: f32, y: f32| Affine::translate(x * s, y * s);
        let pulse = |k: f32| Affine::scale_about(cx, cy, k, k);
        let Some(e) = emote else { return Affine::ID };
        match (slot, e) {
            ("hands", E::ChibiCheer) => {
                let (px, py) = self.px(80.0, 100.0);
                Affine::rotate_about(px, py, 0.1 * (2.0 * ph).sin())
            }
            ("hands", E::ChibiHeart) => {
                let (px, py) = self.px(80.0, 113.0);
                let k = 1.0 + 0.07 * (2.0 * ph).sin();
                Affine::scale_about(px, py, k, k)
            }
            ("eyes", E::StarEyes | E::HeartEyes) => {
                let (px, py) = self.px(80.0, 69.0);
                let k = 1.0 + 0.06 * (2.0 * ph).sin();
                Affine::scale_about(px, py, k, k)
            }
            ("fx", E::ChibiCheer | E::StarEyes) => pulse(1.0 + 0.15 * (2.0 * ph).sin()),
            ("fx", E::ChibiHeart | E::HeartEyes) => tr(0.0, -4.0 * ph.sin()),
            ("fx", E::ComicAnger3d) => pulse(1.0 + 0.18 * ph.sin().abs()),
            ("fx", E::SweatPanic) => tr(0.0, 3.0 * ph.sin()),
            ("fx", E::SparkleIdea) => pulse(1.0 + 0.08 * (2.0 * ph).sin()).then(tr(0.0, -3.0 * ph.sin())),
            ("fx", E::Exclaim3d) => pulse(1.0 + 0.25 * ph.sin().max(0.0)),
            ("fx", E::Question) => Affine::rotate_about(cx, cy, 0.12 * ph.sin()).then(tr(0.0, -3.0 * ph.sin())),
            ("fx", E::ThreatArrow3d) => tr(4.0 * ph.sin(), 0.0),
            ("fx", E::Tears) => {
                let (px, py) = self.px(80.0, 77.0);
                Affine::scale_about(px, py, 1.0, 1.0 + 0.12 * ph.sin()).then(tr(0.0, 2.0 * ph.sin()))
            }
            ("fx", E::DizzySpiral) => tr(4.0 * ph.sin(), 0.0),
            ("fx", E::ReliefSigh) => pulse(1.0 + 0.1 * ph.sin()).then(tr(3.0 * ph.sin(), -2.0 * ph.sin())),
            ("fx", E::MusicNote) => tr(1.5 * ph.cos(), -4.0 * ph.sin()),
            _ => Affine::ID,
        }
    }

    fn render(&self, emote: Option<CharacterEmote>, frame: usize, frames: usize) -> RgbaImage {
        let frames = frames.max(1);
        let ph = TAU * (frame % frames) as f32 / frames as f32;
        let (w, h) = (self.size.0 as usize, self.size.1 as usize);
        let mut canvas: Vec<Col> = vec![[0.0; 4]; w * h];
        let body = self.body_motion(emote, ph);
        for (slot, variant) in self.choice(emote) {
            let Some(layer) = self.layers.get(&slot).and_then(|m| m.get(&variant)) else {
                continue;
            };
            if layer.is_empty() {
                continue;
            }
            let t = self.slot_motion(emote, &slot, layer, ph).then(body);
            let inv = t.invert();
            let b = layer.bbox;
            let corners = [t.apply(b[0], b[1]), t.apply(b[2], b[1]), t.apply(b[0], b[3]), t.apply(b[2], b[3])];
            let x0 = corners.iter().map(|c| c.0).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
            let y0 = corners.iter().map(|c| c.1).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
            let x1 = (corners.iter().map(|c| c.0).fold(f32::MIN, f32::max).ceil().max(0.0) as usize + 1).min(w);
            let y1 = (corners.iter().map(|c| c.1).fold(f32::MIN, f32::max).ceil().max(0.0) as usize + 1).min(h);
            for y in y0..y1 {
                for x in x0..x1 {
                    let (u, v) = inv.apply(x as f32 + 0.5, y as f32 + 0.5);
                    let src = layer.sample(u - layer.offset.0 as f32, v - layer.offset.1 as f32);
                    if src[3] <= 0.0 {
                        continue;
                    }
                    let dst = &mut canvas[y * w + x];
                    for k in 0..4 {
                        dst[k] = src[k] + dst[k] * (1.0 - src[3]);
                    }
                }
            }
        }
        let mut img = RgbaImage::new(w as u32, h as u32);
        for (i, p) in canvas.iter().enumerate() {
            img.put_pixel((i % w) as u32, (i / w) as u32, straight(*p));
        }
        img
    }

    /// One animation frame of an emote stamp (`frame` of `frames`, looping).
    pub fn compose(&self, emote: CharacterEmote, frame: usize, frames: usize) -> RgbaImage {
        self.render(Some(emote), frame, frames)
    }

    /// The neutral, still SD Zena (Track B-1 body).
    pub fn neutral(&self) -> RgbaImage {
        self.render(None, 0, 1)
    }

    /// `frames` frames side by side (horizontal strip, `frames × width` wide).
    pub fn atlas(&self, emote: CharacterEmote, frames: usize) -> RgbaImage {
        let frames = frames.max(1);
        let (w, h) = self.size;
        let mut out = RgbaImage::new(w * frames as u32, h);
        for f in 0..frames {
            let img = self.compose(emote, f, frames);
            image::imageops::replace(&mut out, &img, (f as u32 * w) as i64, 0);
        }
        out
    }

    /// Writes `<dir>/chibi_<EMOTE>.png` (the [`ChibiKit::atlas`]) and returns its path.
    pub fn write_atlas(&self, emote: CharacterEmote, frames: usize, dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("chibi_{}.png", emote.name()));
        self.atlas(emote, frames)
            .save(&path)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(img: &RgbaImage) -> f32 {
        img.pixels().filter(|p| p[3] > 16).count() as f32 / (img.width() * img.height()) as f32
    }

    #[test]
    fn every_emote_composes_a_reasonable_stamp() {
        let kit = ChibiKit::builtin();
        assert_eq!(kit.size(), (160, 160));
        for e in CharacterEmote::ALL {
            let img = kit.compose(e, 0, 8);
            let c = coverage(&img);
            assert!((0.05..0.8).contains(&c), "{}: coverage {c}", e.name());
            let other = kit.compose(e, 2, 8);
            assert_ne!(img, other, "{} does not animate", e.name());
            assert!(kit.choice(Some(e)).len() == 8, "{}: every slot resolves", e.name());
        }
        let neutral = kit.neutral();
        assert!(coverage(&neutral) > 0.05);
        assert_ne!(neutral, kit.compose(CharacterEmote::ChibiCheer, 0, 8));
    }

    #[test]
    fn variants_differ_and_layers_are_not_empty() {
        let kit = ChibiKit::builtin();
        for slot in DEFAULT_SLOTS {
            for v in builtin_variants(slot) {
                let l = &kit.layers[slot][*v];
                assert_eq!(
                    l.is_empty(),
                    (slot, *v) == ("fx", "none") || (slot, *v) == ("hands", "none"),
                    "{slot}/{v}"
                );
            }
        }
        let eyes: Vec<&RgbaImage> = builtin_variants("eyes").iter().map(|v| &kit.layers["eyes"][*v].img).collect();
        for i in 0..eyes.len() {
            for j in i + 1..eyes.len() {
                assert_ne!(eyes[i], eyes[j]);
            }
        }
        // The face sits horizontally centred.
        let face = &kit.layers["face"]["default"];
        assert!((face.centroid.0 - 80.0).abs() < 1.0, "{:?}", face.centroid);
    }

    #[test]
    fn atlas_is_a_horizontal_strip() {
        let kit = ChibiKit::builtin();
        let a = kit.atlas(CharacterEmote::MusicNote, 6);
        assert_eq!((a.width(), a.height()), (6 * 160, 160));
        let dir = std::env::temp_dir().join(format!("zena-mcp-test-{}-atlas", std::process::id()));
        let p = kit.write_atlas(CharacterEmote::MusicNote, 4, &dir).unwrap();
        assert!(p.ends_with("chibi_MUSIC_NOTE.png"));
        assert_eq!(image::open(&p).unwrap().width(), 640);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn external_kit_overrides_a_slot() {
        let dir = std::env::temp_dir().join(format!("zena-mcp-test-{}-kit", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        RgbaImage::from_pixel(20, 20, Rgba([255, 0, 0, 255]))
            .save(dir.join("fx.png"))
            .unwrap();
        std::fs::write(
            dir.join("kit.json"),
            r#"{"size": [160, 160],
                "layers": {"fx": {"red_box": {"png": "fx.png", "offset": [10, 10]}}},
                "emotes": {"QUESTION": {"fx": "red_box", "eyes": "no_such_variant"}}}"#,
        )
        .unwrap();
        let kit = ChibiKit::load(&dir).unwrap();
        let red = |img: &RgbaImage| img.pixels().filter(|p| p[0] > 200 && p[1] < 40 && p[2] < 40 && p[3] > 200).count();
        let q = kit.compose(CharacterEmote::Question, 0, 8);
        assert!(red(&q) > 300, "the red box replaces the ? fx");
        assert_eq!(red(&ChibiKit::builtin().compose(CharacterEmote::Question, 0, 8)), 0);
        let choice = kit.choice(Some(CharacterEmote::Question));
        assert!(
            choice.contains(&("eyes".to_string(), "normal".to_string())),
            "unknown variants fall back: {choice:?}"
        );
        assert!(kit.variants("fx").contains(&"red_box".to_string()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn builtin_scales_to_other_sizes() {
        let kit = ChibiKit::builtin_sized(96, 128);
        let img = kit.compose(CharacterEmote::ChibiCheer, 0, 4);
        assert_eq!((img.width(), img.height()), (96, 128));
        assert!(coverage(&img) > 0.05);
    }

    #[test]
    fn affine_inverse_round_trips() {
        let t = Affine::rotate_about(10.0, 20.0, 0.3)
            .then(Affine::scale_about(5.0, 5.0, 1.2, 0.8))
            .then(Affine::translate(3.0, -2.0));
        let (x, y) = t.invert().apply(t.apply(7.0, 9.0).0, t.apply(7.0, 9.0).1);
        assert!((x - 7.0).abs() < 1e-4 && (y - 9.0).abs() < 1e-4);
    }
}
