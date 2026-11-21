//! CPU text for the overlay: word-wrapped Korean / English lines rasterized with a system font into the speech
//! bubble (with a tail toward Zena's face) and the chat box (with the IME preedit underlined and a caret).
//! Re-rasterized only when the text changes, then uploaded as one sprite texture.

use ab_glyph::{point, Font, FontArc, PxScale, ScaleFont};
use std::path::{Path, PathBuf};

/// Fonts tried when none is given (Korean-capable first).
fn system_fonts() -> Vec<(PathBuf, u32)> {
    let mut v = Vec::new();
    if let Some(win) = std::env::var_os("WINDIR") {
        let f = PathBuf::from(win).join("Fonts");
        v.push((f.join("malgun.ttf"), 0));
        v.push((f.join("gulim.ttc"), 0));
        v.push((f.join("arial.ttf"), 0));
    }
    for p in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ] {
        v.push((PathBuf::from(p), 0));
    }
    v
}

pub struct TextRaster {
    font: Option<FontArc>,
    pub source: String,
}

/// Straight-alpha RGBA8 pixels.
pub struct Image {
    pub w: u32,
    pub h: u32,
    pub rgba: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct BoxStyle {
    pub font_px: f32,
    pub fill: [f32; 4],
    pub border: [f32; 4],
    pub text: [f32; 4],
    pub accent: [f32; 4],
    pub radius: f32,
    pub border_px: f32,
    pub padding: f32,
    pub max_lines: usize,
    /// A tail on the top edge pointing up (toward the face).
    pub tail: bool,
}

impl BoxStyle {
    pub fn bubble(scale: f32) -> Self {
        Self {
            font_px: 17.0 * scale,
            fill: [1.0, 1.0, 1.0, 0.93],
            border: [0.42, 0.25, 0.70, 1.0],
            text: [0.16, 0.12, 0.22, 1.0],
            accent: [0.42, 0.25, 0.70, 1.0],
            radius: 14.0 * scale,
            border_px: 2.0 * scale,
            padding: 10.0 * scale,
            max_lines: 4,
            tail: true,
        }
    }
    pub fn input(scale: f32) -> Self {
        Self {
            font_px: 16.0 * scale,
            fill: [0.12, 0.10, 0.18, 0.92],
            border: [0.62, 0.45, 0.95, 1.0],
            text: [1.0, 1.0, 1.0, 1.0],
            accent: [0.80, 0.70, 1.0, 1.0],
            radius: 10.0 * scale,
            border_px: 2.0 * scale,
            padding: 8.0 * scale,
            max_lines: 2,
            tail: false,
        }
    }

    /// Stage 0 compact sync status pill.
    pub fn pill(scale: f32, border: [f32; 4]) -> Self {
        Self {
            font_px: 12.5 * scale,
            fill: [0.10, 0.09, 0.16, 0.88],
            border,
            text: [0.96, 0.96, 1.0, 1.0],
            accent: border,
            radius: 12.0 * scale,
            border_px: 1.5 * scale,
            padding: 5.0 * scale,
            max_lines: 1,
            tail: false,
        }
    }

    /// Stage 1 unseen rejection alert chip.
    pub fn rejection(scale: f32) -> Self {
        Self {
            font_px: 13.0 * scale,
            fill: [0.24, 0.10, 0.12, 0.94],
            border: [0.95, 0.42, 0.42, 1.0],
            text: [1.0, 0.92, 0.92, 1.0],
            accent: [1.0, 0.65, 0.45, 1.0],
            radius: 10.0 * scale,
            border_px: 2.0 * scale,
            padding: 7.0 * scale,
            max_lines: 2,
            tail: false,
        }
    }

    /// Stage 2 history / verdict drawer panel (non-blended local vs global rows + ▲/▼).
    pub fn drawer(scale: f32) -> Self {
        Self {
            font_px: 12.5 * scale,
            fill: [0.09, 0.08, 0.15, 0.94],
            border: [0.52, 0.38, 0.85, 1.0],
            text: [0.95, 0.94, 0.99, 1.0],
            accent: [0.76, 0.66, 1.0, 1.0],
            radius: 10.0 * scale,
            border_px: 1.8 * scale,
            padding: 7.0 * scale,
            max_lines: 10,
            tail: false,
        }
    }

    /// 1–9 numbered navigation badge (gold lock border when `commit_guarded`).
    pub fn badge(scale: f32, border: [f32; 4], commit_guarded: bool) -> Self {
        Self {
            font_px: 13.0 * scale,
            fill: if commit_guarded {
                [0.24, 0.18, 0.06, 0.94]
            } else {
                [0.12, 0.10, 0.20, 0.92]
            },
            border: if commit_guarded {
                [0.98, 0.78, 0.24, 1.0]
            } else {
                border
            },
            text: [1.0, 1.0, 1.0, 1.0],
            accent: border,
            radius: 8.0 * scale,
            border_px: 2.0 * scale,
            padding: 4.5 * scale,
            max_lines: 1,
            tail: false,
        }
    }
}

/// Premultiplied float canvas.
struct Canvas {
    w: usize,
    h: usize,
    px: Vec<[f32; 4]>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            px: vec![[0.0; 4]; w * h],
        }
    }
    /// Composite `color` (straight) with coverage `a` over pixel `(x, y)`.
    fn blend(&mut self, x: i32, y: i32, color: [f32; 4], a: f32) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h || a <= 0.0 {
            return;
        }
        let a = (a.min(1.0)) * color[3];
        let p = &mut self.px[y as usize * self.w + x as usize];
        for c in 0..3 {
            p[c] = color[c] * a + p[c] * (1.0 - a);
        }
        p[3] = a + p[3] * (1.0 - a);
    }
    fn into_image(self) -> Image {
        let mut rgba = Vec::with_capacity(self.w * self.h * 4);
        for p in self.px {
            let a = p[3];
            for c in 0..3 {
                let v = if a > 0.0 { p[c] / a } else { 0.0 };
                rgba.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            rgba.push((a.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
        Image {
            w: self.w as u32,
            h: self.h as u32,
            rgba,
        }
    }
}

/// Signed distance to a rounded rectangle `(x0, y0, x1, y1)` with corner radius `r`.
fn rounded_rect_sd(x: f32, y: f32, rect: [f32; 4], r: f32) -> f32 {
    let cx = (rect[0] + rect[2]) * 0.5;
    let cy = (rect[1] + rect[3]) * 0.5;
    let hx = (rect[2] - rect[0]) * 0.5 - r;
    let hy = (rect[3] - rect[1]) * 0.5 - r;
    let dx = (x - cx).abs() - hx;
    let dy = (y - cy).abs() - hy;
    let ox = dx.max(0.0);
    let oy = dy.max(0.0);
    (ox * ox + oy * oy).sqrt() + dx.max(dy).min(0.0) - r
}

/// Signed distance to the tail triangle (apex up at `(ax, ay)`, base on `y = by` of half width `hw`).
fn tail_sd(x: f32, y: f32, ax: f32, ay: f32, by: f32, hw: f32) -> f32 {
    // Distances to the two slanted edges and the base, positive outside.
    let h = by - ay;
    let len = (h * h + hw * hw).sqrt();
    let left = (h * (ax - x) - hw * (y - ay)) / len;
    let right = (h * (x - ax) - hw * (y - ay)) / len;
    left.max(right).max(y - by)
}

impl TextRaster {
    /// `path`: a TTF/OTF/TTC file; otherwise the system's Korean font.
    pub fn load(path: Option<&Path>) -> Self {
        let candidates: Vec<(PathBuf, u32)> = match path {
            Some(p) => vec![(p.to_path_buf(), 0)],
            None => system_fonts(),
        };
        for (p, index) in candidates {
            let Ok(bytes) = std::fs::read(&p) else { continue };
            if let Ok(f) = ab_glyph::FontVec::try_from_vec_and_index(bytes, index) {
                return Self {
                    font: Some(FontArc::new(f)),
                    source: p.display().to_string(),
                };
            }
        }
        Self {
            font: None,
            source: "none".into(),
        }
    }

    pub fn has_font(&self) -> bool {
        self.font.is_some()
    }

    fn advance(&self, s: &str, px: f32) -> f32 {
        let Some(f) = &self.font else { return 0.0 };
        let sf = f.as_scaled(PxScale::from(px));
        let mut w = 0.0;
        let mut prev = None;
        for c in s.chars() {
            let id = sf.glyph_id(c);
            if let Some(p) = prev {
                w += sf.kern(p, id);
            }
            w += sf.h_advance(id);
            prev = Some(id);
        }
        w
    }

    /// Greedy word wrap; words wider than a line break between characters (Korean sentences often run long).
    pub fn wrap(&self, text: &str, px: f32, max_w: f32) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for para in text.split('\n') {
            let mut line = String::new();
            for word in para.split(' ').filter(|w| !w.is_empty()) {
                let cand = if line.is_empty() {
                    word.to_string()
                } else {
                    format!("{line} {word}")
                };
                if self.advance(&cand, px) <= max_w {
                    line = cand;
                    continue;
                }
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                for c in word.chars() {
                    let cand = format!("{line}{c}");
                    if self.advance(&cand, px) > max_w && !line.is_empty() {
                        lines.push(std::mem::take(&mut line));
                        line.push(c);
                    } else {
                        line = cand;
                    }
                }
            }
            if !line.is_empty() {
                lines.push(line);
            }
        }
        lines
    }

    fn draw_line(&self, cv: &mut Canvas, s: &str, x: f32, baseline: f32, px: f32, color: [f32; 4]) -> f32 {
        let Some(f) = &self.font else { return x };
        let sf = f.as_scaled(PxScale::from(px));
        let mut pen = x;
        let mut prev = None;
        for c in s.chars() {
            let id = sf.glyph_id(c);
            if let Some(p) = prev {
                pen += sf.kern(p, id);
            }
            let g = id.with_scale_and_position(PxScale::from(px), point(pen, baseline));
            if let Some(og) = f.outline_glyph(g) {
                let b = og.px_bounds();
                og.draw(|gx, gy, cov| cv.blend(b.min.x as i32 + gx as i32, b.min.y as i32 + gy as i32, color, cov));
            }
            pen += sf.h_advance(id);
            prev = Some(id);
        }
        pen
    }

    /// A box `width` px wide holding `text` (the last `max_lines` lines when it is longer), plus `preedit`
    /// (underlined) and a caret when `caret` is set. `None` without a font or text.
    pub fn render_box(&self, text: &str, preedit: &str, caret: bool, width: u32, st: &BoxStyle) -> Option<Image> {
        let font = self.font.as_ref()?;
        if text.trim().is_empty() && preedit.is_empty() && !caret {
            return None;
        }
        let sf = font.as_scaled(PxScale::from(st.font_px));
        let line_h = (sf.ascent() - sf.descent() + sf.line_gap()).ceil();
        let inner_w = width as f32 - 2.0 * st.padding - 2.0 * st.border_px;
        let full = format!("{text}{preedit}");
        let mut lines = self.wrap(&full, st.font_px, inner_w - st.font_px * 0.5);
        if lines.is_empty() {
            lines.push(String::new());
        }
        if lines.len() > st.max_lines {
            lines.drain(..lines.len() - st.max_lines);
        }
        let tail_h = if st.tail { 10.0 * st.font_px / 17.0 } else { 0.0 };
        let box_h = lines.len() as f32 * line_h + 2.0 * st.padding + 2.0 * st.border_px;
        let (w, h) = (width as usize, (box_h + tail_h).ceil() as usize);
        let mut cv = Canvas::new(w, h);
        let rect = [0.5, tail_h + 0.5, w as f32 - 0.5, h as f32 - 0.5];
        let inner = [
            rect[0] + st.border_px,
            rect[1] + st.border_px,
            rect[2] - st.border_px,
            rect[3] - st.border_px,
        ];
        let (ax, hw) = (w as f32 * 0.5, tail_h * 0.9);
        // 3×3 supersampled shape: border colour everywhere inside, fill inside the inner outline.
        const SS: [f32; 3] = [-1.0 / 3.0, 0.0, 1.0 / 3.0];
        for y in 0..h {
            for x in 0..w {
                let (mut outer, mut fill) = (0.0, 0.0);
                for sy in SS {
                    for sx in SS {
                        let (fx, fy) = (x as f32 + 0.5 + sx, y as f32 + 0.5 + sy);
                        let mut d = rounded_rect_sd(fx, fy, rect, st.radius);
                        let mut di = rounded_rect_sd(fx, fy, inner, (st.radius - st.border_px).max(0.0));
                        if st.tail {
                            let t = tail_sd(fx, fy, ax, 0.5, rect[1] + st.border_px + 1.0, hw);
                            d = d.min(t);
                            di = di.min(t + st.border_px * 1.2);
                        }
                        outer += (d <= 0.0) as u32 as f32;
                        fill += (di <= 0.0) as u32 as f32;
                    }
                }
                cv.blend(x as i32, y as i32, st.border, outer / 9.0);
                cv.blend(x as i32, y as i32, st.fill, fill / 9.0);
            }
        }
        // Text; the preedit (IME composition) is underlined in the accent colour.
        let x0 = inner[0] + st.padding;
        let mut consumed = 0usize;
        let pre_from = text.chars().count();
        let n = lines.len();
        for (i, l) in lines.iter().enumerate() {
            let baseline = inner[1] + st.padding + i as f32 * line_h + sf.ascent();
            let chars: Vec<char> = l.chars().collect();
            let split = pre_from.saturating_sub(consumed).min(chars.len());
            let (done, pre): (String, String) = (chars[..split].iter().collect(), chars[split..].iter().collect());
            let mut pen = self.draw_line(&mut cv, &done, x0, baseline, st.font_px, st.text);
            if !pre.is_empty() {
                let start = pen;
                pen = self.draw_line(&mut cv, &pre, pen, baseline, st.font_px, st.accent);
                for x in start as i32..pen as i32 {
                    cv.blend(x, (baseline + 3.0) as i32, st.accent, 1.0);
                }
            }
            if caret && i + 1 == n {
                for y in (baseline - sf.ascent() * 0.85) as i32..(baseline + 2.0) as i32 {
                    cv.blend(pen as i32 + 1, y, st.accent, 1.0);
                    cv.blend(pen as i32 + 2, y, st.accent, 0.6);
                }
            }
            consumed += chars.len() + 1; // the wrap ate one space per break
        }
        Some(cv.into_image())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_have_the_right_sign() {
        let r = [0.0, 0.0, 100.0, 50.0];
        assert!(rounded_rect_sd(50.0, 25.0, r, 10.0) < 0.0);
        assert!(rounded_rect_sd(-1.0, 25.0, r, 10.0) > 0.0);
        assert!(rounded_rect_sd(1.0, 1.0, r, 10.0) > 0.0, "the corner is cut");
        assert!(tail_sd(50.0, 8.0, 50.0, 0.0, 10.0, 8.0) < 0.0);
        assert!(tail_sd(40.0, 1.0, 50.0, 0.0, 10.0, 8.0) > 0.0);
    }

    #[test]
    fn korean_text_wraps_and_renders_when_a_font_exists() {
        let t = TextRaster::load(None);
        if !t.has_font() {
            eprintln!("no system font; skipping");
            return;
        }
        let lines = t.wrap("오른쪽 위에 보스야! 지금 바로 굴러서 피해, 그다음 반격!", 17.0, 150.0);
        assert!(lines.len() >= 2, "{lines:?}");
        assert!(lines.iter().all(|l| t.advance(l, 17.0) <= 150.0 + 1.0));
        let img = t
            .render_box("안녕! 오늘도 같이 하자!", "", false, 330, &BoxStyle::bubble(1.0))
            .unwrap();
        assert_eq!(img.w, 330);
        let opaque = img.rgba.chunks(4).filter(|p| p[3] > 200).count();
        let dark = img.rgba.chunks(4).filter(|p| p[3] > 200 && p[0] < 100).count();
        assert!(opaque > (img.w * img.h / 2) as usize, "the box is filled");
        assert!(dark > 50, "glyphs were drawn ({dark})");
        let input = t.render_box("제나야", "안", true, 330, &BoxStyle::input(1.0)).unwrap();
        assert!(input.h > 20);
        assert!(t.render_box("  ", "", false, 330, &BoxStyle::bubble(1.0)).is_none());
    }
}
