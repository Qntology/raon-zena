//! Stage 1 geometry: the player character sits at the screen centre `(0, 0)`; every event is
//! expressed as a normalised distance `r` and an azimuth `θ` around it.

use serde::{Deserialize, Serialize};

/// Axis-aligned box in screen pixels (`x0,y0` inclusive top-left, `x1,y1` exclusive bottom-right).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl BBox {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            x0: x0.min(x1),
            y0: y0.min(y1),
            x1: x0.max(x1),
            y1: y0.max(y1),
        }
    }
    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }
    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)
    }
    /// Intersection-over-union, 0 when either box is empty.
    pub fn iou(&self, o: &BBox) -> f32 {
        let ix = (self.x1.min(o.x1) - self.x0.max(o.x0)).max(0.0);
        let iy = (self.y1.min(o.y1) - self.y0.max(o.y0)).max(0.0);
        let inter = ix * iy;
        let union = self.area() + o.area() - inter;
        if union <= 0.0 {
            0.0
        } else {
            inter / union
        }
    }
}

/// Polar coordinate around the screen centre.
///
/// * `r` is normalised by the half-diagonal, so `0` is the player and `1` a screen corner.
/// * `theta_deg` is in `[0, 360)`, `0°` = right, `90°` = up (screen y grows downward, so it is flipped).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Polar {
    pub r: f32,
    pub theta_deg: f32,
}

/// Size of the captured desktop / game surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenGeometry {
    pub width: f32,
    pub height: f32,
}

impl ScreenGeometry {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: width as f32,
            height: height as f32,
        }
    }
    pub fn half_diagonal(&self) -> f32 {
        0.5 * (self.width * self.width + self.height * self.height).sqrt()
    }
    /// Pixel → polar around the centre.
    pub fn to_polar(&self, x: f32, y: f32) -> Polar {
        let dx = x - self.width * 0.5;
        let dy = self.height * 0.5 - y; // flip so +y is "up"
        let r = (dx * dx + dy * dy).sqrt() / self.half_diagonal();
        let mut t = dy.atan2(dx).to_degrees();
        if t < 0.0 {
            t += 360.0;
        }
        if t >= 360.0 {
            t -= 360.0;
        }
        Polar { r, theta_deg: t }
    }
    /// Polar → pixel.
    pub fn from_polar(&self, p: Polar) -> (f32, f32) {
        let rad = p.theta_deg.to_radians();
        let d = p.r * self.half_diagonal();
        (self.width * 0.5 + d * rad.cos(), self.height * 0.5 - d * rad.sin())
    }
}

/// Radial motion of an event relative to the player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FlowDirection {
    /// Centre → outside: the player's own skill / attack.
    Outward,
    /// Outside → centre: approaching monster or incoming attack (threat).
    Inward,
    /// Moving around the player at roughly constant distance.
    Tangential,
    /// No meaningful motion between the two samples.
    Static,
}

/// Thresholds for [`classify_flow`], in normalised-r units per second and degrees per second.
#[derive(Debug, Clone, Copy)]
pub struct FlowThresholds {
    pub radial_speed: f32,
    pub angular_speed_deg: f32,
}

impl Default for FlowThresholds {
    fn default() -> Self {
        Self {
            radial_speed: 0.08,
            angular_speed_deg: 25.0,
        }
    }
}

/// Smallest signed angle difference `b - a` in degrees, in `(-180, 180]`.
pub fn angle_delta_deg(a: f32, b: f32) -> f32 {
    let mut d = (b - a) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    }
    if d <= -180.0 {
        d += 360.0;
    }
    d
}

/// Classify the radial flow of one tracked event between two samples `dt_s` seconds apart.
/// Returns the flow and the radial velocity `dr/dt` (negative = approaching).
pub fn classify_flow(prev: Polar, cur: Polar, dt_s: f32, th: FlowThresholds) -> (FlowDirection, f32) {
    if dt_s <= 0.0 {
        return (FlowDirection::Static, 0.0);
    }
    let vr = (cur.r - prev.r) / dt_s;
    // Arc length on the unit-r circle is r·dθ; use the mean radius so motion close to the
    // player is not exaggerated.
    let w = angle_delta_deg(prev.theta_deg, cur.theta_deg) / dt_s;
    let tangential = w.abs() * 0.5 * (prev.r + cur.r);
    if vr.abs() >= th.radial_speed && vr.abs() >= tangential.to_radians() {
        if vr < 0.0 {
            (FlowDirection::Inward, vr)
        } else {
            (FlowDirection::Outward, vr)
        }
    } else if w.abs() >= th.angular_speed_deg {
        (FlowDirection::Tangential, vr)
    } else {
        (FlowDirection::Static, vr)
    }
}

/// Classify a *burst* of change pixels by comparing where its energy sits now versus one frame
/// ago: energy that expands away from the centre is `Outward`, energy that contracts is `Inward`.
/// `prev_r`/`cur_r` are energy-weighted mean radii of the same ROI.
pub fn classify_burst(prev_r: f32, cur_r: f32, min_delta: f32) -> FlowDirection {
    let d = cur_r - prev_r;
    if d > min_delta {
        FlowDirection::Outward
    } else if d < -min_delta {
        FlowDirection::Inward
    } else {
        FlowDirection::Static
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_roundtrip_and_axes() {
        let g = ScreenGeometry::new(1920, 1080);
        let c = g.to_polar(960.0, 540.0);
        assert!(c.r.abs() < 1e-6);
        let right = g.to_polar(1920.0, 540.0);
        assert!((right.theta_deg - 0.0).abs() < 1e-3);
        let up = g.to_polar(960.0, 0.0);
        assert!((up.theta_deg - 90.0).abs() < 1e-3);
        let left = g.to_polar(0.0, 540.0);
        assert!((left.theta_deg - 180.0).abs() < 1e-3);
        let corner = g.to_polar(1920.0, 1080.0);
        assert!((corner.r - 1.0).abs() < 1e-5);
        assert!((corner.theta_deg - 360.0 + 29.357).abs() < 0.01);
        for &(x, y) in &[(10.0, 20.0), (1500.0, 900.0), (960.0, 10.0)] {
            let (bx, by) = g.from_polar(g.to_polar(x, y));
            assert!((bx - x).abs() < 1e-2 && (by - y).abs() < 1e-2);
        }
    }

    #[test]
    fn flow_classification() {
        let th = FlowThresholds::default();
        let far = Polar { r: 0.8, theta_deg: 30.0 };
        let near = Polar { r: 0.5, theta_deg: 32.0 };
        assert_eq!(classify_flow(far, near, 1.0, th).0, FlowDirection::Inward);
        assert_eq!(classify_flow(near, far, 1.0, th).0, FlowDirection::Outward);
        let orbit = Polar { r: 0.5, theta_deg: 80.0 };
        assert_eq!(classify_flow(near, orbit, 1.0, th).0, FlowDirection::Tangential);
        assert_eq!(classify_flow(near, near, 1.0, th).0, FlowDirection::Static);
        assert!((angle_delta_deg(350.0, 10.0) - 20.0).abs() < 1e-4);
        assert!((angle_delta_deg(10.0, 350.0) + 20.0).abs() < 1e-4);
    }

    #[test]
    fn iou_basic() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-6);
        assert_eq!(a.iou(&BBox::new(20.0, 20.0, 30.0, 30.0)), 0.0);
    }
}
