//! Pointing at a numbered target inside another app's window (`contracts/course_nav_v1.json` CN-10 / CN-12).
//!
//! pd_nav_number numbers regions in window-relative coordinates (0..1), while Zena's gaze, the screen polar
//! coordinates and the 3D coaching arrow live in desktop pixels. [`NavTargetProjector`] maps a target rect of
//! a window placed anywhere on the desktop to the desktop point, a polar coordinate centred on that window (for
//! `Situation::polar` and the eight direction words of the prompt), the avatar's [`LookAt`] from her head, the
//! arrow's angle `atan2(−dy, dx)`, and whether the overlay itself covers the target (IoU > 0.05 or the centre
//! inside the overlay rect) so the app can step aside.
//!
//! [`SpokenSlotLock`] collects the badge numbers Zena is saying ("2번 성문으로 가자!") into pd_nav_number's
//! `locked_numbers` bit mask (bit n = number n), held while the line is spoken and for 8 s after it.

use crate::avatar::{look_at_point, AvatarPlacement, LookAt};
use crate::geometry::{BBox, Polar, ScreenGeometry};
use crate::prompt::direction_label;

/// Overlap (IoU) above which the overlay counts as covering the target.
pub const OCCLUSION_IOU: f32 = 0.05;
/// A badge number Zena said stays locked this long after her line ended.
pub const SPOKEN_LOCK_HOLD_MS: u64 = 8_000;
/// Phase 7 Part 2 (`AC-04`, `κ_sr = 2.5`): 스크린리더 활성 시 발화 번호 잠금 유지 시간(20초).
pub const SPOKEN_LOCK_HOLD_SR_MS: u64 = 20_000;

/// One target in desktop terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedNavTarget {
    pub number: u8,
    /// The target centre on the desktop (px).
    pub desktop_point: (f32, f32),
    /// The target rect on the desktop (px).
    pub desktop_rect: BBox,
    /// Around the target window's centre (`r` 0 = centre, 1 = window corner; `θ` 0° right, 90° up).
    pub window_polar: Polar,
    /// The avatar's gaze from her head to the target.
    pub avatar_look: LookAt,
    /// The coaching arrow's rotation from the head toward the target (0° right, 90° up), in [0, 360).
    pub arrow_angle_deg: f32,
    /// The overlay window covers the target: make it click-through / translucent or dock elsewhere.
    pub occluded_by_overlay: bool,
}

impl ProjectedNavTarget {
    /// The prompt's direction words for the target ("오른쪽 아래"), consistent with `window_polar`.
    pub fn direction_label(&self) -> &'static str {
        direction_label(self.window_polar.theta_deg)
    }
}

/// Window-relative point `(rx, ry)` ∈ [0,1]² of `window` (desktop px) → desktop px.
pub fn window_to_desktop(window: &BBox, rx: f32, ry: f32) -> (f32, f32) {
    (
        window.x0 + rx.clamp(0.0, 1.0) * window.width(),
        window.y0 + ry.clamp(0.0, 1.0) * window.height(),
    )
}

/// The arrow angle from `from` to `to` (desktop px, y down): `atan2(−(to.y − from.y), to.x − from.x)` in degrees,
/// normalised to [0, 360).
pub fn arrow_angle_deg(from: (f32, f32), to: (f32, f32)) -> f32 {
    let a = (-(to.1 - from.1)).atan2(to.0 - from.0).to_degrees().rem_euclid(360.0);
    // rem_euclid may round up to 360.0, and keeps the sign of -0.0.
    if a >= 360.0 || a == 0.0 {
        0.0
    } else {
        a
    }
}

/// Whether `overlay` covers `target`: IoU above [`OCCLUSION_IOU`], or the target centre inside the overlay.
pub fn overlay_occludes(overlay: &BBox, target: &BBox) -> bool {
    let (cx, cy) = target.center();
    overlay.iou(target) > OCCLUSION_IOU || (cx >= overlay.x0 && cx < overlay.x1 && cy >= overlay.y0 && cy < overlay.y1)
}

/// Projects window-relative targets for one overlay / avatar placement.
#[derive(Debug, Clone, Copy)]
pub struct NavTargetProjector {
    /// The captured desktop: the fallback window when the target window's rect is unknown (fullscreen).
    pub screen: ScreenGeometry,
    /// The overlay window on the desktop (px), e.g. 350×450.
    pub overlay: BBox,
    /// Zena's head on the desktop.
    pub place: AvatarPlacement,
}

impl NavTargetProjector {
    /// `window`: the target app's window rect in desktop px (`None` → the whole screen). `rel_rect`: the target
    /// region inside it, normalised to [0,1]² (pd_nav_number `Region.rect`).
    pub fn project(&self, number: u8, window: Option<BBox>, rel_rect: BBox) -> ProjectedNavTarget {
        let full = BBox::new(0.0, 0.0, self.screen.width, self.screen.height);
        let w = window.filter(|w| w.width() > 0.0 && w.height() > 0.0).unwrap_or(full);
        let (x0, y0) = window_to_desktop(&w, rel_rect.x0, rel_rect.y0);
        let (x1, y1) = window_to_desktop(&w, rel_rect.x1, rel_rect.y1);
        let desktop_rect = BBox::new(x0, y0, x1, y1);
        let desktop_point = desktop_rect.center();
        let local = ScreenGeometry {
            width: w.width(),
            height: w.height(),
        };
        let window_polar = local.to_polar(desktop_point.0 - w.x0, desktop_point.1 - w.y0);
        let head = (self.place.head_x, self.place.head_y);
        ProjectedNavTarget {
            number,
            desktop_point,
            desktop_rect,
            window_polar,
            avatar_look: look_at_point(&self.place, desktop_point.0, desktop_point.1),
            arrow_angle_deg: arrow_angle_deg(head, desktop_point),
            occluded_by_overlay: overlay_occludes(&self.overlay, &desktop_rect),
        }
    }

    /// [`NavTargetProjector::project`] for a target known only by its relative centre.
    pub fn project_point(&self, number: u8, window: Option<BBox>, rel_center: (f32, f32)) -> ProjectedNavTarget {
        let (x, y) = rel_center;
        self.project(number, window, BBox::new(x, y, x, y))
    }
}

/// Badge numbers 1..=9 named in `text` as a mask (bit n = number n): "2번", "2 번", "#2", "#2번". "2번째", "12번"
/// and "0번" are not badges.
pub fn spoken_slot_mask(text: &str) -> u16 {
    let chars: Vec<char> = text.chars().collect();
    let mut mask = 0u16;
    for (i, &c) in chars.iter().enumerate() {
        // `to_digit` only accepts ASCII digits, so fullwidth digits never count.
        let Some(d) = c.to_digit(10).filter(|d| (1..=9).contains(d)) else {
            continue;
        };
        let prev = i.checked_sub(1).map(|j| chars[j]);
        if prev.is_some_and(|p| p.is_ascii_digit()) || chars.get(i + 1).is_some_and(char::is_ascii_digit) {
            continue;
        }
        let mut j = i + 1;
        while chars.get(j) == Some(&' ') {
            j += 1;
        }
        let beon = chars.get(j) == Some(&'번') && chars.get(j + 1) != Some(&'째');
        if beon || prev == Some('#') {
            mask |= 1 << d;
        }
    }
    mask
}

/// The badge numbers Zena is saying, for pd_nav_number's `locked_numbers`: a full set of badges must not swap a
/// number she is pointing at with her words.
#[derive(Debug, Clone)]
pub struct SpokenSlotLock {
    line: String,
    until_ms: [u64; 10],
    pub hold_ms: u64,
}

impl Default for SpokenSlotLock {
    fn default() -> Self {
        Self {
            line: String::new(),
            until_ms: [0; 10],
            hold_ms: SPOKEN_LOCK_HOLD_MS,
        }
    }
}

impl SpokenSlotLock {
    pub fn new() -> Self {
        Self::default()
    }

    /// A streamed piece of the line being spoken (`StreamEvent::Speech` or a raw delta); numbers split across
    /// pieces ("2" + "번") are recognised. Returns the mask of the current line.
    pub fn push_spoken(&mut self, piece: &str) -> u16 {
        self.line.push_str(piece);
        spoken_slot_mask(&self.line)
    }

    /// The line finished or was cut off (ask epoch): its numbers stay locked for `hold_ms` more.
    pub fn end_line(&mut self, now_ms: u64) {
        let m = spoken_slot_mask(&self.line);
        for n in 1..=9usize {
            if m & (1 << n) != 0 {
                self.until_ms[n] = self.until_ms[n].max(now_ms + self.hold_ms);
            }
        }
        self.line.clear();
    }

    /// The mask to send now: the current line's numbers and those still held.
    pub fn mask(&self, now_ms: u64) -> u16 {
        let held = (1..=9usize).filter(|&n| self.until_ms[n] > now_ms).fold(0u16, |m, n| m | (1 << n));
        held | spoken_slot_mask(&self.line)
    }

    pub fn is_locked(&self, number: u8, now_ms: u64) -> bool {
        (1..=9).contains(&number) && self.mask(now_ms) & (1 << number) != 0
    }

    /// Drops everything (screen change: the old numbers mean nothing).
    pub fn clear(&mut self) {
        self.line.clear();
        self.until_ms = [0; 10];
    }

    /// Phase 7 Part 2 (`AC-04`): 스크린리더 활성 시 번호 잠금 창을 20초로 확장한다.
    pub fn set_screen_reader_mode(&mut self, active: bool) {
        self.hold_ms = if active {
            SPOKEN_LOCK_HOLD_SR_MS
        } else {
            SPOKEN_LOCK_HOLD_MS
        };
    }
}

/// Phase 7 Part 2 (`AC-04`): 시각 장애·저시력 사용자를 위한 3D 공간 오디오 비콘 매개변수.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioBeaconCue {
    /// 스테레오 패닝 `[-1.0 (좌), +1.0 (우)]`.
    pub pan: f32,
    /// 수직 고도 피치(Hz): 아래 ~220Hz, 중앙 440Hz, 위 ~660Hz.
    pub pitch_hz: f32,
    /// 근접 펄스 주기(Hz): 가까울수록 빠르게 맥동.
    pub pulse_hz: f32,
    /// 8방향 언어 힌트(UIA LiveRegion 병행 출력용).
    pub direction: &'static str,
}

/// 화면 극좌표 `(r, θ)` 를 스테레오 패닝·피치·근접 펄스로 음향화한다.
pub fn sonify_polar(polar: &Polar) -> AudioBeaconCue {
    let rad = polar.theta_deg.to_radians();
    let r = polar.r.clamp(0.0, 1.0);
    let pan = (rad.cos() * r).clamp(-1.0, 1.0);
    let pitch_hz = (440.0 + 220.0 * rad.sin()).clamp(220.0, 660.0);
    let pulse_hz = (1.5 + 3.5 * (1.0 - r)).clamp(1.5, 5.0);
    AudioBeaconCue {
        pan,
        pitch_hz,
        pulse_hz,
        direction: direction_label(polar.theta_deg),
    }
}

/// Phase 7 Part 2 (`AC-04`): OS 돋보기(Magnifier) 확대 뷰포트 영역.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnifierViewport {
    pub rect: BBox,
    pub zoom: f32,
}

/// 돋보기 확대 뷰포트 내에서 대상(`target`)을 가리지 않는 가장 먼 모서리에 제나 오버레이를 도킹한다.
pub fn dock_within_magnifier(viewport: &MagnifierViewport, overlay_size: (f32, f32), target: &BBox) -> BBox {
    let (ow, oh) = (
        overlay_size.0.min(viewport.rect.width().max(1.0)),
        overlay_size.1.min(viewport.rect.height().max(1.0)),
    );
    let candidates = [
        BBox::new(viewport.rect.x0, viewport.rect.y0, viewport.rect.x0 + ow, viewport.rect.y0 + oh),
        BBox::new(viewport.rect.x1 - ow, viewport.rect.y0, viewport.rect.x1, viewport.rect.y0 + oh),
        BBox::new(viewport.rect.x0, viewport.rect.y1 - oh, viewport.rect.x0 + ow, viewport.rect.y1),
        BBox::new(viewport.rect.x1 - ow, viewport.rect.y1 - oh, viewport.rect.x1, viewport.rect.y1),
    ];
    let (tx, ty) = target.center();
    candidates
        .into_iter()
        .max_by(|a, b| {
            let occ_a = overlay_occludes(a, target);
            let occ_b = overlay_occludes(b, target);
            let dist = |box_: &BBox| {
                let (cx, cy) = box_.center();
                (cx - tx).hypot(cy - ty)
            };
            // 비가림 우선, 그다음 대상 중심으로부터의 거리 최대화
            occ_b.cmp(&occ_a).then_with(|| dist(a).total_cmp(&dist(b)))
        })
        .unwrap_or(candidates[3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brief::{CourseBrief, NextStepHint};
    use crate::prompt::{build_fast_path_prompt, SessionContext, Situation, ASSISTANT_PREFILL, USER_TURN};

    #[test]
    fn cn10_window_offset_polar_gaze_arrow_agree_and_the_overlay_steps_aside() {
        let screen = ScreenGeometry::new(1920, 1080);
        // The overlay (350×450) is docked over the lower right of the target window; Zena's head is in it.
        let overlay = BBox::new(1450.0, 400.0, 1800.0, 850.0);
        let place = AvatarPlacement {
            head_x: 1625.0,
            head_y: 520.0,
            viewer_depth_px: 900.0,
        };
        let p = NavTargetProjector { screen, overlay, place };
        let window = BBox::new(800.0, 200.0, 800.0 + 960.0, 200.0 + 540.0);
        let t = p.project(1, Some(window), BBox::new(0.88, 0.88, 0.92, 0.92));
        let (x, y) = t.desktop_point;
        assert!((x - 1664.0).abs() < 1e-3 && (y - 686.0).abs() < 1e-3, "{:?}", t.desktop_point);
        // Polar around the window centre (1280, 470): dx = 384, dy = −216.
        let r = (384f32.hypot(216.0)) / (0.5 * 960f32.hypot(540.0));
        assert!((t.window_polar.r - r).abs() < 1e-4, "{:?}", t.window_polar);
        assert!((t.window_polar.theta_deg - (360.0 + (-216f32).atan2(384.0).to_degrees())).abs() < 1e-3);
        assert_eq!(t.direction_label(), "오른쪽 아래");
        // Gaze and arrow both from the head at desktop coordinates: target is right of and below the head.
        let expect = arrow_angle_deg((1625.0, 520.0), (1664.0, 686.0));
        assert!((t.arrow_angle_deg - expect).abs() < 1e-4);
        assert!((t.arrow_angle_deg - (360.0 + (-166f32).atan2(39.0).to_degrees())).abs() < 1e-3);
        assert!(t.avatar_look.yaw_deg > 0.0 && t.avatar_look.pitch_deg < 0.0, "{:?}", t.avatar_look);
        let rad = t.arrow_angle_deg.to_radians();
        assert_eq!(rad.cos() > 0.0, t.avatar_look.yaw_deg > 0.0, "arrow and gaze agree horizontally");
        assert_eq!(rad.sin() > 0.0, t.avatar_look.pitch_deg > 0.0, "and vertically");
        assert!((t.avatar_look.yaw_deg - 39f32.atan2(900.0).to_degrees()).abs() < 1e-3);
        assert!(t.occluded_by_overlay, "the button is under the overlay");
        // Fed into the prompt, the polar gives the same direction words.
        let s = Situation {
            event_label: Some("다음 체크포인트".into()),
            polar: Some(t.window_polar),
            ..Default::default()
        };
        assert!(build_fast_path_prompt(&s).contains("오른쪽 아래"));
        // The same target with the overlay docked top-left is not occluded.
        let away = NavTargetProjector {
            overlay: BBox::new(0.0, 0.0, 350.0, 450.0),
            ..p
        };
        assert!(!away.project(1, Some(window), BBox::new(0.88, 0.88, 0.92, 0.92)).occluded_by_overlay);
    }

    #[test]
    fn fullscreen_fallback_and_point_targets() {
        let screen = ScreenGeometry::new(1920, 1080);
        let p = NavTargetProjector {
            screen,
            overlay: BBox::new(1570.0, 630.0, 1920.0, 1080.0),
            place: AvatarPlacement {
                head_x: 1745.0,
                head_y: 760.0,
                viewer_depth_px: 900.0,
            },
        };
        let t = p.project_point(3, None, (0.5, 0.0));
        assert_eq!(t.desktop_point, (960.0, 0.0));
        assert!((t.window_polar.theta_deg - 90.0).abs() < 1e-3, "straight up from the screen centre");
        assert_eq!(t.direction_label(), "위");
        assert!(!t.occluded_by_overlay);
        let degenerate = p.project_point(1, Some(BBox::new(10.0, 10.0, 10.0, 10.0)), (0.5, 0.5));
        assert_eq!(
            degenerate.desktop_point,
            (960.0, 540.0),
            "an empty window rect falls back to the screen"
        );
        let inside = p.project_point(2, None, (0.95, 0.95));
        assert!(inside.occluded_by_overlay, "a point target counts by its centre");
        assert_eq!(arrow_angle_deg((0.0, 0.0), (10.0, 0.0)), 0.0);
        assert!((arrow_angle_deg((0.0, 0.0), (0.0, -10.0)) - 90.0).abs() < 1e-4, "screen-up is 90°");
        assert!((arrow_angle_deg((0.0, 0.0), (-10.0, 0.0)) - 180.0).abs() < 1e-4);
        assert!((arrow_angle_deg((0.0, 0.0), (0.0, 10.0)) - 270.0).abs() < 1e-4);
    }

    #[test]
    fn cn12_prefix_stays_byte_identical_and_the_spoken_number_is_locked() {
        let brief = CourseBrief {
            passed: 2,
            length: 4,
            vs_best_ms: Some(-1_800),
            vs_public_ms: Some(-3_200),
            next_hint: NextStepHint::NumberedOnScreen {
                number: 2,
                clean_label: "성문 우회로".into(),
                direction_label: "왼쪽".into(),
            },
            verdict_judged_premium: true,
            ..Default::default()
        };
        let bare = build_fast_path_prompt(&Situation::default());
        let with = build_fast_path_prompt(&Situation {
            session: Some(SessionContext {
                course_brief: Some(brief),
                ..Default::default()
            }),
            ..Default::default()
        });
        let head = |p: &str| p[..p.find(USER_TURN).unwrap() + USER_TURN.len()].to_string();
        assert_eq!(head(&with), head(&bare), "system + tools prefix byte-identical");
        assert!(
            with.ends_with(ASSISTANT_PREFILL) && bare.ends_with(ASSISTANT_PREFILL),
            "<think></think> prefill unchanged"
        );
        let user = &with[head(&with).len()..with.len() - ASSISTANT_PREFILL.len()];
        assert!(user.contains("[다음 추천 스텝] #2 '성문 우회로' (왼쪽)"), "{user}");

        // The model streams "2번 성문으로 가자!" while a new Premium region wants a slot in the full set.
        let mut lock = SpokenSlotLock::new();
        assert_eq!(lock.push_spoken("2"), 0, "a bare digit is not a badge yet");
        assert_eq!(lock.push_spoken("번 성문"), 1 << 2);
        lock.push_spoken("으로 가자!");
        assert!(lock.is_locked(2, 10_000), "bit 2 holds slot 2 while speaking");
        assert!(!lock.is_locked(1, 10_000));
        lock.end_line(11_000);
        assert_eq!(lock.mask(11_000 + SPOKEN_LOCK_HOLD_MS - 1), 1 << 2, "held 8 s after the line");
        assert_eq!(lock.mask(11_000 + SPOKEN_LOCK_HOLD_MS), 0, "then released");
        lock.push_spoken("3번");
        lock.clear();
        assert_eq!(lock.mask(0), 0);
    }

    #[test]
    fn badge_numbers_in_speech() {
        assert_eq!(spoken_slot_mask("1번 간편결제 눌러, 아니면 #3"), (1 << 1) | (1 << 3));
        assert_eq!(spoken_slot_mask("2 번 레버"), 1 << 2);
        assert_eq!(spoken_slot_mask("#4번"), 1 << 4);
        assert_eq!(spoken_slot_mask("2번째 문, 12번 방, 0번, 이번엔"), 0);
        assert_eq!(spoken_slot_mask("３번"), 0, "fullwidth digits are not badge numbers");
    }

    #[test]
    fn ac04_screen_reader_lock_sonify_polar_and_magnifier_docking() {
        let mut lock = SpokenSlotLock::new();
        lock.set_screen_reader_mode(true);
        lock.push_spoken("2번 결제 진행");
        lock.end_line(1_000);
        // Under screen reader mode, #2 stays locked at 19.9s (would have expired at 8s normally).
        assert!(lock.is_locked(2, 1_000 + SPOKEN_LOCK_HOLD_SR_MS - 1));
        assert!(!lock.is_locked(2, 1_000 + SPOKEN_LOCK_HOLD_SR_MS));

        let right_up = Polar { r: 0.8, theta_deg: 45.0 };
        let beacon = sonify_polar(&right_up);
        assert!(beacon.pan > 0.4, "right-side target pans right: {}", beacon.pan);
        assert!(beacon.pitch_hz > 550.0, "upper target raises pitch: {}", beacon.pitch_hz);

        let vp = MagnifierViewport {
            rect: BBox::new(100.0, 100.0, 740.0, 460.0),
            zoom: 3.0,
        };
        let target_bottom_right = BBox::new(620.0, 380.0, 710.0, 440.0);
        let docked = dock_within_magnifier(&vp, (180.0, 160.0), &target_bottom_right);
        assert!(!overlay_occludes(&docked, &target_bottom_right));
        assert!(docked.x0 >= vp.rect.x0 && docked.x1 <= vp.rect.x1);
        assert!(docked.y0 >= vp.rect.y0 && docked.y1 <= vp.rect.y1);
        assert_eq!((docked.x0, docked.y0), (100.0, 100.0), "docks in farthest corner from target");
    }
}
