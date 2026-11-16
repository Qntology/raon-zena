//! Avatar abstraction adapter: one `control_zena_avatar` call → Live2D (`.model3.json`) and VRM
//! (`.vrm` / glTF) parameters, plus text-driven viseme lip-sync for Korean and English.

use crate::geometry::{Polar, ScreenGeometry};
use crate::tools::{BodyMotion, EmotionState, ZenaAvatarArgs};

/// Where the avatar's head is on the desktop and how far the "virtual viewer" is (in px) —
/// used to turn a target point on screen into head yaw/pitch.
#[derive(Debug, Clone, Copy)]
pub struct AvatarPlacement {
    pub head_x: f32,
    pub head_y: f32,
    pub viewer_depth_px: f32,
}

/// Look direction relative to facing the viewer (degrees; +yaw = avatar looks to screen-right,
/// +pitch = up).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LookAt {
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

pub const MAX_YAW: f32 = 35.0;
pub const MAX_PITCH: f32 = 25.0;

/// Gaze from the avatar's head toward an event given in player-centred polar coordinates.
pub fn look_at_event(screen: &ScreenGeometry, place: &AvatarPlacement, target: Polar) -> LookAt {
    let (tx, ty) = screen.from_polar(target);
    let dx = tx - place.head_x;
    let dy = place.head_y - ty;
    LookAt {
        yaw_deg: dx.atan2(place.viewer_depth_px).to_degrees().clamp(-MAX_YAW, MAX_YAW),
        pitch_deg: dy.atan2(place.viewer_depth_px).to_degrees().clamp(-MAX_PITCH, MAX_PITCH),
    }
}

// ------------------------------------------------------------------------------------------
// Live2D
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Live2DFrame {
    /// Cubism parameter id → value.
    pub params: Vec<(&'static str, f32)>,
    /// `.exp3.json` expression to cross-fade to.
    pub expression: Option<&'static str>,
    /// `.motion3.json` (group, index) to start.
    pub motion: Option<(&'static str, u32)>,
}

pub fn live2d_expression(e: EmotionState) -> &'static str {
    match e {
        EmotionState::Neutral => "neutral",
        EmotionState::Joy => "joy",
        EmotionState::Excited => "excited",
        EmotionState::Surprised => "surprised",
        EmotionState::SurprisedUrgent => "surprised_urgent",
        EmotionState::Angry => "angry",
        EmotionState::Sorrow => "sorrow",
        EmotionState::Worried => "worried",
        EmotionState::Focused => "focused",
        EmotionState::Relieved => "relieved",
    }
}

pub fn live2d_motion(m: BodyMotion) -> Option<(&'static str, u32)> {
    Some(match m {
        BodyMotion::Idle => return None,
        BodyMotion::Nod => ("Nod", 0),
        BodyMotion::LeanForwardPoint => ("Point", 0),
        BodyMotion::Cheer => ("Cheer", 0),
        BodyMotion::Clap => ("Cheer", 1),
        BodyMotion::Flinch => ("Flinch", 0),
        BodyMotion::Think => ("Think", 0),
        BodyMotion::HideCrouch => ("Hide", 0),
        BodyMotion::WaveGoodbye => ("Farewell", 0),
    })
}

/// Map a look direction to Cubism standard parameters. The head takes 60 % of the rotation and
/// the eyes the remainder, which reads more natural than turning the head all the way.
pub fn live2d_gaze_params(l: LookAt) -> [(&'static str, f32); 5] {
    [
        ("ParamAngleX", (l.yaw_deg * 0.6).clamp(-30.0, 30.0)),
        ("ParamAngleY", (l.pitch_deg * 0.6).clamp(-30.0, 30.0)),
        ("ParamAngleZ", (-l.yaw_deg * 0.12).clamp(-30.0, 30.0)),
        ("ParamEyeBallX", (l.yaw_deg / MAX_YAW).clamp(-1.0, 1.0)),
        ("ParamEyeBallY", (l.pitch_deg / MAX_PITCH).clamp(-1.0, 1.0)),
    ]
}

pub fn to_live2d(args: &ZenaAvatarArgs, look: Option<LookAt>) -> Live2DFrame {
    let mut params = Vec::new();
    if let Some(l) = look {
        params.extend_from_slice(&live2d_gaze_params(l));
    }
    if args.body_motion == BodyMotion::LeanForwardPoint {
        params.push(("ParamBodyAngleY", 8.0));
    }
    Live2DFrame {
        params,
        expression: Some(live2d_expression(args.emotion_state)),
        motion: live2d_motion(args.body_motion),
    }
}

// ------------------------------------------------------------------------------------------
// VRM
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrmVersion {
    /// VRM 0.x `VRMBlendShapeProxy` presets (Joy/Angry/Sorrow/Fun, A/I/U/E/O).
    V0,
    /// VRM 1.0 expression presets (happy/angry/sad/relaxed/surprised, aa/ih/ou/ee/oh).
    V1,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VrmFrame {
    pub look: LookAt,
    /// Expression / blend-shape preset → weight 0..1.
    pub expressions: Vec<(&'static str, f32)>,
    /// Humanoid animation clip to play.
    pub clip: Option<&'static str>,
    /// Extra spine lean (degrees, forward positive) solved by IK.
    pub spine_lean_deg: f32,
}

/// Emotion → (happy, angry, sad, relaxed, surprised) weights.
fn emotion_weights(e: EmotionState) -> [f32; 5] {
    match e {
        EmotionState::Neutral => [0.0, 0.0, 0.0, 0.0, 0.0],
        EmotionState::Joy => [1.0, 0.0, 0.0, 0.0, 0.0],
        EmotionState::Excited => [0.8, 0.0, 0.0, 0.0, 0.4],
        EmotionState::Surprised => [0.0, 0.0, 0.0, 0.0, 1.0],
        EmotionState::SurprisedUrgent => [0.0, 0.25, 0.0, 0.0, 1.0],
        EmotionState::Angry => [0.0, 1.0, 0.0, 0.0, 0.0],
        EmotionState::Sorrow => [0.0, 0.0, 1.0, 0.0, 0.0],
        EmotionState::Worried => [0.0, 0.0, 0.55, 0.0, 0.25],
        EmotionState::Focused => [0.0, 0.2, 0.0, 0.3, 0.0],
        EmotionState::Relieved => [0.3, 0.0, 0.0, 0.8, 0.0],
    }
}

pub fn to_vrm(args: &ZenaAvatarArgs, look: Option<LookAt>, version: VrmVersion) -> VrmFrame {
    let w = emotion_weights(args.emotion_state);
    let names: [&'static str; 5] = match version {
        VrmVersion::V1 => ["happy", "angry", "sad", "relaxed", "surprised"],
        // VRM 0.x has no "Surprised" preset; models conventionally add a custom clip of that name.
        VrmVersion::V0 => ["Joy", "Angry", "Sorrow", "Fun", "Surprised"],
    };
    let expressions = names.iter().zip(w).filter(|(_, w)| *w > 0.0).map(|(n, w)| (*n, w)).collect();
    let clip = match args.body_motion {
        BodyMotion::Idle => None,
        BodyMotion::Nod => Some("nod"),
        BodyMotion::LeanForwardPoint => Some("lean_point"),
        BodyMotion::Cheer => Some("cheer"),
        BodyMotion::Clap => Some("clap"),
        BodyMotion::Flinch => Some("flinch"),
        BodyMotion::Think => Some("think"),
        BodyMotion::HideCrouch => Some("crouch"),
        BodyMotion::WaveGoodbye => Some("wave_goodbye"),
    };
    let spine_lean_deg = match args.body_motion {
        BodyMotion::LeanForwardPoint => 12.0,
        BodyMotion::HideCrouch => 20.0,
        BodyMotion::Flinch => -6.0,
        _ => 0.0,
    };
    VrmFrame {
        look: look.unwrap_or_default(),
        expressions,
        clip,
        spine_lean_deg,
    }
}

// ------------------------------------------------------------------------------------------
// Lip-sync
// ------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Viseme {
    A,
    I,
    U,
    E,
    O,
    Rest,
}

impl Viseme {
    pub fn vrm_name(self, v: VrmVersion) -> Option<&'static str> {
        Some(match (self, v) {
            (Viseme::Rest, _) => return None,
            (Viseme::A, VrmVersion::V1) => "aa",
            (Viseme::I, VrmVersion::V1) => "ih",
            (Viseme::U, VrmVersion::V1) => "ou",
            (Viseme::E, VrmVersion::V1) => "ee",
            (Viseme::O, VrmVersion::V1) => "oh",
            (Viseme::A, VrmVersion::V0) => "A",
            (Viseme::I, VrmVersion::V0) => "I",
            (Viseme::U, VrmVersion::V0) => "U",
            (Viseme::E, VrmVersion::V0) => "E",
            (Viseme::O, VrmVersion::V0) => "O",
        })
    }
    /// Cubism `ParamMouthForm` (-1 = rounded/pout, +1 = spread/smile).
    pub fn mouth_form(self) -> f32 {
        match self {
            Viseme::A => 0.0,
            Viseme::I => 0.8,
            Viseme::E => 0.5,
            Viseme::U => -0.8,
            Viseme::O => -0.5,
            Viseme::Rest => 0.0,
        }
    }
}

/// Viseme of one character: Hangul syllables are decomposed to their medial vowel (중성).
pub fn viseme_of(c: char) -> Option<Viseme> {
    let u = c as u32;
    if (0xAC00..=0xD7A3).contains(&u) {
        let jung = ((u - 0xAC00) % (21 * 28)) / 28;
        return Some(match jung {
            0 | 2 | 9 => Viseme::A,                    // ㅏ ㅑ ㅘ
            1 | 3 | 5 | 7 | 10 | 11 | 15 => Viseme::E, // ㅐ ㅒ ㅔ ㅖ ㅙ ㅚ ㅞ
            4 | 6 | 8 | 12 | 14 => Viseme::O,          // ㅓ ㅕ ㅗ ㅛ ㅝ
            13 | 16 | 17 => Viseme::U,                 // ㅜ ㅟ ㅠ
            _ => Viseme::I,                            // ㅡ ㅢ ㅣ
        });
    }
    match c.to_ascii_lowercase() {
        'a' => Some(Viseme::A),
        'e' => Some(Viseme::E),
        'i' | 'y' => Some(Viseme::I),
        'o' => Some(Viseme::O),
        'u' | 'w' => Some(Viseme::U),
        _ => None,
    }
}

/// Evenly distribute the visemes of `text` over `duration_ms`; sentence punctuation inserts a
/// short `Rest`. Returns `(start_ms, viseme)` keyframes.
pub fn viseme_timeline(text: &str, duration_ms: u32) -> Vec<(u32, Viseme)> {
    let mut units: Vec<(Viseme, f32)> = Vec::new();
    for c in text.chars() {
        if let Some(v) = viseme_of(c) {
            // English letters are much shorter than a Hangul syllable.
            let w = if (c as u32) >= 0xAC00 { 1.0 } else { 0.35 };
            units.push((v, w));
        } else if matches!(c, '.' | '!' | '?' | ',' | '…' | '~') {
            units.push((Viseme::Rest, 0.8));
        }
    }
    let total: f32 = units.iter().map(|u| u.1).sum();
    if total == 0.0 {
        return vec![(0, Viseme::Rest)];
    }
    let mut t = 0.0f32;
    let mut out = Vec::with_capacity(units.len() + 1);
    for (v, w) in units {
        let start = (t / total * duration_ms as f32) as u32;
        if out.last().map(|l: &(u32, Viseme)| l.1) != Some(v) {
            out.push((start, v));
        }
        t += w;
    }
    out.push((duration_ms, Viseme::Rest));
    out
}

/// Mouth parameters for the current audio RMS (0..1 normalised) and active viseme.
pub fn live2d_mouth(rms: f32, v: Viseme) -> [(&'static str, f32); 2] {
    let open = if v == Viseme::Rest { 0.0 } else { (rms * 2.2).clamp(0.0, 1.0) };
    [("ParamMouthOpenY", open), ("ParamMouthForm", v.mouth_form())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{GazeTarget, ZenaAvatarArgs};

    #[test]
    fn hangul_visemes() {
        let v: Vec<_> = "아이우에오".chars().map(|c| viseme_of(c).unwrap()).collect();
        assert_eq!(v, vec![Viseme::A, Viseme::I, Viseme::U, Viseme::E, Viseme::O]);
        assert_eq!(viseme_of('봐'), Some(Viseme::A));
        assert_eq!(viseme_of('써'), Some(Viseme::O));
        assert_eq!(viseme_of('!'), None);
    }

    #[test]
    fn timeline_is_monotonic_and_ends_at_rest() {
        let tl = viseme_timeline("조심해! 오른쪽이야.", 1500);
        assert!(tl.windows(2).all(|w| w[0].0 <= w[1].0));
        assert_eq!(tl.last().unwrap(), &(1500, Viseme::Rest));
        assert!(tl.iter().any(|(_, v)| *v == Viseme::Rest && tl.last().unwrap().0 != 0));
    }

    #[test]
    fn gaze_points_toward_event() {
        let screen = ScreenGeometry::new(1920, 1080);
        // Avatar sits bottom-left; an event to the upper-right should give +yaw, +pitch.
        let place = AvatarPlacement {
            head_x: 200.0,
            head_y: 800.0,
            viewer_depth_px: 900.0,
        };
        let l = look_at_event(&screen, &place, Polar { r: 0.3, theta_deg: 45.0 });
        assert!(l.yaw_deg > 10.0 && l.pitch_deg > 5.0, "{l:?}");
        let p = live2d_gaze_params(l);
        assert!(p.iter().all(|(_, v)| v.abs() <= 30.0));
    }

    #[test]
    fn prd_example_maps_to_both_backends() {
        let a = ZenaAvatarArgs {
            gaze_target: Some(GazeTarget {
                angle_deg: 45.0,
                distance_norm: 0.35,
            }),
            emotion_state: EmotionState::SurprisedUrgent,
            body_motion: BodyMotion::LeanForwardPoint,
            speech_text: Some("회피기 써!".into()),
            ..Default::default()
        };
        let l2d = to_live2d(
            &a,
            Some(LookAt {
                yaw_deg: 20.0,
                pitch_deg: 10.0,
            }),
        );
        assert_eq!(l2d.expression, Some("surprised_urgent"));
        assert_eq!(l2d.motion, Some(("Point", 0)));
        let vrm = to_vrm(&a, None, VrmVersion::V1);
        assert!(vrm.expressions.contains(&("surprised", 1.0)));
        assert_eq!(vrm.clip, Some("lean_point"));
        assert_eq!(Viseme::O.vrm_name(VrmVersion::V1), Some("oh"));
        assert_eq!(live2d_mouth(0.3, Viseme::A)[0].1, 0.66);
    }
}
