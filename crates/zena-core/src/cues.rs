//! Lv2 pre-cached cues: short lines whose audio is synthesised once at start-up and replayed
//! with no LLM call.

use crate::tools::{BodyMotion, EmotionState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CueKind {
    NiceHit,
    Careful,
    HitOnCooldown,
    MidRangeFight,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cue {
    pub kind: CueKind,
    /// Alternatives rotated round-robin so repeats do not sound robotic.
    pub lines: &'static [&'static str],
    pub emotion: EmotionState,
    pub motion: BodyMotion,
}

pub const CUES: &[Cue] = &[
    Cue {
        kind: CueKind::NiceHit,
        lines: &["나이스!", "좋았어!", "명중!"],
        emotion: EmotionState::Joy,
        motion: BodyMotion::Cheer,
    },
    Cue {
        kind: CueKind::Careful,
        lines: &["조심!", "온다!", "피해!"],
        emotion: EmotionState::Surprised,
        motion: BodyMotion::Flinch,
    },
    Cue {
        kind: CueKind::HitOnCooldown,
        lines: &["아야, 버텨!", "쿨 돌 때까지만!"],
        emotion: EmotionState::Worried,
        motion: BodyMotion::Flinch,
    },
    Cue {
        kind: CueKind::MidRangeFight,
        lines: &["간다!", "집중!"],
        emotion: EmotionState::Focused,
        motion: BodyMotion::Nod,
    },
];

pub fn cue(kind: CueKind) -> &'static Cue {
    CUES.iter().find(|c| c.kind == kind).expect("every CueKind has an entry")
}

/// Round-robin line picker.
#[derive(Debug, Default, Clone)]
pub struct CuePlayer {
    counters: [usize; 4],
}

impl CuePlayer {
    pub fn next_line(&mut self, kind: CueKind) -> &'static str {
        let c = cue(kind);
        let slot = &mut self.counters[kind as usize];
        let line = c.lines[*slot % c.lines.len()];
        *slot += 1;
        line
    }
}

/// Pick the cue for an Lv2 decision reason (see `reaction::ReactionRouter::classify`).
pub fn cue_for_reason(reason: &str) -> CueKind {
    match reason {
        "own skill landed" => CueKind::NiceHit,
        "hit while skills on cooldown" => CueKind::HitOnCooldown,
        "close inward threat" | "sharp HP drop" | "HP critical and falling" => CueKind::Careful,
        _ => CueKind::MidRangeFight,
    }
}

/// Phase 7 Part 2: 청각 장애 · 난청 사용자를 위한 화면 가장자리 주변시 호(Peripheral Visual Arc) + 햅틱 패턴.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeSide {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HapticPattern {
    SinglePulse,
    DoubleUrgent,
    SoftHeartbeat,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdgeVisualArc {
    pub side: EdgeSide,
    /// 가장자리 축 상의 위치 비율 `[0.0, 1.0]`.
    pub offset_ratio: f32,
    /// 발광 강도 `[0.25, 1.0]`.
    pub intensity: f32,
    pub haptic: HapticPattern,
    pub text_cue: &'static str,
}

/// 극좌표 각도 `theta_deg`(0° 우, 90° 상) 와 반경 `r` 을 화면 가장자리 시각 호와 햅틱 패턴으로 변환한다.
pub fn peripheral_visual_arc(theta_deg: f32, r: f32, urgent: bool) -> EdgeVisualArc {
    let a = theta_deg.rem_euclid(360.0);
    let rad = a.to_radians();
    let (side, offset_ratio) = if !(45.0..315.0).contains(&a) {
        (EdgeSide::Right, (0.5 - 0.5 * rad.sin()).clamp(0.0, 1.0))
    } else if (45.0..135.0).contains(&a) {
        (EdgeSide::Top, (0.5 + 0.5 * rad.cos()).clamp(0.0, 1.0))
    } else if (135.0..225.0).contains(&a) {
        (EdgeSide::Left, (0.5 - 0.5 * rad.sin()).clamp(0.0, 1.0))
    } else {
        (EdgeSide::Bottom, (0.5 + 0.5 * rad.cos()).clamp(0.0, 1.0))
    };
    let intensity = if urgent { 1.0 } else { (0.35 + 0.55 * r.clamp(0.0, 1.0)).clamp(0.25, 1.0) };
    let haptic = if urgent {
        HapticPattern::DoubleUrgent
    } else if r >= 0.7 {
        HapticPattern::SinglePulse
    } else {
        HapticPattern::SoftHeartbeat
    };
    let text_cue = match (side, urgent) {
        (EdgeSide::Left, true) => "왼쪽 위험!",
        (EdgeSide::Right, true) => "오른쪽 위험!",
        (EdgeSide::Top, true) => "위쪽 주의!",
        (EdgeSide::Bottom, true) => "아래쪽 주의!",
        (EdgeSide::Left, false) => "왼쪽 안내",
        (EdgeSide::Right, false) => "오른쪽 안내",
        (EdgeSide::Top, false) => "위쪽 안내",
        (EdgeSide::Bottom, false) => "아래쪽 안내",
    };
    EdgeVisualArc {
        side,
        offset_ratio,
        intensity,
        haptic,
        text_cue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_robin() {
        let mut p = CuePlayer::default();
        assert_eq!(p.next_line(CueKind::NiceHit), "나이스!");
        assert_eq!(p.next_line(CueKind::NiceHit), "좋았어!");
        assert_eq!(p.next_line(CueKind::Careful), "조심!");
        assert_eq!(cue_for_reason("own skill landed"), CueKind::NiceHit);
        let arc = peripheral_visual_arc(180.0, 0.9, true);
        assert_eq!(arc.side, EdgeSide::Left);
        assert_eq!(arc.haptic, HapticPattern::DoubleUrgent);
        assert_eq!(arc.text_cue, "왼쪽 위험!");
    }
}
