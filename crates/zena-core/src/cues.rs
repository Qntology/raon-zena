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
    }
}
