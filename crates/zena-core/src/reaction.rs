//! Stage 4: reaction-level routing.
//!
//! | level | trigger                                                     | LLM            |
//! |-------|-------------------------------------------------------------|----------------|
//! | Lv1   | far monsters wandering, ordinary movement / farming         | none (0 ms)    |
//! | Lv2   | mid-range fight, own skill hit, hit while on cooldown       | none (cached)  |
//! | Lv3   | close inward threat, sharp HP drop, user natural-language   | Raon-1-nano    |
//!
//! Only events that already survived the NLP-NMS router reach this stage.

use crate::geometry::{FlowDirection, Polar};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReactionLevel {
    /// Gaze / micro-expression only, no speech.
    Lv1Silent = 1,
    /// Pre-cached short voice line + motion.
    Lv2Cached = 2,
    /// Raon-1-nano fast-path generation (tool call + coaching line).
    Lv3Coaching = 3,
}

/// One fused observation for the current tick.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Observation {
    pub polar: Option<Polar>,
    pub flow: Option<FlowDirection>,
    /// `dr/dt` in normalised radius per second (negative = approaching).
    pub radial_velocity: f32,
    /// Normalised event energy 0..1 from the luminance-change sensor.
    pub energy: f32,
    /// Current HP ratio 0..1 if an HP bar ROI is tracked.
    pub hp_ratio: Option<f32>,
    /// HP change per second (negative = losing HP).
    pub hp_rate: Option<f32>,
    /// Zero-shot label from the vision encoder (e.g. "boss"), with its score.
    pub label: Option<String>,
    pub label_score: f32,
    pub own_skill_hit: bool,
    pub hit_during_cooldown: bool,
    /// A user request that survived semantic NMS.
    pub user_query: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReactionConfig {
    /// `r` below this is "close" (근접).
    pub near_r: f32,
    /// `r` below this (and above `near_r`) is "mid-range" (중거리).
    pub mid_r: f32,
    /// HP ratio regarded as critical.
    pub hp_critical: f32,
    /// HP loss per second regarded as a sharp drop (급감), e.g. 0.15 = 15 %/s.
    pub hp_drop_rate: f32,
    /// Minimum energy for a mid-range event to count as combat.
    pub combat_energy: f32,
    /// Minimum spacing of automatic Lv3 generations (user queries are exempt).
    pub min_lv3_interval_ms: u64,
    /// Minimum spacing of Lv2 cues.
    pub min_lv2_interval_ms: u64,
}

impl Default for ReactionConfig {
    fn default() -> Self {
        Self {
            near_r: 0.22,
            mid_r: 0.5,
            hp_critical: 0.3,
            hp_drop_rate: 0.15,
            combat_energy: 0.25,
            min_lv3_interval_ms: 2500,
            min_lv2_interval_ms: 900,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactionDecision {
    pub level: ReactionLevel,
    /// Priority used to arbitrate between simultaneous decisions (higher first).
    pub priority: f32,
    pub reason: &'static str,
    /// The level before rate limiting, when it was downgraded.
    pub downgraded_from: Option<ReactionLevel>,
}

/// Stateful router (keeps the time of the last Lv2/Lv3 reaction for rate limiting).
#[derive(Debug, Clone, Default)]
pub struct ReactionRouter {
    pub cfg: ReactionConfig,
    last_lv3_ms: Option<u64>,
    last_lv2_ms: Option<u64>,
}

impl ReactionRouter {
    pub fn new(cfg: ReactionConfig) -> Self {
        Self {
            cfg,
            last_lv3_ms: None,
            last_lv2_ms: None,
        }
    }

    /// Pure classification without rate limiting.
    pub fn classify(&self, o: &Observation) -> (ReactionLevel, &'static str) {
        let c = &self.cfg;
        if o.user_query.as_deref().is_some_and(|q| !q.trim().is_empty()) {
            return (ReactionLevel::Lv3Coaching, "user natural-language request");
        }
        if let Some(rate) = o.hp_rate {
            if rate <= -c.hp_drop_rate {
                return (ReactionLevel::Lv3Coaching, "sharp HP drop");
            }
        }
        if let (Some(hp), Some(rate)) = (o.hp_ratio, o.hp_rate) {
            if hp <= c.hp_critical && rate < 0.0 {
                return (ReactionLevel::Lv3Coaching, "HP critical and falling");
            }
        }
        let r = o.polar.map(|p| p.r).unwrap_or(1.0);
        let inward = o.flow == Some(FlowDirection::Inward);
        if inward && r < c.near_r {
            return (ReactionLevel::Lv3Coaching, "close inward threat");
        }
        if o.hit_during_cooldown {
            return (ReactionLevel::Lv2Cached, "hit while skills on cooldown");
        }
        if o.own_skill_hit || (o.flow == Some(FlowDirection::Outward) && r < c.mid_r && o.energy >= c.combat_energy) {
            return (ReactionLevel::Lv2Cached, "own skill landed");
        }
        if r < c.mid_r && (inward || o.energy >= c.combat_energy) {
            return (ReactionLevel::Lv2Cached, "mid-range combat");
        }
        (ReactionLevel::Lv1Silent, "ambient movement")
    }

    /// Priority: level dominates; inward threats get a 1.5× boost; closer and more energetic
    /// events rank higher.
    pub fn priority(&self, level: ReactionLevel, o: &Observation) -> f32 {
        let r = o.polar.map(|p| p.r).unwrap_or(1.0).clamp(0.0, 1.0);
        let mut p = (level as u8 as f32) * 10.0 + (1.0 - r) * 3.0 + o.energy.clamp(0.0, 1.0) * 2.0;
        if o.flow == Some(FlowDirection::Inward) {
            p *= 1.5;
        }
        if o.user_query.is_some() {
            p += 100.0;
        }
        p
    }

    /// Classify, then apply rate limits (Lv3→Lv2→Lv1 downgrade). User requests are never limited.
    pub fn route(&mut self, o: &Observation, now_ms: u64) -> ReactionDecision {
        let (level, reason) = self.classify(o);
        let is_user = o.user_query.is_some();
        let mut out = level;
        if out == ReactionLevel::Lv3Coaching && !is_user {
            if let Some(t) = self.last_lv3_ms {
                if now_ms.saturating_sub(t) < self.cfg.min_lv3_interval_ms {
                    out = ReactionLevel::Lv2Cached;
                }
            }
        }
        if out == ReactionLevel::Lv2Cached {
            if let Some(t) = self.last_lv2_ms {
                if now_ms.saturating_sub(t) < self.cfg.min_lv2_interval_ms {
                    out = ReactionLevel::Lv1Silent;
                }
            }
        }
        match out {
            ReactionLevel::Lv3Coaching => self.last_lv3_ms = Some(now_ms),
            ReactionLevel::Lv2Cached => self.last_lv2_ms = Some(now_ms),
            ReactionLevel::Lv1Silent => {}
        }
        ReactionDecision {
            level: out,
            priority: self.priority(out, o),
            reason,
            downgraded_from: (out != level).then_some(level),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(r: f32, flow: FlowDirection, energy: f32) -> Observation {
        Observation {
            polar: Some(Polar { r, theta_deg: 45.0 }),
            flow: Some(flow),
            energy,
            ..Default::default()
        }
    }

    #[test]
    fn table_rows() {
        let r = ReactionRouter::default();
        assert_eq!(r.classify(&obs(0.9, FlowDirection::Tangential, 0.1)).0, ReactionLevel::Lv1Silent);
        assert_eq!(r.classify(&obs(0.35, FlowDirection::Inward, 0.4)).0, ReactionLevel::Lv2Cached);
        assert_eq!(r.classify(&obs(0.1, FlowDirection::Inward, 0.4)).0, ReactionLevel::Lv3Coaching);
        let mut o = obs(0.9, FlowDirection::Static, 0.0);
        o.hp_ratio = Some(0.6);
        o.hp_rate = Some(-0.3);
        assert_eq!(r.classify(&o).0, ReactionLevel::Lv3Coaching);
        let q = Observation {
            user_query: Some("제나야 보스 패턴 어떻게 피해?".into()),
            ..Default::default()
        };
        assert_eq!(r.classify(&q).0, ReactionLevel::Lv3Coaching);
        let mut s = obs(0.3, FlowDirection::Outward, 0.5);
        s.own_skill_hit = true;
        assert_eq!(r.classify(&s).0, ReactionLevel::Lv2Cached);
    }

    #[test]
    fn rate_limit_downgrades_but_not_user_queries() {
        let mut r = ReactionRouter::default();
        let threat = obs(0.1, FlowDirection::Inward, 0.9);
        assert_eq!(r.route(&threat, 0).level, ReactionLevel::Lv3Coaching);
        let d = r.route(&threat, 500);
        assert_eq!(d.level, ReactionLevel::Lv2Cached);
        assert_eq!(d.downgraded_from, Some(ReactionLevel::Lv3Coaching));
        assert_eq!(r.route(&threat, 800).level, ReactionLevel::Lv1Silent);
        let q = Observation {
            user_query: Some("물약 어디 있어?".into()),
            ..Default::default()
        };
        assert_eq!(r.route(&q, 900).level, ReactionLevel::Lv3Coaching);
        assert_eq!(r.route(&threat, 5000).level, ReactionLevel::Lv3Coaching);
    }

    #[test]
    fn inward_gets_priority_boost() {
        let r = ReactionRouter::default();
        let a = obs(0.4, FlowDirection::Inward, 0.5);
        let b = obs(0.4, FlowDirection::Outward, 0.5);
        assert!(r.priority(ReactionLevel::Lv2Cached, &a) > r.priority(ReactionLevel::Lv2Cached, &b));
    }
}
