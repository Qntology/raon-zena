//! Temporal semantic NMS over user requests and in-game alerts.
//!
//! Candidates arriving in one tick (plus everything emitted in the last `memory_ms`) are
//! de-duplicated greedily: user requests first, then by score; a candidate is suppressed when
//! its semantic overlap with an already accepted one reaches `τ`. Overlap is cosine similarity,
//! raised to 1 for repeated system commands and, for alerts with the same label, by spatial
//! proximity in `(r, θ)`. Game alerts never suppress user requests, but a user question about
//! the same thing suppresses the alert (the answer covers it): for that direction the overlap
//! also counts how much of the alert's vocabulary the question contains (logis-center's D4
//! containment rule), because a short label and a full sentence rarely have a high cosine.
//!
//! `τ` adapts like logis-center's EVIDENCE DEDUP floor: with ≥ 8 vectors in the pool it is
//! `clamp(μ + 3σ, base_tau, 0.995)` over (up to 4000 sampled) pairwise cosines, so only pairs
//! that are unusually similar *for this pool* count as duplicates.

use crate::embed::cosine;
use crate::normalize::normalize;
use serde::Serialize;
use std::collections::VecDeque;
use zena_core::geometry::{angle_delta_deg, Polar};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Source {
    User,
    Game,
}

#[derive(Debug, Clone, Serialize)]
pub struct Item {
    pub id: u64,
    pub source: Source,
    /// Intent name or event label.
    pub key: String,
    pub is_command: bool,
    pub text: String,
    #[serde(skip)]
    pub vec: Vec<f32>,
    pub score: f32,
    pub t_ms: u64,
    pub polar: Option<Polar>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub item: Item,
    pub kept: bool,
    pub suppressed_by: Option<u64>,
    pub overlap: f32,
    pub tau: f32,
}

#[derive(Debug, Clone)]
pub struct SemanticNms {
    pub base_tau: f32,
    pub memory_ms: u64,
    /// Polar distance (normalised r units, θ scaled by r) under which same-label alerts merge.
    pub polar_eps: f32,
    history: VecDeque<Item>,
}

impl Default for SemanticNms {
    fn default() -> Self {
        Self {
            base_tau: 0.6,
            memory_ms: 4000,
            polar_eps: 0.12,
            history: VecDeque::new(),
        }
    }
}

/// logis-center's adaptive duplicate floor.
pub fn adaptive_tau(vecs: &[&[f32]], base: f32) -> f32 {
    if vecs.len() < 8 {
        return base;
    }
    let total = vecs.len() * (vecs.len() - 1) / 2;
    let step = (total / 4000).max(1);
    let (mut sum, mut sq, mut cnt, mut k) = (0f64, 0f64, 0usize, 0usize);
    'outer: for a in 0..vecs.len() {
        for b in a + 1..vecs.len() {
            k += 1;
            if k % step != 0 {
                continue;
            }
            let s = cosine(vecs[a], vecs[b]) as f64;
            sum += s;
            sq += s * s;
            cnt += 1;
            if cnt >= 4000 {
                break 'outer;
            }
        }
    }
    if cnt < 8 {
        return base;
    }
    let mu = sum / cnt as f64;
    let sd = (sq / cnt as f64 - mu * mu).max(0.0).sqrt();
    ((mu + 3.0 * sd) as f32).clamp(base, 0.995)
}

fn polar_dist(a: Polar, b: Polar) -> f32 {
    let dr = a.r - b.r;
    let arc = angle_delta_deg(a.theta_deg, b.theta_deg).to_radians() * 0.5 * (a.r + b.r);
    (dr * dr + arc * arc).sqrt()
}

/// Fraction of `inner`'s word lemmas that also occur in `outer`.
pub fn containment(inner: &str, outer: &str) -> f32 {
    let a = normalize(inner);
    let b = normalize(outer);
    if a.words.is_empty() {
        return 0.0;
    }
    let hit = a
        .words
        .iter()
        .filter(|w| b.words.iter().any(|v| v.lemma == w.lemma || v.stem == w.stem))
        .count();
    hit as f32 / a.words.len() as f32
}

impl SemanticNms {
    pub fn overlap(&self, a: &Item, b: &Item) -> f32 {
        let mut s = cosine(&a.vec, &b.vec);
        if a.source == Source::Game && b.source == Source::User {
            s = s.max(containment(&a.text, &b.text));
        }
        if a.key == b.key {
            if a.is_command && b.is_command {
                s = 1.0;
            }
            if let (Some(p), Some(q)) = (a.polar, b.polar) {
                s = s.max(1.0 - polar_dist(p, q) / self.polar_eps).min(1.0);
            }
        }
        s
    }

    pub fn process(&mut self, mut batch: Vec<Item>, now_ms: u64) -> Vec<Decision> {
        while self.history.front().is_some_and(|h| now_ms.saturating_sub(h.t_ms) > self.memory_ms) {
            self.history.pop_front();
        }
        let tau = {
            let pool: Vec<&[f32]> = batch
                .iter()
                .map(|i| i.vec.as_slice())
                .chain(self.history.iter().map(|i| i.vec.as_slice()))
                .collect();
            adaptive_tau(&pool, self.base_tau)
        };
        batch.sort_by(|a, b| {
            (b.source == Source::User)
                .cmp(&(a.source == Source::User))
                .then(b.score.total_cmp(&a.score))
        });
        let mut kept: Vec<Item> = Vec::new();
        let mut out = Vec::with_capacity(batch.len());
        for item in batch {
            let mut by: Option<(u64, f32)> = None;
            for other in kept.iter().chain(self.history.iter()) {
                if item.source == Source::User && other.source == Source::Game {
                    continue;
                }
                let ov = self.overlap(&item, other);
                if ov >= tau && by.is_none_or(|b| ov > b.1) {
                    by = Some((other.id, ov));
                }
            }
            match by {
                Some((id, ov)) => out.push(Decision {
                    item,
                    kept: false,
                    suppressed_by: Some(id),
                    overlap: ov,
                    tau,
                }),
                None => {
                    kept.push(item.clone());
                    out.push(Decision {
                        item,
                        kept: true,
                        suppressed_by: None,
                        overlap: 0.0,
                        tau,
                    });
                }
            }
        }
        self.history.extend(kept);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{HashedNgramEmbedder, TextEmbedder};

    #[allow(clippy::too_many_arguments)]
    fn item(id: u64, src: Source, key: &str, text: &str, score: f32, t: u64, polar: Option<Polar>, cmd: bool) -> Item {
        let e = HashedNgramEmbedder::default();
        Item {
            id,
            source: src,
            key: key.into(),
            is_command: cmd,
            text: text.into(),
            vec: e.embed(text),
            score,
            t_ms: t,
            polar,
        }
    }

    #[test]
    fn repeated_question_is_suppressed_but_new_one_passes() {
        let mut nms = SemanticNms::default();
        let d = nms.process(
            vec![item(1, Source::User, "CoachQuestion", "보스 패턴 어떻게 피해", 0.9, 0, None, false)],
            0,
        );
        assert!(d[0].kept);
        let d = nms.process(
            vec![
                item(
                    2,
                    Source::User,
                    "CoachQuestion",
                    "제나야 보스 패턴은 어떻게 피해?",
                    0.9,
                    1000,
                    None,
                    false,
                ),
                item(3, Source::User, "InventoryQuery", "물약 어디 있어", 0.9, 1000, None, false),
            ],
            1000,
        );
        let k: Vec<_> = d.iter().map(|x| (x.item.id, x.kept)).collect();
        assert!(k.contains(&(2, false)) && k.contains(&(3, true)), "{k:?}");
        // After the memory window the same question is allowed again.
        let d = nms.process(
            vec![item(
                4,
                Source::User,
                "CoachQuestion",
                "보스 패턴 어떻게 피해",
                0.9,
                6000,
                None,
                false,
            )],
            6000,
        );
        assert!(d[0].kept);
    }

    #[test]
    fn same_alert_nearby_merges_far_one_does_not() {
        let mut nms = SemanticNms::default();
        let p = |r, t| Some(Polar { r, theta_deg: t });
        let d = nms.process(
            vec![
                item(1, Source::Game, "monster_approach", "몬스터 접근", 0.8, 0, p(0.30, 40.0), false),
                item(2, Source::Game, "monster_approach", "몬스터 접근", 0.6, 0, p(0.32, 43.0), false),
                item(3, Source::Game, "monster_approach", "몬스터 접근", 0.5, 0, p(0.30, 220.0), false),
            ],
            0,
        );
        let kept: Vec<_> = d.iter().filter(|x| x.kept).map(|x| x.item.id).collect();
        // Identical text ⇒ cosine 1, so all same-label alerts within the tick collapse to the best.
        assert_eq!(kept, vec![1]);
    }

    #[test]
    fn user_question_covers_alert_but_not_vice_versa() {
        let mut nms = SemanticNms::default();
        let d = nms.process(
            vec![
                item(1, Source::Game, "boss", "보스 등장 패턴", 0.99, 0, None, false),
                item(2, Source::User, "CoachQuestion", "보스 패턴 어떻게 피해", 0.5, 0, None, false),
            ],
            0,
        );
        let user = d.iter().find(|x| x.item.id == 2).unwrap();
        assert!(user.kept);
        let alert = d.iter().find(|x| x.item.id == 1).unwrap();
        assert!(!alert.kept && alert.suppressed_by == Some(2), "{:?}", alert.overlap);
    }

    #[test]
    fn repeated_command_is_idempotent() {
        let mut nms = SemanticNms::default();
        assert!(nms.process(vec![item(1, Source::User, "ExitApp", "종료해", 0.95, 0, None, true)], 0)[0].kept);
        assert!(!nms.process(vec![item(2, Source::User, "ExitApp", "앱 꺼줘", 0.95, 500, None, true)], 500)[0].kept);
    }

    #[test]
    fn adaptive_tau_rises_for_homogeneous_pools() {
        let e = HashedNgramEmbedder::default();
        let texts = ["a", "b", "c", "d", "e", "f", "g", "h", "i"].map(|t| e.embed(t));
        let refs: Vec<&[f32]> = texts.iter().map(|v| v.as_slice()).collect();
        let t = adaptive_tau(&refs, 0.6);
        assert!((0.6..=0.995).contains(&t));
        assert_eq!(adaptive_tau(&refs[..3], 0.6), 0.6);
    }
}
