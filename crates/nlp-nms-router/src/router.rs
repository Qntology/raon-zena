//! Glue: utterances + game alerts in, de-duplicated actions out.

use crate::embed::{HashedNgramEmbedder, TextEmbedder};
use crate::intent::{Intent, IntentBank, DEFAULT_BANK};
use crate::normalize::normalize;
use crate::semantic::{Decision, Item, SemanticNms, Source};
use crate::span::span_nms;
use crate::spatial::{select_target, BoxCand};
use zena_core::geometry::Polar;
use zena_core::tools::{SystemCommand, WindowLayerMode};

#[derive(Debug, Clone)]
pub enum RouterInput {
    Utterance {
        text: String,
        t_ms: u64,
    },
    GameEvent {
        label: String,
        score: f32,
        polar: Polar,
        t_ms: u64,
    },
}

#[derive(Debug, Clone, Default)]
pub struct RouterOutput {
    pub system: Vec<(SystemCommand, Option<WindowLayerMode>)>,
    /// Lv3 requests for Raon-1-nano: (intent, full utterance).
    pub queries: Vec<(Intent, String)>,
    /// Alerts that survived NMS: (label, polar, score).
    pub events: Vec<(String, Polar, f32)>,
    pub decisions: Vec<Decision>,
}

pub struct NlpNmsRouter {
    pub bank: IntentBank<HashedNgramEmbedder>,
    pub nms: SemanticNms,
    /// Minimum span score for an intent (span NMS acceptance).
    pub accept: f32,
    next_id: u64,
}

impl Default for NlpNmsRouter {
    fn default() -> Self {
        Self {
            bank: IntentBank::new(HashedNgramEmbedder::default(), DEFAULT_BANK),
            nms: SemanticNms::default(),
            accept: 0.6,
            next_id: 1,
        }
    }
}

impl NlpNmsRouter {
    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn route(&mut self, inputs: Vec<RouterInput>, now_ms: u64) -> RouterOutput {
        let mut items = Vec::new();
        for inp in inputs {
            match inp {
                RouterInput::Utterance { text, t_ms } => {
                    let n = normalize(&text);
                    if n.words.is_empty() {
                        continue;
                    }
                    let hits = span_nms(&n.words, &self.bank, 5, self.accept);
                    let full_vec = self.bank.embedder.embed(&n.text);
                    if hits.is_empty() {
                        if n.addressed {
                            let id = self.id();
                            items.push(Item {
                                id,
                                source: Source::User,
                                key: format!("{:?}", Intent::SmallTalk),
                                is_command: false,
                                text: text.clone(),
                                vec: full_vec,
                                score: 0.5,
                                t_ms,
                                polar: None,
                            });
                        }
                        continue;
                    }
                    for h in hits {
                        let id = self.id();
                        let cmd = h.intent.is_command();
                        // Commands dedupe on their own span; questions on the whole utterance.
                        let (t, v) = if cmd {
                            (h.text.clone(), self.bank.embedder.embed(&h.text))
                        } else {
                            (text.clone(), full_vec.clone())
                        };
                        items.push(Item {
                            id,
                            source: Source::User,
                            key: format!("{:?}", h.intent),
                            is_command: cmd,
                            text: t,
                            vec: v,
                            score: h.score,
                            t_ms,
                            polar: None,
                        });
                    }
                }
                RouterInput::GameEvent { label, score, polar, t_ms } => {
                    let id = self.id();
                    let vec = self.bank.embedder.embed(&label);
                    items.push(Item {
                        id,
                        source: Source::Game,
                        key: label.clone(),
                        is_command: false,
                        text: label,
                        vec,
                        score,
                        t_ms,
                        polar: Some(polar),
                    });
                }
            }
        }
        let decisions = self.nms.process(items, now_ms);
        let mut out = RouterOutput::default();
        for d in &decisions {
            if !d.kept {
                continue;
            }
            match d.item.source {
                Source::Game => out.events.push((d.item.key.clone(), d.item.polar.unwrap(), d.item.score)),
                Source::User => {
                    let intent = intent_from_key(&d.item.key);
                    match intent.and_then(Intent::system_command) {
                        Some(c) => out.system.push(c),
                        None => out.queries.push((intent.unwrap_or(Intent::SmallTalk), d.item.text.clone())),
                    }
                }
            }
        }
        out.decisions = decisions;
        out
    }

    /// Spatial-text NMS: which detected region does `request` refer to?
    pub fn resolve_target(&self, request: &str, boxes: &mut [BoxCand]) -> Option<(usize, Polar)> {
        let q = self.bank.embedder.embed(request);
        select_target(boxes, &q, 0.2)
    }
}

fn intent_from_key(k: &str) -> Option<Intent> {
    use Intent::*;
    [
        ExitApp,
        HideAvatar,
        ShowAvatar,
        TopmostOn,
        TopmostOff,
        InteractiveMode,
        GhostMode,
        CoachQuestion,
        InventoryQuery,
        SmallTalk,
    ]
    .into_iter()
    .find(|i| format!("{i:?}") == k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_burst_is_deduplicated() {
        let mut r = NlpNmsRouter::default();
        let p = |r, t| Polar { r, theta_deg: t };
        let out = r.route(
            vec![
                RouterInput::Utterance {
                    text: "제나야 방금 그 보스 패턴 어떻게 피해?".into(),
                    t_ms: 0,
                },
                RouterInput::Utterance {
                    text: "보스 패턴 어떻게 피해".into(),
                    t_ms: 10,
                },
                RouterInput::GameEvent {
                    label: "보스 패턴".into(),
                    score: 0.9,
                    polar: p(0.15, 30.0),
                    t_ms: 0,
                },
                RouterInput::GameEvent {
                    label: "몬스터 접근".into(),
                    score: 0.7,
                    polar: p(0.6, 200.0),
                    t_ms: 0,
                },
                RouterInput::GameEvent {
                    label: "몬스터 접근".into(),
                    score: 0.6,
                    polar: p(0.61, 201.0),
                    t_ms: 0,
                },
            ],
            20,
        );
        assert_eq!(out.queries.len(), 1, "{:?}", out.queries);
        assert_eq!(out.queries[0].0, Intent::CoachQuestion);
        let labels: Vec<_> = out.events.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(
            labels,
            vec!["몬스터 접근"],
            "boss alert is covered by the question; duplicate approach merged"
        );
    }

    #[test]
    fn commands_route_to_system_and_repeat_is_ignored() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 오늘 그만 켜둘게, 종료해 줘".into(),
                t_ms: 0,
            }],
            0,
        );
        assert_eq!(out.system, vec![(SystemCommand::ExitApp, None)]);
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "종료해".into(),
                t_ms: 800,
            }],
            800,
        );
        assert!(out.system.is_empty());
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "잠깐 숨어있어".into(),
                t_ms: 900,
            }],
            900,
        );
        assert_eq!(out.system, vec![(SystemCommand::HideAvatar, None)]);
    }

    #[test]
    fn small_talk_when_addressed_without_intent() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 오늘 기분 어때".into(),
                t_ms: 0,
            }],
            0,
        );
        assert_eq!(out.queries.first().map(|q| q.0), Some(Intent::SmallTalk));
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "음 그렇구나".into(),
                t_ms: 5000,
            }],
            5000,
        );
        assert!(out.queries.is_empty() && out.system.is_empty());
    }
}
