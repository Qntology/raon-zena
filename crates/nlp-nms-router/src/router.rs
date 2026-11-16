//! Glue: utterances + game alerts in, de-duplicated actions out.
//!
//! The user's words are caught into the expression domains of the word dictionary (`zena-lexicon`); each kept
//! domain becomes what its node says: a system command, a dictionary expression Zena plays at once (no LLM), or a
//! request for Raon-1-nano.

use crate::domain::{DomainBank, Hit};
use crate::embed::{HashedNgramEmbedder, TextEmbedder};
use crate::normalize::normalize;
use crate::semantic::{Decision, Item, SemanticNms, Source};
use crate::span::span_nms;
use crate::spatial::{select_target, BoxCand};
use std::collections::HashMap;
use zena_core::geometry::Polar;
use zena_core::tools::{AssetAction, SystemCommand, WindowLayerMode, ZenaAssetArgs};
use zena_lexicon::{Action, Lexicon};

/// The dictionary node an utterance is answered with when Zena is addressed but no domain is caught.
pub const SMALL_TALK: &str = "chitchat.small_talk";

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

/// A domain caught in an utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Caught {
    /// Dictionary key (`group.name`).
    pub key: String,
    pub action: Action,
    /// The words that named the domain.
    pub span: String,
    /// The whole utterance.
    pub utterance: String,
    pub score: f32,
    pub hit: Hit,
}

#[derive(Debug, Clone, Default)]
pub struct RouterOutput {
    pub system: Vec<(SystemCommand, Option<WindowLayerMode>)>,
    /// Avatar asset commands (track switches, accessories), in the order they were said.
    pub assets: Vec<ZenaAssetArgs>,
    /// Dictionary expressions Zena plays without the LLM, best first (a command's own expression included).
    pub expressions: Vec<Caught>,
    /// Requests for Raon-1-nano (Lv3), best first.
    pub queries: Vec<Caught>,
    /// Alerts that survived NMS: (label, polar, score).
    pub events: Vec<(String, Polar, f32)>,
    pub decisions: Vec<Decision>,
}

pub struct NlpNmsRouter {
    pub bank: DomainBank<HashedNgramEmbedder>,
    pub nms: SemanticNms,
    /// Minimum span score for a domain (span NMS acceptance).
    pub accept: f32,
    next_id: u64,
}

impl Default for NlpNmsRouter {
    fn default() -> Self {
        Self::new(&Lexicon::bundled())
    }
}

impl NlpNmsRouter {
    pub fn new(lex: &Lexicon) -> Self {
        Self {
            bank: DomainBank::from_lexicon(lex, HashedNgramEmbedder::default()),
            nms: SemanticNms::default(),
            accept: 0.6,
            next_id: 1,
        }
    }

    /// Swaps in an updated dictionary; the NMS memory of recent requests stays.
    pub fn reload(&mut self, lex: &Lexicon) {
        self.bank = DomainBank::from_lexicon(lex, HashedNgramEmbedder::default());
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub fn route(&mut self, inputs: Vec<RouterInput>, now_ms: u64) -> RouterOutput {
        let mut items = Vec::new();
        let mut caught: HashMap<u64, Caught> = HashMap::new();
        for inp in inputs {
            match inp {
                RouterInput::Utterance { text, t_ms } => {
                    let n = normalize(&text);
                    if n.words.is_empty() {
                        continue;
                    }
                    let mut hits: Vec<(usize, String, f32, Hit)> = span_nms(&n.words, &self.bank, 5, self.accept)
                        .into_iter()
                        .map(|h| (h.index, h.text, h.score, h.hit))
                        .collect();
                    if hits.is_empty() && n.addressed {
                        if let Some(i) = self.bank.index_of(SMALL_TALK) {
                            hits.push((i, n.text.clone(), 0.5, Hit::Vector));
                        }
                    }
                    let full_vec = self.bank.embedder.embed(&n.text);
                    for (index, span, score, hit) in hits {
                        let id = self.id();
                        let e = self.bank.entry(index);
                        // Commands and expressions dedupe on their own words; questions on the whole utterance.
                        let ask = matches!(e.action, Action::Ask { .. });
                        let (t, v) = if ask {
                            (text.clone(), full_vec.clone())
                        } else {
                            (span.clone(), self.bank.embedder.embed(&span))
                        };
                        caught.insert(
                            id,
                            Caught {
                                key: e.key.clone(),
                                action: e.action.clone(),
                                span,
                                utterance: text.clone(),
                                score,
                                hit,
                            },
                        );
                        items.push(Item {
                            id,
                            source: Source::User,
                            key: e.key.clone(),
                            is_command: e.action.is_command(),
                            text: t,
                            vec: v,
                            score,
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
        // Utterances that gave a command: they are not also a question unless one was caught exactly.
        let mut commanding: Vec<String> = Vec::new();
        for d in decisions.iter().filter(|d| d.kept) {
            match d.item.source {
                Source::Game => out.events.push((d.item.key.clone(), d.item.polar.unwrap(), d.item.score)),
                Source::User => {
                    let Some(c) = caught.remove(&d.item.id) else { continue };
                    match &c.action {
                        Action::System { command, mode, express } => {
                            out.system.push((*command, *mode));
                            commanding.push(c.utterance.clone());
                            if express.is_some() {
                                out.expressions.push(c);
                            }
                        }
                        Action::Asset { asset, express } => {
                            if !out.assets.contains(asset) {
                                out.assets.push(asset.clone());
                            }
                            commanding.push(c.utterance.clone());
                            if express.is_some() {
                                out.expressions.push(c);
                            }
                        }
                        Action::Express(_) => out.expressions.push(c),
                        Action::Ask { .. } => out.queries.push(c),
                        Action::None => {}
                    }
                }
            }
        }
        out.queries.retain(|q| !(q.hit == Hit::Vector && commanding.contains(&q.utterance)));
        // "안경 벗어" names the part and the removal: taking off wins over putting on.
        let removing = |a: &ZenaAssetArgs| matches!(a.action, AssetAction::RemovePart | AssetAction::ResetParts);
        if out.assets.iter().any(removing) {
            out.assets.retain(|a| a.action != AssetAction::EquipPart);
            out.expressions
                .retain(|c| !matches!(&c.action, Action::Asset { asset, .. } if asset.action == AssetAction::EquipPart));
        }
        out.expressions.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.queries.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.decisions = decisions;
        out
    }

    /// Spatial-text NMS: which detected region does `request` refer to?
    pub fn resolve_target(&self, request: &str, boxes: &mut [BoxCand]) -> Option<(usize, Polar)> {
        let q = self.bank.embedder.embed(request);
        select_target(boxes, &q, 0.2)
    }
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
        assert_eq!(out.queries[0].key, "coach.game_tip");
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
    fn avatar_asset_requests_route_to_assets() {
        use zena_core::tools::{AvatarPart, AvatarTrack};
        let say = |r: &mut NlpNmsRouter, text: &str, t: u64| {
            r.route(
                vec![RouterInput::Utterance {
                    text: text.into(),
                    t_ms: t,
                }],
                t,
            )
        };
        let mut r = NlpNmsRouter::default();
        let out = say(&mut r, "제나야, 로봇 모드로 변신해 줘", 0);
        assert_eq!(out.assets.len(), 1, "{:?}", out.decisions);
        assert_eq!(
            (out.assets[0].action, out.assets[0].track),
            (AssetAction::SwitchTrack, Some(AvatarTrack::Solidcraft))
        );
        assert!(out.expressions.iter().any(|c| c.key == "avatar.robot_mode"), "plays its line too");
        let out = say(&mut r, "안경이랑 헤드셋 써 봐", 5_000);
        let mut parts: Vec<_> = out.assets.iter().filter_map(|a| a.part).collect();
        parts.sort();
        assert_eq!(parts, vec![AvatarPart::Headset, AvatarPart::Glasses], "{:?}", out.assets);
        assert!(
            out.queries.is_empty(),
            "a command is not also a question for the LLM: {:?}",
            out.queries.iter().map(|q| (&q.key, &q.span, q.score, q.hit)).collect::<Vec<_>>()
        );
        // The same command amid game alerts (as in the live pipeline) is still no question.
        let out = r.route(
            vec![
                RouterInput::GameEvent {
                    label: "근접 위협".into(),
                    score: 0.9,
                    polar: Polar { r: 0.1, theta_deg: 30.0 },
                    t_ms: 7_000,
                },
                RouterInput::Utterance {
                    text: "제나야 리본 달아 봐".into(),
                    t_ms: 7_000,
                },
            ],
            7_000,
        );
        assert_eq!(
            out.assets.iter().filter_map(|a| a.part).collect::<Vec<_>>(),
            vec![AvatarPart::Ribbon]
        );
        assert!(
            out.queries.is_empty(),
            "{:?}",
            out.queries.iter().map(|q| &q.key).collect::<Vec<_>>()
        );
        let out = say(&mut r, "안경 벗어", 10_000);
        assert_eq!(
            out.assets.iter().map(|a| a.action).collect::<Vec<_>>(),
            vec![AssetAction::ResetParts],
            "taking off wins"
        );
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
        assert_eq!(out.queries.first().map(|q| q.key.as_str()), Some(SMALL_TALK));
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "음 그렇구나".into(),
                t_ms: 5000,
            }],
            5000,
        );
        assert!(out.queries.is_empty() && out.system.is_empty());
    }

    #[test]
    fn feelings_are_caught_into_dictionary_expressions() {
        use zena_core::tools::{BodyMotion, EmotionState};
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "와 보스 잡았다! 이겼어!".into(),
                t_ms: 0,
            }],
            0,
        );
        let e = out.expressions.first().expect("an expression");
        assert_eq!(e.key, "emotion.celebrate");
        let x = e.action.express().unwrap();
        assert_eq!((x.emotion, x.motion), (EmotionState::Excited, BodyMotion::Cheer));
        assert!(out.queries.is_empty(), "no LLM for a feeling the dictionary covers");
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "아 또 죽었어 짜증나".into(),
                t_ms: 5000,
            }],
            5000,
        );
        assert_eq!(out.expressions.first().map(|c| c.key.as_str()), Some("emotion.frustrated"));
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "오늘 주식 떡락했어".into(),
                t_ms: 9000,
            }],
            9000,
        );
        assert_eq!(out.expressions.first().map(|c| c.key.as_str()), Some("market.stock_down"));
        // The same thanks twice in a row is said once (semantic NMS over the span).
        let a = r.route(
            vec![RouterInput::Utterance {
                text: "고마워".into(),
                t_ms: 12000,
            }],
            12000,
        );
        let b = r.route(
            vec![RouterInput::Utterance {
                text: "고마워".into(),
                t_ms: 12500,
            }],
            12500,
        );
        assert_eq!(a.expressions.len(), 1);
        assert!(b.expressions.is_empty());
        // Quitting waves goodbye before the command.
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 종료해 줘".into(),
                t_ms: 20000,
            }],
            20000,
        );
        assert_eq!(out.system, vec![(SystemCommand::ExitApp, None)]);
        assert_eq!(
            out.expressions.first().and_then(|c| c.action.express()).map(|x| x.motion),
            Some(BodyMotion::WaveGoodbye)
        );
    }

    #[test]
    fn a_reloaded_dictionary_catches_new_words() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "개꿀잼".into(),
                t_ms: 0,
            }],
            0,
        );
        assert!(out.expressions.is_empty());
        let mut lex = Lexicon::bundled();
        let i = lex.nodes.iter().position(|n| n.key == "emotion.celebrate").unwrap();
        lex.nodes[i].exact_match.push("개꿀잼".into());
        r.reload(&lex);
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "개꿀잼".into(),
                t_ms: 100,
            }],
            100,
        );
        assert_eq!(
            out.expressions.first().map(|c| (c.key.as_str(), c.hit)),
            Some(("emotion.celebrate", Hit::Exact))
        );
    }
}
