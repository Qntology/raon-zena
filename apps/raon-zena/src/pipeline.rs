//! Worker thread: capture → event sensor → tracker → NLP-NMS router (word dictionary) → reaction routing →
//! dictionary expressions / (Lv3) generation → avatar commands for the UI thread. Every few seconds a downscaled
//! frame also goes to the screen watcher, whose observations come back as dictionary expressions too.

use crate::capture::FrameSource;
use crate::lexicon::LexiconMsg;
use crate::llm::{LlmJob, LlmWorker};
use crate::watcher::{ScreenFrame, ScreenObservation};
use nlp_nms_router::{Caught, Hit, NlpNmsRouter, RouterInput};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::Instant;
use zena_core::avatar::viseme_timeline;
use zena_core::cues::{cue, cue_for_reason, CuePlayer};
use zena_core::geometry::{FlowDirection, ScreenGeometry};
use zena_core::prompt::{build_fast_path_prompt, Situation};
use zena_core::reaction::{Observation, ReactionLevel, ReactionRouter};
use zena_core::tools::{
    parse_model_output, BodyMotion, EmotionState, GazeTarget, SystemCommand, ToolCall, WindowLayerMode, ZenaAvatarArgs,
};
use zena_lexicon::{Action, Express, ExpressLevel, Lexicon, VisionKind};
use zena_sensor::naflex::bgra_to_rgb;
use zena_sensor::{extract_rois, LogLumSensor, RoiTracker};

/// What the UI thread is asked to do.
#[derive(Debug, Clone)]
pub enum AvatarCmd {
    /// Lv1: look toward an event, no speech.
    Gaze(GazeTarget),
    /// Lv2/Lv3: full expression with optional speech and its viseme timeline.
    Act {
        args: ZenaAvatarArgs,
        level: ReactionLevel,
        visemes: Vec<(u32, zena_core::avatar::Viseme)>,
        speech_ms: u32,
    },
    System(SystemCommand, Option<WindowLayerMode>),
}

pub struct PipelineConfig {
    pub sensor_threshold: f32,
    pub min_tile_events: u32,
    /// How often a frame goes to the screen watcher (it skips unchanged frames itself).
    pub screen_interval_ms: u64,
    /// Learn spans caught by vectors alone with at least this score (`None`: no learning).
    pub learn_min: Option<f32>,
    /// Minimum spacing of two dictionary expressions of the same domain.
    pub expression_gap_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sensor_threshold: 0.15,
            min_tile_events: 6,
            screen_interval_ms: 3000,
            learn_min: None,
            expression_gap_ms: 1500,
        }
    }
}

/// The channels the pipeline talks through besides the frame source.
pub struct Links {
    pub llm: LlmWorker,
    pub utterances: Receiver<String>,
    pub out: Sender<AvatarCmd>,
    /// Dictionary revisions (hot reload).
    pub lexicon: Receiver<Lexicon>,
    pub learn: Option<Sender<LexiconMsg>>,
    /// Frames for the screen watcher and what it saw.
    pub screen_frames: Option<SyncSender<ScreenFrame>>,
    pub screen: Option<Receiver<ScreenObservation>>,
}

/// Rough speaking time for Korean TTS (≈ 7 syllables per second) until real TTS durations exist.
fn speech_ms(text: &str) -> u32 {
    (text.chars().filter(|c| !c.is_whitespace()).count() as u32 * 140).max(400)
}

/// Plays dictionary expressions: lines rotate per domain, the same domain is not repeated within `gap_ms`.
#[derive(Default)]
pub struct ExpressionPlayer {
    next_line: HashMap<String, usize>,
    last_ms: HashMap<String, u64>,
}

impl ExpressionPlayer {
    pub fn play(&mut self, key: &str, e: &Express, gaze: Option<GazeTarget>, now: u64, gap_ms: u64) -> Option<AvatarCmd> {
        if self.last_ms.get(key).is_some_and(|&t| now.saturating_sub(t) < gap_ms) {
            return None;
        }
        self.last_ms.insert(key.to_string(), now);
        let line = (e.level == ExpressLevel::Lv2 && !e.lines.is_empty()).then(|| {
            let i = self.next_line.entry(key.to_string()).or_default();
            let l = e.lines[*i % e.lines.len()].clone();
            *i += 1;
            l
        });
        let (level, ms, visemes) = match &line {
            Some(l) => {
                let ms = speech_ms(l);
                (ReactionLevel::Lv2Cached, ms, viseme_timeline(l, ms))
            }
            None => (ReactionLevel::Lv1Silent, 0, Vec::new()),
        };
        Some(AvatarCmd::Act {
            args: ZenaAvatarArgs {
                gaze_target: gaze,
                emotion_state: e.emotion,
                body_motion: e.motion,
                speech_text: line,
            },
            level,
            visemes,
            speech_ms: ms,
        })
    }
}

/// When a screen observation is worth reacting to: a scene when it changes, an event or a private screen at most
/// once per its cooldown.
#[derive(Default)]
pub struct ScreenGate {
    scene: Option<String>,
    last_ms: HashMap<String, u64>,
}

impl ScreenGate {
    pub fn admit(&mut self, o: &ScreenObservation, now: u64) -> bool {
        let recent = self
            .last_ms
            .get(&o.key)
            .is_some_and(|&t| now.saturating_sub(t) < o.cooldown_s as u64 * 1000);
        let ok = match o.kind {
            VisionKind::Scene => {
                let changed = self.scene.as_deref() != Some(o.key.as_str());
                self.scene = Some(o.key.clone());
                changed && !recent
            }
            VisionKind::Event | VisionKind::Privacy => !recent,
        };
        if ok {
            self.last_ms.insert(o.key.clone(), now);
        }
        ok
    }
}

/// Extra lines the LLM gets for a request topic.
fn topic_context(c: &Caught, lex: &Lexicon, screen: Option<&(String, f32)>) -> Vec<String> {
    let mut out = Vec::new();
    if let Action::Ask { topic, .. } = &c.action {
        if topic == "screen" || topic == "market" {
            if let Some((key, score)) = screen {
                let what = lex.vision_node(key).map(|v| v.semantic.clone()).unwrap_or_else(|| key.clone());
                out.push(format!("지금 화면: {what} (확신도 {:.0}%)", score * 100.0));
            } else {
                out.push("지금 화면은 아직 모른다.".into());
            }
        }
        if topic == "market" {
            out.push("투자 조언(사라/팔아라)은 하지 말고 플레이어의 기분에 공감한다.".into());
        }
    }
    out
}

pub fn run(mut src: Box<dyn FrameSource>, links: Links, mut lexicon: Lexicon, cfg: PipelineConfig) -> anyhow::Result<()> {
    let Links {
        llm,
        utterances,
        out,
        lexicon: reloads,
        learn,
        screen_frames,
        screen,
    } = links;
    let (w, h) = src.size();
    let geo = ScreenGeometry::new(w as u32, h as u32);
    let mut sensor = LogLumSensor::new(w, h, cfg.sensor_threshold);
    let mut tracker = RoiTracker::default();
    let mut router = NlpNmsRouter::new(&lexicon);
    let mut reactions = ReactionRouter::default();
    let mut cues = CuePlayer::default();
    let mut expressions = ExpressionPlayer::default();
    let mut gate = ScreenGate::default();
    let mut last_screen: Option<(String, f32)> = None;
    let mut last_frame_sent: Option<u64> = None;
    let t0 = Instant::now();
    eprintln!(
        "pipeline: {w}x{h} sensor, dictionary revision {} ({} domains), Lv3 backend = {}, screen watcher {}",
        lexicon.version,
        router.bank.len(),
        llm.name,
        if screen_frames.is_some() { "on" } else { "off" }
    );
    loop {
        let Some(frame) = src.next_frame()? else {
            continue;
        };
        if (frame.width, frame.height) != (w, h) {
            anyhow::bail!("frame size changed {}x{} -> {}x{}", w, h, frame.width, frame.height);
        }
        let now = t0.elapsed().as_millis() as u64;

        // A new dictionary revision (edited with `zena-lexicon`, or learned).
        for lex in reloads.try_iter() {
            router.reload(&lex);
            eprintln!("[lexicon] router now on revision {} ({} domains)", lex.version, router.bank.len());
            lexicon = lex;
        }

        // Periodic screen sampling: one RGB copy every interval, dropped if the watcher is still busy.
        if let Some(tx) = &screen_frames {
            if last_frame_sent.is_none_or(|t| now.saturating_sub(t) >= cfg.screen_interval_ms) {
                last_frame_sent = Some(now);
                let _ = tx.try_send(ScreenFrame {
                    rgb: bgra_to_rgb(&frame.bgra),
                    w,
                    h,
                });
            }
        }
        if let Some(rx) = &screen {
            for o in rx.try_iter() {
                last_screen = (o.kind != VisionKind::Privacy).then(|| (o.key.clone(), o.score));
                if !gate.admit(&o, now) {
                    continue;
                }
                let Some(e) = lexicon.vision_express(&o.key).cloned() else {
                    continue;
                };
                let gaze = match (o.kind, o.peak) {
                    (VisionKind::Privacy, _) | (_, None) => None,
                    (_, Some((x, y))) => {
                        let p = geo.to_polar(x * w as f32, y * h as f32);
                        Some(GazeTarget {
                            angle_deg: p.theta_deg,
                            distance_norm: p.r,
                        })
                    }
                };
                eprintln!(
                    "[screen] {} ({:.2}, {} of {} tokens, {:.0} ms) → {}",
                    o.key, o.score, o.tokens, o.cells, o.encode_ms, o.link
                );
                if let Some(cmd) = expressions.play(&o.link, &e, gaze, now, cfg.expression_gap_ms) {
                    out.send(cmd)?;
                }
            }
        }

        let stats = sensor.process_bgra(&frame.bgra);
        let rois = extract_rois(&stats, &geo, cfg.min_tile_events);
        let tracks = tracker.update(rois, now).to_vec();
        let max_energy = tracks.iter().map(|t| t.roi.energy).fold(1.0f32, f32::max);

        let mut inputs: Vec<RouterInput> = utterances
            .try_iter()
            .map(|text| RouterInput::Utterance { text, t_ms: now })
            .collect();
        for t in tracks.iter().filter(|t| t.age >= 1) {
            let label = match (t.flow, t.roi.polar.r) {
                (FlowDirection::Inward, r) if r < 0.22 => "근접 위협",
                (FlowDirection::Inward, _) => "몬스터 접근",
                (FlowDirection::Outward, _) => "스킬 시전",
                _ => "움직임",
            };
            inputs.push(RouterInput::GameEvent {
                label: label.into(),
                score: t.roi.energy / max_energy,
                polar: t.roi.polar,
                t_ms: now,
            });
        }
        let routed = router.route(inputs, now);

        // Learning: spans the dictionary caught by vectors alone.
        if let (Some(tx), Some(min)) = (&learn, cfg.learn_min) {
            for c in routed
                .expressions
                .iter()
                .chain(&routed.queries)
                .filter(|c| c.hit == Hit::Vector && c.score >= min)
            {
                let _ = tx.send(LexiconMsg::Learn {
                    key: c.key.clone(),
                    span: c.span.clone(),
                });
            }
        }
        // Dictionary expressions first, then the commands (quitting plays the overlay's own farewell sequence).
        if let Some(c) = routed.expressions.iter().find(|c| !matches!(c.action, Action::System { command: SystemCommand::ExitApp, .. })) {
            if let Some(e) = c.action.express() {
                if let Some(cmd) = expressions.play(&c.key, e, None, now, cfg.expression_gap_ms) {
                    out.send(cmd)?;
                }
            }
        }
        for (c, m) in routed.system {
            out.send(AvatarCmd::System(c, m))?;
        }

        // Pick the strongest surviving signal for this tick.
        let mut obs = Observation::default();
        let mut label = None;
        let query = routed.queries.first().cloned();
        if let Some(q) = &query {
            obs.user_query = Some(q.utterance.clone());
            // Acknowledge at once (e.g. a thinking pose) while Raon-1-nano answers.
            if let Some(e) = q.action.express() {
                if let Some(cmd) = expressions.play(&q.key, e, None, now, cfg.expression_gap_ms) {
                    out.send(cmd)?;
                }
            }
        }
        if let Some((l, p, s)) = routed.events.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
            let tr = tracks.iter().min_by(|a, b| {
                let da = (a.roi.polar.r - p.r).abs() + (a.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                let db = (b.roi.polar.r - p.r).abs() + (b.roi.polar.theta_deg - p.theta_deg).abs() / 360.0;
                da.total_cmp(&db)
            });
            obs.polar = Some(*p);
            obs.energy = *s;
            obs.flow = tr.map(|t| t.flow);
            obs.radial_velocity = tr.map(|t| t.radial_velocity).unwrap_or(0.0);
            label = Some(l.clone());
        }
        if obs.user_query.is_none() && obs.polar.is_none() {
            continue;
        }
        let d = reactions.route(&obs, now);
        match d.level {
            ReactionLevel::Lv1Silent => {
                if let Some(p) = obs.polar {
                    out.send(AvatarCmd::Gaze(GazeTarget {
                        angle_deg: p.theta_deg,
                        distance_norm: p.r,
                    }))?;
                }
            }
            ReactionLevel::Lv2Cached => {
                out.send(cached_cue(&mut cues, d.reason, obs.polar, d.level))?;
            }
            ReactionLevel::Lv3Coaching => {
                let mut context = Vec::new();
                if let Some(q) = &query {
                    context = topic_context(q, &lexicon, last_screen.as_ref());
                }
                let sit = Situation {
                    event_label: label,
                    polar: obs.polar,
                    flow: obs.flow,
                    hp_ratio: obs.hp_ratio,
                    user_query: obs.user_query.clone(),
                    context,
                };
                let prompt = build_fast_path_prompt(&sit);
                // Generation runs on the LLM thread (loading the model first if it is not resident); if it is
                // busy, fall back to an immediate cached cue so Zena still reacts.
                if !llm.try_submit(LlmJob {
                    prompt,
                    situation: sit,
                    level: d.level,
                }) {
                    out.send(cached_cue(&mut cues, d.reason, obs.polar, ReactionLevel::Lv2Cached))?;
                }
            }
        }
    }
}

fn cached_cue(cues: &mut CuePlayer, reason: &str, polar: Option<zena_core::Polar>, level: ReactionLevel) -> AvatarCmd {
    let kind = cue_for_reason(reason);
    let c = cue(kind);
    let line = cues.next_line(kind).to_string();
    let ms = speech_ms(&line);
    AvatarCmd::Act {
        args: ZenaAvatarArgs {
            gaze_target: polar.map(|p| GazeTarget {
                angle_deg: p.theta_deg,
                distance_norm: p.r,
            }),
            emotion_state: c.emotion,
            body_motion: c.motion,
            speech_text: Some(line.clone()),
        },
        level,
        visemes: viseme_timeline(&line, ms),
        speech_ms: ms,
    }
}

/// Turn a model completion into avatar commands (runs on the LLM thread).
pub fn completion_to_cmds(completion: &str, level: ReactionLevel) -> Vec<AvatarCmd> {
    let mut cmds = Vec::new();
    let parsed = parse_model_output(completion);
    if parsed.calls.is_empty() && !parsed.preamble.is_empty() {
        // A plain-text answer (no tool call): speak it with a neutral expression.
        let args = ZenaAvatarArgs {
            gaze_target: None,
            emotion_state: EmotionState::Neutral,
            body_motion: BodyMotion::Nod,
            speech_text: None,
        };
        let args = ToolCall::ControlZenaAvatar(ZenaAvatarArgs {
            speech_text: Some(parsed.preamble.clone()),
            ..args
        })
        .sanitized();
        if let ToolCall::ControlZenaAvatar(args) = args {
            let text = args.speech_text.clone().unwrap_or_default();
            let ms = speech_ms(&text);
            cmds.push(AvatarCmd::Act {
                visemes: viseme_timeline(&text, ms),
                speech_ms: ms,
                args,
                level,
            });
        }
        return cmds;
    }
    for call in parsed.calls {
        match call {
            ToolCall::ControlZenaAvatar(args) => {
                let text = args.speech_text.clone().unwrap_or_default();
                let ms = speech_ms(&text);
                cmds.push(AvatarCmd::Act {
                    visemes: viseme_timeline(&text, ms),
                    speech_ms: ms,
                    args,
                    level,
                });
            }
            ToolCall::ManageZenaSystem(s) => cmds.push(AvatarCmd::System(s.command, s.window_layer_mode)),
        }
    }
    cmds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_completion_is_spoken_and_tool_calls_are_mapped() {
        let c = completion_to_cmds("인벤토리 한번 같이 찾아볼까?", ReactionLevel::Lv3Coaching);
        assert!(matches!(&c[..], [AvatarCmd::Act { args, .. }] if args.speech_text.as_deref() == Some("인벤토리 한번 같이 찾아볼까?")));
        let c = completion_to_cmds(
            "<tool_call>\n<function=manage_zena_system>\n<parameter=command>\nEXIT_APP\n</parameter>\n</function>\n</tool_call>",
            ReactionLevel::Lv3Coaching,
        );
        assert!(matches!(&c[..], [AvatarCmd::System(SystemCommand::ExitApp, None)]));
    }

    #[test]
    fn dictionary_expressions_rotate_lines_and_respect_the_gap() {
        let lex = Lexicon::bundled();
        let e = lex.node("emotion.celebrate").unwrap().action.express().unwrap().clone();
        let mut p = ExpressionPlayer::default();
        let line = |c: Option<AvatarCmd>| match c {
            Some(AvatarCmd::Act { args, level, .. }) => (args.speech_text, level),
            other => panic!("{other:?}"),
        };
        let (a, lv) = line(p.play("emotion.celebrate", &e, None, 0, 1500));
        assert_eq!(lv, ReactionLevel::Lv2Cached);
        assert!(p.play("emotion.celebrate", &e, None, 500, 1500).is_none(), "within the gap");
        let (b, _) = line(p.play("emotion.celebrate", &e, None, 2000, 1500));
        assert_ne!(a, b, "the next line of the node");
        let silent = lex.node("react.combat").unwrap().action.express().unwrap().clone();
        assert_eq!(
            line(p.play("react.combat", &silent, None, 0, 1500)),
            (None, ReactionLevel::Lv1Silent)
        );
    }

    #[test]
    fn scenes_react_on_change_and_events_after_their_cooldown() {
        let obs = |key: &str, kind: VisionKind, cooldown_s: u32| ScreenObservation {
            key: key.into(),
            kind,
            link: String::new(),
            score: 0.9,
            peak: None,
            cooldown_s,
            encode_ms: 0.0,
            tokens: 0,
            cells: 0,
        };
        let mut g = ScreenGate::default();
        assert!(g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 0));
        assert!(!g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 3000), "same scene");
        assert!(g.admit(&obs("vision.coding", VisionKind::Scene, 600), 6000), "the scene changed");
        assert!(
            !g.admit(&obs("vision.stock_chart", VisionKind::Scene, 600), 9000),
            "back to a scene seen within its cooldown"
        );
        assert!(g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 10_000));
        assert!(!g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 15_000));
        assert!(g.admit(&obs("vision.game_victory", VisionKind::Event, 20), 31_000));
    }

    #[test]
    fn utterances_and_screen_observations_become_dictionary_expressions() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let (utx, urx) = channel();
        let (otx, orx) = channel();
        let (_ltx, lrx) = channel();
        let (stx, srx) = channel();
        let links = Links {
            llm: LlmWorker::spawn(Box::new(crate::llm::TemplateBackend), |_, _| {}),
            utterances: urx,
            out: otx,
            lexicon: lrx,
            learn: None,
            screen_frames: None,
            screen: Some(srx),
        };
        let src = Box::new(crate::capture::SyntheticSource::new(160, 90));
        std::thread::spawn(move || run(src, links, Lexicon::bundled(), PipelineConfig::default()));
        utx.send("와 보스 잡았다! 이겼어!".to_string()).unwrap();
        stx.send(ScreenObservation {
            key: "vision.stock_chart".into(),
            kind: VisionKind::Scene,
            link: "react.market".into(),
            score: 0.8,
            peak: Some((0.75, 0.25)),
            cooldown_s: 600,
            encode_ms: 0.0,
            tokens: 0,
            cells: 0,
        })
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let (mut cheered, mut chart) = (false, false);
        while std::time::Instant::now() < deadline && !(cheered && chart) {
            if let Ok(AvatarCmd::Act { args, .. }) = orx.recv_timeout(Duration::from_millis(200)) {
                if (args.emotion_state, args.body_motion) == (EmotionState::Excited, BodyMotion::Cheer) {
                    cheered = args.speech_text.is_some();
                }
                if args.body_motion == BodyMotion::Think && args.gaze_target.is_some_and(|g| g.angle_deg < 90.0) {
                    chart = true; // looks up-right, where the chart's peak cell is
                }
            }
        }
        assert!(cheered, "the user's win is played from the dictionary");
        assert!(chart, "the chart on screen is played from react.market, looking at its peak");
    }
}
