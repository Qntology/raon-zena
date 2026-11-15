//! Worker thread: capture → event sensor → tracker → NLP-NMS router → reaction routing →
//! (Lv3) generation → avatar commands for the UI thread.

use crate::capture::FrameSource;
use crate::llm::LlmBackend;
use nlp_nms_router::{NlpNmsRouter, RouterInput};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use zena_core::avatar::viseme_timeline;
use zena_core::cues::{cue, cue_for_reason, CuePlayer};
use zena_core::geometry::{FlowDirection, ScreenGeometry};
use zena_core::prompt::{build_fast_path_prompt, Situation};
use zena_core::reaction::{Observation, ReactionLevel, ReactionRouter};
use zena_core::tools::{parse_model_output, GazeTarget, SystemCommand, ToolCall, WindowLayerMode, ZenaAvatarArgs};
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
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sensor_threshold: 0.15,
            min_tile_events: 6,
        }
    }
}

/// Rough speaking time for Korean TTS (≈ 7 syllables per second) until real TTS durations exist.
fn speech_ms(text: &str) -> u32 {
    (text.chars().filter(|c| !c.is_whitespace()).count() as u32 * 140).max(400)
}

pub fn run(
    mut src: Box<dyn FrameSource>,
    mut llm: Box<dyn LlmBackend>,
    utterances: Receiver<String>,
    out: Sender<AvatarCmd>,
    cfg: PipelineConfig,
) -> anyhow::Result<()> {
    let (w, h) = src.size();
    let geo = ScreenGeometry::new(w as u32, h as u32);
    let mut sensor = LogLumSensor::new(w, h, cfg.sensor_threshold);
    let mut tracker = RoiTracker::default();
    let mut router = NlpNmsRouter::default();
    let mut reactions = ReactionRouter::default();
    let mut cues = CuePlayer::default();
    let t0 = Instant::now();
    eprintln!("pipeline: {w}x{h} sensor, Lv3 backend = {}", llm.name());
    loop {
        let Some(frame) = src.next_frame()? else {
            continue;
        };
        if (frame.width, frame.height) != (w, h) {
            anyhow::bail!("frame size changed {}x{} -> {}x{}", w, h, frame.width, frame.height);
        }
        let now = t0.elapsed().as_millis() as u64;
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
        for (c, m) in routed.system {
            out.send(AvatarCmd::System(c, m))?;
        }

        // Pick the strongest surviving signal for this tick.
        let mut obs = Observation::default();
        let mut label = None;
        if let Some((intent_q, text)) = routed.queries.first() {
            obs.user_query = Some(text.clone());
            let _ = intent_q;
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
                let kind = cue_for_reason(d.reason);
                let c = cue(kind);
                let line = cues.next_line(kind).to_string();
                let ms = speech_ms(&line);
                let args = ZenaAvatarArgs {
                    gaze_target: obs.polar.map(|p| GazeTarget {
                        angle_deg: p.theta_deg,
                        distance_norm: p.r,
                    }),
                    emotion_state: c.emotion,
                    body_motion: c.motion,
                    speech_text: Some(line.clone()),
                };
                out.send(AvatarCmd::Act {
                    args,
                    level: d.level,
                    visemes: viseme_timeline(&line, ms),
                    speech_ms: ms,
                })?;
            }
            ReactionLevel::Lv3Coaching => {
                let sit = Situation {
                    event_label: label,
                    polar: obs.polar,
                    flow: obs.flow,
                    hp_ratio: obs.hp_ratio,
                    user_query: obs.user_query.clone(),
                    context: vec![],
                };
                let prompt = build_fast_path_prompt(&sit);
                let completion = llm.generate(&prompt, &sit);
                for call in parse_model_output(&completion).calls {
                    match call {
                        ToolCall::ControlZenaAvatar(args) => {
                            let text = args.speech_text.clone().unwrap_or_default();
                            let ms = speech_ms(&text);
                            out.send(AvatarCmd::Act {
                                visemes: viseme_timeline(&text, ms),
                                speech_ms: ms,
                                args,
                                level: d.level,
                            })?;
                        }
                        ToolCall::ManageZenaSystem(s) => out.send(AvatarCmd::System(s.command, s.window_layer_mode))?,
                    }
                }
            }
        }
    }
}
