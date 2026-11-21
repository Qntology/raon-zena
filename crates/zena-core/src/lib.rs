//! # zena-core
//!
//! Platform-neutral core of **KRAFTON Raon Zena**:
//!
//! * [`geometry`] – Stage 1: screen-centre polar coordinates `(r, θ)` and inward/outward flow.
//! * [`reaction`] – Stage 4: reaction-level routing (Lv1 silent / Lv2 cached cue / Lv3 LLM coaching).
//! * [`tools`]    – the `control_zena_avatar` / `manage_zena_system` tool-call schema and a parser for
//!   Raon-1-nano's `<tool_call><function=…><parameter=…>` output format.
//! * [`avatar`]   – the avatar abstraction adapter mapping one tool call to Live2D and VRM parameters,
//!   including Hangul-aware viseme lip-sync.
//! * [`window`]   – Win32 extended-style math, system commands, graceful shutdown and FPS throttling.
//! * [`prompt`]   – ChatML prompt builder for the non-thinking fast path, multi-turn memory and the session
//!   summary (kept out of the cached system prefix).
//! * [`stream`]   – incremental `<tool_call>` parsing: expressions before the line, the line by sentence.
//! * [`session`]  – play-session tracking (scene time, streaks, affinity, tension) and proactive chatter timing.
//! * [`brief`]    – per-request coaching slots of the user turn: course (ghost splits, next step), friction tip,
//!   web shopping coaching with the CommitGuard, operator analytics with the small-sample guard.
//! * [`nav_target`] – window-relative targets → desktop gaze, polar, coaching-arrow angle and overlay occlusion;
//!   the badge numbers Zena is saying as a `locked_numbers` mask.
//! * [`coach`]    – proactive friction coaching: five interrupt gates, the pending coach hook, refusal fatigue,
//!   ephemeral offers outside the dialogue window, Lv2 / Lv3 routing.
//! * [`reference`] – deictic / anaphoric / elliptical reference restoration over window-isolated topic frames and
//!   the (cc, to) three-level query scopes.
//! * [`claims`]   – number-preserving claims and the three verdict slots (facts only when nobody judged).
//! * [`exposure`] – the 5-minute ledger of records Zena actually said.
//! * [`hydrate`]  – labels for text-less remote hits without blocking calls.
//! * [`ingress_guard`] – the front gate for player text: hard secrets blocked and zeroed, contact data masked.
//!
//! Nothing here depends on Windows, a GPU or a model file, so all of it is unit-tested on any host.

pub mod avatar;
pub mod brief;
pub mod claims;
pub mod coach;
pub mod cues;
pub mod exposure;
pub mod geometry;
pub mod hydrate;
pub mod ingress_guard;
pub mod nav_target;
pub mod prompt;
pub mod reaction;
pub mod reference;
pub mod session;
pub mod stream;
mod text;
pub mod tools;
pub mod window;

pub use geometry::{BBox, FlowDirection, Polar, ScreenGeometry};
pub use reaction::{Observation, ReactionDecision, ReactionLevel, ReactionRouter};
pub use tools::{CharacterEmote, ToolCall, ZenaAssetArgs, ZenaAvatarArgs, ZenaSystemArgs};
