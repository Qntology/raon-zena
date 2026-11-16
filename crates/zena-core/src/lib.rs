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
//!
//! Nothing here depends on Windows, a GPU or a model file, so all of it is unit-tested on any host.

pub mod avatar;
pub mod cues;
pub mod geometry;
pub mod prompt;
pub mod reaction;
pub mod session;
pub mod stream;
pub mod tools;
pub mod window;

pub use geometry::{BBox, FlowDirection, Polar, ScreenGeometry};
pub use reaction::{Observation, ReactionDecision, ReactionLevel, ReactionRouter};
pub use tools::{CharacterEmote, ToolCall, ZenaAssetArgs, ZenaAvatarArgs, ZenaSystemArgs};
