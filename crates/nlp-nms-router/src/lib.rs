//! # nlp-nms-router
//!
//! Stage 3 of Raon Zena: decide *which one* of many simultaneous requests/alerts Zena reacts to.
//!
//! The pipeline ports the natural-language search + NMS design of Qntology/logis-center
//! (Apache-2.0; see `NOTICE`) to a real-time game companion:
//!
//! 1. [`korean`] / [`normalize`] – wake-word removal, closed-class josa/copula peeling with
//!    batchim agreement (logis-center `NOMINAL_TAILS_ML`, `hangul_tail_agrees`).
//! 2. [`embed`]   – a dependency-free hashed char/jamo n-gram embedder (384-d, like logis-center's
//!    `EMBED_DIM`), pluggable behind [`embed::TextEmbedder`] for a neural encoder.
//! 3. [`intent`]  – bias/prejudice phrase banks with max-pool cosine and a prejudice penalty
//!    (logis-center field scoring) plus an exact-keyword fast path.
//! 4. [`span`]    – span NMS: greedy score→length, any word overlap suppresses (logis-center
//!    commerce/shipping span NMS), so one utterance can carry several intents.
//! 5. [`semantic`]– temporal semantic NMS over requests and in-game alerts, with logis-center's
//!    adaptive duplicate floor `clamp(μ + 3σ, τ, 0.995)`.
//! 6. [`spatial`] – spatial-text NMS: text-similarity confidence, greedy box suppression with the
//!    logis-center snap/twin IoU rules, and one-to-one `exclusive_assign`.
//! 7. [`router`]  – glue producing system commands, Lv3 queries and a single gaze target.

pub mod embed;
pub mod intent;
pub mod korean;
pub mod normalize;
pub mod router;
pub mod semantic;
pub mod span;
pub mod spatial;

pub use router::{NlpNmsRouter, RouterInput, RouterOutput};
