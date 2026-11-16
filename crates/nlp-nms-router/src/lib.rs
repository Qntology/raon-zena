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
//! 3. [`domain`]  – the word dictionary (`zena-lexicon`, logis-center `bias.json` format) as a domain bank:
//!    exact / agglutinative-prefix match before vectors, then bias/prejudice max-pool cosine with a
//!    prejudice penalty (logis-center field scoring); the `function` group explains request endings away.
//! 4. [`span`]    – span NMS: greedy score→length, any word overlap suppresses (logis-center
//!    commerce/shipping span NMS), so one utterance can carry several expression domains.
//! 5. [`semantic`]– temporal semantic NMS over requests and in-game alerts, with logis-center's
//!    adaptive duplicate floor `clamp(μ + 3σ, τ, 0.995)`.
//! 6. [`spatial`] – spatial-text NMS: text-similarity confidence, greedy box suppression with the
//!    logis-center snap/twin IoU rules, and one-to-one `exclusive_assign`.
//! 7. [`router`]  – glue producing system commands, dictionary expressions (no LLM), Lv3 queries and a
//!    single gaze target.

pub mod analyze;
pub mod domain;
pub mod embed;
pub mod korean;
pub mod normalize;
pub mod router;
pub mod semantic;
pub mod span;
pub mod spatial;

pub use domain::{DomainBank, Hit};
pub use router::{Caught, NlpNmsRouter, RouterInput, RouterOutput};

/// Where the granite model is installed (tests and diagnostics).
#[cfg(feature = "neural")]
#[doc(hidden)]
pub fn __granite_dir() -> Option<std::path::PathBuf> {
    zena_nlp::models::granite_dir()
}
