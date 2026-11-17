//! # nlp-nms-router
//!
//! Stage 3 of Raon Zena: decide *which one* of many simultaneous requests/alerts Zena reacts to.
//!
//! The domain-neutral NLP-NMS core (josa peeling, role / negation analysis, hashed and granite embedders, the
//! generic domain bank, span / temporal-semantic / spatial-text NMS) is premium-discount's `premium-discount-nms`
//! (moved there from this crate, originally ported from Qntology/logis-center); its modules are re-exported here
//! under their old paths. What stays is Raon Zena's own:
//!
//! * [`domain`] – the word dictionary (`zena-lexicon`) as a domain bank whose entries carry Zena's
//!   [`zena_lexicon::Action`], and Zena's wake words.
//! * [`router`] – glue producing system commands, dictionary expressions (no LLM), Lv3 queries and a single gaze
//!   target.

pub use premium_discount_nms::{analyze, embed, korean, normalize, semantic, span, spatial};

pub mod domain;
pub mod router;

pub use domain::{DomainBank, Hit, LexiconBank};
pub use router::{Caught, NlpNmsRouter, RouterInput, RouterOutput};

/// Where the granite model is installed (tests and diagnostics).
#[cfg(feature = "neural")]
#[doc(hidden)]
pub fn __granite_dir() -> Option<std::path::PathBuf> {
    premium_discount_nms::__granite_dir()
}
