//! # zena-lexicon
//!
//! The word dictionary that drives how Raon Zena expresses herself.
//!
//! * [`model`]  – the dictionary: Qntology/logis-center `bias.json` nodes (`semantic`, `bias`, `prejudice`,
//!   `exact_match`) extended with the character's expression (`express`), system commands and LLM topics, plus
//!   noto-sandbox-style `vision` nodes (SigLIP2 prompts) linked to the same expressions.
//! * [`phrase`] – phrase-list parsing shared with logis-center / noto (comma lists, digit weighting).
//! * `store`    – (feature `lancedb`) the dictionary kept in a LanceDB database: import/export, phrase and line
//!   edits, learned phrases, SigLIP2 anchors with vector search, full-text search, versions for hot reload and
//!   rollback.

pub mod model;
pub mod phrase;
#[cfg(feature = "lancedb")]
pub mod store;

pub use model::{Action, Express, ExpressLevel, Lexicon, Node, VisionKind, VisionNode, VisionPrompting};
#[cfg(feature = "lancedb")]
pub use store::{Anchors, LexiconStore, Role};
