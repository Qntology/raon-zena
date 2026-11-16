//! # zena-nlp
//!
//! Text understanding for Raon Zena's NLP-NMS v2 (`docs/plans/relative-value-sds.md`, Phase 0):
//!
//! * [`granite`] – `ibm-granite/granite-embedding-97m-multilingual-r2` (ModernBERT, 384-d, CLS pooling) on candle.
//! * `stanza`    – (feature `stanza`) the Stanza pipeline (tokens → MWT → UPOS/XPOS/feats → lemma → dependencies) from
//!   the stanza-onnx-export format 2 models on ONNX Runtime, plus query language detection.
//! * [`models`]  – where the models are found on this machine.

pub mod granite;
pub mod models;
#[cfg(feature = "stanza")]
pub mod stanza;
