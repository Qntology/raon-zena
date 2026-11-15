//! # raon-quant
//!
//! Single-pass, 4-tier "pyramid" quantizer for the KRAFTON Raon models used by Raon Zena
//! (PRD §3): original `safetensors` (BF16/F32, read through `mmap`) → GGUF v3 with
//! ggml-compatible Q8_0/Q6_K/Q5_K/Q4_K blocks.
//!
//! * [`kquant`]      – importance-weighted LS block encoders (vendored from Qntology/logis-center).
//! * [`kernels`]     – block decoders, fused dequant-dot kernels, start-up self-verify.
//! * [`pyramid`]     – tier policy for Raon-1-nano and Raon-VisionEncoder + streaming converter.
//! * [`vocab`]       – `kept_ids` construction from the byte-level BPE tokenizer.
//! * [`lmhead`]      – exact arg-max over a sliced head (cold-row cluster bounds).
//! * [`gguf`], [`safetensors`], [`imatrix`] – file formats.

pub mod gguf;
pub mod imatrix;
pub mod kernels;
// Vendored verbatim; keep upstream style.
#[allow(clippy::manual_is_multiple_of)]
#[rustfmt::skip]
pub mod kquant;
pub mod lmhead;
pub mod pyramid;
pub mod safetensors;
pub mod vocab;
