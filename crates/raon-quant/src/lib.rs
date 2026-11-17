//! # raon-quant
//!
//! Single-pass, 4-tier "pyramid" quantizer for the KRAFTON Raon models used by Raon Zena
//! (PRD §3): original `safetensors` (BF16/F32, read through `mmap`) → GGUF v3 with
//! ggml-compatible Q8_0/Q6_K/Q5_K/Q4_K blocks.
//!
//! * [`kquant`]      – importance-weighted LS block encoders (vendored from Qntology/logis-center).
//! * [`kernels`]     – block decoders, fused dequant-dot kernels, start-up self-verify.
//! * [`pyramid`]     – tier policy for Raon-1-nano and Raon-VisionEncoder + streaming converter.
//! * [`vocab`]       – `kept_ids` construction from the byte-level BPE tokenizer; [`trim`] the per-language rule.
//! * [`lmhead`]      – exact arg-max over a sliced head (cold-row cluster bounds).
//! * [`gguf`], [`safetensors`], [`imatrix`] – file formats.

// The model-independent parts (formats, encoders, kernels, vocabulary slicing) are premium-discount's
// `premium-discount-quant` (moved there from this crate); re-exported under their old paths. What stays here is
// Raon-specific: the pyramid tier plans for Raon-1-nano / Raon-VisionEncoder and the Raon-1-nano block cache.
pub use premium_discount_quant::{gguf, imatrix, kernels, kquant, lmhead, safetensors, trim, vocab};

pub mod cache;
pub mod pyramid;
