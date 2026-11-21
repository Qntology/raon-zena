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
pub use router::{Caught, FeedbackCatch, FeedbackSpan, FeedbackWord, NlpNmsRouter, RouterInput, RouterOutput};

/// Where the granite model is installed (tests and diagnostics).
#[cfg(feature = "neural")]
#[doc(hidden)]
pub fn __granite_dir() -> Option<std::path::PathBuf> {
    premium_discount_nms::__granite_dir()
}

/// Installs the multi-core `raon-cpu-kernels` SiLU-GLU kernel into `premium-discount-nlp::granite`
/// (bit-identical to candle's CPU `silu(gate) * up`, zero narrow copies, falls back off-CPU).
pub fn install_cpu_glu_kernel() -> bool {
    #[cfg(feature = "neural")]
    {
        fn cpu_silu_glu(x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
            if x.device().is_cpu() && x.dtype() == candle_core::DType::F32 && x.is_contiguous() {
                let dims = x.dims();
                if let Some(&last) = dims.last() {
                    if last > 0 && last % 2 == 0 {
                        let half = last / 2;
                        let src = x.flatten_all()?.to_vec1::<f32>()?;
                        let mut out = vec![0.0f32; src.len() / 2];
                        raon_cpu_kernels::silu_mul_halves(&src, half, &mut out);
                        let mut out_dims = dims.to_vec();
                        *out_dims.last_mut().unwrap() = half;
                        return candle_core::Tensor::from_vec(out, out_dims, x.device());
                    }
                }
            }
            premium_discount_nlp::granite::silu_glu(x)
        }
        premium_discount_nlp::granite::set_glu_kernel(cpu_silu_glu)
    }
    #[cfg(not(feature = "neural"))]
    {
        false
    }
}
