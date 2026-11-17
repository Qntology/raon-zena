//! # zena-sds
//!
//! Score Dynamics for Raon Zena: premium-discount's `premium-discount-sds` (the logis-center port that started here),
//! re-exported so Raon Zena's paths (`zena_sds::Ledger`, `zena_sds::matrix::double_center` …) stay the same. The
//! ledger recipe is unchanged, so ledgers already on disk are read as before.

pub use premium_discount_sds::*;
