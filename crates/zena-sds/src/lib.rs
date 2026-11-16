//! # zena-sds
//!
//! Score Dynamics (SDS) for **KRAFTON Raon Zena**, ported from Qntology/logis-center
//! (`commerce/src-tauri/src/utils/score_dynamics.rs` and the assignment helpers of `ai_utils.rs`).
//! Fixed thresholds become per-scope adaptive statistics: "is this score unusual *here*?".
//!
//! * [`scope`]  – [`Scope`] `{ track, primary, secondary }`, its three keys and the per-track [`MinObs`].
//! * [`stats`]  – mergeable statistics: [`Welford`] (μ, σ, n + recent ring, `drift_z`), [`decay_shape`] of a score
//!   list, the confusion ledger ([`ConfusionStat`]), reject counts ([`RejectStat`]) and [`ScopeStat`].
//! * [`ledger`] – [`Ledger`]: a persisted base plus this run's observations (one run file per process), reads with
//!   backoff secondary → primary → global → unscoped, confusion tie-breaks and transition priors.
//! * [`matrix`] – double centering and margin-ordered 1:1 assignment with tie shapes for the caller to settle.
//!
//! No global state: a [`Ledger`] is a plain value (`Send + Sync`); share it behind a `Mutex` / `RwLock`.
//! Nothing judges on too few samples — every read returns `None` until a level has its [`MinObs`].

pub mod ledger;
pub mod matrix;
pub mod scope;
pub mod stats;

mod persist;

pub use ledger::{Band, Ledger, Level};
pub use matrix::{AssignDiag, TieKind};
pub use scope::{MinObs, Scope};
pub use stats::{decay_shape, tie_verdict, ConfusionStat, DecayShape, DecayStat, RejectStat, ScopeStat, Welford};

/// Generation tag written into every ledger and run file; a file with another recipe is set aside, not merged.
pub const RECIPE: &str = "zena-sds-v1:welford+ring/run-files/backoff/confusion/transition";

/// A tie is broken toward the historical winner only at this win rate or above (a bare majority would freeze
/// pairs that genuinely alternate).
pub const TIE_WIN_RATE: f32 = 0.65;

/// The historical winner may not be rejected more than this much more often than its rival
/// ([`Ledger::learned_specificity`]) for the tie to go its way.
pub const TIE_SPECIFICITY_SLACK: f32 = 0.15;
