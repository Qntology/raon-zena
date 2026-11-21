//! PD-Verdict: premium (추천) / discount (비추천) of one evaluation relative to its scope.
//!
//! premium-discount owns the rule and its implementation: the judge itself is `premium_discount_verdict` (premium-discount
//! `crates/premium-discount-verdict`, pinned by git revision in the workspace `Cargo.toml`), re-exported here unchanged so the
//! server's cron, premium-discount-app and Raon Zena run one implementation. This crate keeps only what is
//! Zena's own: turning spoken feedback into observations ([`feedback`]), sharing an event's credit among the
//! candidates that may have caused it ([`credit`]) and how Zena talks about a local verdict next to the global one
//! ([`speak`]).
//!
//! `contracts/verdict_v1.json` and `contracts/verdict_v2.json` here are byte copies of premium-discount's contracts.
//! The tests run every case against the re-exported judge, so moving the pinned revision onto a rule change fails here
//! first. Local verdicts use `pd-verdict/2` ([`LOCAL_RECIPE`], [`local_params`]); the server's global verdicts carry
//! their own recipe, and verdicts of different recipes are never compared ([`speak::VerdictPairView`]).

pub use premium_discount_verdict::*;

pub mod credit;
pub mod feedback;
pub mod speak;

pub use credit::{
    attribute_and_update_duels, duel_for_item, outcome_gate, outperforms, record_demotion, AttributionCandidate, AttributionReport,
    CandidateRelation, CreditShare, CreditTargetKind, CrossTrialMemory, Exclusion, OutcomeGate, RivalFailure,
};
pub use feedback::{
    apply_feedback_to_ledger, extract_pragmatic_feedback, extract_pragmatic_feedback_with, feedback_obs, manual_obs, mentioned_label,
    ExtractedFeedback, FeedbackContext, FeedbackIntent, FeedbackLexicon, FeedbackToken, ObsKind,
};
pub use speak::{
    outbound_votes, stance, toggle_pin, DifferKind, GlobalVerdict, LocalVerdictEntry, MemoryFreshness, PinToggle, Stance, UserPin,
    VerdictLocale, VerdictPairView,
};

/// `pd-verdict/1`, under the name this crate has always used (premium-discount calls it `RECIPE_V1`).
pub const RECIPE: &str = premium_discount_verdict::RECIPE_V1;

/// The recipe of every local (on-device) verdict: `pd-verdict/2`.
pub const LOCAL_RECIPE: &str = premium_discount_verdict::RECIPE_V2;

/// The parameters of every local verdict ([`LOCAL_RECIPE`]).
pub fn local_params() -> Params {
    Params::v2()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const CONTRACT: &str = include_str!("../contracts/verdict_v1.json");
    /// SHA-256 of premium-discount's `contracts/verdict_v1.json` this copy was taken from (LF line endings).
    const CONTRACT_SHA256: &str = "c5f97f687425ef8ccd0bdebe58beab324a1be6f4652f18ffb2981387fbfa0fec";
    const CONTRACT_V2: &str = include_str!("../contracts/verdict_v2.json");
    /// SHA-256 of premium-discount's `contracts/verdict_v2.json` this copy was taken from (LF line endings).
    const CONTRACT_V2_SHA256: &str = "f66720540cd7f9ed819eeed267473cecfdeadaed2a24057f0a3b5c807678f895";

    fn lf(text: &str) -> String {
        text.replace("\r\n", "\n")
    }

    fn sha256(text: &str) -> String {
        Sha256::digest(lf(text).as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[derive(Deserialize)]
    struct Contract {
        recipe: String,
        tolerance: f64,
        params: Params,
        cases: Vec<Named>,
    }

    #[derive(Deserialize)]
    struct Named {
        name: String,
        #[serde(flatten)]
        case: Case,
        expect: Verdict,
    }

    /// Every case of `text` through the re-exported judge with `params`.
    fn check(text: &str, recipe: &str, params: Params, min_cases: usize) -> Contract {
        let c: Contract = serde_json::from_str(text).unwrap();
        assert_eq!(c.recipe, recipe);
        assert_eq!(c.params, params, "{recipe} params");
        assert_eq!(Params::for_recipe(recipe).as_ref(), Some(&params));
        assert!(c.cases.len() >= min_cases, "{recipe}: {} cases", c.cases.len());
        let close = |a: Option<f64>, b: Option<f64>, what: &str| match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= c.tolerance, "{what}: {a} vs {b}"),
            _ => assert_eq!(a, b, "{what}"),
        };
        for n in &c.cases {
            let got = verdict(&n.case, &params).unwrap();
            let want = &n.expect;
            close(Some(got.evidence), Some(want.evidence), &format!("{recipe} {} evidence", n.name));
            close(Some(got.value), Some(want.value), &format!("{recipe} {} value", n.name));
            close(got.rank, want.rank, &format!("{recipe} {} rank", n.name));
            assert_eq!(got.level, want.level, "{recipe} {} level", n.name);
            assert_eq!(got.tier, want.tier, "{recipe} {} tier", n.name);
            assert_eq!(got.grade, want.grade, "{recipe} {} grade", n.name);
            assert_eq!(got.since_ms, want.since_ms, "{recipe} {} since", n.name);
            assert_eq!(got.reasons, want.reasons, "{recipe} {} reasons", n.name);
        }
        c
    }

    #[test]
    fn the_copy_is_the_one_recorded() {
        assert_eq!(
            sha256(CONTRACT),
            CONTRACT_SHA256,
            "contracts/verdict_v1.json was edited here: copy it again from premium-discount"
        );
    }

    #[test]
    fn the_copy_matches_premium_discount_when_it_is_checked_out_next_to_us() {
        let upstream = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../premium-discount/contracts/verdict_v1.json");
        let Ok(text) = std::fs::read_to_string(&upstream) else {
            eprintln!("skipped: no {}", upstream.display());
            return;
        };
        assert_eq!(
            lf(&text),
            lf(CONTRACT),
            "premium-discount changed the contract: copy it again and update CONTRACT_SHA256"
        );
    }

    #[test]
    fn params_and_every_case_match_the_contract() {
        let c = check(CONTRACT, RECIPE, Params::v1(), 15);
        assert!(c.cases.len() >= 15);
    }

    #[test]
    fn the_v2_copy_is_the_one_recorded() {
        assert_eq!(
            sha256(CONTRACT_V2),
            CONTRACT_V2_SHA256,
            "contracts/verdict_v2.json was edited here: copy it again from premium-discount"
        );
    }

    #[test]
    fn the_v2_copy_matches_premium_discount_when_it_is_checked_out_next_to_us() {
        let upstream = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../premium-discount/contracts/verdict_v2.json");
        let Ok(text) = std::fs::read_to_string(&upstream) else {
            eprintln!("skipped: no {}", upstream.display());
            return;
        };
        assert_eq!(
            lf(&text),
            lf(CONTRACT_V2),
            "premium-discount changed the v2 contract: copy it again and update CONTRACT_V2_SHA256"
        );
    }

    #[test]
    fn v2_params_and_every_case_match_the_contract() {
        assert_eq!(LOCAL_RECIPE, RECIPE_V2);
        assert_eq!(local_params(), Params::v2());
        let v2 = check(CONTRACT_V2, RECIPE_V2, Params::v2(), 25);
        // v2 holds every v1 case, with the same expectations.
        let v1 = check(CONTRACT, RECIPE_V1, Params::v1(), 15);
        for old in &v1.cases {
            let same = v2
                .cases
                .iter()
                .find(|n| n.name == old.name)
                .unwrap_or_else(|| panic!("v2 lacks {}", old.name));
            assert_eq!(same.case, old.case, "{} input", old.name);
            assert_eq!(same.expect, old.expect, "{} expectation", old.name);
        }
        assert!(v2
            .cases
            .iter()
            .any(|n| n.case.track == NAV && n.case.obs.iter().any(|o| o.kind == IMPLICIT)));
        assert_eq!(Params::v2().weight(IMPLICIT), 0.3);
        assert_eq!(Params::v1().weight(IMPLICIT), 0.0);
    }

    #[test]
    fn verdicts_serialize_with_the_contract_field_names() {
        let c: Value = serde_json::from_str(CONTRACT).unwrap();
        let first: Case = serde_json::from_value(c["cases"][0].clone()).unwrap();
        let v = serde_json::to_value(verdict(&first, &Params::v1()).unwrap()).unwrap();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        let want: Vec<&str> = c["cases"][0]["expect"].as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.len(), want.len());
        for k in want {
            assert!(keys.contains(&k), "{k}");
        }
        assert_eq!(v["tier"], "premium");
        assert_eq!(v["level"], "secondary");
    }

    #[test]
    fn unknown_tracks_and_weights() {
        let params = Params::v1();
        let case = Case {
            now_ms: 0,
            track: "nope".into(),
            prev: Prev {
                tier: Tier::Unrated,
                since_ms: 0,
            },
            obs: vec![],
            baseline: Baselines::default(),
            duel: None,
            superseded: false,
        };
        assert!(verdict(&case, &params).is_none());
        assert_eq!(params.weight("manual"), 2.0);
        assert_eq!(params.weight("unknown"), 0.0);
        let b = baseline_of(&[1.0, -1.0, 1.0, -1.0]);
        assert_eq!((b.n, b.mean, b.sd), (4, 0.0, 1.0));
        assert_eq!(baseline_of(&[]).n, 0);
    }
}
