//! PD-Verdict: premium (추천) / discount (비추천) of one evaluation relative to its scope.
//!
//! premium-discount owns the rule and its implementation: the judge itself is `premium_discount_verdict` (premium-discount
//! `crates/premium-discount-verdict`, pinned by git revision in the workspace `Cargo.toml`), re-exported here unchanged so the
//! server's cron, premium-discount-app and Raon Zena run one implementation. This crate keeps only what is
//! Zena's own: turning spoken feedback into observations ([`feedback`]) and how Zena talks about a local verdict
//! next to the global one ([`speak`]).
//!
//! `contracts/verdict_v1.json` here is a byte copy of premium-discount's contract. The tests run every case
//! against the re-exported judge, so moving the pinned revision onto a rule change fails here first.

pub use premium_discount_verdict::*;

pub mod feedback;
pub mod speak;

pub use feedback::{feedback_obs, manual_obs, FeedbackIntent};
pub use speak::{stance, Stance};

/// `pd-verdict/1`, under the name this crate has always used (premium-discount calls it `RECIPE_V1`).
pub const RECIPE: &str = premium_discount_verdict::RECIPE_V1;

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    const CONTRACT: &str = include_str!("../contracts/verdict_v1.json");
    /// SHA-256 of premium-discount's `contracts/verdict_v1.json` this copy was taken from (LF line endings).
    const CONTRACT_SHA256: &str = "c5f97f687425ef8ccd0bdebe58beab324a1be6f4652f18ffb2981387fbfa0fec";

    fn lf(text: &str) -> String {
        text.replace("\r\n", "\n")
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

    #[test]
    fn the_copy_is_the_one_recorded() {
        let digest: String = Sha256::digest(lf(CONTRACT).as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            digest, CONTRACT_SHA256,
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
        let c: Contract = serde_json::from_str(CONTRACT).unwrap();
        assert_eq!(c.recipe, RECIPE);
        assert_eq!(c.params, Params::v1());
        assert!(c.cases.len() >= 15);
        let params = Params::v1();
        let close = |a: Option<f64>, b: Option<f64>, what: &str| match (a, b) {
            (Some(a), Some(b)) => assert!((a - b).abs() <= c.tolerance, "{what}: {a} vs {b}"),
            _ => assert_eq!(a, b, "{what}"),
        };
        for n in &c.cases {
            let got = verdict(&n.case, &params).unwrap();
            let want = &n.expect;
            close(Some(got.evidence), Some(want.evidence), &format!("{} evidence", n.name));
            close(Some(got.value), Some(want.value), &format!("{} value", n.name));
            close(got.rank, want.rank, &format!("{} rank", n.name));
            assert_eq!(got.level, want.level, "{} level", n.name);
            assert_eq!(got.tier, want.tier, "{} tier", n.name);
            assert_eq!(got.grade, want.grade, "{} grade", n.name);
            assert_eq!(got.since_ms, want.since_ms, "{} since", n.name);
            assert_eq!(got.reasons, want.reasons, "{} reasons", n.name);
        }
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
