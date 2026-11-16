//! PD-Verdict: premium (추천) / discount (비추천) of one evaluation relative to its scope.
//!
//! premium-discount owns the rule. Its `contracts/verdict_v1.json` (written by its reference implementation
//! `contracts/gen_verdict_v1.py`) holds the parameters and the cases every implementation must reproduce: the
//! server's cron (TypeScript) for global verdicts from votes, and this crate for Raon Zena's local verdicts from
//! outcomes, feedback and manual corrections. `contracts/verdict_v1.json` here is a byte copy, checked by the tests
//! (and against `../premium-discount` when that checkout sits next to this one).
//!
//! 1. Each observation weighs `u = weight × credit × 2^(−age / H)` (`age = now − at_ms`, never negative).
//!    Withdrawn observations (`sign` 0) do not count.
//! 2. `E = Σu`, `v = Σ(u × sign) / (E + k)`.
//! 3. Baseline: the first of secondary → primary → global with `n ≥ min_obs`; then
//!    `r = tanh((v − mean) / (sd + eps))`.
//! 4. Tier: superseded → discount; `E < min_evidence` → unrated; enter premium / discount; hold the previous
//!    tier (hysteresis); else neutral.
//! 5. Grade: premium +2 (`r ≥ strong`) or +1, neutral 0, discount −2 (superseded or `r ≤ −strong`) or −1,
//!    unrated none.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub mod feedback;
pub mod speak;

pub use feedback::{feedback_obs, manual_obs, FeedbackIntent};
pub use speak::{stance, Stance};

pub const RECIPE: &str = "pd-verdict/1";

const DAY_MS: f64 = 86_400_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Secondary,
    Primary,
    Global,
}

impl Level {
    pub const ALL: [Level; 3] = [Level::Secondary, Level::Primary, Level::Global];

    pub fn name(self) -> &'static str {
        match self {
            Level::Secondary => "secondary",
            Level::Primary => "primary",
            Level::Global => "global",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Premium,
    Neutral,
    Discount,
    Unrated,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Premium => "premium",
            Tier::Neutral => "neutral",
            Tier::Discount => "discount",
            Tier::Unrated => "unrated",
        }
    }
}

/// One observation of an evaluation (a vote, an outcome credit, a piece of feedback …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Obs {
    pub kind: String,
    /// +1 towards premium, −1 towards discount, 0 withdrawn
    pub sign: i32,
    pub weight: f64,
    /// 0..1 (outcome credit; 1 for votes)
    pub credit: f64,
    pub at_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub n: u64,
    pub mean: f64,
    pub sd: f64,
}

/// Value distribution of the judged evaluations in each scope level (the evaluation itself included).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Baselines {
    #[serde(default)]
    pub secondary: Option<Baseline>,
    #[serde(default)]
    pub primary: Option<Baseline>,
    #[serde(default)]
    pub global: Option<Baseline>,
}

impl Baselines {
    pub fn get(&self, level: Level) -> Option<Baseline> {
        match level {
            Level::Secondary => self.secondary,
            Level::Primary => self.primary,
            Level::Global => self.global,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinObs {
    pub secondary: u64,
    pub primary: u64,
    pub global: u64,
}

impl MinObs {
    pub fn get(&self, level: Level) -> u64 {
        match level {
            Level::Secondary => self.secondary,
            Level::Primary => self.primary,
            Level::Global => self.global,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrackParams {
    pub half_life_ms: f64,
    pub min_obs: MinObs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Params {
    pub k: f64,
    pub eps: f64,
    pub enter: f64,
    pub exit: f64,
    pub floor: f64,
    pub strong: f64,
    pub min_evidence: f64,
    pub duel_rate: f64,
    pub duel_min: u32,
    pub tracks: BTreeMap<String, TrackParams>,
    /// Default weight of each observation kind (producers put it into `Obs::weight`).
    pub weights: BTreeMap<String, f64>,
}

impl Params {
    /// `pd-verdict/1` (must equal the contract's `params`; the tests compare them).
    pub fn v1() -> Params {
        let track = |days: f64, secondary, primary, global| TrackParams {
            half_life_ms: days * DAY_MS,
            min_obs: MinObs {
                secondary,
                primary,
                global,
            },
        };
        Params {
            k: 2.0,
            eps: 0.05,
            enter: 0.5,
            exit: 0.3,
            floor: 0.1,
            strong: 0.8,
            min_evidence: 2.0,
            duel_rate: 0.65,
            duel_min: 3,
            tracks: BTreeMap::from([
                ("pd".to_string(), track(90.0, 5, 10, 30)),
                ("drop".to_string(), track(14.0, 6, 3, 10)),
                ("web".to_string(), track(14.0, 8, 4, 12)),
                ("chat".to_string(), track(14.0, 6, 3, 10)),
                ("coach".to_string(), track(14.0, 6, 3, 10)),
            ]),
            weights: BTreeMap::from(
                [
                    ("vote", 1.0),
                    ("vote_outcome", 0.5),
                    ("manual", 2.0),
                    ("feedback", 1.0),
                    ("outcome", 1.0),
                    ("reaction", 0.6),
                    ("engage", 0.2),
                    ("model", 0.3),
                ]
                .map(|(k, w)| (k.to_string(), w)),
            ),
        }
    }

    pub fn track(&self, name: &str) -> Option<&TrackParams> {
        self.tracks.get(name)
    }

    /// The default weight of an observation kind (0 for an unknown kind).
    pub fn weight(&self, kind: &str) -> f64 {
        self.weights.get(kind).copied().unwrap_or(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prev {
    pub tier: Tier,
    pub since_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Duel {
    pub wins: u32,
    pub losses: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub evidence: f64,
    pub value: f64,
    pub level: Option<Level>,
    pub rank: Option<f64>,
    pub tier: Tier,
    pub grade: Option<i32>,
    pub since_ms: i64,
    pub reasons: Vec<String>,
}

/// Evidence `E` and value `v` of the observations at `now_ms`.
pub fn evidence(obs: &[Obs], now_ms: i64, track: &TrackParams, params: &Params) -> (f64, f64) {
    let (mut e, mut s) = (0.0, 0.0);
    for o in obs {
        if o.sign == 0 {
            continue;
        }
        let age = (now_ms - o.at_ms).max(0) as f64;
        let u = o.weight * o.credit * 2f64.powf(-age / track.half_life_ms);
        e += u;
        s += u * o.sign as f64;
    }
    (e, s / (e + params.k))
}

/// What [`judge`] needs besides `E` and `v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Context<'a> {
    pub track: &'a TrackParams,
    pub baselines: &'a Baselines,
    pub prev: Prev,
    pub now_ms: i64,
    pub duel: Option<Duel>,
    pub superseded: bool,
}

/// The tier of an evaluation whose `E` and `v` are known (peers are re-judged without re-reading observations).
pub fn judge(evidence: f64, value: f64, cx: &Context<'_>, params: &Params) -> Verdict {
    let (e, v) = (evidence, value);
    let base = Level::ALL
        .into_iter()
        .find_map(|lv| cx.baselines.get(lv).filter(|b| b.n >= cx.track.min_obs.get(lv)).map(|b| (lv, b)));
    let level = base.map(|(lv, _)| lv);
    let r = base.map(|(_, b)| ((v - b.mean) / (b.sd + params.eps)).tanh());
    let (win, loss) = match cx.duel {
        Some(d) if d.wins + d.losses >= params.duel_min => {
            let meetings = (d.wins + d.losses) as f64;
            (Some(d.wins as f64 / meetings), Some(d.losses as f64 / meetings))
        }
        _ => (None, None),
    };
    let prev = cx.prev.tier;

    let mut reasons = Vec::new();
    let tier = if cx.superseded {
        reasons.push("superseded".to_string());
        Tier::Discount
    } else if e < params.min_evidence {
        reasons.push("low_evidence".to_string());
        Tier::Unrated
    } else {
        let rank_high = r.is_some_and(|r| r >= params.enter);
        let rank_low = r.is_some_and(|r| r <= -params.enter);
        let duel_win = win.is_some_and(|w| w >= params.duel_rate);
        let duel_loss = loss.is_some_and(|l| l >= params.duel_rate);
        if v >= params.floor && (rank_high || duel_win) {
            reasons.push(if rank_high { "rank_high" } else { "duel_win" }.to_string());
            Tier::Premium
        } else if v <= -params.floor && (rank_low || duel_loss) {
            reasons.push(if rank_low { "rank_low" } else { "duel_loss" }.to_string());
            Tier::Discount
        } else if prev == Tier::Premium && v > 0.0 && (r.is_some_and(|r| r >= params.exit) || (r.is_none() && duel_win)) {
            reasons.push("held".to_string());
            Tier::Premium
        } else if prev == Tier::Discount && v < 0.0 && (r.is_some_and(|r| r <= -params.exit) || (r.is_none() && duel_loss)) {
            reasons.push("held".to_string());
            Tier::Discount
        } else {
            if rank_high || rank_low || duel_win || duel_loss {
                reasons.push("floor".to_string());
            }
            Tier::Neutral
        }
    };
    reasons.push(match level {
        Some(lv) => format!("level:{}", lv.name()),
        None => "no_baseline".to_string(),
    });

    let grade = match tier {
        Tier::Premium => Some(if r.is_some_and(|r| r >= params.strong) { 2 } else { 1 }),
        Tier::Discount => Some(if cx.superseded || r.is_some_and(|r| r <= -params.strong) {
            -2
        } else {
            -1
        }),
        Tier::Neutral => Some(0),
        Tier::Unrated => None,
    };
    let since_ms = if tier == prev { cx.prev.since_ms } else { cx.now_ms };
    Verdict {
        evidence: e,
        value: v,
        level,
        rank: r,
        tier,
        grade,
        since_ms,
        reasons,
    }
}

/// One contract case's input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Case {
    pub now_ms: i64,
    pub track: String,
    pub prev: Prev,
    pub obs: Vec<Obs>,
    #[serde(default)]
    pub baseline: Baselines,
    #[serde(default)]
    pub duel: Option<Duel>,
    #[serde(default)]
    pub superseded: bool,
}

/// Observations to verdict. `None` for an unknown track.
pub fn verdict(case: &Case, params: &Params) -> Option<Verdict> {
    let track = params.track(&case.track)?;
    let (e, v) = evidence(&case.obs, case.now_ms, track, params);
    let cx = Context {
        track,
        baselines: &case.baseline,
        prev: case.prev,
        now_ms: case.now_ms,
        duel: case.duel,
        superseded: case.superseded,
    };
    Some(judge(e, v, &cx, params))
}

/// Baseline snapshot (population standard deviation) of the judged values in one scope.
pub fn baseline_of(values: &[f64]) -> Baseline {
    let n = values.len();
    if n == 0 {
        return Baseline { n: 0, mean: 0.0, sd: 0.0 };
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    Baseline {
        n: n as u64,
        mean,
        sd: variance.sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
