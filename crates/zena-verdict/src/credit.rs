//! Credit attribution (Scenario 4 Part 2, `feedback_credit_v1` FC-05 … FC-07): which of the candidates the player
//! touched in the last 5 minutes (briefed records, passed checkpoints, numbered screen regions) earned a victory, a
//! defeat or a spoken grade, and how much.
//!
//! 1. **Gates before any matrix.** A shielded defeat (the player blamed their own hands, [`crate::feedback`])
//!    distributes nothing. A warned trap never shares a victory, nor a defeat unless the player named it. When a
//!    bottleneck (friction z ≥ 1) is in the window of a defeat, the checkpoints cruised through without friction
//!    (z ≤ 0) are protected (credit 0.0).
//! 2. **N × 4 matrix** `matrix[candidate][channel]` with every cell in \[0, 1] (SDS's absent marker −1.0 can never
//!    appear): contact × time `exposure · 2^(−age / 90 s)`, relevance (explicit mention 0.95, else `max(0, cos)`),
//!    causal friction fit (defeat: `tanh(max(0, z) / 1.5)`; victory: `1 / (1 + e^z)`; no friction: 0.5, or the
//!    segment's friction for a defeat) and, as the anchor column, the candidate's ambient prior. Each row is
//!    double-centred *alone* — its residual against its own ambient prior decides whether it is a candidate at all,
//!    so one candidate never degenerates to 0.0 — and the whole matrix is double-centred, whose residuals decide who
//!    stands out. Both are projected with 0.30 / 0.35 / 0.35 on the three event columns (SDS `double_center`) and
//!    read on the scale of the raw columns ([`attribution_scores`]).
//! 3. **Split by the number of valid candidates.** None → nobody (Flat). One → 1.0 (Decisive). Two → 1.0 to the first
//!    when the gap reaches τ₂ (SDS `credit.margin`, default 0.08), else a pair. Three or more → SDS `line_tie_shape`
//!    (a Decisive margin feeds `credit.margin`). A pair of complementary candidates splits 0.5 / 0.5 (synergy, no
//!    ledger entry); exclusive alternatives (same slot group) go to the historical winner (`tie_verdict`) or split
//!    0.5 / 0.5 and count a tie.
//! 4. **Repeat decay.** Repeated automatic outcomes of the same sign weigh 0.5ᵏ (floor 0.125); spoken and manual
//!    grades never decay.
//! 5. **Cross-trial duels.** An exclusive alternative that wins (credit ≥ 0.5) within 15 minutes after its rival lost
//!    (credit ≥ 0.5) beats it in the confusion ledger ([`CrossTrialMemory`] keeps the failures); the ledger becomes
//!    PD-Verdict [`Duel`]s ([`duel_for_item`]) and "outperforms" edges ([`outperforms`]).
//!
//! Observations keep `weight` (kind × strength × decay) and `credit` apart, as PD-Verdict multiplies them.

use crate::feedback::{ExtractedFeedback, ObsKind};
use crate::{Duel, Obs, Params};
pub use premium_discount_sds::matrix::{double_center, line_tie_shape, ABSENT};
pub use premium_discount_sds::{tie_verdict, ConfusionStat, RejectStat, ScopeStat, TieKind, Welford};
use serde::{Deserialize, Serialize};

/// Candidates older than this are out of the window (5 minutes).
pub const EXPOSURE_TTL_MS: u64 = 300_000;
/// Half-life of the contact column.
pub const EXPOSURE_HALF_LIFE_MS: f64 = 90_000.0;
/// How long a rival's failure waits for the alternative's win (15 minutes).
pub const CROSS_TRIAL_WINDOW_MS: i64 = 900_000;
/// Relevance of an explicitly named candidate.
pub const MENTION_RELEVANCE: f32 = 0.95;
/// A bottleneck: friction z at or above this.
pub const BOTTLENECK_Z: f64 = 1.0;
/// A cruised checkpoint: friction z at or below this.
pub const CRUISE_Z: f64 = 0.0;
/// The friction fit of a candidate without friction data.
pub const NEUTRAL_FIT: f32 = 0.5;
/// Projection weights of the three event columns (contact, relevance, friction).
pub const EVENT_WEIGHTS: [f32; 3] = [0.30, 0.35, 0.35];
/// τ₂: the gap that makes the first of two candidates decisive, until `credit.margin` has enough samples.
pub const PAIR_MARGIN: f32 = 0.08;
/// The SDS baseline axis of decisive margins.
pub const CREDIT_MARGIN_AXIS: &str = "credit.margin";
/// Decay per repeated automatic outcome of the same sign, and its floor.
pub const REPEAT_DECAY: f64 = 0.5;
pub const REPEAT_DECAY_FLOOR: f64 = 0.125;
/// The credit a win or a loss needs to count in a cross-trial duel.
pub const DUEL_CREDIT: f64 = 0.5;
/// Reject kind written by a ▼ pin ([`record_demotion`]).
pub const DEMOTED: &str = "demoted";

const MARGIN_RING: usize = 32;
const MARGIN_MIN_OBS: u64 = 5;
const VALID_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditTargetKind {
    /// A drop / chat / coach record Zena briefed.
    RagRecord,
    /// A time-trial checkpoint (Scenario 1).
    CourseCheckpoint,
    /// A numbered screen region (1–9).
    NavRegion,
}

/// How two candidates that tie at the top relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRelation {
    /// Used together (weapon + talisman, a guide + a checkpoint): 0.5 / 0.5 synergy.
    Complementary,
    /// Alternatives for one fork or slot (front gate vs right ladder): history decides, else a counted tie.
    MutuallyExclusive,
}

impl CandidateRelation {
    /// Exclusive when both name the same slot group, complementary otherwise.
    pub fn of(a: &AttributionCandidate, b: &AttributionCandidate) -> Self {
        match (&a.exclusive_slot_group, &b.exclusive_slot_group) {
            (Some(x), Some(y)) if x == y => CandidateRelation::MutuallyExclusive,
            _ => CandidateRelation::Complementary,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionCandidate {
    pub id: String,
    pub kind: CreditTargetKind,
    /// Fork / slot group: equal groups are exclusive alternatives, different (or none) complementary.
    pub exclusive_slot_group: Option<String>,
    /// Zena warned against it (an outdated version, a trap).
    pub is_warned_trap: bool,
    /// 0..1: how much of the briefing was heard (1.0 for a checkpoint passed).
    pub exposure_weight: f32,
    /// Since the contact (≤ [`EXPOSURE_TTL_MS`]).
    pub age_ms: u64,
    /// Cosine with the event context; negative values count as 0.
    pub semantic_sim: f32,
    /// The utterance named this candidate (`ExtractedFeedback::target_mention`, `feedback::mentioned_label`).
    pub matches_explicit_mention: bool,
    /// Friction z of the segment the candidate covers (None for a plain note).
    pub friction_z: Option<f64>,
    /// The anchor column: how present the candidate is regardless of this event (0.45 app-wide memo, 0.20 a
    /// checkpoint- or boss-specific record).
    pub ambient_prior: f32,
    /// Earlier automatic outcomes of the same sign on this candidate in this session and scope (k − 1).
    pub prior_same_sign_outcomes: u32,
}

impl AttributionCandidate {
    /// A fully heard, fresh candidate with no relevance or friction yet and a specific ambient prior (0.20).
    pub fn new(id: impl Into<String>, kind: CreditTargetKind) -> Self {
        Self {
            id: id.into(),
            kind,
            exclusive_slot_group: None,
            is_warned_trap: false,
            exposure_weight: 1.0,
            age_ms: 0,
            semantic_sim: 0.0,
            matches_explicit_mention: false,
            friction_z: None,
            ambient_prior: 0.20,
            prior_same_sign_outcomes: 0,
        }
    }
}

/// Why a candidate left before the matrix (its credit is 0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// A warned trap in a victory, or in a defeat nobody blamed it for.
    WarnedTrap,
    /// A checkpoint cruised through (z ≤ 0) while a bottleneck (z ≥ 1) caused the defeat.
    CruiseProtected,
    /// Older than [`EXPOSURE_TTL_MS`].
    Expired,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreditShare {
    pub target_id: String,
    pub target_kind: CreditTargetKind,
    /// 1.0 alone, 0.5 in a pair; the shares of one event never sum above 1.0.
    pub credit: f64,
    /// `weight(kind) × strength × credit × decay`.
    pub effective_weight: f64,
    pub sign: i32,
    #[serde(with = "tie_serde")]
    pub tie_kind: TieKind,
    /// The observation to store for the target (`weight` = kind × strength × decay, `credit` apart).
    pub obs: Obs,
}

/// A rival's recent failure for the cross-trial duel ([`CrossTrialMemory`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RivalFailure {
    pub id: String,
    pub slot_group: String,
    pub at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionReport {
    /// Candidates that received credit (everyone else got 0.0).
    pub shares: Vec<CreditShare>,
    /// Cross-trial duels recorded now: `(winner, loser, winner's decided win rate, outperforms active)`.
    pub duel_updates: Vec<(String, String, f32, bool)>,
    /// A defeat the player's self-blame shielded: nothing was distributed.
    pub outcome_shielded: bool,
    /// The shape of the decision (Flat when nobody got credit).
    #[serde(with = "tie_serde")]
    pub tie: TieKind,
    /// Candidates removed before the matrix.
    pub excluded: Vec<(String, Exclusion)>,
    /// Exclusive pairs recorded as an undecided meeting (`ties += 1`), for a `zena_sds::Ledger::record_tie` mirror.
    pub ties: Vec<(String, String)>,
    /// The decisive margin pushed to `credit.margin`.
    pub credit_margin: Option<f32>,
    /// Per eligible candidate: `(id, own residual, relative residual)`.
    pub scores: Vec<(String, f32, f32)>,
}

impl Default for AttributionReport {
    fn default() -> Self {
        Self {
            shares: Vec::new(),
            duel_updates: Vec::new(),
            outcome_shielded: false,
            tie: TieKind::Flat,
            excluded: Vec::new(),
            ties: Vec::new(),
            credit_margin: None,
            scores: Vec::new(),
        }
    }
}

impl AttributionReport {
    pub fn share(&self, id: &str) -> Option<&CreditShare> {
        self.shares.iter().find(|s| s.target_id == id)
    }

    /// The credit `id` received (0.0 when it got none).
    pub fn credit_of(&self, id: &str) -> f64 {
        self.share(id).map_or(0.0, |s| s.credit)
    }
}

/// `decisive`, `pair_confusion`, `flat`.
pub fn tie_name(t: TieKind) -> &'static str {
    match t {
        TieKind::Decisive => "decisive",
        TieKind::PairConfusion => "pair_confusion",
        TieKind::Flat => "flat",
    }
}

mod tie_serde {
    use super::{tie_name, TieKind};
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(t: &TieKind, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(tie_name(*t))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TieKind, D::Error> {
        let s = String::deserialize(d)?;
        [TieKind::Decisive, TieKind::PairConfusion, TieKind::Flat]
            .into_iter()
            .find(|t| tie_name(*t) == s)
            .ok_or_else(|| D::Error::custom(format!("unknown tie kind {s:?}")))
    }
}

fn unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn unit64(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn friction_fit(sign: i32, z: Option<f64>, segment_z: Option<f64>) -> f32 {
    let blame = |z: f64| (z.max(0.0) / 1.5).tanh() as f32;
    if sign < 0 {
        match z {
            Some(z) => blame(z),
            // A plain note inherits the segment's friction, at least neutral.
            None => segment_z.map_or(NEUTRAL_FIT, |s| blame(s).max(NEUTRAL_FIT)),
        }
    } else {
        z.map_or(NEUTRAL_FIT, |z| (1.0 / (1.0 + z.exp())) as f32)
    }
}

/// The raw N × 4 matrix (contact × time, relevance, friction fit, ambient prior) of an event of `sign`; every cell
/// in \[0, 1].
pub fn attribution_matrix(sign: i32, candidates: &[AttributionCandidate]) -> Vec<Vec<f32>> {
    let finite = |z: Option<f64>| z.filter(|z| z.is_finite());
    let segment_z = candidates
        .iter()
        .filter_map(|c| finite(c.friction_z))
        .fold(None, |m: Option<f64>, z| Some(m.map_or(z, |m| m.max(z))));
    candidates
        .iter()
        .map(|c| {
            let contact = unit(c.exposure_weight) * 2f64.powf(-(c.age_ms as f64) / EXPOSURE_HALF_LIFE_MS) as f32;
            let relevance = if c.matches_explicit_mention {
                MENTION_RELEVANCE
            } else {
                unit(c.semantic_sim)
            };
            let fit = friction_fit(sign, finite(c.friction_z), segment_z);
            vec![unit(contact), relevance, unit(fit), unit(c.ambient_prior)]
        })
        .collect()
}

/// `(own, relative)` residual scores of each row: `own` from the row double-centred alone (its event columns against
/// its own ambient anchor), `relative` from the whole matrix double-centred, both projected with [`EVENT_WEIGHTS`]
/// and scaled by the column count. Row centering takes a quarter of every column away, so the scale puts gaps back on
/// the scale of the raw event columns: the effective weights are 0.2 / 0.4 / 0.4 on the events and −1 on the ambient
/// anchor (`own > 0` ⇔ `0.2·contact + 0.4·relevance + 0.4·fit > ambient`). With one row both are equal.
pub fn attribution_scores(raw: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    const COLUMNS: f32 = 4.0;
    let project = |row: &[f32]| COLUMNS * row.iter().zip(EVENT_WEIGHTS).map(|(v, w)| v * w).sum::<f32>();
    let own = raw
        .iter()
        .map(|row| project(&double_center(std::slice::from_ref(row))[0]))
        .collect();
    let relative = double_center(raw).iter().map(|row| project(row)).collect();
    (own, relative)
}

/// τ₂: the usual decisive margin minus its spread (`credit.margin`, clamped to 0.04 … 0.16) once it has 5 samples,
/// [`PAIR_MARGIN`] before.
pub fn pair_margin(st: &ScopeStat) -> f32 {
    match st.baseline.get(CREDIT_MARGIN_AXIS) {
        Some(w) if w.n >= MARGIN_MIN_OBS => ((w.mean - w.sd()) as f32).clamp(PAIR_MARGIN * 0.5, PAIR_MARGIN * 2.0),
        _ => PAIR_MARGIN,
    }
}

fn pair_key(a: &str, b: &str) -> (String, bool) {
    if a <= b {
        (format!("{a}|{b}"), true)
    } else {
        (format!("{b}|{a}"), false)
    }
}

/// `winner` beat `loser` (`zena_sds::Ledger::record_confusion`, on a plain [`ScopeStat`]).
fn record_win(st: &mut ScopeStat, winner: &str, loser: &str, margin: f32) {
    if winner.is_empty() || loser.is_empty() || winner == loser {
        return;
    }
    let (key, winner_is_a) = pair_key(winner, loser);
    let e = st.confusion.entry(key).or_default();
    e.ties += 1;
    if winner_is_a {
        e.a_wins += 1;
    } else {
        e.b_wins += 1;
    }
    if margin.is_finite() {
        e.margin.push(margin as f64, MARGIN_RING);
    }
}

/// `a` and `b` met and neither won (`zena_sds::Ledger::record_tie`).
fn record_tie(st: &mut ScopeStat, a: &str, b: &str, margin: f32) {
    if a.is_empty() || b.is_empty() || a == b {
        return;
    }
    let e = st.confusion.entry(pair_key(a, b).0).or_default();
    e.ties += 1;
    if margin.is_finite() {
        e.margin.push(margin as f64, MARGIN_RING);
    }
}

/// Reject shares of both items, known as soon as either has a reject entry (a ▼ pin makes one).
fn specificities(st: &ScopeStat, a: &str, b: &str) -> (Option<f32>, Option<f32>) {
    if !st.reject.contains_key(a) && !st.reject.contains_key(b) {
        return (None, None);
    }
    let spec = |x: &str| st.reject.get(x).map_or(0.0, RejectStat::specificity);
    (Some(spec(a)), Some(spec(b)))
}

/// The historical winner of the exclusive pair, when `tie_verdict` lets the tie go its way (at least
/// `params.duel_min` meetings).
fn history_winner(st: &ScopeStat, a: &str, b: &str, params: &Params) -> Option<String> {
    let (key, a_is_x) = pair_key(a, b);
    let c = st.confusion.get(&key)?;
    if c.ties < params.duel_min as u64 {
        return None;
    }
    let (x, y) = if a_is_x { (a, b) } else { (b, a) };
    let (winner, loser, wins) = if c.a_wins >= c.b_wins { (x, y, c.a_wins) } else { (y, x, c.b_wins) };
    let rate = wins as f32 / c.ties.max(1) as f32;
    let (spec_w, spec_l) = specificities(st, winner, loser);
    tie_verdict(Some((true, rate)), spec_w, spec_l).then(|| winner.to_string())
}

/// `(winner's decided win rate, outperforms)` of the pair.
fn duel_rate(st: &ScopeStat, winner: &str, loser: &str, params: &Params) -> (f32, bool) {
    let (key, winner_is_a) = pair_key(winner, loser);
    let Some(c) = st.confusion.get(&key) else { return (0.0, false) };
    let (wins, losses) = if winner_is_a { (c.a_wins, c.b_wins) } else { (c.b_wins, c.a_wins) };
    let decided = wins + losses;
    if decided == 0 {
        return (0.0, false);
    }
    let rate = wins as f64 / decided as f64;
    (rate as f32, decided >= params.duel_min as u64 && rate >= params.duel_rate)
}

/// Whether `winner --outperforms--> loser`: at least `duel_min` decided meetings, won at `duel_rate` or more.
pub fn outperforms(st: &ScopeStat, winner: &str, loser: &str, params: &Params) -> bool {
    duel_rate(st, winner, loser, params).1
}

/// The PD-Verdict duel of `item`: decided wins and losses summed over every exclusive pair it is in (undecided ties
/// left out, so a synergy split never dilutes a win rate). `None` before its first decided meeting. Item ids must not
/// contain `|` (the ledger's key separator).
pub fn duel_for_item(st: &ScopeStat, item_id: &str) -> Option<Duel> {
    let (mut wins, mut losses) = (0u64, 0u64);
    for (key, c) in &st.confusion {
        let Some((a, b)) = key.split_once('|') else { continue };
        if a == item_id {
            wins += c.a_wins;
            losses += c.b_wins;
        } else if b == item_id {
            wins += c.b_wins;
            losses += c.a_wins;
        }
    }
    let narrow = |n: u64| n.min(u32::MAX as u64) as u32;
    (wins + losses > 0).then(|| Duel {
        wins: narrow(wins),
        losses: narrow(losses),
    })
}

/// A ▼ pin rejects `item_id` once (`demoted`), so ties never break toward it; un-pinning (`demoted == false`) takes
/// that reject back.
pub fn record_demotion(st: &mut ScopeStat, item_id: &str, demoted: bool, now_ms: i64) {
    if item_id.is_empty() {
        return;
    }
    if demoted {
        let e = st.reject.entry(item_id.to_string()).or_default();
        e.rejected += 1;
        *e.kinds.entry(DEMOTED.to_string()).or_insert(0) += 1;
    } else if let Some(e) = st.reject.get_mut(item_id) {
        if let Some(k) = e.kinds.get_mut(DEMOTED) {
            if *k > 0 {
                *k -= 1;
                e.rejected = e.rejected.saturating_sub(1);
            }
            if *k == 0 {
                e.kinds.remove(DEMOTED);
            }
        }
        if e.seen == 0 && e.rejected == 0 && e.kinds.is_empty() {
            st.reject.remove(item_id);
        }
    }
    st.updated_ms = st.updated_ms.max(now_ms);
}

/// What to do with a physical outcome given the feedback said around it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeGate {
    Distribute,
    /// A defeat the player blamed on their own hands: distribute nothing.
    Shielded,
    /// A victory followed by an explicit disapproval: only the disapproval is distributed.
    SuppressedByFeedback,
}

pub fn outcome_gate(outcome_sign: i32, feedback: Option<&ExtractedFeedback>) -> OutcomeGate {
    match feedback {
        Some(f) if outcome_sign < 0 && f.shield_defeat_outcome => OutcomeGate::Shielded,
        Some(f) if outcome_sign > 0 && matches!(f.evaluation, Some((ObsKind::Feedback, s, _)) if s < 0) => {
            OutcomeGate::SuppressedByFeedback
        }
        _ => OutcomeGate::Distribute,
    }
}

/// N × 4 double-centred credit attribution for one event of `event_kind` ("feedback" | "reaction" | "outcome" |
/// "manual" | "implicit") and `sign`, updating `scope_stat` (`credit.margin`, the confusion ledger) — see the module
/// docs. `strength` is the extracted strength (0.5 or 1.0); `shield_defeat` shields an `outcome` −1;
/// `recent_failures` are the rivals' failures of this session and (cc, to) ([`CrossTrialMemory::failures`]).
#[allow(clippy::too_many_arguments)]
pub fn attribute_and_update_duels(
    event_kind: &str,
    sign: i32,
    strength: f64,
    shield_defeat: bool,
    candidates: &[AttributionCandidate],
    scope_stat: &mut ScopeStat,
    recent_failures: &[RivalFailure],
    now_ms: i64,
    params: &Params,
) -> AttributionReport {
    let mut report = AttributionReport::default();
    let sign = sign.signum();
    let outcome = event_kind == "outcome";
    if sign == 0 || candidates.is_empty() {
        return report;
    }
    if outcome && sign < 0 && shield_defeat {
        report.outcome_shielded = true;
        return report;
    }

    let bottleneck = outcome && sign < 0 && candidates.iter().any(|c| c.friction_z.is_some_and(|z| z >= BOTTLENECK_Z));
    let mut eligible: Vec<AttributionCandidate> = Vec::new();
    for c in candidates {
        let why = if c.age_ms > EXPOSURE_TTL_MS {
            Some(Exclusion::Expired)
        } else if c.is_warned_trap && (sign > 0 || !c.matches_explicit_mention) {
            Some(Exclusion::WarnedTrap)
        } else if bottleneck && c.friction_z.is_some_and(|z| z <= CRUISE_Z) {
            Some(Exclusion::CruiseProtected)
        } else {
            None
        };
        match why {
            Some(w) => report.excluded.push((c.id.clone(), w)),
            None => eligible.push(c.clone()),
        }
    }

    let raw = attribution_matrix(sign, &eligible);
    let (own, rel) = attribution_scores(&raw);
    report.scores = eligible.iter().enumerate().map(|(f, c)| (c.id.clone(), own[f], rel[f])).collect();
    let mut valid: Vec<usize> = (0..eligible.len()).filter(|&f| own[f] > VALID_EPS).collect();
    valid.sort_by(|&a, &b| rel[b].total_cmp(&rel[a]).then(own[b].total_cmp(&own[a])).then(a.cmp(&b)));

    let mut assigned: Vec<(usize, f64)> = Vec::new();
    let mut pair = |top: usize, second: usize, margin: f32, st: &mut ScopeStat, report: &mut AttributionReport| {
        let (a, b) = (&eligible[top], &eligible[second]);
        let winner = match CandidateRelation::of(a, b) {
            CandidateRelation::Complementary => None,
            CandidateRelation::MutuallyExclusive => {
                let w = history_winner(st, &a.id, &b.id, params);
                if w.is_none() {
                    record_tie(st, &a.id, &b.id, margin);
                    report.ties.push((a.id.clone(), b.id.clone()));
                }
                w
            }
        };
        match winner {
            Some(w) => assigned.push((if w == a.id { top } else { second }, 1.0)),
            None => assigned.extend([(top, 0.5), (second, 0.5)]),
        }
    };
    let tie = match valid.len() {
        0 => TieKind::Flat,
        1 => {
            assigned.push((valid[0], 1.0));
            TieKind::Decisive
        }
        2 => {
            let (top, second) = (valid[0], valid[1]);
            let margin = rel[top] - rel[second];
            if margin >= pair_margin(scope_stat) {
                assigned.push((top, 1.0));
                TieKind::Decisive
            } else {
                pair(top, second, margin, scope_stat, &mut report);
                TieKind::PairConfusion
            }
        }
        _ => {
            // Shifted to start at 0: the shape is shift-invariant and no cell falls below the threshold.
            let low = valid.iter().map(|&f| rel[f]).fold(f32::INFINITY, f32::min);
            let proj: Vec<Vec<f32>> = valid.iter().map(|&f| vec![rel[f] - low]).collect();
            match line_tie_shape(&proj, 0, 0.0) {
                Some((TieKind::Decisive, top, _, m12, _)) => {
                    assigned.push((valid[top], 1.0));
                    scope_stat
                        .baseline
                        .entry(CREDIT_MARGIN_AXIS.to_string())
                        .or_default()
                        .push(m12 as f64, MARGIN_RING);
                    report.credit_margin = Some(m12);
                    TieKind::Decisive
                }
                Some((TieKind::PairConfusion, top, second, m12, _)) => {
                    pair(valid[top], valid[second], m12, scope_stat, &mut report);
                    TieKind::PairConfusion
                }
                _ => TieKind::Flat,
            }
        }
    };
    report.tie = tie;

    let automatic = matches!(event_kind, "outcome" | "implicit");
    let base = params.weight(event_kind) * unit64(strength);
    for (f, credit) in assigned {
        let c = &eligible[f];
        let decay = if automatic {
            REPEAT_DECAY.powi(c.prior_same_sign_outcomes.min(64) as i32).max(REPEAT_DECAY_FLOOR)
        } else {
            1.0
        };
        let weight = base * decay;
        report.shares.push(CreditShare {
            target_id: c.id.clone(),
            target_kind: c.kind,
            credit,
            effective_weight: weight * credit,
            sign,
            tie_kind: tie,
            obs: Obs {
                kind: event_kind.to_string(),
                sign,
                weight,
                credit,
                at_ms: now_ms,
            },
        });
    }

    if sign > 0 {
        for s in report.shares.iter().filter(|s| s.credit >= DUEL_CREDIT) {
            let Some(group) = eligible
                .iter()
                .find(|c| c.id == s.target_id)
                .and_then(|c| c.exclusive_slot_group.as_deref())
            else {
                continue;
            };
            let rival = recent_failures
                .iter()
                .filter(|r| r.id != s.target_id && r.slot_group == group && now_ms >= r.at_ms && now_ms - r.at_ms <= CROSS_TRIAL_WINDOW_MS)
                .max_by_key(|r| r.at_ms);
            if let Some(r) = rival {
                record_win(scope_stat, &s.target_id, &r.id, f32::NAN);
                let (rate, active) = duel_rate(scope_stat, &s.target_id, &r.id, params);
                report.duel_updates.push((s.target_id.clone(), r.id.clone(), rate, active));
            }
        }
    }
    if report.credit_margin.is_some() || !report.ties.is_empty() || !report.duel_updates.is_empty() {
        scope_stat.updated_ms = scope_stat.updated_ms.max(now_ms);
    }
    report
}

/// The rivals' failures a cross-trial duel looks back on: keep one per session and (cc, to).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossTrialMemory {
    pub failures: Vec<RivalFailure>,
}

impl CrossTrialMemory {
    pub fn failures(&self) -> &[RivalFailure] {
        &self.failures
    }

    /// After an attribution over `candidates` at `now_ms`: each duel it recorded uses up its rival's failure, its own
    /// losses (sign −1, credit ≥ 0.5, in a slot group) are remembered, and failures older than 15 minutes are
    /// forgotten.
    pub fn remember(&mut self, report: &AttributionReport, candidates: &[AttributionCandidate], now_ms: i64) {
        let group_of = |id: &str| candidates.iter().find(|c| c.id == id).and_then(|c| c.exclusive_slot_group.clone());
        for (winner, loser, _, _) in &report.duel_updates {
            let group = group_of(winner);
            let used = self
                .failures
                .iter()
                .enumerate()
                .filter(|(_, f)| &f.id == loser && Some(&f.slot_group) == group.as_ref())
                .max_by_key(|(_, f)| f.at_ms)
                .map(|(i, _)| i);
            if let Some(i) = used {
                self.failures.remove(i);
            }
        }
        for s in report.shares.iter().filter(|s| s.sign < 0 && s.credit >= DUEL_CREDIT) {
            if let Some(slot_group) = group_of(&s.target_id) {
                self.failures.push(RivalFailure {
                    id: s.target_id.clone(),
                    slot_group,
                    at_ms: now_ms,
                });
            }
        }
        self.failures.retain(|f| now_ms - f.at_ms <= CROSS_TRIAL_WINDOW_MS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::{extract_pragmatic_feedback, FeedbackContext};

    fn p() -> Params {
        Params::v2()
    }

    fn checkpoint(id: &str, z: f64, age_ms: u64) -> AttributionCandidate {
        AttributionCandidate {
            friction_z: Some(z),
            age_ms,
            semantic_sim: 0.5,
            ..AttributionCandidate::new(id, CreditTargetKind::CourseCheckpoint)
        }
    }

    fn record(id: &str, age_ms: u64, sim: f32) -> AttributionCandidate {
        AttributionCandidate {
            age_ms,
            semantic_sim: sim,
            ..AttributionCandidate::new(id, CreditTargetKind::RagRecord)
        }
    }

    fn run(kind: &str, sign: i32, cands: &[AttributionCandidate], st: &mut ScopeStat) -> AttributionReport {
        attribute_and_update_duels(kind, sign, 1.0, false, cands, st, &[], 1_000_000, &p())
    }

    #[test]
    fn fc05_cruise_checkpoint_is_protected_and_the_bottleneck_takes_the_blame() {
        let a = checkpoint("cp-a", -0.8, 40_000);
        let b = checkpoint("cp-b", 2.1, 5_000);
        let mut st = ScopeStat::default();
        let r = run("outcome", -1, &[a.clone(), b.clone()], &mut st);
        assert_eq!(r.credit_of("cp-a"), 0.0, "the cruised checkpoint is protected");
        assert_eq!(r.excluded, vec![("cp-a".to_string(), Exclusion::CruiseProtected)]);
        let s = r.share("cp-b").unwrap();
        assert_eq!(
            (s.credit, s.sign, s.tie_kind),
            (1.0, -1, TieKind::Decisive),
            "−1.0 on the bottleneck alone"
        );
        assert_eq!(
            (s.obs.kind.as_str(), s.obs.sign, s.obs.weight, s.obs.credit),
            ("outcome", -1, 1.0, 1.0)
        );
        // No 0.0 degeneration and no absent marker: one row double-centred against its ambient anchor.
        let (id, own, rel) = &r.scores[0];
        assert_eq!(id, "cp-b");
        assert!(*own > 0.05 && own == rel && *own != ABSENT, "{own} {rel}");
        let raw = attribution_matrix(-1, std::slice::from_ref(&b));
        assert!(raw[0].iter().all(|v| (0.0..=1.0).contains(v)), "{raw:?}");
        assert!(double_center(&raw)[0].iter().all(|v| v.is_finite() && *v != ABSENT));
        assert!(raw[0][2] > 0.85, "tanh(2.1 / 1.5) friction fit: {raw:?}");
        // Without a bottleneck nobody is protected.
        let mild = run("outcome", -1, &[a.clone(), checkpoint("cp-c", 0.6, 5_000)], &mut st);
        assert!(mild.excluded.is_empty());
        // A victory protects nobody either.
        assert!(run("outcome", 1, &[a, b], &mut st).excluded.is_empty());
    }

    #[test]
    fn fc06_warned_trap_is_excluded_before_the_matrix() {
        let slot1 = record("slot-1", 120_000, 0.7);
        let slot3 = AttributionCandidate {
            is_warned_trap: true,
            ..record("slot-3", 120_000, 0.75)
        };
        let mut st = ScopeStat::default();
        let r = run("outcome", 1, &[slot1.clone(), slot3.clone()], &mut st);
        assert_eq!(r.credit_of("slot-3"), 0.0);
        assert_eq!(r.excluded, vec![("slot-3".to_string(), Exclusion::WarnedTrap)]);
        let s = r.share("slot-1").unwrap();
        assert_eq!((s.credit, s.sign, s.tie_kind), (1.0, 1, TieKind::Decisive));
        assert_eq!(r.scores.len(), 1, "the trap never entered the matrix");
        // A defeat nobody blamed it for leaves it out too; naming it lets it take the blame.
        assert_eq!(
            run("outcome", -1, &[slot1.clone(), slot3.clone()], &mut st).credit_of("slot-3"),
            0.0
        );
        let named = AttributionCandidate {
            matches_explicit_mention: true,
            ..slot3
        };
        let r = run("feedback", -1, &[slot1, named], &mut st);
        assert_eq!(r.credit_of("slot-3"), 1.0, "{r:?}");
    }

    #[test]
    fn fc07_two_complementary_candidates_split_half_and_half() {
        let weapon = AttributionCandidate {
            exclusive_slot_group: Some("weapon".into()),
            ..record("weapon-guide", 60_000, 0.60)
        };
        let talisman = AttributionCandidate {
            exclusive_slot_group: Some("talisman".into()),
            ..record("talisman-guide", 62_000, 0.62)
        };
        let mut st = ScopeStat::default();
        let r = run("outcome", 1, &[weapon, talisman], &mut st);
        assert_eq!(r.tie, TieKind::PairConfusion, "N = 2 still reaches the pair rule");
        assert_eq!((r.credit_of("weapon-guide"), r.credit_of("talisman-guide")), (0.5, 0.5));
        assert!(r.shares.iter().map(|s| s.credit).sum::<f64>() <= 1.0);
        assert!(
            st.confusion.is_empty() && r.ties.is_empty(),
            "synergy leaves the confusion ledger alone"
        );
        assert_eq!(r.share("weapon-guide").unwrap().effective_weight, 0.5);
    }

    #[test]
    fn fc07_exclusive_rival_three_losses_then_three_wins_outperforms() {
        let params = p();
        let route = |id: &str| AttributionCandidate {
            exclusive_slot_group: Some("gate-2".into()),
            ..record(id, 30_000, 0.6)
        };
        let (a, b) = (route("route-a"), route("route-b"));
        let mut st = ScopeStat::default();
        let mut memory = CrossTrialMemory::default();
        let mut t = 10_000_000;
        for _ in 0..3 {
            let lost = attribute_and_update_duels(
                "outcome",
                -1,
                1.0,
                false,
                std::slice::from_ref(&a),
                &mut st,
                memory.failures(),
                t,
                &params,
            );
            assert_eq!(lost.credit_of("route-a"), 1.0);
            memory.remember(&lost, std::slice::from_ref(&a), t);
            t += 60_000;
        }
        assert_eq!(memory.failures().len(), 3);
        let mut last = AttributionReport::default();
        for _ in 0..3 {
            last = attribute_and_update_duels(
                "outcome",
                1,
                1.0,
                false,
                std::slice::from_ref(&b),
                &mut st,
                memory.failures(),
                t,
                &params,
            );
            assert_eq!(last.credit_of("route-b"), 1.0);
            memory.remember(&last, std::slice::from_ref(&b), t);
            t += 60_000;
        }
        assert!(memory.failures().is_empty(), "each failure meets one win");
        assert_eq!(duel_for_item(&st, "route-b"), Some(Duel { wins: 3, losses: 0 }));
        assert_eq!(duel_for_item(&st, "route-a"), Some(Duel { wins: 0, losses: 3 }));
        assert!(outperforms(&st, "route-b", "route-a", &params), "B --outperforms--> A");
        assert!(!outperforms(&st, "route-a", "route-b", &params));
        assert_eq!(last.duel_updates, vec![("route-b".to_string(), "route-a".to_string(), 1.0, true)]);
        let c = &st.confusion["route-a|route-b"];
        assert_eq!((c.ties, c.a_wins, c.b_wins), (3, 0, 3));
        // The duel feeds PD-Verdict: three meetings, rate 1.0 → duel_win without a baseline.
        let duel = duel_for_item(&st, "route-b");
        let baselines = crate::Baselines::default();
        let cx = crate::Context {
            track: params.track("coach").unwrap(),
            baselines: &baselines,
            prev: crate::Prev {
                tier: crate::Tier::Unrated,
                since_ms: 0,
            },
            now_ms: t,
            duel,
            superseded: false,
        };
        let v = crate::judge(3.0, 0.6, &cx, &params);
        assert_eq!((v.tier, v.reasons[0].as_str()), (crate::Tier::Premium, "duel_win"));
    }

    #[test]
    fn a_failure_older_than_fifteen_minutes_or_from_another_slot_is_no_duel() {
        let params = p();
        let mut st = ScopeStat::default();
        let b = AttributionCandidate {
            exclusive_slot_group: Some("gate-2".into()),
            ..record("route-b", 10_000, 0.6)
        };
        let old = [RivalFailure {
            id: "route-a".into(),
            slot_group: "gate-2".into(),
            at_ms: 0,
        }];
        let r = attribute_and_update_duels(
            "outcome",
            1,
            1.0,
            false,
            std::slice::from_ref(&b),
            &mut st,
            &old,
            CROSS_TRIAL_WINDOW_MS + 1,
            &params,
        );
        assert!(r.duel_updates.is_empty() && st.confusion.is_empty());
        let other = [RivalFailure {
            id: "route-x".into(),
            slot_group: "gate-9".into(),
            at_ms: 0,
        }];
        let r = attribute_and_update_duels("outcome", 1, 1.0, false, &[b], &mut st, &other, 1_000, &params);
        assert!(r.duel_updates.is_empty());
        assert_eq!(duel_for_item(&st, "route-b"), None);
    }

    #[test]
    fn nothing_relevant_is_flat_and_three_candidates_use_the_tie_shape() {
        let mut st = ScopeStat::default();
        let stale = AttributionCandidate {
            ambient_prior: 0.45,
            ..record("old-memo", 290_000, 0.2)
        };
        let r = run("outcome", 1, &[stale], &mut st);
        assert_eq!(
            (r.tie, r.shares.len()),
            (TieKind::Flat, 0),
            "an unrelated memo takes no credit: {r:?}"
        );
        let expired = record("ancient", EXPOSURE_TTL_MS + 1, 0.9);
        assert_eq!(run("outcome", 1, &[expired], &mut st).excluded[0].1, Exclusion::Expired);

        let star = AttributionCandidate {
            matches_explicit_mention: true,
            ..record("star", 5_000, 0.9)
        };
        let r = run(
            "feedback",
            1,
            &[star, record("x", 100_000, 0.45), record("y", 100_000, 0.47)],
            &mut st,
        );
        assert_eq!((r.tie, r.credit_of("star")), (TieKind::Decisive, 1.0), "{r:?}");
        let m = r.credit_margin.unwrap();
        assert!(m > 0.0);
        assert_eq!(st.baseline[CREDIT_MARGIN_AXIS].n, 1);
    }

    #[test]
    fn exclusive_ties_count_a_meeting_unless_history_or_a_demotion_decides() {
        let params = p();
        let alt = |id: &str, sim: f32| AttributionCandidate {
            exclusive_slot_group: Some("fork".into()),
            ..record(id, 20_000, sim)
        };
        let cands = [alt("left", 0.80), alt("right", 0.81), record("z-far", 280_000, 0.05)];
        let mut st = ScopeStat::default();
        let r = attribute_and_update_duels("outcome", 1, 1.0, false, &cands[..2], &mut st, &[], 0, &params);
        assert_eq!(
            (r.tie, r.credit_of("left"), r.credit_of("right")),
            (TieKind::PairConfusion, 0.5, 0.5)
        );
        assert_eq!(r.ties, vec![("right".to_string(), "left".to_string())]);
        assert_eq!(st.confusion["left|right"].ties, 1);
        // History: right won 4 of 5 meetings → the tie goes to right.
        let c = st.confusion.get_mut("left|right").unwrap();
        (c.ties, c.b_wins) = (5, 4);
        let r = attribute_and_update_duels("outcome", 1, 1.0, false, &cands[..2], &mut st, &[], 0, &params);
        assert_eq!((r.credit_of("right"), r.credit_of("left")), (1.0, 0.0));
        assert_eq!(st.confusion["left|right"].ties, 5, "a decided tie is not a new meeting");
        // A ▼ pin on right: rejected more often than left by more than the slack → back to a counted tie.
        record_demotion(&mut st, "right", true, 5);
        let r = attribute_and_update_duels("outcome", 1, 1.0, false, &cands[..2], &mut st, &[], 0, &params);
        assert_eq!((r.credit_of("right"), r.credit_of("left")), (0.5, 0.5));
        record_demotion(&mut st, "right", false, 6);
        assert!(!st.reject.contains_key("right"), "un-pinning takes the reject back");
        assert_eq!(st.updated_ms, 6);
    }

    #[test]
    fn repeat_decay_applies_to_automatic_outcomes_only() {
        let mut st = ScopeStat::default();
        let tired = AttributionCandidate {
            prior_same_sign_outcomes: 2,
            ..record("farm", 10_000, 0.8)
        };
        assert_eq!(run("outcome", 1, std::slice::from_ref(&tired), &mut st).shares[0].obs.weight, 0.25);
        let worn = AttributionCandidate {
            prior_same_sign_outcomes: 9,
            ..tired.clone()
        };
        assert_eq!(run("outcome", 1, &[worn], &mut st).shares[0].obs.weight, REPEAT_DECAY_FLOOR);
        assert_eq!(
            run("feedback", 1, std::slice::from_ref(&tired), &mut st).shares[0].obs.weight,
            1.0,
            "spoken: no decay"
        );
        assert_eq!(
            run("manual", -1, std::slice::from_ref(&tired), &mut st).shares[0].obs.weight,
            2.0,
            "manual: no decay"
        );
        let half = attribute_and_update_duels("feedback", 1, 0.5, false, &[tired], &mut st, &[], 0, &p());
        assert_eq!(
            (half.shares[0].obs.weight, half.shares[0].obs.credit),
            (0.5, 1.0),
            "strength in weight, credit apart"
        );
    }

    #[test]
    fn a_shielded_defeat_distributes_nothing_and_the_gate_reads_the_feedback() {
        let mut st = ScopeStat::default();
        let r = attribute_and_update_duels("outcome", -1, 1.0, true, &[record("g", 1_000, 0.9)], &mut st, &[], 0, &p());
        assert!(r.outcome_shielded && r.shares.is_empty());
        let spoken = attribute_and_update_duels("feedback", -1, 1.0, true, &[record("g", 1_000, 0.9)], &mut st, &[], 0, &p());
        assert!(!spoken.outcome_shielded, "only physical outcomes are shielded");
        let defeat = FeedbackContext {
            recent_defeat_within_15s: true,
            ..Default::default()
        };
        let blame = extract_pragmatic_feedback("공략은 맞는데 내 손이 꼬였어", &[], "ko", defeat).unwrap();
        assert_eq!(outcome_gate(-1, Some(&blame)), OutcomeGate::Shielded);
        let disapproval = extract_pragmatic_feedback("깨긴 했는데 비추", &[], "ko", FeedbackContext::default()).unwrap();
        assert_eq!(outcome_gate(1, Some(&disapproval)), OutcomeGate::SuppressedByFeedback);
        assert_eq!(outcome_gate(1, None), OutcomeGate::Distribute);
    }

    #[test]
    fn broken_inputs_stay_inside_the_unit_square() {
        let odd = AttributionCandidate {
            exposure_weight: f32::NAN,
            semantic_sim: -0.4,
            ambient_prior: 3.0,
            friction_z: Some(f64::INFINITY),
            ..record("odd", 0, 0.0)
        };
        let raw = attribution_matrix(1, &[odd]);
        assert!(raw[0].iter().all(|v| (0.0..=1.0).contains(v)), "{raw:?}");
        assert_eq!(raw[0][1], 0.0, "a negative cosine is no relevance");
        assert_eq!(pair_margin(&ScopeStat::default()), PAIR_MARGIN);
        let mut st = ScopeStat::default();
        for _ in 0..5 {
            st.baseline.entry(CREDIT_MARGIN_AXIS.into()).or_default().push(0.3, 32);
        }
        assert_eq!(pair_margin(&st), PAIR_MARGIN * 2.0, "clamped");
        assert_eq!(duel_for_item(&st, "nobody"), None);
        let json = serde_json::to_value(run("outcome", 1, &[record("k", 0, 0.9)], &mut st)).unwrap();
        assert_eq!(
            (json["tie"].as_str(), json["shares"][0]["tie_kind"].as_str()),
            (Some("decisive"), Some("decisive"))
        );
        let back: AttributionReport = serde_json::from_value(json).unwrap();
        assert_eq!(back.tie, TieKind::Decisive);
    }
}
