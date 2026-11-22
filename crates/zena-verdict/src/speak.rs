//! What Zena may say about a recommendation: the local verdict (my own record) and the global one (other
//! users, premium-discount's server) side by side. They are never mixed into one number: "it worked for me but
//! the others say it does not" is the most useful thing to say, and a blend would hide it.
//!
//! | local | global | Zena says |
//! |---|---|---|
//! | judged | same tier | both agree |
//! | judged | other tier | both, explicitly |
//! | judged | unrated | the local verdict only |
//! | unrated | judged | "the others …" |
//! | unrated | unrated | facts only: never "recommended" / "not recommended" |
//!
//! On top of [`Stance`] (Scenario 4 Part 3, `feedback_credit_v1` FC-08 · FC-09), [`VerdictPairView`] is what the prompt
//! builder and the Stage 2 drawer share:
//!
//! * a [`Stance::Differ`] splits into four [`DifferKind`]s (all six ordered pairs of judged tiers);
//! * [`MemoryFreshness::FadedRecall`]: a past Premium / Discount whose evidence only decayed (1.0 ≤ E < 2.0, no
//!   opposite vote since) is recalled instead of forgotten. The confirmation lives apart from PD-Verdict's `prev`
//!   ([`LocalVerdictEntry::last_confirmed`]), so it survives every later query;
//! * [`MemoryFreshness::PinnedByUser`]: a manual ▲ / ▼ ([`toggle_pin`]) overrides the statistics at once and a second
//!   click restores them; pins and local verdicts never go to the server ([`outbound_votes`]);
//! * recipes are never mixed: the view holds each side's tier, grade and recipe, never a raw value or rank, and the
//!   local side is always `pd-verdict/2` (`Params::v2`), the global one what the server says (`pd-verdict/1` when a
//!   search hit carries no recipe).

use crate::feedback::manual_obs;
use crate::{verdict, Case, Obs, Params, Prev, Tier, Verdict, RECIPE_V1, RECIPE_V2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stance {
    Agree(Tier),
    Differ {
        local: Tier,
        global: Tier,
    },
    LocalOnly(Tier),
    GlobalOnly(Tier),
    /// Neither side has enough evidence.
    FactsOnly,
}

fn judged(t: Option<Tier>) -> Option<Tier> {
    t.filter(|t| *t != Tier::Unrated)
}

pub fn stance(local: Option<Tier>, global: Option<Tier>) -> Stance {
    match (judged(local), judged(global)) {
        (Some(l), Some(g)) if l == g => Stance::Agree(l),
        (Some(l), Some(g)) => Stance::Differ { local: l, global: g },
        (Some(l), None) => Stance::LocalOnly(l),
        (None, Some(g)) => Stance::GlobalOnly(g),
        (None, None) => Stance::FactsOnly,
    }
}

/// 추천 / 비추천 / 보통 (판정 전은 말하지 않으므로 빈 문자열)
pub fn word(t: Tier) -> &'static str {
    match t {
        Tier::Premium => "추천",
        Tier::Discount => "비추천",
        Tier::Neutral => "보통",
        Tier::Unrated => "",
    }
}

impl Stance {
    /// One context line for the model (Korean, as the rest of Zena's prompt), about the record `label`.
    pub fn line(self, label: &str) -> String {
        match self {
            Stance::Agree(t) => format!("{label}: 내 기록과 서버 평가 모두 {}", word(t)),
            Stance::Differ { local, global } => {
                format!("{label}: 내 기록에선 {}인데 서버에선 {} (둘 다 말할 것)", word(local), word(global))
            }
            Stance::LocalOnly(t) => format!("{label}: 내 기록으로는 {} (서버 판정 전)", word(t)),
            Stance::GlobalOnly(t) => format!("{label}: 다른 사용자들은 {} (내 기록은 판정 전)", word(t)),
            Stance::FactsOnly => format!("{label}: 판정 전 (추천·비추천이라고 말하지 말고 있었던 일만)"),
        }
    }

    /// The character emote the briefing opens with (`zena_core::tools::CharacterEmote` names).
    pub fn emote(self) -> Option<&'static str> {
        match self {
            Stance::Differ { .. } => Some("QUESTION"),
            Stance::Agree(t) | Stance::LocalOnly(t) | Stance::GlobalOnly(t) => match t {
                Tier::Premium => Some("STAR_EYES"),
                Tier::Discount => Some("SWEAT_PANIC"),
                _ => None,
            },
            Stance::FactsOnly => None,
        }
    }

    /// The finer meaning of a [`Stance::Differ`]; `None` for every other stance.
    pub fn differ_kind(self) -> Option<DifferKind> {
        match self {
            Stance::Differ { local, global } => DifferKind::of(local, global),
            _ => None,
        }
    }
}

/// Both sides judged (premium, neutral or discount) but different: the six ordered pairs in four meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferKind {
    /// Local premium, global discount: others avoid it, it works for me (my own trick).
    WorksForMeOnly,
    /// Local discount, global premium: the standard way that failed for me (suggest a detour first).
    FailsForMe,
    /// Local premium or discount, global neutral: the crowd is split, my record is clear.
    LocalClearGlobalSplit,
    /// Local neutral, global premium or discount: my record is mixed, the crowd leans.
    LocalUndecided,
}

impl DifferKind {
    pub fn of(local: Tier, global: Tier) -> Option<DifferKind> {
        use Tier::*;
        match (local, global) {
            (Premium, Discount) => Some(DifferKind::WorksForMeOnly),
            (Discount, Premium) => Some(DifferKind::FailsForMe),
            (Premium | Discount, Neutral) => Some(DifferKind::LocalClearGlobalSplit),
            (Neutral, Premium | Discount) => Some(DifferKind::LocalUndecided),
            _ => None,
        }
    }
}

/// A manual ▲ / ▼ in the history view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserPin {
    /// ▲: recommended, whatever the statistics say.
    Promoted,
    /// ▼: not recommended (blocked), whatever the statistics say.
    Demoted,
}

impl UserPin {
    pub fn tier(self) -> Tier {
        match self {
            UserPin::Promoted => Tier::Premium,
            UserPin::Demoted => Tier::Discount,
        }
    }

    /// The strongest grade of the pinned tier (+2 / −2).
    pub fn grade(self) -> i32 {
        match self {
            UserPin::Promoted => 2,
            UserPin::Demoted => -2,
        }
    }

    /// The sign of the pin's `manual` observation.
    pub fn sign(self) -> i32 {
        self.grade().signum()
    }
}

/// Why the local side reads as it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryFreshness {
    /// The current verdict as judged (E ≥ 2.0, or unrated with nothing to recall).
    Active,
    /// A past premium / discount whose evidence only decayed (1.0 ≤ E < 2.0, no opposite vote since).
    FadedRecall { past_tier: Tier, past_grade: i32 },
    /// The user's ▲ / ▼ overrides the statistics.
    PinnedByUser(UserPin),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictLocale {
    Ko,
    En,
    Ja,
}

impl VerdictLocale {
    /// "ko", "ko-KR", "korean" → Ko; "en…" / "english" → En; "ja…" / "japanese" → Ja; anything else Ko.
    pub fn from_code(code: &str) -> VerdictLocale {
        let c = code.trim().to_lowercase();
        if c.starts_with("en") {
            VerdictLocale::En
        } else if c.starts_with("ja") || c.starts_with("jp") {
            VerdictLocale::Ja
        } else {
            VerdictLocale::Ko
        }
    }
}

/// The evidence below which a past verdict is forgotten rather than recalled.
pub const FADED_RECALL_MIN_EVIDENCE: f64 = 1.0;

/// A verdict that was premium or discount once (kept apart from PD-Verdict's `prev`, which every judgement
/// overwrites).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmedTierRecord {
    pub tier: Tier,
    pub grade: i32,
    pub confirmed_at_ms: i64,
}

/// The local state of one record between judgements: PD-Verdict's `prev` (hysteresis) and the last confirmed tier
/// (FadedRecall). Store it with the record's observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalVerdictEntry {
    pub prev: Prev,
    #[serde(default)]
    pub last_confirmed: Option<ConfirmedTierRecord>,
}

impl Default for LocalVerdictEntry {
    fn default() -> Self {
        Self {
            prev: Prev {
                tier: Tier::Unrated,
                since_ms: 0,
            },
            last_confirmed: None,
        }
    }
}

/// No observation against the confirmed tier arrived after it was confirmed.
pub fn no_opposite_since(obs: &[Obs], confirmed: &ConfirmedTierRecord) -> bool {
    let past = match confirmed.tier {
        Tier::Premium => 1,
        Tier::Discount => -1,
        _ => return true,
    };
    !obs.iter().any(|o| o.at_ms > confirmed.confirmed_at_ms && o.sign.signum() == -past)
}

/// The five guards of a faded recall: the verdict is unrated for low evidence only (not superseded), its evidence
/// is still `1.0 ≤ E < min_evidence`, there is a past premium / discount, the value still points its way
/// (`v ≥ +floor` / `v ≤ −floor`), and no opposite vote arrived since.
fn recall(v: &Verdict, past_tier: Tier, past_grade: i32, no_opposite: bool, params: &Params) -> Option<(Tier, i32)> {
    let has = |r: &str| v.reasons.iter().any(|x| x == r);
    let low = v.tier == Tier::Unrated && has("low_evidence") && !has("superseded");
    let fading = v.evidence >= FADED_RECALL_MIN_EVIDENCE && v.evidence < params.min_evidence;
    let consistent = match past_tier {
        Tier::Premium => v.value >= params.floor,
        Tier::Discount => v.value <= -params.floor,
        _ => false,
    };
    (low && fading && consistent && no_opposite).then_some((past_tier, past_grade))
}

impl LocalVerdictEntry {
    /// Takes in a verdict judged with `self.prev` at `now_ms` from `obs`: a premium / discount (not superseded) is
    /// confirmed, a neutral forgets the confirmation, an unrated one is recalled when the five guards hold. `prev`
    /// becomes the verdict's tier, as PD-Verdict's hysteresis needs.
    pub fn observe(&mut self, v: &Verdict, obs: &[Obs], now_ms: i64, params: &Params) -> MemoryFreshness {
        self.prev = Prev {
            tier: v.tier,
            since_ms: v.since_ms,
        };
        let superseded = v.reasons.iter().any(|r| r == "superseded");
        match v.tier {
            Tier::Premium | Tier::Discount if !superseded => {
                self.last_confirmed = Some(ConfirmedTierRecord {
                    tier: v.tier,
                    grade: v.grade.unwrap_or(0),
                    confirmed_at_ms: now_ms,
                });
                MemoryFreshness::Active
            }
            Tier::Neutral => {
                self.last_confirmed = None;
                MemoryFreshness::Active
            }
            Tier::Unrated => self
                .last_confirmed
                .and_then(|c| recall(v, c.tier, c.grade, no_opposite_since(obs, &c), params))
                .map_or(MemoryFreshness::Active, |(past_tier, past_grade)| MemoryFreshness::FadedRecall {
                    past_tier,
                    past_grade,
                }),
            _ => MemoryFreshness::Active,
        }
    }

    /// Judges `case` with this entry's `prev` (the case's own `prev` is ignored) and takes the verdict in.
    pub fn judge(&mut self, case: &Case, params: &Params) -> Option<(Verdict, MemoryFreshness)> {
        let mut case = case.clone();
        case.prev = self.prev;
        let v = verdict(&case, params)?;
        let freshness = self.observe(&v, &case.obs, case.now_ms, params);
        Some((v, freshness))
    }

    /// The `last_confirmed` argument of [`VerdictPairView::resolve`].
    pub fn recall_input(&self, obs: &[Obs]) -> Option<(Tier, i32, bool)> {
        self.last_confirmed.map(|c| (c.tier, c.grade, no_opposite_since(obs, &c)))
    }
}

/// What a ▲ / ▼ click did ([`toggle_pin`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PinToggle {
    /// The pin now in force.
    pub pin: Option<UserPin>,
    /// The `manual` observation appended (none when the click only released a pin).
    pub recorded: Option<Obs>,
    /// Earlier pin observations turned into withdrawals (`sign` 0) in place.
    pub withdrawn: usize,
}

/// A click on ▲ / ▼ for one record, like the desktop vote buttons: a first click pins (a `manual` ±1, weight 2.0,
/// appended to `history`); clicking the pinned button again releases it (its observation becomes `sign` 0, so the
/// statistics are exactly as before); clicking the other button switches. Local only: nothing here is a server vote.
pub fn toggle_pin(history: &mut Vec<Obs>, current: Option<UserPin>, clicked: UserPin, now_ms: i64, params: &Params) -> PinToggle {
    let mut withdrawn = 0;
    if let Some(pin) = current {
        if let Some(i) = history.iter().rposition(|o| o.kind == "manual" && o.sign == pin.sign()) {
            history[i].sign = 0;
            withdrawn = 1;
        }
        if pin == clicked {
            return PinToggle {
                pin: None,
                recorded: None,
                withdrawn,
            };
        }
    }
    let obs = manual_obs(clicked == UserPin::Promoted, now_ms, params);
    history.push(obs.clone());
    PinToggle {
        pin: Some(clicked),
        recorded: Some(obs),
        withdrawn,
    }
}

/// Observation kinds the server counts (`eval_votes`); everything else (`manual`, `feedback`, `reaction`, `outcome`,
/// `implicit`, …) and every local verdict stay on the device.
pub const SERVER_VOTE_KINDS: &[&str] = &["vote", "vote_outcome"];

pub fn is_server_vote_kind(kind: &str) -> bool {
    SERVER_VOTE_KINDS.contains(&kind)
}

/// The observations a sync loop may send at all (server vote kinds only; consent gates apply on top).
pub fn outbound_votes(obs: &[Obs]) -> Vec<&Obs> {
    obs.iter().filter(|o| is_server_vote_kind(&o.kind)).collect()
}

/// The server's verdict of a record as [`VerdictPairView::resolve`] takes it: `(tier, grade, up / down votes, recipe)`
/// (a `VerdictDto` row, or a search hit's `MatchVerdict` with an empty recipe).
pub type GlobalVerdict<'a> = (Tier, Option<i32>, Option<(u32, u32)>, &'a str);

/// The local and global verdicts side by side, never blended: one line for the prompt, the Stage 2 drawer's chips.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictPairView {
    /// Of the effective local tier (recalled or pinned) and the global tier.
    pub stance: Stance,
    pub differ_kind: Option<DifferKind>,
    /// The effective local tier: the verdict's, the recalled one, or the pin's.
    pub local_tier: Tier,
    /// `None` exactly when `local_tier` is unrated (0 means neutral, as the desktop `tierView`).
    pub local_grade: Option<i32>,
    pub local_freshness: MemoryFreshness,
    /// Always `pd-verdict/2`.
    pub local_recipe: String,
    /// My wins / losses (shown on this device only; never sent).
    pub local_wl: Option<(u32, u32)>,
    /// Chips: the verdict's reasons ("rank_high", "duel_win", "held", …) plus "faded_recall" / "pinned_by_user".
    pub local_reasons: Vec<String>,
    pub global_tier: Tier,
    /// `None` exactly when `global_tier` is unrated.
    pub global_grade: Option<i32>,
    pub global_up_down: Option<(u32, u32)>,
    /// `pd-verdict/1` | `pd-verdict/2`, as the server labelled it (`pd-verdict/1` when it did not).
    pub global_recipe: String,
    /// The ▲-pinned record was superseded (patched) after the pin: ask whether to keep it.
    pub pin_superseded_warning: bool,
}

fn word_en(t: Tier) -> &'static str {
    match t {
        Tier::Premium => "recommended",
        Tier::Discount => "not recommended",
        Tier::Neutral => "so-so",
        Tier::Unrated => "",
    }
}

fn word_ja(t: Tier) -> &'static str {
    match t {
        Tier::Premium => "おすすめ",
        Tier::Discount => "非推奨",
        Tier::Neutral => "普通",
        Tier::Unrated => "",
    }
}

impl VerdictPairView {
    /// Puts the local verdict (judged with `Params::v2`), its recall input (`(past tier, past grade, no opposite vote
    /// since)`, [`LocalVerdictEntry::recall_input`]), the user's pin (`(pin, superseded after the pin)`), my wins /
    /// losses and the server's verdict (`(tier, grade, up / down votes, recipe)`) side by side. A pin beats a recall,
    /// which beats the verdict; the global side is taken as given.
    pub fn resolve(
        local: Option<&Verdict>,
        last_confirmed: Option<(Tier, i32, bool)>,
        user_pin: Option<(UserPin, bool)>,
        local_wl: Option<(u32, u32)>,
        global: Option<GlobalVerdict<'_>>,
    ) -> Self {
        let params = Params::v2();
        let mut local_reasons = local.map(|v| v.reasons.clone()).unwrap_or_default();
        let recalled = local
            .zip(last_confirmed)
            .and_then(|(v, (t, g, clean))| recall(v, t, g, clean, &params));
        let (local_tier, local_grade, local_freshness) = if let Some((pin, _)) = user_pin {
            local_reasons.push("pinned_by_user".into());
            (pin.tier(), Some(pin.grade()), MemoryFreshness::PinnedByUser(pin))
        } else if let Some((past_tier, past_grade)) = recalled {
            local_reasons.push("faded_recall".into());
            (past_tier, Some(past_grade), MemoryFreshness::FadedRecall { past_tier, past_grade })
        } else {
            let tier = local.map_or(Tier::Unrated, |v| v.tier);
            let grade = local.and_then(|v| v.grade).filter(|_| tier != Tier::Unrated);
            (tier, grade, MemoryFreshness::Active)
        };
        let (global_tier, global_grade, global_up_down, global_recipe) = match global {
            Some((t, g, ud, recipe)) => {
                let recipe = if recipe.trim().is_empty() { RECIPE_V1 } else { recipe.trim() };
                (t, g.filter(|_| t != Tier::Unrated), ud, recipe.to_string())
            }
            None => (Tier::Unrated, None, None, RECIPE_V1.to_string()),
        };
        let stance = stance(Some(local_tier), Some(global_tier));
        VerdictPairView {
            stance,
            differ_kind: stance.differ_kind(),
            local_tier,
            local_grade,
            local_freshness,
            local_recipe: RECIPE_V2.to_string(),
            local_wl,
            local_reasons,
            global_tier,
            global_grade,
            global_up_down,
            global_recipe,
            pin_superseded_warning: matches!(user_pin, Some((UserPin::Promoted, true))),
        }
    }

    /// One context line in `locale`. Without a verdict on either side it never names a recommendation either way.
    pub fn prompt_line(&self, label: &str, locale: VerdictLocale) -> String {
        match locale {
            VerdictLocale::Ko => self.line_ko(label),
            VerdictLocale::En => self.line_en(label),
            VerdictLocale::Ja => self.line_ja(label),
        }
    }

    fn global_judged(&self) -> Option<Tier> {
        judged(Some(self.global_tier))
    }

    fn line_ko(&self, label: &str) -> String {
        let global = self
            .global_judged()
            .map(|g| format!("; 서버 평가는 {}", word(g)))
            .unwrap_or_default();
        let warn = if self.pin_superseded_warning {
            " — 고정한 뒤 패치로 바뀜: 그대로 둘지 물어볼 것"
        } else {
            ""
        };
        match self.local_freshness {
            MemoryFreshness::PinnedByUser(UserPin::Promoted) => {
                format!("{label}: 사용자가 직접 ▲ 고정한 추천 (통계보다 우선){global}{warn}")
            }
            MemoryFreshness::PinnedByUser(UserPin::Demoted) => format!("{label}: 사용자가 직접 ▼ 고정한 비추천 (권하지 말 것){global}"),
            MemoryFreshness::FadedRecall { past_tier, .. } => {
                format!(
                    "{label}: 예전 내 기록으로는 {} (오래돼 흐릿함: 예전에 잘 쓰던 것처럼 회상하듯 말할 것){global}",
                    word(past_tier)
                )
            }
            MemoryFreshness::Active => match (self.stance, self.differ_kind) {
                (_, Some(DifferKind::WorksForMeOnly)) => {
                    format!("{label}: 내 기록에선 추천인데 서버에선 비추천 (나만의 비기: 둘 다 말하되 내 성공 루트를 앞세울 것)")
                }
                (_, Some(DifferKind::FailsForMe)) => {
                    format!("{label}: 내 기록에선 비추천인데 서버에선 추천 (정석이지만 내겐 안 맞았음: 강요하지 말고 우회로부터)")
                }
                (_, Some(DifferKind::LocalClearGlobalSplit)) => {
                    format!(
                        "{label}: 내 기록에선 {}인데 서버 평가는 보통 (호불호가 갈림: 내 전적 중심으로)",
                        word(self.local_tier)
                    )
                }
                (_, Some(DifferKind::LocalUndecided)) => {
                    format!(
                        "{label}: 내 기록은 보통(반반)인데 서버에선 {} (내 반반 전적을 먼저, 서버 여론은 보조로)",
                        word(self.global_tier)
                    )
                }
                (Stance::FactsOnly, None) => format!("{label}: 판정 전 (평가를 단정하지 말고 있었던 일만 말할 것)"),
                (s, None) => s.line(label),
            },
        }
    }

    fn line_en(&self, label: &str) -> String {
        let global = self
            .global_judged()
            .map(|g| format!("; the server says {}", word_en(g)))
            .unwrap_or_default();
        let warn = if self.pin_superseded_warning {
            " — patched since the pin: ask whether to keep it"
        } else {
            ""
        };
        match self.local_freshness {
            MemoryFreshness::PinnedByUser(UserPin::Promoted) => {
                format!("{label}: pinned ▲ by the user, recommended (respect it over the statistics){global}{warn}")
            }
            MemoryFreshness::PinnedByUser(UserPin::Demoted) => {
                format!("{label}: pinned ▼ by the user, not recommended (do not suggest it){global}")
            }
            MemoryFreshness::FadedRecall { past_tier, .. } => {
                format!(
                    "{label}: used to be {} by my record (a while ago: recall it like an old favourite){global}",
                    word_en(past_tier)
                )
            }
            MemoryFreshness::Active => match (self.stance, self.differ_kind) {
                (_, Some(DifferKind::WorksForMeOnly)) => {
                    format!("{label}: worked for me, but the server says not recommended (my own trick: mention both, lead with my route)")
                }
                (_, Some(DifferKind::FailsForMe)) => {
                    format!("{label}: failed for me, though the server recommends it (the standard way that did not suit me: offer a detour first)")
                }
                (_, Some(DifferKind::LocalClearGlobalSplit)) => {
                    format!(
                        "{label}: {} by my record, the server is split (lead with my record)",
                        word_en(self.local_tier)
                    )
                }
                (_, Some(DifferKind::LocalUndecided)) => {
                    format!(
                        "{label}: my record is mixed, the server says {} (mention my mixed record first)",
                        word_en(self.global_tier)
                    )
                }
                (Stance::Agree(t), None) => format!("{label}: my record and the server both say {}", word_en(t)),
                (Stance::LocalOnly(t), None) => format!("{label}: {} by my record (no server verdict yet)", word_en(t)),
                (Stance::GlobalOnly(t), None) => format!("{label}: other players say {} (no verdict from my record yet)", word_en(t)),
                (Stance::Differ { local, global }, None) => {
                    format!(
                        "{label}: {} by my record, {} by the server (say both)",
                        word_en(local),
                        word_en(global)
                    )
                }
                (Stance::FactsOnly, None) => format!("{label}: no verdict yet (state only what happened, make no judgement)"),
            },
        }
    }

    fn line_ja(&self, label: &str) -> String {
        let global = self
            .global_judged()
            .map(|g| format!("；サーバーでは{}", word_ja(g)))
            .unwrap_or_default();
        let warn = if self.pin_superseded_warning {
            " — 固定後にパッチで変更: このままにするか確認する"
        } else {
            ""
        };
        match self.local_freshness {
            MemoryFreshness::PinnedByUser(UserPin::Promoted) => {
                format!("{label}: ユーザーが▲で固定したおすすめ（統計より優先）{global}{warn}")
            }
            MemoryFreshness::PinnedByUser(UserPin::Demoted) => format!("{label}: ユーザーが▼で固定した非推奨（勧めない）{global}"),
            MemoryFreshness::FadedRecall { past_tier, .. } => {
                format!(
                    "{label}: 以前の自分の記録では{}（久しぶり: 思い出すように話す）{global}",
                    word_ja(past_tier)
                )
            }
            MemoryFreshness::Active => match (self.stance, self.differ_kind) {
                (_, Some(DifferKind::WorksForMeOnly)) => {
                    format!("{label}: 自分には合ったがサーバーでは非推奨（自分だけの攻略: 両方伝えつつ自分のルートを優先）")
                }
                (_, Some(DifferKind::FailsForMe)) => {
                    format!("{label}: サーバーではおすすめだが自分には合わなかった（無理に勧めず回り道から提案）")
                }
                (_, Some(DifferKind::LocalClearGlobalSplit)) => {
                    format!(
                        "{label}: 自分の記録では{}、サーバーは賛否が分かれる（自分の戦績を中心に）",
                        word_ja(self.local_tier)
                    )
                }
                (_, Some(DifferKind::LocalUndecided)) => {
                    format!(
                        "{label}: 自分の記録は半々、サーバーでは{}（自分の半々の戦績を先に）",
                        word_ja(self.global_tier)
                    )
                }
                (Stance::Agree(t), None) => format!("{label}: 自分の記録でもサーバーでも{}", word_ja(t)),
                (Stance::LocalOnly(t), None) => format!("{label}: 自分の記録では{}（サーバー判定前）", word_ja(t)),
                (Stance::GlobalOnly(t), None) => format!("{label}: 他のプレイヤーは{}（自分の記録は判定前）", word_ja(t)),
                (Stance::Differ { local, global }, None) => format!(
                    "{label}: 自分の記録では{}、サーバーでは{}（両方伝える）",
                    word_ja(local),
                    word_ja(global)
                ),
                (Stance::FactsOnly, None) => format!("{label}: 判定前（評価は断定せず、起きたことだけを伝える）"),
            },
        }
    }

    /// The emote to open with: a patched pin warns; a pin or a recall follows its tier (a recalled premium is an
    /// idea to revisit); each [`DifferKind`] has its own; otherwise [`Stance::emote`].
    pub fn recommended_emote(&self) -> Option<&'static str> {
        if self.pin_superseded_warning {
            return Some("EXCLAIM_3D");
        }
        match self.local_freshness {
            MemoryFreshness::PinnedByUser(UserPin::Promoted) => return Some("STAR_EYES"),
            MemoryFreshness::PinnedByUser(UserPin::Demoted) => return Some("SWEAT_PANIC"),
            MemoryFreshness::FadedRecall {
                past_tier: Tier::Premium, ..
            } => return Some("SPARKLE_IDEA"),
            MemoryFreshness::FadedRecall {
                past_tier: Tier::Discount, ..
            } => return Some("SWEAT_PANIC"),
            _ => {}
        }
        match self.differ_kind {
            Some(DifferKind::WorksForMeOnly) => Some("SPARKLE_IDEA"),
            Some(DifferKind::FailsForMe) | Some(DifferKind::LocalUndecided) => Some("QUESTION"),
            Some(DifferKind::LocalClearGlobalSplit) => Some(if self.local_tier == Tier::Premium {
                "CHIBI_CHEER"
            } else {
                "SWEAT_PANIC"
            }),
            None => self.stance.emote(),
        }
    }
}

/// Phase 5 Part 1: 신규 지름길이 패치 등으로 망가져 `Discount (grade <= -1)` 에 도달하고
/// 충분한 증거(`REINSTATE_MIN_EVIDENCE = 3.0`)가 쌓이면 과거 `superseded` 표준 코스를 자가 복권한다.
pub const REINSTATE_MIN_EVIDENCE: f64 = 3.0;

pub fn should_reinstate_standard_course(current_verdict: &Verdict) -> bool {
    current_verdict.tier == Tier::Discount
        && current_verdict.grade.is_some_and(|g| g <= -1)
        && current_verdict.evidence >= REINSTATE_MIN_EVIDENCE
}

/// Phase 4 Part 2 & Phase 5 Part 1: 코스·지름길 전용 로컬/글로벌 병렬 스탠스 브리핑 라인.
/// 원본 해시·레시피 혼합 없이 `LocalOnly` · `WorksForMeOnly` · `FadedRecall` · `PinnedByUser` 를 발화 지침으로 변환한다.
pub fn course_stance_brief_line(
    view: &VerdictPairView,
    label: &str,
    saved_ms: Option<u32>,
    locale: VerdictLocale,
) -> String {
    let base = view.prompt_line(label, locale);
    match saved_ms.filter(|&ms| ms >= 1_000) {
        Some(ms) => {
            let secs = f64::from(ms) / 1000.0;
            match locale {
                VerdictLocale::Ko => format!("{base} (평균 {secs:.1}초 단축)"),
                VerdictLocale::En => format!("{base} (saves {secs:.1}s on average)"),
                VerdictLocale::Ja => format!("{base}（平均{secs:.1}秒短縮）"),
            }
        }
        None => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evidence, Baseline, Baselines};

    #[test]
    fn the_five_rows_of_the_table() {
        use Tier::*;
        assert_eq!(stance(Some(Premium), Some(Premium)), Stance::Agree(Premium));
        assert_eq!(
            stance(Some(Premium), Some(Discount)),
            Stance::Differ {
                local: Premium,
                global: Discount
            }
        );
        assert_eq!(stance(Some(Neutral), Some(Unrated)), Stance::LocalOnly(Neutral));
        assert_eq!(stance(None, Some(Discount)), Stance::GlobalOnly(Discount));
        assert_eq!(stance(Some(Unrated), None), Stance::FactsOnly);

        assert_eq!(
            stance(Some(Premium), Some(Premium)).line("빙결 공략"),
            "빙결 공략: 내 기록과 서버 평가 모두 추천"
        );
        assert!(stance(Some(Premium), Some(Discount)).line("x").contains("둘 다"));
        let facts = Stance::FactsOnly.line("구르기 공략");
        assert!(!facts.contains("추천이야") && facts.contains("판정 전"), "{facts}");

        assert_eq!(Stance::Agree(Premium).emote(), Some("STAR_EYES"));
        assert_eq!(Stance::GlobalOnly(Discount).emote(), Some("SWEAT_PANIC"));
        assert_eq!(
            Stance::Differ {
                local: Premium,
                global: Neutral
            }
            .emote(),
            Some("QUESTION")
        );
        assert_eq!(Stance::LocalOnly(Neutral).emote(), None);
    }

    const T0: i64 = 1_700_000_000_000;

    fn outcome(sign: i32, credit: f64, at_ms: i64) -> Obs {
        Obs {
            kind: "outcome".into(),
            sign,
            weight: 1.0,
            credit,
            at_ms,
        }
    }

    /// A coach-track case with a secondary baseline that ranks a clearly positive value high.
    fn case(obs: &[Obs], now_ms: i64) -> Case {
        Case {
            now_ms,
            track: "coach".into(),
            prev: Prev {
                tier: Tier::Unrated,
                since_ms: 0,
            },
            obs: obs.to_vec(),
            baseline: Baselines {
                secondary: Some(Baseline { n: 6, mean: 0.0, sd: 0.2 }),
                ..Default::default()
            },
            duel: None,
            superseded: false,
        }
    }

    fn global(t: Tier, g: Option<i32>) -> Option<GlobalVerdict<'static>> {
        Some((t, g, Some((12, 3)), RECIPE_V1))
    }

    fn view(local: &Verdict, g: Tier) -> VerdictPairView {
        let grade = match g {
            Tier::Premium => Some(1),
            Tier::Discount => Some(-1),
            Tier::Neutral => Some(0),
            Tier::Unrated => None,
        };
        VerdictPairView::resolve(Some(local), None, None, None, global(g, grade))
    }

    fn judged_as(tier: Tier, grade: Option<i32>) -> Verdict {
        Verdict {
            evidence: 4.0,
            value: 0.5,
            level: None,
            rank: Some(0.7),
            tier,
            grade,
            since_ms: 0,
            reasons: vec!["rank_high".into(), "level:secondary".into()],
        }
    }

    #[test]
    fn fc08_differ_kinds_split_works_for_me_only_and_fails_for_me() {
        use Tier::*;
        let mine = view(&judged_as(Premium, Some(2)), Discount);
        assert_eq!(mine.differ_kind, Some(DifferKind::WorksForMeOnly));
        assert_eq!(mine.recommended_emote(), Some("SPARKLE_IDEA"));
        let theirs = view(&judged_as(Discount, Some(-1)), Premium);
        assert_eq!(theirs.differ_kind, Some(DifferKind::FailsForMe));
        assert_eq!(theirs.recommended_emote(), Some("QUESTION"));
        // All six ordered pairs of different judged tiers have a kind; equal or unrated pairs have none.
        let tiers = [Premium, Neutral, Discount];
        for l in tiers {
            for g in tiers {
                assert_eq!(DifferKind::of(l, g).is_some(), l != g, "{l:?} {g:?}");
            }
        }
        assert_eq!(DifferKind::of(Premium, Neutral), Some(DifferKind::LocalClearGlobalSplit));
        assert_eq!(DifferKind::of(Discount, Neutral), Some(DifferKind::LocalClearGlobalSplit));
        assert_eq!(DifferKind::of(Neutral, Premium), Some(DifferKind::LocalUndecided));
        assert_eq!(DifferKind::of(Unrated, Premium), None);
        let split = view(&judged_as(Premium, Some(1)), Neutral);
        assert_eq!(
            (split.recommended_emote(), split.stance.emote()),
            (Some("CHIBI_CHEER"), Some("QUESTION"))
        );
        assert!(mine.prompt_line("우측 성문", VerdictLocale::Ko).contains("나만의 비기"));
        assert!(theirs.prompt_line("정문", VerdictLocale::En).contains("detour"));
        assert!(view(&judged_as(Neutral, Some(0)), Discount)
            .prompt_line("x", VerdictLocale::Ja)
            .contains("半々"));
    }

    #[test]
    fn fc08_faded_recall_survives_repeated_queries_and_clears_on_an_opposite_vote() {
        let p = Params::v2();
        let h = p.track("coach").unwrap().half_life_ms;
        let mut obs = vec![outcome(1, 1.0, T0), outcome(1, 1.0, T0), outcome(1, 1.0, T0)];
        let mut entry = LocalVerdictEntry::default();
        let (v, f) = entry.judge(&case(&obs, T0), &p).unwrap();
        assert_eq!((v.tier, v.grade, f), (Tier::Premium, Some(2), MemoryFreshness::Active));

        // Only time passes: E decays to 1.90 and PD-Verdict says unrated (low evidence).
        let t1 = T0 + (h * (3.0f64 / 1.9).log2()) as i64;
        let (v1, f1) = entry.judge(&case(&obs, t1), &p).unwrap();
        assert!((v1.evidence - 1.9).abs() < 1e-6, "{}", v1.evidence);
        assert_eq!(v1.tier, Tier::Unrated);
        let recalled = MemoryFreshness::FadedRecall {
            past_tier: Tier::Premium,
            past_grade: 2,
        };
        assert_eq!(f1, recalled);
        assert_eq!(entry.prev.tier, Tier::Unrated, "PD-Verdict's prev is overwritten …");
        // … but the second query still recalls it.
        let t2 = t1 + 3_600_000;
        let (v2, f2) = entry.judge(&case(&obs, t2), &p).unwrap();
        assert_eq!((v2.tier, f2), (Tier::Unrated, recalled));
        let pair = VerdictPairView::resolve(Some(&v2), entry.recall_input(&obs), None, None, None);
        assert_eq!(pair.local_freshness, recalled);
        assert_eq!(
            (pair.local_tier, pair.local_grade),
            (Tier::Premium, Some(2)),
            "not demoted to GlobalOnly / FactsOnly"
        );
        assert_eq!(pair.stance, Stance::LocalOnly(Tier::Premium));
        assert_eq!(pair.recommended_emote(), Some("SPARKLE_IDEA"));
        assert!(pair.local_reasons.contains(&"faded_recall".to_string()));
        assert!(pair.prompt_line("우측 성문 루트", VerdictLocale::Ko).contains("예전"));

        // An opposite vote (too small to re-judge) ends the recall at once.
        obs.push(outcome(-1, 0.05, t2 + 1_000));
        let (v3, f3) = entry.judge(&case(&obs, t2 + 2_000), &p).unwrap();
        assert!(v3.evidence < 2.0 && v3.value >= p.floor, "{v3:?}");
        assert_eq!((v3.tier, f3), (Tier::Unrated, MemoryFreshness::Active));
        let pair = VerdictPairView::resolve(Some(&v3), entry.recall_input(&obs), None, None, None);
        assert_eq!(
            (pair.local_tier, pair.local_grade, pair.stance),
            (Tier::Unrated, None, Stance::FactsOnly)
        );

        // Withdrawn (sign 0), the opposite vote no longer counts; below E = 1.0 the memory is gone.
        obs.last_mut().unwrap().sign = 0;
        assert_eq!(entry.judge(&case(&obs, t2 + 3_000), &p).unwrap().1, recalled);
        let t4 = T0 + (h * (3.0f64 / 0.9).log2()) as i64;
        let (v4, f4) = entry.judge(&case(&obs, t4), &p).unwrap();
        assert!(v4.evidence < FADED_RECALL_MIN_EVIDENCE);
        assert_eq!(f4, MemoryFreshness::Active);
        // A neutral verdict forgets the confirmation.
        let mut neutral = LocalVerdictEntry {
            last_confirmed: entry.last_confirmed,
            ..Default::default()
        };
        let mut n = judged_as(Tier::Neutral, Some(0));
        n.reasons = vec!["level:secondary".into()];
        assert_eq!(neutral.observe(&n, &[], t4, &p), MemoryFreshness::Active);
        assert_eq!(neutral.last_confirmed, None);
    }

    #[test]
    fn fc08_recipes_are_never_mixed() {
        let p = Params::v2();
        let obs = vec![outcome(1, 1.0, T0); 3];
        let local = verdict(&case(&obs, T0), &p).unwrap();
        let pair = VerdictPairView::resolve(Some(&local), None, None, Some((3, 0)), global(Tier::Discount, Some(-1)));
        assert_eq!((pair.local_recipe.as_str(), pair.global_recipe.as_str()), (RECIPE_V2, RECIPE_V1));
        let json = serde_json::to_value(&pair).unwrap();
        for raw in ["value", "rank", "evidence", "level"] {
            assert!(json.get(raw).is_none(), "the view carries no raw {raw}");
        }
        let value = format!("{:.2}", local.value);
        for locale in [VerdictLocale::Ko, VerdictLocale::En, VerdictLocale::Ja] {
            assert!(!pair.prompt_line("x", locale).contains(&value), "no raw value in the prompt");
        }
        // A search hit (MatchVerdict) carries no recipe: the server default.
        let hit = VerdictPairView::resolve(None, None, None, None, Some((Tier::Premium, Some(1), None, "")));
        assert_eq!(hit.global_recipe, RECIPE_V1);
        let v2_server = VerdictPairView::resolve(None, None, None, None, Some((Tier::Premium, Some(1), None, RECIPE_V2)));
        assert_eq!(v2_server.global_recipe, RECIPE_V2);
        // Unrated sides never carry a grade (0 is neutral).
        let unrated = VerdictPairView::resolve(None, None, None, None, Some((Tier::Unrated, Some(0), None, RECIPE_V1)));
        assert_eq!((unrated.local_grade, unrated.global_grade), (None, None));
    }

    #[test]
    fn fc09_manual_pin_overrides_at_once_and_reverses_on_a_second_click() {
        let p = Params::v2();
        let now = T0;
        let mut history = vec![outcome(1, 1.0, now); 5];
        let mut entry = LocalVerdictEntry::default();
        let (before, _) = entry.judge(&case(&history, now), &p).unwrap();
        assert_eq!((before.tier, before.evidence), (Tier::Premium, 5.0));
        let server = global(Tier::Premium, Some(1));

        // ▼: the manual −1 alone would not move a 5-win record, the pin overrides it at once.
        let click = toggle_pin(&mut history, None, UserPin::Demoted, now + 1, &p);
        assert_eq!(click.pin, Some(UserPin::Demoted));
        let m = click.recorded.as_ref().unwrap();
        assert_eq!((m.kind.as_str(), m.sign, m.weight, m.credit), ("manual", -1, 2.0, 1.0));
        // Judged at the same instant throughout, so E compares exactly (the clicks come 1 ms and 2 ms later and count
        // as now).
        let (stats, _) = entry.judge(&case(&history, now), &p).unwrap();
        assert_eq!((stats.tier, stats.evidence), (Tier::Premium, 7.0), "statistical inertia");
        let pinned = VerdictPairView::resolve(
            Some(&stats),
            entry.recall_input(&history),
            click.pin.map(|pin| (pin, false)),
            Some((5, 0)),
            server,
        );
        assert_eq!(pinned.local_tier, Tier::Discount);
        assert_eq!(pinned.local_grade, Some(-2));
        assert_eq!(pinned.local_freshness, MemoryFreshness::PinnedByUser(UserPin::Demoted));
        assert_eq!(pinned.differ_kind, Some(DifferKind::FailsForMe));
        assert_eq!(pinned.recommended_emote(), Some("SWEAT_PANIC"));
        assert!(pinned.prompt_line("정문", VerdictLocale::Ko).contains("▼"));
        assert!(outbound_votes(&history).is_empty(), "nothing to send to the server");
        assert!(!is_server_vote_kind("manual") && is_server_vote_kind("vote"));

        // ▼ again: released, the statistics are exactly as before.
        let again = toggle_pin(&mut history, click.pin, UserPin::Demoted, now + 2, &p);
        assert_eq!((again.pin, again.recorded.is_none(), again.withdrawn), (None, true, 1));
        assert_eq!(history.len(), 6, "withdrawn in place");
        let (after, _) = entry.judge(&case(&history, now), &p).unwrap();
        assert_eq!((after.tier, after.grade, after.evidence), (before.tier, before.grade, 5.0));
        let restored = VerdictPairView::resolve(
            Some(&after),
            entry.recall_input(&history),
            again.pin.map(|pin| (pin, false)),
            Some((5, 0)),
            server,
        );
        assert_eq!(
            (restored.local_tier, restored.local_freshness),
            (Tier::Premium, MemoryFreshness::Active)
        );
        assert!(outbound_votes(&history).is_empty());
    }

    #[test]
    fn pins_switch_and_warn_after_a_patch() {
        let p = Params::v2();
        let mut history = Vec::new();
        let up = toggle_pin(&mut history, None, UserPin::Promoted, 1, &p);
        let down = toggle_pin(&mut history, up.pin, UserPin::Demoted, 2, &p);
        assert_eq!((down.pin, down.withdrawn), (Some(UserPin::Demoted), 1));
        assert_eq!(
            history.iter().map(|o| o.sign).collect::<Vec<_>>(),
            [0, -1],
            "switching withdraws the ▲"
        );
        let (e, _) = evidence(&history, 2, p.track("coach").unwrap(), &p);
        assert_eq!(e, 2.0);
        let warned = VerdictPairView::resolve(None, None, Some((UserPin::Promoted, true)), None, None);
        assert!(warned.pin_superseded_warning);
        assert_eq!(warned.recommended_emote(), Some("EXCLAIM_3D"));
        assert!(warned.prompt_line("세팅", VerdictLocale::Ko).contains("패치"));
        assert!(warned.prompt_line("setup", VerdictLocale::En).contains("patched"));
        assert!(!VerdictPairView::resolve(None, None, Some((UserPin::Demoted, true)), None, None).pin_superseded_warning);
    }

    #[test]
    fn facts_only_never_names_a_recommendation_in_any_locale() {
        let none = VerdictPairView::resolve(None, None, None, None, None);
        assert_eq!(none.stance, Stance::FactsOnly);
        assert_eq!(none.recommended_emote(), None);
        for locale in [VerdictLocale::Ko, VerdictLocale::En, VerdictLocale::Ja] {
            let line = none.prompt_line("구르기 공략", locale);
            for banned in ["추천", "비추", "recommend", "おすすめ", "非推奨"] {
                assert!(!line.contains(banned), "{locale:?}: {line}");
            }
        }
        assert_eq!(VerdictLocale::from_code("ko-KR"), VerdictLocale::Ko);
        assert_eq!(VerdictLocale::from_code("English"), VerdictLocale::En);
        assert_eq!(VerdictLocale::from_code("ja"), VerdictLocale::Ja);
        assert_eq!(VerdictLocale::from_code(""), VerdictLocale::Ko);
        // Plain stances keep the existing Korean line.
        let agree = view(&judged_as(Tier::Premium, Some(2)), Tier::Premium);
        assert_eq!(
            agree.prompt_line("빙결 공략", VerdictLocale::Ko),
            Stance::Agree(Tier::Premium).line("빙결 공략")
        );
        assert_eq!(
            agree.prompt_line("ice", VerdictLocale::En),
            "ice: my record and the server both say recommended"
        );
    }

    #[test]
    fn cn252_local_shortcut_stance_separates_works_for_me_only_and_reversible_pin() {
        let p = Params::v2();
        let local_prem = judged_as(Tier::Premium, Some(2));
        // Local Premium vs Global Discount => WorksForMeOnly (no blend, leads with personal shortcut)
        let wfm = VerdictPairView::resolve(
            Some(&local_prem),
            None,
            None,
            Some((3, 0)),
            Some((Tier::Discount, Some(-1), Some((1, 4)), RECIPE_V2)),
        );
        assert_eq!(wfm.differ_kind, Some(DifferKind::WorksForMeOnly));
        let line = course_stance_brief_line(&wfm, "우측 사다리 지름길", Some(4_200), VerdictLocale::Ko);
        assert!(line.contains("나만의 비기") && line.contains("4.2초 단축"), "{line}");

        // Reversible ▼ demotion pin withdraws cleanly on second click and stays local-only
        let mut hist = Vec::new();
        let pin1 = toggle_pin(&mut hist, None, UserPin::Demoted, 100, &p);
        assert_eq!(pin1.pin, Some(UserPin::Demoted));
        assert!(outbound_votes(&hist).is_empty(), "manual pin must never go to server");
        let pin2 = toggle_pin(&mut hist, pin1.pin, UserPin::Demoted, 200, &p);
        assert_eq!(pin2.pin, None);
        assert_eq!(pin2.withdrawn, 1);
    }

    #[test]
    fn cn260_faded_recall_recalls_past_route_and_superseded_reinstates_when_new_route_fails() {
        let p = Params::v2();
        let day_ms = 86_400_000i64;
        let obs = vec![
            Obs { kind: "outcome".into(), sign: 1, weight: 1.0, credit: 1.0, at_ms: 0 },
            Obs { kind: "outcome".into(), sign: 1, weight: 1.0, credit: 1.0, at_ms: 1_000 },
            Obs { kind: "outcome".into(), sign: 1, weight: 1.0, credit: 1.0, at_ms: 2_000 },
        ];
        let mut entry = LocalVerdictEntry::default();
        // Day 0: nav track confirms Premium (E=3.0 >= 2.0)
        let c0 = Case {
            now_ms: 3_000,
            track: "nav".into(),
            prev: Prev { tier: Tier::Unrated, since_ms: 0 },
            obs: obs.clone(),
            baseline: Baselines {
                primary: Some(crate::baseline_of(&[-0.4, -0.1, 0.0, 0.2, 0.3])),
                ..Default::default()
            },
            duel: None,
            superseded: false,
        };
        let (v0, f0) = entry.judge(&c0, &p).unwrap();
        assert_eq!(v0.tier, Tier::Premium);
        assert_eq!(f0, MemoryFreshness::Active);

        // Day 18 (~1.3 half-lives on 14d nav track): E ~ 1.23 in [1.0, 2.0) -> Unrated + FadedRecall!
        let c18 = Case {
            now_ms: 18 * day_ms,
            ..c0
        };
        let (v18, f18) = entry.judge(&c18, &p).unwrap();
        assert_eq!(v18.tier, Tier::Unrated);
        assert!(matches!(f18, MemoryFreshness::FadedRecall { past_tier: Tier::Premium, .. }));
        let view = VerdictPairView::resolve(Some(&v18), entry.recall_input(&obs), None, Some((3, 0)), None);
        let line = course_stance_brief_line(&view, "지하 통로", None, VerdictLocale::Ko);
        assert!(line.contains("예전 내 기록으로는 추천"), "{line}");

        // Self-reinstatement when newer shortcut breaks to Discount (grade <= -1, evidence >= 3.0)
        let broken_new = Verdict {
            evidence: 3.5,
            value: -0.8,
            level: None,
            rank: None,
            tier: Tier::Discount,
            grade: Some(-2),
            since_ms: 0,
            reasons: vec!["value_low".into()],
        };
        assert!(should_reinstate_standard_course(&broken_new));
    }
}
