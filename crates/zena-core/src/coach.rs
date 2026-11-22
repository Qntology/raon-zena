//! Proactive friction coaching that never gets in the player's way (`contracts/proactive_friction_v1.json`,
//! scenario 2). Everything here is a pure state machine fed with plain values; the app reads the sensors.
//!
//! * [`PendingCoachHook`] — the five-gate check on [`InterruptSignals`]: privacy lock / hidden avatar
//!   (abort), the player typing or composing (hold), high-tension combat (hold), Zena's own speech channel busy
//!   (hold), the same bottleneck coached within 90 s (abort). A held tip waits in a single slot and goes out at
//!   the golden moment — 1.5 s after a defeat ([`CoachDeliveryMode::PostDefeatComfort`]), 0.8 s after the chat
//!   box was closed without sending ([`CoachDeliveryMode::AfterInputAborted`]), right after the bubble cleared —
//!   or is dropped when the player asks (ask epoch bump), solves it alone (`Submitted` / next step), leaves the
//!   window or screen, or it gets old.
//! * [`CoachingFatigueController`] — two ignored offers dampen coaching, three (or an explicit "혼자 해볼게")
//!   mute it until the player talks to Zena. The offer itself is an [`EphemeralOffer`] kept outside the 4-turn
//!   [`Dialogue`](crate::prompt::Dialogue): it becomes a [`DialogueTurn`] only when the player follows up within
//!   45 s, otherwise the dialogue is left exactly as it was. Picking the suggested number within the accept
//!   window yields an [`AcceptedOffer`] (coach-track engage 0.2 + an exposure entry).
//! * [`route_proactive`] — Lv3 with a [`FrictionBrief`] when there is evidence (a verified alternative or a
//!   confirmed trap), a cached Lv2 line with no LLM when there is none, nothing when the player's own question
//!   takes the channel.

use crate::brief::FrictionBrief;
use crate::exposure::{ExposedRecord, ExposureRole, ExposureTargetKind};
use crate::prompt::DialogueTurn;
use crate::reaction::ReactionLevel;
use crate::text::clip_chars;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Session tension from which the combat gate holds proactive coaching (`SessionTracker`, half-life 20 s).
pub const COMBAT_TENSION: f32 = 0.65;
/// The same (screen, bottleneck) is not coached again within this many ms.
pub const ANCHOR_COOLDOWN_MS: u64 = 90_000;
/// A held tip older than this is dropped (ordinary holds / holds that waited out a fight).
pub const NORMAL_TTL_MS: u64 = 20_000;
pub const COMBAT_TTL_MS: u64 = 35_000;
/// Breathing room after a defeat before the comforting tip.
pub const POST_DEFEAT_DELAY_MS: u64 = 1_500;
/// Delay after the chat box was closed without sending.
pub const INPUT_CANCEL_DELAY_MS: u64 = 800;
/// An unanswered proactive offer is forgotten after this many ms.
pub const OFFER_TTL_MS: u64 = 45_000;
/// Picking the offered number within this window counts as accepting the offer.
pub const ACCEPT_WINDOW_MS: u64 = 15_000;
/// Phase 7 Part 2 (`κ_sr = 2.5`): 스크린리더 활성 시 코칭 수락 창을 35초로 확장한다.
pub const SCREEN_READER_ACCEPT_WINDOW_MS: u64 = 35_000;
/// Phase 6 Part 2: Lv3 코칭을 0ms Lv2 반사 템플릿으로 증류하기 위한 연속 검증 수락 횟수.
pub const REFLEX_DISTILL_MIN_ACCEPTS: u32 = 2;
/// Coach-track observation weight of an accepted offer (`Obs { kind: "engage", sign: 1, weight: 0.2 }`).
pub const COACH_ENGAGE_WEIGHT: f32 = 0.2;

/// The live state the five gates look at, sampled by the app every tick.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InterruptSignals {
    pub now_ms: u64,
    /// G1: privacy lock (FullLockdown), avatar hidden (`!WindowState::visible`) or shutdown in progress.
    pub privacy_or_hidden: bool,
    /// G2: chat box open / IME composing (`WindowState::typing`), a drag of ≥ 5 px in progress, or voice activity.
    pub user_composing: bool,
    /// G3: an urgent / mid-range fight (see [`InterruptSignals::combat_from`]) or tension ≥ [`COMBAT_TENSION`].
    pub high_tension_combat: bool,
    /// G4: Zena is streaming a line, her bubble is still up (2.5 s), or the Lv3 spacing has not elapsed.
    pub speech_channel_busy: bool,
    /// The player was just defeated (`SessionTracker::defeat`): a defeat ends the fight for the combat gate.
    pub just_defeated: bool,
}

impl InterruptSignals {
    /// The combat gate from the reaction router's decision and the session tension: every Lv3 reason except a
    /// user request (close inward threat, sharp HP drop, HP critical), every Lv2 reason (a fight), or tension
    /// ≥ [`COMBAT_TENSION`].
    pub fn combat_from(level: ReactionLevel, reason: &str, tension: f32) -> bool {
        let fighting = match level {
            ReactionLevel::Lv3Coaching => reason != "user natural-language request",
            ReactionLevel::Lv2Cached => true,
            ReactionLevel::Lv1Silent => false,
        };
        fighting || tension >= COMBAT_TENSION
    }

    /// Whether any gate keeps Zena from starting an unprompted line now (the `quiet` of `ProactiveTimer::due`).
    pub fn blocks_proactive(&self) -> bool {
        self.privacy_or_hidden || self.user_composing || self.high_tension_combat || self.speech_channel_busy
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HoldReason {
    UserComposing,
    HighTensionCombat,
    ChannelBusy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GateVerdict {
    /// Speak now.
    Pass,
    /// Not now: keep it in the hook's slot.
    Hold(HoldReason),
    /// Privacy / hidden, or this bottleneck was coached recently: drop it, keep nothing.
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoachDeliveryMode {
    /// No wait.
    Immediate,
    /// After a defeat, once the breathing delay passed: comfort first, then the tip.
    PostDefeatComfort,
    /// The chat box was closed with Esc (nothing sent): pick up what the player was about to ask.
    AfterInputAborted,
    /// Right after Zena's previous bubble cleared.
    AfterSpeechCleared,
}

impl CoachDeliveryMode {
    /// Tone tag of the brief (`"immediate_tip"`, `"post_defeat_comfort"`, ...).
    pub fn tone_tag(self) -> &'static str {
        match self {
            CoachDeliveryMode::Immediate => "immediate_tip",
            CoachDeliveryMode::PostDefeatComfort => "post_defeat_comfort",
            CoachDeliveryMode::AfterInputAborted => "after_input_aborted",
            CoachDeliveryMode::AfterSpeechCleared => "after_speech_cleared",
        }
    }

    /// The tone instruction for the model (Korean, as the rest of the prompt).
    pub fn tone_hint(self) -> &'static str {
        match self {
            CoachDeliveryMode::Immediate => "짧게 팁 한 가지",
            CoachDeliveryMode::PostDefeatComfort => "방금 졌다 — 다그치지 말고 먼저 위로한 뒤 팁 한 가지",
            CoachDeliveryMode::AfterInputAborted => "채팅창을 열었다 그냥 닫았다 — '혹시 막혀서 부르려던 거면' 하고 가볍게",
            CoachDeliveryMode::AfterSpeechCleared => "앞 대사에 이어 짧게",
        }
    }
}

/// Who gets the speech channel when several want it (higher first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpeechPriority {
    Chatter,
    Break,
    FrictionCoach,
    SurvivalWarning,
    UserQuestion,
}

/// One proactive friction tip (a `CoachReady` signal of premium-discount-nav's live friction detector).
#[derive(Debug, Clone, PartialEq)]
pub struct PendingCoach {
    pub cc: String,
    pub screen_key: String,
    pub bottleneck_anchor: Option<String>,
    pub friction_z: Option<f64>,
    /// `"rage_click"` | `"retried"` | `"pre_action_stall"` | `"ghost_overrun"`.
    pub reason_tag: String,
    pub created_at_ms: u64,
    /// Set by the hook when it holds the tip.
    pub held_by: Option<HoldReason>,
    /// The app's ask epoch when the tip was raised; any later question supersedes it.
    pub ask_epoch_at_capture: u64,
}

impl PendingCoach {
    pub fn new(cc: &str, screen_key: &str, bottleneck_anchor: Option<&str>, reason_tag: &str, created_at_ms: u64, ask_epoch: u64) -> Self {
        Self {
            cc: cc.to_string(),
            screen_key: screen_key.to_string(),
            bottleneck_anchor: bottleneck_anchor.map(str::to_string),
            friction_z: None,
            reason_tag: reason_tag.to_string(),
            created_at_ms,
            held_by: None,
            ask_epoch_at_capture: ask_epoch,
        }
    }

    /// The player asked something since the tip was raised: their question owns the channel.
    pub fn superseded_by_question(&self, current_ask_epoch: u64) -> bool {
        current_ask_epoch != self.ask_epoch_at_capture
    }

    /// Short cause wording for the brief.
    pub fn cause_summary(&self) -> &'static str {
        match self.reason_tag.as_str() {
            "rage_click" => "반복 클릭 무응답",
            "retried" => "입력 재시도 반복",
            "pre_action_stall" => "진행 경로 탐색 지연",
            "ghost_overrun" => "고스트보다 오래 정체",
            _ => "진행이 막힘",
        }
    }
}

/// The five gates plus a single-slot hold with golden-moment release.
#[derive(Debug, Clone)]
pub struct PendingCoachHook {
    pending: Option<PendingCoach>,
    /// (screen, anchor) → when it was last coached.
    recent_coached: Vec<((String, Option<String>), u64)>,
    defeated_at: Option<u64>,
    composing_seen: bool,
    composing_ended_at: Option<u64>,
    pub anchor_cooldown_ms: u64,
    pub normal_ttl_ms: u64,
    pub combat_ttl_ms: u64,
    pub post_defeat_delay_ms: u64,
    pub input_cancel_delay_ms: u64,
}

impl Default for PendingCoachHook {
    fn default() -> Self {
        Self {
            pending: None,
            recent_coached: Vec::new(),
            defeated_at: None,
            composing_seen: false,
            composing_ended_at: None,
            anchor_cooldown_ms: ANCHOR_COOLDOWN_MS,
            normal_ttl_ms: NORMAL_TTL_MS,
            combat_ttl_ms: COMBAT_TTL_MS,
            post_defeat_delay_ms: POST_DEFEAT_DELAY_MS,
            input_cancel_delay_ms: INPUT_CANCEL_DELAY_MS,
        }
    }
}

fn hold_reason(sig: &InterruptSignals) -> Option<HoldReason> {
    if sig.user_composing {
        Some(HoldReason::UserComposing)
    } else if sig.high_tension_combat || sig.just_defeated {
        Some(HoldReason::HighTensionCombat)
    } else if sig.speech_channel_busy {
        Some(HoldReason::ChannelBusy)
    } else {
        None
    }
}

/// The five interrupt gates as one pure function: G1 privacy / hidden / shutdown → abort, G5 the same
/// bottleneck coached recently (`same_bottleneck_recent`) → abort, G2 the player composing → hold, G3 a tense
/// fight (a defeat in progress counts) → hold, G4 Zena's speech channel busy → hold, otherwise pass.
pub fn interrupt_gate(sig: &InterruptSignals, same_bottleneck_recent: bool) -> GateVerdict {
    if sig.privacy_or_hidden || same_bottleneck_recent {
        return GateVerdict::Abort;
    }
    match hold_reason(sig) {
        Some(r) => GateVerdict::Hold(r),
        None => GateVerdict::Pass,
    }
}

impl PendingCoachHook {
    pub fn new() -> Self {
        Self::default()
    }

    /// The held tip, if any.
    pub fn pending(&self) -> Option<&PendingCoach> {
        self.pending.as_ref()
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether (screen, anchor) was coached within the cooldown.
    pub fn on_cooldown(&self, screen_key: &str, anchor: Option<&str>, now_ms: u64) -> bool {
        self.recent_coached
            .iter()
            .any(|((s, a), t)| s == screen_key && a.as_deref() == anchor && now_ms.saturating_sub(*t) < self.anchor_cooldown_ms)
    }

    /// The pure gate verdict for `candidate` (no state change): G1 privacy → abort, G5 cooldown → abort,
    /// G2 composing / G3 combat (a defeat in progress counts) / G4 channel → hold, otherwise pass.
    pub fn evaluate(&self, candidate: &PendingCoach, sig: &InterruptSignals) -> GateVerdict {
        let recent = self.on_cooldown(&candidate.screen_key, candidate.bottleneck_anchor.as_deref(), sig.now_ms);
        interrupt_gate(sig, recent)
    }

    /// Records that (screen, anchor) was coached now (also call it when coaching went out another way, e.g. as
    /// the answer to the player's question).
    pub fn note_coached(&mut self, screen_key: &str, anchor: Option<&str>, now_ms: u64) {
        self.recent_coached
            .retain(|((s, a), _)| !(s == screen_key && a.as_deref() == anchor));
        self.recent_coached
            .push(((screen_key.to_string(), anchor.map(str::to_string)), now_ms));
    }

    fn prune(&mut self, now_ms: u64) {
        let cd = self.anchor_cooldown_ms;
        self.recent_coached.retain(|(_, t)| now_ms.saturating_sub(*t) < cd);
    }

    fn clear(&mut self) {
        self.pending = None;
        self.defeated_at = None;
        self.composing_seen = false;
        self.composing_ended_at = None;
    }

    /// A new `CoachReady`: delivered now (`Some(.., Immediate)`), held in the slot or dropped (`None`).
    pub fn submit(&mut self, mut candidate: PendingCoach, sig: &InterruptSignals) -> Option<(PendingCoach, CoachDeliveryMode)> {
        let now = sig.now_ms;
        self.prune(now);
        match self.evaluate(&candidate, sig) {
            GateVerdict::Abort => None,
            GateVerdict::Pass => {
                // A fresher tip supersedes whatever was held.
                self.clear();
                self.note_coached(&candidate.screen_key, candidate.bottleneck_anchor.as_deref(), now);
                Some((candidate, CoachDeliveryMode::Immediate))
            }
            GateVerdict::Hold(reason) => {
                candidate.held_by = Some(reason);
                let held = self
                    .pending
                    .as_ref()
                    .map(|p| (p.cc == candidate.cc && p.screen_key == candidate.screen_key, p.friction_z));
                let replace = match held {
                    None => true,
                    // Another window / screen: the old tip is stale.
                    Some((false, _)) => {
                        self.clear();
                        true
                    }
                    // Same screen: keep the stronger signal (ties go to the newer one).
                    Some((true, z)) => candidate.friction_z.unwrap_or(0.0) >= z.unwrap_or(0.0),
                };
                if replace {
                    self.pending = Some(candidate);
                } else if let Some(p) = &mut self.pending {
                    p.held_by = Some(reason);
                }
                if sig.just_defeated && self.defeated_at.is_none() {
                    self.defeated_at = Some(now);
                }
                if sig.user_composing {
                    self.composing_seen = true;
                    self.composing_ended_at = None;
                }
                None
            }
        }
    }

    /// Every tick: drops a held tip that went stale (the player solved it — `progress_advanced` —, asked
    /// something, changed window or screen, privacy, TTL) and releases it at the golden moment.
    pub fn poll(
        &mut self,
        current_cc: &str,
        current_screen: &str,
        progress_advanced: bool,
        current_ask_epoch: u64,
        sig: &InterruptSignals,
    ) -> Option<(PendingCoach, CoachDeliveryMode)> {
        let now = sig.now_ms;
        let p = self.pending.as_ref()?;
        let ttl = if p.held_by == Some(HoldReason::HighTensionCombat) || self.defeated_at.is_some() {
            self.combat_ttl_ms
        } else {
            self.normal_ttl_ms
        };
        let stale = progress_advanced
            || p.superseded_by_question(current_ask_epoch)
            || p.cc != current_cc
            || p.screen_key != current_screen
            || sig.privacy_or_hidden
            || now.saturating_sub(p.created_at_ms) > ttl;
        if stale {
            self.clear();
            return None;
        }
        if sig.just_defeated && self.defeated_at.is_none() {
            self.defeated_at = Some(now);
        }
        if sig.user_composing {
            self.composing_seen = true;
            self.composing_ended_at = None;
        } else if self.composing_seen && self.composing_ended_at.is_none() {
            self.composing_ended_at = Some(now);
        }
        // A recorded defeat ends the fight for this tip even while the tension is still high.
        let combat = sig.high_tension_combat && self.defeated_at.is_none();
        let blocking = if sig.user_composing {
            Some(HoldReason::UserComposing)
        } else if combat {
            Some(HoldReason::HighTensionCombat)
        } else if sig.speech_channel_busy {
            Some(HoldReason::ChannelBusy)
        } else {
            None
        };
        if let Some(r) = blocking {
            if let Some(p) = &mut self.pending {
                p.held_by = Some(r);
            }
            return None;
        }
        let held_by = self.pending.as_ref()?.held_by;
        let (mode, at) = if let Some(t) = self.defeated_at {
            (CoachDeliveryMode::PostDefeatComfort, t + self.post_defeat_delay_ms)
        } else if let Some(t) = self.composing_ended_at {
            (CoachDeliveryMode::AfterInputAborted, t + self.input_cancel_delay_ms)
        } else if held_by == Some(HoldReason::ChannelBusy) {
            (CoachDeliveryMode::AfterSpeechCleared, now)
        } else {
            (CoachDeliveryMode::Immediate, now)
        };
        if now < at {
            return None;
        }
        let coach = self.pending.take()?;
        self.clear();
        self.note_coached(&coach.screen_key, coach.bottleneck_anchor.as_deref(), now);
        Some((coach, mode))
    }
}

/// How willing the player currently is to hear unprompted coaching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FatigueState {
    #[default]
    Receptive,
    /// Two offers in a row ignored (or a mild negative reaction): higher bar, doubled cooldown, Lv2 cues only.
    Dampened,
    /// Three ignored, an explicit "혼자 해볼게" / ▼ / closed bubble: no proactive coaching until the player talks.
    MutedUntilAsk,
}

/// A proactive offer Zena just made, kept outside the 4-turn dialogue window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EphemeralOffer {
    pub cc: String,
    /// The record / checkpoint the offer points at (for credit), if known.
    pub target_id: Option<String>,
    /// The numbered badge the offer names ("1번 간편결제 써볼래?" → 1).
    pub target_number: Option<u8>,
    /// What Zena said (clipped to 60 characters like a dialogue turn).
    pub spoken_line: String,
    pub offered_at_ms: u64,
}

impl EphemeralOffer {
    pub fn new(cc: &str, target_id: Option<&str>, target_number: Option<u8>, spoken_line: &str, offered_at_ms: u64) -> Self {
        Self {
            cc: cc.to_string(),
            target_id: target_id.map(str::to_string),
            target_number,
            spoken_line: clip_chars(spoken_line, 60),
            offered_at_ms,
        }
    }
}

/// The player picked the number Zena suggested within the accept window.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedOffer {
    pub offer: EphemeralOffer,
    pub picked_at_ms: u64,
    /// Coach-track engage observation weight ([`COACH_ENGAGE_WEIGHT`], sign +1, credit 1.0).
    pub engage_weight: f32,
}

impl AcceptedOffer {
    pub fn picked_after_ms(&self) -> u64 {
        self.picked_at_ms.saturating_sub(self.offer.offered_at_ms)
    }

    /// The exposure entry for the 5-minute ledger (`None` when the offer named no record).
    pub fn exposure(&self) -> Option<ExposedRecord> {
        let id = self.offer.target_id.clone()?;
        Some(ExposedRecord {
            record_id: id,
            cc: self.offer.cc.clone(),
            to: None,
            canonical_topic: self.offer.spoken_line.clone(),
            role: ExposureRole::RecommendedSolution,
            w_spoken: 1.0,
            exposed_at_ms: self.offer.offered_at_ms,
            target_kind: ExposureTargetKind::RagRecord,
            exclusive_slot_group: Some("tactics".to_string()),
        })
    }
}

/// Refusal-fatigue hysteresis plus the ephemeral offer slot.
#[derive(Debug, Clone)]
pub struct CoachingFatigueController {
    state: FatigueState,
    consecutive_ignores: u8,
    pub ephemeral_offer: Option<EphemeralOffer>,
    pub offer_ttl_ms: u64,
    pub accept_window_ms: u64,
}

impl Default for CoachingFatigueController {
    fn default() -> Self {
        Self {
            state: FatigueState::Receptive,
            consecutive_ignores: 0,
            ephemeral_offer: None,
            offer_ttl_ms: OFFER_TTL_MS,
            accept_window_ms: ACCEPT_WINDOW_MS,
        }
    }
}

impl CoachingFatigueController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Phase 7 Part 2 (`AC-04`): 스크린리더 활성 시 수락 창(15s → 35s)을 동적으로 조정한다.
    pub fn set_screen_reader_mode(&mut self, active: bool) {
        self.accept_window_ms = if active {
            SCREEN_READER_ACCEPT_WINDOW_MS
        } else {
            ACCEPT_WINDOW_MS
        };
    }

    pub fn state(&self) -> FatigueState {
        self.state
    }

    pub fn consecutive_ignores(&self) -> u8 {
        self.consecutive_ignores
    }

    /// (friction threshold multiplier, coaching cooldown ms) for the current state; `None` while muted.
    pub fn thresholds(&self) -> Option<(f64, u64)> {
        match self.state {
            FatigueState::Receptive => Some((1.0, ANCHOR_COOLDOWN_MS)),
            FatigueState::Dampened => Some((1.5, ANCHOR_COOLDOWN_MS * 2)),
            FatigueState::MutedUntilAsk => None,
        }
    }

    /// Whether any proactive coaching may start (survival warnings are not coaching and are not gated here).
    pub fn allows_proactive(&self) -> bool {
        self.state != FatigueState::MutedUntilAsk
    }

    /// Dampened: only short Lv2 cues, no LLM coaching.
    pub fn lv2_only(&self) -> bool {
        self.state == FatigueState::Dampened
    }

    /// Zena made a proactive offer: remember it here, NOT in the dialogue.
    pub fn offer(&mut self, offer: EphemeralOffer) {
        self.ephemeral_offer = Some(offer);
    }

    /// The player picked badge `picked_number`: if it is the offered one, in the same window and within the
    /// accept window, coaching becomes receptive again and the acceptance is returned.
    pub fn on_target_picked(&mut self, cc: &str, picked_number: u8, now_ms: u64) -> Option<AcceptedOffer> {
        let o = self.ephemeral_offer.as_ref()?;
        let accepted =
            o.cc == cc && o.target_number == Some(picked_number) && now_ms.saturating_sub(o.offered_at_ms) <= self.accept_window_ms;
        if !accepted {
            return None;
        }
        let offer = self.ephemeral_offer.take()?;
        self.state = FatigueState::Receptive;
        self.consecutive_ignores = 0;
        Some(AcceptedOffer {
            offer,
            picked_at_ms: now_ms,
            engage_weight: COACH_ENGAGE_WEIGHT,
        })
    }

    /// An offer went unanswered or was dismissed: Receptive → (2) Dampened → (3) MutedUntilAsk; an explicit mute
    /// ("혼자 해볼게", "조용히 해", ▼, closing the bubble) mutes at once.
    pub fn on_ignored_or_dismissed(&mut self, explicit_mute: bool) {
        self.ephemeral_offer = None;
        self.consecutive_ignores = self.consecutive_ignores.saturating_add(1);
        self.state = if explicit_mute || self.consecutive_ignores >= 3 {
            FatigueState::MutedUntilAsk
        } else if self.consecutive_ignores >= 2 {
            FatigueState::Dampened
        } else {
            self.state
        };
    }

    /// Every tick: an offer nobody followed up within the TTL is dropped and counted as ignored. Returns it.
    pub fn expire(&mut self, now_ms: u64) -> Option<EphemeralOffer> {
        let expired = self
            .ephemeral_offer
            .as_ref()
            .is_some_and(|o| now_ms.saturating_sub(o.offered_at_ms) > self.offer_ttl_ms);
        if !expired {
            return None;
        }
        let o = self.ephemeral_offer.take();
        self.on_ignored_or_dismissed(false);
        o
    }

    /// The player said something to Zena: coaching becomes receptive again, and an offer made in this window
    /// within the TTL is returned as the dialogue turn to push before the player's own exchange.
    pub fn on_user_utterance(&mut self, current_cc: &str, now_ms: u64) -> Option<DialogueTurn> {
        self.state = FatigueState::Receptive;
        self.consecutive_ignores = 0;
        let o = self.ephemeral_offer.take()?;
        (o.cc == current_cc && now_ms.saturating_sub(o.offered_at_ms) <= self.offer_ttl_ms).then_some(DialogueTurn {
            user: None,
            zena: o.spoken_line,
        })
    }
}

/// What the app knows, without blocking, about how to get past the bottleneck.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CoachEvidence {
    /// Scrubbed label of the stuck element ("쿠폰 적용 버튼").
    pub bottleneck_label: Option<String>,
    /// A verified alternative: a Premium / GoldenPath step ("#1 '간편결제' (유저 검증 지름길)").
    pub verified_alternative: Option<String>,
    /// A confirmed FrictionTrap to warn about.
    pub confirmed_trap: Option<String>,
    /// The badge to point at (Lv2 points without words about the record).
    pub point_at: Option<u8>,
}

impl CoachEvidence {
    pub fn has_evidence(&self) -> bool {
        self.verified_alternative.is_some() || self.confirmed_trap.is_some()
    }
}

/// A cached line for the Lv2 path (the app may swap in its lexicon line for the same situation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedCoachLine {
    pub line: String,
    pub point_at: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteCancel {
    /// The player asked something: their question takes the channel.
    UserQuestion,
    /// Coaching is muted until the player talks to Zena.
    Muted,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProactiveRoute {
    /// Raon-1-nano with this brief after `<|im_start|>user\n` (`SessionContext::friction_brief`).
    Lv3(FrictionBrief),
    /// No LLM: a cached line, pointing at the badge if one is known.
    Lv2(CachedCoachLine),
    Cancel(RouteCancel),
}

/// Lv2 / Lv3 decision for a released tip: a newer question cancels; muted coaching stays silent; evidence with
/// the model available (and not dampened) goes to Lv3; anything else is a cached Lv2 line.
pub fn route_proactive(
    coach: &PendingCoach,
    mode: CoachDeliveryMode,
    evidence: &CoachEvidence,
    lv3_available: bool,
    current_ask_epoch: u64,
    fatigue: &CoachingFatigueController,
) -> ProactiveRoute {
    if coach.superseded_by_question(current_ask_epoch) {
        return ProactiveRoute::Cancel(RouteCancel::UserQuestion);
    }
    if !fatigue.allows_proactive() {
        return ProactiveRoute::Cancel(RouteCancel::Muted);
    }
    if evidence.has_evidence() && lv3_available && !fatigue.lv2_only() {
        return ProactiveRoute::Lv3(FrictionBrief {
            bottleneck_label: evidence.bottleneck_label.clone(),
            cause_summary: coach.cause_summary().to_string(),
            verified_alternative: evidence.verified_alternative.clone(),
            trap_warning: evidence.confirmed_trap.clone(),
            delivery: mode,
        });
    }
    let line = match (mode, evidence.point_at) {
        (CoachDeliveryMode::PostDefeatComfort, _) => "아까운 타이밍이었어! 숨 한번 고르고 다시 가보자!".to_string(),
        (_, Some(n)) => format!("{n}번 쪽부터 차근차근 확인해 볼까?"),
        (CoachDeliveryMode::AfterInputAborted, None) => "혹시 막혀서 부르려던 거야? 천천히 같이 보자!".to_string(),
        _ => "막히는 데가 있으면 언제든 물어봐!".to_string(),
    };
    ProactiveRoute::Lv2(CachedCoachLine {
        line,
        point_at: evidence.point_at,
    })
}

/// Phase 6 Part 2: 반복 검증된 Lv3 코칭을 0ms LLM-Free Lv2 반사 신경으로 증류한 템플릿.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReflexTemplate {
    pub cc: String,
    pub screen_key: String,
    pub anchor: Option<String>,
    pub line: String,
    pub point_at: Option<u8>,
    pub verified_accepts: u32,
    pub distilled: bool,
}

/// Phase 6 Part 2: Lv3 → Lv2 반사 신경 증류 컴파일러.
/// 동일 `(cc, screen_key, anchor)` 에서 Lv3 코칭이 `min_accepts`(기본 2회) 연속 검증 수락되면
/// 이후 같은 병목 발생 시 LLM 을 깨우지 않고 0ms Lv2 즉시 응답으로 전환한다.
#[derive(Debug, Clone)]
pub struct ReflexCompiler {
    templates: BTreeMap<(String, String, Option<String>), ReflexTemplate>,
    pub min_accepts: u32,
}

impl Default for ReflexCompiler {
    fn default() -> Self {
        Self {
            templates: BTreeMap::new(),
            min_accepts: REFLEX_DISTILL_MIN_ACCEPTS,
        }
    }
}

impl ReflexCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_lv3_success(
        &mut self,
        cc: &str,
        screen_key: &str,
        anchor: Option<&str>,
        line: &str,
        point_at: Option<u8>,
    ) -> bool {
        let key = (cc.to_string(), screen_key.to_string(), anchor.map(str::to_string));
        let entry = self.templates.entry(key).or_insert_with(|| ReflexTemplate {
            cc: cc.to_string(),
            screen_key: screen_key.to_string(),
            anchor: anchor.map(str::to_string),
            line: clip_chars(line, 60),
            point_at,
            verified_accepts: 0,
            distilled: false,
        });
        entry.line = clip_chars(line, 60);
        entry.point_at = point_at;
        entry.verified_accepts = entry.verified_accepts.saturating_add(1);
        if entry.verified_accepts >= self.min_accepts {
            entry.distilled = true;
        }
        entry.distilled
    }

    pub fn revoke(&mut self, cc: &str, screen_key: &str, anchor: Option<&str>) {
        self.templates
            .remove(&(cc.to_string(), screen_key.to_string(), anchor.map(str::to_string)));
    }

    pub fn lookup(&self, cc: &str, screen_key: &str, anchor: Option<&str>) -> Option<&ReflexTemplate> {
        self.templates
            .get(&(cc.to_string(), screen_key.to_string(), anchor.map(str::to_string)))
            .filter(|t| t.distilled)
    }
}

/// 증류된 0ms Lv2 반사 템플릿을 우선 확인한 뒤 없을 때만 기존 `route_proactive` 로 라우팅한다.
pub fn route_proactive_with_reflex(
    coach: &PendingCoach,
    mode: CoachDeliveryMode,
    evidence: &CoachEvidence,
    lv3_available: bool,
    current_ask_epoch: u64,
    fatigue: &CoachingFatigueController,
    compiler: &ReflexCompiler,
) -> ProactiveRoute {
    if coach.superseded_by_question(current_ask_epoch) {
        return ProactiveRoute::Cancel(RouteCancel::UserQuestion);
    }
    if !fatigue.allows_proactive() {
        return ProactiveRoute::Cancel(RouteCancel::Muted);
    }
    if let Some(tpl) = compiler.lookup(&coach.cc, &coach.screen_key, coach.bottleneck_anchor.as_deref()) {
        return ProactiveRoute::Lv2(CachedCoachLine {
            line: tpl.line.clone(),
            point_at: tpl.point_at.or(evidence.point_at),
        });
    }
    route_proactive(coach, mode, evidence, lv3_available, current_ask_epoch, fatigue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{build_fast_path_prompt, Dialogue, SessionContext, Situation, USER_TURN};

    fn sig(now_ms: u64) -> InterruptSignals {
        InterruptSignals {
            now_ms,
            ..Default::default()
        }
    }

    fn coach(t: u64, epoch: u64) -> PendingCoach {
        PendingCoach::new("eldenring.exe", "screen-a", Some("#2"), "rage_click", t, epoch)
    }

    #[test]
    fn pf04_combat_holds_then_one_post_defeat_release_after_1500ms() {
        let mut hook = PendingCoachHook::new();
        let tension = 0.78;
        let combat = InterruptSignals {
            high_tension_combat: InterruptSignals::combat_from(ReactionLevel::Lv1Silent, "ambient movement", tension),
            ..sig(10_000)
        };
        assert!(combat.high_tension_combat, "tension 0.78 ≥ 0.65 holds");
        let c = coach(10_000, 1);
        assert_eq!(hook.evaluate(&c, &combat), GateVerdict::Hold(HoldReason::HighTensionCombat));
        assert_eq!(hook.submit(c, &combat), None, "nothing is said during the fight");
        let mut released = Vec::new();
        for t in (10_000..18_000).step_by(250) {
            let s = InterruptSignals { now_ms: t, ..combat };
            released.extend(hook.poll("eldenring.exe", "screen-a", false, 1, &s));
        }
        assert!(released.is_empty(), "zero lines while fighting");
        // 8 s later: defeat, the fight is over.
        let defeat_at = 18_000;
        for t in (defeat_at..defeat_at + 1_500).step_by(100) {
            let s = InterruptSignals {
                just_defeated: t == defeat_at,
                ..sig(t)
            };
            released.extend(hook.poll("eldenring.exe", "screen-a", false, 1, &s));
        }
        assert!(released.is_empty(), "the breathing delay is respected");
        for t in (defeat_at + 1_500..defeat_at + 5_000).step_by(100) {
            released.extend(hook.poll("eldenring.exe", "screen-a", false, 1, &sig(t)));
        }
        assert_eq!(released.len(), 1, "exactly one release");
        assert_eq!(released[0].1, CoachDeliveryMode::PostDefeatComfort);
        assert_eq!(released[0].0.held_by, Some(HoldReason::HighTensionCombat));
    }

    #[test]
    fn a_defeat_releases_even_while_tension_is_still_high() {
        let mut hook = PendingCoachHook::new();
        let hot = InterruptSignals {
            high_tension_combat: true,
            ..sig(0)
        };
        hook.submit(coach(0, 0), &hot);
        // The defeat bumps tension, so the app may still report combat; the recorded defeat wins.
        let d = InterruptSignals {
            just_defeated: true,
            ..hot
        };
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", false, 0, &InterruptSignals { now_ms: 3_000, ..d }),
            None
        );
        let after = hook.poll("eldenring.exe", "screen-a", false, 0, &InterruptSignals { now_ms: 4_500, ..hot });
        assert_eq!(after.map(|r| r.1), Some(CoachDeliveryMode::PostDefeatComfort));
    }

    #[test]
    fn pf05_typing_yields_then_a_question_absorbs_or_esc_releases_after_800ms() {
        let typing = InterruptSignals {
            user_composing: true,
            ..sig(1_000)
        };
        let mut base = PendingCoachHook::new();
        assert_eq!(
            base.evaluate(&coach(1_000, 1), &typing),
            GateVerdict::Hold(HoldReason::UserComposing)
        );
        assert_eq!(base.submit(coach(1_000, 1), &typing), None);
        // Branch A: the question is sent (ask epoch 1 → 2): the held tip is gone, nothing is ever said.
        let mut a = base.clone();
        let mut out = Vec::new();
        out.extend(a.poll("eldenring.exe", "screen-a", false, 2, &sig(2_000)));
        assert!(!a.is_pending(), "absorbed by the question");
        for t in (2_000..30_000).step_by(500) {
            out.extend(a.poll("eldenring.exe", "screen-a", false, 2, &sig(t)));
        }
        assert!(out.is_empty(), "no duplicate line");
        // Branch B: Esc closes the box (typing → false, no new question): released 800 ms later.
        let mut b = base.clone();
        assert_eq!(b.poll("eldenring.exe", "screen-a", false, 1, &sig(3_000)), None);
        assert_eq!(b.poll("eldenring.exe", "screen-a", false, 1, &sig(3_799)), None);
        let r = b.poll("eldenring.exe", "screen-a", false, 1, &sig(3_800)).expect("released");
        assert_eq!(r.1, CoachDeliveryMode::AfterInputAborted);
        // Reopening the box within the 800 ms holds the tip again; the clock restarts at the next close.
        let mut c = base;
        assert_eq!(c.poll("eldenring.exe", "screen-a", false, 1, &sig(3_000)), None);
        let typing_again = InterruptSignals {
            user_composing: true,
            ..sig(3_500)
        };
        assert_eq!(c.poll("eldenring.exe", "screen-a", false, 1, &typing_again), None);
        assert_eq!(c.poll("eldenring.exe", "screen-a", false, 1, &sig(4_000)), None);
        assert_eq!(c.poll("eldenring.exe", "screen-a", false, 1, &sig(4_799)), None);
        assert_eq!(
            c.poll("eldenring.exe", "screen-a", false, 1, &sig(4_800)).map(|r| r.1),
            Some(CoachDeliveryMode::AfterInputAborted)
        );
    }

    #[test]
    fn a_question_or_a_window_switch_during_a_fight_drops_the_held_tip() {
        let hot = InterruptSignals {
            high_tension_combat: true,
            ..sig(0)
        };
        let mut hook = PendingCoachHook::new();
        hook.submit(coach(0, 5), &hot);
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", false, 6, &InterruptSignals { now_ms: 1_000, ..hot }),
            None
        );
        assert!(!hook.is_pending(), "the player's question supersedes it");
        hook.submit(coach(0, 5), &hot);
        assert_eq!(
            hook.poll("steampowered.com", "screen-a", false, 5, &InterruptSignals { now_ms: 1_000, ..hot }),
            None
        );
        assert!(!hook.is_pending(), "another window");
        hook.submit(coach(0, 5), &hot);
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", false, 5, &InterruptSignals { now_ms: 35_001, ..hot }),
            None
        );
        assert!(!hook.is_pending(), "combat TTL 35 s");
    }

    #[test]
    fn pf06_self_resolved_tip_is_dropped_and_the_same_bottleneck_is_cooled_down_90s() {
        let busy = InterruptSignals {
            speech_channel_busy: true,
            ..sig(5_000)
        };
        let mut hook = PendingCoachHook::new();
        assert_eq!(hook.submit(coach(5_000, 0), &busy), None);
        assert_eq!(hook.pending().and_then(|p| p.held_by), Some(HoldReason::ChannelBusy));
        // The player clicks the right button on their own (Signal::Submitted).
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", true, 0, &InterruptSignals { now_ms: 6_000, ..busy }),
            None
        );
        let mut late = Vec::new();
        for t in (6_000..30_000).step_by(500) {
            late.extend(hook.poll("eldenring.exe", "screen-a", false, 0, &sig(t)));
        }
        assert!(late.is_empty(), "no stale tip after the bubble cleared");

        // A tip delivered for (screen-a, #2) …
        let mut hook = PendingCoachHook::new();
        let first = hook.submit(coach(40_000, 0), &sig(40_000));
        assert_eq!(first.map(|r| r.1), Some(CoachDeliveryMode::Immediate));
        // … and the same bottleneck 30 s later is aborted by G5.
        let again = coach(70_000, 0);
        assert_eq!(hook.evaluate(&again, &sig(70_000)), GateVerdict::Abort);
        assert_eq!(hook.submit(again, &sig(70_000)), None);
        assert!(!hook.is_pending(), "aborted, not held");
        // Another anchor is fine; after 90 s the same one is too.
        let other = PendingCoach::new("eldenring.exe", "screen-a", Some("#3"), "retried", 70_000, 0);
        assert_eq!(hook.evaluate(&other, &sig(70_000)), GateVerdict::Pass);
        assert_eq!(hook.evaluate(&coach(130_001, 0), &sig(130_001)), GateVerdict::Pass);
    }

    #[test]
    fn privacy_aborts_and_drops_a_held_tip() {
        let mut hook = PendingCoachHook::new();
        let locked = InterruptSignals {
            privacy_or_hidden: true,
            user_composing: true,
            ..sig(0)
        };
        assert_eq!(hook.evaluate(&coach(0, 0), &locked), GateVerdict::Abort);
        assert_eq!(hook.submit(coach(0, 0), &locked), None);
        assert!(!hook.is_pending(), "nothing kept, not even in the slot");
        let busy = InterruptSignals {
            speech_channel_busy: true,
            ..sig(0)
        };
        hook.submit(coach(0, 0), &busy);
        assert!(hook.is_pending());
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", false, 0, &InterruptSignals { now_ms: 100, ..locked }),
            None
        );
        assert!(!hook.is_pending());
    }

    #[test]
    fn held_tips_expire_change_with_the_screen_and_release_after_the_bubble() {
        let busy = InterruptSignals {
            speech_channel_busy: true,
            ..sig(0)
        };
        let mut hook = PendingCoachHook::new();
        hook.submit(coach(0, 0), &busy);
        assert_eq!(
            hook.poll("eldenring.exe", "screen-a", false, 0, &InterruptSignals { now_ms: 20_001, ..busy }),
            None
        );
        assert!(!hook.is_pending(), "TTL 20 s");

        hook.submit(coach(0, 0), &busy);
        assert_eq!(hook.poll("eldenring.exe", "screen-b", false, 0, &sig(100)), None);
        assert!(!hook.is_pending(), "another screen");

        hook.submit(coach(0, 0), &busy);
        let r = hook
            .poll("eldenring.exe", "screen-a", false, 0, &sig(2_500))
            .expect("bubble cleared");
        assert_eq!(r.1, CoachDeliveryMode::AfterSpeechCleared);
        assert!(hook.on_cooldown("screen-a", Some("#2"), 2_600), "a release starts the cooldown");
    }

    #[test]
    fn the_slot_keeps_one_tip_the_stronger_on_the_same_screen() {
        let busy = InterruptSignals {
            speech_channel_busy: true,
            ..sig(0)
        };
        let mut hook = PendingCoachHook::new();
        let mut strong = coach(0, 0);
        strong.friction_z = Some(2.4);
        hook.submit(strong, &busy);
        let mut weak = PendingCoach::new("eldenring.exe", "screen-a", Some("#5"), "retried", 100, 0);
        weak.friction_z = Some(1.6);
        hook.submit(weak, &busy);
        assert_eq!(hook.pending().unwrap().bottleneck_anchor.as_deref(), Some("#2"));
        let elsewhere = PendingCoach::new("eldenring.exe", "screen-b", None, "retried", 200, 0);
        hook.submit(elsewhere, &busy);
        assert_eq!(hook.pending().unwrap().screen_key, "screen-b", "a new screen replaces the old tip");
    }

    #[test]
    fn the_gate_table_puts_aborts_before_holds() {
        let all = InterruptSignals {
            now_ms: 0,
            privacy_or_hidden: true,
            user_composing: true,
            high_tension_combat: true,
            speech_channel_busy: true,
            just_defeated: false,
        };
        assert_eq!(interrupt_gate(&all, false), GateVerdict::Abort, "G1 first");
        let holds = InterruptSignals {
            privacy_or_hidden: false,
            ..all
        };
        assert_eq!(interrupt_gate(&holds, true), GateVerdict::Abort, "G5 before any hold");
        assert_eq!(interrupt_gate(&holds, false), GateVerdict::Hold(HoldReason::UserComposing));
        let fight = InterruptSignals {
            user_composing: false,
            ..holds
        };
        assert_eq!(interrupt_gate(&fight, false), GateVerdict::Hold(HoldReason::HighTensionCombat));
        let talking = InterruptSignals {
            high_tension_combat: false,
            ..fight
        };
        assert_eq!(interrupt_gate(&talking, false), GateVerdict::Hold(HoldReason::ChannelBusy));
        assert_eq!(interrupt_gate(&sig(0), false), GateVerdict::Pass);
        assert!(talking.blocks_proactive() && !sig(0).blocks_proactive());
    }

    #[test]
    fn combat_gate_reasons_and_priorities() {
        assert!(InterruptSignals::combat_from(
            ReactionLevel::Lv3Coaching,
            "close inward threat",
            0.0
        ));
        assert!(!InterruptSignals::combat_from(
            ReactionLevel::Lv3Coaching,
            "user natural-language request",
            0.1
        ));
        assert!(InterruptSignals::combat_from(ReactionLevel::Lv2Cached, "mid-range combat", 0.0));
        assert!(!InterruptSignals::combat_from(ReactionLevel::Lv1Silent, "ambient movement", 0.6));
        assert!(SpeechPriority::UserQuestion > SpeechPriority::SurvivalWarning);
        assert!(SpeechPriority::SurvivalWarning > SpeechPriority::FrictionCoach);
        assert!(SpeechPriority::FrictionCoach > SpeechPriority::Break && SpeechPriority::Break > SpeechPriority::Chatter);
    }

    #[test]
    fn pf10_evidence_goes_to_lv3_unrated_to_cached_lv2_and_a_question_cancels() {
        let fatigue = CoachingFatigueController::new();
        let c = coach(0, 3);
        // A: a verified Premium alternative.
        let ev = CoachEvidence {
            bottleneck_label: Some("쿠폰 적용 버튼".into()),
            verified_alternative: Some("#1 '간편결제' (유저 검증 지름길)".into()),
            point_at: Some(1),
            ..Default::default()
        };
        let brief = match route_proactive(&c, CoachDeliveryMode::Immediate, &ev, true, 3, &fatigue) {
            ProactiveRoute::Lv3(b) => b,
            other => panic!("{other:?}"),
        };
        let lines = brief.to_prompt_lines();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("쿠폰 적용 버튼") && lines[0].contains("반복 클릭 무응답"),
            "{lines:?}"
        );
        assert!(lines[1].contains("#1 '간편결제'"), "{lines:?}");
        let bare = build_fast_path_prompt(&Situation::default());
        let with = build_fast_path_prompt(&Situation {
            session: Some(SessionContext {
                friction_brief: Some(brief),
                ..Default::default()
            }),
            ..Default::default()
        });
        let cut = |p: &str| p[..p.find(USER_TURN).unwrap() + USER_TURN.len()].to_string();
        assert_eq!(cut(&with), cut(&bare), "system + tools prefix unchanged");
        assert!(with.ends_with("<|im_start|>assistant\n<think></think>\n\n"));
        assert!(with[with.find(USER_TURN).unwrap()..].contains("[막힌 곳]"));
        // B: all Unrated, first visit: no LLM.
        let none = CoachEvidence {
            point_at: Some(1),
            ..Default::default()
        };
        match route_proactive(&c, CoachDeliveryMode::Immediate, &none, true, 3, &fatigue) {
            ProactiveRoute::Lv2(l) => assert_eq!((l.line.as_str(), l.point_at), ("1번 쪽부터 차근차근 확인해 볼까?", Some(1))),
            other => panic!("{other:?}"),
        }
        // Evidence but the model is cooling down / not loaded: Lv2 too.
        assert!(matches!(
            route_proactive(&c, CoachDeliveryMode::PostDefeatComfort, &ev, false, 3, &fatigue),
            ProactiveRoute::Lv2(_)
        ));
        // C: the player asks while the proactive line is starting.
        assert_eq!(
            route_proactive(&c, CoachDeliveryMode::Immediate, &ev, true, 4, &fatigue),
            ProactiveRoute::Cancel(RouteCancel::UserQuestion)
        );
        assert!(
            c.superseded_by_question(4),
            "a running proactive stream checks the same epoch and stops"
        );
    }

    #[test]
    fn pf11_unanswered_offer_leaves_the_four_turns_and_a_quick_follow_up_promotes_it() {
        let mut dialogue = Dialogue::new(4);
        for i in 0..4 {
            dialogue.push(Some(&format!("질문 {i}")), &format!("답 {i}"));
        }
        let before = dialogue.turns();
        let offer = EphemeralOffer::new(
            "shop.example.com",
            Some("rec-1"),
            Some(1),
            "쿠폰 칸 말고 1번 간편결제 써볼래?",
            100_000,
        );
        // Branch A: 50 s of silence.
        let mut a = CoachingFatigueController::new();
        a.offer(offer.clone());
        assert_eq!(a.expire(130_000), None, "still within 45 s");
        assert_eq!(a.expire(150_000).map(|o| o.target_number), Some(Some(1)));
        assert_eq!(a.on_user_utterance("shop.example.com", 151_000), None, "nothing left to promote");
        assert_eq!(dialogue.turns(), before, "the four real turns are untouched");
        // Branch B: "응 그거 수수료 있어?" 10 s later.
        let mut b = CoachingFatigueController::new();
        b.offer(offer);
        let turn = b.on_user_utterance("shop.example.com", 110_000).expect("promoted");
        assert_eq!(turn.user, None);
        assert_eq!(turn.zena, "쿠폰 칸 말고 1번 간편결제 써볼래?");
        dialogue.push_turn(turn.clone());
        assert_eq!(dialogue.turns().last(), Some(&turn), "the offer is now the previous turn");
        assert_eq!(dialogue.len(), 4);
        // Another window does not inherit the offer.
        let mut c = CoachingFatigueController::new();
        c.offer(EphemeralOffer::new("eldenring.exe", None, Some(2), "2번 레버!", 0));
        assert_eq!(c.on_user_utterance("shop.example.com", 1_000), None);
    }

    #[test]
    fn pf12_three_ignores_mute_until_asked_and_picking_the_number_is_an_engage() {
        let mut f = CoachingFatigueController::new();
        assert_eq!(f.thresholds(), Some((1.0, 90_000)));
        f.on_ignored_or_dismissed(false);
        assert_eq!(f.state(), FatigueState::Receptive);
        f.on_ignored_or_dismissed(false);
        assert_eq!(f.state(), FatigueState::Dampened);
        assert_eq!(f.thresholds(), Some((1.5, 180_000)));
        assert!(f.lv2_only());
        f.on_ignored_or_dismissed(false);
        assert_eq!(f.state(), FatigueState::MutedUntilAsk);
        assert_eq!(f.thresholds(), None);
        let c = coach(0, 0);
        let ev = CoachEvidence {
            verified_alternative: Some("#1 '간편결제'".into()),
            ..Default::default()
        };
        for t in 0..5 {
            assert_eq!(
                route_proactive(&c, CoachDeliveryMode::Immediate, &ev, true, 0, &f),
                ProactiveRoute::Cancel(RouteCancel::Muted),
                "tick {t}: no proactive line while muted"
            );
        }
        f.on_user_utterance("eldenring.exe", 10_000);
        assert_eq!(f.state(), FatigueState::Receptive, "a question lifts the mute");
        assert!(matches!(
            route_proactive(&c, CoachDeliveryMode::Immediate, &ev, true, 0, &f),
            ProactiveRoute::Lv3(_)
        ));

        // Offer, then the player picks the suggested number 5 s later.
        f.offer(EphemeralOffer::new(
            "shop.example.com",
            Some("rec-42"),
            Some(1),
            "1번 간편결제 써볼래?",
            20_000,
        ));
        assert_eq!(
            f.on_target_picked("shop.example.com", 2, 22_000),
            None,
            "another number is not an engage"
        );
        let acc = f.on_target_picked("shop.example.com", 1, 25_000).expect("accepted");
        assert_eq!(acc.picked_after_ms(), 5_000);
        assert_eq!(acc.engage_weight, 0.2);
        let rec = acc.exposure().expect("exposure entry");
        assert_eq!(
            (rec.record_id.as_str(), rec.role, rec.w_spoken),
            ("rec-42", ExposureRole::RecommendedSolution, 1.0)
        );
        let mut ledger = crate::exposure::ExposureLedger::default();
        ledger.insert(rec);
        assert_eq!(ledger.active_in_window("shop.example.com", 26_000).len(), 1);
        // Too late does not count.
        f.offer(EphemeralOffer::new("shop.example.com", Some("rec-43"), Some(2), "2번?", 30_000));
        assert_eq!(f.on_target_picked("shop.example.com", 2, 30_000 + ACCEPT_WINDOW_MS + 1), None);
    }

    #[test]
    fn explicit_mute_is_immediate() {
        let mut f = CoachingFatigueController::new();
        f.on_ignored_or_dismissed(true);
        assert_eq!(f.state(), FatigueState::MutedUntilAsk);
        assert!(!f.allows_proactive());
    }

    #[test]
    fn cn256_ingame_defeat_trap_routes_to_post_defeat_comfort_without_combat_interruption() {
        let mut hook = PendingCoachHook::new();
        let fatigue = CoachingFatigueController::new();
        let combat = InterruptSignals {
            high_tension_combat: true,
            ..sig(10_000)
        };
        let trap_coach = PendingCoach::new("eldenring.exe", "boss-gate", Some("trap-door"), "defeat_trap", 10_000, 1);
        assert_eq!(hook.submit(trap_coach, &combat), None, "silent during combat");
        // Defeat triggers 1.5s delay then releases PostDefeatComfort with confirmed_trap evidence -> Lv3 comfort + warning.
        assert_eq!(
            hook.poll(
                "eldenring.exe",
                "boss-gate",
                false,
                1,
                &InterruptSignals { just_defeated: true, ..sig(15_000) }
            ),
            None
        );
        let (released, mode) = hook
            .poll("eldenring.exe", "boss-gate", false, 1, &sig(16_500))
            .expect("released after 1500ms");
        assert_eq!(mode, CoachDeliveryMode::PostDefeatComfort);
        let ev = CoachEvidence {
            bottleneck_label: Some("중앙 정문".into()),
            confirmed_trap: Some("정면 돌입 함정".into()),
            verified_alternative: Some("#1 '우측 사다리'".into()),
            point_at: Some(1),
        };
        match route_proactive(&released, mode, &ev, true, 1, &fatigue) {
            ProactiveRoute::Lv3(brief) => {
                let text = brief.to_prompt_lines().join("\n");
                assert!(text.contains("정면 돌입 함정") && text.contains("먼저 위로"), "{text}");
            }
            other => panic!("expected Lv3 PostDefeatComfort, got {other:?}"),
        }
    }

    #[test]
    fn cn274_reflex_compiler_distills_lv3_coaching_to_zero_ms_lv2_after_two_accepts() {
        let mut compiler = ReflexCompiler::new();
        let fatigue = CoachingFatigueController::new();
        let c = coach(1_000, 0);
        let ev = CoachEvidence {
            verified_alternative: Some("#1 '간편결제'".into()),
            point_at: Some(1),
            ..Default::default()
        };
        // Before 2 verified accepts -> still routes to Lv3.
        assert!(!compiler.record_lv3_success("eldenring.exe", "screen-a", Some("#2"), "1번 우측 사다리 써보자!", Some(1)));
        assert!(matches!(
            route_proactive_with_reflex(&c, CoachDeliveryMode::Immediate, &ev, true, 0, &fatigue, &compiler),
            ProactiveRoute::Lv3(_)
        ));
        // Second verified accept -> distills into 0ms Lv2 reflex (LLM never wakes even when lv3_available == true)!
        assert!(compiler.record_lv3_success("eldenring.exe", "screen-a", Some("#2"), "1번 우측 사다리 써보자!", Some(1)));
        match route_proactive_with_reflex(&c, CoachDeliveryMode::Immediate, &ev, true, 0, &fatigue, &compiler) {
            ProactiveRoute::Lv2(cached) => {
                assert_eq!(cached.line, "1번 우측 사다리 써보자!");
                assert_eq!(cached.point_at, Some(1));
            }
            other => panic!("expected 0ms Lv2 distilled reflex, got {other:?}"),
        }
        // Revocation restores Lv3 path.
        compiler.revoke("eldenring.exe", "screen-a", Some("#2"));
        assert!(matches!(
            route_proactive_with_reflex(&c, CoachDeliveryMode::Immediate, &ev, true, 0, &fatigue, &compiler),
            ProactiveRoute::Lv3(_)
        ));
    }
}
