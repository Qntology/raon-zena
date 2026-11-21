//! The 5-minute exposure ledger (`contracts/contextual_rag_v1.json` CR-12, scenario 3 part 4 → scenario 4's
//! credit back-propagation, row 0 `w_spoken`): which records Zena actually *said*, and in which role.
//!
//! Only sentences that really went out (the bubble / TTS — what `ToolCallStream` delivered) are matched against
//! the slots that were injected into the prompt. A generation cancelled before its first sentence delivers
//! nothing and registers nothing; a slot that was in the prompt but not spoken is not registered either.
//! A spoken warning slot is registered as [`ExposureRole::WarnedTrap`], so a later win never credits a trap.

use crate::claims::{PackedVerdictSlot, SlotRole};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Exposures are kept this long.
pub const EXPOSURE_TTL_MS: u64 = 300_000;
/// At most this many exposures are kept.
pub const MAX_EXPOSURES: usize = 32;
/// Semantic similarity (the app's sentence embedding cosine) from which an unnamed slot counts as said.
pub const SEMANTIC_MATCH_MIN: f32 = 0.68;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExposureRole {
    /// Said as a solution / alternative (a win credits it, a loss may debit it).
    RecommendedSolution,
    /// Said as an outdated version or a trap to avoid (a win must never credit it).
    WarnedTrap,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExposedRecord {
    pub record_id: String,
    pub cc: String,
    pub to: Option<String>,
    pub canonical_topic: String,
    pub role: ExposureRole,
    /// 1.0 when the topic was named, 0.75..1.0 for a semantic match.
    pub w_spoken: f32,
    pub exposed_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ExposureLedger {
    entries: VecDeque<ExposedRecord>,
    pub ttl_ms: u64,
    pub max_entries: usize,
}

impl Default for ExposureLedger {
    fn default() -> Self {
        Self {
            entries: VecDeque::new(),
            ttl_ms: EXPOSURE_TTL_MS,
            max_entries: MAX_EXPOSURES,
        }
    }
}

/// How strongly `delivered` says `slot`: 1.0 when one of its topic words is in it, else a semantic weight in
/// 0.75..=1.0 when `similarity` reaches [`SEMANTIC_MATCH_MIN`], else `None`.
pub fn spoken_weight(slot: &PackedVerdictSlot, delivered: &str, similarity: impl Fn(&str, &str) -> f32) -> Option<f32> {
    let lower = delivered.to_lowercase();
    if slot.match_terms().iter().any(|t| lower.contains(t.as_str())) {
        return Some(1.0);
    }
    let sim = similarity(slot.gist(), delivered);
    (sim >= SEMANTIC_MATCH_MIN).then(|| (0.75 + 0.25 * (sim - SEMANTIC_MATCH_MIN) / (1.0 - SEMANTIC_MATCH_MIN)).clamp(0.75, 1.0))
}

impl ExposureLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Adds (or refreshes) an exposure; the same record in the same role is kept once with the higher weight.
    pub fn insert(&mut self, rec: ExposedRecord) {
        let mut rec = rec;
        if let Some(i) = self.entries.iter().position(|e| e.record_id == rec.record_id && e.role == rec.role) {
            let old = self.entries.remove(i).expect("index from position");
            rec.w_spoken = rec.w_spoken.max(old.w_spoken);
        }
        self.entries.push_back(rec);
        while self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Matches the injected slots against the sentences actually delivered this turn and registers the spoken
    /// ones. Returns how many were registered (0 when nothing was delivered).
    pub fn record_delivered_turn(
        &mut self,
        injected_slots: &[PackedVerdictSlot],
        delivered_speech: &str,
        similarity_fn: impl Fn(&str, &str) -> f32,
        now_ms: u64,
    ) -> usize {
        let spoken = delivered_speech.trim();
        if spoken.is_empty() {
            return 0;
        }
        let mut n = 0;
        for slot in injected_slots
            .iter()
            .filter(|s| s.role != SlotRole::Barometer && !s.record_id.is_empty())
        {
            let Some(w) = spoken_weight(slot, spoken, &similarity_fn) else {
                continue;
            };
            self.insert(ExposedRecord {
                record_id: slot.record_id.clone(),
                cc: slot.cc.clone(),
                to: slot.to.clone(),
                canonical_topic: slot.canonical_topic.clone(),
                role: if slot.role == SlotRole::Warning {
                    ExposureRole::WarnedTrap
                } else {
                    ExposureRole::RecommendedSolution
                },
                w_spoken: w,
                exposed_at_ms: now_ms,
            });
            n += 1;
        }
        n
    }

    /// Drops exposures older than the TTL.
    pub fn prune(&mut self, now_ms: u64) {
        let ttl = self.ttl_ms;
        self.entries.retain(|e| now_ms.saturating_sub(e.exposed_at_ms) <= ttl);
    }

    /// Exposures of the last `ttl_ms` in window `cc`.
    pub fn active_in_window(&mut self, cc: &str, now_ms: u64) -> Vec<ExposedRecord> {
        self.prune(now_ms);
        self.entries.iter().filter(|e| e.cc == cc).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{CandidateKind, Grade, VerdictCandidate, VerdictSlotPacker};
    use crate::prompt::{build_fast_path_prompt, SessionContext, Situation, ASSISTANT_PREFILL, USER_TURN};

    fn slots() -> Vec<PackedVerdictSlot> {
        let c = |id: &str, topic: &str, label: &str, score: f32, kind: CandidateKind, g: Grade| VerdictCandidate {
            record_id: id.into(),
            cc: "eldenring.exe".into(),
            canonical_topic: topic.into(),
            label: label.into(),
            local: g,
            global: g,
            kind,
            score,
            ..Default::default()
        };
        VerdictSlotPacker::default().pack(
            &[
                c("dodge", "회피", "2페이즈 회피", 0.95, CandidateKind::Solution, Grade::Premium),
                c("jar", "항아리", "항아리 던지기", 0.80, CandidateKind::Solution, Grade::Premium),
                c("shield", "대방패", "대방패 막기", 0.70, CandidateKind::Trap, Grade::Discount),
            ],
            None,
        )
    }

    #[test]
    fn cr12_only_spoken_slots_are_registered_with_their_role_and_the_prefix_is_unchanged() {
        let injected = slots();
        assert_eq!(
            injected.iter().map(|s| s.role).collect::<Vec<_>>(),
            vec![SlotRole::Solution, SlotRole::Solution, SlotRole::Warning]
        );
        let bare = build_fast_path_prompt(&Situation::default());
        let prompt = build_fast_path_prompt(&Situation {
            user_query: Some("말레니아 2페이즈 어떻게 해?".into()),
            session: Some(SessionContext {
                verdict_slots: injected.clone(),
                ..Default::default()
            }),
            ..Default::default()
        });
        let cut = |p: &str| p[..p.find(USER_TURN).unwrap() + USER_TURN.len()].to_string();
        assert_eq!(cut(&prompt), cut(&bare), "system + tools prefix byte-identical");
        assert!(prompt.ends_with(ASSISTANT_PREFILL));

        let mut ledger = ExposureLedger::new();
        let delivered = "2페이즈는 회피가 답이야! 대방패로 막는 건 패치 뒤로 안 통해.";
        let n = ledger.record_delivered_turn(&injected, delivered, |_, _| 0.0, 1_000);
        assert_eq!(n, 2);
        let active = ledger.active_in_window("eldenring.exe", 2_000);
        let dodge = active.iter().find(|e| e.record_id == "dodge").unwrap();
        assert_eq!((dodge.role, dodge.w_spoken), (ExposureRole::RecommendedSolution, 1.0));
        let shield = active.iter().find(|e| e.record_id == "shield").unwrap();
        assert_eq!(shield.role, ExposureRole::WarnedTrap, "a spoken trap is sign-protected");
        assert!(active.iter().all(|e| e.record_id != "jar"), "the unspoken slot 2 is not registered");
    }

    #[test]
    fn a_cancelled_turn_registers_nothing_and_entries_expire_after_five_minutes() {
        let mut ledger = ExposureLedger::new();
        assert_eq!(
            ledger.record_delivered_turn(&slots(), "", |_, _| 1.0, 0),
            0,
            "epoch cancelled before the first sentence"
        );
        assert_eq!(ledger.record_delivered_turn(&slots(), "회피!", |_, _| 0.0, 10_000), 1);
        assert_eq!(ledger.active_in_window("eldenring.exe", 10_000 + EXPOSURE_TTL_MS).len(), 1);
        assert!(ledger.active_in_window("eldenring.exe", 10_001 + EXPOSURE_TTL_MS).is_empty());
        assert!(ledger.active_in_window("other.exe", 10_000).is_empty());
    }

    #[test]
    fn semantic_matches_weigh_between_075_and_1_and_duplicates_merge() {
        let s = &slots()[1];
        assert_eq!(spoken_weight(s, "그 단지 같은 걸 던져 봐", |_, _| 0.5), None);
        let w = spoken_weight(s, "그 단지 같은 걸 던져 봐", |_, _| 0.84).unwrap();
        assert!((w - 0.875).abs() < 1e-4, "{w}");
        assert_eq!(spoken_weight(s, "그 단지", |_, _| f32::NAN), None);
        let mut ledger = ExposureLedger::new();
        ledger.record_delivered_turn(&slots(), "항아리 던지기 좋아", |_, _| 0.0, 0);
        ledger.record_delivered_turn(&slots(), "항아리 또 던져", |_, _| 0.0, 5_000);
        assert_eq!(ledger.len(), 1, "one record, one role, refreshed");
        assert_eq!(ledger.active_in_window("eldenring.exe", 5_000)[0].exposed_at_ms, 5_000);
        let mut small = ExposureLedger {
            max_entries: 1,
            ..ExposureLedger::default()
        };
        small.record_delivered_turn(&slots(), "회피하고 항아리", |_, _| 0.0, 0);
        assert_eq!(small.len(), 1, "capped");
    }
}
