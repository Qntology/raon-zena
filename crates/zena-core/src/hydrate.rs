//! Labels for remote search hits that come without text (`contracts/contextual_rag_v1.json` CR-10, scenario 3
//! part 4). The server keeps no text and no text hash, so a hit is only ids and hashes; inventing a label for it
//! would be a hallucination. Three steps, none of which blocks:
//!
//! 1. its `to` hash is on the current screen (NavMap region / element hashes, course checkpoints, the topic
//!    tracker's reverse table) → that label;
//! 2. its thread parent (`ref_`) or root (`root`) is one of my own notes in the local mirror → "my thread";
//! 3. otherwise it is only counted, and all such hits share one [`Barometer`] line with no label at all.
//!
//! The bounded single graph lookup of the plan's step 2-B (≤ 200 ms, top hit only) is an app decision: see
//! [`HydrationReport::top_unresolved`].

use crate::brief::prompt_label;
use crate::claims::{Barometer, CandidateKind, Grade, RecordStatus, VerdictCandidate};

/// A remote search hit without text (premium-discount-mcp `MergedMatch` with `text: None`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RemoteHit {
    pub id: String,
    pub cc: String,
    pub to: Option<String>,
    pub root: Option<String>,
    pub ref_: Option<String>,
    pub score: f32,
    pub global: Grade,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HydrationSource {
    /// Step 1: the `to` hash is a label on the current screen / course / topic table.
    ScreenLabel,
    /// Step 2: the hit replies in a thread of one of my own notes.
    MyThread,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HydratedHit {
    pub hit: RemoteHit,
    pub source: HydrationSource,
    /// What the slot may call it: "'우측 성문' 관련 평가", "'근접 원형 달리기' 스레드의 후속 평가".
    pub label: String,
}

impl HydratedHit {
    /// A slot candidate: someone else's evaluation, so the local grade is Unrated and the global one is the hit's.
    pub fn to_candidate(&self, canonical_topic: &str) -> VerdictCandidate {
        VerdictCandidate {
            record_id: self.hit.id.clone(),
            cc: self.hit.cc.clone(),
            to: self.hit.to.clone(),
            canonical_topic: canonical_topic.to_string(),
            label: self.label.clone(),
            claim: None,
            local: Grade::Unrated,
            global: self.hit.global,
            kind: if self.hit.global == Grade::Discount {
                CandidateKind::Trap
            } else {
                CandidateKind::Solution
            },
            status: RecordStatus::Active,
            score: self.hit.score,
            value_diff: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct HydrationReport {
    pub hydrated: Vec<HydratedHit>,
    pub unresolved: Vec<RemoteHit>,
    /// One line for every unresolved hit (`None` when all were hydrated).
    pub barometer: Option<Barometer>,
}

impl HydrationReport {
    /// The best unresolved hit — the only one a bounded (≤ 200 ms) graph lookup may be spent on, if the soft
    /// deadline still has ≥ 250 ms left and the app is online.
    pub fn top_unresolved(&self) -> Option<&RemoteHit> {
        self.unresolved.iter().max_by(|a, b| a.score.total_cmp(&b.score))
    }
}

/// The three-step label restorer over in-memory lookups.
pub struct LabelHydrator<'a> {
    /// Step 1: `to` hash → clean label (NavMap region / element hashes, course checkpoints, topic reverse table).
    pub label_for_to: &'a dyn Fn(&str) -> Option<String>,
    /// Step 2: record id (a hit's `ref_` / `root`) → title of my own note in the local mirror.
    pub my_note_for_id: &'a dyn Fn(&str) -> Option<String>,
}

impl LabelHydrator<'_> {
    pub fn hydrate(&self, hits: &[RemoteHit]) -> HydrationReport {
        let mut report = HydrationReport::default();
        for hit in hits {
            let screen = hit
                .to
                .as_deref()
                .and_then(|t| (self.label_for_to)(t))
                .map(|l| prompt_label(&l))
                .filter(|l| !l.is_empty());
            if let Some(label) = screen {
                report.hydrated.push(HydratedHit {
                    hit: hit.clone(),
                    source: HydrationSource::ScreenLabel,
                    label: format!("'{label}' 관련 평가"),
                });
                continue;
            }
            let thread = [hit.ref_.as_deref(), hit.root.as_deref()]
                .into_iter()
                .flatten()
                .find_map(|id| (self.my_note_for_id)(id))
                .map(|l| prompt_label(&l))
                .filter(|l| !l.is_empty());
            if let Some(note) = thread {
                report.hydrated.push(HydratedHit {
                    hit: hit.clone(),
                    source: HydrationSource::MyThread,
                    label: format!("'{note}' 스레드의 후속 평가"),
                });
                continue;
            }
            report.unresolved.push(hit.clone());
        }
        if !report.unresolved.is_empty() {
            report.barometer = Some(Barometer {
                total: report.unresolved.len(),
                premium: report.unresolved.iter().filter(|h| h.global == Grade::Premium).count(),
                discount: report.unresolved.iter().filter(|h| h.global == Grade::Discount).count(),
            });
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::{SlotRole, VerdictSlotPacker};
    use std::cell::Cell;
    use std::collections::HashMap;

    #[test]
    fn cr10_screen_label_my_thread_and_one_barometer_line_with_no_invented_text() {
        let gate = format!("0x{}", "1".repeat(40));
        let navmap: HashMap<String, String> = [(gate.clone(), "우측 성문".to_string())].into();
        let mirror: HashMap<String, String> = [("note-7".to_string(), "근접 원형 달리기".to_string())].into();
        let calls = Cell::new(0);
        let label_for_to = |t: &str| {
            calls.set(calls.get() + 1);
            navmap.get(t).cloned()
        };
        let my_note_for_id = |id: &str| {
            calls.set(calls.get() + 1);
            mirror.get(id).cloned()
        };
        let hits = [
            RemoteHit {
                id: "r1".into(),
                cc: "eldenring.exe".into(),
                to: Some(gate),
                score: 0.9,
                global: Grade::Premium,
                ..Default::default()
            },
            RemoteHit {
                id: "r2".into(),
                cc: "eldenring.exe".into(),
                to: Some(format!("0x{}", "2".repeat(40))),
                ref_: Some("note-7".into()),
                score: 0.8,
                global: Grade::Premium,
                ..Default::default()
            },
            RemoteHit {
                id: "r3".into(),
                cc: "eldenring.exe".into(),
                to: Some(format!("0x{}", "3".repeat(40))),
                root: Some("someone-else".into()),
                score: 0.7,
                global: Grade::Premium,
                ..Default::default()
            },
        ];
        let report = LabelHydrator {
            label_for_to: &label_for_to,
            my_note_for_id: &my_note_for_id,
        }
        .hydrate(&hits);
        assert!(calls.get() <= 2 * hits.len(), "in-memory lookups only, no graph calls");
        assert_eq!(report.hydrated.len(), 2);
        assert_eq!(report.hydrated[0].source, HydrationSource::ScreenLabel);
        assert_eq!(report.hydrated[0].label, "'우측 성문' 관련 평가");
        assert_eq!(report.hydrated[1].source, HydrationSource::MyThread);
        assert_eq!(report.hydrated[1].label, "'근접 원형 달리기' 스레드의 후속 평가");
        assert_eq!(report.unresolved.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), vec!["r3"]);
        let b = report.barometer.unwrap();
        assert_eq!((b.total, b.premium, b.discount), (1, 1, 0));
        let line = b.line();
        assert!(
            line.starts_with("[서버 여론]") && line.contains("1건") && line.contains("지어내지 말 것"),
            "{line}"
        );
        assert!(!line.contains("0x") && !line.contains("r3"), "no hash, no id, no invented label");
        assert_eq!(report.top_unresolved().map(|h| h.id.as_str()), Some("r3"));

        // Into the slots: two labelled solutions plus the barometer, still no hashes.
        let cands: Vec<_> = report.hydrated.iter().map(|h| h.to_candidate("")).collect();
        let slots = VerdictSlotPacker::default().pack(&cands, report.barometer.as_ref());
        assert_eq!(slots.len(), 3);
        assert_eq!(slots[2].role, SlotRole::Barometer);
        assert!(
            slots[0].line.contains("'우측 성문' 관련 평가: 다른 사용자 추천"),
            "{}",
            slots[0].line
        );
        assert!(slots.iter().all(|s| !s.line.contains("0x")));
    }

    #[test]
    fn nothing_hydrated_means_one_facts_only_barometer_and_hash_labels_are_rejected() {
        let navmap = |t: &str| Some(t.to_string()); // a broken lookup that returns the hash itself
        let none = |_: &str| None;
        let hits = [RemoteHit {
            id: "x".into(),
            to: Some(format!("0x{}", "ab".repeat(20))),
            ..Default::default()
        }];
        let r = LabelHydrator {
            label_for_to: &navmap,
            my_note_for_id: &none,
        }
        .hydrate(&hits);
        assert!(r.hydrated.is_empty(), "a label that is only a hash is no label");
        assert!(r.barometer.unwrap().facts_only());
        assert!(!r.barometer.unwrap().line().contains("추천"));
        assert_eq!(
            LabelHydrator {
                label_for_to: &none,
                my_note_for_id: &none
            }
            .hydrate(&[]),
            HydrationReport::default()
        );
    }
}
