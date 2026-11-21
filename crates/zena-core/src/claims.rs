//! Prompt material from evaluation records (`contracts/contextual_rag_v1.json`, scenario 3 part 4):
//!
//! * [`extract_claim`] / [`scrub_claim_for_prompt`] — one claim of at most [`MAX_CLAIM_CHARS`] characters per
//!   record that keeps the numbers that matter ("데미지 2500", "20%→15%", "2페이즈") and drops only personal data
//!   (e-mail, phone, card, resident number, wallet key, hashes). premium-discount-nav's `scrub_name` drops every
//!   word with four or more digits, which is right for screen labels but would erase strategy numbers.
//! * [`VerdictSlotPacker`] — the (at most) three verdict slots of `SessionContext`: the best solution, the
//!   runner-up (`[경합]` when it is a near tie) and a patch-superseded / trap warning, plus the remote-hit
//!   [`Barometer`] line. Local and global verdicts are said side by side, never blended; a record neither side
//!   judged (Unrated / Neutral locally and globally) is facts only — `[사실기록-판정전]`, never the word "추천".
//!   Rejected and failed records are left out; a record the server no longer has (missing) stays as my note
//!   with its global grade forced to Unrated.

use crate::brief::prompt_label;
use crate::ingress_guard::{find_pii, PiiKind};
use crate::text::{clip_chars, collapse_ws, drop_verdict_words, strip_hex_hashes};
use serde::{Deserialize, Serialize};

/// A claim in a verdict slot is at most this many characters.
pub const MAX_CLAIM_CHARS: usize = 38;
/// Local note text read for a claim is capped at this many bytes.
pub const MAX_CLAIM_SOURCE_BYTES: usize = 8_192;
/// At most this many verdict slots go into one prompt.
pub const MAX_VERDICT_SLOTS: usize = 3;

/// One side's verdict on a record — premium-discount-verdict's `Tier`, mirrored so zena-core stays free of
/// that dependency (the app maps `Tier::Premium` → `Grade::Premium`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Grade {
    Premium,
    Neutral,
    Discount,
    #[default]
    Unrated,
}

impl Grade {
    /// Whether this side says something Zena may repeat: Premium (추천) or Discount (비추천).
    pub fn asserts(self) -> bool {
        matches!(self, Grade::Premium | Grade::Discount)
    }

    fn word(self) -> &'static str {
        match self {
            Grade::Premium => "추천",
            Grade::Discount => "비추천",
            Grade::Neutral | Grade::Unrated => "",
        }
    }
}

/// Removes personal data from a claim — e-mail, phone number, Luhn card number, resident registration number,
/// private key, `0x…` hashes, a password after its keyword — and keeps every other number and unit.
pub fn scrub_claim_for_prompt(text: &str) -> String {
    let mut spans: Vec<_> = find_pii(text, None).into_iter().filter(|s| s.kind != PiiKind::LongDigits).collect();
    spans.sort_by_key(|s| s.start);
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    for s in spans {
        if s.start < at {
            continue;
        }
        out.push_str(&text[at..s.start]);
        out.push(' ');
        at = s.end;
    }
    out.push_str(&text[at..]);
    strip_hex_hashes(&out)
}

fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let next = chars.get(i + 1).copied();
        let end = match c {
            '\n' | '!' | '?' | '。' | '…' => true,
            '.' => next.is_none_or(char::is_whitespace),
            _ => false,
        };
        if c != '\n' {
            cur.push(c);
        }
        if end {
            let s = collapse_ws(&cur);
            if !s.is_empty() {
                out.push(s);
            }
            cur.clear();
        }
    }
    let s = collapse_ws(&cur);
    if !s.is_empty() {
        out.push(s);
    }
    out
}

fn numeric_tokens(s: &str) -> usize {
    s.split_whitespace().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count()
}

/// Cuts `s` to `max` characters on word boundaries, choosing the contiguous run of words that keeps the most
/// numbers (the earliest such run): numbers are never cut in half and are the last thing dropped.
pub fn clip_claim(s: &str, max: usize) -> String {
    let s = collapse_ws(s);
    if s.chars().count() <= max {
        return s;
    }
    let words: Vec<&str> = s.split(' ').collect();
    let lens: Vec<usize> = words.iter().map(|w| w.chars().count()).collect();
    let mut best: Option<(usize, usize, usize)> = None; // (numbers, start, end)
    for start in 0..words.len() {
        let mut len = 0;
        let mut end = start;
        while end < words.len() {
            let add = lens[end] + usize::from(end > start);
            if len + add > max {
                break;
            }
            len += add;
            end += 1;
        }
        if end == start {
            continue;
        }
        let nums = words[start..end].iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).count();
        if best.is_none_or(|(n, _, _)| nums > n) {
            best = Some((nums, start, end));
        }
    }
    match best {
        Some((_, start, end)) => words[start..end].join(" "),
        None => clip_chars(&s, max),
    }
}

/// The claim of a (long) local note: the sentence `score` ranks highest (the app passes its query similarity;
/// ties go to the sentence with more numbers, then the earlier one), scrubbed and clipped to
/// [`MAX_CLAIM_CHARS`]. `None` when nothing printable is left.
pub fn extract_claim(text: &str, score: impl Fn(&str) -> f32) -> Option<String> {
    let mut cut = text.len().min(MAX_CLAIM_SOURCE_BYTES);
    while !text.is_char_boundary(cut) {
        cut -= 1;
    }
    let mut best: Option<(f32, usize, String)> = None;
    for sentence in split_sentences(&text[..cut]) {
        let clean = scrub_claim_for_prompt(&sentence);
        if clean.is_empty() {
            continue;
        }
        let sc = score(&sentence);
        let sc = if sc.is_finite() { sc } else { f32::MIN };
        let nums = numeric_tokens(&clean);
        let better = match &best {
            None => true,
            Some((b, n, _)) => sc > *b || (sc == *b && nums > *n),
        };
        if better {
            best = Some((sc, nums, clean));
        }
    }
    best.map(|(_, _, s)| clip_claim(&s, MAX_CLAIM_CHARS)).filter(|s| !s.is_empty())
}

/// [`extract_claim`] ranked by word overlap with `query` (when no embedder is at hand).
pub fn extract_claim_for_query(text: &str, query: &str) -> Option<String> {
    let q: Vec<String> = query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.chars().count() >= 2)
        .collect();
    extract_claim(text, |s| {
        let lower = s.to_lowercase();
        q.iter().filter(|w| lower.contains(w.as_str())).count() as f32
    })
}

/// Where the record stands in my local outbox / the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordStatus {
    #[default]
    Active,
    /// Gone from the server (404 / retention) but still on my disk: kept as my note, global grade Unrated.
    Missing,
    /// Refused by a filter or the server's review: never briefed.
    Rejected,
    /// Sync failed for good: never briefed.
    Failed,
}

/// How the record relates to the question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    #[default]
    Solution,
    /// An older version a patch superseded (`Confirmed { value_diff }`).
    SupersededOld,
    /// A known trap (Discount / FrictionTrap).
    Trap,
}

/// One record that may go into a verdict slot.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VerdictCandidate {
    pub record_id: String,
    pub cc: String,
    pub to: Option<String>,
    /// Short topic key used to recognise the record in what Zena said ("회피", "대방패").
    pub canonical_topic: String,
    /// Display label (hydrated, scrubbed): "회피 공략".
    pub label: String,
    /// A claim from [`extract_claim`].
    pub claim: Option<String>,
    pub local: Grade,
    pub global: Grade,
    pub kind: CandidateKind,
    pub status: RecordStatus,
    pub score: f32,
    /// Old → new value of a superseded record ("20%", "15%").
    pub value_diff: Option<(String, String)>,
}

impl VerdictCandidate {
    fn is_warning(&self) -> bool {
        let discount = self.local == Grade::Discount || self.global == Grade::Discount;
        let premium = self.local == Grade::Premium || self.global == Grade::Premium;
        self.kind != CandidateKind::Solution || (discount && !premium)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotRole {
    /// The best (or a clearly second) solution.
    Solution,
    /// A runner-up within the contest margin of the best one.
    Contested,
    /// A patch-superseded version or a trap: said as a warning.
    Warning,
    /// The aggregate line of remote hits whose label could not be restored.
    Barometer,
}

/// One packed verdict line and what it refers to (for the exposure ledger).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedVerdictSlot {
    /// 1-based position.
    pub slot: u8,
    pub role: SlotRole,
    /// Empty for the barometer.
    pub record_id: String,
    pub cc: String,
    pub to: Option<String>,
    pub canonical_topic: String,
    pub label: String,
    pub claim: Option<String>,
    pub facts_only: bool,
    /// The line rendered into the user turn.
    pub line: String,
}

const GENERIC_TOPIC_WORDS: [&str; 8] = ["공략", "평가", "관련", "기록", "방법", "팁", "스레드", "후속"];
const PARTICLES: [&str; 15] = [
    "으로", "에서", "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만",
];

fn stem(word: &str) -> String {
    let w = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    for p in PARTICLES {
        if let Some(s) = w.strip_suffix(p) {
            if s.chars().count() >= 2 {
                return s.to_string();
            }
        }
    }
    w
}

impl PackedVerdictSlot {
    /// Words whose presence in a spoken line means this slot was said: the topic key's words with particles
    /// stripped (the label's when the key is empty), generic words like "공략" left out.
    pub fn match_terms(&self) -> Vec<String> {
        let source = if self.canonical_topic.trim().is_empty() {
            &self.label
        } else {
            &self.canonical_topic
        };
        let mut terms: Vec<String> = source
            .split_whitespace()
            .map(stem)
            .filter(|w| w.chars().count() >= 2 && !GENERIC_TOPIC_WORDS.contains(&w.as_str()))
            .collect();
        terms.dedup();
        terms
    }

    /// The text compared semantically with what was said: the claim, else the label.
    pub fn gist(&self) -> &str {
        self.claim.as_deref().filter(|c| !c.is_empty()).unwrap_or(&self.label)
    }
}

/// Counts of remote hits whose label could not be restored, merged into one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Barometer {
    pub total: usize,
    pub premium: usize,
    pub discount: usize,
}

impl Barometer {
    /// No hit carries a judgement: the line says nothing about recommending.
    pub fn facts_only(&self) -> bool {
        self.premium + self.discount == 0
    }

    pub fn line(&self) -> String {
        const NO_TEXT: &str = "세부 원문 없음 — 없는 수치·조작법 지어내지 말 것";
        if self.facts_only() {
            format!("[서버 여론] 관련 평가 {}건 (판정 전 · {NO_TEXT})", self.total)
        } else {
            format!(
                "[서버 여론] 다른 사용자 평가 {}건 중 추천 {} · 비추천 {} ({NO_TEXT})",
                self.total, self.premium, self.discount
            )
        }
    }
}

/// Packs records into at most [`MAX_VERDICT_SLOTS`] lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VerdictSlotPacker {
    /// A runner-up within this score margin of the best solution is tagged `[경합]`.
    pub contested_margin: f32,
    /// Each line is cut to this many characters.
    pub max_line_chars: usize,
}

impl Default for VerdictSlotPacker {
    fn default() -> Self {
        Self {
            contested_margin: 0.05,
            max_line_chars: 80,
        }
    }
}

fn stance_words(local: Grade, global: Grade) -> String {
    match (local.asserts(), global.asserts()) {
        (true, true) if local == global => format!("내 기록·서버 모두 {}", local.word()),
        (true, true) => format!("내 기록은 {}·서버는 {}", local.word(), global.word()),
        (true, false) => format!("내 기록으로 {}", local.word()),
        (false, true) => format!("다른 사용자 {}", global.word()),
        (false, false) => String::new(),
    }
}

impl VerdictSlotPacker {
    fn render(&self, c: &VerdictCandidate, role: SlotRole) -> (String, bool) {
        let facts = !(c.local.asserts() || c.global.asserts());
        let tidy = |s: &str| if facts { drop_verdict_words(s) } else { s.to_string() };
        let label = tidy(&prompt_label(&c.label));
        let claim = c
            .claim
            .as_deref()
            .map(|s| tidy(&clip_claim(&scrub_claim_for_prompt(s), MAX_CLAIM_CHARS)))
            .filter(|s| !s.is_empty());
        let diff = c
            .value_diff
            .as_ref()
            .map(|(old, new)| format!("{}→{}", prompt_label(old), prompt_label(new)));
        let header = if facts {
            "[사실기록-판정전]"
        } else {
            match role {
                SlotRole::Solution => "[해법]",
                SlotRole::Contested => "[경합]",
                SlotRole::Warning if diff.is_some() || c.kind == CandidateKind::SupersededOld => "[주의-패치대체]",
                SlotRole::Warning => "[주의-함정]",
                SlotRole::Barometer => "[서버 여론]",
            }
        };
        let mut body = label;
        if !facts {
            body.push_str(": ");
            body.push_str(&stance_words(c.local, c.global));
        }
        if let Some(d) = diff.filter(|_| role == SlotRole::Warning) {
            body.push_str(&format!(" · {d} (패치로 바뀜)"));
        }
        if let Some(cl) = claim {
            body.push_str(" — ");
            body.push_str(&cl);
        }
        (clip_chars(&format!("{header} {body}"), self.max_line_chars), facts)
    }

    fn slot(&self, n: usize, c: &VerdictCandidate, role: SlotRole) -> PackedVerdictSlot {
        let (line, facts_only) = self.render(c, role);
        PackedVerdictSlot {
            slot: n as u8,
            role,
            record_id: c.record_id.clone(),
            cc: c.cc.clone(),
            to: c.to.clone(),
            canonical_topic: c.canonical_topic.clone(),
            label: prompt_label(&c.label),
            claim: c.claim.clone(),
            facts_only,
            line,
        }
    }

    /// The slots for `candidates` (any order) and the optional barometer of unrestored remote hits.
    pub fn pack(&self, candidates: &[VerdictCandidate], barometer: Option<&Barometer>) -> Vec<PackedVerdictSlot> {
        let live: Vec<VerdictCandidate> = candidates
            .iter()
            .filter(|c| !matches!(c.status, RecordStatus::Rejected | RecordStatus::Failed))
            .map(|c| {
                let mut c = c.clone();
                if c.status == RecordStatus::Missing {
                    c.global = Grade::Unrated;
                }
                c
            })
            .collect();
        let by_score = |a: &&VerdictCandidate, b: &&VerdictCandidate| b.score.total_cmp(&a.score);
        let mut solutions: Vec<&VerdictCandidate> = live.iter().filter(|c| !c.is_warning()).collect();
        let mut warnings: Vec<&VerdictCandidate> = live.iter().filter(|c| c.is_warning()).collect();
        solutions.sort_by(by_score);
        warnings.sort_by(by_score);

        let mut plan: Vec<(&VerdictCandidate, SlotRole)> = Vec::new();
        if let Some(first) = solutions.first() {
            plan.push((first, SlotRole::Solution));
            if let Some(second) = solutions.get(1) {
                let role = if (first.score - second.score).abs() < self.contested_margin {
                    SlotRole::Contested
                } else {
                    SlotRole::Solution
                };
                plan.push((second, role));
            }
        }
        if let Some(w) = warnings.first() {
            plan.push((w, SlotRole::Warning));
        }
        let mut out: Vec<PackedVerdictSlot> = plan.into_iter().enumerate().map(|(i, (c, r))| self.slot(i + 1, c, r)).collect();
        if let Some(b) = barometer.filter(|b| b.total > 0) {
            if out.len() < MAX_VERDICT_SLOTS {
                out.push(PackedVerdictSlot {
                    slot: out.len() as u8 + 1,
                    role: SlotRole::Barometer,
                    record_id: String::new(),
                    cc: String::new(),
                    to: None,
                    canonical_topic: String::new(),
                    label: String::new(),
                    claim: None,
                    facts_only: b.facts_only(),
                    line: b.line(),
                });
            }
        }
        let rest = solutions
            .iter()
            .skip(2)
            .map(|c| (*c, SlotRole::Solution))
            .chain(warnings.iter().skip(1).map(|c| (*c, SlotRole::Warning)));
        for (c, r) in rest {
            if out.len() >= MAX_VERDICT_SLOTS {
                break;
            }
            let n = out.len() + 1;
            out.push(self.slot(n, c, r));
        }
        out.truncate(MAX_VERDICT_SLOTS);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{build_fast_path_prompt, SessionContext, Situation};

    fn cand(id: &str, topic: &str, local: Grade, global: Grade, score: f32) -> VerdictCandidate {
        VerdictCandidate {
            record_id: id.into(),
            cc: "eldenring.exe".into(),
            canonical_topic: topic.into(),
            label: format!("{topic} 공략"),
            local,
            global,
            score,
            ..Default::default()
        }
    }

    #[test]
    fn claims_keep_strategy_numbers_and_drop_personal_data() {
        let s = scrub_claim_for_prompt("데미지 2500, 배송비 3000원, 20%→15% 너프. 문의 hong@example.com 010-1234-5678");
        assert!(s.contains("2500") && s.contains("3000원") && s.contains("20%→15%"), "{s}");
        assert!(!s.contains("hong@") && !s.contains("5678"), "{s}");
        let card = scrub_claim_for_prompt("카드 4111 1111 1111 1111 로 결제");
        assert!(!card.contains("4111"), "{card}");
        let hash = scrub_claim_for_prompt(&format!("체크포인트 0x{} 통과", "ab12".repeat(10)));
        assert_eq!(hash, "체크포인트 통과");
    }

    #[test]
    fn the_claim_is_the_best_sentence_within_38_chars_with_its_numbers_whole() {
        let note =
            "오늘 보스 잡았다. 패치 이후 말레니아 2페이즈 물새 난무 데미지가 20%→15%로 줄어서 이제 맞고 버틸 만하다. 다음엔 출혈 빌드.";
        let c = extract_claim_for_query(note, "물새 난무 데미지").unwrap();
        assert!(c.chars().count() <= MAX_CLAIM_CHARS, "{c}");
        assert!(c.contains("2페이즈") && c.contains("20%→15%로"), "{c}");
        assert_eq!(clip_claim("짧은 문장 1개", 38), "짧은 문장 1개");
        let long = "아주 긴 수식어가 앞에 잔뜩 붙어 있는 문장인데 마지막에야 핵심 수치 3회 반복 후 9초 대기가 나온다";
        let k = clip_claim(long, 38);
        assert!(k.contains("3회") && k.contains("9초"), "the window keeps the numbers: {k}");
        assert_eq!(extract_claim("   ", |_| 1.0), None);
        assert_eq!(
            extract_claim("1.5초 뒤 회피. 끝", |s| if s.contains("1.5초") { 1.0 } else { 0.0 }).as_deref(),
            Some("1.5초 뒤 회피.")
        );
    }

    #[test]
    fn slots_are_solution_runner_up_and_warning_side_by_side() {
        let packer = VerdictSlotPacker::default();
        let mut trap = cand("c", "대방패", Grade::Discount, Grade::Discount, 0.7);
        trap.kind = CandidateKind::Trap;
        let mut old = cand("d", "구버전 회피", Grade::Unrated, Grade::Premium, 0.9);
        old.kind = CandidateKind::SupersededOld;
        old.value_diff = Some(("20%".into(), "15%".into()));
        let slots = packer.pack(
            &[
                cand("a", "회피", Grade::Premium, Grade::Premium, 0.91),
                cand("b", "항아리", Grade::Premium, Grade::Discount, 0.89),
                trap,
                old,
            ],
            None,
        );
        assert_eq!(slots.len(), 3);
        assert_eq!((slots[0].role, slots[0].record_id.as_str()), (SlotRole::Solution, "a"));
        assert!(
            slots[0].line.starts_with("[해법] 회피 공략: 내 기록·서버 모두 추천"),
            "{}",
            slots[0].line
        );
        assert_eq!(slots[1].role, SlotRole::Contested, "0.02 apart");
        assert!(
            slots[1].line.contains("내 기록은 추천·서버는 비추천"),
            "never blended: {}",
            slots[1].line
        );
        assert_eq!(slots[2].role, SlotRole::Warning);
        assert_eq!(slots[2].record_id, "d", "the higher-scored warning");
        assert!(
            slots[2].line.starts_with("[주의-패치대체]") && slots[2].line.contains("20%→15%"),
            "{}",
            slots[2].line
        );
        assert!(slots.iter().all(|s| s.line.chars().count() <= 80));
    }

    #[test]
    fn cr11_rejected_is_gone_missing_survives_and_unrated_is_facts_only() {
        let mut rejected = cand("rej", "가드 반격", Grade::Premium, Grade::Premium, 0.99);
        rejected.status = RecordStatus::Rejected;
        let mut failed = cand("fail", "점프 공격", Grade::Premium, Grade::Unrated, 0.98);
        failed.status = RecordStatus::Failed;
        let mut missing = cand("miss", "근접 원형 달리기", Grade::Unrated, Grade::Premium, 0.8);
        missing.status = RecordStatus::Missing;
        missing.claim = Some("보스 왼쪽으로 원형 달리기 3바퀴. 추천함".into());
        let unrated = cand("new", "물병 투척", Grade::Unrated, Grade::Neutral, 0.7);
        let slots = VerdictSlotPacker::default().pack(
            &[rejected, failed, missing, unrated],
            Some(&Barometer {
                total: 2,
                ..Default::default()
            }),
        );
        let ids: Vec<&str> = slots.iter().map(|s| s.record_id.as_str()).collect();
        assert!(!ids.contains(&"rej") && !ids.contains(&"fail"), "{ids:?}");
        assert!(ids.contains(&"miss"), "my note survives as a local memo");
        let missing_slot = slots.iter().find(|s| s.record_id == "miss").unwrap();
        assert!(
            missing_slot.facts_only,
            "the server's old Premium does not count once it is missing"
        );
        for s in &slots {
            assert!(s.facts_only, "{s:?}");
            assert!(!s.line.contains("추천"), "{}", s.line);
        }
        assert!(slots
            .iter()
            .filter(|s| s.role != SlotRole::Barometer)
            .all(|s| s.line.starts_with("[사실기록-판정전]")));
        assert!(slots
            .iter()
            .any(|s| s.role == SlotRole::Barometer && s.line.starts_with("[서버 여론] 관련 평가 2건")));
        let prompt = build_fast_path_prompt(&Situation {
            session: Some(SessionContext {
                verdict_slots: slots,
                verdicts: vec!["옛 경로의 줄: 모두 추천".into()],
                ..Default::default()
            }),
            ..Default::default()
        });
        assert!(prompt.contains("[사실기록-판정전] 근접 원형 달리기"), "{prompt}");
        assert!(!prompt.contains("추천"), "packed slots replace the legacy [추천] lines");
    }

    #[test]
    fn a_discount_only_record_is_a_warning_and_a_barometer_fills_a_free_slot() {
        let slots = VerdictSlotPacker::default().pack(
            &[cand("t", "쿠폰 칸", Grade::Unrated, Grade::Discount, 0.5)],
            Some(&Barometer {
                total: 3,
                premium: 2,
                discount: 1,
            }),
        );
        assert_eq!(slots[0].role, SlotRole::Warning);
        assert!(
            slots[0].line.starts_with("[주의-함정] 쿠폰 칸 공략: 다른 사용자 비추천"),
            "{}",
            slots[0].line
        );
        assert_eq!(slots[1].role, SlotRole::Barometer);
        assert!(slots[1].line.contains("추천 2 · 비추천 1"));
        assert_eq!(slots[1].slot, 2);
        assert!(VerdictSlotPacker::default().pack(&[], None).is_empty());
    }

    #[test]
    fn match_terms_strip_particles_and_generic_words() {
        let s = VerdictSlotPacker::default().slot(1, &cand("x", "대방패로", Grade::Premium, Grade::Unrated, 1.0), SlotRole::Solution);
        assert_eq!(s.match_terms(), vec!["대방패".to_string()]);
        let mut c = cand("y", "", Grade::Premium, Grade::Unrated, 1.0);
        c.label = "우측 사다리 공략".into();
        assert_eq!(
            VerdictSlotPacker::default().slot(1, &c, SlotRole::Solution).match_terms(),
            vec!["우측", "사다리"]
        );
    }
}
