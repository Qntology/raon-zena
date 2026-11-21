//! Per-request coaching slots of Raon-1-nano's user turn, re-exported from [`crate::prompt`]:
//!
//! * [`CourseBrief`] — time-trial ghost splits and the next step (`contracts/course_nav_v1.json` CN-11 / CN-12);
//! * [`FrictionBrief`] — a proactive tip about where the player is stuck (`proactive_friction_v1` PF-10);
//! * [`WebCoachBrief`] — shopping-page coaching with the final-payment CommitGuard and the sold-out fallback
//!   (`web_analytic_v1` WA-10);
//! * [`AnalyticBrief`] — a site operator's funnel report with the small-sample guard (WA-11).
//!
//! They all render inside `SessionContext`, i.e. after `<|im_start|>user\n`: the cached system + tools prefix
//! and the `<think></think>` prefill never change. They never print raw `0x…` hashes (a model shown a hash
//! invents a button name for it), and a record nobody judged yet is described as facts only — no "추천" wording.

use crate::coach::CoachDeliveryMode;
use crate::text::{clip_chars, collapse_ws, contains_any, drop_verdict_words, has_batchim, josa, strip_hex_hashes};

/// Split differences below this many ms are "the same pace" (splits are rounded to 100 ms; ±100..300 ms is noise
/// that would only burn the 90-character line on decimals).
pub const SPLIT_DEADBAND_MS: i64 = 1_000;
/// Labels in a brief are cut to this many characters (premium-discount-nav's `scrub_name` limit).
pub const LABEL_MAX_CHARS: usize = 40;
/// Funnel windows with fewer entering sessions than this are not compared.
pub const MIN_FUNNEL_SESSIONS: u32 = 15;
/// An operator answer is at most this many characters.
pub const ANALYTIC_ANSWER_MAX_CHARS: usize = 90;

/// A label as a brief may print it: hash-like tokens removed, whitespace collapsed, at most 40 characters.
pub fn prompt_label(s: &str) -> String {
    clip_chars(&strip_hex_hashes(s), LABEL_MAX_CHARS)
}

const AWAIT_TRANSITION_LINE: &str = "[다음 스텝] 다음 화면 전환 대기 (화면에 없는 버튼 이름을 지어내지 말 것)";

/// The next step of the course, as Zena may talk about it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum NextStepHint {
    /// The next checkpoint is badge `number` on screen.
    NumberedOnScreen {
        number: u8,
        clean_label: String,
        direction_label: String,
    },
    /// Not on screen, but the gesture to reach it is known ("아래로 스크롤해 다음 구간으로").
    GesturePrompt { action_label: String },
    /// Only a hash is known (the next screen, an unopened modal): no label is printed at all.
    #[default]
    AwaitScreenTransition,
    /// The finish checkpoint is behind.
    Finished,
}

/// The run relative to the tracked ghost (premium-discount-nav `GhostSyncState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CourseSync {
    #[default]
    OnTrack,
    Detouring,
    Rejoined,
}

impl CourseSync {
    /// `"on_track"` | `"detouring"` | `"rejoined"`.
    pub fn tag(self) -> &'static str {
        match self {
            CourseSync::OnTrack => "on_track",
            CourseSync::Detouring => "detouring",
            CourseSync::Rejoined => "rejoined",
        }
    }
}

/// The public split `vs_public_ms` is measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PublicRef {
    #[default]
    Average,
    Record,
}

impl PublicRef {
    fn subject(self) -> &'static str {
        match self {
            PublicRef::Average => "타인 평균",
            PublicRef::Record => "타인 최고",
        }
    }
}

/// The live course coaching slot (from premium-discount-nav's `LiveGuide`, mapped by the app).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CourseBrief {
    pub passed: usize,
    pub length: usize,
    /// vs my best (ms, negative = ahead).
    pub vs_best_ms: Option<i64>,
    /// vs the public reference ([`PublicRef`]).
    pub vs_public_ms: Option<i64>,
    pub public_ref: PublicRef,
    /// A verified public shortcut beats my usual route by this much (ms).
    pub habit_upgrade_saved_ms: Option<u32>,
    pub sync: CourseSync,
    pub next_hint: NextStepHint,
    /// The course is judged Premium; `false` (Unrated / Neutral) → facts only.
    pub verdict_judged_premium: bool,
}

/// "내 최고와 동등" / "타인 평균보다 3.2초 빠름" with the 1-second dead band.
pub fn split_phrase(subject: &str, ms: i64) -> String {
    if ms.abs() < SPLIT_DEADBAND_MS {
        format!("{} 동등", josa(subject, "과", "와"))
    } else {
        let secs = ms.unsigned_abs() as f64 / 1000.0;
        format!("{subject}보다 {secs:.1}초 {}", if ms < 0 { "빠름" } else { "느림" })
    }
}

impl CourseBrief {
    pub const MAX_LINES: usize = 2;

    /// Unrated / Neutral course: say what happened, never recommend.
    pub fn facts_only(&self) -> bool {
        !self.verdict_judged_premium
    }

    /// At most two representative splits (my best, then the public reference).
    pub fn split_summary(&self) -> Vec<String> {
        let mut v = Vec::new();
        if let Some(ms) = self.vs_best_ms {
            v.push(split_phrase("내 최고", ms));
        }
        if let Some(ms) = self.vs_public_ms {
            v.push(split_phrase(self.public_ref.subject(), ms));
        }
        v
    }

    fn label(&self, s: &str) -> String {
        let l = prompt_label(s);
        if self.facts_only() {
            drop_verdict_words(&l)
        } else {
            l
        }
    }

    fn next_step_line(&self) -> String {
        let tag = if self.facts_only() {
            "[다음 관찰 스텝]"
        } else {
            "[다음 추천 스텝]"
        };
        match &self.next_hint {
            NextStepHint::NumberedOnScreen {
                number,
                clean_label,
                direction_label,
            } if (1..=9).contains(number) => {
                let mut s = format!("{tag} #{number}");
                let label = self.label(clean_label);
                if !label.is_empty() {
                    s.push_str(&format!(" '{label}'"));
                }
                let dir = prompt_label(direction_label);
                if !dir.is_empty() {
                    s.push_str(&format!(" ({dir})"));
                }
                s
            }
            NextStepHint::GesturePrompt { action_label } if !self.label(action_label).is_empty() => {
                format!("{tag} {}", self.label(action_label))
            }
            NextStepHint::Finished => "[코스 완주] 결승 체크포인트 통과".to_string(),
            _ => AWAIT_TRANSITION_LINE.to_string(),
        }
    }

    /// The (at most two) lines after `<|im_start|>user\n`.
    pub fn to_prompt_lines(&self) -> Vec<String> {
        let progress = format!("{}/{}", self.passed.min(self.length), self.length);
        let splits = self.split_summary();
        let first = if self.facts_only() {
            let mut parts = vec![format!("{progress} 구간")];
            match self.sync {
                CourseSync::Detouring => parts.push("고스트 경로 밖".into()),
                CourseSync::Rejoined => parts.push("본선 재합류".into()),
                CourseSync::OnTrack => {}
            }
            parts.extend(splits);
            format!("[주행 기록-판정전] {} (표본 부족: 좋다·나쁘다 단정 말고 사실만)", parts.join(" · "))
        } else if let Some(saved) = self.habit_upgrade_saved_ms.filter(|ms| i64::from(*ms) >= SPLIT_DEADBAND_MS) {
            format!(
                "[코스 분기 알림 {progress}] 평소 루트보다 평균 {:.1}초 빠른 검증 지름길 있음",
                f64::from(saved) / 1000.0
            )
        } else {
            let (head, what) = match self.sync {
                CourseSync::OnTrack => (format!("[코스 진행 {progress}]"), None),
                CourseSync::Detouring => (format!("[코스 우회 중 {progress}]"), Some("고스트 경로 밖")),
                CourseSync::Rejoined => (format!("[코스 재합류 {progress}]"), Some("우회 후 본선 합류")),
            };
            let body: Vec<String> = what.map(str::to_string).into_iter().chain(splits).collect();
            if body.is_empty() {
                head
            } else {
                format!("{head} {}", body.join(" · "))
            }
        };
        vec![first, self.next_step_line()]
    }
}

/// The Lv3 proactive friction tip (two lines after `<|im_start|>user\n`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrictionBrief {
    /// Scrubbed label of the stuck element ("쿠폰 적용 버튼").
    pub bottleneck_label: Option<String>,
    /// "반복 클릭 무응답" | "입력 재시도 반복" | "진행 경로 탐색 지연" | ...
    pub cause_summary: String,
    /// A verified way around ("#1 '간편결제' (유저 검증 지름길)").
    pub verified_alternative: Option<String>,
    /// A confirmed trap to warn about.
    pub trap_warning: Option<String>,
    pub delivery: CoachDeliveryMode,
}

impl FrictionBrief {
    /// `"immediate_tip"` | `"post_defeat_comfort"` | `"after_input_aborted"` | `"after_speech_cleared"`.
    pub fn delivery_tone(&self) -> &'static str {
        self.delivery.tone_tag()
    }

    pub fn to_prompt_lines(&self) -> Vec<String> {
        let clean = |s: &Option<String>| s.as_deref().map(prompt_label).filter(|l| !l.is_empty());
        let cause = prompt_label(&self.cause_summary);
        let cause = if cause.is_empty() { "진행이 막힘".to_string() } else { cause };
        let first = match clean(&self.bottleneck_label) {
            Some(l) => format!("[막힌 곳] '{l}' — {cause}"),
            None => format!("[막힌 곳] {cause}"),
        };
        let (alt, trap) = (clean(&self.verified_alternative), clean(&self.trap_warning));
        let mut second = match (&alt, &trap) {
            (Some(a), Some(t)) => format!("[우회 제안] {a} · '{t}' 피하기"),
            (Some(a), None) => format!("[우회 제안] {a}"),
            (None, Some(t)) => format!("[주의] '{t}'에서 다들 막힘"),
            (None, None) => "[우회 제안] 검증된 대안 없음 — 조작법을 지어내지 말고 공감만".to_string(),
        };
        second.push_str(&format!(" / 말투: {}", self.delivery.tone_hint()));
        vec![first, second]
    }
}

const COMMIT_WORDS: [&str; 20] = [
    "결제하기",
    "결제 하기",
    "주문완료",
    "주문 완료",
    "지금 결제",
    "바로 결제",
    "구매 확정",
    "구매확정",
    "주문 확정",
    "주문확정",
    "결제 완료",
    "결제완료",
    "pay now",
    "place order",
    "place your order",
    "complete purchase",
    "confirm and pay",
    "注文を確定する",
    "注文確定",
    "購入を確定",
];

const UNAVAILABLE_WORDS: [&str; 15] = [
    "품절",
    "일시품절",
    "매진",
    "판매종료",
    "판매 종료",
    "종료",
    "점검중",
    "점검 중",
    "비활성",
    "sold out",
    "out of stock",
    "unavailable",
    "売り切れ",
    "在庫なし",
    "メンテナンス中",
];

/// A final money-moving button ("결제하기", "Pay Now", "Place Order", "注文を確定する"); steps that only lead
/// toward checkout ("장바구니 담기", "Proceed to Checkout", "Buy Now") are not.
pub fn is_commit_label(label: &str) -> bool {
    contains_any(label, &COMMIT_WORDS)
}

/// A sold-out / closed / under-maintenance label (Korean, English, Japanese).
pub fn is_unavailable_label(label: &str) -> bool {
    contains_any(label, &UNAVAILABLE_WORDS)
}

/// `label` without its status words ("간편결제 (일시품절)" → "간편결제").
fn without_status_words(label: &str) -> String {
    let mut words = UNAVAILABLE_WORDS;
    words.sort_by_key(|w| std::cmp::Reverse(w.len()));
    let mut s = label.to_string();
    for w in words {
        while let Some(i) = s.to_ascii_lowercase().find(w) {
            s.replace_range(i..i + w.len(), " ");
        }
    }
    let mut s = collapse_ws(&s);
    for pair in ["( )", "()", "[ ]", "[]"] {
        s = s.replace(pair, " ");
    }
    collapse_ws(&s)
        .trim_matches(|c: char| matches!(c, '-' | '·' | '/' | ','))
        .trim()
        .to_string()
}

/// One numbered button of a shopping page as the overlay sees it (pd_nav_number order).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WebChoice {
    pub number: u8,
    pub label: String,
    /// GoldenPath / Premium for this page (local analysis or the server).
    pub verified: bool,
    /// UIA `IsEnabled == false` or `aria-disabled="true"`.
    pub disabled: bool,
}

impl WebChoice {
    pub fn unavailable(&self) -> bool {
        self.disabled || is_unavailable_label(&self.label)
    }
}

/// What a number hotkey / voice pick may do with a badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotDispatch {
    /// Click it (UIA `InvokePattern::Invoke`).
    Invoke,
    /// CommitGuard: move focus / the cursor there only, never click a final payment button.
    FocusOnly,
}

/// Shopping-page coaching for the consumer, after `<|im_start|>user\n`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct WebCoachBrief {
    /// Normalised site ("amazon.com").
    pub host: String,
    /// Scrubbed page label ("결제 단계").
    pub page_label: Option<String>,
    /// The step to suggest: the first available, non-final button.
    pub primary: Option<WebChoice>,
    /// A verified first choice that is sold out / disabled right now (the suggestion moved on to the next path).
    pub runtime_blocked_primary: Option<String>,
    /// A confirmed FrictionTrap on this page.
    pub friction_trap: Option<String>,
    /// CommitGuard: badges of final-payment buttons, never clicked for the player.
    pub commit_numbers: Vec<u8>,
}

impl WebCoachBrief {
    /// From the page's numbered buttons: final-payment buttons go to the CommitGuard, the first available other
    /// button is suggested, and a verified button that is unavailable becomes `runtime_blocked_primary`.
    pub fn from_choices(host: &str, page_label: Option<&str>, choices: &[WebChoice], friction_trap: Option<&str>) -> Self {
        let mut sorted: Vec<&WebChoice> = choices.iter().filter(|c| (1..=9).contains(&c.number)).collect();
        sorted.sort_by_key(|c| c.number);
        let commit_numbers: Vec<u8> = sorted.iter().filter(|c| is_commit_label(&c.label)).map(|c| c.number).collect();
        let primary = sorted
            .iter()
            .find(|c| !is_commit_label(&c.label) && !c.unavailable())
            .map(|c| (*c).clone());
        let runtime_blocked_primary = sorted
            .iter()
            .find(|c| c.verified && c.unavailable() && !is_commit_label(&c.label))
            .filter(|c| primary.as_ref().is_none_or(|p| p.number != c.number))
            .map(|c| prompt_label(&without_status_words(&c.label)))
            .filter(|l| !l.is_empty());
        Self {
            host: host.trim().to_lowercase(),
            page_label: page_label.map(prompt_label).filter(|l| !l.is_empty()),
            primary,
            runtime_blocked_primary,
            friction_trap: friction_trap.map(prompt_label).filter(|l| !l.is_empty()),
            commit_numbers,
        }
    }

    /// A final-payment step is on screen: confirm-first wording, no clicking for the player.
    pub fn commit_safety_guard(&self) -> bool {
        !self.commit_numbers.is_empty()
    }

    /// What the hotkey / voice dispatcher may do with badge `number`.
    pub fn dispatch(&self, number: u8) -> SlotDispatch {
        if self.commit_numbers.contains(&number) {
            SlotDispatch::FocusOnly
        } else {
            SlotDispatch::Invoke
        }
    }

    /// At most three lines.
    pub fn to_prompt_lines(&self) -> Vec<String> {
        let judged = self.primary.as_ref().is_some_and(|p| p.verified);
        let mut head = if judged {
            "[웹 코칭] ".to_string()
        } else {
            "[웹 코칭-판정전] ".to_string()
        };
        head.push_str(&prompt_label(&self.host));
        if let Some(p) = &self.page_label {
            head.push_str(&format!(" · {p}"));
        }
        match &self.primary {
            Some(p) if p.verified => head.push_str(&format!(
                " — 다음: #{} '{}' (다른 사용자 검증 경로)",
                p.number,
                prompt_label(&p.label)
            )),
            Some(p) => head.push_str(&format!(
                " — 화면 #{} '{}' (사실만, 좋다·나쁘다 단정 말 것)",
                p.number,
                drop_verdict_words(&prompt_label(&p.label))
            )),
            None => {}
        }
        if let Some(b) = &self.runtime_blocked_primary {
            head.push_str(&format!(
                " / '{b}'{} 지금 품절·비활성 — 다음 경로로 안내",
                if has_batchim(b) { "은" } else { "는" }
            ));
        }
        let mut lines = vec![head];
        if let Some(t) = &self.friction_trap {
            lines.push(format!("[주의] '{t}'에서 막히는 사람이 많음"));
        }
        if self.commit_safety_guard() {
            let nums: Vec<String> = self.commit_numbers.iter().map(|n| format!("#{n}")).collect();
            lines.push(format!(
                "[결제 확인] {} 최종 결제 버튼 — 대신 누르거나 재촉하지 말고, 금액·수량·배송지를 먼저 확인하자고 안내",
                josa(&nums.join(", "), "은", "는")
            ));
        }
        lines
    }
}

/// Words that assert a sharp drop; never said on an insufficient sample.
pub const ASSERTIVE_DROP_WORDS: [&str; 9] = ["급락", "폭락", "급감", "곤두박질", "추락", "반토막", "plunge", "crash", "collapse"];

/// A site operator's funnel report, after `<|im_start|>user\n`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AnalyticBrief {
    pub host: String,
    /// "this_week (+37.0h) vs last_week (+37.0h)".
    pub window_summary: String,
    /// Both windows have at least [`MIN_FUNNEL_SESSIONS`] entering sessions.
    pub confidence_sufficient: bool,
    pub sessions: u32,
    /// Only printed when the sample is sufficient ("14.2% → 9.8% (-4.4%p)").
    pub conversion_delta: Option<String>,
    pub top_friction_trap: Option<String>,
    pub top_golden_path: Option<String>,
}

impl AnalyticBrief {
    /// The sample guard: sufficient only when the current window and the comparison window (if any) both
    /// reached [`MIN_FUNNEL_SESSIONS`].
    pub fn sample_sufficient(sessions: u32, prev_sessions: Option<u32>) -> bool {
        sessions >= MIN_FUNNEL_SESSIONS && prev_sessions.is_none_or(|p| p >= MIN_FUNNEL_SESSIONS)
    }

    pub fn to_prompt_lines(&self) -> Vec<String> {
        let host = prompt_label(&self.host);
        let window = clip_chars(&strip_hex_hashes(&self.window_summary), 2 * LABEL_MAX_CHARS);
        if !self.confidence_sufficient {
            return vec![
                format!(
                    "[분석-표본부족] {host} · {window} · 세션 {} (최소 {MIN_FUNNEL_SESSIONS})",
                    self.sessions
                ),
                format!("[답변] 증감·원인을 단정하지 말고 표본이 적다고만 말할 것, {ANALYTIC_ANSWER_MAX_CHARS}자 이내"),
            ];
        }
        let mut first = format!("[분석] {host} · {window} · 세션 {}", self.sessions);
        if let Some(d) = self.conversion_delta.as_deref().map(prompt_label).filter(|d| !d.is_empty()) {
            first.push_str(&format!(" · 전환율 {d}"));
        }
        let mut second = Vec::new();
        if let Some(t) = self.top_friction_trap.as_deref().map(prompt_label).filter(|t| !t.is_empty()) {
            second.push(format!("[병목] '{t}'"));
        }
        if let Some(g) = self.top_golden_path.as_deref().map(prompt_label).filter(|g| !g.is_empty()) {
            second.push(format!("[우수 경로] '{g}'"));
        }
        second.push(format!("답은 {ANALYTIC_ANSWER_MAX_CHARS}자 이내"));
        vec![first, second.join(" · ")]
    }

    /// The model's answer made safe to show: on an insufficient sample an assertive drop is replaced by a
    /// neutral line; always at most [`ANALYTIC_ANSWER_MAX_CHARS`] characters.
    pub fn sanitize_answer(&self, answer: &str) -> String {
        let answer = answer.trim();
        let text = if !self.confidence_sufficient && contains_any(answer, &ASSERTIVE_DROP_WORDS) {
            format!(
                "아직 세션이 {}건뿐이라 변화를 단정하기 어려워요. 조금 더 모아서 볼게요.",
                self.sessions
            )
        } else {
            answer.to_string()
        };
        crate::tools::truncate_speech(&text, ANALYTIC_ANSWER_MAX_CHARS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::{build_fast_path_prompt, SessionContext, Situation, USER_TURN};

    fn premium() -> CourseBrief {
        CourseBrief {
            passed: 2,
            length: 4,
            vs_best_ms: Some(-1_800),
            vs_public_ms: Some(-3_200),
            next_hint: NextStepHint::NumberedOnScreen {
                number: 1,
                clean_label: "우측 사다리".into(),
                direction_label: "오른쪽 위".into(),
            },
            verdict_judged_premium: true,
            ..Default::default()
        }
    }

    #[test]
    fn splits_inside_one_second_are_the_same_pace() {
        assert_eq!(split_phrase("내 최고", 999), "내 최고와 동등");
        assert_eq!(split_phrase("타인 평균", -300), "타인 평균과 동등");
        assert_eq!(split_phrase("내 최고", -1_000), "내 최고보다 1.0초 빠름");
        assert_eq!(split_phrase("타인 최고", 3_100), "타인 최고보다 3.1초 느림");
    }

    #[test]
    fn a_premium_course_brief_is_two_lines_with_at_most_two_splits() {
        let lines = premium().to_prompt_lines();
        assert_eq!(lines.len(), CourseBrief::MAX_LINES);
        assert_eq!(lines[0], "[코스 진행 2/4] 내 최고보다 1.8초 빠름 · 타인 평균보다 3.2초 빠름");
        assert_eq!(lines[1], "[다음 추천 스텝] #1 '우측 사다리' (오른쪽 위)");
        let habit = CourseBrief {
            habit_upgrade_saved_ms: Some(8_000),
            ..premium()
        };
        assert_eq!(
            habit.to_prompt_lines()[0],
            "[코스 분기 알림 2/4] 평소 루트보다 평균 8.0초 빠른 검증 지름길 있음"
        );
        let rejoined = CourseBrief {
            passed: 3,
            sync: CourseSync::Rejoined,
            vs_public_ms: None,
            vs_best_ms: Some(2_500),
            ..premium()
        };
        assert_eq!(
            rejoined.to_prompt_lines()[0],
            "[코스 재합류 3/4] 우회 후 본선 합류 · 내 최고보다 2.5초 느림"
        );
        assert_eq!(CourseSync::Rejoined.tag(), "rejoined");
        let over = CourseBrief {
            passed: 9,
            vs_best_ms: Some(400),
            vs_public_ms: None,
            ..premium()
        };
        assert_eq!(
            over.to_prompt_lines()[0],
            "[코스 진행 4/4] 내 최고와 동등",
            "progress is clamped, jitter is the same pace"
        );
        let done = CourseBrief {
            next_hint: NextStepHint::Finished,
            ..premium()
        };
        assert_eq!(done.to_prompt_lines()[1], "[코스 완주] 결승 체크포인트 통과");
        let gesture = CourseBrief {
            next_hint: NextStepHint::GesturePrompt {
                action_label: "아래로 스크롤해 다음 구간으로".into(),
            },
            ..premium()
        };
        assert_eq!(gesture.to_prompt_lines()[1], "[다음 추천 스텝] 아래로 스크롤해 다음 구간으로");
    }

    #[test]
    fn cn11_unrated_brief_hides_hashes_and_recommendations() {
        let next_hash = format!("0xabc{}", &"0123456789".repeat(4)[..37]);
        assert_eq!(next_hash.len(), 42);
        let brief = CourseBrief {
            passed: 1,
            length: 3,
            vs_public_ms: Some(400),
            next_hint: NextStepHint::AwaitScreenTransition,
            verdict_judged_premium: false, // Tier::Unrated, decided = 2 < 3
            ..Default::default()
        };
        let p = build_fast_path_prompt(&Situation {
            session: Some(SessionContext {
                course_brief: Some(brief.clone()),
                ..Default::default()
            }),
            ..Default::default()
        });
        assert!(!p.contains("0xabc"), "no raw hash anywhere");
        assert!(p.contains("[주행 기록-판정전] 1/3 구간 · 타인 평균과 동등"), "{p}");
        assert!(p.contains("[다음 스텝] 다음 화면 전환 대기"));
        assert!(!p.contains("추천"), "no recommendation wording in the whole prompt");
        assert!(brief.to_prompt_lines().len() <= 2);
        // A hash that slipped into a label is removed too, and a label's "추천" is dropped on facts only.
        let leaky = CourseBrief {
            next_hint: NextStepHint::NumberedOnScreen {
                number: 3,
                clean_label: format!("중앙 통로 {next_hash} 추천"),
                direction_label: "위".into(),
            },
            ..brief
        };
        let line = &leaky.to_prompt_lines()[1];
        assert_eq!(line, "[다음 관찰 스텝] #3 '중앙 통로' (위)");
        let bad_number = CourseBrief {
            next_hint: NextStepHint::NumberedOnScreen {
                number: 12,
                clean_label: "x".into(),
                direction_label: String::new(),
            },
            ..premium()
        };
        assert!(bad_number.to_prompt_lines()[1].starts_with("[다음 스텝] 다음 화면 전환 대기"));
    }

    #[test]
    fn friction_brief_lines_carry_the_tone() {
        let b = FrictionBrief {
            bottleneck_label: Some("쿠폰 적용 버튼".into()),
            cause_summary: "반복 클릭 무응답".into(),
            verified_alternative: None,
            trap_warning: None,
            delivery: CoachDeliveryMode::PostDefeatComfort,
        };
        let l = b.to_prompt_lines();
        assert_eq!(l[0], "[막힌 곳] '쿠폰 적용 버튼' — 반복 클릭 무응답");
        assert!(
            l[1].starts_with("[우회 제안] 검증된 대안 없음") && l[1].contains("먼저 위로"),
            "{}",
            l[1]
        );
        assert_eq!(b.delivery_tone(), "post_defeat_comfort");
    }

    #[test]
    fn wa10_sold_out_first_choice_moves_on_and_the_payment_button_is_guarded() {
        let choices = [
            WebChoice {
                number: 1,
                label: "간편결제 (일시품절)".into(),
                verified: true,
                disabled: false,
            },
            WebChoice {
                number: 2,
                label: "일반결제".into(),
                verified: true,
                disabled: false,
            },
            WebChoice {
                number: 3,
                label: "결제하기".into(),
                verified: false,
                disabled: false,
            },
        ];
        let b = WebCoachBrief::from_choices("Shop.Example.com", Some("결제 단계"), &choices, Some("쿠폰 적용"));
        assert_eq!(
            b.primary.as_ref().map(|p| p.label.as_str()),
            Some("일반결제"),
            "the next path inherits #1's role"
        );
        assert_eq!(
            b.runtime_blocked_primary.as_deref(),
            Some("간편결제"),
            "the status words are not part of the name"
        );
        assert_eq!(b.commit_numbers, vec![3]);
        assert!(b.commit_safety_guard());
        assert_eq!(b.dispatch(3), SlotDispatch::FocusOnly, "never auto-click the final payment button");
        assert_eq!(b.dispatch(2), SlotDispatch::Invoke);
        let lines = b.to_prompt_lines();
        assert!(lines.len() <= 3);
        assert!(
            lines[0].starts_with("[웹 코칭] shop.example.com · 결제 단계 — 다음: #2 '일반결제'"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("품절·비활성 — 다음 경로"));
        let guard = lines.iter().find(|l| l.starts_with("[결제 확인]")).expect("confirm-first line");
        assert!(
            guard.contains("#3") && guard.contains("확인") && guard.contains("누르거나 재촉하지 말고"),
            "{guard}"
        );
        assert!(!lines
            .iter()
            .any(|l| l.contains("#3") && (l.contains("눌러") || l.contains("클릭해"))));
        // A physically disabled first choice behaves the same.
        let disabled = [
            WebChoice {
                number: 1,
                label: "간편결제".into(),
                verified: true,
                disabled: true,
            },
            choices[1].clone(),
        ];
        let d = WebCoachBrief::from_choices("shop.example.com", None, &disabled, None);
        assert_eq!(d.runtime_blocked_primary.as_deref(), Some("간편결제"));
        assert!(!d.commit_safety_guard());
        // Inside the prompt, after the user turn only.
        let bare = build_fast_path_prompt(&Situation::default());
        let p = build_fast_path_prompt(&Situation {
            session: Some(SessionContext {
                web_coach: Some(b),
                ..Default::default()
            }),
            ..Default::default()
        });
        let cut = |p: &str| p[..p.find(USER_TURN).unwrap() + USER_TURN.len()].to_string();
        assert_eq!(cut(&p), cut(&bare));
    }

    #[test]
    fn commit_and_unavailable_labels() {
        for l in ["결제하기", "Pay Now", "Place Order", "注文を確定する", "주문 완료"] {
            assert!(is_commit_label(l), "{l}");
        }
        for l in [
            "장바구니 담기",
            "Add to Cart",
            "Proceed to Checkout",
            "Buy Now",
            "간편결제",
            "일반결제",
        ] {
            assert!(!is_commit_label(l), "{l}");
        }
        for l in ["Sold Out", "out of stock", "매진", "売り切れ", "점검중"] {
            assert!(is_unavailable_label(l), "{l}");
        }
        assert!(!is_unavailable_label("지금 구매"));
        let unjudged = WebCoachBrief::from_choices(
            "a.com",
            None,
            &[WebChoice {
                number: 1,
                label: "추천 상품 보기".into(),
                ..Default::default()
            }],
            None,
        );
        let line = &unjudged.to_prompt_lines()[0];
        assert!(
            line.starts_with("[웹 코칭-판정전] a.com — 화면 #1 '상품 보기'") && !line.contains("추천"),
            "{line}"
        );
    }

    #[test]
    fn wa11_insufficient_sample_never_asserts_a_drop_and_answers_fit_90_chars() {
        let thin = AnalyticBrief {
            host: "shop.example.com".into(),
            window_summary: "this_week (+37.0h) vs last_week (+37.0h)".into(),
            confidence_sufficient: AnalyticBrief::sample_sufficient(4, Some(30)),
            sessions: 4,
            conversion_delta: Some("14.2% → 9.8% (-4.4%p)".into()),
            top_friction_trap: Some("쿠폰 적용".into()),
            top_golden_path: None,
        };
        assert!(!thin.confidence_sufficient);
        let lines = thin.to_prompt_lines();
        let joined = lines.join("\n");
        assert!(
            joined.starts_with("[분석-표본부족] shop.example.com · this_week (+37.0h) vs last_week (+37.0h) · 세션 4 (최소 15)"),
            "{joined}"
        );
        assert!(
            !joined.contains("9.8%") && !joined.contains("쿠폰"),
            "no delta, no ranking on 4 sessions"
        );
        assert!(ASSERTIVE_DROP_WORDS.iter().all(|w| !joined.contains(w)));
        let safe = thin.sanitize_answer("이번 주 전환율이 급락했어요! 쿠폰 칸 때문이에요.");
        assert!(ASSERTIVE_DROP_WORDS.iter().all(|w| !safe.contains(w)), "{safe}");
        assert!(safe.contains("4건"));
        assert!(safe.chars().count() <= ANALYTIC_ANSWER_MAX_CHARS);
        let long = "가".repeat(200);
        assert!(thin.sanitize_answer(&long).chars().count() <= ANALYTIC_ANSWER_MAX_CHARS);

        let thick = AnalyticBrief {
            confidence_sufficient: AnalyticBrief::sample_sufficient(120, Some(140)),
            sessions: 120,
            top_golden_path: Some("간편결제".into()),
            ..thin.clone()
        };
        let l = thick.to_prompt_lines();
        assert_eq!(
            l[0],
            "[분석] shop.example.com · this_week (+37.0h) vs last_week (+37.0h) · 세션 120 · 전환율 14.2% → 9.8% (-4.4%p)"
        );
        assert_eq!(l[1], "[병목] '쿠폰 적용' · [우수 경로] '간편결제' · 답은 90자 이내");
        assert_eq!(
            thick.sanitize_answer("전환율이 급락했어요."),
            "전환율이 급락했어요.",
            "a sufficient sample may say so"
        );
        assert!(!AnalyticBrief::sample_sufficient(20, Some(10)), "the comparison window counts too");
    }
}
