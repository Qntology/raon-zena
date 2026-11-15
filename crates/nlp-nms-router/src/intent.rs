//! Intent scoring with bias/prejudice phrase banks.
//!
//! Port of logis-center's field scoring: `bias` = max-pool cosine against the intent's phrases,
//! `prej` = max-pool cosine against phrases that look similar but mean something else, and
//! `score = max(bias − prej·w, 0)` with `w = 0.3` for ≤ 2-word spans and `0.7` otherwise. An
//! exact keyword hit short-circuits to a high score (logis-center "exact match before vectors").

use crate::embed::{cosine, TextEmbedder};
use crate::normalize::normalize;
use serde::Serialize;
use zena_core::tools::{SystemCommand, WindowLayerMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, PartialOrd, Ord)]
pub enum Intent {
    ExitApp,
    HideAvatar,
    ShowAvatar,
    TopmostOn,
    TopmostOff,
    InteractiveMode,
    GhostMode,
    CoachQuestion,
    InventoryQuery,
    SmallTalk,
}

impl Intent {
    pub fn system_command(self) -> Option<(SystemCommand, Option<WindowLayerMode>)> {
        Some(match self {
            Intent::ExitApp => (SystemCommand::ExitApp, None),
            Intent::HideAvatar => (SystemCommand::HideAvatar, None),
            Intent::ShowAvatar => (SystemCommand::ShowAvatar, None),
            Intent::TopmostOn => (SystemCommand::SetTopmostOn, None),
            Intent::TopmostOff => (SystemCommand::SetTopmostOff, None),
            Intent::InteractiveMode => (SystemCommand::KeepActive, Some(WindowLayerMode::InteractivePet)),
            Intent::GhostMode => (SystemCommand::KeepActive, Some(WindowLayerMode::GhostClickThrough)),
            _ => return None,
        })
    }
    pub fn is_command(self) -> bool {
        self.system_command().is_some()
    }
}

struct Entry {
    intent: Intent,
    bias: Vec<Vec<f32>>,
    prej: Vec<Vec<f32>>,
    keywords: &'static [&'static str],
}

/// `(intent, bias phrases, prejudice phrases, exact keywords)`.
type Spec = (Intent, &'static [&'static str], &'static [&'static str], &'static [&'static str]);

pub const DEFAULT_BANK: &[Spec] = &[
    (
        Intent::ExitApp,
        &[
            "종료해 줘",
            "앱 꺼 줘",
            "프로그램 종료",
            "오늘 그만 켜둘게",
            "이제 꺼도 돼",
            "quit the app",
            "exit",
            "close zena",
        ],
        &["숨어 있어", "게임 종료됐어", "창 뒤로 가"],
        &["종료해", "종료해줘", "꺼줘", "꺼 줘", "앱 꺼", "quit", "exit app"],
    ),
    (
        Intent::HideAvatar,
        &["잠깐 숨어 있어", "모습 숨겨 줘", "안 보이게 해 줘", "잠깐 사라져", "hide yourself"],
        &["종료해 줘", "다시 나와"],
        &["숨어있어", "숨어 있어", "숨겨줘", "모습만 숨겨", "hide"],
    ),
    (
        Intent::ShowAvatar,
        &["다시 나와", "다시 보여 줘", "돌아와", "모습 보여 줘", "show yourself"],
        &["맨 앞으로 나와", "숨어 있어"],
        &["다시 나와", "돌아와"],
    ),
    (
        Intent::TopmostOn,
        &["화면 맨 앞으로 나와", "항상 위에 떠 있어", "제일 앞에 있어", "always on top"],
        &["다른 창 뒤로 가", "다시 나와"],
        &["맨 앞으로", "항상 위"],
    ),
    (
        Intent::TopmostOff,
        &["다른 창 뒤로 가 있어", "창 뒤로 가", "뒤에 가 있어", "go behind other windows"],
        &["화면 맨 앞으로 나와", "숨어 있어"],
        &["뒤로 가 있어", "창 뒤로"],
    ),
    (
        Intent::InteractiveMode,
        &["위치 옮길게", "너 좀 움직일게", "드래그할게", "let me move you"],
        &["클릭 통과", "전투 모드"],
        &["위치 옮길", "드래그"],
    ),
    (
        Intent::GhostMode,
        &["클릭 통과시켜 줘", "전투 모드로 해", "클릭 안 걸리게", "click through"],
        &["위치 옮길게"],
        &["클릭 통과", "전투 모드"],
    ),
    (
        Intent::CoachQuestion,
        &[
            "보스 패턴 어떻게 피해",
            "이거 어떻게 잡아",
            "공략 알려 줘",
            "지금 뭐 해야 돼",
            "어디로 가야 돼",
            "how do i dodge this",
            "any tips",
        ],
        &["인벤토리에서 물약 어디 있어"],
        &["어떻게 피해", "공략", "어떻게 잡아"],
    ),
    (
        Intent::InventoryQuery,
        &[
            "인벤토리에서 물약 어디 있어",
            "가방에 아이템 어디 있어",
            "포션 몇 개 남았어",
            "where is my potion",
        ],
        &["보스 패턴 어떻게 피해"],
        &["인벤토리", "물약", "포션", "가방"],
    ),
];

pub struct IntentBank<E: TextEmbedder> {
    entries: Vec<Entry>,
    pub embedder: E,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct IntentScore {
    pub intent: Intent,
    pub score: f32,
    pub bias: f32,
    pub prej: f32,
    pub exact: bool,
}

impl<E: TextEmbedder> IntentBank<E> {
    pub fn new(embedder: E, spec: &[Spec]) -> Self {
        let entries = spec
            .iter()
            .map(|(intent, bias, prej, keywords)| Entry {
                intent: *intent,
                bias: bias.iter().map(|p| embedder.embed(p)).collect(),
                prej: prej.iter().map(|p| embedder.embed(p)).collect(),
                keywords,
            })
            .collect();
        Self { entries, embedder }
    }

    /// All intents scored for `text`, best first.
    pub fn score(&self, text: &str) -> Vec<IntentScore> {
        let n = normalize(text);
        let wc = n.words.len();
        let v = self.embedder.embed(text);
        let joined = n.text.clone();
        let compact: String = joined.split_whitespace().collect();
        let pw = if wc <= 2 { 0.3 } else { 0.7 };
        let mut out: Vec<IntentScore> = self
            .entries
            .iter()
            .map(|e| {
                let bias = e.bias.iter().map(|b| cosine(&v, b)).fold(0f32, f32::max);
                let prej = e.prej.iter().map(|p| cosine(&v, p)).fold(0f32, f32::max);
                let exact = e.keywords.iter().any(|k| {
                    let kc: String = k.split_whitespace().collect();
                    joined.contains(k) || compact.contains(&kc)
                });
                let mut score = (bias - prej * pw).max(0.0);
                if exact {
                    score = score.max(0.95);
                }
                IntentScore {
                    intent: e.intent,
                    score,
                    bias,
                    prej,
                    exact,
                }
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashedNgramEmbedder;

    fn bank() -> IntentBank<HashedNgramEmbedder> {
        IntentBank::new(HashedNgramEmbedder::default(), DEFAULT_BANK)
    }

    #[test]
    fn prd_utterances_map_to_intents() {
        let b = bank();
        let top = |t: &str| b.score(t)[0].intent;
        assert_eq!(top("제나야 오늘 그만 켜둘게, 종료해 줘"), Intent::ExitApp);
        assert_eq!(top("잠깐 숨어있어"), Intent::HideAvatar);
        assert_eq!(top("잠깐 모습만 숨겨줘"), Intent::HideAvatar);
        assert_eq!(top("화면 맨 앞으로 나와"), Intent::TopmostOn);
        assert_eq!(top("다른 창 뒤로 가 있어"), Intent::TopmostOff);
        assert_eq!(top("방금 그 보스 패턴 어떻게 피해?"), Intent::CoachQuestion);
        assert_eq!(top("내 인벤토리에서 물약 어디 있어?"), Intent::InventoryQuery);
    }
}
