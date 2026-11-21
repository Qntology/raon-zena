//! A spoken recommendation becomes an observation. The rules are deterministic on purpose: the observation
//! comes from the dictionary match and Stanza's negation scope (NLP-NMS v2), never from the model, so a
//! hallucinated "vote" cannot pollute a verdict.
//!
//! Unlike a command (NLP-NMS holds a negated command instead of running it), negated feedback flips its sign:
//!
//! | utterance | intent | negated | observation |
//! |---|---|---|---|
//! | "그 공략 추천" | recommend | no | +1 |
//! | "그거 추천 안 해" | recommend | yes | −1 |
//! | "별로였어" | not_recommend | no | −1 |
//! | "별로지 않았어" · "나쁘지 않았어" | not_recommend | yes | +1, half weight |
//!
//! [`extract_pragmatic_feedback`] reads a whole utterance the same way, with four deterministic steps on top
//! (Scenario 4 Part 1, `feedback_credit_v1` FC-01 … FC-04):
//!
//! 1. **Modality gate** — a request ("다른 공략 추천해 **줘**"), a question ("이거 추천이야**?**") or a hypothetical
//!    ("좋**으면** 추천할게") is not feedback: no vote.
//! 2. **Adversative split** — "-는데 · -지만 · -어도 · 근데 · but · けど" cut the utterance into concessions and the
//!    main clause; when both grade the target, the main clause wins ("딜은 쎈데 \[rejected] 스태미나가 너무 딸려서
//!    비추 \[kept]").
//! 3. **Blame shield and sarcasm** — a main clause that blames the player's own hands ("내 손이 꼬였어") grades
//!    nothing and shields the defeat just before it; after a defeat or a friction burst, exaggeration plus a damage
//!    verb ("참 좋~은 길이다 죽기 딱 좋네") cancels every positive vote.
//! 4. **Retraction** — "아까 말 취소" turns the previous feedback (5 minutes at most) into a withdrawal *in place*
//!    ([`apply_feedback_to_ledger`]): PD-Verdict skips `sign` 0, so the old vote leaves `E` and `v` entirely.
//!
//! The vocabulary is [`FeedbackLexicon::builtin`] (Korean, English, Japanese); a dictionary's `feedback_rules`
//! section (`zena_lexicon::Lexicon::feedback_rules`) extends it through [`extract_pragmatic_feedback_with`].

use crate::{Obs, Params};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackIntent {
    /// `feedback.recommend` ("추천", "좋았어", "덕분에 깼다", "통했어")
    Recommend,
    /// `feedback.not_recommend` ("비추", "별로", "안 통해", "틀렸어")
    NotRecommend,
}

/// The observation for one piece of feedback about a record (`credit` from the target resolution, 0..1).
pub fn feedback_obs(intent: FeedbackIntent, negated: bool, credit: f64, at_ms: i64, params: &Params) -> Obs {
    let base = params.weight("feedback");
    let (sign, weight) = match (intent, negated) {
        (FeedbackIntent::Recommend, false) => (1, base),
        (FeedbackIntent::Recommend, true) => (-1, base),
        (FeedbackIntent::NotRecommend, false) => (-1, base),
        // A double negative ("not bad") is a weak recommendation.
        (FeedbackIntent::NotRecommend, true) => (1, base * 0.5),
    };
    Obs {
        kind: "feedback".into(),
        sign,
        weight,
        credit: credit.clamp(0.0, 1.0),
        at_ms,
    }
}

/// A manual ▲ / ▼ correction in the history view: the strongest local observation.
pub fn manual_obs(up: bool, at_ms: i64, params: &Params) -> Obs {
    Obs {
        kind: "manual".into(),
        sign: if up { 1 } else { -1 },
        weight: params.weight("manual"),
        credit: 1.0,
        at_ms,
    }
}

/// How far back a retraction ("아까 말 취소") reaches: 5 minutes.
pub const RETRACT_WINDOW_MS: i64 = 300_000;
/// How far back a self-blame ("내 손이 꼬였어") shields a defeat outcome: 15 seconds.
pub const DEFEAT_SHIELD_WINDOW_MS: i64 = 15_000;

/// The observation kind a spoken evaluation becomes (`Params::weight` gives its base weight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObsKind {
    /// An explicit recommendation / disapproval (weight 1.0).
    Feedback,
    /// Pure thanks or relief ("고마워!", "살았다!"; weight 0.6).
    Reaction,
}

impl ObsKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ObsKind::Feedback => "feedback",
            ObsKind::Reaction => "reaction",
        }
    }
}

/// What the session knows around an utterance (the screen events of the last seconds).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackContext {
    pub recent_defeat_within_15s: bool,
    pub recent_victory_within_15s: bool,
    /// A friction burst (rage clicks, retries) is in progress or just ended.
    pub high_friction_burst: bool,
}

/// One word as an analyzer saw it (NLP-NMS `Normalized` words with Stanza / rule roles; `nlp_nms_router::FeedbackWord`
/// maps onto it field by field). Optional: [`extract_pragmatic_feedback`] reads the raw text with its own rules and
/// takes these flags only as extra evidence, so the result is the same with or without a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeedbackToken<'a> {
    pub surface: &'a str,
    pub lemma: &'a str,
    /// A request auxiliary after a predicate: "줘", "줄래", "볼래", "please".
    pub is_auxiliary_request: bool,
    /// A negator: "안", "못", "-지 않-", "not".
    pub is_negation: bool,
    /// Inside a negation's scope.
    pub negated_in_clause: bool,
    /// Stanza `nsubj` / `dislocated`, or a noun with "-은 / -는 / -이 / -가".
    pub is_subject_or_topic: bool,
}

/// What one utterance says about the recommendation it follows (no model involved).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractedFeedback {
    /// `(kind, sign ±1, strength 0.5 | 1.0)`; `None` when the utterance only shields, retracts or was sarcastic.
    pub evaluation: Option<(ObsKind, i32, f64)>,
    /// Turn the last feedback of the last 5 minutes on the same target into a withdrawal (`sign` 0) in place.
    pub retracts_previous: bool,
    /// The player blamed their own hands right after a defeat: the defeat outcome (−1) must not grade the guide.
    pub shield_defeat_outcome: bool,
    /// An explicit pointer for the credit matrix ("1번" …): `"{n}번"` for a numbered slot, or a quoted name.
    pub target_mention: Option<String>,
    /// Positive votes cancelled as sarcasm (exaggeration + damage verb after a defeat or a friction burst).
    pub sarcasm_blocked: bool,
}

/// The pragmatic vocabulary, one phrase list per category. Phrases are lowercase; a phrase with a space matches
/// consecutive words (each a prefix of its word: "아까 말" matches "아까 말을"), Korean phrases match a word's start
/// (and, from two syllables, anywhere inside it: "개비추"), English ones a whole word with a short inflection,
/// Japanese ones anywhere. The JSON form is the dictionary's `feedback_rules` section (every key optional; a list is
/// an array of strings or one comma-separated string, as the dictionary's other phrase lists).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct FeedbackLexicon {
    /// Explicit recommendation (strength 1.0).
    #[serde(deserialize_with = "phrase_list")]
    pub recommend: Vec<String>,
    /// Explicit disapproval (strength 1.0).
    #[serde(deserialize_with = "phrase_list")]
    pub not_recommend: Vec<String>,
    /// A good attribute ("쎈데", "편해"; strength 0.5).
    #[serde(deserialize_with = "phrase_list")]
    pub positive_attribute: Vec<String>,
    /// A bad attribute ("딸려서", "막혀서"; strength 0.5).
    #[serde(deserialize_with = "phrase_list")]
    pub negative_attribute: Vec<String>,
    /// Thanks / relief ([`ObsKind::Reaction`]).
    #[serde(deserialize_with = "phrase_list")]
    pub reaction: Vec<String>,
    /// Request auxiliaries after the evaluated word ("줘", "줄래", "please").
    #[serde(deserialize_with = "phrase_list")]
    pub request: Vec<String>,
    /// Retraction markers ("취소", "아까 말", "scratch that").
    #[serde(deserialize_with = "phrase_list")]
    pub retraction: Vec<String>,
    /// Self-blame phrases ("내 실수", "my bad").
    #[serde(deserialize_with = "phrase_list")]
    pub self_blame: Vec<String>,
    /// Exaggeration ("참 좋", "기가 막히", "yeah right"); an elongated word ("좋~은") counts too.
    #[serde(deserialize_with = "phrase_list")]
    pub exaggeration: Vec<String>,
    /// Damage verbs for sarcasm ("죽", "막히", "stuck").
    #[serde(deserialize_with = "phrase_list")]
    pub damage: Vec<String>,
    /// Hedges that weaken an evaluation to 0.5 ("좀", "약간", "kinda").
    #[serde(deserialize_with = "phrase_list")]
    pub hedge: Vec<String>,
}

/// A phrase list as the dictionary writes it: an array of strings, or one comma-separated string.
fn phrase_list<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum List {
        One(String),
        Many(Vec<String>),
    }
    let raw = match List::deserialize(d)? {
        List::One(s) => s.split(',').map(str::to_string).collect(),
        List::Many(v) => v,
    };
    Ok(raw.into_iter().map(|p| p.trim().to_lowercase()).filter(|p| !p.is_empty()).collect())
}

const RECOMMEND: &[&str] = &[
    "추천",
    "강추",
    "최고",
    "굿",
    "꿀",
    "좋",
    "괜찮",
    "통했",
    "통하네",
    "통함",
    "통한다",
    "통하더",
    "먹혔",
    "먹히네",
    "먹힘",
    "먹힌다",
    "효과 있",
    "덕분",
    "쓸만",
    "쓸 만",
    "유용",
    "완벽",
    "recommend",
    "good",
    "great",
    "awesome",
    "useful",
    "worked",
    "works",
    "helpful",
    "perfect",
    "nice",
    "love",
    "おすすめ",
    "オススメ",
    "お勧め",
    "最高",
    "良",
    "よかった",
    "使える",
    "便利",
];
const NOT_RECOMMEND: &[&str] = &[
    "비추",
    "별로",
    "최악",
    "구려",
    "구리",
    "구림",
    "구렸",
    "쓰레기",
    "나쁘",
    "나빴",
    "나쁜",
    "노답",
    "비효율",
    "후회",
    "안 통",
    "안통",
    "소용없",
    "소용 없",
    "쓸모없",
    "쓸모 없",
    "쓸데없",
    "틀렸",
    "틀린",
    "틀려",
    "엉터리",
    "bad",
    "terrible",
    "awful",
    "useless",
    "worst",
    "sucks",
    "trash",
    "garbage",
    "wrong",
    "ダメ",
    "だめ",
    "駄目",
    "微妙",
    "最悪",
    "悪",
    "使えな",
];
const POSITIVE_ATTRIBUTE: &[&str] = &[
    "쎄", "쎈", "강하", "강력", "빠르", "빨라", "편하", "편해", "쉽", "쉬워", "안정", "strong", "fast", "easy", "強い", "楽",
];
const NEGATIVE_ATTRIBUTE: &[&str] = &[
    "딸려",
    "딸리",
    "딸림",
    "딸렸",
    "막혀",
    "막혔",
    "막히",
    "막힘",
    "느리",
    "느려",
    "약하",
    "약해",
    "어렵",
    "어려워",
    "힘들",
    "불편",
    "애매",
    "실패",
    "weak",
    "slow",
    "hard",
    "clunky",
    "遅い",
    "弱い",
    "難しい",
];
const REACTION: &[&str] = &[
    "고마워",
    "고맙",
    "감사",
    "땡큐",
    "살았다",
    "살았어",
    "살았네",
    "thanks",
    "thank",
    "thx",
    "ありがとう",
    "助かった",
];
const REQUEST: &[&str] = &[
    "줘",
    "줘요",
    "주세요",
    "줄래",
    "줄래요",
    "주라",
    "주실래요",
    "주시겠어요",
    "달라",
    "달라고",
    "봐",
    "봐줘",
    "볼래",
    "볼래요",
    "할까",
    "할까요",
    "할래",
    "할래요",
    "please",
];
const RETRACTION: &[&str] = &[
    "아까 말",
    "방금 말",
    "아까 추천",
    "방금 추천",
    "말 취소",
    "말 바꿀",
    "말을 바꿀",
    "취소",
    "정정",
    "무효",
    "철회",
    "take back",
    "take it back",
    "take that back",
    "scratch that",
    "never mind",
    "nevermind",
    "さっきのなし",
    "取り消し",
    "撤回",
    "やっぱなし",
];
const SELF_BLAME: &[&str] = &[
    "내 실수",
    "제 실수",
    "내가 못",
    "내가 잘못",
    "내 탓",
    "제 탓",
    "my bad",
    "my fault",
    "i choked",
    "my timing",
    "my mistake",
    "操作ミス",
    "私のミス",
    "自分のミス",
    "俺のミス",
];
const EXAGGERATION: &[&str] = &[
    "참 좋",
    "참 잘",
    "참 대단",
    "아주 잘",
    "아주 대단",
    "아주 좋",
    "기가 막히",
    "기가막히",
    "퍽이나",
    "yeah right",
    "great job",
    "nice job",
    "just great",
    "just perfect",
    "さすが",
];
const DAMAGE: &[&str] = &[
    "죽",
    "망하",
    "망했",
    "망함",
    "망해",
    "막히",
    "막혔",
    "막혀",
    "막힘",
    "터지",
    "터졌",
    "터져",
    "날아가",
    "날아갔",
    "날아감",
    "털렸",
    "털림",
    "끔살",
    "die",
    "died",
    "dead",
    "dying",
    "stuck",
    "wiped",
    "死",
    "詰ん",
];
const HEDGE: &[&str] = &[
    "좀",
    "조금",
    "약간",
    "살짝",
    "그럭저럭",
    "kinda",
    "somewhat",
    "ちょっと",
    "まあまあ",
];

// Grammar (closed classes; not dictionary vocabulary).
const NEG_ADVERBS: &[&str] = &["안", "못"];
const EN_NEGATORS: &[&str] = &["not", "never", "no", "cannot"];
const LONG_NEGATION: &[&str] = &["않", "못하", "못해", "못했", "못한", "못함"];
const FUSED_LONG_NEGATION: &[&str] = &["지않", "지못하", "지못해", "지못했", "지못한"];
/// A negating adverb fused to the light verb after a noun: "추천 못해", "추천 안함".
const FUSED_NEGATED_LIGHT_VERBS: &[&str] = &[
    "못하", "못해", "못했", "못한", "못함", "안하", "안해", "안했", "안한", "안함", "안돼", "안됐", "안되", "안된",
];
const COPULA_NEGATION: &[&str] = &["아니", "아냐", "아닌", "아님"];
const JA_NEGATION: &[&str] = &["ない", "なかった", "ません", "なく"];
const LIGHT_VERBS: &[&str] = &["하", "해", "했", "한", "할", "합", "되", "돼", "됐", "된", "될"];
const NOMINALIZERS: &[&str] = &["건", "거", "게", "것", "것은", "것도", "거는"];
const REQUEST_ENDINGS: &[&str] = &[
    "해줘",
    "해줘요",
    "해주세요",
    "해줄래",
    "해줄래요",
    "해달라",
    "해달라고",
    "해봐",
    "해볼래",
    "해주라",
    "줘",
    "줘요",
    "주세요",
    "줄래",
];
const JA_REQUEST: &[&str] = &["して", "ください", "教えて", "ちょうだい"];
const EN_REQUEST_OPENERS: &[&[&str]] = &[
    &["can", "you"],
    &["could", "you"],
    &["would", "you"],
    &["recommend", "me"],
    &["give", "me"],
];
const QUESTION_ENDINGS: &[&str] = &[
    "니",
    "냐",
    "나요",
    "까",
    "까요",
    "어때",
    "어때요",
    "는지",
    "인가",
    "인가요",
    "은가",
    "은가요",
    "을까",
    "을까요",
];
/// Question words (whole words: "몇 번 죽었지만" and "어떻게든" are no questions).
const INTERROGATIVES: &[&str] = &[
    "무슨",
    "어느",
    "어떤",
    "어떻게",
    "어디",
    "어디가",
    "어디서",
    "어디야",
    "왜",
    "누구",
    "누가",
    "뭐가",
    "뭘",
    "뭐야",
    "뭔데",
    "what",
    "which",
    "how",
    "why",
    "where",
    "who",
    "何",
    "どれ",
    "どの",
    "どう",
    "なぜ",
];
const CONDITIONAL_ENDINGS: &[&str] = &[
    "으면", "다면", "라면", "거든", "하면", "되면", "보면", "쓰면", "이면", "가면", "오면",
];
/// Nouns ending in 면 that are not the conditional "-면".
const MYEON_NOUNS: &[&str] = &[
    "화면", "측면", "정면", "장면", "표면", "전면", "후면", "단면", "국면", "방면", "체면", "수면",
];
const SPECULATIVE_ENDINGS: &[&str] = &[
    "겠지",
    "겠어",
    "겠네",
    "겠다",
    "겠는데",
    "을걸",
    "할걸",
    "일걸",
    "할게",
    "할께",
    "할거야",
    "할듯",
    "일듯",
    "을듯",
    "을지도",
    "일지도",
    "할지도",
];
const EN_SPECULATIVE: &[&str] = &["if", "unless", "might", "maybe", "probably", "would"];
const JA_HYPOTHETICAL: &[&str] = &["たら", "れば", "なら", "もし", "かも", "だろう", "でしょう"];
/// A plan, not an experience ("좋아, 가보자", "let's go"): every grade in the sentence is unexecuted.
const PROPOSITIVE_ENDINGS: &[&str] = &["보자", "가자", "하자", "쓰자", "깨자", "잡자", "뛰자"];
const JA_PROPOSITIVE: &[&str] = &["しよう", "行こう", "やろう", "みよう"];
/// Hearsay on the graded word ("좋대", "추천이래", "좋다고 하던데"): not the player's own experience.
const HEARSAY_ENDINGS: &[&str] = &["대", "대요", "래", "래요", "다고", "라고", "다던데", "다더라", "라던데", "라더라"];
/// Set phrases whose evaluative word grades nothing ("좋은 아침", "good luck").
const IDIOMS: &[&str] = &[
    "좋은 아침",
    "좋은 하루",
    "좋은 밤",
    "좋은 저녁",
    "좋은 꿈",
    "good morning",
    "good night",
    "good evening",
    "good luck",
    "nice to meet",
];
const ADVERSATIVE_WORDS: &[&str] = &[
    "근데",
    "그런데",
    "하지만",
    "다만",
    "그렇지만",
    "그치만",
    "그래도",
    "but",
    "however",
    "yet",
    "though",
    "although",
    "でも",
    "だけど",
    "しかし",
];
const KO_ADVERSATIVE_TAILS: &[&str] = &["어도", "아도", "해도", "여도", "워도", "와도", "봐도", "써도"];
const JA_ADVERSATIVE: &[&str] = &["けれども", "けれど", "けど", "なのに", "ものの"];
const FIRST_PERSON: &[&str] = &[
    "나", "내", "내가", "나는", "나도", "제가", "저", "제", "저는", "i", "i'm", "im", "my", "me", "私", "俺", "僕", "自分",
];
const FIRST_PERSON_SUBJECTS: &[&str] = &["내가", "나는", "제가", "저는", "i", "i'm", "im"];
const BODY_CONTROL: &[&str] = &[
    "손",
    "손가락",
    "손꾸락",
    "컨트롤",
    "타이밍",
    "피지컬",
    "반응",
    "조작",
    "에임",
    "hands",
    "hand",
    "fingers",
    "timing",
    "control",
    "aim",
    "操作",
    "指",
    "反応",
];
const MISTAKE: &[&str] = &[
    "꼬였",
    "꼬여",
    "꼬임",
    "미끄러",
    "삑",
    "놓쳤",
    "놓침",
    "늦었",
    "실수",
    "망쳤",
    "choked",
    "messed",
    "missed",
    "fumbled",
    "slipped",
    "ミス",
    "しくじ",
];
/// What a guide is called: "맞는데" grades a guide only next to one of these ("공략은 맞는데").
const GUIDE_NOUNS: &[&str] = &[
    "공략", "루트", "방법", "길", "빌드", "세팅", "팁", "전략", "말", "정보", "guide", "route", "build", "tip", "攻略",
];
/// Words that point back at the earlier vote, which a bare "취소" needs to be a retraction.
const RETRACTION_REFERENCES: &[&str] = &["아까", "방금", "전에", "그거", "이거", "말", "앞에"];

fn owned(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

impl FeedbackLexicon {
    /// The built-in vocabulary (Korean, English, Japanese): what runs without a dictionary.
    pub fn builtin() -> &'static FeedbackLexicon {
        static BUILTIN: OnceLock<FeedbackLexicon> = OnceLock::new();
        BUILTIN.get_or_init(|| FeedbackLexicon {
            recommend: owned(RECOMMEND),
            not_recommend: owned(NOT_RECOMMEND),
            positive_attribute: owned(POSITIVE_ATTRIBUTE),
            negative_attribute: owned(NEGATIVE_ATTRIBUTE),
            reaction: owned(REACTION),
            request: owned(REQUEST),
            retraction: owned(RETRACTION),
            self_blame: owned(SELF_BLAME),
            exaggeration: owned(EXAGGERATION),
            damage: owned(DAMAGE),
            hedge: owned(HEDGE),
        })
    }

    /// The built-in vocabulary extended by `rules` (a dictionary's `feedback_rules`).
    pub fn with_builtin(rules: &FeedbackLexicon) -> FeedbackLexicon {
        let mut lex = FeedbackLexicon::builtin().clone();
        lex.extend(rules);
        lex
    }

    /// Adds `other`'s phrases (lowercased, without duplicates).
    pub fn extend(&mut self, other: &FeedbackLexicon) {
        let add = |into: &mut Vec<String>, from: &[String]| {
            for p in from {
                let p = p.trim().to_lowercase();
                if !p.is_empty() && !into.contains(&p) {
                    into.push(p);
                }
            }
        };
        add(&mut self.recommend, &other.recommend);
        add(&mut self.not_recommend, &other.not_recommend);
        add(&mut self.positive_attribute, &other.positive_attribute);
        add(&mut self.negative_attribute, &other.negative_attribute);
        add(&mut self.reaction, &other.reaction);
        add(&mut self.request, &other.request);
        add(&mut self.retraction, &other.retraction);
        add(&mut self.self_blame, &other.self_blame);
        add(&mut self.exaggeration, &other.exaggeration);
        add(&mut self.damage, &other.damage);
        add(&mut self.hedge, &other.hedge);
    }
}

/// [`extract_pragmatic_feedback_with`] on the built-in vocabulary.
pub fn extract_pragmatic_feedback(
    raw_text: &str,
    tokens: &[FeedbackToken<'_>],
    locale: &str,
    ctx: FeedbackContext,
) -> Option<ExtractedFeedback> {
    extract_pragmatic_feedback_with(FeedbackLexicon::builtin(), raw_text, tokens, locale, ctx)
}

/// The feedback in one utterance, or `None` when it holds none (a request, a question, a hypothetical, small talk):
/// no vote is cast for `None`. `tokens` may be empty; `locale` ("ko", "en", "ja", …) only decides the Japanese "〜が、"
/// split, the script decides everything else.
pub fn extract_pragmatic_feedback_with(
    lex: &FeedbackLexicon,
    raw_text: &str,
    tokens: &[FeedbackToken<'_>],
    locale: &str,
    ctx: FeedbackContext,
) -> Option<ExtractedFeedback> {
    let japanese = locale.trim().to_lowercase().starts_with("ja") || raw_text.chars().any(is_kana);
    let mut toks = tokenize(raw_text, japanese);
    if toks.is_empty() {
        return None;
    }
    merge_hints(&mut toks, tokens);
    let a = Analysis::new(toks);

    let retracting = a.retraction_clauses(lex);
    let retracts_previous = !retracting.is_empty();
    let mut cues: Vec<Cue> = a
        .cues(lex)
        .into_iter()
        .filter(|c| !retracting.iter().any(|r| r.contains(&c.tok)))
        .filter(|c| !a.gated(lex, c))
        .collect();

    let mut sarcasm_blocked = false;
    if (ctx.recent_defeat_within_15s || ctx.high_friction_burst) && a.exaggerated(lex) && a.damaged(lex) {
        let before = cues.len();
        cues.retain(|c| c.sign < 0);
        sarcasm_blocked = cues.len() < before;
    }

    let segs = &a.segments;
    let blame: Vec<bool> = segs.iter().map(|s| a.self_blame(lex, s.range())).collect();
    let in_seg = |s: &Segment| cues.iter().filter(|c| s.range().contains(&c.tok)).copied().collect::<Vec<Cue>>();
    let deciding = segs.iter().rposition(|s| !s.concessive).unwrap_or(segs.len() - 1);
    let evaluation = if blame[deciding] {
        // The main clause blames the player's hands: it grades nothing. A concession that praised the guide stays,
        // at half strength ("공략은 맞는데 내 손이 꼬였어").
        let praised = segs
            .iter()
            .enumerate()
            .any(|(i, s)| i != deciding && in_seg(s).iter().any(|c| c.kind == ObsKind::Feedback && c.sign > 0));
        praised.then_some((ObsKind::Feedback, 1, 0.5))
    } else {
        combine(&in_seg(&segs[deciding])).or_else(|| {
            // Nothing graded in the main clause: the nearest other clause that grades stands (no contrast to lose).
            (0..segs.len())
                .rev()
                .filter(|&i| i != deciding && !blame[i])
                .find_map(|i| combine(&in_seg(&segs[i])))
        })
    };
    // A shield needs a self-blame and no disapproval of the guide.
    let shield_defeat_outcome =
        ctx.recent_defeat_within_15s && blame.iter().any(|b| *b) && !matches!(evaluation, Some((_, sign, _)) if sign < 0);
    if evaluation.is_none() && !retracts_previous && !shield_defeat_outcome && !sarcasm_blocked {
        return None;
    }
    Some(ExtractedFeedback {
        evaluation,
        retracts_previous,
        shield_defeat_outcome,
        target_mention: target_mention(raw_text, &a.toks),
        sarcasm_blocked,
    })
}

/// Records `extracted` in one target's observation history: a retraction first turns the last `feedback` observation
/// of the last `retract_window_ms` (default [`RETRACT_WINDOW_MS`]) into `sign` 0 in place, a blame shield turns the
/// defeat outcomes (−1) of the last [`DEFEAT_SHIELD_WINDOW_MS`] into `sign` 0, then the new evaluation is appended
/// alone (`weight` = kind weight × strength, `credit` kept apart, as PD-Verdict multiplies them). Returns that new
/// observation. History rows withdrawn this way leave `E` and `v` completely (PD-Verdict skips `sign` 0).
///
/// An app that holds a defeat back until the next utterance passes `shield_defeat` to
/// [`crate::credit::attribute_and_update_duels`] instead; this function covers the defeat that was already stored.
pub fn apply_feedback_to_ledger(
    history: &mut Vec<Obs>,
    extracted: &ExtractedFeedback,
    credit: f64,
    now_ms: i64,
    retract_window_ms: i64,
    params: &Params,
) -> Option<Obs> {
    let within = |o: &Obs, window: i64| o.at_ms <= now_ms && now_ms - o.at_ms <= window;
    if extracted.retracts_previous {
        let last = history
            .iter()
            .enumerate()
            .filter(|(_, o)| o.kind == ObsKind::Feedback.as_str() && o.sign != 0 && within(o, retract_window_ms))
            .max_by_key(|(i, o)| (o.at_ms, *i))
            .map(|(i, _)| i);
        if let Some(i) = last {
            history[i].sign = 0;
        }
    }
    if extracted.shield_defeat_outcome {
        for o in history.iter_mut() {
            if o.kind == "outcome" && o.sign < 0 && within(o, DEFEAT_SHIELD_WINDOW_MS) {
                o.sign = 0;
            }
        }
    }
    let (kind, sign, strength) = extracted.evaluation?;
    if sign == 0 {
        return None;
    }
    let unit = |x: f64| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
    let obs = Obs {
        kind: kind.as_str().into(),
        sign: sign.signum(),
        weight: params.weight(kind.as_str()) * unit(strength),
        credit: unit(credit),
        at_ms: now_ms,
    };
    history.push(obs.clone());
    Some(obs)
}

/// Which of `labels` (record titles, checkpoint names) `raw_text` names, longest first, ignoring spaces and case:
/// the explicit-mention flag of a credit candidate ("지하수로는 별로야" → "지하수로").
pub fn mentioned_label(raw_text: &str, labels: &[&str]) -> Option<usize> {
    let text = compact(raw_text);
    labels
        .iter()
        .enumerate()
        .map(|(i, l)| (i, compact(l)))
        .filter(|(_, l)| l.chars().count() >= 2 && text.contains(l.as_str()))
        .max_by_key(|(i, l)| (l.chars().count(), std::cmp::Reverse(*i)))
        .map(|(i, _)| i)
}

// ── text helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

fn is_hangul(c: char) -> bool {
    ('\u{AC00}'..='\u{D7A3}').contains(&c) || ('\u{3131}'..='\u{318E}').contains(&c)
}

fn is_kana(c: char) -> bool {
    ('\u{3040}'..='\u{30FF}').contains(&c)
}

fn is_cjk(c: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&c)
}

fn is_japanese_word(s: &str) -> bool {
    s.chars().any(|c| is_kana(c) || is_cjk(c))
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

fn compact(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

fn starts_any(s: &str, list: &[&str]) -> bool {
    list.iter().any(|p| s.starts_with(p))
}

fn ends_any(s: &str, list: &[&str]) -> bool {
    list.iter().any(|p| s.ends_with(p))
}

fn contains_any(s: &str, list: &[&str]) -> bool {
    list.iter().any(|p| s.contains(p))
}

/// An English word or its short inflection ("recommend" → "recommended", "works").
fn english_word(t: &str, w: &str) -> bool {
    t == w
        || t.strip_prefix(w)
            .is_some_and(|rest| rest.len() <= 3 && rest.chars().all(|c| c.is_ascii_alphabetic()))
}

/// One word of a phrase against one token: English whole word, Japanese anywhere, Korean prefix.
fn word_matches(t: &str, w: &str) -> bool {
    if w.is_ascii() {
        t.is_ascii() && english_word(t, w)
    } else if is_japanese_word(w) {
        t.contains(w)
    } else {
        t.starts_with(w)
    }
}

fn english_negator(t: &str) -> bool {
    EN_NEGATORS.contains(&t) || t.ends_with("n't") || matches!(t, "dont" | "doesnt" | "didnt" | "isnt" | "wasnt" | "cant" | "wont" | "aint")
}

fn is_light_verb(t: &str) -> bool {
    starts_any(t, LIGHT_VERBS)
}

fn is_conditional(t: &str) -> bool {
    if t.is_ascii() {
        return matches!(t, "if" | "unless");
    }
    if is_japanese_word(t) {
        return contains_any(t, &["たら", "れば", "なら", "もし"]);
    }
    ends_any(t, CONDITIONAL_ENDINGS) || (t.ends_with('면') && chars(t) >= 2 && !MYEON_NOUNS.contains(&t))
}

fn is_adversative_ending(t: &str) -> bool {
    let n = chars(t);
    if n < 2 || !t.chars().all(is_hangul) || matches!(t, "그런데" | "근데") {
        return false;
    }
    t.ends_with('데') || (n >= 3 && t.ends_with("지만")) || t.ends_with("건만") || t.ends_with("길래") || ends_any(t, KO_ADVERSATIVE_TAILS)
}

// ── tokens ───────────────────────────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct Tok {
    /// Lowercase, punctuation and elongation marks removed.
    s: String,
    /// Written with "~" ("좋~은").
    elongated: bool,
    question: bool,
    sentence_end: bool,
    clause_end: bool,
    /// Cut from a fused adversative ending ("했는데|별로지", "良かったけど").
    adversative_end: bool,
    hint_request: bool,
    hint_negation: bool,
    hint_negated: bool,
    hint_subject: bool,
}

fn tokenize(raw: &str, japanese: bool) -> Vec<Tok> {
    let mut toks: Vec<Tok> = Vec::new();
    let mut cur = String::new();
    let mut elongated = false;
    fn flush(toks: &mut Vec<Tok>, cur: &mut String, elongated: &mut bool) {
        if !cur.is_empty() {
            toks.push(Tok {
                s: std::mem::take(cur),
                elongated: *elongated,
                ..Default::default()
            });
        }
        *elongated = false;
    }
    for c in raw.chars() {
        let c = if matches!(c, '\u{2019}' | '\u{2018}') { '\'' } else { c };
        if c.is_alphanumeric() || (c == '\'' && cur.chars().last().is_some_and(|p| p.is_ascii_alphabetic())) {
            cur.extend(c.to_lowercase());
        } else if matches!(c, '~' | '\u{FF5E}' | '\u{301C}') {
            if cur.is_empty() {
                if let Some(last) = toks.last_mut() {
                    last.elongated = true;
                }
            } else {
                elongated = true;
            }
        } else if c.is_whitespace() {
            flush(&mut toks, &mut cur, &mut elongated);
            if c == '\n' {
                if let Some(last) = toks.last_mut() {
                    last.sentence_end = true;
                    last.clause_end = true;
                }
            }
        } else {
            flush(&mut toks, &mut cur, &mut elongated);
            if let Some(last) = toks.last_mut() {
                match c {
                    '?' | '\u{FF1F}' => {
                        last.question = true;
                        last.sentence_end = true;
                        last.clause_end = true;
                    }
                    '.' | '!' | '\u{3002}' | '\u{FF01}' | '\u{2026}' => {
                        last.sentence_end = true;
                        last.clause_end = true;
                    }
                    ',' | ';' | ':' | '\u{3001}' | '\u{FF0C}' => last.clause_end = true,
                    _ => {}
                }
            }
        }
    }
    flush(&mut toks, &mut cur, &mut elongated);
    if let Some(last) = toks.last_mut() {
        last.sentence_end = true;
        last.clause_end = true;
    }
    split_fused(toks, japanese)
}

/// Splits an adversative ending fused to the next word ("했는데별로지" → "했는데" + "별로지") and Japanese adversative
/// particles ("良かったけど微妙" → "良かったけど" + "微妙").
fn split_fused(toks: Vec<Tok>, japanese: bool) -> Vec<Tok> {
    let mut out = Vec::with_capacity(toks.len());
    for mut t in toks {
        loop {
            let cs: Vec<char> = t.s.chars().collect();
            let cut = if cs.iter().all(|c| is_hangul(*c)) {
                ["는데", "은데", "지만"]
                    .iter()
                    .filter_map(|m| {
                        let m: Vec<char> = m.chars().collect();
                        (1..cs.len().saturating_sub(1))
                            .find(|&k| cs[k..].starts_with(&m) && cs.len() - (k + 2) >= 2)
                            .map(|k| k + 2)
                    })
                    .min()
            } else if is_japanese_word(&t.s) {
                JA_ADVERSATIVE
                    .iter()
                    .filter_map(|m| {
                        let m: Vec<char> = m.chars().collect();
                        (1..cs.len())
                            .find(|&k| cs[k..].starts_with(&m) && cs.len() > k + m.len())
                            .map(|k| k + m.len())
                    })
                    .min()
            } else {
                None
            };
            let Some(k) = cut else { break };
            let head: String = cs[..k].iter().collect();
            let tail: String = cs[k..].iter().collect();
            out.push(Tok {
                s: head,
                elongated: t.elongated,
                adversative_end: true,
                ..Default::default()
            });
            t.s = tail;
        }
        if is_japanese_word(&t.s) && !t.sentence_end {
            // "良かったけど、" ends a concession; so does "〜が、" in Japanese.
            let ga = japanese && t.clause_end && t.s.ends_with('が') && chars(&t.s) >= 3;
            if ends_any(&t.s, JA_ADVERSATIVE) || ga {
                t.adversative_end = true;
            }
        }
        out.push(t);
    }
    out
}

fn merge_hints(toks: &mut [Tok], tokens: &[FeedbackToken<'_>]) {
    for h in tokens {
        let key = compact(h.surface);
        if key.is_empty() {
            continue;
        }
        for t in toks.iter_mut().filter(|t| t.s == key) {
            t.hint_request |= h.is_auxiliary_request;
            t.hint_negation |= h.is_negation;
            t.hint_negated |= h.negated_in_clause;
            t.hint_subject |= h.is_subject_or_topic;
        }
    }
}

// ── analysis ─────────────────────────────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Segment {
    start: usize,
    end: usize,
    /// A concession ("딜은 쎈데", "although …"): its grade gives way to the main clause's.
    concessive: bool,
}

impl Segment {
    fn range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// One evaluative word with its final polarity.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cue {
    tok: usize,
    sign: i32,
    /// 0.5 (attribute, double negative, hedge) or 1.0.
    strength: f64,
    kind: ObsKind,
}

struct Analysis {
    toks: Vec<Tok>,
    segments: Vec<Segment>,
    /// Per token: its comma-delimited clause and its sentence.
    clause: Vec<Range<usize>>,
    sentence: Vec<Range<usize>>,
}

/// A vocabulary entry for cue matching.
struct CueDef<'a> {
    text: &'a str,
    sign: i32,
    weak: bool,
    kind: ObsKind,
}

#[derive(Debug, Clone, Copy)]
struct Hit {
    start: usize,
    end: usize,
    fused_negation: bool,
}

fn match_cue(t: &str, c: &str) -> Option<Hit> {
    if c.is_empty() || t.is_empty() {
        return None;
    }
    let len = chars(c);
    let at = |byte: usize| chars(&t[..byte]);
    if c.is_ascii() {
        return (t.is_ascii() && english_word(t, c)).then_some(Hit {
            start: 0,
            end: len,
            fused_negation: false,
        });
    }
    if is_japanese_word(c) {
        return t.find(c).map(|b| Hit {
            start: at(b),
            end: at(b) + len,
            fused_negation: false,
        });
    }
    if t.starts_with(c) {
        return Some(Hit {
            start: 0,
            end: len,
            fused_negation: false,
        });
    }
    // "안좋아", "못써먹겠다": a negating adverb fused to the predicate.
    for neg in NEG_ADVERBS {
        if t.strip_prefix(neg).is_some_and(|rest| rest.starts_with(c)) {
            return Some(Hit {
                start: 1,
                end: 1 + len,
                fused_negation: true,
            });
        }
    }
    if len >= 2 {
        if let Some(b) = t.find(c) {
            return Some(Hit {
                start: at(b),
                end: at(b) + len,
                fused_negation: false,
            });
        }
    }
    None
}

fn ranges(toks: &[Tok], segments: &[Segment]) -> (Vec<Range<usize>>, Vec<Range<usize>>) {
    let n = toks.len();
    let mut clause = vec![0..0; n];
    let mut sentence = vec![0..0; n];
    let (mut cs, mut ss) = (0, 0);
    for i in 0..n {
        let seg_start = segments.iter().any(|s| s.start == i);
        if i > 0 && (toks[i - 1].clause_end || seg_start) {
            cs = i;
        }
        if i > 0 && toks[i - 1].sentence_end {
            ss = i;
        }
        clause[i] = cs..cs;
        sentence[i] = ss..ss;
    }
    // Close the ranges from the right.
    let mut end = n;
    for i in (0..n).rev() {
        clause[i].end = end;
        if clause[i].start == i {
            end = i;
        }
    }
    let mut end = n;
    for i in (0..n).rev() {
        sentence[i].end = end;
        if sentence[i].start == i {
            end = i;
        }
    }
    (clause, sentence)
}

fn segments(toks: &[Tok]) -> Vec<Segment> {
    let n = toks.len();
    let mut segs = Vec::new();
    let mut start = 0;
    let mut until_comma = false; // "although X, Y": the concession runs to the comma
    let mut reverse = false; // "X though Y": the concession is what follows
    for i in 0..n {
        let s = toks[i].s.as_str();
        let sentence_start = i == 0 || toks[i - 1].sentence_end;
        let though = matches!(s, "though" | "although");
        if i > start && ADVERSATIVE_WORDS.contains(&s) && !(though && sentence_start) {
            segs.push(Segment {
                start,
                end: i,
                concessive: !though,
            });
            start = i;
            reverse = though;
            until_comma = false;
        } else if though && sentence_start {
            if i > start {
                segs.push(Segment {
                    start,
                    end: i,
                    concessive: true,
                });
                start = i;
            }
            until_comma = true;
        }
        let adversative = toks[i].adversative_end || (is_adversative_ending(s) && !toks[i].sentence_end);
        if i + 1 < n && (adversative || (until_comma && toks[i].clause_end)) {
            segs.push(Segment {
                start,
                end: i + 1,
                concessive: true,
            });
            start = i + 1;
            until_comma = false;
            reverse = false;
        }
    }
    if start < n {
        segs.push(Segment {
            start,
            end: n,
            concessive: reverse,
        });
    }
    segs
}

/// Final `(sign, weak)` of a cue word of `sign` under `negations` negators.
fn polarity(sign: i32, weak: bool, negations: u8) -> (i32, bool) {
    match (sign > 0, negations) {
        (_, 0) => (sign, weak),
        // "추천 안 해": a negated recommendation is a disapproval.
        (true, 1) => (-1, weak),
        // "나쁘지 않았어": a double negative is a weak recommendation.
        (false, 1) => (1, true),
        (true, _) => (1, true),
        (false, _) => (-1, true),
    }
}

/// One clause's grade: the only sign present, or the stronger of two (equal strengths are ambivalent: no vote).
fn combine(cues: &[Cue]) -> Option<(ObsKind, i32, f64)> {
    let strongest = |kind: ObsKind, sign: i32| {
        cues.iter()
            .filter(|c| c.kind == kind && c.sign == sign)
            .map(|c| c.strength)
            .fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))))
    };
    match (strongest(ObsKind::Feedback, 1), strongest(ObsKind::Feedback, -1)) {
        (Some(p), None) => Some((ObsKind::Feedback, 1, p)),
        (None, Some(n)) => Some((ObsKind::Feedback, -1, n)),
        (Some(p), Some(n)) if p > n => Some((ObsKind::Feedback, 1, p)),
        (Some(p), Some(n)) if n > p => Some((ObsKind::Feedback, -1, n)),
        (Some(_), Some(_)) => None,
        (None, None) => strongest(ObsKind::Reaction, 1).map(|s| (ObsKind::Reaction, 1, s)),
    }
}

impl Analysis {
    fn new(toks: Vec<Tok>) -> Self {
        let segments = segments(&toks);
        let (clause, sentence) = ranges(&toks, &segments);
        Self {
            toks,
            segments,
            clause,
            sentence,
        }
    }

    fn s(&self, i: usize) -> &str {
        &self.toks[i].s
    }

    /// Whether the phrase `p` (words separated by spaces) starts at token `i` inside one clause.
    fn phrase_at(&self, i: usize, p: &str) -> Option<usize> {
        let words: Vec<&str> = p.split_whitespace().collect();
        if words.is_empty() || i + words.len() > self.clause[i].end {
            return None;
        }
        words
            .iter()
            .enumerate()
            .all(|(k, w)| word_matches(self.s(i + k), w))
            .then(|| i + words.len() - 1)
    }

    /// Whether any phrase of `list` occurs in `range`.
    fn has_phrase(&self, range: Range<usize>, list: &[String]) -> bool {
        range
            .clone()
            .any(|i| list.iter().any(|p| self.phrase_at(i, p).is_some_and(|last| last < range.end)))
    }

    fn cues(&self, lex: &FeedbackLexicon) -> Vec<Cue> {
        let defs: Vec<CueDef> = [
            (&lex.not_recommend, -1, false, ObsKind::Feedback),
            (&lex.recommend, 1, false, ObsKind::Feedback),
            (&lex.negative_attribute, -1, true, ObsKind::Feedback),
            (&lex.positive_attribute, 1, true, ObsKind::Feedback),
            (&lex.reaction, 1, false, ObsKind::Reaction),
        ]
        .into_iter()
        .flat_map(|(list, sign, weak, kind)| {
            list.iter().map(move |t| CueDef {
                text: t.as_str(),
                sign,
                weak,
                kind,
            })
        })
        .collect();
        // "공략은 맞는데": "맞다" grades a guide only when one is named in the clause.
        let guide_is_right = CueDef {
            text: "맞",
            sign: 1,
            weak: false,
            kind: ObsKind::Feedback,
        };
        let n = self.toks.len();
        let mut used = vec![false; n];
        let mut out = Vec::new();
        // Set phrases grade nothing ("좋은 아침").
        for i in 0..n {
            for p in IDIOMS {
                if let Some(last) = self.phrase_at(i, p) {
                    used[i..=last].iter_mut().for_each(|u| *u = true);
                }
            }
        }
        // Phrases first ("안 통해", "쓸 만해"): their words are not matched again.
        for i in 0..n {
            for d in defs.iter().filter(|d| d.text.contains(' ')) {
                if let Some(last) = self.phrase_at(i, d.text) {
                    if used[i..=last].iter().any(|u| *u) {
                        continue;
                    }
                    used[i..=last].iter_mut().for_each(|u| *u = true);
                    let end = chars(self.s(last));
                    out.push(self.cue(lex, i, last, end, false, d));
                }
            }
        }
        for i in (0..n).filter(|i| !used[*i]) {
            let t = self.s(i);
            // "별로 안 좋아": 별로 is the adverb "not very", the predicate after it carries the grade.
            if t == "별로" && self.negator_follows(i) {
                continue;
            }
            let mut hits: Vec<(Hit, &CueDef)> = defs
                .iter()
                .filter(|d| !d.text.contains(' '))
                .filter_map(|d| match_cue(t, d.text).map(|h| (h, d)))
                .collect();
            let graded_form = t.starts_with('맞') && !ends_any(t, &["아서", "고", "으면", "으니까", "다가"]);
            if graded_form && self.clause[i].clone().any(|j| starts_any(self.s(j), GUIDE_NOUNS)) {
                let h = Hit {
                    start: 0,
                    end: 1,
                    fused_negation: false,
                };
                hits.push((h, &guide_is_right));
            }
            // Longest first, disapproval before approval ("비추천" is 비추, not 추천); overlaps are dropped.
            hits.sort_by(|a, b| (b.0.end - b.0.start).cmp(&(a.0.end - a.0.start)).then(a.1.sign.cmp(&b.1.sign)));
            let mut taken: Vec<Range<usize>> = Vec::new();
            for (h, d) in hits {
                if taken.iter().any(|r| h.start < r.end && r.start < h.end) {
                    continue;
                }
                taken.push(h.start..h.end);
                out.push(self.cue(lex, i, i, h.end, h.fused_negation, d));
            }
        }
        out.retain(|c| c.sign != 0);
        out
    }

    /// The cue of definition `d` spanning tokens `first..=last` (ending at char `end` of the last).
    fn cue(&self, lex: &FeedbackLexicon, first: usize, last: usize, end: usize, fused: bool, d: &CueDef) -> Cue {
        let negations = self.negations(first, last, end, fused);
        if d.kind == ObsKind::Reaction {
            // A negated thanks is no reaction at all.
            let sign = if negations % 2 == 1 { 0 } else { 1 };
            return Cue {
                tok: last,
                sign,
                strength: 1.0,
                kind: d.kind,
            };
        }
        let (sign, weak) = polarity(d.sign, d.weak, negations);
        let hedged = self.clause[last].clone().any(|j| self.is_hedge(lex, j));
        Cue {
            tok: last,
            sign,
            strength: if weak || hedged { 0.5 } else { 1.0 },
            kind: d.kind,
        }
    }

    fn is_hedge(&self, lex: &FeedbackLexicon, j: usize) -> bool {
        let t = self.s(j);
        lex.hedge.iter().any(|h| if chars(h) == 1 { t == h } else { word_matches(t, h) })
    }

    fn negator_follows(&self, i: usize) -> bool {
        i + 1 < self.clause[i].end && (NEG_ADVERBS.contains(&self.s(i + 1)) || starts_any(self.s(i + 1), LONG_NEGATION))
    }

    /// Negators acting on the predicate of the cue at tokens `first..=last` (saturating at 2): fused into the word
    /// ("안좋아", "나쁘지않아", "悪くない"), an adverb before it ("안 좋아", "not bad"), and after it through light verbs
    /// and nominalisers: "추천 안 해", "나쁘지 않았어", "추천은 하지 않아", "나쁜 건 아니야".
    fn negations(&self, first: usize, last: usize, end: usize, fused: bool) -> u8 {
        let clause = self.clause[last].clone();
        let t = self.s(last);
        let after: String = t.chars().skip(end).collect();
        let mut n = 0u8;
        if fused {
            n += 1;
        }
        if contains_any(&after, FUSED_LONG_NEGATION) {
            n += 1;
        }
        if is_japanese_word(t) && contains_any(&after.chars().take(6).collect::<String>(), JA_NEGATION) {
            n += 1;
        }
        if first > clause.start && !self.toks[first - 1].clause_end && NEG_ADVERBS.contains(&self.s(first - 1)) {
            n += 1;
        }
        if t.is_ascii() && (first.saturating_sub(2).max(clause.start)..first).any(|j| english_negator(self.s(j))) {
            n += 1;
        }
        let mut ends_ji = t.ends_with('지');
        let mut j = last + 1;
        while j < clause.end && !self.toks[j - 1].clause_end {
            let w = self.s(j);
            if NEG_ADVERBS.contains(&w) {
                // "추천 안 해" negates the light verb of 추천하다; "별로 안 좋아" negates 좋다, not 별로.
                if j + 1 < clause.end && !self.toks[j].clause_end && is_light_verb(self.s(j + 1)) {
                    n += 1;
                    ends_ji = self.s(j + 1).ends_with('지');
                    j += 2;
                    continue;
                }
                break;
            }
            if ends_ji && starts_any(w, LONG_NEGATION) {
                n += 1;
                ends_ji = false;
            } else if starts_any(w, FUSED_NEGATED_LIGHT_VERBS) {
                n += 1;
                ends_ji = w.ends_with('지');
            } else if starts_any(w, COPULA_NEGATION) || self.toks[j].hint_negation {
                n += 1;
                ends_ji = false;
            } else if is_light_verb(w) || NOMINALIZERS.contains(&w) {
                ends_ji = w.ends_with('지');
            } else {
                break;
            }
            j += 1;
        }
        if n == 0 && self.toks[last].hint_negated {
            n = 1;
        }
        n.min(2)
    }

    /// Requests, questions, hypotheticals and hearsay are not feedback.
    fn gated(&self, lex: &FeedbackLexicon, c: &Cue) -> bool {
        self.request(lex, c.tok) || self.question(c.tok) || self.hypothetical(c.tok) || self.hearsay(c.tok)
    }

    /// "좋대", "추천이래", "좋다고 하던데": someone else's experience.
    fn hearsay(&self, i: usize) -> bool {
        let t = self.s(i);
        t.chars().all(is_hangul) && chars(t) >= 2 && ends_any(t, HEARSAY_ENDINGS)
    }

    /// A request: the evaluated word carries a request ending ("추천해줘"), or a request auxiliary follows it in the
    /// same sentence ("다른 공략 추천해 줘", "쓸 만한 빌드 추천해 줄래": the whole sentence asks), or the sentence
    /// asks in English ("please", "can you …").
    fn request(&self, lex: &FeedbackLexicon, i: usize) -> bool {
        let t = self.s(i);
        let sentence = self.sentence[i].clone();
        let hangul_request = |w: &str| w.chars().all(is_hangul) && ends_any(w, REQUEST_ENDINGS) && chars(w) > 2;
        if hangul_request(t) || (is_japanese_word(t) && contains_any(t, JA_REQUEST)) {
            return true;
        }
        let asks_later = (i + 1..sentence.end).any(|j| {
            let w = self.s(j);
            self.toks[j].hint_request || lex.request.iter().any(|r| !r.is_ascii() && w == r) || w.starts_with("알려") || hangul_request(w)
        });
        if asks_later {
            return true;
        }
        let words: Vec<&str> = sentence.clone().map(|j| self.s(j)).collect();
        if lex.request.iter().any(|r| r.is_ascii() && words.contains(&r.as_str())) {
            return true;
        }
        EN_REQUEST_OPENERS.iter().any(|o| words.len() >= o.len() && words[..o.len()] == **o)
    }

    fn question(&self, i: usize) -> bool {
        let sentence = self.sentence[i].clone();
        if sentence.clone().any(|j| self.toks[j].question) {
            return true;
        }
        let last = self.s(sentence.end - 1);
        if last != "아까" && last.chars().all(is_hangul) && ends_any(last, QUESTION_ENDINGS) {
            return true;
        }
        if is_japanese_word(last) && (last.ends_with('か') || last.ends_with("かな")) {
            return true;
        }
        self.clause[i].clone().any(|j| INTERROGATIVES.contains(&self.s(j)))
    }

    /// A hypothetical: the evaluated word is a condition ("좋으면") or a guess / promise ("추천할게", "좋겠지",
    /// "추천할 거야"), a condition precedes it in its clause, or its sentence proposes a plan ("좋아, 가보자",
    /// "let's go"); English "if / might / would" in the clause.
    fn hypothetical(&self, i: usize) -> bool {
        let t = self.s(i);
        let clause = self.clause[i].clone();
        if is_conditional(t) || (t.chars().all(is_hangul) && ends_any(t, SPECULATIVE_ENDINGS)) {
            return true;
        }
        let plan = self.sentence[i].clone().any(|j| {
            let w = self.s(j);
            (w.chars().all(is_hangul) && chars(w) >= 2 && ends_any(w, PROPOSITIVE_ENDINGS))
                || matches!(w, "let's" | "lets")
                || (is_japanese_word(w) && contains_any(w, JA_PROPOSITIVE))
        });
        if plan {
            return true;
        }
        if is_japanese_word(t) && contains_any(t, JA_HYPOTHETICAL) {
            return true;
        }
        if (t.ends_with('할') || t.ends_with('을')) && i + 1 < clause.end && self.s(i + 1).starts_with('거') {
            return true;
        }
        (clause.start..i).any(|j| is_conditional(self.s(j)))
            || (t.is_ascii() && clause.clone().any(|j| EN_SPECULATIVE.contains(&self.s(j))))
    }

    /// The clauses that withdraw the earlier vote ("아까 말 취소", "scratch that").
    fn retraction_clauses(&self, lex: &FeedbackLexicon) -> Vec<Range<usize>> {
        let mut seen: Vec<Range<usize>> = Vec::new();
        for i in 0..self.toks.len() {
            let clause = self.clause[i].clone();
            if seen.contains(&clause) {
                continue;
            }
            let phrase = lex
                .retraction
                .iter()
                .filter(|p| p.contains(' ') || is_japanese_word(p))
                .any(|p| self.has_phrase(clause.clone(), std::slice::from_ref(p)));
            let bare = lex.retraction.iter().filter(|p| !p.contains(' ') && !is_japanese_word(p)).any(|p| {
                clause.clone().any(|j| {
                    if p.is_ascii() {
                        self.s(j) == p
                    } else {
                        self.s(j).starts_with(p.as_str())
                    }
                })
            });
            let refers = clause.len() <= 3 || clause.clone().any(|j| starts_any(self.s(j), RETRACTION_REFERENCES));
            if phrase || (bare && refers) {
                seen.push(clause);
            }
        }
        seen
    }

    /// A clause in which the player blames their own hands ("내 손이 꼬였어", "my bad").
    fn self_blame(&self, lex: &FeedbackLexicon, seg: Range<usize>) -> bool {
        if self.has_phrase(seg.clone(), &lex.self_blame) {
            return true;
        }
        let guide = seg.clone().any(|j| starts_any(self.s(j), GUIDE_NOUNS));
        if guide {
            return false;
        }
        let first_person = |t: &str| FIRST_PERSON.contains(&t);
        let body = |t: &str| BODY_CONTROL.iter().any(|b| word_matches(t, b));
        let possessed_body = seg.clone().any(|j| {
            let t = self.s(j);
            (first_person(t) && j + 1 < seg.end && body(self.s(j + 1)))
                || ((t.starts_with('내') || t.starts_with('제')) && chars(t) > 1 && body(&t.chars().skip(1).collect::<String>()))
        });
        // "내가 / 제가 / i …", or a bare "나 / 저" opening the clause or marked as the subject by the analyzer.
        let subject = seg.clone().any(|j| {
            let t = self.s(j);
            FIRST_PERSON_SUBJECTS.contains(&t) || (FIRST_PERSON.contains(&t) && (j == seg.start || self.toks[j].hint_subject))
        });
        let mistake = seg.clone().any(|j| MISTAKE.iter().any(|m| match_cue(self.s(j), m).is_some()));
        possessed_body || (subject && mistake)
    }

    fn exaggerated(&self, lex: &FeedbackLexicon) -> bool {
        self.toks.iter().any(|t| t.elongated) || self.has_phrase(0..self.toks.len(), &lex.exaggeration)
    }

    fn damaged(&self, lex: &FeedbackLexicon) -> bool {
        self.toks.iter().any(|t| lex.damage.iter().any(|d| word_matches(&t.s, d)))
    }
}

/// `"{n}번"` for a numbered slot ("1번", "3번째", "#3", "slot 3", "3番"), or a quoted name.
fn target_mention(raw: &str, toks: &[Tok]) -> Option<String> {
    let number = |s: &str| -> Option<String> {
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        let rest = &s[digits.len()..];
        (!digits.is_empty() && (rest.starts_with('번') || rest.starts_with('番'))).then(|| format!("{digits}번"))
    };
    for (i, t) in toks.iter().enumerate() {
        if let Some(n) = number(&t.s) {
            return Some(n);
        }
        if matches!(t.s.as_str(), "slot" | "number" | "option" | "no") {
            if let Some(next) = toks
                .get(i + 1)
                .filter(|n| !n.s.is_empty() && n.s.chars().all(|c| c.is_ascii_digit()))
            {
                return Some(format!("{}번", next.s));
            }
        }
    }
    let raw_chars: Vec<char> = raw.chars().collect();
    if let Some(p) = raw_chars.iter().position(|c| *c == '#') {
        let digits: String = raw_chars[p + 1..].iter().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            return Some(format!("{digits}번"));
        }
    }
    for (open, close) in [('"', '"'), ('\u{300C}', '\u{300D}'), ('\u{201C}', '\u{201D}')] {
        if let Some(a) = raw.find(open) {
            let rest = &raw[a + open.len_utf8()..];
            if let Some(b) = rest.find(close) {
                let name = rest[..b].trim();
                if !name.is_empty() && chars(name) <= 40 {
                    return Some(name.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evidence, Params};

    fn ex(text: &str) -> Option<ExtractedFeedback> {
        extract_pragmatic_feedback(text, &[], "ko", FeedbackContext::default())
    }

    fn eval(text: &str) -> Option<(ObsKind, i32, f64)> {
        ex(text).and_then(|e| e.evaluation)
    }

    const F: ObsKind = ObsKind::Feedback;

    #[test]
    fn negation_flips_feedback_and_a_double_negative_is_weak() {
        let p = Params::v1();
        let o = |i, n| {
            let x = feedback_obs(i, n, 0.8, 7, &p);
            (x.sign, x.weight)
        };
        assert_eq!(o(FeedbackIntent::Recommend, false), (1, 1.0));
        assert_eq!(o(FeedbackIntent::Recommend, true), (-1, 1.0));
        assert_eq!(o(FeedbackIntent::NotRecommend, false), (-1, 1.0));
        assert_eq!(o(FeedbackIntent::NotRecommend, true), (1, 0.5));
        let x = feedback_obs(FeedbackIntent::Recommend, false, 1.7, 7, &p);
        assert_eq!((x.kind.as_str(), x.credit, x.at_ms), ("feedback", 1.0, 7), "credit is clamped");
        let m = manual_obs(false, 9, &p);
        assert_eq!((m.sign, m.weight, m.kind.as_str()), (-1, 2.0, "manual"));
    }

    #[test]
    fn fc01_a_request_is_not_feedback() {
        assert_eq!(ex("다른 공략 추천해 줘"), None, "the request auxiliary 줘 casts no vote");
        assert_eq!(ex("쓸 만한 빌드 추천해 줄래"), None);
        assert_eq!(ex("다른 공략 추천해줘"), None, "fused request ending");
        assert_eq!(ex("이거 추천이야?"), None, "a question");
        assert_eq!(ex("어떤 무기가 제일 좋아?"), None);
        assert_eq!(ex("나중에 써보고 좋으면 추천할게"), None, "a hypothetical");
        assert_eq!(ex("can you recommend a better route?"), None);
    }

    #[test]
    fn fc01_long_form_double_negation_is_a_weak_recommend() {
        assert_eq!(eval("우회하긴 했는데 나쁘지 않았어"), Some((F, 1, 0.5)));
        assert_eq!(
            eval("우회하긴 했는데별로지 않았어"),
            Some((F, 1, 0.5)),
            "fused -는데 split from the next word"
        );
        assert_eq!(eval("별로지 않았어"), Some((F, 1, 0.5)));
        assert_eq!(eval("나쁘지않네"), Some((F, 1, 0.5)), "fused long form");
        assert_eq!(eval("나쁜 건 아니야"), Some((F, 1, 0.5)), "copula negation");
    }

    #[test]
    fn fc01_explicit_recommend_sums_to_full_strength() {
        assert_eq!(eval("우회하긴 했는데 나쁘지 않았어, 추천!"), Some((F, 1, 1.0)));
    }

    #[test]
    fn fc01_tokens_from_an_analyzer_give_the_same_result() {
        let text = "우회하긴 했는데 나쁘지 않았어";
        let word = |s, neg, negated| FeedbackToken {
            surface: s,
            lemma: s,
            is_auxiliary_request: false,
            is_negation: neg,
            negated_in_clause: negated,
            is_subject_or_topic: false,
        };
        let tokens = [
            word("우회하긴", false, false),
            word("했는데", false, false),
            word("나쁘지", false, true),
            word("않았어", true, true),
        ];
        assert_eq!(
            extract_pragmatic_feedback(text, &tokens, "ko", FeedbackContext::default()),
            extract_pragmatic_feedback(text, &[], "ko", FeedbackContext::default())
        );
        let mut req = [
            word("다른", false, false),
            word("공략", false, false),
            word("추천해", false, false),
            word("줘", false, false),
        ];
        req[3].is_auxiliary_request = true;
        assert_eq!(
            extract_pragmatic_feedback("다른 공략 추천해 줘", &req, "ko", FeedbackContext::default()),
            None
        );
    }

    #[test]
    fn fc02_adversative_main_clause_wins() {
        let e = ex("딜은 쎈데 스태미나가 너무 딸려서 비추").unwrap();
        assert_eq!(
            e.evaluation,
            Some((F, -1, 1.0)),
            "the concession 쎈데 is rejected, the main clause decides"
        );
        assert!(!e.retracts_previous && !e.shield_defeat_outcome && !e.sarcasm_blocked);
        assert_eq!(eval("2페는 별로인데 그래도 추천"), Some((F, 1, 1.0)));
        assert_eq!(
            eval("길은 좋지만 너무 느려"),
            Some((F, -1, 0.5)),
            "an attribute in the main clause still wins"
        );
        assert_eq!(eval("the route was good but the boss part is bad"), Some((F, -1, 1.0)));
        assert_eq!(eval("although it is slow, i recommend it"), Some((F, 1, 1.0)));
        assert_eq!(eval("it worked, though it was a bit slow"), Some((F, 1, 1.0)));
    }

    #[test]
    fn fc03_self_blame_shields_the_defeat_and_keeps_the_praise() {
        let after_defeat = FeedbackContext {
            recent_defeat_within_15s: true,
            ..Default::default()
        };
        let e = extract_pragmatic_feedback("공략은 맞는데 내 손이 꼬였어", &[], "ko", after_defeat).unwrap();
        assert!(e.shield_defeat_outcome);
        assert_eq!(
            e.evaluation,
            Some((F, 1, 0.5)),
            "the conceded praise of the guide stays at half strength"
        );

        // The defeat 5 s before the utterance leaves the ledger; the half praise is the only vote.
        let p = Params::v2();
        let mut history = vec![Obs {
            kind: "outcome".into(),
            sign: -1,
            weight: 1.0,
            credit: 1.0,
            at_ms: 100_000,
        }];
        let new = apply_feedback_to_ledger(&mut history, &e, 1.0, 105_000, RETRACT_WINDOW_MS, &p).unwrap();
        assert_eq!(history[0].sign, 0, "the defeat −1 is not stored");
        assert_eq!((new.sign, new.weight, new.kind.as_str()), (1, 0.5, "feedback"));
        let (e_, v) = evidence(&history, 105_000, p.track("coach").unwrap(), &p);
        assert!((e_ - 0.5).abs() < 1e-12 && v > 0.0, "{e_} {v}");

        // Without a defeat there is nothing to shield, but the hands still grade nothing.
        let calm = ex("공략은 맞는데 내 손이 꼬였어").unwrap();
        assert!(!calm.shield_defeat_outcome);
        assert_eq!(calm.evaluation, Some((F, 1, 0.5)));
        let only = extract_pragmatic_feedback("내 손이 꼬였어", &[], "ko", after_defeat).unwrap();
        assert_eq!((only.evaluation, only.shield_defeat_outcome), (None, true));
        assert!(
            extract_pragmatic_feedback("my bad, the route is fine", &[], "en", after_defeat)
                .unwrap()
                .shield_defeat_outcome
        );
        // Blaming the guide is no shield.
        let guide = extract_pragmatic_feedback("내가 실수하긴 했는데 이 공략 진짜 별로야", &[], "ko", after_defeat).unwrap();
        assert_eq!((guide.evaluation, guide.shield_defeat_outcome), (Some((F, -1, 1.0)), false));
    }

    #[test]
    fn fc03_sarcasm_after_a_friction_burst_casts_no_positive_vote() {
        let burst = FeedbackContext {
            high_friction_burst: true,
            ..Default::default()
        };
        let e = extract_pragmatic_feedback("참 좋~은 길이다 죽기 딱 좋네", &[], "ko", burst).unwrap();
        assert!(e.sarcasm_blocked);
        assert_eq!(e.evaluation, None, "no +1");
        let p = Params::v2();
        let mut history = Vec::new();
        assert_eq!(apply_feedback_to_ledger(&mut history, &e, 1.0, 0, RETRACT_WINDOW_MS, &p), None);
        assert!(history.is_empty(), "nothing stored");
        let defeat = FeedbackContext {
            recent_defeat_within_15s: true,
            ..Default::default()
        };
        assert!(
            extract_pragmatic_feedback("yeah right, great route, i died twice", &[], "en", defeat)
                .unwrap()
                .sarcasm_blocked
        );
        // The same words in a calm moment are taken at face value (the guard is context-gated).
        assert_eq!(eval("참 좋~은 길이다 죽기 딱 좋네"), Some((F, 1, 1.0)));
        // Negative votes survive the guard.
        let neg = extract_pragmatic_feedback("참 좋~네 또 막혔어 비추", &[], "ko", burst).unwrap();
        assert_eq!(neg.evaluation, Some((F, -1, 1.0)));
    }

    #[test]
    fn fc04_retraction_zeroes_the_previous_vote_in_place() {
        let p = Params::v2();
        let first = ex("이거 추천!").unwrap();
        assert_eq!(first.evaluation, Some((F, 1, 1.0)));
        let mut history = Vec::new();
        apply_feedback_to_ledger(&mut history, &first, 1.0, 0, RETRACT_WINDOW_MS, &p).unwrap();

        let second = ex("아까 말 취소, 2페 막혀서 별로야").unwrap();
        assert!(second.retracts_previous);
        assert_eq!(second.evaluation, Some((F, -1, 1.0)));
        let new = apply_feedback_to_ledger(&mut history, &second, 1.0, 60_000, RETRACT_WINDOW_MS, &p).unwrap();
        assert_eq!(history.len(), 2, "withdrawn in place, not a third row");
        assert_eq!(history[0].sign, 0, "the t=0 vote is now a withdrawal");
        assert_eq!(new.sign, -1);
        let (e, v) = evidence(&history, 60_000, p.track("coach").unwrap(), &p);
        assert_eq!(e, 1.0, "only the new −1 counts");
        assert!(v < 0.0, "{v}");
    }

    #[test]
    fn retraction_reaches_back_five_minutes_and_only_feedback() {
        let p = Params::v2();
        let retract = ex("아까 말 취소").unwrap();
        assert_eq!((retract.evaluation, retract.retracts_previous), (None, true));
        let fb = |sign, at_ms| Obs {
            kind: "feedback".into(),
            sign,
            weight: 1.0,
            credit: 1.0,
            at_ms,
        };
        let mut old = vec![fb(1, 0)];
        assert_eq!(
            apply_feedback_to_ledger(&mut old, &retract, 1.0, 300_001, RETRACT_WINDOW_MS, &p),
            None
        );
        assert_eq!(old[0].sign, 1, "older than 5 minutes stays");
        let mut two = vec![
            fb(1, 10_000),
            fb(-1, 20_000),
            Obs {
                kind: "manual".into(),
                ..fb(1, 30_000)
            },
        ];
        apply_feedback_to_ledger(&mut two, &retract, 1.0, 40_000, RETRACT_WINDOW_MS, &p);
        assert_eq!(
            two.iter().map(|o| o.sign).collect::<Vec<_>>(),
            [1, 0, 1],
            "the last feedback only; manual untouched"
        );
        assert!(
            ex("방금 추천한 거 취소할게").unwrap().evaluation.is_none(),
            "the 추천 inside a retraction is no vote"
        );
        assert!(ex("scratch that, it is bad").unwrap().retracts_previous);
        assert!(
            !ex("스킬 취소 타이밍이 진짜 좋아 추천").unwrap().retracts_previous,
            "a bare 취소 needs a pointer back"
        );
    }

    #[test]
    fn reactions_hedges_negations_and_targets() {
        assert_eq!(eval("고마워!"), Some((ObsKind::Reaction, 1, 1.0)));
        assert_eq!(eval("덕분에 살았어 추천"), Some((F, 1, 1.0)), "feedback outranks the reaction");
        assert_eq!(eval("추천 안 해"), Some((F, -1, 1.0)));
        assert_eq!(eval("추천은 하지 않아"), Some((F, -1, 1.0)));
        assert_eq!(eval("추천 못해"), Some((F, -1, 1.0)), "못 fused to the light verb");
        assert_eq!(eval("몇 번 죽었지만 추천"), Some((F, 1, 1.0)), "몇 번 is no question");
        assert_eq!(eval("별로 안 좋아"), Some((F, -1, 1.0)), "별로 + 안 좋아 is one disapproval");
        assert_eq!(eval("안좋아"), Some((F, -1, 1.0)));
        assert_eq!(eval("비추천"), Some((F, -1, 1.0)), "비추천 is not 추천");
        assert_eq!(eval("좀 별로"), Some((F, -1, 0.5)), "a hedge halves it");
        assert_eq!(eval("not bad at all"), Some((F, 1, 0.5)));
        assert_eq!(eval("i don't recommend it"), Some((F, -1, 1.0)));
        assert_eq!(eval("좋은 점도 있고 나쁜 점도 있어"), None, "ambivalent: no vote");
        assert_eq!(ex("좋은 아침"), None, "a set phrase");
        assert_eq!(ex("good luck"), None);
        assert_eq!(ex("좋아, 이제 가보자"), None, "a plan, not an experience");
        assert_eq!(ex("다들 이 공략 좋대"), None, "hearsay");
        assert_eq!(eval("이 공략 진짜 좋다"), Some((F, 1, 1.0)));
        assert_eq!(ex("음 그렇구나"), None, "small talk");
        assert_eq!(ex(""), None);
        assert_eq!(ex("1번 루트 추천").unwrap().target_mention.as_deref(), Some("1번"));
        assert_eq!(ex("slot 3 is great").unwrap().target_mention.as_deref(), Some("3번"));
        assert_eq!(ex("\"지하수로\" 비추").unwrap().target_mention.as_deref(), Some("지하수로"));
        assert_eq!(mentioned_label("지하수로는 별로야", &["성문", "지하 수로", "지하"]), Some(1));
        assert_eq!(mentioned_label("그냥 별로야", &["성문"]), None);
    }

    #[test]
    fn japanese_utterances() {
        let ja = |t: &str| extract_pragmatic_feedback(t, &[], "ja", FeedbackContext::default()).and_then(|e| e.evaluation);
        assert_eq!(ja("このルートおすすめ!"), Some((F, 1, 1.0)));
        assert_eq!(ja("悪くない"), Some((F, 1, 0.5)));
        assert_eq!(ja("良かったけど、ボスは微妙"), Some((F, -1, 1.0)));
        assert_eq!(ja("おすすめしない"), Some((F, -1, 1.0)));
        assert_eq!(
            extract_pragmatic_feedback("おすすめ教えて", &[], "ja", FeedbackContext::default()),
            None
        );
    }

    #[test]
    fn a_dictionary_extends_the_builtin_vocabulary() {
        let rules: FeedbackLexicon =
            serde_json::from_str(r#"{"recommend": ["개꿀잼"], "self_blame": ["손이 미끄러"], "retraction": "없던 걸로, Forget It"}"#)
                .unwrap();
        assert_eq!(rules.retraction, ["없던 걸로", "forget it"], "a comma list, lowercased");
        let lex = FeedbackLexicon::with_builtin(&rules);
        assert!(lex.recommend.contains(&"개꿀잼".to_string()) && lex.recommend.contains(&"추천".to_string()));
        let e = extract_pragmatic_feedback_with(&lex, "개꿀잼이었어", &[], "ko", FeedbackContext::default()).unwrap();
        assert_eq!(e.evaluation, Some((F, 1, 1.0)));
        assert_eq!(FeedbackLexicon::default().recommend.len(), 0);
        let json = serde_json::to_value(ex("이거 추천!").unwrap()).unwrap();
        assert_eq!(json["evaluation"][0], "feedback");
    }
}
