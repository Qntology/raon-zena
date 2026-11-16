//! NLP-NMS v2, step ②–③: what each word of an utterance does — content, function word, auxiliary / request
//! predicate, or negation — so that only content words can name a domain and negated requests are not acted on.
//!
//! * [`StanzaAnalyzer`] (feature `stanza`) runs the Stanza pipeline (`zena-nlp`, ONNX): UPOS, content stem (lemma),
//!   dependency relation per eojeol. Function units are those logis-center drops from search queries
//!   (`STANZA_DROP_TAGS`: ADP, PUNCT, PART, SCONJ, CCONJ, PRON) when they hold no content morpheme; a verb that
//!   follows a predicate as its `compound`/`aux` with an auxiliary lemma (보 · 주 · 드리 …) is an auxiliary
//!   ("써 봐" → 보, "변신해 줘" → 주); 마 · 말 after a predicate is the prohibitive "-지 마", 안 · 못 before one and
//!   `Polarity=Neg` are negations.
//! * [`RuleAnalyzer`] gives the same roles from closed word lists when no model is installed.
//!
//! Negation scope: a negation covers its clause — the words since the previous clause end (a connective "-고 ·
//! -서" predicate or an earlier negation) — so "끄지 말고 계속 켜 둬" negates only "끄지 말고".

use crate::normalize::{Normalized, Role, Word};

pub trait Analyzer: Send + Sync {
    fn name(&self) -> &'static str;
    /// Set roles (and analyzer lemmas) on the words of a normalised utterance, then mark negation scopes.
    fn analyze(&self, n: &mut Normalized);
}

/// No analysis: every word is content (the v1 router).
pub struct NoAnalyzer;

impl Analyzer for NoAnalyzer {
    fn name(&self) -> &'static str {
        "none"
    }
    fn analyze(&self, _n: &mut Normalized) {}
}

/// Korean auxiliary / request predicates that follow a main predicate.
const AUX_SURFACES: &[&str] = &[
    "봐",
    "봐요",
    "봐라",
    "봐줘",
    "볼래",
    "보자",
    "줘",
    "줘요",
    "주세요",
    "줄래",
    "줄래요",
    "주라",
    "드려",
    "드릴게",
];
/// Lemmas of the same, as Stanza gives them.
#[cfg(feature = "stanza")]
const AUX_LEMMAS: &[&str] = &["보", "주", "드리", "버리", "놓", "두"];
/// The prohibitive auxiliary "-지 마 / 말다".
const PROHIBIT_SURFACES: &[&str] = &["마", "마라", "마요", "말고", "말아", "말아줘", "말아요", "말자", "말래"];
#[cfg(feature = "stanza")]
const PROHIBIT_LEMMAS: &[&str] = &["마", "말"];
/// Negating adverbs before a predicate.
const NEG_ADVERBS: &[&str] = &["안", "못", "not", "never", "don't", "dont"];
/// Korean determiners before a noun ("이 보스", "그 패턴").
const KO_DETERMINERS: &[&str] = &["이", "그", "저", "이런", "그런", "저런"];
/// English function words (tokens Stanza would tag ADP / DET / PRON / PART).
const EN_FUNCTION: &[&str] = &[
    "the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "me", "my", "you", "your", "it", "please",
];

fn ends_any(s: &str, tails: &[&str]) -> bool {
    tails.iter().any(|t| s.ends_with(t))
}

/// A predicate form that can host an auxiliary or a prohibition ("써", "변신해", "끄지", "알려주지").
fn is_predicate_like(w: &Word) -> bool {
    if matches!(w.upos.as_str(), "VERB" | "ADJ" | "AUX") {
        return true;
    }
    // Hangul syllables are atomic, so the common verb-final syllables are listed as they are written.
    ends_any(
        &w.surface,
        &[
            "지", "어", "아", "여", "해", "워", "와", "져", "게", "줘", "써", "봐", "가", "꺼", "켜", "떠", "놔", "둬", "차",
        ],
    )
}

/// A clause-ending connective predicate ("끄지 말고", "와서").
fn ends_clause(w: &Word) -> bool {
    w.role != Role::Function && ends_any(&w.surface, &["고", "서", "는데", "지만"]) && w.surface.chars().count() >= 2
}

/// Mark every word in the scope of a negation (`Word::negated`).
pub fn mark_negation(words: &mut [Word]) {
    let mut clause_start = 0;
    for i in 0..words.len() {
        if words[i].role == Role::Negation {
            // A negating adverb ("안 가") scopes forward over its predicate; a prohibition ("끄지 마") backward.
            let adverb = NEG_ADVERBS.contains(&words[i].surface.as_str()) && words[i].upos != "AUX";
            let (a, b) = if adverb {
                (i, (i + 1).min(words.len() - 1))
            } else {
                (clause_start, i)
            };
            for w in &mut words[a..=b] {
                w.negated = true;
            }
            clause_start = b + 1;
        } else if ends_clause(&words[i]) && i > 0 && words[i - 1].role != Role::Negation {
            // "와서" closes a clause unless the clause is the negated "-지 말고" itself (handled above).
            let next_is_neg = words.get(i + 1).is_some_and(|w| w.role == Role::Negation);
            if !next_is_neg {
                clause_start = i + 1;
            }
        }
    }
    // A prohibition fused into one word ("끄지마", "가지말고").
    let mut start = 0;
    for i in 0..words.len() {
        let s = words[i].surface.as_str();
        if ["지마", "지말고", "지말아", "지마라"].iter().any(|t| s.ends_with(t)) && s.chars().count() > 2 {
            for w in &mut words[start..=i] {
                w.negated = true;
            }
            start = i + 1;
        }
    }
}

/// Closed-class rules (no model).
pub struct RuleAnalyzer;

impl RuleAnalyzer {
    pub fn roles(words: &mut [Word]) {
        for i in 0..words.len() {
            let s = words[i].surface.as_str();
            let after_predicate = i > 0 && is_predicate_like(&words[i - 1]);
            words[i].role = if PROHIBIT_SURFACES.contains(&s) && after_predicate {
                Role::Negation
            } else if NEG_ADVERBS.contains(&s) {
                Role::Negation
            } else if AUX_SURFACES.contains(&s) && after_predicate {
                Role::Auxiliary
            } else if EN_FUNCTION.contains(&s) || (KO_DETERMINERS.contains(&s) && i + 1 < words.len()) {
                Role::Function
            } else {
                Role::Content
            };
        }
    }
}

impl Analyzer for RuleAnalyzer {
    fn name(&self) -> &'static str {
        "rules"
    }
    fn analyze(&self, n: &mut Normalized) {
        Self::roles(&mut n.words);
        mark_negation(&mut n.words);
    }
}

/// The Stanza pipeline (ONNX) for Korean and the other converted languages; falls back to [`RuleAnalyzer`] for a
/// language it cannot analyze or when the units do not line up with the words.
#[cfg(feature = "stanza")]
pub struct StanzaAnalyzer {
    /// The app language used to break ties in language detection.
    pub ui_language: String,
}

#[cfg(feature = "stanza")]
impl StanzaAnalyzer {
    /// Opens the model of `lang` now (the first open takes about half a second) so the first utterance is not slow.
    pub fn warm(lang: &str) -> anyhow::Result<()> {
        zena_nlp::stanza::pipeline(lang).map(|_| ())
    }

    fn roles(words: &mut [Word], qa: &zena_nlp::stanza::QueryAnalysis) {
        for (i, u) in qa.units.iter().enumerate() {
            let lemma = u.words.first().map(|w| w.lemma.as_str()).unwrap_or("");
            // A predicate before, by the parser or by its ending ("가지 마": the tagger may read 가지 as a noun).
            let after_predicate = i > 0 && is_predicate_like(&words[i - 1]);
            let neg_feat = u.words.iter().any(|w| w.feats.contains("Polarity=Neg"));
            let w = &mut words[i];
            w.upos = u.upos.clone();
            w.deprel = u.deprel.clone();
            if !u.stem.is_empty() && u.stem != w.surface {
                w.alt = Some(u.stem.to_lowercase());
            }
            let verbal = matches!(u.upos.as_str(), "VERB" | "AUX");
            w.role = if (verbal && PROHIBIT_LEMMAS.contains(&lemma) && after_predicate)
                || neg_feat
                || (u.upos == "ADV" && NEG_ADVERBS.contains(&w.surface.as_str()))
            {
                Role::Negation
            } else if verbal && after_predicate && AUX_LEMMAS.contains(&lemma) {
                // "-아/어 보다 · 주다" after the main predicate, whatever relation the parser gave it.
                Role::Auxiliary
            } else if (u.functional && zena_nlp::stanza::is_drop_tag(&u.upos)) || u.upos == "DET" {
                // Determiners ("이 보스") name nothing either.
                Role::Function
            } else {
                Role::Content
            };
        }
    }
}

#[cfg(feature = "stanza")]
impl Analyzer for StanzaAnalyzer {
    fn name(&self) -> &'static str {
        "stanza"
    }
    fn analyze(&self, n: &mut Normalized) {
        let lang = zena_nlp::stanza::detect_query_lang(&n.text, &self.ui_language);
        let analysis = lang.and_then(|l| zena_nlp::stanza::analyze_query(&n.text, l).ok());
        match analysis {
            Some(qa) if qa.whitespace_units && qa.units.len() == n.words.len() => Self::roles(&mut n.words, &qa),
            _ => RuleAnalyzer::roles(&mut n.words),
        }
        mark_negation(&mut n.words);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize;

    fn roles(text: &str) -> Vec<(String, Role, bool)> {
        let mut n = normalize(text);
        RuleAnalyzer.analyze(&mut n);
        n.words.into_iter().map(|w| (w.surface, w.role, w.negated)).collect()
    }

    #[test]
    fn auxiliaries_requests_and_negation_scopes() {
        let r = roles("안경이랑 헤드셋 써 봐");
        assert_eq!(r[3].1, Role::Auxiliary, "{r:?}");
        assert!(r.iter().all(|x| !x.2));
        let r = roles("앱 끄지 마");
        assert_eq!(r[2].1, Role::Negation);
        assert!(r.iter().all(|x| x.2), "the whole clause is negated: {r:?}");
        let r = roles("끄지 말고 계속 켜 둬");
        assert_eq!(
            r.iter().map(|x| x.2).collect::<Vec<_>>(),
            [true, true, false, false, false],
            "{r:?}"
        );
        let r = roles("앱끄지마");
        assert!(r[0].2, "fused prohibition");
        let r = roles("나 오늘 안 가");
        assert_eq!((r[2].1, r[3].2, r[0].2), (Role::Negation, true, false), "{r:?}");
        let r = roles("이거 봐 봐");
        assert_eq!(r[1].1, Role::Content, "a lone 봐 after a noun is the main verb");
        assert_eq!(r[2].1, Role::Auxiliary);
    }
}
