//! Zena's expression domains: the word dictionary (`zena-lexicon`) as premium-discount-nms's generic domain bank.
//!
//! Every text node of the dictionary becomes one bank entry carrying its [`Action`]; the scoring itself (exact /
//! agglutinative-prefix match before vectors, bias / prejudice max-pool cosine with the prejudice penalty) is
//! premium-discount-nms's. The dictionary's `function` group (request endings, fillers) explains words away.

pub use premium_discount_nms::domain::{DomainEntry, DomainScore, DomainSpec, Hit, EXACT_SCORE, PREFIX_SCORE};
use premium_discount_nms::embed::TextEmbedder;
use zena_lexicon::{Action, Lexicon};

/// Zena's domain bank: entries carry the dictionary node's [`Action`].
pub type DomainBank<E> = premium_discount_nms::domain::DomainBank<Action, E>;

/// What an utterance may start with to address Zena; removed before matching (`Normalized::addressed`).
pub const WAKE_WORDS: &[&str] = &["제나야", "제나", "라온 제나", "라온제나", "hey zena", "zena", "raon zena"];

/// Makes Zena's [`WAKE_WORDS`] the NMS core's wake words (process-wide; the router and the banks call it).
pub fn install_wake_words() {
    premium_discount_nms::normalize::set_wake_words(WAKE_WORDS);
}

/// The dictionary's text nodes as bank specs, and its function phrases.
pub fn specs(lex: &Lexicon) -> (Vec<DomainSpec<Action>>, Vec<String>) {
    let specs = lex
        .text_nodes()
        .map(|n| DomainSpec {
            key: n.key.clone(),
            payload: n.action.clone(),
            bias: n.bias.iter().map(|p| (p.text.clone(), p.weight)).collect(),
            prejudice: n.prejudice.iter().map(|p| p.text.clone()).collect(),
            exact: n.exact_match.clone(),
        })
        .collect();
    (specs, lex.function_phrases().to_vec())
}

/// Building a [`DomainBank`] from the word dictionary.
pub trait LexiconBank<E: TextEmbedder>: Sized {
    fn from_lexicon(lex: &Lexicon, embedder: E) -> Self;
    /// Each phrase embedded by its content words (the rule analyzer strips request endings and auxiliaries), the
    /// way spans are embedded when an analyzer runs.
    fn from_lexicon_content(lex: &Lexicon, embedder: E) -> Self;
}

impl<E: TextEmbedder> LexiconBank<E> for DomainBank<E> {
    fn from_lexicon(lex: &Lexicon, embedder: E) -> Self {
        install_wake_words();
        let (specs, function) = specs(lex);
        premium_discount_nms::domain::DomainBank::from_specs(specs, &function, embedder)
    }

    fn from_lexicon_content(lex: &Lexicon, embedder: E) -> Self {
        install_wake_words();
        let (specs, function) = specs(lex);
        premium_discount_nms::domain::DomainBank::from_specs_content(specs, &function, embedder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashedNgramEmbedder;
    use crate::normalize::normalize;

    pub(crate) fn bank() -> DomainBank<HashedNgramEmbedder> {
        DomainBank::from_lexicon(&Lexicon::bundled(), HashedNgramEmbedder::default())
    }

    /// The span test that used to live in the NMS core: it needs this dictionary.
    #[test]
    fn one_utterance_two_domains() {
        let bank = bank();
        let n = normalize("제나야 화면 맨 앞으로 나와서 보스 패턴 어떻게 피해");
        let hits = crate::span::span_nms(&n.words, &bank, 5, 0.6);
        let keys: Vec<_> = hits.iter().map(|h| bank.entry(h.index).key.as_str()).collect();
        assert_eq!(keys, vec!["system.topmost_on", "coach.game_tip"], "{hits:?}");
        assert!(hits.windows(2).all(|w| w[0].end <= w[1].start));
        let n = normalize("와 보스 잡았다 이겼어 고마워 제나");
        let hits = crate::span::span_nms(&n.words, &bank, 5, 0.6);
        let keys: Vec<_> = hits.iter().map(|h| bank.entry(h.index).key.as_str()).collect();
        assert!(keys.contains(&"emotion.celebrate") && keys.contains(&"emotion.thanks"), "{keys:?}");
    }

    #[test]
    fn prd_utterances_map_to_their_domains() {
        let b = bank();
        let top = |t: &str| b.entry(b.score(t)[0].index).key.clone();
        assert_eq!(top("잠깐 숨어있어"), "system.hide_avatar");
        assert_eq!(top("잠깐 모습만 숨겨줘"), "system.hide_avatar");
        assert_eq!(top("화면 맨 앞으로 나와"), "system.topmost_on");
        assert_eq!(top("다른 창 뒤로 가 있어"), "system.topmost_off");
        assert_eq!(top("방금 그 보스 패턴 어떻게 피해"), "coach.game_tip");
        assert_eq!(top("내 인벤토리에서 물약 어디 있어"), "coach.inventory");
    }

    #[test]
    fn exact_and_agglutinative_prefix_matches_come_before_vectors() {
        let b = bank();
        let w = |t: &str| normalize(t).words;
        let key = |r: Option<(usize, Hit)>| r.map(|(i, h)| (b.entry(i).key.clone(), h));
        assert_eq!(key(b.exact(&w("클리어"))), Some(("emotion.celebrate".into(), Hit::Exact)));
        assert_eq!(
            key(b.exact(&w("클리어했다"))),
            Some(("emotion.celebrate".into(), Hit::Prefix)),
            "two syllables of ending"
        );
        assert_eq!(key(b.exact(&w("떡락했어"))), Some(("market.stock_down".into(), Hit::Prefix)));
        assert_eq!(
            key(b.exact(&w("종료해 줘"))),
            Some(("system.exit_app".into(), Hit::Exact)),
            "spaces are ignored"
        );
        assert_eq!(
            key(b.exact(&w("물약을"))),
            Some(("coach.inventory".into(), Hit::Exact)),
            "the josa-peeled lemma matches"
        );
        assert_eq!(
            b.exact(&w("하이라이트")),
            None,
            "three syllables left over is another word, not an ending"
        );
        assert_eq!(b.exact(&w("highlight")), None, "prefix matching is for Hangul only");
        let s = b.score("떡상 가즈아");
        assert_eq!(b.entry(s[0].index).key, "market.stock_up");
    }

    #[test]
    fn function_words_alone_name_nothing() {
        let b = bank();
        assert!(b.best(&normalize("해 줘").words).is_none());
        assert!(b.best(&normalize("좀 진짜").words).is_none());
        assert!(b.best(&normalize("종료해 줘").words).is_some());
    }
}
