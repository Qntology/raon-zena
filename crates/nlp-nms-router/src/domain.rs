//! Expression-domain scoring from the word dictionary (`zena-lexicon`).
//!
//! Every text node of the dictionary becomes one bank entry. A span of the user's words is scored the way
//! Qntology/logis-center scores fields against `bias.json`:
//!
//! 1. **exact match before vectors** — the span equals an `exact_match` entry (surface, josa-peeled lemma or
//!    request-stripped stem, spaces ignored) → 0.95; or, for Hangul, an entry is a prefix of the span's last word
//!    with at most two syllables left over (logis-center's agglutinative prefix match: "클리어했다" → "클리어")
//!    → 0.90. The longest entry wins; two domains tied on the same length → no exact verdict.
//! 2. **bias / prejudice vectors** — `bias` = max-pool cosine against the node's bias phrases (times the phrase
//!    weight), `prej` = max-pool cosine against its prejudice phrases (look-alikes that mean something else),
//!    `score = max(bias − prej·w, 0)` with `w = 0.3` for ≤ 2-word spans and `0.7` otherwise.
//!
//! The dictionary's `function` group (request endings, fillers) explains words away: a span made only of them
//! never names a domain.

use crate::embed::{cosine, TextEmbedder};
use crate::korean::is_hangul;
use crate::normalize::{normalize, Word};
use serde::Serialize;
use std::collections::HashSet;
use zena_lexicon::model::compact;
use zena_lexicon::{Action, Lexicon};

/// How a domain was caught.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Hit {
    Exact,
    Prefix,
    Vector,
}

pub const EXACT_SCORE: f32 = 0.95;
pub const PREFIX_SCORE: f32 = 0.90;

#[derive(Debug, Clone)]
pub struct DomainEntry {
    pub key: String,
    pub action: Action,
    bias: Vec<(Vec<f32>, f32)>,
    prej: Vec<Vec<f32>>,
    /// `(compact form, word count)` of each exact-match entry.
    exact: Vec<(String, usize)>,
}

pub struct DomainBank<E: TextEmbedder> {
    entries: Vec<DomainEntry>,
    function: HashSet<String>,
    pub embedder: E,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DomainScore {
    /// Index into the bank (`DomainBank::entry`).
    pub index: usize,
    pub score: f32,
    pub bias: f32,
    pub prej: f32,
    pub hit: Hit,
}

impl<E: TextEmbedder> DomainBank<E> {
    pub fn from_lexicon(lex: &Lexicon, embedder: E) -> Self {
        let entries = lex
            .text_nodes()
            .map(|n| DomainEntry {
                key: n.key.clone(),
                action: n.action.clone(),
                bias: n.bias.iter().map(|p| (embedder.embed(&p.text), p.weight)).collect(),
                prej: n.prejudice.iter().map(|p| embedder.embed(&p.text)).collect(),
                exact: n
                    .exact_match
                    .iter()
                    .map(|e| (compact(&normalize(e).text), normalize(e).words.len().max(1)))
                    .filter(|(c, _)| !c.is_empty())
                    .collect(),
            })
            .collect();
        let function = lex
            .function_phrases()
            .iter()
            .flat_map(|p| {
                let n = normalize(p);
                let whole = compact(&n.text);
                n.words.into_iter().map(|w| compact(&w.surface)).chain(std::iter::once(whole))
            })
            .filter(|w| !w.is_empty())
            .collect();
        Self {
            entries,
            function,
            embedder,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn entry(&self, i: usize) -> &DomainEntry {
        &self.entries[i]
    }
    pub fn index_of(&self, key: &str) -> Option<usize> {
        self.entries.iter().position(|e| e.key == key)
    }

    /// A word the function bank explains away (request ending, filler).
    pub fn is_function_word(&self, w: &Word) -> bool {
        [&w.surface, &w.lemma, &w.stem].iter().any(|f| self.function.contains(&compact(f)))
    }

    /// STEP 1 (logis-center): exact or agglutinative-prefix match of a span against every `exact_match` entry.
    pub fn exact(&self, words: &[Word]) -> Option<(usize, Hit)> {
        if words.is_empty() {
            return None;
        }
        let n = words.len();
        let head: String = words[..n - 1].iter().map(|w| compact(&w.surface)).collect();
        let last = &words[n - 1];
        // The span with its last word as written, josa-peeled, or request-stripped.
        let forms: Vec<String> = [&last.surface, &last.lemma, &last.stem]
            .iter()
            .map(|f| format!("{head}{}", compact(f)))
            .collect();
        let surface = &forms[0];
        // (entry, matched length, exact beats prefix)
        let mut hits: Vec<(usize, usize, Hit)> = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            for (ec, ew) in &e.exact {
                let len = ec.chars().count();
                // Spacing does not matter for an exact hit ("종료해 줘" = "종료해줘"); a prefix only extends the
                // span's last word, so the entry must have as many words as the span.
                if forms.iter().any(|f| f == ec) {
                    hits.push((i, len, Hit::Exact));
                } else if *ew == n && len >= 2 && surface.starts_with(ec.as_str()) && ec.chars().last().is_some_and(is_hangul) {
                    let rest: Vec<char> = surface[ec.len()..].chars().collect();
                    if !rest.is_empty() && rest.len() <= 2 && rest.iter().all(|c| is_hangul(*c)) {
                        hits.push((i, len, Hit::Prefix));
                    }
                }
            }
        }
        let rank = |h: &(usize, usize, Hit)| (h.1, h.2 == Hit::Exact);
        let top = hits.iter().map(rank).max()?;
        let mut winners: Vec<&(usize, usize, Hit)> = hits.iter().filter(|h| rank(h) == top).collect();
        winners.dedup_by_key(|h| h.0);
        // The longest entry wins; two domains tied on it give no exact verdict (logis-center).
        (winners.len() == 1).then(|| (winners[0].0, winners[0].2))
    }

    /// Every domain scored for `words` (a span), best first.
    pub fn score_words(&self, words: &[Word]) -> Vec<DomainScore> {
        let text = words.iter().map(|w| w.surface.as_str()).collect::<Vec<_>>().join(" ");
        let v = self.embedder.embed(&text);
        let pw = if words.len() <= 2 { 0.3 } else { 0.7 };
        let exact = self.exact(words);
        let mut out: Vec<DomainScore> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let bias = e.bias.iter().map(|(b, w)| cosine(&v, b) * w).fold(0f32, f32::max);
                let prej = e.prej.iter().map(|p| cosine(&v, p)).fold(0f32, f32::max);
                let mut s = DomainScore {
                    index: i,
                    score: (bias - prej * pw).max(0.0),
                    bias,
                    prej,
                    hit: Hit::Vector,
                };
                if let Some((j, h)) = exact {
                    if j == i {
                        let x = if h == Hit::Exact { EXACT_SCORE } else { PREFIX_SCORE };
                        if x >= s.score {
                            s.score = x;
                            s.hit = h;
                        }
                    }
                }
                s
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out
    }

    /// Every domain scored for free text, best first.
    pub fn score(&self, text: &str) -> Vec<DomainScore> {
        self.score_words(&normalize(text).words)
    }

    /// The best domain of a span, unless the span is only function words.
    pub fn best(&self, words: &[Word]) -> Option<DomainScore> {
        if words.iter().all(|w| self.is_function_word(w)) {
            return None;
        }
        self.score_words(words).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashedNgramEmbedder;

    pub(crate) fn bank() -> DomainBank<HashedNgramEmbedder> {
        DomainBank::from_lexicon(&Lexicon::bundled(), HashedNgramEmbedder::default())
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
