//! Span NMS over an utterance (port of logis-center `query_shipping.rs` / `query_commerce.rs`).
//!
//! Every word span of length 1..=`max_len` is scored against the dictionary's domain bank; candidates are
//! sorted by score, then by length (longer wins ties), and any candidate that overlaps an already accepted span
//! by at least one word is suppressed. The survivors, in reading order, are the expression domains of the
//! utterance ("맨 앞으로 나와서 보스 패턴 알려줘" → system.topmost_on + coach.game_tip).

use crate::domain::{DomainBank, Hit};
use crate::embed::TextEmbedder;
use crate::normalize::Word;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpanHit {
    pub start: usize,
    pub end: usize,
    pub text: String,
    /// Index into the bank.
    pub index: usize,
    pub score: f32,
    pub hit: Hit,
}

pub fn span_nms<E: TextEmbedder>(words: &[Word], bank: &DomainBank<E>, max_len: usize, accept: f32) -> Vec<SpanHit> {
    let mut cands = Vec::new();
    for s in 0..words.len() {
        for e in (s + 1)..=(s + max_len).min(words.len()) {
            let Some(best) = bank.best(&words[s..e]) else { continue };
            if best.score >= accept {
                let text = words[s..e].iter().map(|w| w.surface.as_str()).collect::<Vec<_>>().join(" ");
                cands.push(SpanHit {
                    start: s,
                    end: e,
                    text,
                    index: best.index,
                    score: best.score,
                    hit: best.hit,
                });
            }
        }
    }
    cands.sort_by(|a, b| b.score.total_cmp(&a.score).then((b.end - b.start).cmp(&(a.end - a.start))));
    let mut winners: Vec<SpanHit> = Vec::new();
    for c in cands {
        if !winners.iter().any(|w| c.start < w.end && c.end > w.start) {
            winners.push(c);
        }
    }
    // The same domain twice in one utterance collapses to the stronger span.
    winners.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut seen = Vec::new();
    winners.retain(|w| {
        let fresh = !seen.contains(&w.index);
        seen.push(w.index);
        fresh
    });
    winners.sort_by_key(|w| w.start);
    winners
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashedNgramEmbedder;
    use crate::normalize::normalize;
    use zena_lexicon::Lexicon;

    #[test]
    fn one_utterance_two_domains() {
        let bank = DomainBank::from_lexicon(&Lexicon::bundled(), HashedNgramEmbedder::default());
        let n = normalize("제나야 화면 맨 앞으로 나와서 보스 패턴 어떻게 피해");
        let hits = span_nms(&n.words, &bank, 5, 0.6);
        let keys: Vec<_> = hits.iter().map(|h| bank.entry(h.index).key.as_str()).collect();
        assert_eq!(keys, vec!["system.topmost_on", "coach.game_tip"], "{hits:?}");
        assert!(hits.windows(2).all(|w| w[0].end <= w[1].start));
        let n = normalize("와 보스 잡았다 이겼어 고마워 제나");
        let hits = span_nms(&n.words, &bank, 5, 0.6);
        let keys: Vec<_> = hits.iter().map(|h| bank.entry(h.index).key.as_str()).collect();
        assert!(keys.contains(&"emotion.celebrate") && keys.contains(&"emotion.thanks"), "{keys:?}");
    }
}
