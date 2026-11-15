//! Span NMS over an utterance (port of logis-center `query_shipping.rs` / `query_commerce.rs`).
//!
//! Every word span of length 1..=`max_len` is scored against the intent bank; candidates are
//! sorted by score, then by length (longer wins ties), and any candidate that overlaps an
//! already accepted span by at least one word is suppressed. The survivors, in reading order,
//! are the intents of the utterance ("맨 앞으로 나와서 보스 패턴 알려줘" → TopmostOn + CoachQuestion).

use crate::embed::TextEmbedder;
use crate::intent::{Intent, IntentBank};
use crate::normalize::Word;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpanHit {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub intent: Intent,
    pub score: f32,
}

pub fn span_nms<E: TextEmbedder>(words: &[Word], bank: &IntentBank<E>, max_len: usize, accept: f32) -> Vec<SpanHit> {
    let mut cands = Vec::new();
    for s in 0..words.len() {
        for e in (s + 1)..=(s + max_len).min(words.len()) {
            let text = words[s..e].iter().map(|w| w.surface.as_str()).collect::<Vec<_>>().join(" ");
            let best = bank.score(&text)[0];
            if best.score >= accept {
                cands.push(SpanHit {
                    start: s,
                    end: e,
                    text,
                    intent: best.intent,
                    score: best.score,
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
    // Same intent twice in one utterance collapses to the stronger span.
    winners.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut seen = Vec::new();
    winners.retain(|w| {
        let fresh = !seen.contains(&w.intent);
        seen.push(w.intent);
        fresh
    });
    winners.sort_by_key(|w| w.start);
    winners
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashedNgramEmbedder;
    use crate::intent::DEFAULT_BANK;
    use crate::normalize::normalize;

    #[test]
    fn one_utterance_two_intents() {
        let bank = IntentBank::new(HashedNgramEmbedder::default(), DEFAULT_BANK);
        let n = normalize("제나야 화면 맨 앞으로 나와서 보스 패턴 어떻게 피해");
        let hits = span_nms(&n.words, &bank, 5, 0.6);
        let intents: Vec<_> = hits.iter().map(|h| h.intent).collect();
        assert_eq!(intents, vec![Intent::TopmostOn, Intent::CoachQuestion], "{hits:?}");
        assert!(hits.windows(2).all(|w| w[0].end <= w[1].start));
    }
}
