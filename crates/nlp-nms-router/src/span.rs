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

/// Thresholds of the v2 selection ([`span_assign`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AssignConfig {
    pub max_len: usize,
    /// Minimum raw score (bias − w·prejudice) of a vector hit.
    pub accept: f32,
    /// Minimum lead of the winning domain over the runner-up on the same span, in double-centred residuals.
    pub margin: f32,
    /// Minimum double-centred residual of a vector hit (how far above its row and column means it stands).
    pub residual: f32,
    /// Hybrid scoring: a lexical (hashed n-gram) score above `lexical_floor` adds `lexical_weight × (score − floor)`
    /// to the semantic one before centring, so shared words break semantic near-ties (dense + sparse).
    pub lexical_weight: f32,
    pub lexical_floor: f32,
}

/// Characters of a span's content words (vector hits need at least two: a bare "이", "뭐" names nothing).
fn content_chars(words: &[Word]) -> usize {
    words
        .iter()
        .filter(|w| w.is_content())
        .map(|w| w.surface.chars().filter(|c| c.is_alphanumeric()).count())
        .sum()
}

/// What [`span_assign`] chose, plus the observations SDS keeps.
#[derive(Debug, Clone, Default)]
pub struct Assigned {
    pub hits: Vec<SpanHit>,
    /// Raw domain scores of each winning vector span (decay shapes for SDS).
    pub columns: Vec<Vec<f32>>,
    /// Ties the margin gate held back (domain index, rival index).
    pub held: Vec<(usize, usize)>,
}

/// NLP-NMS v2 span selection — logis-center's Plinko laws 2–3 over spans × domains:
///
/// 1. every span with a content word (up to `max_len` words) is scored against every domain in one batch;
/// 2. exact / prefix hits are kept as in v1 (they decide before vectors);
/// 3. the vector scores are **double-centred** (`zena_sds::matrix::double_center`): a domain that scores high on
///    every span, or a span that scores high on every domain (a bare "봐"), loses its pull;
/// 4. each domain claims at most one span and each span one domain, largest residual first, only with a lead of
///    `margin` over the runner-up (`exclusive_assign_adaptive`; near ties go to `allow_tie`, e.g. the SDS
///    confusion ledger) and a raw score of at least `accept`;
/// 5. overlapping winners are suppressed greedily by score, and a domain appears once — as in [`span_nms`].
pub fn span_assign<E: TextEmbedder, L: TextEmbedder>(
    words: &[Word],
    bank: &DomainBank<E>,
    lexical: Option<&DomainBank<L>>,
    cfg: &AssignConfig,
    allow_tie: impl Fn(usize, usize, zena_sds::TieKind) -> bool,
) -> Assigned {
    let mut ranges = Vec::new();
    for s in 0..words.len() {
        for e in (s + 1)..=(s + cfg.max_len).min(words.len()) {
            if bank.eligible(&words[s..e]) {
                ranges.push((s, e));
            }
        }
    }
    let mut out = Assigned::default();
    if ranges.is_empty() || bank.is_empty() {
        return out;
    }
    let spans: Vec<&[Word]> = ranges.iter().map(|&(s, e)| &words[s..e]).collect();
    let mut raw = bank.raw_scores(&spans);
    // Dense + sparse: the lexical bank (same dictionary, same order) adds its evidence above the floor.
    if let Some(lex) = lexical.filter(|l| l.len() == bank.len() && cfg.lexical_weight > 0.0) {
        for (col, lcol) in raw.iter_mut().zip(lex.raw_scores(&spans)) {
            for (d, l) in col.iter_mut().zip(lcol) {
                if d.hit == Hit::Vector {
                    d.score += cfg.lexical_weight * (l.score - cfg.lexical_floor).max(0.0);
                }
            }
        }
    }
    let text = |s: usize, e: usize| words[s..e].iter().map(|w| w.surface.as_str()).collect::<Vec<_>>().join(" ");

    let mut cands: Vec<SpanHit> = Vec::new();
    // Exact / prefix hits first; their cells stay out of the vector matrix.
    let mut matrix = vec![vec![zena_sds::matrix::ABSENT; ranges.len()]; bank.len()];
    for (j, scores) in raw.iter().enumerate() {
        let short = content_chars(spans[j]) < 2;
        for d in scores {
            if d.hit == Hit::Vector {
                if !short {
                    matrix[d.index][j] = d.score;
                }
            } else {
                let (s, e) = ranges[j];
                cands.push(SpanHit {
                    start: s,
                    end: e,
                    text: text(s, e),
                    index: d.index,
                    score: d.score,
                    hit: d.hit,
                });
            }
        }
    }
    let centred = zena_sds::matrix::double_center(&matrix);
    let (assigned, diags) = zena_sds::matrix::exclusive_assign_adaptive(&centred, cfg.residual, cfg.margin, allow_tie);
    for d in diags.iter().filter(|d| !d.accepted) {
        out.held.push((d.field, d.rival));
    }
    for (domain, a) in assigned.iter().enumerate() {
        let Some((j, _residual, _margin)) = *a else { continue };
        let raw_score = raw[j][domain].score;
        if raw_score < cfg.accept {
            continue;
        }
        let (s, e) = ranges[j];
        cands.push(SpanHit {
            start: s,
            end: e,
            text: text(s, e),
            index: domain,
            score: raw_score,
            hit: Hit::Vector,
        });
        out.columns.push(raw[j].iter().map(|d| d.score).collect());
    }
    cands.sort_by(|a, b| b.score.total_cmp(&a.score).then((b.end - b.start).cmp(&(a.end - a.start))));
    let mut winners: Vec<SpanHit> = Vec::new();
    for c in cands {
        let overlaps = winners.iter().any(|w| c.start < w.end && c.end > w.start);
        if !overlaps && !winners.iter().any(|w| w.index == c.index) {
            winners.push(c);
        }
    }
    winners.sort_by_key(|w| w.start);
    out.hits = winners;
    out
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
