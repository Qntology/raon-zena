//! Text embedding.
//!
//! [`HashedNgramEmbedder`] is a deterministic, model-free embedder: signed feature hashing of
//! word lemmas, character bi/tri-grams and Hangul **jamo** tri-grams into 384 dimensions,
//! L2-normalised. It is lexical, not semantic — paraphrases that share no surface form score
//! low — so the router keeps it behind [`TextEmbedder`] and a neural encoder (e.g. pooled
//! Raon-1-nano hidden states) can replace it without touching the NMS code.

use crate::korean::{is_hangul, jamo};
use crate::normalize::normalize;

pub trait TextEmbedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

#[derive(Debug, Clone)]
pub struct HashedNgramEmbedder {
    pub dim: usize,
}

impl Default for HashedNgramEmbedder {
    fn default() -> Self {
        Self { dim: 384 }
    }
}

fn fnv1a(s: &str, salt: u8) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ salt as u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl HashedNgramEmbedder {
    fn add(&self, v: &mut [f32], feat: &str, salt: u8, w: f32) {
        let h = fnv1a(feat, salt);
        let i = (h % self.dim as u64) as usize;
        v[i] += if h >> 63 == 0 { w } else { -w };
    }

    fn char_ngrams(&self, v: &mut [f32], s: &str, ns: &[usize], salt: u8, w: f32) {
        let cs: Vec<char> = std::iter::once('^').chain(s.chars()).chain(std::iter::once('$')).collect();
        for &n in ns {
            if cs.len() < n {
                continue;
            }
            for win in cs.windows(n) {
                let g: String = win.iter().collect();
                self.add(v, &g, salt + n as u8, w);
            }
        }
    }
}

impl TextEmbedder for HashedNgramEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let n = normalize(text);
        let mut v = vec![0f32; self.dim];
        for w in &n.words {
            self.add(&mut v, &w.lemma, 1, 1.0);
            if w.stem != w.lemma {
                self.add(&mut v, &w.stem, 1, 0.8);
            }
            let base = if w.stem.chars().count() < w.lemma.chars().count() {
                &w.stem
            } else {
                &w.lemma
            };
            self.char_ngrams(&mut v, base, &[2, 3], 10, 0.5);
            if base.chars().any(is_hangul) {
                self.char_ngrams(&mut v, &jamo(base), &[3], 20, 0.3);
            }
        }
        for pair in n.words.windows(2) {
            self.add(&mut v, &format!("{} {}", pair[0].lemma, pair[1].lemma), 30, 0.6);
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn similar_requests_are_closer_than_unrelated() {
        let e = HashedNgramEmbedder::default();
        let a = e.embed("보스 패턴 어떻게 피해");
        let b = e.embed("제나야 보스 패턴은 어떻게 피하지?");
        let c = e.embed("인벤토리에서 물약 찾아줘");
        assert!(cosine(&a, &b) > 0.5, "{}", cosine(&a, &b));
        assert!(cosine(&a, &b) > cosine(&a, &c) + 0.3);
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-5);
        assert_eq!(cosine(&a, &vec![0.0; 384]), 0.0);
    }
}
