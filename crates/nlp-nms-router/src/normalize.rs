//! Utterance normalisation: wake-word removal, punctuation/whitespace cleanup, ASCII lowercase,
//! and per-word lemma candidates.

use crate::korean::{is_hangul, noun_lemma, predicate_stem};

pub const WAKE_WORDS: &[&str] = &["제나야", "제나", "라온 제나", "라온제나", "hey zena", "zena", "raon zena"];

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub surface: String,
    /// Noun lemma (josa peeled) and predicate stem (request ending stripped).
    pub lemma: String,
    pub stem: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Normalized {
    pub text: String,
    pub words: Vec<Word>,
    pub addressed: bool,
}

pub fn normalize(raw: &str) -> Normalized {
    let lower: String = raw.chars().map(|c| if c.is_ascii() { c.to_ascii_lowercase() } else { c }).collect();
    let mut text = lower.trim().to_string();
    let mut addressed = false;
    for w in WAKE_WORDS {
        if let Some(rest) = text.strip_prefix(w) {
            text = rest.trim_start_matches([',', '!', '~', ' ', '.']).to_string();
            addressed = true;
            break;
        }
    }
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || is_hangul(c) || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<Word> = cleaned
        .split_whitespace()
        .map(|w| Word {
            surface: w.to_string(),
            lemma: noun_lemma(w),
            stem: predicate_stem(w),
        })
        .collect();
    let text = words.iter().map(|w| w.surface.as_str()).collect::<Vec<_>>().join(" ");
    Normalized { text, words, addressed }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_wake_word_and_punctuation() {
        let n = normalize("제나야, 방금 그 보스 패턴 어떻게 피해?");
        assert!(n.addressed);
        assert_eq!(n.text, "방금 그 보스 패턴 어떻게 피해");
        assert_eq!(n.words.len(), 6);
        let n = normalize("Hey Zena, EXIT the app!");
        assert_eq!(n.text, "exit the app");
    }
}
