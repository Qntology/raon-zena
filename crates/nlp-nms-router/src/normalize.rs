//! Utterance normalisation: wake-word removal, punctuation/whitespace cleanup, ASCII lowercase,
//! and per-word lemma candidates.

use crate::korean::{is_hangul, noun_lemma, predicate_stem};

pub const WAKE_WORDS: &[&str] = &["제나야", "제나", "라온 제나", "라온제나", "hey zena", "zena", "raon zena"];

/// What a word does in the utterance (set by an [`crate::analyze::Analyzer`]; [`Role::Content`] without one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
pub enum Role {
    /// Carries meaning: may name a domain.
    #[default]
    Content,
    /// Particles, conjunctions, pronouns, punctuation (Stanza ADP/PART/SCONJ/CCONJ/PRON/PUNCT).
    Function,
    /// An auxiliary or request predicate after the main one: "써 **봐**", "변신해 **줘**".
    Auxiliary,
    /// A negation of the predicate before it: "끄지 **마**", "**안** 가".
    Negation,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Word {
    pub surface: String,
    /// Noun lemma (josa peeled) and predicate stem (request ending stripped).
    pub lemma: String,
    pub stem: String,
    /// The analyzer's lemma / content stem (Stanza), another form for exact matching.
    pub alt: Option<String>,
    /// Universal POS and dependency relation when an analyzer gave them (empty otherwise).
    pub upos: String,
    pub deprel: String,
    pub role: Role,
    /// Inside the scope of a negation ("앱 끄지 마": 앱, 끄지 and 마).
    pub negated: bool,
}

impl Word {
    /// Whether the word can name a domain.
    pub fn is_content(&self) -> bool {
        self.role == Role::Content
    }
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
            ..Default::default()
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
