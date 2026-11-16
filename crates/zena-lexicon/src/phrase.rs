//! Phrase lists of the bias.json node format.
//!
//! `bias` and `prejudice` are one string of phrases separated by `,` (also `\n`, `|`, and `/` unless the slash sits
//! inside a short abbreviation such as `w/o` or `n/a`), as in Qntology/logis-center's `bias.json` and noto-sandbox's
//! `world_lexicon.json` (port of `noto-search::lexicon::split_phrase_list` / `weighted_phrases`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Phrase {
    pub text: String,
    /// 1.0 for words; phrases made mostly of digits (ids, prices) weigh less, as in noto's lexicon.
    pub weight: f32,
}

/// Splits a phrase list, trims, drops empties and repeats (first occurrence kept).
pub fn split_phrase_list(raw: &str) -> Vec<String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, c) in chars.iter().enumerate() {
        let sep = match *c {
            ',' | '\n' | '|' => true,
            '/' => !is_abbreviation_slash(&chars, i),
            _ => false,
        };
        if sep {
            push_trimmed(&mut out, &cur);
            cur.clear();
        } else {
            cur.push(*c);
        }
    }
    push_trimmed(&mut out, &cur);
    let mut seen = std::collections::HashSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

fn push_trimmed(out: &mut Vec<String>, s: &str) {
    let t = s.trim();
    if !t.is_empty() {
        out.push(t.to_string());
    }
}

/// `w/o`, `n/a`, `I/O`: one or two letters (at least one alphabetic) on both sides of the slash.
fn is_abbreviation_slash(chars: &[char], i: usize) -> bool {
    let side = |range: &mut dyn Iterator<Item = usize>| {
        let (mut n, mut alpha) = (0usize, false);
        for j in range {
            let c = chars[j];
            if !c.is_ascii_alphanumeric() {
                break;
            }
            n += 1;
            alpha |= c.is_ascii_alphabetic();
        }
        (n, alpha)
    };
    let (l, la) = side(&mut (0..i).rev());
    let (r, ra) = side(&mut (i + 1..chars.len()));
    la && ra && (1..=2).contains(&l) && (1..=2).contains(&r)
}

/// The phrase list with weights: a quarter or more digits → 0.80, some digits → 0.95, else 1.0.
pub fn weighted_phrases(raw: &str) -> Vec<Phrase> {
    split_phrase_list(raw)
        .into_iter()
        .map(|text| {
            let compact: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
            let digits = compact.iter().filter(|c| c.is_ascii_digit()).count();
            let ratio = digits as f32 / compact.len().max(1) as f32;
            let weight = if ratio >= 0.25 {
                0.80
            } else if digits > 0 {
                0.95
            } else {
                1.0
            };
            Phrase { text, weight }
        })
        .collect()
}

/// Joins phrases back into the bias.json string form.
pub fn join_phrases<'a>(phrases: impl IntoIterator<Item = &'a str>) -> String {
    phrases.into_iter().collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_split_dedupe_and_keep_short_abbreviations() {
        assert_eq!(
            split_phrase_list("a, b ,a|c\nd / e, w/o sugar, n/a"),
            vec!["a", "b", "c", "d", "e", "w/o sugar", "n/a"]
        );
        assert_eq!(split_phrase_list(" , ,"), Vec::<String>::new());
        assert_eq!(split_phrase_list("종료해 줘, 앱 꺼 줘"), vec!["종료해 줘", "앱 꺼 줘"]);
    }

    #[test]
    fn digit_heavy_phrases_weigh_less() {
        let w = weighted_phrases("이겼다, 1등 먹었어요, 603145678912");
        assert_eq!(w.iter().map(|p| p.weight).collect::<Vec<_>>(), vec![1.0, 0.95, 0.80]);
        assert_eq!(
            join_phrases(w.iter().map(|p| p.text.as_str())),
            "이겼다, 1등 먹었어요, 603145678912"
        );
    }
}
