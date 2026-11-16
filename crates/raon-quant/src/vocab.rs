//! `kept_ids` construction for vocabulary row slicing (PRD §3.4).
//!
//! Raon-1-nano uses a GPT-2-style byte-level BPE. Every vocabulary string is mapped back to raw
//! bytes and kept when it is:
//! * a special/structural added token (`<|im_start|>`, `<tool_call>`, `<think>`, …; the unused
//!   `<|reserved_N|>` placeholders are dropped),
//! * one of the 256 single-byte tokens (byte fallback: any text stays producible),
//! * text made only of the target scripts — ASCII (English, digits, JSON/markup), Hangul
//!   syllables/Jamo, and common punctuation/symbol/emoji blocks — including tokens that start or
//!   end with a partial UTF-8 sequence that can only complete into such characters.
//!
//! Slicing only changes which rows are *resident*; [`crate::lmhead`] keeps decoding exact.

use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Inverse of GPT-2's `bytes_to_unicode`.
pub fn byte_decoder() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap(), b as u8)).collect()
}

const ALLOWED: &[(u32, u32)] = &[
    (0x0000, 0x007F),   // ASCII
    (0x00A0, 0x00BF),   // Latin-1 punctuation & symbols
    (0x00D7, 0x00D7),   // ×
    (0x00F7, 0x00F7),   // ÷
    (0x1100, 0x11FF),   // Hangul Jamo
    (0x2000, 0x23FF),   // punctuation, super/subscripts, currency, letterlike, arrows, math, technical
    (0x2460, 0x27BF),   // enclosed alnum, box drawing, shapes, misc symbols, dingbats
    (0x3000, 0x303F),   // CJK symbols & punctuation
    (0x3130, 0x318F),   // Hangul compatibility Jamo
    (0x3200, 0x32FF),   // enclosed CJK (㈜ ㉠ …)
    (0xA960, 0xA97F),   // Hangul Jamo extended-A
    (0xAC00, 0xD7A3),   // Hangul syllables
    (0xD7B0, 0xD7FF),   // Hangul Jamo extended-B
    (0xFE0F, 0xFE0F),   // emoji variation selector
    (0xFF00, 0xFFEF),   // half/full-width forms
    (0x1F300, 0x1FAFF), // emoji
];

pub fn allowed_char(c: u32) -> bool {
    ALLOWED.iter().any(|&(a, b)| (a..=b).contains(&c))
}

/// Can an incomplete UTF-8 prefix still complete into an allowed character?
fn allowed_tail(t: &[u8]) -> bool {
    let (len, mut lo, mut hi) = match t[0] {
        0xC2..=0xDF => (2, (t[0] & 0x1F) as u32, (t[0] & 0x1F) as u32),
        0xE0..=0xEF => (3, (t[0] & 0x0F) as u32, (t[0] & 0x0F) as u32),
        0xF0..=0xF4 => (4, (t[0] & 0x07) as u32, (t[0] & 0x07) as u32),
        _ => return false,
    };
    for i in 1..len {
        let (a, b) = match t.get(i) {
            Some(&c) => ((c & 0x3F) as u32, (c & 0x3F) as u32),
            None => (0, 0x3F),
        };
        lo = (lo << 6) | a;
        hi = (hi << 6) | b;
    }
    ALLOWED.iter().any(|&(a, b)| lo <= b && hi >= a)
}

/// Classify one token's raw bytes.
pub fn keep_bytes(b: &[u8]) -> bool {
    if b.len() <= 1 {
        return true;
    }
    let lead = b.iter().take(3).take_while(|&&x| (0x80..=0xBF).contains(&x)).count();
    let rest = &b[lead..];
    if rest.is_empty() {
        // Pure continuation fragment: cheap to keep when short (tails of split Hangul syllables).
        return b.len() <= 2;
    }
    let (body, tail) = match std::str::from_utf8(rest) {
        Ok(s) => (s, None),
        Err(e) if e.error_len().is_none() => (
            std::str::from_utf8(&rest[..e.valid_up_to()]).unwrap(),
            Some(&rest[e.valid_up_to()..]),
        ),
        Err(_) => return false,
    };
    body.chars().all(|c| allowed_char(c as u32)) && tail.is_none_or(allowed_tail)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VocabSlice {
    pub vocab_size: usize,
    /// Kept token ids, ascending.
    pub kept: Vec<u32>,
    /// Dropped token ids, ascending.
    pub cold: Vec<u32>,
    pub stats: BTreeMap<String, usize>,
}

impl VocabSlice {
    /// Everything kept (slicing disabled).
    pub fn full(vocab_size: usize) -> Self {
        VocabSlice {
            vocab_size,
            kept: (0..vocab_size as u32).collect(),
            cold: vec![],
            stats: BTreeMap::new(),
        }
    }
}

/// Build `kept_ids` from a Hugging Face `tokenizer.json`. Ids past the tokenizer's range (padding
/// rows of the embedding) are dropped.
///
/// `keep_top`: additionally cap the script-filtered text tokens to the `N` lowest ids. Byte-level
/// BPE assigns ids in merge order, so low ids are the frequent tokens. Specials and the 256 byte
/// tokens are always kept. Raon-1-nano's vocabulary is ~96 % Korean/English already, so without
/// this cap slicing removes only ~4 % of the rows.
pub fn kept_ids_from_tokenizer(path: &Path, vocab_size: usize, keep_top: Option<usize>) -> Result<VocabSlice> {
    let t: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)?;
    let dec = byte_decoder();
    let mut keep = vec![false; vocab_size];
    let mut known = vec![false; vocab_size];
    let mut text_ids: Vec<usize> = Vec::new();
    let mut stats = BTreeMap::<String, usize>::new();
    for a in t["added_tokens"].as_array().context("added_tokens")? {
        let id = a["id"].as_u64().context("id")? as usize;
        if id >= vocab_size {
            continue;
        }
        known[id] = true;
        let content = a["content"].as_str().unwrap_or("");
        let reserved = content.starts_with("<|reserved_");
        keep[id] = !reserved;
        *stats
            .entry(if reserved { "special_reserved_dropped" } else { "special_kept" }.into())
            .or_default() += 1;
    }
    for (tok, id) in t["model"]["vocab"].as_object().context("model.vocab")? {
        let id = id.as_u64().context("vocab id")? as usize;
        if id >= vocab_size || known[id] {
            continue;
        }
        known[id] = true;
        let bytes: Option<Vec<u8>> = tok.chars().map(|c| dec.get(&c).copied()).collect();
        let k = bytes.as_deref().is_some_and(keep_bytes);
        let class = match &bytes {
            Some(b) if b.len() == 1 => "byte_fallback",
            Some(_) if k => "text_kept",
            _ => "text_dropped",
        };
        *stats.entry(class.into()).or_default() += 1;
        keep[id] = k;
        if k && class == "text_kept" {
            text_ids.push(id);
        }
    }
    if let Some(n) = keep_top {
        text_ids.sort_unstable();
        for &id in text_ids.iter().skip(n) {
            keep[id] = false;
        }
        stats.insert("text_beyond_keep_top".into(), text_ids.len().saturating_sub(n));
    }
    stats.insert("unassigned_rows_dropped".into(), known.iter().filter(|k| !**k).count());
    let kept: Vec<u32> = (0..vocab_size as u32).filter(|&i| keep[i as usize]).collect();
    let cold: Vec<u32> = (0..vocab_size as u32).filter(|&i| !keep[i as usize]).collect();
    stats.insert("kept".into(), kept.len());
    stats.insert("cold".into(), cold.len());
    Ok(VocabSlice {
        vocab_size,
        kept,
        cold,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_decoder_is_a_bijection() {
        let d = byte_decoder();
        assert_eq!(d.len(), 256);
        assert_eq!(d[&'Ġ'], b' ');
        assert_eq!(d[&'Ċ'], b'\n');
        assert_eq!(d[&'a'], b'a');
    }

    #[test]
    fn keeps_target_scripts_and_drops_others() {
        assert!(keep_bytes("안녕하세요".as_bytes()));
        assert!(keep_bytes(" hello".as_bytes()));
        assert!(keep_bytes("{\"angle_deg\":".as_bytes()));
        assert!(keep_bytes("→".as_bytes()));
        assert!(keep_bytes("😀".as_bytes()));
        assert!(!keep_bytes("中文".as_bytes()));
        assert!(!keep_bytes("Привет".as_bytes()));
        assert!(!keep_bytes("ありがとう".as_bytes()));
        // "가" = EA B0 80: an incomplete prefix of a Hangul syllable is kept…
        assert!(keep_bytes(&[b' ', 0xEA, 0xB0]));
        // …but a prefix that can only become CJK ideographs (E4–E9 lead) is not.
        assert!(!keep_bytes(&[b' ', 0xE4, 0xB8]));
        // A Hangul syllable preceded by the tail of a previous character.
        let mut v = vec![0x80];
        v.extend_from_slice("요".as_bytes());
        assert!(keep_bytes(&v));
        assert!(keep_bytes(&[0xB0]));
    }

    #[test]
    fn builds_slice_from_tokenizer_json() {
        let dir = std::env::temp_dir().join(format!("raonq-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tokenizer.json");
        let tok = serde_json::json!({
            "added_tokens": [{"id": 5, "content": "<|im_start|>"}, {"id": 6, "content": "<|reserved_0|>"}],
            "model": {"vocab": {"a": 0, "ĠìķĪ": 1, "ä¸Ń": 2, "Ġhello": 3, "ì": 4}}
        });
        std::fs::write(&p, tok.to_string()).unwrap();
        let s = kept_ids_from_tokenizer(&p, 8, None).unwrap();
        assert_eq!(s.kept, vec![0, 1, 3, 4, 5]);
        assert_eq!(s.cold, vec![2, 6, 7]);
        // keep_top=1 keeps the lowest-id text token (1) plus bytes/specials.
        let s = kept_ids_from_tokenizer(&p, 8, Some(1)).unwrap();
        assert_eq!(s.kept, vec![0, 1, 4, 5]);
        std::fs::remove_dir_all(dir).ok();
    }
}
