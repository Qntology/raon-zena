//! Small text helpers shared by the prompt slots, the claim extractor, the reference resolver and the ingress
//! guard. Crate-private: the public surface is the modules that use them.

/// `s` trimmed and cut to its first `n` characters (no ellipsis).
pub(crate) fn clip_chars(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>().trim_end().to_string()
    }
}

/// Whitespace runs collapsed to one space, trimmed.
pub(crate) fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Removes hash-like tokens — `0x` followed by three or more hex digits (premium-discount `hashId` addresses,
/// keys) and bare runs of 16+ hex digits mixing digits and letters — then collapses whitespace. A model shown a
/// raw hash invents a button name for it, so no prompt line may carry one.
pub(crate) fn strip_hex_hashes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let boundary_before = |i: usize| i == 0 || !chars[i - 1].is_ascii_alphanumeric();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '0' && boundary_before(i) && matches!(chars.get(i + 1), Some('x' | 'X')) {
            let mut j = i + 2;
            while j < chars.len() && chars[j].is_ascii_hexdigit() {
                j += 1;
            }
            if j - (i + 2) >= 3 {
                i = j;
                continue;
            }
        }
        if chars[i].is_ascii_hexdigit() && boundary_before(i) {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_hexdigit() {
                j += 1;
            }
            let run = &chars[i..j];
            let at_boundary = j == chars.len() || !chars[j].is_ascii_alphanumeric();
            if at_boundary && run.len() >= 16 && run.iter().any(char::is_ascii_digit) && run.iter().any(char::is_ascii_alphabetic) {
                i = j;
                continue;
            }
            out.extend(run);
            i = j;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    collapse_ws(&out)
}

/// Whether the last syllable of `word` ends in a consonant (받침). Digits are read in Sino-Korean
/// (일 · 삼 · 육 · 칠 · 팔 · 영 end in one); other characters count as open.
pub(crate) fn has_batchim(word: &str) -> bool {
    match word.trim_end().chars().last() {
        Some(c @ '\u{AC00}'..='\u{D7A3}') => !(c as u32 - 0xAC00).is_multiple_of(28),
        Some(c) if c.is_ascii_digit() => matches!(c, '0' | '1' | '3' | '6' | '7' | '8'),
        _ => false,
    }
}

/// `word` followed by the particle form its ending takes: `josa("평균", "과", "와")` → `평균과`.
pub(crate) fn josa(word: &str, after_consonant: &str, after_vowel: &str) -> String {
    format!("{word}{}", if has_batchim(word) { after_consonant } else { after_vowel })
}

/// `s` without the verdict words "비추천" / "추천" (for facts-only lines of records nobody judged yet).
pub(crate) fn drop_verdict_words(s: &str) -> String {
    collapse_ws(&s.replace("비추천", " ").replace("추천", " "))
}

/// Whether `haystack` contains any of `needles` (ASCII compared case-insensitively).
pub(crate) fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    let lower = haystack.to_lowercase();
    needles.iter().any(|n| lower.contains(&n.to_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_removed_but_words_and_short_numbers_stay() {
        let addr = format!("0x{}", "abc123".repeat(7));
        assert_eq!(strip_hex_hashes(&format!("성문 {addr} 열기")), "성문 열기");
        assert_eq!(strip_hex_hashes("데미지 2500 · cafe 버튼"), "데미지 2500 · cafe 버튼");
        assert_eq!(strip_hex_hashes("deadbeef00112233 끝"), "끝");
        assert_eq!(strip_hex_hashes("0x1 버튼"), "0x1 버튼", "too short to be a hash");
        assert_eq!(strip_hex_hashes("abcdefgh"), "abcdefgh");
    }

    #[test]
    fn particles_follow_the_final_consonant() {
        assert_eq!(josa("내 최고", "과", "와"), "내 최고와");
        assert_eq!(josa("타인 평균", "과", "와"), "타인 평균과");
        assert_eq!(josa("3", "은", "는"), "3은");
        assert_eq!(josa("2", "은", "는"), "2는");
    }

    #[test]
    fn clipping_matching_and_verdict_words() {
        assert_eq!(clip_chars("  가나다라  ", 2), "가나");
        assert!(contains_any("Sold Out", &["sold out"]));
        assert_eq!(drop_verdict_words("이 공략 추천, 저건 비추천"), "이 공략 , 저건");
    }
}
