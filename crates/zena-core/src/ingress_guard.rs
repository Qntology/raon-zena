//! The front gate for every text the player types, says or drops (`contracts/privacy_shield_v1.json`, PS-12).
//! Before a line reaches the router, the dialogue, Raon-1-nano, the outbox or the lexicon learner it is checked
//! here, in place, without a model:
//!
//! * **hard secrets** — a Luhn-valid payment card number, a Korean resident / alien registration number, a
//!   BIP-39 seed phrase (12+ consecutive list words), a 256-bit private key (`0x` + 64 hex, or a bare 64-hex
//!   token), a password typed right after its keyword — block the whole line: the input buffer is overwritten
//!   with zeros and only the [`SecretKind`] comes back (never the digits).
//! * **soft personal data** — e-mail, phone number, the user name in `C:\Users\<name>\`, long order numbers —
//!   is masked; the masked copy goes on (but never into the lexicon learner) and the original buffer is zeroed.
//! * a public address (`0x` + 40 hex), request ids and ordinary numbers ("데미지 2500") pass untouched.
//!
//! zena-core has no word list and no crypto crate: the BIP-39 English list comes in as a word predicate (the app
//! wires premium-discount-identity's list), and zeroing is done by hand ([`zeroize_string`]).

use crate::tools::{BodyMotion, CharacterEmote, EmotionState, ZenaAvatarArgs};
use serde::{Deserialize, Serialize};

/// Consecutive BIP-39 words from which a run is treated as a seed phrase (phrases have 12, 15, 18, 21 or 24).
pub const SEED_PHRASE_MIN_WORDS: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecretKind {
    PaymentCardLuhn,
    WalletSeedPhrase,
    WalletPrivateKeyHex,
    GovernmentIdNumber,
    ExplicitPasswordToken,
}

impl SecretKind {
    /// Zena's fixed reply: what kind of thing it was, never what it said.
    pub fn warning_line(self) -> &'static str {
        match self {
            SecretKind::PaymentCardLuhn => "앗, 카드번호 같은 건 여기 적으면 안 돼! 방금 건 바로 지웠어.",
            SecretKind::WalletSeedPhrase => "지갑 복구 단어는 누구한테도 보여 주면 안 돼! 방금 건 바로 지웠어.",
            SecretKind::WalletPrivateKeyHex => "지갑 개인키는 절대 적지 마! 방금 건 바로 지웠어.",
            SecretKind::GovernmentIdNumber => "주민등록번호 같은 건 적지 마! 방금 건 바로 지웠어.",
            SecretKind::ExplicitPasswordToken => "비밀번호는 나한테도 말하지 마! 방금 건 바로 지웠어.",
        }
    }
}

/// What may go on after the gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngressVerdict {
    /// Nothing sensitive: route it normally (lexicon learning allowed).
    Clean(String),
    /// Contact / path identifiers masked: coach on the masked copy, but never learn from it.
    Scrubbed { sanitized_text: String, allow_lexicon_learn: bool },
    /// A hard secret: the input was zeroed; nothing reaches the LLM, the dialogue, the outbox or the lexicon.
    HardBlocked { kind: SecretKind },
}

impl IngressVerdict {
    /// The text the router / dialogue / LLM / outbox may see (`None` when blocked).
    pub fn routable_text(&self) -> Option<&str> {
        match self {
            IngressVerdict::Clean(s) => Some(s),
            IngressVerdict::Scrubbed { sanitized_text, .. } => Some(sanitized_text),
            IngressVerdict::HardBlocked { .. } => None,
        }
    }

    /// Whether the lexicon learner may record the line (only clean lines).
    pub fn allows_lexicon_learn(&self) -> bool {
        match self {
            IngressVerdict::Clean(_) => true,
            IngressVerdict::Scrubbed { allow_lexicon_learn, .. } => *allow_lexicon_learn,
            IngressVerdict::HardBlocked { .. } => false,
        }
    }

    pub fn is_blocked(&self) -> bool {
        matches!(self, IngressVerdict::HardBlocked { .. })
    }

    /// The model-free Lv2 reaction to a block: HIDE_CROUCH with a fixed line that never echoes the input.
    pub fn block_reaction(&self) -> Option<ZenaAvatarArgs> {
        let IngressVerdict::HardBlocked { kind } = self else {
            return None;
        };
        Some(ZenaAvatarArgs {
            gaze_target: None,
            emotion_state: EmotionState::Worried,
            body_motion: BodyMotion::HideCrouch,
            character_emote: Some(CharacterEmote::SweatPanic),
            speech_text: Some(kind.warning_line().to_string()),
        })
    }
}

/// Overwrites the string's bytes with zeros before its buffer is released, and leaves it empty. Copies made
/// earlier by reallocation are out of reach; callers should not grow sensitive buffers more than needed.
pub fn zeroize_string(s: &mut String) {
    let mut bytes = std::mem::take(s).into_bytes();
    bytes.iter_mut().for_each(|b| *b = 0);
    // Keeps the stores: the optimiser must assume the buffer is read here.
    std::hint::black_box(&mut bytes);
}

/// Luhn mod-10 checksum over decimal digits (0..=9 each).
pub fn luhn_valid(digits: &[u8]) -> bool {
    if digits.is_empty() {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            let d = u32::from(d);
            if i % 2 == 1 {
                let x = d * 2;
                if x > 9 {
                    x - 9
                } else {
                    x
                }
            } else {
                d
            }
        })
        .sum();
    sum.is_multiple_of(10)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PiiKind {
    Hard(SecretKind),
    Email,
    Phone,
    UserPath,
    LongDigits,
}

/// A byte range of `text` and what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PiiSpan {
    pub start: usize,
    pub end: usize,
    pub kind: PiiKind,
}

struct DigitRun {
    start: usize,
    end: usize,
    digits: Vec<u8>,
    groups: Vec<usize>,
    /// Every separator is a hyphen (or there is none).
    hyphens_only: bool,
}

/// Runs of ASCII digits joined by single spaces or hyphens, on ASCII-word boundaries.
fn digit_runs(text: &str) -> Vec<DigitRun> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !(b[i].is_ascii_digit() && (i == 0 || !b[i - 1].is_ascii_alphanumeric())) {
            i += 1;
            continue;
        }
        let start = i;
        let (mut digits, mut groups, mut hyphens_only) = (Vec::new(), Vec::new(), true);
        loop {
            let g0 = digits.len();
            while i < b.len() && b[i].is_ascii_digit() {
                digits.push(b[i] - b'0');
                i += 1;
            }
            groups.push(digits.len() - g0);
            if i + 1 < b.len() && matches!(b[i], b' ' | b'-') && b[i + 1].is_ascii_digit() {
                hyphens_only &= b[i] == b'-';
                i += 1;
                continue;
            }
            break;
        }
        if i == b.len() || !b[i].is_ascii_alphanumeric() {
            out.push(DigitRun {
                start,
                end: i,
                digits,
                groups,
                hyphens_only,
            });
        }
    }
    out
}

fn is_rrn(d: &[u8]) -> bool {
    let month = d[2] * 10 + d[3];
    let day = d[4] * 10 + d[5];
    (1..=12).contains(&month) && (1..=31).contains(&day) && (1..=8).contains(&d[6])
}

fn card_grouping(groups: &[usize]) -> bool {
    match groups {
        [_] => true,
        [4, 6, 5] | [4, 6, 4] => true,
        [init @ .., last] => init.iter().all(|g| *g == 4) && (1..=4).contains(last),
        [] => false,
    }
}

fn classify_run(r: &DigitRun) -> Option<PiiKind> {
    let n = r.digits.len();
    if n == 13 && (r.groups == [13] || r.groups == [6, 7]) && is_rrn(&r.digits) {
        return Some(PiiKind::Hard(SecretKind::GovernmentIdNumber));
    }
    if (13..=19).contains(&n)
        && card_grouping(&r.groups)
        && (2..=6).contains(&r.digits[0])
        && r.digits.iter().any(|d| *d != r.digits[0])
        && luhn_valid(&r.digits)
    {
        return Some(PiiKind::Hard(SecretKind::PaymentCardLuhn));
    }
    if (9..=11).contains(&n) && r.hyphens_only && r.digits[0] == 0 && (1..=7).contains(&r.digits[1]) {
        return Some(PiiKind::Phone);
    }
    if r.hyphens_only && n >= 9 {
        return Some(PiiKind::LongDigits);
    }
    None
}

/// `0x` + 64 hex (private key; `0x` + 40 hex addresses pass) and bare 64-hex tokens mixing digits and letters.
fn hex_secrets(text: &str) -> Vec<PiiSpan> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        if boundary && b[i] == b'0' && i + 1 < b.len() && (b[i + 1] == b'x' || b[i + 1] == b'X') {
            let mut j = i + 2;
            while j < b.len() && b[j].is_ascii_hexdigit() {
                j += 1;
            }
            let after = j == b.len() || !b[j].is_ascii_alphanumeric();
            if after && j - (i + 2) == 64 {
                out.push(PiiSpan {
                    start: i,
                    end: j,
                    kind: PiiKind::Hard(SecretKind::WalletPrivateKeyHex),
                });
            }
            i = j;
            continue;
        }
        if boundary && b[i].is_ascii_hexdigit() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_hexdigit() {
                j += 1;
            }
            let run = &b[i..j];
            let after = j == b.len() || !b[j].is_ascii_alphanumeric();
            if after && run.len() == 64 && run.iter().any(u8::is_ascii_digit) && run.iter().any(u8::is_ascii_alphabetic) {
                out.push(PiiSpan {
                    start: i,
                    end: j,
                    kind: PiiKind::Hard(SecretKind::WalletPrivateKeyHex),
                });
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    out
}

fn emails(text: &str) -> Vec<PiiSpan> {
    let b = text.as_bytes();
    let local = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'%' | b'+' | b'-');
    let domain = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-');
    let mut out = Vec::new();
    for (p, _) in text.match_indices('@') {
        let mut s = p;
        while s > 0 && local(b[s - 1]) {
            s -= 1;
        }
        let mut e = p + 1;
        while e < b.len() && domain(b[e]) {
            e += 1;
        }
        while e > p + 1 && b[e - 1] == b'.' {
            e -= 1;
        }
        let dom = &text[p + 1..e];
        let tld_ok = dom
            .rsplit('.')
            .next()
            .is_some_and(|t| t.len() >= 2 && t.bytes().all(|c| c.is_ascii_alphabetic()));
        if s < p && dom.contains('.') && tld_ok {
            out.push(PiiSpan {
                start: s,
                end: e,
                kind: PiiKind::Email,
            });
        }
    }
    out
}

/// The user-name component of `C:\Users\<name>\…` / `C:/Users/<name>/…` and of a path starting `/Users/<name>/`.
fn user_paths(text: &str) -> Vec<PiiSpan> {
    let lower = text.to_ascii_lowercase();
    let b = text.as_bytes();
    let mut out = Vec::new();
    for pat in ["\\users\\", "/users/"] {
        for (pos, _) in lower.match_indices(pat) {
            let drive = pos >= 2 && b[pos - 1] == b':' && b[pos - 2].is_ascii_alphabetic();
            let rooted = pat == "/users/" && (pos == 0 || matches!(b[pos - 1], b' ' | b'\t' | b'\n' | b'"' | b'\'' | b'('));
            if !(drive || rooted) {
                continue;
            }
            let s = pos + pat.len();
            let mut e = s;
            while e < b.len() && !matches!(b[e], b'\\' | b'/' | b' ' | b'\t' | b'\n' | b'"' | b'\'') {
                e += 1;
            }
            let name = text[s..e].to_ascii_lowercase();
            if !name.is_empty() && !matches!(name.as_str(), "public" | "default" | "[user]") {
                out.push(PiiSpan {
                    start: s,
                    end: e,
                    kind: PiiKind::UserPath,
                });
            }
        }
    }
    out
}

const KO_SECRET_KEYS: [&str; 6] = ["비밀번호", "비번", "패스워드", "암호", "인증번호", "핀번호"];
const EN_SECRET_KEYS: [&str; 7] = ["password", "passwd", "passcode", "pwd", "pw", "pin", "otp"];
const KEY_FILLERS: [&str; 8] = ["is", ":", "=", "：", "는", "은", "-", "→"];

fn strip_key_tail(rest: &str) -> &str {
    let mut r = rest;
    for p in ["는요", "는", "은", "가", "이", "를", "을", "도", "요", ":", "=", "："] {
        if let Some(x) = r.strip_prefix(p) {
            r = x;
            break;
        }
    }
    r.trim_start_matches([':', '=', '：'])
}

/// The token right after a secret keyword, as a password would be typed.
fn looks_secret(token: &str) -> bool {
    let t = token.trim_end_matches(['.', ',', '!', '?', ')', '"', '\'']);
    let ascii_graphic = !t.is_empty() && t.chars().all(|c| c.is_ascii_graphic());
    let digit = t.chars().any(|c| c.is_ascii_digit());
    let letter = t.chars().any(|c| c.is_ascii_alphabetic());
    let symbol = t.chars().any(|c| c.is_ascii_punctuation());
    ascii_graphic && t.chars().count() >= 4 && (digit || (letter && symbol))
}

fn password_tokens(text: &str) -> Vec<PiiSpan> {
    let mut tokens: Vec<(usize, usize)> = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (c.is_whitespace(), start) {
            (true, Some(s)) => {
                tokens.push((s, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        tokens.push((s, text.len()));
    }
    let mut out = Vec::new();
    for (k, &(s, e)) in tokens.iter().enumerate() {
        let tok = &text[s..e];
        let lower = tok.to_ascii_lowercase();
        let rest = if let Some(key) = KO_SECRET_KEYS.iter().find(|key| tok.starts_with(**key)) {
            Some(strip_key_tail(&tok[key.len()..]))
        } else {
            let head_len = lower.bytes().take_while(u8::is_ascii_alphabetic).count();
            EN_SECRET_KEYS
                .contains(&&lower[..head_len])
                .then(|| strip_key_tail(&tok[head_len..]))
        };
        let Some(rest) = rest else { continue };
        let found = if !rest.is_empty() {
            looks_secret(rest).then_some(e)
        } else {
            tokens[k + 1..]
                .iter()
                .take(3)
                .find(|(a, b)| !KEY_FILLERS.contains(&&text[*a..*b]))
                .and_then(|&(a, b)| looks_secret(&text[a..b]).then_some(b))
        };
        if let Some(end) = found {
            out.push(PiiSpan {
                start: s,
                end,
                kind: PiiKind::Hard(SecretKind::ExplicitPasswordToken),
            });
        }
    }
    out
}

/// Runs of at least [`SEED_PHRASE_MIN_WORDS`] list words; numbering and punctuation between them ("1. abandon
/// 2. ability") do not break a run, any other word does.
fn seed_phrases(text: &str, is_word: &dyn Fn(&str) -> bool) -> Vec<PiiSpan> {
    let b = text.as_bytes();
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() {
            let s = i;
            while i < b.len() && b[i].is_ascii_alphabetic() {
                i += 1;
            }
            words.push((s, i));
        } else {
            i += 1;
        }
    }
    let joins = |sep: &str| {
        sep.chars()
            .all(|c| c.is_whitespace() || c.is_ascii_digit() || ".,;:-()[]#".contains(c))
    };
    let mut out = Vec::new();
    let mut run: Option<(usize, usize)> = None; // (first word index, length)
    let flush = |run: Option<(usize, usize)>, out: &mut Vec<PiiSpan>| {
        if let Some((first, len)) = run.filter(|(_, len)| *len >= SEED_PHRASE_MIN_WORDS) {
            out.push(PiiSpan {
                start: words[first].0,
                end: words[first + len - 1].1,
                kind: PiiKind::Hard(SecretKind::WalletSeedPhrase),
            });
        }
    };
    for (k, &(s, e)) in words.iter().enumerate() {
        if !is_word(&text[s..e].to_ascii_lowercase()) {
            flush(run.take(), &mut out);
            continue;
        }
        run = match run {
            Some((first, len)) if joins(&text[words[k - 1].1..s]) => Some((first, len + 1)),
            other => {
                flush(other, &mut out);
                Some((k, 1))
            }
        };
    }
    flush(run, &mut out);
    out
}

#[derive(Clone, Copy)]
enum SeedCheck<'a> {
    None,
    Word(&'a dyn Fn(&str) -> bool),
    Span(&'a dyn Fn(&str) -> Option<(usize, usize)>),
}

/// Every sensitive span of `text`, sorted by start (longer first on ties).
pub(crate) fn find_pii(text: &str, is_seed_word: Option<&dyn Fn(&str) -> bool>) -> Vec<PiiSpan> {
    find_pii_with(
        text,
        match is_seed_word {
            Some(f) => SeedCheck::Word(f),
            None => SeedCheck::None,
        },
    )
}

fn find_pii_with(text: &str, seed_check: SeedCheck<'_>) -> Vec<PiiSpan> {
    let mut spans: Vec<PiiSpan> = digit_runs(text)
        .iter()
        .filter_map(|r| {
            classify_run(r).map(|kind| PiiSpan {
                start: r.start,
                end: r.end,
                kind,
            })
        })
        .collect();
    spans.extend(hex_secrets(text));
    spans.extend(emails(text));
    spans.extend(user_paths(text));
    spans.extend(password_tokens(text));
    match seed_check {
        SeedCheck::None => {}
        SeedCheck::Word(f) => spans.extend(seed_phrases(text, f)),
        SeedCheck::Span(det) => {
            if let Some((start, end)) = det(text).filter(|&(s, e)| s < e && e <= text.len()) {
                spans.push(PiiSpan {
                    start,
                    end,
                    kind: PiiKind::Hard(SecretKind::WalletSeedPhrase),
                });
            }
        }
    }
    spans.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
    spans
}

fn mask_token(kind: PiiKind) -> &'static str {
    match kind {
        PiiKind::Email => "[이메일]",
        PiiKind::Phone => "[전화번호]",
        PiiKind::UserPath => "[USER]",
        PiiKind::LongDigits | PiiKind::Hard(_) => "####",
    }
}

/// The ingress gate with an optional BIP-39 word predicate or full span detector.
#[derive(Clone, Copy)]
pub struct IngressPiiScrubber<'a> {
    seed_check: SeedCheck<'a>,
}

impl std::fmt::Debug for IngressPiiScrubber<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IngressPiiScrubber")
            .field("seed_list", &!matches!(self.seed_check, SeedCheck::None))
            .finish()
    }
}

impl<'a> IngressPiiScrubber<'a> {
    /// `is_seed_word`: whether a lowercase word is in the BIP-39 English list.
    pub fn new(is_seed_word: &'a dyn Fn(&str) -> bool) -> Self {
        Self {
            seed_check: SeedCheck::Word(is_seed_word),
        }
    }

    /// Uses a full seed-phrase span detector (e.g. `premium_discount_identity::detect_seed_phrase_span` or a
    /// checksum / context-aware checker that avoids false positives on plain English gaming words).
    pub fn with_seed_detector(detector: &'a dyn Fn(&str) -> Option<(usize, usize)>) -> Self {
        Self {
            seed_check: SeedCheck::Span(detector),
        }
    }

    /// Without a word list seed phrases are not detected (everything else is).
    pub fn without_seed_list() -> Self {
        Self {
            seed_check: SeedCheck::None,
        }
    }

    /// Read-only check: the first hard secret in `text`, if any.
    pub fn detect(&self, text: &str) -> Option<SecretKind> {
        find_pii_with(text, self.seed_check).into_iter().find_map(|s| match s.kind {
            PiiKind::Hard(k) => Some(k),
            _ => None,
        })
    }

    /// Checks `raw` in place. A hard secret zeroes the buffer and blocks; soft data is masked into a copy and the
    /// buffer zeroed; a clean line is moved out of the buffer unchanged. `raw` is left empty in every case.
    pub fn inspect_and_scrub(&self, raw: &mut String) -> IngressVerdict {
        let spans = find_pii_with(raw, self.seed_check);
        if let Some(kind) = spans.iter().find_map(|s| match s.kind {
            PiiKind::Hard(k) => Some(k),
            _ => None,
        }) {
            zeroize_string(raw);
            return IngressVerdict::HardBlocked { kind };
        }
        if spans.is_empty() {
            return IngressVerdict::Clean(std::mem::take(raw));
        }
        let mut sanitized = String::with_capacity(raw.len());
        let mut at = 0;
        for s in &spans {
            if s.start < at {
                continue;
            }
            sanitized.push_str(&raw[at..s.start]);
            sanitized.push_str(mask_token(s.kind));
            at = s.end;
        }
        sanitized.push_str(&raw[at..]);
        zeroize_string(raw);
        IngressVerdict::Scrubbed {
            sanitized_text: sanitized,
            allow_lexicon_learn: false,
        }
    }
}

/// [`IngressPiiScrubber::inspect_and_scrub`] with a BIP-39 word predicate.
pub fn scrub_or_block_ingress(raw: &mut String, is_seed_word: &dyn Fn(&str) -> bool) -> IngressVerdict {
    IngressPiiScrubber::new(is_seed_word).inspect_and_scrub(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reaction::ReactionLevel;
    use crate::tools::WindowLayerMode;
    use crate::window::{PrivacyAction, PrivacyEtiquette, ShutdownPhase, WindowState, HOVER_ALPHA, WS_EX_NOACTIVATE, WS_EX_TRANSPARENT};
    use std::time::Duration;

    /// The first 24 words of the BIP-39 English list.
    const STUB: [&str; 24] = [
        "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse", "access", "accident", "account",
        "accuse", "achieve", "acid", "acoustic", "acquire", "across", "act", "action", "actor", "actress", "actual",
    ];

    fn stub(w: &str) -> bool {
        STUB.contains(&w)
    }

    fn check(s: &str) -> IngressVerdict {
        scrub_or_block_ingress(&mut s.to_string(), &stub)
    }

    #[test]
    fn ps12_lock_forces_click_through_blocks_secrets_and_stops_capture_at_farewell() {
        // InteractivePet, then a privacy lock starts.
        let st = WindowState {
            mode: WindowLayerMode::InteractivePet,
            ..WindowState::default()
        };
        assert_eq!(st.ex_style() & WS_EX_TRANSPARENT, 0, "pet mode catches clicks normally");
        let mut etiquette = PrivacyEtiquette::default();
        let actions = etiquette.set_locked(true, 1_000);
        for a in [
            PrivacyAction::CancelGeneration,
            PrivacyAction::ClearSpeech,
            PrivacyAction::HideCrouch,
            PrivacyAction::ApplyStyle,
        ] {
            assert!(actions.contains(&a), "{actions:?}");
        }
        let style = etiquette.ex_style(&st);
        assert_eq!(style & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE), WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
        assert_eq!(etiquette.alpha(&st, false), HOVER_ALPHA);
        assert_eq!(etiquette.alpha(&st, true), 76);
        // The lock cleared the bubble (speaking = false) and cancelled the Lv3 stream: 15 FPS.
        assert_eq!(
            etiquette.frame_interval(ReactionLevel::Lv3Coaching, false, true),
            Some(Duration::from_micros(66_666))
        );
        assert_eq!(st.mode, WindowLayerMode::InteractivePet, "the mode itself is kept for the way back");

        // Chat box input while locked.
        let mut card = String::from("결제 카드 4111 1111 1111 1111 맞아?");
        let v = scrub_or_block_ingress(&mut card, &stub);
        assert_eq!(
            v,
            IngressVerdict::HardBlocked {
                kind: SecretKind::PaymentCardLuhn
            }
        );
        assert!(
            card.is_empty() && card.capacity() == 0,
            "the original buffer is zeroed and released"
        );
        assert_eq!(v.routable_text(), None, "nothing for the router / dialogue / LLM / outbox");
        assert!(!v.allows_lexicon_learn());
        assert!(!format!("{v:?}").contains("4111"), "the verdict carries the kind only");
        let r = v.block_reaction().unwrap();
        assert_eq!(r.body_motion, BodyMotion::HideCrouch);
        assert!(!r.speech_text.unwrap().chars().any(|c| c.is_ascii_digit()));

        let mut seed = STUB[..12].join(" ");
        let v = scrub_or_block_ingress(&mut seed, &stub);
        assert_eq!(
            v,
            IngressVerdict::HardBlocked {
                kind: SecretKind::WalletSeedPhrase
            }
        );
        assert!(seed.is_empty());

        let address = format!("내 주소 0x{} 맞지?", "a1b2c3d4e5".repeat(4));
        let mut buf = address.clone();
        assert_eq!(
            scrub_or_block_ingress(&mut buf, &stub),
            IngressVerdict::Clean(address),
            "a public address passes"
        );

        // ExitApp: capture stops at the first millisecond of the farewell.
        assert!(ShutdownPhase::Running.allows_capture());
        let farewell = ShutdownPhase::begin(5_000);
        assert!(!farewell.allows_capture());
        assert!(!farewell.step(5_000).allows_capture());
        assert!(!ShutdownPhase::StopCapture.allows_capture());
    }

    #[test]
    fn cards_need_luhn_a_card_prefix_and_card_grouping() {
        assert_eq!(
            check("4111-1111-1111-1111"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::PaymentCardLuhn
            }
        );
        assert_eq!(
            check("5500000000000004"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::PaymentCardLuhn
            }
        );
        assert_eq!(
            check("아멕스 3782 822463 10005"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::PaymentCardLuhn
            }
        );
        assert!(matches!(check("4111 1111 1111 1112"), IngressVerdict::Clean(_)), "Luhn fails");
        assert!(
            matches!(check("체력 1200 1500 1800 2000"), IngressVerdict::Clean(_)),
            "stats are not a card"
        );
        assert!(matches!(check("데미지 2500 들어갔어"), IngressVerdict::Clean(_)));
        assert!(luhn_valid(&[7, 9, 9, 2, 7, 3, 9, 8, 7, 1, 3]));
        assert!(!luhn_valid(&[]));
    }

    #[test]
    fn resident_numbers_keys_and_passwords_are_hard_blocked() {
        assert_eq!(
            check("900101-1234567"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::GovernmentIdNumber
            }
        );
        assert_eq!(
            check("주민번호 9001012234567 이야"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::GovernmentIdNumber
            }
        );
        let key = format!("0x{}", "0123456789abcdef".repeat(4));
        assert_eq!(
            check(&key),
            IngressVerdict::HardBlocked {
                kind: SecretKind::WalletPrivateKeyHex
            }
        );
        assert_eq!(
            check(&"9f".repeat(32)),
            IngressVerdict::HardBlocked {
                kind: SecretKind::WalletPrivateKeyHex
            }
        );
        assert_eq!(
            check("비밀번호는 qwer1234 야"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::ExplicitPasswordToken
            }
        );
        assert_eq!(
            check("password: hunter2"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::ExplicitPasswordToken
            }
        );
        assert_eq!(
            check("인증번호 482910"),
            IngressVerdict::HardBlocked {
                kind: SecretKind::ExplicitPasswordToken
            }
        );
        assert!(matches!(check("비밀번호 바꾸는 법 알려줘"), IngressVerdict::Clean(_)));
        assert!(matches!(check("password reset 어떻게 해?"), IngressVerdict::Clean(_)));
        assert!(
            matches!(check("spin 1234 combo"), IngressVerdict::Clean(_)),
            "keywords match whole words only"
        );
    }

    #[test]
    fn seed_phrases_need_twelve_consecutive_list_words() {
        let numbered: String = STUB[..12].iter().enumerate().map(|(i, w)| format!("{}. {w} ", i + 1)).collect();
        assert_eq!(
            check(&numbered),
            IngressVerdict::HardBlocked {
                kind: SecretKind::WalletSeedPhrase
            }
        );
        assert_eq!(
            check(&STUB.join(" ")),
            IngressVerdict::HardBlocked {
                kind: SecretKind::WalletSeedPhrase
            }
        );
        assert!(matches!(check(&STUB[..11].join(" ")), IngressVerdict::Clean(_)), "11 words");
        let broken = format!("{} boss {}", STUB[..6].join(" "), STUB[6..12].join(" "));
        assert!(matches!(check(&broken), IngressVerdict::Clean(_)), "another word breaks the run");
        assert!(matches!(check("I want to know about the boss"), IngressVerdict::Clean(_)));
        let no_list = IngressPiiScrubber::without_seed_list();
        assert_eq!(
            no_list.detect(&STUB[..12].join(" ")),
            None,
            "without the list seed phrases are not judged"
        );
    }

    #[test]
    fn soft_data_is_masked_zeroed_and_never_learned() {
        let mut raw = String::from("hong@example.com로 보내고 010-1234-5678로 연락 줘");
        let v = IngressPiiScrubber::new(&stub).inspect_and_scrub(&mut raw);
        assert!(raw.is_empty());
        assert_eq!(
            v,
            IngressVerdict::Scrubbed {
                sanitized_text: "[이메일]로 보내고 [전화번호]로 연락 줘".into(),
                allow_lexicon_learn: false
            }
        );
        assert!(!v.allows_lexicon_learn() && v.routable_text().is_some());
        let p = check(r"C:\Users\hong\Documents\save.dat 가 안 열려");
        assert_eq!(p.routable_text(), Some(r"C:\Users\[USER]\Documents\save.dat 가 안 열려"));
        let o = check("주문번호 202411210001234 배송 언제 와?");
        assert_eq!(o.routable_text(), Some("주문번호 #### 배송 언제 와?"));
        assert!(matches!(check("2024-11-21 패치 노트"), IngressVerdict::Clean(_)), "dates pass");
        let clean = check("말레니아 2페이즈 어떻게 피해?");
        assert!(clean.allows_lexicon_learn());
        assert_eq!(clean.block_reaction(), None);
    }

    #[test]
    fn zeroize_empties_the_string() {
        let mut s = String::from("secret");
        zeroize_string(&mut s);
        assert!(s.is_empty());
    }
}
