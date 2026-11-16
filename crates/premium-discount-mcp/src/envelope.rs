//! The `cc` (domain) of a premium.discount evaluation.
//!
//! The server accepts only a registrable domain (`www/src/evaluations.ts`:
//! `^([a-z0-9-]{1,63}\.)+[a-z]{2,63}$`, at most 64 characters) and premium-discount-app is stricter (no label
//! starting or ending with `-`). Raon Zena talks about three kinds of places, mapped as:
//!
//! * `store.steampowered.com` — kept (lowercased, trailing dot dropped).
//! * `https://www.example.com:8443/a?b` — the host, without a leading `www.`.
//! * `app://Code.exe/coding` — a desktop program: `code.exe`. Labels are folded to `[a-z0-9-]`, a name without a
//!   letters-only last label gets `.exe` (`app://eldenring` → `eldenring.exe`), and a label with no ASCII left
//!   (`카카오톡.exe`) becomes `x` + 8 hex of its SHA-256. The scene after the slash is returned separately: the
//!   server compares `cc` exactly, so a scene inside it would split one program into many domains.

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

/// Where a `cc` came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CcKind {
    Domain,
    Web,
    App,
}

impl CcKind {
    pub fn name(self) -> &'static str {
        match self {
            CcKind::Domain => "domain",
            CcKind::Web => "web",
            CcKind::App => "app",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cc {
    /// The value sent as `cc`.
    pub cc: String,
    pub kind: CcKind,
    /// `app://exe/<scene>`: the scene label (folded like a domain label), not part of `cc`.
    pub scene: Option<String>,
}

const MAX_CC: usize = 64;
const MAX_LABEL: usize = 63;

/// premium-discount-app's `normalize_cc` (which is at least as strict as the server).
pub fn check_domain(cc: &str) -> Result<()> {
    let chars_ok = cc
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-');
    let labels_ok = cc
        .split('.')
        .all(|l| !l.is_empty() && l.len() <= MAX_LABEL && !l.starts_with('-') && !l.ends_with('-'));
    let tld_ok = cc
        .rsplit('.')
        .next()
        .is_some_and(|t| (2..=MAX_LABEL).contains(&t.len()) && t.chars().all(|c| c.is_ascii_lowercase()));
    if cc.len() > MAX_CC || !cc.contains('.') || !chars_ok || !labels_ok || !tld_ok {
        bail!("cc must be a registrable domain: {cc}");
    }
    Ok(())
}

/// Folds one label to `[a-z0-9-]`: other characters become `-`, runs of `-` collapse, `-` is trimmed at the
/// ends. A label with nothing left is replaced by `x` + 8 hex of the original's SHA-256.
fn fold_label(raw: &str) -> String {
    let mut out = String::new();
    for c in raw.chars().flat_map(char::to_lowercase) {
        let c = if c.is_ascii_lowercase() || c.is_ascii_digit() { c } else { '-' };
        if c == '-' && (out.is_empty() || out.ends_with('-')) {
            continue;
        }
        out.push(c);
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        let digest = Sha256::digest(raw.as_bytes());
        out = format!("x{}", digest[..4].iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
    out.truncate(MAX_LABEL);
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn app_cc(rest: &str) -> Result<Cc> {
    let rest = rest.trim_end_matches('/');
    let (program, scene) = match rest.split_once('/') {
        Some((p, s)) => (p, Some(s)),
        None => (rest, None),
    };
    // A Windows path (`C:\Program Files\x.exe`) keeps only its file name.
    let program = program.rsplit(['\\', '/']).next().unwrap_or(program).trim();
    if program.is_empty() {
        bail!("app:// needs a program name");
    }
    let mut labels: Vec<String> = program.split('.').filter(|l| !l.trim().is_empty()).map(fold_label).collect();
    let tld_ok = labels.len() >= 2
        && labels
            .last()
            .is_some_and(|t| t.len() >= 2 && t.chars().all(|c| c.is_ascii_lowercase()));
    if !tld_ok {
        labels.push("exe".into());
    }
    let mut cc = labels.join(".");
    if cc.len() > MAX_CC {
        // Shorten the first (name) label, keeping the rest.
        let tail = cc.len() - labels[0].len();
        let keep = MAX_CC.saturating_sub(tail).max(1);
        labels[0].truncate(keep);
        while labels[0].ends_with('-') {
            labels[0].pop();
        }
        cc = labels.join(".");
    }
    check_domain(&cc)?;
    let scene = scene
        .map(|s| s.trim_matches('/').replace('/', "-"))
        .filter(|s| !s.trim().is_empty())
        .map(|s| fold_label(&s));
    Ok(Cc {
        cc,
        kind: CcKind::App,
        scene,
    })
}

fn host_of(url: &str) -> &str {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    let authority = after_scheme.split(['/', '?', '#']).next().unwrap_or("");
    let host_port = authority.rsplit_once('@').map(|(_, h)| h).unwrap_or(authority);
    match host_port.rsplit_once(':') {
        Some((h, port)) if port.chars().all(|c| c.is_ascii_digit()) => h,
        _ => host_port,
    }
}

/// Maps a domain, a web URL or an `app://` place to the `cc` premium.discount accepts.
pub fn normalize_cc(raw: &str) -> Result<Cc> {
    let s = raw.trim();
    if s.is_empty() {
        bail!("cc is required");
    }
    if let Some(rest) = s.strip_prefix("app://").or_else(|| s.strip_prefix("APP://")) {
        return app_cc(rest);
    }
    let (host, kind) = if s.contains("://") {
        let h = host_of(s).to_ascii_lowercase();
        let h = h.strip_prefix("www.").map(str::to_string).unwrap_or(h);
        (h, CcKind::Web)
    } else {
        (s.to_ascii_lowercase(), CcKind::Domain)
    };
    let cc = host.trim_end_matches('.').to_string();
    check_domain(&cc)?;
    Ok(Cc { cc, kind, scene: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(s: &str) -> String {
        normalize_cc(s).unwrap().cc
    }

    #[test]
    fn domains_and_urls() {
        assert_eq!(cc(" Store.SteamPowered.com. "), "store.steampowered.com");
        assert_eq!(cc("https://www.Example.com:8443/a/b?c=d#e"), "example.com");
        assert_eq!(cc("http://user:pw@news.ycombinator.com/item?id=1"), "news.ycombinator.com");
        assert_eq!(normalize_cc("https://store.steampowered.com/app/1").unwrap().kind, CcKind::Web);
        for bad in ["localhost", "http://127.0.0.1:8080/", "a_b.com", "-a.com", "example.c0m", "", "x.y"] {
            assert!(normalize_cc(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn apps_become_exe_domains() {
        let c = normalize_cc("app://Code.exe/coding").unwrap();
        assert_eq!(
            (c.cc.as_str(), c.kind, c.scene.as_deref()),
            ("code.exe", CcKind::App, Some("coding"))
        );
        assert_eq!(cc("app://eldenring.exe/boss_fight"), "eldenring.exe");
        assert_eq!(
            normalize_cc("app://eldenring.exe/boss_fight").unwrap().scene.as_deref(),
            Some("boss-fight")
        );
        assert_eq!(cc("app://Code - Insiders.exe"), "code-insiders.exe");
        assert_eq!(cc("app://steam_api64.exe"), "steam-api64.exe");
        assert_eq!(cc("app://firefox"), "firefox.exe");
        assert_eq!(cc("app://C:\\Program Files\\Notepad++\\notepad++.exe/editing"), "notepad.exe");
        assert_eq!(cc("app://python3.13"), "python3.13.exe");
        let hangul = cc("app://카카오톡.exe/chat");
        assert!(
            hangul.starts_with('x') && hangul.ends_with(".exe") && hangul.len() == "x01234567.exe".len(),
            "{hangul}"
        );
        assert_eq!(hangul, cc("app://카카오톡.exe"), "stable");
        let long = cc(&format!("app://{}.exe", "a".repeat(100)));
        assert!(long.len() <= 64 && long.ends_with(".exe"), "{long}");
        assert!(normalize_cc("app:///scene").is_err());
        for s in [
            "app://Code.exe/coding",
            "app://카카오톡.exe",
            "app://firefox",
            "https://www.example.com/",
        ] {
            check_domain(&cc(s)).unwrap();
        }
    }
}
