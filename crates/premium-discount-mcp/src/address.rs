//! premium.discount addresses: `0x` + 40 lowercase hex.
//!
//! [`hash_id`] is the server project's `hashId(text)` (legacy `www/src/index.ts`, `www/public/static/js`):
//! `ethers.computeAddress(ethers.hashMessage(text)).toLowerCase()`, i.e. the EIP-191 message hash used as a
//! secp256k1 private key, and the address of its public key. The server only checks `ethers.isAddress`, so any
//! 20 bytes would pass; using the same recipe keeps `to` / `bcc` comparable with the other clients.

use anyhow::{bail, Result};
use k256::elliptic_curve::sec1::ToEncodedPoint;
use tiny_keccak::{Hasher, Keccak};
use unicode_normalization::UnicodeNormalization;

fn keccak256(parts: &[&[u8]]) -> [u8; 32] {
    let mut k = Keccak::v256();
    for p in parts {
        k.update(p);
    }
    let mut out = [0u8; 32];
    k.finalize(&mut out);
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `ethers.hashMessage(text)`: keccak256 of `"\x19Ethereum Signed Message:\n" + byte length + text`.
pub fn hash_message(text: &str) -> [u8; 32] {
    let len = text.len().to_string();
    keccak256(&[b"\x19Ethereum Signed Message:\n", len.as_bytes(), text.as_bytes()])
}

/// `ethers.computeAddress(private_key)`: the last 20 bytes of keccak256 of the uncompressed public key.
pub fn compute_address(private_key: &[u8; 32]) -> Result<String> {
    let Ok(secret) = k256::SecretKey::from_slice(private_key) else {
        // Zero or ≥ the curve order: ethers throws too. Probability ~2^-128 for a hash.
        bail!("not a valid secp256k1 private key");
    };
    let point = secret.public_key().to_encoded_point(false);
    let digest = keccak256(&[&point.as_bytes()[1..]]);
    Ok(format!("0x{}", hex(&digest[12..])))
}

/// `hashId(text)` of the premium.discount web client.
pub fn hash_id(text: &str) -> Result<String> {
    compute_address(&hash_message(text))
}

/// The text a content address (`bcc`) is computed from: NFC, trimmed, whitespace runs folded to one space,
/// so the same error message or item description read from two windows lands on the same address.
pub fn normalize_content(text: &str) -> String {
    text.nfc().collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `hash_id(normalize_content(text))`.
pub fn content_address(text: &str) -> Result<String> {
    let norm = normalize_content(text);
    if norm.is_empty() {
        bail!("cannot address empty content");
    }
    hash_id(&norm)
}

/// `0x` + 40 hex in any case, lowercased (the app's rule; the server additionally checks mixed-case checksums,
/// which lowercasing sidesteps).
pub fn normalize_address(field: &str, raw: &str) -> Result<String> {
    let a = raw.trim().to_ascii_lowercase();
    if a.len() == 42 && a.starts_with("0x") && a[2..].chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(a)
    } else {
        bail!("{field} must be an address (0x + 40 hex)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generated with the ethers 6.9.0 bundle the server project ships (www/public/static/js/ethers.umd.min.js):
    // { hashMessage(s), computeAddress(hashMessage(s)).toLowerCase() }.
    const REFERENCE: [(&str, &str, &str); 8] = [
        (
            "",
            "5f35dce98ba4fba25530a026ed80b2cecdaa31091ba4958b99b52ea1d068adad",
            "0x7404358246d491ed2f9dac694f7dea789037424c",
        ),
        (
            "hello",
            "50b2c43fd39106bafbba0da34fc430e1f91e3c96ea2acee2bc34119f92b37750",
            "0x56d67386939607c11bd60bb009eb02f4dd29c318",
        ),
        (
            "SurfaceError: Outdated",
            "268c9fed744753daf4ddc9f99d7daebf0610e9dab5d59c0b49d2c81566d71c32",
            "0xa0328cd123a2b578ba1a6a211a29af590fec75c7",
        ),
        (
            "패치 1.4: 보스 체력 15% 너프",
            "98f220dae2d6ec4549c64dcfa6a12f256d6994b01d2a865ad550bb38ab6264f5",
            "0x75252908349507fec4859cf5429f5e19a3710381",
        ),
        (
            "app://code.exe/coding",
            "bc0452b4008d14a3edbd27fceb6ec87aa2c8963014e6e8dccebd2602c00dabff",
            "0x4a2215e2a6342518db6e67379af597cda515362e",
        ),
        (
            "Visual Studio Code — main.rs",
            "2b7470a469e21bb0b9aa371535df4ce2a0d388658ce4356616ca336da154a852",
            "0x1e5f4e065ef66523e315255c39c796f89aafb61d",
        ),
        (
            "store.steampowered.com",
            "926cadd235c4ef546f5e41b355dfb0b7f79cee0a35d9b8ae9eb7824ffc5c4b60",
            "0x0fee18a53a5dd368d3ce799e31cdef5763533d15",
        ),
        (
            "a@b.c",
            "24df26c8284cef6af1c1bb86438655a1c0ffd9096be308aa423904aa81b357d1",
            "0x8f88b3418daf380f7ccace748d53c63f949356db",
        ),
    ];

    #[test]
    fn matches_ethers() {
        for (text, msg, addr) in REFERENCE {
            assert_eq!(hex(&hash_message(text)), msg, "hashMessage({text:?})");
            assert_eq!(hash_id(text).unwrap(), addr, "hashId({text:?})");
        }
    }

    #[test]
    fn content_is_normalized_before_hashing() {
        let a = content_address("  SurfaceError:\n\tOutdated  ").unwrap();
        assert_eq!(a, hash_id("SurfaceError: Outdated").unwrap());
        // NFD Hangul (as macOS file names carry it) folds to the NFC address.
        let nfd: String = "패치".nfd().collect();
        assert_ne!(nfd, "패치");
        assert_eq!(content_address(&nfd).unwrap(), content_address("패치").unwrap());
        assert!(content_address(" \n ").is_err());
    }

    #[test]
    fn addresses_are_checked_and_lowercased() {
        assert_eq!(
            normalize_address("to", " 0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD ").unwrap(),
            "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        );
        assert!(normalize_address("to", "0x1234").is_err());
        assert!(normalize_address("to", "abcdefabcdefabcdefabcdefabcdefabcdefabcdef").is_err());
    }

    #[test]
    fn invalid_private_keys_fail() {
        assert!(compute_address(&[0u8; 32]).is_err());
        assert!(compute_address(&[0xff; 32]).is_err());
    }
}
