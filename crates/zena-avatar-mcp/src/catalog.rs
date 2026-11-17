//! The avatar catalog: openly licensed 3D `.vrm` models and 2D layered kits Zena can wear, and the downloader
//! that puts them in the asset cache (`%LOCALAPPDATA%\RaonZena\assets\catalog\<id>\`).
//!
//! Every built-in entry is pinned: a fixed URL (a commit on GitHub, an Arweave transaction) with its byte size and
//! SHA-256, checked after the download, so a changed or truncated file never lands in the cache. The rules are
//! premium-discount-app's model downloader's, shared through `premium-discount-manifest`: downloads go to
//! `<file>.part` and are renamed only when size and hash match ([`premium_discount_manifest::promote`]); a cached file
//! is re-hashed before it is trusted; listed paths stay inside the entry's folder.
//!
//! The built-in list holds what could be verified on 2026-11-17 (file, license in the model's own VRM meta):
//!
//! | id | model | license (VRM meta) |
//! |---|---|---|
//! | `vrm1-twist-sample` (default) | pixiv's VRM 1.0 sample girl (three-vrm, vrm-specification) | VRM Public License 1.0: avatar use by everyone, commercial use by corporations, redistribution and modification allowed, credit unnecessary |
//! | `seed-san` | VirtualCast's VRM 1.0 sample | VRM Public License 1.0, credit required ("Seed-san model by VirtualCast, Inc.") |
//! | `100avatars-rose`, `100avatars-rabbit` | Polygonal Mind's 100Avatars (low-poly, VRM 0.x) | CC0 |
//!
//! VRoid's official CC0 sample avatars sit behind a VRoid Hub sign-in, so they cannot be pinned here. No public
//! 2D kit in this crate's `kit.json` layout (hair_back · body · face · eyes · mouth · hair_front · hands · fx) was
//! found either, so the only built-in 2D entry is the built-in vector kit. More entries — a VRoid Hub model the
//! user downloaded, a PNGTuber kit converted to `kit.json` — go in `<cache root>/catalog.json` (same shape as
//! [`Entry`], a JSON array); user entries with the id of a built-in one replace it.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use premium_discount_manifest::{is_sha256_hex, part_path, promote, safe_relative};
use serde::{Deserialize, Serialize};

/// The per-user asset cache: `%LOCALAPPDATA%\RaonZena\assets` (else `$XDG_CACHE_HOME`, else the temp dir).
pub fn default_cache_root() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("RaonZena")
        .join("assets")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// A `.vrm` model (Track B-2, `AvatarTrack::Subculture3d`).
    Vrm,
    /// A `kit.json` directory of layer PNGs (Track B-1, `AvatarTrack::Subculture2d`, see [`crate::chibi`]).
    ChibiKit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CatalogFile {
    /// File name inside the entry's folder (`model.vrm`, `kit.json`, `eyes/happy.png`).
    pub path: String,
    pub url: String,
    /// Lowercase hex.
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub kind: Kind,
    pub name: String,
    pub author: String,
    /// SPDX-ish name (`CC0-1.0`, `VRM-PL-1.0`) as the model declares it.
    pub license: String,
    pub license_url: String,
    /// What the license asks of us (credit line), or empty.
    #[serde(default)]
    pub credit: String,
    #[serde(default)]
    pub note: String,
    /// Fetched automatically when the track has no model configured.
    #[serde(default)]
    pub default: bool,
    /// Empty only for the built-in vector kit.
    #[serde(default)]
    pub files: Vec<CatalogFile>,
}

impl Entry {
    /// The file the track loads: the `.vrm` for a model, the folder holding `kit.json` for a kit.
    pub fn main_path(&self, root: &Path) -> PathBuf {
        let dir = entry_dir(root, &self.id);
        match self.kind {
            Kind::Vrm => self.files.first().map(|f| dir.join(&f.path)).unwrap_or(dir),
            Kind::ChibiKit => dir,
        }
    }

    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// The built-in vector kit's id (no files: [`crate::chibi::ChibiKit::builtin`]).
pub const BUILTIN_KIT: &str = "builtin-chibi";

const VRM_SPEC: &str = "https://raw.githubusercontent.com/vrm-c/vrm-specification/94e82dd346fa6cf0337c4421728640e5252dd38e/samples";

/// The pinned entries (see the module docs).
pub fn builtin() -> Vec<Entry> {
    let file = |path: &str, url: String, sha256: &str, size: u64| CatalogFile {
        path: path.into(),
        url,
        sha256: sha256.into(),
        size,
    };
    vec![
        Entry {
            id: "vrm1-twist-sample".into(),
            kind: Kind::Vrm,
            name: "VRM1_Constraint_Twist_Sample".into(),
            author: "pixiv Inc.".into(),
            license: "VRM-PL-1.0".into(),
            license_url: "https://vrm.dev/licenses/1.0/".into(),
            credit: String::new(),
            note: "VRM 1.0 sample of three-vrm / vrm-specification: MToon, expressions, LookAt bones, spring bones".into(),
            default: true,
            files: vec![file(
                "model.vrm",
                format!("{VRM_SPEC}/VRM1_Constraint_Twist_Sample/vrm/VRM1_Constraint_Twist_Sample.vrm"),
                "12c2b97e95e700783a6a550dc0eee2d7880aeedccef9ae67bc4c5a2f0f2631a2",
                10_776_032,
            )],
        },
        Entry {
            id: "seed-san".into(),
            kind: Kind::Vrm,
            name: "Seed-san".into(),
            author: "VirtualCast, Inc.".into(),
            license: "VRM-PL-1.0".into(),
            license_url: "https://vrm.dev/licenses/1.0/".into(),
            credit: "Seed-san model by VirtualCast, Inc.".into(),
            note: "VRM 1.0 sample: MToon and PBR materials, expressions (morph, UV), LookAt by expression".into(),
            default: false,
            files: vec![file(
                "model.vrm",
                format!("{VRM_SPEC}/Seed-san/vrm/Seed-san.vrm"),
                "624d0d554bc205bbdc33e22a68a2c3c20edebb3e573011ead8878a65e5329b23",
                10_917_800,
            )],
        },
        Entry {
            id: "100avatars-rose".into(),
            kind: Kind::Vrm,
            name: "Rose (100Avatars 057)".into(),
            author: "Polygonal Mind".into(),
            license: "CC0-1.0".into(),
            license_url: "https://creativecommons.org/publicdomain/zero/1.0/".into(),
            credit: String::new(),
            note: "Low-poly VRM 0.x, stored on Arweave (open-source-avatars registry)".into(),
            default: false,
            files: vec![file(
                "model.vrm",
                "https://arweave.net/Ea1KXujzJatQgCFSMzGOzp_UtHqB1pyia--U3AtkMAY".into(),
                "8e44e5638ffdf935e5d6fa990aa6cba428269f342f2db024c2961ee286a41e5f",
                2_400_964,
            )],
        },
        Entry {
            id: "100avatars-rabbit".into(),
            kind: Kind::Vrm,
            name: "Rabbit (100Avatars 059)".into(),
            author: "Polygonal Mind".into(),
            license: "CC0-1.0".into(),
            license_url: "https://creativecommons.org/publicdomain/zero/1.0/".into(),
            credit: String::new(),
            note: "Low-poly VRM 0.x, stored on Arweave (open-source-avatars registry)".into(),
            default: false,
            files: vec![file(
                "model.vrm",
                "https://arweave.net/RymRtrmhHx_f9ZDvtvIQb1noTHvILdjoTg5G7L2DR-8".into(),
                "ea5453d2d74a9bd23a1fb27159e40e8a0d402f3d42a538b28118c1639bc9eae9",
                2_633_752,
            )],
        },
        Entry {
            id: BUILTIN_KIT.into(),
            kind: Kind::ChibiKit,
            name: "Zena chibi (built-in vector kit)".into(),
            author: "Raon Zena".into(),
            license: "Apache-2.0".into(),
            license_url: "https://www.apache.org/licenses/LICENSE-2.0".into(),
            credit: String::new(),
            note: "Drawn in code; nothing to download".into(),
            default: true,
            files: Vec::new(),
        },
    ]
}

/// `<root>/catalog.json`: user entries, a JSON array of [`Entry`].
pub const USER_CATALOG: &str = "catalog.json";

/// Built-in entries, then `<root>/catalog.json` (an entry with a built-in id replaces it; a user entry marked
/// `default` takes the default from the built-in ones of its kind). A broken user file is an error, not ignored.
pub fn load(root: &Path) -> Result<Vec<Entry>> {
    let mut entries = builtin();
    let path = root.join(USER_CATALOG);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let user: Vec<Entry> = serde_json::from_str(&text).with_context(|| format!("{} is not a list of entries", path.display()))?;
    for u in user {
        check_entry(&u).with_context(|| format!("{}: entry {}", path.display(), u.id))?;
        if u.default {
            for e in entries.iter_mut().filter(|e| e.kind == u.kind) {
                e.default = false;
            }
        }
        match entries.iter_mut().find(|e| e.id == u.id) {
            Some(e) => *e = u,
            None => entries.push(u),
        }
    }
    Ok(entries)
}

fn check_entry(e: &Entry) -> Result<()> {
    if e.id.is_empty() || !e.id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        bail!("id must be ASCII letters, digits, - or _");
    }
    if e.kind == Kind::Vrm && e.files.len() != 1 {
        bail!("a vrm entry has exactly one file");
    }
    if e.kind == Kind::ChibiKit && !e.files.is_empty() && !e.files.iter().any(|f| f.path == "kit.json") {
        bail!("a chibi_kit entry needs kit.json");
    }
    for f in &e.files {
        safe_relative(&f.path).map_err(|m| anyhow!(m))?;
        if !is_sha256_hex(&f.sha256) {
            bail!("{}: sha256 must be 64 hex digits", f.path);
        }
        if !(f.url.starts_with("https://") || f.url.starts_with("http://127.0.0.1") || f.url.starts_with("http://localhost")) {
            bail!("{}: only https URLs (or a local test server)", f.path);
        }
    }
    Ok(())
}

pub fn entry_dir(root: &Path, id: &str) -> PathBuf {
    root.join("catalog").join(id)
}

/// The default entry of a kind.
pub fn default_of(entries: &[Entry], kind: Kind) -> Option<&Entry> {
    entries.iter().find(|e| e.kind == kind && e.default)
}

pub fn find<'a>(entries: &'a [Entry], id: &str) -> Option<&'a Entry> {
    entries.iter().find(|e| e.id == id)
}

fn verified(path: &Path, f: &CatalogFile) -> bool {
    premium_discount_manifest::verified(path, f.size, &f.sha256)
}

/// Whether every file of the entry is in the cache with the pinned size and hash.
pub fn is_cached(root: &Path, e: &Entry) -> bool {
    let dir = entry_dir(root, &e.id);
    e.files.iter().all(|f| verified(&dir.join(&f.path), f))
}

/// Download progress: `(file path, bytes so far, bytes of this file)`.
pub type Progress<'a> = &'a mut dyn FnMut(&str, u64, u64);

/// Puts every file of `e` in the cache (skipping files already there and verified) and returns
/// [`Entry::main_path`]. A size or hash mismatch removes the partial file and fails.
pub fn fetch(root: &Path, e: &Entry, progress: Progress<'_>) -> Result<PathBuf> {
    check_entry(e)?;
    let dir = entry_dir(root, &e.id);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        .build();
    for f in &e.files {
        let dest = dir.join(&f.path);
        if verified(&dest, f) {
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let part = part_path(&dest);
        let result = download(&agent, f, &part, progress)
            .and_then(|()| promote(&part, &dest, f.size, &f.sha256).map_err(anyhow::Error::from));
        if let Err(err) = result {
            let _ = std::fs::remove_file(&part);
            return Err(err.context(format!("{} ({})", f.path, f.url)));
        }
    }
    Ok(e.main_path(root))
}

fn download(agent: &ureq::Agent, f: &CatalogFile, part: &Path, progress: Progress<'_>) -> Result<()> {
    let resp = agent.get(&f.url).call().map_err(|e| match e {
        ureq::Error::Status(code, _) => anyhow::anyhow!("HTTP {code}"),
        other => anyhow::anyhow!("{other}"),
    })?;
    let mut reader = resp.into_reader().take(f.size + 1);
    let mut out = std::fs::File::create(part).with_context(|| format!("creating {}", part.display()))?;
    let mut buf = vec![0u8; 1 << 16];
    let mut got = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        got += n as u64;
        if got > f.size {
            bail!("larger than the pinned {} bytes", f.size);
        }
        out.write_all(&buf[..n])?;
        progress(&f.path, got, f.size);
    }
    out.flush()?;
    if got != f.size {
        bail!("got {got} of {} bytes", f.size);
    }
    Ok(())
}

/// One entry for `list_avatar_models`: the entry plus where it lives and whether it is already cached.
pub fn describe(root: &Path, e: &Entry) -> serde_json::Value {
    let mut v = serde_json::to_value(e).unwrap_or_default();
    let cached = e.files.is_empty() || is_cached(root, e);
    if let Some(o) = v.as_object_mut() {
        o.insert("cached".into(), cached.into());
        o.insert("download_bytes".into(), e.download_size().into());
        if !e.files.is_empty() {
            o.insert("path".into(), e.main_path(root).to_string_lossy().into_owned().into());
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;
    use std::net::TcpListener;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zena-catalog-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Serves `body` to every GET, `n` times, and returns the base URL.
    fn serve(body: Vec<u8>, n: usize) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            for s in l.incoming().take(n) {
                let mut s = s.unwrap();
                let mut r = std::io::BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                while r.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
                    line.clear();
                }
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                s.write_all(head.as_bytes()).unwrap();
                s.write_all(&body).unwrap();
            }
        });
        format!("http://127.0.0.1:{}", addr.port())
    }

    fn hex_sha(b: &[u8]) -> String {
        use premium_discount_manifest::{hex, Digest, Sha256};
        hex(&Sha256::digest(b))
    }

    fn entry(id: &str, url: &str, body: &[u8], kind: Kind) -> Entry {
        Entry {
            id: id.into(),
            kind,
            name: id.into(),
            author: "test".into(),
            license: "CC0-1.0".into(),
            license_url: String::new(),
            credit: String::new(),
            note: String::new(),
            default: false,
            files: vec![CatalogFile {
                path: if kind == Kind::Vrm { "model.vrm".into() } else { "kit.json".into() },
                url: format!("{url}/file"),
                sha256: hex_sha(body),
                size: body.len() as u64,
            }],
        }
    }

    #[test]
    fn builtin_entries_are_pinned_and_have_one_default_per_kind() {
        let all = builtin();
        for e in &all {
            check_entry(e).unwrap();
            assert!(e.files.iter().all(|f| f.url.starts_with("https://")), "{}", e.id);
        }
        assert_eq!(default_of(&all, Kind::Vrm).unwrap().id, "vrm1-twist-sample");
        assert_eq!(default_of(&all, Kind::ChibiKit).unwrap().id, BUILTIN_KIT);
        assert_eq!(all.iter().filter(|e| e.default).count(), 2);
        assert!(all.iter().any(|e| e.license == "CC0-1.0"));
    }

    #[test]
    fn fetch_verifies_and_reuses_the_cache() {
        let root = temp("fetch");
        let body = b"glTF pretend model".to_vec();
        let url = serve(body.clone(), 1);
        let e = entry("t1", &url, &body, Kind::Vrm);
        assert!(!is_cached(&root, &e));
        let mut seen = 0u64;
        let path = fetch(&root, &e, &mut |_, got, _| seen = got).unwrap();
        assert_eq!(path, entry_dir(&root, "t1").join("model.vrm"));
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert_eq!(seen, body.len() as u64);
        assert!(is_cached(&root, &e));
        // The server served once; a second fetch never asks it.
        assert_eq!(fetch(&root, &e, &mut |_, _, _| panic!("downloaded again")).unwrap(), path);
        assert_eq!(describe(&root, &e)["cached"], true);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_wrong_hash_or_size_leaves_nothing_behind() {
        let root = temp("bad");
        let url = serve(b"tampered!".to_vec(), 2);
        let mut e = entry("t2", &url, b"original!", Kind::Vrm);
        let err = format!("{:#}", fetch(&root, &e, &mut |_, _, _| {}).unwrap_err());
        assert!(err.contains("SHA-256"), "{err}");
        let dir = entry_dir(&root, "t2");
        assert!(std::fs::read_dir(&dir).unwrap().next().is_none(), "no partial or final file");
        e.files[0].size = 4;
        let err = format!("{:#}", fetch(&root, &e, &mut |_, _, _| {}).unwrap_err());
        assert!(err.contains("larger than"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn user_catalog_adds_replaces_and_moves_the_default() {
        let root = temp("user");
        let mut kit = entry("my-kit", "https://example.com", b"{}", Kind::ChibiKit);
        kit.default = true;
        let mut rose = builtin().into_iter().find(|e| e.id == "100avatars-rose").unwrap();
        rose.default = true;
        std::fs::write(root.join(USER_CATALOG), serde_json::to_string(&vec![kit, rose]).unwrap()).unwrap();
        let all = load(&root).unwrap();
        assert_eq!(default_of(&all, Kind::ChibiKit).unwrap().id, "my-kit");
        assert_eq!(default_of(&all, Kind::Vrm).unwrap().id, "100avatars-rose");
        assert_eq!(all.iter().filter(|e| e.id == "100avatars-rose").count(), 1);

        let mut evil = entry("evil", "https://example.com", b"x", Kind::Vrm);
        evil.files[0].path = "../../outside.vrm".into();
        std::fs::write(root.join(USER_CATALOG), serde_json::to_string(&vec![evil]).unwrap()).unwrap();
        assert!(load(&root).is_err());
        std::fs::write(root.join(USER_CATALOG), "not json").unwrap();
        assert!(load(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Downloads every built-in model from its real URL and loads it with the crate's own parser:
    /// `cargo test -p zena-avatar-mcp catalog -- --ignored`.
    #[test]
    #[ignore = "network: downloads ~27 MB"]
    fn builtin_models_download_and_parse() {
        let root = std::env::var_os("ZENA_CATALOG_TEST_ROOT").map(PathBuf::from).unwrap_or_else(|| temp("real"));
        for e in builtin().iter().filter(|e| e.kind == Kind::Vrm) {
            let path = fetch(&root, e, &mut |_, _, _| {}).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let mesh = crate::mesh::load_glb(&bytes).unwrap();
            let info = crate::vrm::VrmInfo::from_glb(&bytes).unwrap().expect("a VRM");
            assert!(!mesh.positions.is_empty(), "{}", e.id);
            eprintln!(
                "{}: {:?}, {} vertices, {} expressions",
                e.id,
                info.version,
                mesh.positions.len(),
                info.expressions.len()
            );
        }
    }
}
