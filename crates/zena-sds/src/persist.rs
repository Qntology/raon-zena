//! On-disk layout of a [`crate::Ledger`] named `n` under `dir` (logis "ledger-split / run-files"):
//!
//! * `dir/n.json` – the base ledger ([`BaseFile`]), rewritten only by `open` (folding crashed runs) and `close`.
//! * `dir/n.lock` – serialises those read-modify-writes across processes.
//! * `dir/runs/n/<run>.json` – one live run's observations ([`RunFile`]), rewritten by `flush`.
//! * `dir/runs/n/<run>.lock` – held (OS file lock) by the live run, so another process never folds it.
//!
//! Writes go through a temp file and a rename. Unreadable files are renamed `*.corrupt`, files of another recipe or
//! ledger name `*.stale`; both are then ignored.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::stats::ScopeStat;
use crate::RECIPE;

/// The base ledger stops growing at this size: the least observed, oldest scopes are dropped (logis `CAP_BYTES`).
pub(crate) const BASE_CAP_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct BaseFile {
    pub recipe: String,
    pub name: String,
    #[serde(default)]
    pub updated_ms: i64,
    /// Runs already folded in whose file may still exist (a crash between writing this file and deleting theirs).
    #[serde(default)]
    pub folded: Vec<String>,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct RunFile {
    pub recipe: String,
    pub name: String,
    pub run: String,
    #[serde(default)]
    pub started_ms: i64,
    #[serde(default)]
    pub updated_ms: i64,
    #[serde(default)]
    pub observations: u64,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
}

/// [`RunFile`] by reference, for writing without cloning the run.
#[derive(Serialize)]
pub(crate) struct RunFileOut<'a> {
    pub recipe: &'a str,
    pub name: &'a str,
    pub run: &'a str,
    pub started_ms: i64,
    pub updated_ms: i64,
    pub observations: u64,
    pub scopes: &'a HashMap<String, ScopeStat>,
}

pub(crate) trait Stamped {
    fn stamp(&self) -> (&str, &str);
}

impl Stamped for BaseFile {
    fn stamp(&self) -> (&str, &str) {
        (&self.recipe, &self.name)
    }
}

impl Stamped for RunFile {
    fn stamp(&self) -> (&str, &str) {
        (&self.recipe, &self.name)
    }
}

pub(crate) enum Loaded<T> {
    Missing,
    Valid(T),
    /// Corrupt or stale; renamed to this path (or left in place when the rename failed).
    SetAside(PathBuf),
}

/// Reads a ledger / run file. I/O errors other than "not found" are returned (the file may be fine, just busy).
pub(crate) fn load<T: DeserializeOwned + Stamped>(path: &Path, name: &str) -> io::Result<Loaded<T>> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Loaded::Missing),
        Err(e) => return Err(e),
    };
    Ok(match serde_json::from_slice::<T>(&bytes) {
        Err(_) => Loaded::SetAside(set_aside(path, "corrupt")),
        Ok(f) if f.stamp() != (RECIPE, name) => Loaded::SetAside(set_aside(path, "stale")),
        Ok(f) => Loaded::Valid(f),
    })
}

/// Renames `path` to `path.<ms>.<suffix>`; returns where it is now.
fn set_aside(path: &Path, suffix: &str) -> PathBuf {
    let target = with_suffix(path, &format!(".{}.{}", crate::ledger::now_ms(), suffix));
    match fs::rename(path, &target) {
        Ok(()) => target,
        Err(_) => path.to_path_buf(),
    }
}

pub(crate) fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Writes `bytes` to `path.tmp` and renames it over `path`; falls back to a direct write when the rename fails
/// (logis `write_atomic`).
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = with_suffix(path, ".tmp");
    match fs::write(&tmp, bytes).and_then(|_| fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(_) => {
            let _ = fs::remove_file(&tmp);
            fs::write(path, bytes)
        }
    }
}

/// Pretty JSON of the base, shedding the least observed (then oldest) scopes while it exceeds `cap` bytes.
/// Returns the text and how many scopes were dropped.
pub(crate) fn encode_base(file: &mut BaseFile, cap: usize) -> serde_json::Result<(String, usize)> {
    let mut txt = serde_json::to_string_pretty(file)?;
    if txt.len() <= cap {
        return Ok((txt, 0));
    }
    let mut order: Vec<(String, u64, i64, usize)> = file
        .scopes
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.observations(),
                v.updated_ms,
                k.len() + serde_json::to_string_pretty(v).map_or(0, |s| s.len()),
            )
        })
        .collect();
    order.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
    let mut dropped = 0usize;
    let mut order = order.into_iter();
    while txt.len() > cap {
        let mut excess = txt.len() - cap;
        let mut any = false;
        for (k, _, _, size) in order.by_ref() {
            file.scopes.remove(&k);
            dropped += 1;
            any = true;
            excess = excess.saturating_sub(size);
            if excess == 0 {
                break;
            }
        }
        txt = serde_json::to_string_pretty(file)?;
        if !any {
            break;
        }
    }
    Ok((txt, dropped))
}

fn open_lock_file(path: &Path, create: bool) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).create(create).truncate(false).open(path)
}

/// Blocking exclusive lock on `path` for a read-modify-write of the base. Best effort: `None` when the file system
/// cannot lock (the write then proceeds unguarded, as logis did).
pub(crate) fn lock_base(path: &Path) -> Option<File> {
    let f = open_lock_file(path, true).ok()?;
    f.lock().ok()?;
    Some(f)
}

/// Creates and locks a live run's lock file; the lock lasts as long as the returned handle.
pub(crate) fn hold_run_lock(path: &Path) -> io::Result<File> {
    let f = open_lock_file(path, true)?;
    match f.try_lock() {
        Ok(()) => Ok(f),
        Err(TryLockError::WouldBlock) => Err(io::Error::new(io::ErrorKind::AlreadyExists, "run lock is held by another ledger")),
        // No locking on this file system: keep the handle, liveness just cannot be shown to others.
        Err(TryLockError::Error(_)) => Ok(f),
    }
}

/// Whether the run owning `lock_path` is still alive (its lock is held). A missing lock file means dead; a lock we
/// cannot test counts as alive, so a live run is never folded twice.
pub(crate) fn run_is_live(lock_path: &Path) -> bool {
    let f = match open_lock_file(lock_path, false) {
        Ok(f) => f,
        Err(e) => return e.kind() != io::ErrorKind::NotFound,
    };
    match f.try_lock() {
        Ok(()) => false,
        Err(TryLockError::WouldBlock) | Err(TryLockError::Error(_)) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Welford;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("zena-sds-test-{}-{}", std::process::id(), name));
            let _ = fs::remove_dir_all(&p);
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn write_atomic_replaces_without_leaving_a_temp() {
        let t = TempDir::new("persist-atomic");
        let path = t.0.join("x.json");
        write_atomic(&path, b"{\"a\":1}").unwrap();
        write_atomic(&path, b"{\"a\":2}").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\"a\":2}");
        assert_eq!(fs::read_dir(&t.0).unwrap().count(), 1);
    }

    #[test]
    fn load_sets_aside_corrupt_and_stale_files() {
        let t = TempDir::new("persist-load");
        let path = t.0.join("drop.json");
        assert!(matches!(load::<BaseFile>(&path, "drop").unwrap(), Loaded::Missing));

        fs::write(&path, b"{ not json").unwrap();
        match load::<BaseFile>(&path, "drop").unwrap() {
            Loaded::SetAside(p) => assert!(p.to_string_lossy().ends_with(".corrupt") && p.exists()),
            _ => panic!("corrupt file must be set aside"),
        }
        assert!(!path.exists());

        let other = BaseFile {
            recipe: RECIPE.into(),
            name: "web".into(),
            ..Default::default()
        };
        fs::write(&path, serde_json::to_vec(&other).unwrap()).unwrap();
        assert!(matches!(load::<BaseFile>(&path, "drop").unwrap(), Loaded::SetAside(p) if p.to_string_lossy().ends_with(".stale")));

        let good = BaseFile {
            recipe: RECIPE.into(),
            name: "drop".into(),
            ..Default::default()
        };
        fs::write(&path, serde_json::to_vec(&good).unwrap()).unwrap();
        assert!(matches!(load::<BaseFile>(&path, "drop").unwrap(), Loaded::Valid(_)));
    }

    #[test]
    fn encode_base_sheds_the_least_observed_scopes() {
        let mut f = BaseFile {
            recipe: RECIPE.into(),
            name: "drop".into(),
            ..Default::default()
        };
        for i in 0..40u64 {
            let mut s = ScopeStat::default();
            let mut w = Welford::default();
            for k in 0..=i {
                w.push(k as f64, 0);
            }
            s.baseline.insert("valence".into(), w);
            f.scopes.insert(format!("drop|p{i}|"), s);
        }
        let full = serde_json::to_string_pretty(&f).unwrap().len();
        let cap = full / 2;
        let (txt, dropped) = encode_base(&mut f, cap).unwrap();
        assert!(txt.len() <= cap && dropped > 0, "{} > {cap}", txt.len());
        assert!(f.scopes.contains_key("drop|p39|"), "the most observed scope stays");
        assert!(!f.scopes.contains_key("drop|p0|"), "the least observed goes first");
        let (_, none) = encode_base(&mut f, usize::MAX).unwrap();
        assert_eq!(none, 0);
    }

    #[test]
    fn run_lock_shows_liveness() {
        let t = TempDir::new("persist-lock");
        let lock = t.0.join("r1.lock");
        assert!(!run_is_live(&lock), "no lock file: dead");
        let held = hold_run_lock(&lock).unwrap();
        assert!(run_is_live(&lock));
        drop(held);
        assert!(!run_is_live(&lock), "released on drop (crash)");
        let base = lock_base(&t.0.join("drop.lock"));
        assert!(base.is_some());
    }
}
