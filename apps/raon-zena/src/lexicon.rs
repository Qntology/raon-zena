//! Where Zena's word dictionary comes from, and how its updates reach the running pipeline.
//!
//! With the `lexicon-db` feature (default) the dictionary lives in a LanceDB database: seeded from
//! `data/zena_lexicon.json` on first run, edited with `zena-lexicon` (add / remove / anchors / rollback ...) while
//! Zena runs, and — with `--learn` — taught by use: a span the router caught by vectors alone (no exact word) is
//! counted, and after three sightings it becomes a bias phrase of its domain. A service thread polls the
//! dictionary's revision and pushes every new revision to the router and the screen watcher (hot reload).
//! Without the feature the bundled (or `--lexicon`) JSON is used as is.

use crate::watcher::AnchorsUpdate;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use zena_lexicon::Lexicon;

pub struct Loaded {
    pub lexicon: Lexicon,
    pub anchors: HashMap<String, Vec<f32>>,
    pub source: String,
}

/// Something the pipeline asks the dictionary service to do.
#[cfg_attr(not(feature = "lexicon-db"), allow(dead_code))]
pub enum LexiconMsg {
    /// A span caught by vectors alone, for learning.
    Learn { key: String, span: String },
}

pub fn read_json(path: Option<&Path>) -> Result<Lexicon> {
    match path {
        Some(p) => Lexicon::from_json(&std::fs::read_to_string(p).with_context(|| p.display().to_string())?),
        None => Ok(Lexicon::bundled()),
    }
}

/// The per-user dictionary location: `%LOCALAPPDATA%\RaonZena\lexicon.lance` on Windows,
/// `$XDG_DATA_HOME/raon-zena/lexicon.lance` (or `~/.local/share/...`) elsewhere.
pub fn default_db() -> PathBuf {
    if cfg!(windows) {
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(d).join("RaonZena").join("lexicon.lance");
        }
    }
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("raon-zena").join("lexicon.lance")
}

#[cfg(feature = "lexicon-db")]
pub fn open(db: &Path, seed: Option<&Path>) -> Result<Loaded> {
    use zena_lexicon::LexiconStore;
    let store = LexiconStore::open(db)?;
    if store.revision()? == 0 {
        let lex = read_json(seed)?;
        let rep = store.import(&lex)?;
        eprintln!(
            "[lexicon] seeded {} with {} domains, {} vision nodes",
            db.display(),
            rep.domains,
            rep.vision
        );
    }
    let lexicon = store.load()?;
    let anchors = store.anchors()?;
    eprintln!("[lexicon] {}", store.stats()?);
    Ok(Loaded {
        source: format!("LanceDB {} (revision {})", db.display(), lexicon.version),
        lexicon,
        anchors: anchors.vectors,
    })
}

#[cfg(not(feature = "lexicon-db"))]
pub fn open(_db: &Path, seed: Option<&Path>) -> Result<Loaded> {
    Ok(Loaded {
        lexicon: read_json(seed)?,
        anchors: HashMap::new(),
        source: "JSON (built without the LanceDB store)".into(),
    })
}

/// Starts the dictionary service; returns where to send learning requests.
#[cfg(feature = "lexicon-db")]
pub fn spawn_service(
    db: PathBuf,
    poll: std::time::Duration,
    promote_at: u32,
    to_router: Sender<Lexicon>,
    to_watcher: Option<Sender<AnchorsUpdate>>,
) -> Sender<LexiconMsg> {
    use std::sync::mpsc::{channel, RecvTimeoutError};
    use zena_lexicon::store::Learned;
    let (tx, rx) = channel::<LexiconMsg>();
    std::thread::Builder::new()
        .name("zena-lexicon".into())
        .spawn(move || {
            let store = match zena_lexicon::LexiconStore::open(&db) {
                Ok(s) => s,
                Err(e) => return eprintln!("[lexicon] service not started: {e:#}"),
            };
            let mut last = store.revision().unwrap_or(0);
            loop {
                match rx.recv_timeout(poll) {
                    Ok(LexiconMsg::Learn { key, span }) => match store.learn(&key, &span, promote_at) {
                        Ok(Learned::Promoted) => eprintln!("[lexicon] learned '{span}' for {key}"),
                        Ok(_) => {}
                        Err(e) => eprintln!("[lexicon] learn failed: {e:#}"),
                    },
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                let rev = match store.revision() {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if rev == last {
                    continue;
                }
                last = rev;
                match (store.load(), store.anchors()) {
                    (Ok(lex), Ok(anchors)) => {
                        eprintln!("[lexicon] revision {rev} loaded");
                        if let Some(w) = &to_watcher {
                            let _ = w.send(AnchorsUpdate {
                                lexicon: lex.clone(),
                                anchors: anchors.vectors,
                            });
                        }
                        if to_router.send(lex).is_err() {
                            return;
                        }
                    }
                    (Err(e), _) | (_, Err(e)) => eprintln!("[lexicon] revision {rev} not loaded: {e:#}"),
                }
            }
        })
        .expect("spawn lexicon service");
    tx
}

#[cfg(not(feature = "lexicon-db"))]
pub fn spawn_service(
    _db: PathBuf,
    _poll: std::time::Duration,
    _promote_at: u32,
    _to_router: Sender<Lexicon>,
    _to_watcher: Option<Sender<AnchorsUpdate>>,
) -> Sender<LexiconMsg> {
    std::sync::mpsc::channel().0
}
