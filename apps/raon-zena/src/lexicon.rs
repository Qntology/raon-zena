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
    /// The vision checkpoint the anchors were computed with (`raon_vision::fingerprint`), empty if none.
    pub anchors_model: String,
    pub source: String,
}

/// What the dictionary service needs to recompute anchors itself when the screen model changes.
#[derive(Clone, Debug)]
pub struct Reanchor {
    /// The screen watcher's vision checkpoint (`--screen`).
    pub model: PathBuf,
    /// Its SigLIP2 tokenizer: `--screen-tokenizer`, else `tokenizer.json` next to the checkpoint.
    pub tokenizer: PathBuf,
}

impl Reanchor {
    /// `tokenizer` or the checkpoint folder's `tokenizer.json` (google's SigLIP2 ships one there); `None` when
    /// neither exists.
    pub fn find(model: &Path, tokenizer: Option<&Path>) -> Option<Reanchor> {
        let dir = if model.is_dir() { model.to_path_buf() } else { model.parent()?.to_path_buf() };
        let tok = tokenizer.map(Path::to_path_buf).unwrap_or_else(|| dir.join("tokenizer.json"));
        tok.is_file().then(|| Reanchor {
            model: model.to_path_buf(),
            tokenizer: tok,
        })
    }
}

/// When the stored anchors were computed by another checkpoint than `r.model` (Raon-VE swapped for google's
/// SigLIP2, or a new download), every screen prompt is embedded again with `r.model`'s text tower (CPU, Q8_0: it is
/// loaded for this only and dropped) and stored under the new fingerprint. `Ok(None)` when they already match.
#[cfg(feature = "lexicon-db")]
pub fn reanchor_if_stale(store: &zena_lexicon::LexiconStore, r: &Reanchor) -> Result<Option<usize>> {
    let want = raon_vision::fingerprint(&r.model);
    let have = store.anchors()?;
    let lex = store.load()?;
    let phrases = lex.anchor_phrases();
    if phrases.is_empty() || (have.model == want && phrases.iter().all(|p| have.vectors.contains_key(p))) {
        return Ok(None);
    }
    eprintln!(
        "[lexicon] screen anchors were computed by '{}', the screen model is '{want}': embedding {} prompts again",
        if have.model.is_empty() { "nothing" } else { &have.model },
        phrases.len()
    );
    let t = std::time::Instant::now();
    let text = raon_vision::text::TextTower::load(
        &raon_vision::checkpoint(&r.model),
        &raon_infer::device::select_device(raon_infer::device::Backend::Cpu, 0)?,
        raon_vision::WeightFormat::Q8,
    )?;
    let tok = raon_vision::text::SiglipTokenizer::load(&r.tokenizer)?;
    let anchors = raon_vision::text::compute_anchors(&text, &tok, &lex.prompting, &phrases)?;
    drop(text);
    let n = store.set_anchors(&want, &anchors)?;
    eprintln!("[lexicon] {n} anchors recomputed in {:.1}s", t.elapsed().as_secs_f32());
    Ok(Some(n))
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
        anchors_model: anchors.model,
    })
}

#[cfg(not(feature = "lexicon-db"))]
pub fn open(_db: &Path, seed: Option<&Path>) -> Result<Loaded> {
    Ok(Loaded {
        lexicon: read_json(seed)?,
        anchors: HashMap::new(),
        anchors_model: String::new(),
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
    reanchor: Option<Reanchor>,
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
            // Once per service: a failed recompute is reported, not retried every poll.
            if let Some(r) = &reanchor {
                if let Err(e) = reanchor_if_stale(&store, r) {
                    eprintln!("[lexicon] anchors not recomputed for {}: {e:#}", r.model.display());
                }
            }
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
                                model: anchors.model,
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
    _reanchor: Option<Reanchor>,
) -> Sender<LexiconMsg> {
    std::sync::mpsc::channel().0
}
