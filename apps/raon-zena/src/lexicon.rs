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

/// Phase 6 Part 2 (`CN2-75` / `PF-09`): 0.7B TextTower 백그라운드 재임베딩을 허용하기 위한 최소 가용 RAM(MiB).
pub const REANCHOR_MIN_AVAIL_RAM_MB: u64 = 650;

/// Phase 6 Part 2 (`CN2-75`): 타임트라이얼 주행 중·고긴장 전투 중·저사양 메모리 부족(<650 MiB)일 때는
/// 0.7B TextTower 로딩을 유휴 시점까지 지연시킨다.
pub fn should_run_deferred_reanchor(
    trial_active: bool,
    high_tension_combat: bool,
    avail_ram_mb: Option<u64>,
) -> bool {
    if trial_active || high_tension_combat {
        return false;
    }
    avail_ram_mb.is_none_or(|mb| mb >= REANCHOR_MIN_AVAIL_RAM_MB)
}

/// Phase 6 Part 2 (`CN2-75`): TextTower 를 깨우지 않고 1152-d 이미지 임베딩만으로
/// 시각 프로토타입을 누적·합의(독립 3회 연속 `cos >= 0.90`, ±10초 프라이버시 버퍼 제외)하는 축적기.
#[derive(Debug, Clone)]
pub struct VisualPrototypeAccumulator {
    min_runs: usize,
    min_cos: f32,
    privacy_guard_ms: u64,
    last_privacy_ms: Option<u64>,
    buckets: HashMap<String, Vec<Vec<f32>>>,
}

impl Default for VisualPrototypeAccumulator {
    fn default() -> Self {
        Self {
            min_runs: 3,
            min_cos: 0.90,
            privacy_guard_ms: 10_000,
            last_privacy_ms: None,
            buckets: HashMap::new(),
        }
    }
}

impl VisualPrototypeAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn note_privacy_event(&mut self, now_ms: u64) {
        self.last_privacy_ms = Some(now_ms);
        self.buckets.clear();
    }

    /// 깔끔한 비프라이버시 주행에서 관측된 1152-d 단위 이미지 벡터를 누적하고,
    /// `min_runs`(3회) 이상 `cos >= 0.90` 일관성이 확인되면 L2 정규화 프로토타입을 반환한다.
    pub fn observe(&mut self, vision_key: &str, unit_vec: &[f32], now_ms: u64) -> Option<Vec<f32>> {
        if vision_key.contains("privacy") || vision_key.contains("private") || unit_vec.is_empty() {
            return None;
        }
        if let Some(p_ms) = self.last_privacy_ms {
            if now_ms.saturating_sub(p_ms) <= self.privacy_guard_ms {
                return None;
            }
        }
        let norm = unit_vec.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if norm <= 1e-6 || !norm.is_finite() {
            return None;
        }
        let norm_v: Vec<f32> = unit_vec.iter().map(|&x| (x as f64 / norm) as f32).collect();
        let entry = self.buckets.entry(vision_key.to_string()).or_default();
        if let Some(first) = entry.first() {
            if first.len() != norm_v.len() {
                entry.clear();
            } else {
                let cos: f32 = first.iter().zip(&norm_v).map(|(a, b)| a * b).sum();
                if cos < self.min_cos {
                    entry.clear();
                }
            }
        }
        entry.push(norm_v);
        if entry.len() < self.min_runs {
            return None;
        }
        let dims = entry[0].len();
        let mut mean = vec![0.0f64; dims];
        for v in entry.iter() {
            for (m, &x) in mean.iter_mut().zip(v.iter()) {
                *m += x as f64;
            }
        }
        let mnorm = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mnorm <= 1e-12 {
            return None;
        }
        let proto: Vec<f32> = mean.into_iter().map(|x| (x / mnorm) as f32).collect();
        entry.clear();
        Some(proto)
    }
}

/// Something the pipeline asks the dictionary service to do.
#[cfg_attr(not(feature = "lexicon-db"), allow(dead_code))]
pub enum LexiconMsg {
    /// Legacy learn message (routes through safe 4-gate pipeline when metadata is provided).
    #[allow(dead_code)]
    Learn { key: String, span: String },
    /// Phase 6 Part 2: 4-게이트 안전 자동학습(Tier 2 스코프 바인딩 및 Command Air-Gap 포함).
    LearnSafe {
        cc: Option<String>,
        key: String,
        span: String,
        session_id: String,
        score: f32,
        margin: f32,
        outcome_sign: i32,
    },
    /// Phase 6 Part 2: TextTower 없이 이미지 프로토타입 벡터 직접 주입(Privacy 불변 보호).
    #[allow(dead_code)]
    UpsertPrototype {
        vision_key: String,
        prototype: Vec<f32>,
        model: String,
    },
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
            // Phase 6 Part 2: 저사양 게임 중 0.7B TextTower 기동 차단 — 가용 RAM >= 650 MiB 유휴 조건에서만 재임베딩.
            if let Some(r) = &reanchor {
                if should_run_deferred_reanchor(false, false, crate::pd_bridge::avail_ram_mb()) {
                    if let Err(e) = reanchor_if_stale(&store, r) {
                        eprintln!("[lexicon] anchors not recomputed for {}: {e:#}", r.model.display());
                    }
                } else {
                    eprintln!("[lexicon] deferred reanchor skipped (low available RAM < {REANCHOR_MIN_AVAIL_RAM_MB} MiB)");
                }
            }
            loop {
                match rx.recv_timeout(poll) {
                    Ok(LexiconMsg::Learn { key, span }) => match store.learn(&key, &span, promote_at) {
                        Ok(Learned::Promoted) => eprintln!("[lexicon] learned '{span}' for {key}"),
                        Ok(_) => {}
                        Err(e) => eprintln!("[lexicon] learn failed: {e:#}"),
                    },
                    Ok(LexiconMsg::LearnSafe {
                        cc,
                        key,
                        span,
                        session_id,
                        score,
                        margin,
                        outcome_sign,
                    }) => match store.learn_safe(
                        cc.as_deref(),
                        &key,
                        &span,
                        &session_id,
                        score,
                        margin,
                        outcome_sign,
                    ) {
                        Ok(zena_lexicon::SafeLearnOutcome::Promoted) => {
                            eprintln!("[lexicon] safe-learned '{span}' for {key} (cc={cc:?})")
                        }
                        Ok(_) => {}
                        Err(e) => eprintln!("[lexicon] learn_safe failed: {e:#}"),
                    },
                    Ok(LexiconMsg::UpsertPrototype {
                        vision_key,
                        prototype,
                        model,
                    }) => match store.upsert_visual_prototype(&vision_key, &prototype, &model) {
                        Ok(true) => eprintln!("[lexicon] visual prototype updated for {vision_key}"),
                        Ok(false) => eprintln!("[lexicon] visual prototype blocked (immutable privacy): {vision_key}"),
                        Err(e) => eprintln!("[lexicon] visual prototype failed: {e:#}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cn275_visual_prototype_accumulator_and_deferred_reanchor_guard_never_wake_text_tower_in_combat() {
        // Deferred reanchor blocks during active trial, combat, or low RAM (< 650 MiB).
        assert!(!should_run_deferred_reanchor(true, false, Some(2048)));
        assert!(!should_run_deferred_reanchor(false, true, Some(2048)));
        assert!(!should_run_deferred_reanchor(false, false, Some(512)));
        assert!(should_run_deferred_reanchor(false, false, Some(1024)));

        // VisualPrototypeAccumulator produces a unit prototype after 3 consistent clean runs (cos >= 0.90)
        // and blocks privacy screens and +-10s privacy buffer windows!
        let mut acc = VisualPrototypeAccumulator::new();
        let v1 = vec![0.99f32, 0.141, 0.0];
        let v2 = vec![0.985f32, 0.172, 0.0];
        let v3 = vec![0.992f32, 0.126, 0.0];
        assert!(acc.observe("vision.private", &v1, 1_000).is_none());
        acc.note_privacy_event(5_000);
        assert!(acc.observe("vision.game_battle", &v1, 10_000).is_none(), "inside 10s privacy buffer");
        assert!(acc.observe("vision.game_battle", &v1, 20_000).is_none());
        assert!(acc.observe("vision.game_battle", &v2, 25_000).is_none());
        let proto = acc.observe("vision.game_battle", &v3, 30_000).expect("3 clean consistent runs emit prototype");
        let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }
}
