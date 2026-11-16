//! A named SDS ledger: the persisted base plus this run's observations (logis "ledger-split / run-files").
//!
//! Every `record_*` goes into the run under the scope's full key (`track|primary|secondary`). Reads merge base and
//! run *per queried entry* — never whole scopes — and back off secondary → primary → global → unscoped, each level
//! answering only once it holds its [`MinObs`]; otherwise `None` (no judgement on too few samples). As in logis,
//! a level holds only what was recorded at exactly that key: a primary-level answer comes from observations made
//! with an empty secondary, a global one from an empty primary.
//!
//! Persistence (see `persist`): `flush` rewrites this run's file, `close` folds the run into the base file, and
//! `open` folds run files left by crashed processes. A live run holds a file lock, so concurrent processes never
//! fold each other's runs.

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};

use crate::persist::{self, BaseFile, Loaded, RunFile, BASE_CAP_BYTES};
use crate::scope::{MinObs, Scope, UNSCOPED_KEY, UNSCOPED_RING};
use crate::stats::{decay_shape, tie_verdict, ConfusionStat, DecayStat, ScopeStat, Welford};
use crate::RECIPE;

/// Which backoff level answered a read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Secondary,
    Primary,
    Global,
    Unscoped,
}

impl Level {
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Secondary => "secondary",
            Level::Primary => "primary",
            Level::Global => "global",
            Level::Unscoped => "unscoped",
        }
    }
}

/// One baseline read with backoff: which level answered and its numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Band {
    pub mean: f32,
    pub sd: f32,
    pub n: u64,
    pub recent_mean: f32,
    pub drift_z: f32,
    pub level: Level,
}

impl Band {
    /// `mean + k·sd`.
    pub fn upper(&self, k: f32) -> f32 {
        self.mean + k * self.sd
    }

    /// `mean − k·sd`.
    pub fn lower(&self, k: f32) -> f32 {
        self.mean - k * self.sd
    }
}

/// A named ledger: the persisted base + this run's observations (logis "ledger-split / run-files").
///
/// Plain data, `Send + Sync`: wrap it in a `Mutex` / `RwLock` to share. Dropping it without [`Ledger::close`] acts
/// like a crash: the last flushed run file is folded by the next [`Ledger::open`], unflushed observations are lost.
#[derive(Debug)]
pub struct Ledger {
    name: String,
    store: Option<Store>,
    min_obs: HashMap<String, MinObs>,
    base: HashMap<String, ScopeStat>,
    run: Run,
    recovered: usize,
    set_aside: Vec<PathBuf>,
    shed: usize,
}

#[derive(Debug)]
struct Store {
    base: PathBuf,
    base_lock: PathBuf,
    runs: PathBuf,
}

impl Store {
    fn new(dir: &Path, name: &str) -> Self {
        Self {
            base: dir.join(format!("{name}.json")),
            base_lock: dir.join(format!("{name}.lock")),
            runs: dir.join("runs").join(name),
        }
    }

    fn run_file(&self, id: &str) -> PathBuf {
        self.runs.join(format!("{id}.json"))
    }

    fn run_lock(&self, id: &str) -> PathBuf {
        self.runs.join(format!("{id}.lock"))
    }

    /// Deletes a finished run's file, temp file and lock file (the lock handle must already be dropped).
    fn remove_run(&self, id: &str) {
        let file = self.run_file(id);
        let _ = fs::remove_file(persist::with_suffix(&file, ".tmp"));
        let _ = fs::remove_file(&file);
        let _ = fs::remove_file(self.run_lock(id));
    }
}

#[derive(Debug)]
struct Run {
    id: String,
    started_ms: i64,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
    /// Changed since the last flush.
    dirty: bool,
    /// Held from the first flush until close: marks the run file as live.
    lock: Option<File>,
    /// The run file exists on disk.
    written: bool,
}

impl Run {
    fn fresh() -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let started_ms = now_ms();
        let id = format!("{}-{}-{}", started_ms, std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed));
        Self {
            id,
            started_ms,
            observations: 0,
            scopes: HashMap::new(),
            dirty: false,
            lock: None,
            written: false,
        }
    }
}

/// Unix time in ms.
pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with('.') || !name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')) {
        bail!("invalid SDS ledger name {name:?} (use ASCII letters, digits, '-', '_', '.')");
    }
    Ok(())
}

impl Ledger {
    fn new(name: &str, store: Option<Store>) -> Self {
        Self {
            name: name.to_string(),
            store,
            min_obs: HashMap::new(),
            base: HashMap::new(),
            run: Run::fresh(),
            recovered: 0,
            set_aside: Vec::new(),
            shed: 0,
        }
    }

    /// In memory only (tests, no files).
    pub fn in_memory(name: &str) -> Ledger {
        Self::new(name, None)
    }

    /// `<dir>/<name>.json` is the base; this run writes `<dir>/runs/<name>/<unique>.json` on flush. Leftover run files of
    /// crashed earlier runs are folded into the base on open (then deleted). Corrupt files are renamed `*.corrupt` and
    /// ignored, never panic.
    pub fn open(dir: &Path, name: &str) -> Result<Ledger> {
        Self::open_with(dir, name, &[])
    }

    /// [`Ledger::open`] with track floors set first, so crashed runs are folded with the right ring lengths.
    pub fn open_with(dir: &Path, name: &str, tracks: &[(&str, MinObs)]) -> Result<Ledger> {
        check_name(name)?;
        fs::create_dir_all(dir).with_context(|| format!("creating SDS directory {}", dir.display()))?;
        let store = Store::new(dir, name);
        let mut ledger = Self::new(name, None);
        for (track, m) in tracks {
            ledger.min_obs.insert(track.trim().to_string(), *m);
        }
        ledger.load_and_recover(&store)?;
        ledger.store = Some(store);
        Ok(ledger)
    }

    /// Reads the base and folds dead run files into it, under the base lock.
    fn load_and_recover(&mut self, store: &Store) -> Result<()> {
        let _guard = persist::lock_base(&store.base_lock);
        let mut file = self.read_base(store)?.unwrap_or_default();

        let mut dead_folded: Vec<String> = Vec::new();
        let mut newly: Vec<String> = Vec::new();
        let entries = match fs::read_dir(&store.runs) {
            Ok(e) => e.flatten().map(|e| e.path()).collect::<Vec<_>>(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).with_context(|| format!("listing {}", store.runs.display())),
        };
        for path in entries {
            let Some(id) = path
                .file_name()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_suffix(".json"))
                .map(str::to_string)
            else {
                continue;
            };
            if persist::run_is_live(&store.run_lock(&id)) {
                continue;
            }
            if file.folded.contains(&id) {
                dead_folded.push(id);
                continue;
            }
            match persist::load::<RunFile>(&path, &self.name) {
                Ok(Loaded::Valid(run)) => {
                    for (k, v) in run.scopes {
                        let ring = self.ring_for_key(&k);
                        file.scopes.entry(k).or_default().absorb(v, ring);
                    }
                    newly.push(id);
                }
                Ok(Loaded::SetAside(p)) => {
                    self.set_aside.push(p);
                    let _ = fs::remove_file(store.run_lock(&id));
                }
                // Vanished or busy: leave it for the next open.
                Ok(Loaded::Missing) | Err(_) => {}
            }
        }
        if !newly.is_empty() {
            file.folded.retain(|id| store.run_file(id).exists());
            file.folded.extend(newly.iter().cloned());
            self.write_base(store, &mut file)?;
            self.recovered += newly.len();
        }
        for id in newly.iter().chain(&dead_folded) {
            store.remove_run(id);
        }
        self.base = file.scopes;
        Ok(())
    }

    /// The base file, `None` when missing or set aside.
    fn read_base(&mut self, store: &Store) -> Result<Option<BaseFile>> {
        match persist::load::<BaseFile>(&store.base, &self.name).with_context(|| format!("reading {}", store.base.display()))? {
            Loaded::Valid(f) => Ok(Some(f)),
            Loaded::Missing => Ok(None),
            Loaded::SetAside(p) => {
                self.set_aside.push(p);
                Ok(None)
            }
        }
    }

    fn write_base(&mut self, store: &Store, file: &mut BaseFile) -> Result<()> {
        file.recipe = RECIPE.to_string();
        file.name = self.name.clone();
        file.updated_ms = now_ms();
        let (txt, shed) = persist::encode_base(file, BASE_CAP_BYTES)?;
        persist::write_atomic(&store.base, txt.as_bytes()).with_context(|| format!("writing {}", store.base.display()))?;
        self.shed += shed;
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// This run's id (its file stem under `runs/<name>/`).
    pub fn run_id(&self) -> &str {
        &self.run.id
    }

    /// The base file, `None` in memory.
    pub fn path(&self) -> Option<&Path> {
        self.store.as_ref().map(|s| s.base.as_path())
    }

    /// Corrupt (`*.corrupt`) or stale (`*.stale`, other recipe or ledger name) files this ledger renamed and ignored.
    pub fn set_aside_files(&self) -> &[PathBuf] {
        &self.set_aside
    }

    /// Sets a track's floors (and its ring length = `m.primary`); rings already held for that track are re-trimmed.
    pub fn set_min_obs(&mut self, track: &str, m: MinObs) {
        let track = track.trim().to_string();
        let ring = m.ring_len();
        for (k, st) in self.base.iter_mut().chain(self.run.scopes.iter_mut()) {
            if k != UNSCOPED_KEY && k.split('|').next() == Some(track.as_str()) {
                st.trim_rings(ring);
            }
        }
        self.min_obs.insert(track, m);
    }

    /// The track's floors; [`MinObs::default`] `(6, 3, 10)` when never set.
    pub fn min_obs(&self, track: &str) -> MinObs {
        self.min_obs.get(track.trim()).copied().unwrap_or_default()
    }

    fn ring_for_key(&self, key: &str) -> usize {
        if key == UNSCOPED_KEY {
            UNSCOPED_RING
        } else {
            self.min_obs(key.split('|').next().unwrap_or("")).ring_len()
        }
    }

    fn key_for(s: &Scope) -> String {
        if s.is_empty() {
            UNSCOPED_KEY.to_string()
        } else {
            s.key_secondary()
        }
    }

    fn with_scope_mut(&mut self, s: &Scope, f: impl FnOnce(&mut ScopeStat, usize)) {
        let key = Self::key_for(s);
        let ring = self.ring_for_key(&key);
        self.run.observations = self.run.observations.saturating_add(1);
        let e = self.run.scopes.entry(key).or_default();
        f(e, ring);
        e.updated_ms = now_ms();
        self.run.dirty = true;
    }

    // ── recording ───────────────────────────────────────────────────────────────────────────────────────────────

    /// One value on an axis (`valence`, `credit.margin`, …); non-finite values are ignored.
    pub fn record_baseline(&mut self, s: &Scope, axis: &str, value: f32) {
        if !value.is_finite() {
            return;
        }
        self.with_scope_mut(s, |st, ring| {
            st.baseline.entry(axis.to_string()).or_default().push(value as f64, ring)
        });
    }

    /// The shape of one ranked score list ([`decay_shape`]); ignored below two finite scores.
    pub fn record_decay(&mut self, s: &Scope, axis: &str, scores: &[f32]) {
        let Some(shape) = decay_shape(scores) else { return };
        self.with_scope_mut(s, |st, ring| st.decay.entry(axis.to_string()).or_default().push(&shape, ring));
    }

    /// `winner` beat `loser` by `margin` in a close meeting.
    pub fn record_confusion(&mut self, s: &Scope, winner: &str, loser: &str, margin: f32) {
        if winner.is_empty() || loser.is_empty() || winner == loser {
            return;
        }
        let winner_is_a = winner <= loser;
        let key = if winner_is_a {
            format!("{winner}|{loser}")
        } else {
            format!("{loser}|{winner}")
        };
        self.with_scope_mut(s, |st, ring| {
            let e = st.confusion.entry(key).or_default();
            e.ties += 1;
            if winner_is_a {
                e.a_wins += 1;
            } else {
                e.b_wins += 1;
            }
            if margin.is_finite() {
                e.margin.push(margin as f64, ring);
            }
        });
    }

    /// `a` and `b` met closely and neither won (counts as a meeting, lowers both win rates).
    pub fn record_tie(&mut self, s: &Scope, a: &str, b: &str, margin: f32) {
        if a.is_empty() || b.is_empty() || a == b {
            return;
        }
        let key = if a <= b { format!("{a}|{b}") } else { format!("{b}|{a}") };
        self.with_scope_mut(s, |st, ring| {
            let e = st.confusion.entry(key).or_default();
            e.ties += 1;
            if margin.is_finite() {
                e.margin.push(margin as f64, ring);
            }
        });
    }

    /// `item` was considered (the denominator of [`Ledger::learned_specificity`]).
    pub fn record_seen(&mut self, s: &Scope, item: &str) {
        if item.is_empty() {
            return;
        }
        self.with_scope_mut(s, |st, _| st.reject.entry(item.to_string()).or_default().seen += 1);
    }

    /// `item` was rejected for `kind` (free-form; empty becomes `unspecified`). Does not count as seen.
    pub fn record_reject(&mut self, s: &Scope, item: &str, kind: &str) {
        if item.is_empty() {
            return;
        }
        let kind = match kind.trim() {
            "" => "unspecified",
            k => k,
        };
        self.with_scope_mut(s, |st, _| {
            let e = st.reject.entry(item.to_string()).or_default();
            e.rejected += 1;
            *e.kinds.entry(kind.to_string()).or_insert(0) += 1;
        });
    }

    /// One move `from → to` (domain transitions for [`Ledger::transition_prior`]).
    pub fn record_transition(&mut self, s: &Scope, from: &str, to: &str) {
        if from.is_empty() || to.is_empty() {
            return;
        }
        let key = format!("{from}>{to}");
        self.with_scope_mut(s, |st, _| *st.transition.entry(key).or_insert(0) += 1);
    }

    // ── reads ───────────────────────────────────────────────────────────────────────────────────────────────────

    /// The backoff ladder `(key, floor, level)`; `None` for a scope without track (logis: no active scope).
    fn levels(&self, s: &Scope) -> Option<[(String, u64, Level); 4]> {
        if s.is_empty() {
            return None;
        }
        let m = self.min_obs(&s.track);
        Some([
            (s.key_secondary(), m.secondary, Level::Secondary),
            (s.key_primary(), m.primary, Level::Primary),
            (s.key_global(), m.global, Level::Global),
            (UNSCOPED_KEY.to_string(), m.global.max(m.primary), Level::Unscoped),
        ])
    }

    /// Reads one entry with backoff, merging base and run for that entry only (logis `resolve_entry`): entries
    /// merge independently, so this equals absorbing the whole run scope first, without cloning it. `pick` returns
    /// the value and its sample count, checked against the level's floor; a floor of 0 skips the level.
    fn resolve_entry<E: Clone, T>(
        &self,
        s: &Scope,
        get: impl Fn(&ScopeStat) -> Option<&E>,
        merge: impl Fn(&mut E, &E, usize),
        pick: impl Fn(&E) -> Option<(T, u64)>,
    ) -> Option<(T, Level)> {
        for (key, need, level) in self.levels(s)? {
            if need == 0 {
                continue;
            }
            let base = self.base.get(&key).and_then(&get);
            let run = self.run.scopes.get(&key).and_then(&get);
            let picked = match (base, run) {
                (Some(b), Some(r)) => {
                    let mut m = b.clone();
                    merge(&mut m, r, self.ring_for_key(&key));
                    pick(&m)
                }
                (Some(e), None) | (None, Some(e)) => pick(e),
                (None, None) => None,
            };
            if let Some((v, n)) = picked {
                if n >= need {
                    return Some((v, level));
                }
            }
        }
        None
    }

    /// The axis baseline of the first level with enough samples.
    pub fn baseline(&self, s: &Scope, axis: &str) -> Option<Band> {
        self.resolve_entry(
            s,
            |st| st.baseline.get(axis),
            Welford::merge,
            |w| Some(((w.mean as f32, w.sd() as f32, w.n, w.recent_mean() as f32, w.drift_z() as f32), w.n)),
        )
        .map(|((mean, sd, n, recent_mean, drift_z), level)| Band {
            mean,
            sd,
            n,
            recent_mean,
            drift_z,
            level,
        })
    }

    /// `mean + k·sd` of [`Ledger::baseline`].
    pub fn upper_bound(&self, s: &Scope, axis: &str, k: f32) -> Option<f32> {
        self.baseline(s, axis).map(|b| b.upper(k))
    }

    /// `mean − k·sd` of [`Ledger::baseline`].
    pub fn lower_bound(&self, s: &Scope, axis: &str, k: f32) -> Option<f32> {
        self.baseline(s, axis).map(|b| b.lower(k))
    }

    /// The usual decay shape: (flatness mean, flatness sd, margin mean, margin sd).
    pub fn decay_baseline(&self, s: &Scope, axis: &str) -> Option<(f32, f32, f32, f32)> {
        self.resolve_entry(
            s,
            |st| st.decay.get(axis),
            DecayStat::merge_from,
            |d| {
                let v = (
                    d.tail_flatness.mean as f32,
                    d.tail_flatness.sd() as f32,
                    d.margin.mean as f32,
                    d.margin.sd() as f32,
                );
                Some((v, d.margin.n))
            },
        )
        .map(|(v, _)| v)
    }

    /// Historical winner of the pair: (winner, its wins over all meetings, meetings).
    pub fn confusion_winner(&self, s: &Scope, a: &str, b: &str) -> Option<(String, f32, u64)> {
        if a.is_empty() || b.is_empty() || a == b {
            return None;
        }
        let (x, y) = if a <= b { (a, b) } else { (b, a) };
        let key = format!("{x}|{y}");
        self.resolve_entry(
            s,
            |st| st.confusion.get(&key),
            ConfusionStat::merge_from,
            |c: &ConfusionStat| {
                let total = c.ties.max(1) as f32;
                let (w, wins) = if c.a_wins >= c.b_wins { (x, c.a_wins) } else { (y, c.b_wins) };
                Some(((w.to_string(), wins as f32 / total, c.ties), c.ties))
            },
        )
        .map(|(v, _)| v)
    }

    /// Learned penalty: the share of `item`'s considerations that ended in rejection (needs `seen` ≥ the floor).
    pub fn learned_specificity(&self, s: &Scope, item: &str) -> Option<f32> {
        self.resolve_entry(
            s,
            |st| st.reject.get(item),
            |a, b, r| a.merge_from(b, r),
            |r| Some((r.specificity(), r.seen)),
        )
        .map(|(v, _)| v)
    }

    /// Whether a tie between `item` and `rival` may go to `item` (logis `tie_break_allows` → [`tie_verdict`]).
    /// Callers usually refuse [`crate::TieKind::Flat`] ties before asking.
    pub fn tie_break_allows(&self, s: &Scope, item: &str, rival: &str) -> bool {
        let history = self
            .confusion_winner(s, item, rival)
            .map(|(winner, rate, _)| (winner == item, rate));
        tie_verdict(history, self.learned_specificity(s, item), self.learned_specificity(s, rival))
    }

    /// Laplace-smoothed `P(to | from) = (hit + 1) / (total + k)`, `k` = distinct transitions in the scope. Secondary
    /// then primary level only; `None` while `from` has fewer departures than the level's floor.
    pub fn transition_prior(&self, s: &Scope, from: &str, to: &str) -> Option<f32> {
        if s.is_empty() {
            return None;
        }
        let m = self.min_obs(&s.track);
        let prefix = format!("{from}>");
        let hit_key = format!("{from}>{to}");
        for (key, need) in [(s.key_secondary(), m.secondary), (s.key_primary(), m.primary)] {
            let base = self.base.get(&key).map(|st| &st.transition);
            let run = self.run.scopes.get(&key).map(|st| &st.transition);
            if base.is_none() && run.is_none() {
                continue;
            }
            let maps = [base, run];
            let total: u64 = maps
                .iter()
                .flatten()
                .flat_map(|t| t.iter())
                .filter(|(k, _)| k.starts_with(&prefix))
                .map(|(_, v)| *v)
                .sum();
            if total < need {
                continue;
            }
            let hit: u64 = maps.iter().flatten().filter_map(|t| t.get(&hit_key)).sum();
            let distinct = match (base, run) {
                (Some(b), Some(r)) => b.len() + r.keys().filter(|k| !b.contains_key(*k)).count(),
                (Some(t), None) | (None, Some(t)) => t.len(),
                (None, None) => 0,
            };
            let k = distinct.max(1) as f64;
            return Some((((hit as f64) + 1.0) / ((total as f64) + k)) as f32);
        }
        None
    }

    // ── run management ──────────────────────────────────────────────────────────────────────────────────────────

    /// A more precise scope became known (logis `refine_primary`): move this run's observations from `old` to `new`.
    /// Returns how many entries moved; the base is untouched.
    pub fn refine(&mut self, old: &Scope, new: &Scope) -> usize {
        let (ok, nk) = (Self::key_for(old), Self::key_for(new));
        if ok == nk {
            return 0;
        }
        let Some(prev) = self.run.scopes.remove(&ok) else { return 0 };
        if prev.is_empty() {
            return 0;
        }
        let moved = prev.baseline.len() + prev.decay.len() + prev.confusion.len() + prev.reject.len() + prev.transition.len();
        let ring = self.ring_for_key(&nk);
        self.run.scopes.entry(nk).or_default().absorb(prev, ring);
        self.run.dirty = true;
        moved
    }

    /// Observations recorded in this run.
    pub fn observations(&self) -> u64 {
        self.run.observations
    }

    /// Write this run's file atomically (tmp + rename). No-op in memory or when nothing changed.
    pub fn flush(&mut self) -> Result<()> {
        let Some(store) = &self.store else { return Ok(()) };
        if !self.run.dirty {
            return Ok(());
        }
        fs::create_dir_all(&store.runs).with_context(|| format!("creating {}", store.runs.display()))?;
        if self.run.lock.is_none() {
            let lock = store.run_lock(&self.run.id);
            self.run.lock = Some(persist::hold_run_lock(&lock).with_context(|| format!("locking {}", lock.display()))?);
        }
        let out = persist::RunFileOut {
            recipe: RECIPE,
            name: &self.name,
            run: &self.run.id,
            started_ms: self.run.started_ms,
            updated_ms: now_ms(),
            observations: self.run.observations,
            scopes: &self.run.scopes,
        };
        let txt = serde_json::to_string_pretty(&out)?;
        let path = store.run_file(&self.run.id);
        persist::write_atomic(&path, txt.as_bytes()).with_context(|| format!("writing {}", path.display()))?;
        self.run.written = true;
        self.run.dirty = false;
        Ok(())
    }

    /// Fold the run into the base, write the base atomically, delete the run file, start a fresh run.
    ///
    /// The base is re-read from disk under its lock first, so runs other processes closed meanwhile are kept. On a
    /// write error the run stays as it was (retry later).
    pub fn close(&mut self) -> Result<()> {
        let Some(store) = self.store.take() else {
            for (k, v) in std::mem::take(&mut self.run.scopes) {
                let ring = self.ring_for_key(&k);
                self.base.entry(k).or_default().absorb(v, ring);
            }
            self.run = Run::fresh();
            return Ok(());
        };
        let result = self.close_file(&store);
        self.store = Some(store);
        result
    }

    fn close_file(&mut self, store: &Store) -> Result<()> {
        if self.run.scopes.is_empty() && !self.run.written {
            self.run = Run::fresh();
            return Ok(());
        }
        let guard = persist::lock_base(&store.base_lock);
        let mut file =
            match persist::load::<BaseFile>(&store.base, &self.name).with_context(|| format!("reading {}", store.base.display()))? {
                Loaded::Valid(f) => f,
                Loaded::Missing => BaseFile::default(),
                // Unreadable on disk: what this process loaded is the best base left.
                Loaded::SetAside(p) => {
                    self.set_aside.push(p);
                    BaseFile {
                        scopes: self.base.clone(),
                        ..Default::default()
                    }
                }
            };
        for (k, v) in &self.run.scopes {
            let ring = self.ring_for_key(k);
            file.scopes.entry(k.clone()).or_default().absorb(v.clone(), ring);
        }
        file.folded.retain(|id| store.run_file(id).exists());
        if self.run.written && !file.folded.contains(&self.run.id) {
            file.folded.push(self.run.id.clone());
        }
        self.write_base(store, &mut file)?;
        drop(guard);

        let old = std::mem::replace(&mut self.run, Run::fresh());
        let had_file = old.written;
        drop(old.lock);
        if had_file {
            store.remove_run(&old.id);
        }
        self.base = file.scopes;
        Ok(())
    }

    /// One-line human summary for logs (like logis `report`).
    pub fn report(&self) -> String {
        let all = || self.base.values().chain(self.run.scopes.values());
        let scopes = self.base.len() + self.run.scopes.keys().filter(|k| !self.base.contains_key(*k)).count();
        let baseline: u64 = all().flat_map(|s| s.baseline.values()).map(|w| w.n).sum();
        let decay: u64 = all().flat_map(|s| s.decay.values()).map(|d| d.margin.n).sum();
        let confusion: u64 = all().flat_map(|s| s.confusion.values()).map(|c| c.ties).sum();
        let seen: u64 = all().flat_map(|s| s.reject.values()).map(|r| r.seen).sum();
        let rejected: u64 = all().flat_map(|s| s.reject.values()).map(|r| r.rejected).sum();
        let transition: u64 = all().flat_map(|s| s.transition.values()).sum();
        let mut extra = String::new();
        if self.store.is_none() {
            extra.push_str(" | in memory");
        } else if self.run.dirty {
            extra.push_str(" (unflushed)");
        }
        if self.recovered > 0 {
            extra.push_str(&format!(" | recovered {} crashed run(s)", self.recovered));
        }
        if !self.set_aside.is_empty() {
            extra.push_str(&format!(" | set aside {} unreadable file(s)", self.set_aside.len()));
        }
        if self.shed > 0 {
            extra.push_str(&format!(" | shed {} scope(s) over the size cap", self.shed));
        }
        format!(
            "[SDS REPORT] {}(scopes {} | baseline {} | decay {} | confusion {} | reject {}/{} seen | transition {}) | run {} obs{}",
            self.name, scopes, baseline, decay, confusion, rejected, seen, transition, self.run.observations, extra
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::tests::lcg;

    fn sc(p: &str, s: &str) -> Scope {
        Scope::new("drop", p, s)
    }

    #[test]
    fn ledger_is_send_and_sync() {
        fn check<T: Send + Sync>() {}
        check::<Ledger>();
    }

    #[test]
    fn backoff_walks_secondary_primary_global_then_none() {
        let mut l = Ledger::in_memory("drop");
        l.set_min_obs("drop", MinObs::new(4, 3, 5));
        let topic = sc("app://eldenring.exe", "boss-malenia");
        assert_eq!(l.baseline(&topic, "valence"), None, "no samples, no verdict");

        for v in [10.0, 12.0, 14.0, 16.0, 18.0] {
            l.record_baseline(&Scope::new("drop", "", ""), "valence", v);
        }
        assert_eq!(l.baseline(&topic, "valence").unwrap().level, Level::Global);
        assert_eq!(l.baseline(&topic, "valence").unwrap().n, 5);

        for v in [1.0, 2.0] {
            l.record_baseline(&sc("app://eldenring.exe", ""), "valence", v);
        }
        assert_eq!(l.baseline(&topic, "valence").unwrap().level, Level::Global, "primary has 2 < 3");
        l.record_baseline(&sc("app://eldenring.exe", ""), "valence", 3.0);
        let b = l.baseline(&topic, "valence").unwrap();
        assert_eq!((b.level, b.n, b.mean), (Level::Primary, 3, 2.0));

        for v in [0.1, 0.2, 0.3] {
            l.record_baseline(&topic, "valence", v);
        }
        assert_eq!(l.baseline(&topic, "valence").unwrap().level, Level::Primary, "secondary has 3 < 4");
        l.record_baseline(&topic, "valence", 0.4);
        let b = l.baseline(&topic, "valence").unwrap();
        assert_eq!((b.level, b.n), (Level::Secondary, 4));
        assert!((b.mean - 0.25).abs() < 1e-6);
        let up = l.upper_bound(&topic, "valence", 2.0).unwrap();
        let lo = l.lower_bound(&topic, "valence", 2.0).unwrap();
        assert!((up - (b.mean + 2.0 * b.sd)).abs() < 1e-6 && (lo - (b.mean - 2.0 * b.sd)).abs() < 1e-6);

        // Other axes, scopes and tracks do not leak.
        assert_eq!(l.baseline(&topic, "credit.margin"), None);
        assert_eq!(
            l.baseline(&Scope::new("web", "app://eldenring.exe", "boss-malenia"), "valence"),
            None
        );
        assert_eq!(
            l.baseline(&Scope::new("", "x", "y"), "valence"),
            None,
            "trackless scope never reads"
        );
    }

    #[test]
    fn unscoped_is_the_last_level_and_zero_floors_skip() {
        let mut l = Ledger::in_memory("nms");
        l.set_min_obs("nms", MinObs::new(2, 2, 0));
        for v in [0.5, 0.6] {
            l.record_baseline(&Scope::new("", "", ""), "dup_cos", v);
        }
        let s = Scope::new("nms", "a", "b");
        assert_eq!(
            l.baseline(&s, "dup_cos").unwrap().level,
            Level::Unscoped,
            "unscoped floor = max(global 0, primary 2)"
        );
        for v in [0.1, 0.2, 0.3] {
            l.record_baseline(&Scope::new("nms", "", ""), "dup_cos", v);
        }
        assert_eq!(
            l.baseline(&s, "dup_cos").unwrap().level,
            Level::Unscoped,
            "global floor 0 disables the global level"
        );
    }

    #[test]
    fn base_plus_run_read_equals_absorb() {
        let mut l = Ledger::in_memory("drop");
        let s = sc("app://game.exe", "topic");
        let xs = lcg(7, 40);
        for x in &xs[..25] {
            l.record_baseline(&s, "valence", *x as f32);
            l.record_decay(&s, "credit", &[*x as f32, 0.5, 0.1, -0.2]);
        }
        l.close().unwrap();
        assert_eq!(l.observations(), 0);
        for x in &xs[25..] {
            l.record_baseline(&s, "valence", *x as f32);
            l.record_decay(&s, "credit", &[*x as f32, 0.4, 0.0]);
        }
        let read = l.baseline(&s, "valence").unwrap();
        let read_decay = l.decay_baseline(&s, "credit").unwrap();

        let key = s.key_secondary();
        let mut merged = l.base[&key].clone();
        merged.absorb(l.run.scopes[&key].clone(), l.ring_for_key(&key));
        let w = &merged.baseline["valence"];
        assert_eq!(read.n, 40);
        assert_eq!(
            (read.mean, read.sd, read.recent_mean, read.drift_z),
            (w.mean as f32, w.sd() as f32, w.recent_mean() as f32, w.drift_z() as f32)
        );
        let d = &merged.decay["credit"];
        assert_eq!(
            read_decay,
            (
                d.tail_flatness.mean as f32,
                d.tail_flatness.sd() as f32,
                d.margin.mean as f32,
                d.margin.sd() as f32
            )
        );
        assert_eq!(w.ring.len(), 3, "ring = default primary floor");

        // Closing again folds the rest: the base alone now answers the same.
        l.close().unwrap();
        assert_eq!(l.baseline(&s, "valence").unwrap(), read);
    }

    #[test]
    fn confusion_winner_and_tie_break_with_specificity() {
        let mut l = Ledger::in_memory("drop");
        l.set_min_obs("drop", MinObs::new(4, 4, 4));
        let s = sc("app://game.exe", "boss");
        assert_eq!(l.confusion_winner(&s, "r2", "r1"), None);
        assert!(!l.tie_break_allows(&s, "r2", "r1"));
        for _ in 0..3 {
            l.record_confusion(&s, "r2", "r1", 0.02);
        }
        l.record_tie(&s, "r1", "r2", 0.0);
        assert_eq!(
            l.confusion_winner(&s, "r1", "r2"),
            Some(("r2".to_string(), 0.75, 4)),
            "3 wins over 4 meetings"
        );
        assert_eq!(l.confusion_winner(&s, "r2", "r1"), l.confusion_winner(&s, "r1", "r2"), "order-free");
        // 0.75 ≥ TIE_WIN_RATE.
        assert!(l.tie_break_allows(&s, "r2", "r1"), "specificity unknown: rate alone");
        assert!(!l.tie_break_allows(&s, "r1", "r2"), "the loser never gets the tie");

        // r2 rejected 2/4, r1 never: 0.5 > 0 + slack → refused.
        for _ in 0..4 {
            l.record_seen(&s, "r1");
            l.record_seen(&s, "r2");
        }
        l.record_reject(&s, "r2", "demoted");
        l.record_reject(&s, "r2", "");
        assert_eq!(l.learned_specificity(&s, "r2"), Some(0.5));
        assert_eq!(l.learned_specificity(&s, "r1"), Some(0.0));
        // 0.5 > 0.0 + TIE_SPECIFICITY_SLACK.
        assert!(!l.tie_break_allows(&s, "r2", "r1"));
        assert_eq!(l.run.scopes[&s.key_secondary()].reject["r2"].kinds.get("unspecified"), Some(&1));

        // r1 rejected 2/4 as well: equal specificity, within slack → allowed again.
        l.record_reject(&s, "r1", "outcome_failed");
        l.record_reject(&s, "r1", "outcome_failed");
        assert!(l.tie_break_allows(&s, "r2", "r1"));

        // A tie record dilutes the win rate below the threshold.
        l.record_tie(&s, "r1", "r2", 0.0);
        assert_eq!(l.confusion_winner(&s, "r1", "r2").unwrap().1, 0.6);
        assert!(!l.tie_break_allows(&s, "r2", "r1"));
        assert_eq!(l.confusion_winner(&s, "r1", "r1"), None);
    }

    #[test]
    fn transition_prior_is_laplace_smoothed_and_backs_off_once() {
        let mut l = Ledger::in_memory("web");
        l.set_min_obs("web", MinObs::new(4, 2, 0));
        let topic = Scope::new("web", "web://store", "malenia");
        let domain = Scope::new("web", "web://store", "");
        assert_eq!(l.transition_prior(&topic, "a", "b"), None);
        l.record_transition(&domain, "a", "b");
        l.record_transition(&domain, "a", "c");
        l.record_transition(&domain, "x", "y");
        // Primary level: from 'a' total 2 ≥ 2; k = 3 distinct keys; hit a>b = 1 → (1+1)/(2+3).
        assert!((l.transition_prior(&topic, "a", "b").unwrap() - 0.4).abs() < 1e-6);
        assert!(
            (l.transition_prior(&topic, "a", "z").unwrap() - 0.2).abs() < 1e-6,
            "unseen target keeps mass"
        );
        assert_eq!(l.transition_prior(&topic, "x", "y"), None, "from 'x' has 1 < 2");

        for _ in 0..3 {
            l.record_transition(&topic, "a", "b");
        }
        assert!(
            (l.transition_prior(&topic, "a", "b").unwrap() - 0.4).abs() < 1e-6,
            "secondary 3 < 4: still primary"
        );
        l.close().unwrap();
        l.record_transition(&topic, "a", "d");
        // Secondary: base a>b 3 + run a>d 1 = 4; k = 2; hit 3 → 4/6.
        assert!((l.transition_prior(&topic, "a", "b").unwrap() - 4.0 / 6.0).abs() < 1e-6);
        assert_eq!(
            l.transition_prior(&Scope::new("web", "", ""), "a", "b"),
            None,
            "global is not a level for priors"
        );
    }

    #[test]
    fn refine_moves_this_runs_observations() {
        let mut l = Ledger::in_memory("drop");
        let loading = sc("app://game.exe", "loading");
        let real = sc("app://game.exe", "boss-malenia");
        l.record_baseline(&real, "valence", 1.0);
        l.close().unwrap();
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] {
            l.record_baseline(&loading, "valence", v);
        }
        l.record_confusion(&loading, "r1", "r2", 0.1);
        assert_eq!(l.observations(), 7);
        assert_eq!(l.refine(&loading, &loading), 0);
        assert_eq!(l.refine(&loading, &real), 2);
        assert_eq!(l.refine(&loading, &real), 0, "nothing left at the old key");
        assert_eq!(l.observations(), 7);
        assert!(l.baseline(&loading, "valence").is_none());
        let b = l.baseline(&real, "valence").unwrap();
        assert_eq!((b.n, b.level), (7, Level::Secondary), "merged with the base at the new key");
        assert_eq!(l.confusion_winner(&real, "r1", "r2"), None, "1 meeting < 6");
        assert!(l.run.scopes[&real.key_secondary()].confusion.contains_key("r1|r2"));
    }

    #[test]
    fn set_min_obs_retrims_rings_and_report_counts() {
        let mut l = Ledger::in_memory("drop");
        let s = sc("p", "s");
        for v in 0..5 {
            l.record_baseline(&s, "valence", v as f32);
        }
        assert_eq!(l.run.scopes[&s.key_secondary()].baseline["valence"].ring.len(), 3);
        l.set_min_obs("drop", MinObs::new(1, 2, 1));
        assert_eq!(l.run.scopes[&s.key_secondary()].baseline["valence"].ring, vec![3.0, 4.0]);
        assert_eq!(l.min_obs("drop"), MinObs::new(1, 2, 1));
        assert_eq!(l.min_obs("other"), MinObs::default());

        l.record_decay(&s, "credit", &[1.0]);
        l.record_baseline(&s, "valence", f32::NAN);
        assert_eq!(l.observations(), 5, "ignored records are not observations");
        l.record_seen(&s, "r1");
        l.record_reject(&s, "r1", "format");
        l.record_transition(&s, "a", "b");
        let r = l.report();
        assert!(r.starts_with("[SDS REPORT] drop(scopes 1 | baseline 5 |"), "{r}");
        assert!(
            r.contains("reject 1/1 seen") && r.contains("transition 1") && r.contains("run 8 obs") && r.contains("in memory"),
            "{r}"
        );
    }
}
