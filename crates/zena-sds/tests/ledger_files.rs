//! On-disk ledger behaviour: run files, crash recovery, close, corrupt and stale files, concurrent ledgers.

use std::fs;
use std::path::{Path, PathBuf};

use zena_sds::{Ledger, Level, MinObs, Scope, RECIPE};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let p = std::env::temp_dir().join(format!("zena-sds-test-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&p);
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn topic() -> Scope {
    Scope::new("drop", "app://eldenring.exe", "boss-malenia")
}

fn run_dir(dir: &Path) -> PathBuf {
    dir.join("runs").join("drop")
}

fn files_with(dir: &Path, ext: &str) -> Vec<PathBuf> {
    match fs::read_dir(dir) {
        Ok(e) => e
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.to_string_lossy().ends_with(ext))
            .collect(),
        Err(_) => Vec::new(),
    }
}

fn record(l: &mut Ledger, values: &[f32]) {
    for v in values {
        l.record_baseline(&topic(), "valence", *v);
    }
}

#[test]
fn crashed_run_is_folded_on_open() {
    let t = TempDir::new("crash");
    let mut l = Ledger::open(&t.0, "drop").unwrap();
    record(&mut l, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]);
    l.flush().unwrap();
    let run_file = run_dir(&t.0).join(format!("{}.json", l.run_id()));
    assert!(run_file.exists());
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&run_file).unwrap()).unwrap();
    assert_eq!(
        (v["recipe"].as_str(), v["name"].as_str(), v["observations"].as_u64()),
        (Some(RECIPE), Some("drop"), Some(7))
    );
    assert!(!t.0.join("drop.json").exists(), "flush writes only the run file");

    record(&mut l, &[9.0]); // never flushed: lost in the crash
    drop(l);

    let l = Ledger::open(&t.0, "drop").unwrap();
    let b = l.baseline(&topic(), "valence").unwrap();
    assert_eq!((b.n, b.level), (7, Level::Secondary));
    assert!((b.mean - 0.4).abs() < 1e-6);
    assert_eq!(l.observations(), 0);
    assert!(files_with(&run_dir(&t.0), ".json").is_empty(), "folded run file deleted");
    assert!(files_with(&run_dir(&t.0), ".lock").is_empty());
    assert!(t.0.join("drop.json").exists(), "base written while folding");
    assert!(l.report().contains("recovered 1 crashed run"), "{}", l.report());
}

#[test]
fn close_writes_base_and_removes_run_file() {
    let t = TempDir::new("close");
    let mut l = Ledger::open(&t.0, "drop").unwrap();
    l.flush().unwrap();
    assert!(!run_dir(&t.0).exists(), "nothing recorded, nothing written");
    record(&mut l, &[1.0, 2.0, 3.0]);
    l.flush().unwrap();
    let first = run_dir(&t.0).join(format!("{}.json", l.run_id()));
    fs::remove_file(&first).unwrap();
    l.flush().unwrap();
    assert!(!first.exists(), "flush is a no-op when nothing changed");
    record(&mut l, &[4.0, 5.0, 6.0]);
    l.flush().unwrap();
    assert!(first.exists());

    let old_run = l.run_id().to_string();
    l.close().unwrap();
    assert_ne!(l.run_id(), old_run, "a fresh run starts");
    assert_eq!(l.observations(), 0);
    assert!(files_with(&run_dir(&t.0), ".json").is_empty() && files_with(&run_dir(&t.0), ".lock").is_empty());
    let v: serde_json::Value = serde_json::from_slice(&fs::read(t.0.join("drop.json")).unwrap()).unwrap();
    assert_eq!((v["recipe"].as_str(), v["name"].as_str()), (Some(RECIPE), Some("drop")));
    assert_eq!(v["scopes"][topic().key_secondary()]["baseline"]["valence"]["n"].as_u64(), Some(6));
    assert_eq!(l.baseline(&topic(), "valence").unwrap().n, 6, "reads keep working from the base");

    l.close().unwrap(); // empty run: no-op
    record(&mut l, &[7.0]);
    l.close().unwrap(); // never flushed: folded directly
    drop(l);
    let l = Ledger::open(&t.0, "drop").unwrap();
    let b = l.baseline(&topic(), "valence").unwrap();
    assert_eq!(b.n, 7);
    assert!((b.mean - 4.0).abs() < 1e-6);
    assert_eq!(b.recent_mean, 6.0, "ring of the default primary floor 3: [5, 6, 7]");
}

#[test]
fn corrupt_and_stale_files_are_set_aside() {
    let t = TempDir::new("corrupt");
    fs::create_dir_all(run_dir(&t.0)).unwrap();
    fs::write(t.0.join("drop.json"), b"{\"recipe\": \"zena-sds-v1").unwrap();
    fs::write(run_dir(&t.0).join("1-2-3.json"), [0xff, 0xfe, 0x00]).unwrap();
    let stale = serde_json::json!({ "recipe": "sds-v0", "name": "drop", "run": "4-5-6", "scopes": {} });
    fs::write(run_dir(&t.0).join("4-5-6.json"), stale.to_string()).unwrap();

    let mut l = Ledger::open(&t.0, "drop").unwrap();
    assert_eq!(l.baseline(&topic(), "valence"), None);
    assert_eq!(l.set_aside_files().len(), 3, "{:?}", l.set_aside_files());
    assert_eq!(files_with(&t.0, ".corrupt").len(), 1);
    assert_eq!(files_with(&run_dir(&t.0), ".corrupt").len(), 1);
    assert_eq!(files_with(&run_dir(&t.0), ".stale").len(), 1);
    assert!(files_with(&run_dir(&t.0), ".json").is_empty());

    record(&mut l, &[1.0; 6]);
    l.close().unwrap();
    let l = Ledger::open(&t.0, "drop").unwrap();
    assert_eq!(l.baseline(&topic(), "valence").unwrap().n, 6);
    assert!(l.set_aside_files().is_empty());

    assert!(Ledger::open(&t.0, "../escape").is_err());
    assert!(Ledger::open(&t.0, "").is_err());
}

#[test]
fn a_live_run_is_never_folded_by_another_ledger() {
    let t = TempDir::new("live");
    let mut a = Ledger::open(&t.0, "drop").unwrap();
    record(&mut a, &[1.0; 4]);
    a.flush().unwrap();

    let mut b = Ledger::open(&t.0, "drop").unwrap();
    assert_eq!(files_with(&run_dir(&t.0), ".json").len(), 1, "A's run file is left alone");
    assert!(!b.report().contains("recovered"));
    record(&mut b, &[3.0; 4]);
    b.flush().unwrap();

    a.close().unwrap();
    b.close().unwrap(); // re-reads the base A just wrote
    let c = Ledger::open(&t.0, "drop").unwrap();
    let band = c.baseline(&topic(), "valence").unwrap();
    assert_eq!(band.n, 8, "both runs counted once");
    assert!((band.mean - 2.0).abs() < 1e-6);
    assert!(files_with(&run_dir(&t.0), ".json").is_empty());
}

#[test]
fn a_folded_run_left_behind_is_not_counted_twice() {
    let t = TempDir::new("folded");
    let mut l = Ledger::open(&t.0, "drop").unwrap();
    record(&mut l, &[2.0; 6]);
    l.flush().unwrap();
    let run_file = run_dir(&t.0).join(format!("{}.json", l.run_id()));
    let saved = fs::read(&run_file).unwrap();
    l.close().unwrap();
    // A crash between writing the base and deleting the run file.
    fs::write(&run_file, saved).unwrap();
    drop(l);

    let l = Ledger::open(&t.0, "drop").unwrap();
    assert_eq!(l.baseline(&topic(), "valence").unwrap().n, 6);
    assert!(!run_file.exists(), "already-folded leftover deleted");
}

#[test]
fn open_with_folds_crashed_runs_with_configured_rings() {
    let t = TempDir::new("rings");
    let web = MinObs::new(2, 5, 0);
    let s = Scope::new("web", "web://store", "");
    let mut l = Ledger::open_with(&t.0, "drop", &[("web", web)]).unwrap();
    for v in 0..8 {
        l.record_baseline(&s, "align.cos", v as f32);
    }
    l.flush().unwrap();
    drop(l);
    let l = Ledger::open_with(&t.0, "drop", &[("web", web)]).unwrap();
    assert_eq!(l.min_obs("web"), web);
    let b = l.baseline(&s, "align.cos").unwrap();
    assert_eq!((b.n, b.recent_mean), (8, 5.0), "ring of 5: [3..=7]");
}
