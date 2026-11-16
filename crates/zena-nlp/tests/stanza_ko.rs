//! The Korean Stanza model end to end on this machine (tokens → POS → lemma → dependencies), through the run-time
//! loaded ONNX Runtime. Skips (passes) when the model folder or the library is missing.

#![cfg(feature = "stanza")]

use zena_nlp::stanza::{analyze_query, pipeline};

fn ready() -> bool {
    if zena_nlp::models::stanza_dir("ko").is_none() {
        eprintln!("no Korean Stanza model (RAON_STANZA_DIR / %LOCALAPPDATA%\\RaonZena\\models\\stanza\\ko); skipping");
        return false;
    }
    if let Err(e) = onnxruntime_sys::dynload::api_available() {
        eprintln!("ONNX Runtime unavailable ({e}); skipping");
        return false;
    }
    true
}

#[test]
fn korean_utterances_get_units_lemmas_and_a_tree() {
    if !ready() {
        return;
    }
    let t = std::time::Instant::now();
    pipeline("ko").expect("open the Korean model");
    eprintln!("load {} ms", t.elapsed().as_millis());
    for q in [
        "안경이랑 헤드셋 써 봐",
        "제나야 보스 패턴 어떻게 피해?",
        "보스 패턴 알려주지 마",
        "와 잡았다 고마워",
        "로봇 모드로 변신해 줘",
    ] {
        let a = analyze_query(q, "ko").expect("analyze");
        eprintln!("{q} ({} ms): {:?}", a.elapsed_ms, a.describe());
        assert_eq!(a.lang, "ko");
        assert!(a.whitespace_units, "Korean units are space-separated eojeols");
        assert_eq!(a.units.len(), q.split_whitespace().count(), "{q}");
        let words: Vec<_> = a.units.iter().flat_map(|u| u.words.iter()).collect();
        assert!(!words.is_empty());
        let roots = words.iter().filter(|w| w.head == 0).count();
        assert!(roots >= 1, "a dependency root in {q}");
        assert!(words.iter().all(|w| !w.upos.is_empty() && !w.lemma.is_empty()), "{q}");
    }
    // Content words keep their stems; particles are marked functional or stripped from the stem.
    let a = analyze_query("안경이랑 헤드셋 써 봐", "ko").unwrap();
    assert_eq!(a.units[0].stem, "안경", "{:?}", a.describe());
    assert_eq!(a.units[1].stem, "헤드셋");
    let a = analyze_query("인벤토리를 열어 줘", "ko").unwrap();
    assert_eq!(a.units[0].stem, "인벤토리", "{:?}", a.describe());
}
