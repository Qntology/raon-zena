//! The granite embedder against Hugging Face transformers, plus semantic sanity checks.
//!
//! Reference vectors: `tests/data/granite_reference.json`, written by `python -I scripts/granite_reference.py`
//! (transformers `ModernBertModel`, float32, CPU, CLS pooling, L2-normalised, truncation at 512 tokens).
//! Model folder: `RAON_GRANITE_DIR`, else `%LOCALAPPDATA%\logis-center\models\granite-embedding-97m-multilingual-r2`.
//! Without the model the model-backed tests pass after printing a note.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::Device;
use serde::Deserialize;
use zena_nlp::granite::{Granite, DIM, MAX_TOKENS, SPACE_TAG};

/// Minimum cosine with the Hugging Face vector, and between `embed_batch` and `embed`.
const MIN_COS: f32 = 0.9999;

#[derive(Deserialize)]
struct Reference {
    space_tag: String,
    pooling: String,
    max_tokens: usize,
    dim: usize,
    transformers_version: String,
    items: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    text: String,
    n_tokens: usize,
    n_tokens_untruncated: usize,
    embedding: Vec<f32>,
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("data/granite_reference.json")).expect("parse granite_reference.json")
}

fn model_dir() -> Option<PathBuf> {
    let dir = match std::env::var_os("RAON_GRANITE_DIR") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
            .join("logis-center")
            .join("models")
            .join("granite-embedding-97m-multilingual-r2"),
    };
    ["config.json", "model.safetensors", "tokenizer.json"]
        .iter()
        .all(|f| dir.join(f).is_file())
        .then_some(dir)
}

/// One model shared by the tests of this file (each load is ~400 MB of F32 weights).
fn model() -> Option<&'static Granite> {
    static MODEL: OnceLock<Option<Granite>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let dir = model_dir()?;
            let started = Instant::now();
            let m = Granite::load(&dir, &Device::Cpu).expect("load granite");
            eprintln!("[granite] loaded {} in {:.2?}", dir.display(), started.elapsed());
            Some(m)
        })
        .as_ref()
}

macro_rules! model_or_skip {
    () => {
        match model() {
            Some(m) => m,
            None => {
                eprintln!("[granite] model folder not found (set RAON_GRANITE_DIR); skipping");
                return;
            }
        }
    };
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn short(text: &str) -> String {
    let s: String = text.chars().take(28).collect();
    if s.len() < text.len() {
        format!("{s}…")
    } else {
        s
    }
}

#[test]
fn fixture_describes_this_embedding_space() {
    let r = reference();
    assert_eq!(r.space_tag, SPACE_TAG);
    assert_eq!(r.pooling, "cls");
    assert_eq!(r.dim, DIM);
    assert_eq!(r.max_tokens, MAX_TOKENS);
    assert!(r.items.len() >= 12);
    for item in &r.items {
        assert_eq!(item.embedding.len(), DIM, "{}", item.text);
        let norm = item.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{}: norm {norm}", item.text);
    }
    assert!(
        r.items.iter().any(|i| i.n_tokens_untruncated > 600 && i.n_tokens == MAX_TOKENS),
        "the fixture needs a long input that exercises truncation"
    );
}

#[test]
fn embed_matches_hugging_face() {
    let m = model_or_skip!();
    let r = reference();
    assert_eq!((m.dim(), m.space_tag()), (DIM, SPACE_TAG));
    eprintln!(
        "[granite] reference: transformers {}, {} sentences",
        r.transformers_version,
        r.items.len()
    );
    let (mut min_cos, mut max_diff) = (f32::INFINITY, 0f32);
    let mut failures = Vec::new();
    for item in &r.items {
        let v = m.embed(&item.text).expect("embed");
        assert_eq!(v.len(), DIM);
        let tokens = m.token_count(&item.text).expect("token_count");
        let (c, d) = (cos(&v, &item.embedding), max_abs_diff(&v, &item.embedding));
        eprintln!("[granite] {tokens:4} tok  cos {c:.7}  max|Δ| {d:.2e}  {:?}", short(&item.text));
        min_cos = min_cos.min(c);
        max_diff = max_diff.max(d);
        if tokens != item.n_tokens {
            failures.push(format!("{:?}: {tokens} tokens, Hugging Face {}", short(&item.text), item.n_tokens));
        }
        if c < MIN_COS {
            failures.push(format!("{:?}: cosine {c} < {MIN_COS}", short(&item.text)));
        }
    }
    eprintln!("[granite] embed vs Hugging Face: min cosine {min_cos:.7}, max abs diff {max_diff:.3e}");
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn embed_batch_equals_embed() {
    let m = model_or_skip!();
    let r = reference();
    let texts: Vec<&str> = r.items.iter().map(|i| i.text.as_str()).collect();
    let single: Vec<Vec<f32>> = texts.iter().map(|t| m.embed(t).expect("embed")).collect();
    let batch = m.embed_batch(&texts).expect("embed_batch");
    assert_eq!(batch.len(), texts.len());

    let (mut min_self, mut max_diff, mut min_ref) = (f32::INFINITY, 0f32, f32::INFINITY);
    for ((s, b), item) in single.iter().zip(&batch).zip(&r.items) {
        assert_eq!(b.len(), DIM);
        min_self = min_self.min(cos(s, b));
        max_diff = max_diff.max(max_abs_diff(s, b));
        min_ref = min_ref.min(cos(b, &item.embedding));
    }
    eprintln!(
        "[granite] embed_batch vs embed: min cosine {min_self:.7}, max abs diff {max_diff:.3e}; vs Hugging Face: min cosine {min_ref:.7}"
    );
    assert!(min_self >= MIN_COS, "embed_batch differs from embed: {min_self}");
    assert!(min_ref >= MIN_COS, "embed_batch differs from Hugging Face: {min_ref}");

    // Order and batch composition do not matter: reversed, with duplicates, mixed with short spans only.
    let mixed: Vec<&str> = texts.iter().rev().chain(texts.iter().take(4)).copied().collect();
    let again = m.embed_batch(&mixed).expect("embed_batch");
    for (t, v) in mixed.iter().zip(&again) {
        let i = texts.iter().position(|x| x == t).unwrap();
        assert!(cos(v, &single[i]) >= MIN_COS, "{:?}", short(t));
    }
    assert!(m.embed_batch(&[]).expect("empty batch").is_empty());
}

#[test]
fn semantic_sanity() {
    let m = model_or_skip!();
    let texts = [
        "또 죽었어",
        "보스한테 졌어",
        "인벤토리 열어 줘",
        "보스 패턴 알려줘",
        "tell me the boss pattern",
        "안경 써 봐",
        "wgpu SurfaceError Outdated",
        "와 잡았다 고마워",
    ];
    let v = m.embed_batch(&texts).expect("embed_batch");
    let c = |a: usize, b: usize| cos(&v[a], &v[b]);
    for (i, a) in texts.iter().enumerate() {
        let row: Vec<String> = (0..texts.len()).map(|j| format!("{:.3}", c(i, j))).collect();
        eprintln!("[granite] {:>28} {}", a, row.join(" "));
    }

    let (died_lost, died_inventory) = (c(0, 1), c(0, 2));
    eprintln!("[granite] cos(또 죽었어, 보스한테 졌어) {died_lost:.4} > cos(또 죽었어, 인벤토리 열어 줘) {died_inventory:.4}");
    assert!(died_lost > died_inventory);

    // Korean ↔ English paraphrase: high in absolute terms, and above both sentences' similarity to anything unrelated.
    let paraphrase = c(3, 4);
    let unrelated = [2, 5, 6, 7].iter().flat_map(|&j| [c(3, j), c(4, j)]).fold(f32::MIN, f32::max);
    eprintln!("[granite] cos(보스 패턴 알려줘, tell me the boss pattern) {paraphrase:.4}; best unrelated {unrelated:.4}");
    assert!(paraphrase > 0.85, "paraphrase cosine {paraphrase}");
    assert!(paraphrase > unrelated + 0.02, "paraphrase {paraphrase} vs unrelated {unrelated}");
}
