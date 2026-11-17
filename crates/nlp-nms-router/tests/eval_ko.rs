//! NLP-NMS evaluation on `data/eval/nlp_nms_ko.jsonl` (Phase 0 of `docs/plans/relative-value-sds.md`).
//!
//! Every utterance has the set of dictionary domains it should name. Exact phrases and paraphrases of every
//! routable domain, multi-intent sentences, negations (must name nothing) and everyday sentences not meant for Zena
//! (must name nothing). Reported per configuration: exact-set accuracy, accuracy by kind, **false commands**
//! (a system or avatar-asset command nobody asked for) and the mean time per utterance.
//!
//! `cargo test -p nlp-nms-router --test eval_ko -- --nocapture` prints the table; with `--features stanza` and the
//! models installed it adds the v2 rows (granite + Stanza).

use nlp_nms_router::router::{Models, NlpNmsRouter};
use serde_json::Value;
use std::collections::BTreeSet;
use std::time::Instant;
use zena_lexicon::{Action, Lexicon};

struct Case {
    id: String,
    kind: String,
    text: String,
    expect: BTreeSet<String>,
}

fn cases() -> Vec<Case> {
    let raw = include_str!("../../../data/eval/nlp_nms_ko.jsonl");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).expect("eval line");
            Case {
                id: v["id"].as_str().unwrap().into(),
                kind: v["kind"].as_str().unwrap().into(),
                text: v["text"].as_str().unwrap().into(),
                expect: v["expect"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_str().unwrap().to_string())
                    .collect(),
            }
        })
        .collect()
}

#[derive(Default)]
pub struct Report {
    pub label: String,
    pub n: usize,
    pub correct: usize,
    pub false_commands: usize,
    pub by_kind: std::collections::BTreeMap<String, (usize, usize)>,
    pub misses: Vec<String>,
    pub ms: f64,
}

impl Report {
    fn accuracy(&self) -> f64 {
        self.correct as f64 / self.n.max(1) as f64
    }
    fn line(&self) -> String {
        let kinds: Vec<String> = self.by_kind.iter().map(|(k, (c, n))| format!("{k} {c}/{n}")).collect();
        format!(
            "{:<28} acc {:>3}/{:<3} ({:>5.1}%)  false cmd {:>2}  {:>6.1} ms/utt  | {}",
            self.label,
            self.correct,
            self.n,
            self.accuracy() * 100.0,
            self.false_commands,
            self.ms,
            kinds.join("  ")
        )
    }
}

fn evaluate(label: &str, router: &NlpNmsRouter) -> Report {
    let mut r = Report {
        label: label.into(),
        ..Default::default()
    };
    // Warm-up (first Stanza call, embedding caches of the dictionary already built).
    let _ = router.catch("안녕");
    let t = Instant::now();
    for c in cases() {
        let got = router.catch(&c.text);
        let keys: BTreeSet<String> = got.hits.iter().map(|h| router.bank.entry(h.0).key.clone()).collect();
        let ok = keys == c.expect;
        r.n += 1;
        r.correct += ok as usize;
        let k = r.by_kind.entry(c.kind.clone()).or_default();
        k.0 += ok as usize;
        k.1 += 1;
        let false_cmd = got.hits.iter().any(|h| {
            let e = router.bank.entry(h.0);
            e.payload.is_command() && !c.expect.contains(&e.key)
        });
        r.false_commands += false_cmd as usize;
        if !ok {
            r.misses
                .push(format!("  {:<12} {:<34} want {:?} got {:?}", c.id, c.text, c.expect, keys));
        }
    }
    r.ms = t.elapsed().as_secs_f64() * 1e3 / r.n.max(1) as f64;
    r
}

fn print(r: &Report, misses: bool) {
    eprintln!("{}", r.line());
    if misses {
        for m in &r.misses {
            eprintln!("{m}");
        }
    }
}

#[test]
fn the_eval_set_is_well_formed() {
    let lex = Lexicon::bundled();
    let cs = cases();
    assert!(cs.len() >= 90, "{}", cs.len());
    for c in &cs {
        for k in &c.expect {
            let n = lex.node(k).unwrap_or_else(|| panic!("{}: unknown domain {k}", c.id));
            assert!(!matches!(n.action, Action::None), "{}: {k} has no action", c.id);
        }
        if matches!(c.kind.as_str(), "negation" | "negative") {
            assert!(c.expect.is_empty(), "{}", c.id);
        }
    }
    // Every routable domain is covered.
    let covered: BTreeSet<&String> = cs.iter().flat_map(|c| c.expect.iter()).collect();
    for n in lex.text_nodes().filter(|n| n.key != "chitchat.small_talk") {
        assert!(covered.contains(&n.key), "no utterance for {}", n.key);
    }
}

#[test]
fn rules_never_act_on_a_negated_command_and_do_not_lose_accuracy() {
    let lex = Lexicon::bundled();
    let v1 = evaluate("v1 (hash)", &NlpNmsRouter::with_models(&lex, Models::v1()));
    let rules = evaluate("v1 + rules", &NlpNmsRouter::with_models(&lex, Models::rules()));
    print(&v1, false);
    print(&rules, false);
    let neg = |r: &Report| r.by_kind.get("negation").copied().unwrap_or_default();
    assert_eq!(neg(&rules).0, neg(&rules).1, "every negation names nothing with the rule analyzer");
    assert!(rules.false_commands <= v1.false_commands);
    assert!(rules.correct >= v1.correct, "{} < {}", rules.correct, v1.correct);
}

/// Diagnostics: `EVAL_DEBUG="문장1|문장2" cargo test -p nlp-nms-router --features stanza --test eval_ko -- --ignored
/// --nocapture debug_utterances` prints each word's role, negation and the v2 catch.
#[cfg(feature = "neural")]
#[test]
#[ignore = "diagnostics (EVAL_DEBUG)"]
fn debug_utterances() {
    let Ok(list) = std::env::var("EVAL_DEBUG") else { return };
    let r = NlpNmsRouter::with_models(&Lexicon::bundled(), Models::auto("korean"));
    for t in list.split('|') {
        let c = r.catch(t);
        let n = c.normalized.as_ref().unwrap();
        let words: Vec<String> = n
            .words
            .iter()
            .map(|w| {
                format!(
                    "{}[{}/{:?}{}{}]",
                    w.surface,
                    w.upos,
                    w.role,
                    if w.negated { "/neg" } else { "" },
                    w.alt.as_deref().map(|a| format!("/{a}")).unwrap_or_default()
                )
            })
            .collect();
        let hits: Vec<String> = c
            .hits
            .iter()
            .map(|h| format!("{} '{}' {:.3} {:?}", r.bank.entry(h.0).key, h.1, h.2, h.3))
            .collect();
        eprintln!("{t}\n  words {words:?}\n  hits {hits:?}\n  negated {:?}", c.negated);
        for (s, e) in [(0usize, n.words.len())] {
            let top: Vec<String> = r
                .bank
                .score_words(&n.words[s..e])
                .iter()
                .take(4)
                .map(|d| format!("{} {:.3}", r.bank.entry(d.index).key, d.score))
                .collect();
            eprintln!("  whole-utterance top {top:?}");
        }
    }
}

/// Threshold sweep for the v2 selection: `cargo test -p nlp-nms-router --features stanza --release --test eval_ko --
/// --ignored --nocapture calibrate_v2`. Objective: correct − 2 × false commands (an unwanted command costs more than
/// a miss). The chosen values go to `router::GRANITE_ASSIGN`.
#[cfg(feature = "neural")]
#[test]
#[ignore = "calibration sweep"]
fn calibrate_v2() {
    use nlp_nms_router::router::Selection;
    use nlp_nms_router::span::AssignConfig;
    if nlp_nms_router::__granite_dir().is_none() {
        return;
    }
    let lex = Lexicon::bundled();
    let mut r = NlpNmsRouter::with_models(&lex, Models::auto("korean"));
    let mut rows = Vec::new();
    for accept in [0.62f32, 0.66, 0.70] {
        for margin in [0.02f32, 0.04, 0.06] {
            for (lexical_weight, lexical_floor) in [(0.0f32, 0.5f32), (0.15, 0.4), (0.3, 0.4), (0.3, 0.5), (0.5, 0.5)] {
                let cfg = AssignConfig {
                    max_len: 5,
                    accept,
                    margin,
                    residual: 0.0,
                    lexical_weight,
                    lexical_floor,
                };
                r.set_selection(&lex, Selection::Assign(cfg));
                let rep = evaluate(&format!("a{accept:.2} m{margin:.2} lw{lexical_weight:.2} lf{lexical_floor:.1}"), &r);
                let objective = rep.correct as i64 - 2 * rep.false_commands as i64;
                rows.push((objective, rep));
            }
        }
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.correct.cmp(&a.1.correct)));
    for (o, rep) in rows.iter().take(12) {
        eprintln!("obj {o:>3}  {}", rep.line());
    }
}

/// v2 with the installed models (granite + Stanza). Skips when granite is not installed.
#[cfg(feature = "neural")]
#[test]
fn v2_beats_v1_on_the_eval_set() {
    if nlp_nms_router::__granite_dir().is_none() {
        eprintln!("granite not installed; skipping v2");
        return;
    }
    let lex = Lexicon::bundled();
    let t = Instant::now();
    let models = Models::auto("korean");
    let v2_router = NlpNmsRouter::with_models(&lex, models.clone());
    eprintln!("v2 models ready in {:.1}s ({})", t.elapsed().as_secs_f32(), models.label);
    let v1 = evaluate("v1 (hash)", &NlpNmsRouter::with_models(&lex, Models::v1()));
    let rules = evaluate("v1 + rules", &NlpNmsRouter::with_models(&lex, Models::rules()));
    let v2 = evaluate(&format!("v2 ({})", models.label), &v2_router);
    print(&v1, false);
    print(&rules, false);
    print(&v2, true);
    assert!(v2.correct > v1.correct, "v2 {} vs v1 {}", v2.correct, v1.correct);
    assert!(v2.false_commands <= rules.false_commands, "v2 false commands {}", v2.false_commands);
}
