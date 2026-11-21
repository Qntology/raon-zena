//! Glue: utterances + game alerts in, de-duplicated actions out.
//!
//! The user's words are caught into the expression domains of the word dictionary (`zena-lexicon`); each kept
//! domain becomes what its node says: a system command, a dictionary expression Zena plays at once (no LLM), or a
//! request for Raon-1-nano.
//!
//! A negated span is held back ("앱 끄지 마" runs nothing), except for feedback: a `feedback.*` span is kept with its
//! negation count in a channel of its own ([`Catch::feedback`], [`RouterOutput::feedback`]), because a negated
//! recommendation is the opposite recommendation ("추천 안 해" = −1, "나쁘지 않았어" = +1 weak) and must reach
//! `zena_verdict::feedback` instead of vanishing.

use crate::analyze::{Analyzer, NoAnalyzer, RuleAnalyzer};
use crate::domain::{DomainBank, Hit, LexiconBank};
use crate::embed::{HashedNgramEmbedder, SharedEmbedder, TextEmbedder};
use crate::normalize::{normalize, Normalized, Role, Word};
use crate::semantic::{Decision, Item, SemanticNms, Source};
use crate::span::{span_assign, span_nms, AssignConfig, SpanHit};
use crate::spatial::{select_target, BoxCand};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use zena_core::geometry::Polar;
use zena_core::tools::{AssetAction, SystemCommand, WindowLayerMode, ZenaAssetArgs};
use zena_lexicon::{Action, Lexicon};

/// The dictionary node an utterance is answered with when Zena is addressed but no domain is caught.
pub const SMALL_TALK: &str = "chitchat.small_talk";

/// Dictionary keys of feedback about a recommendation (`feedback.recommend`, `feedback.not_recommend`): negated,
/// they are the opposite feedback, not a command to hold back.
pub const FEEDBACK_PREFIX: &str = "feedback.";

/// A `feedback.*` span of one utterance, kept whether or not it is negated.
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackSpan {
    /// Dictionary key (`feedback.recommend` …).
    pub key: String,
    /// The words that named the domain, and their range in the utterance's words.
    pub span: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
    pub hit: Hit,
    /// Negators acting on the span's predicate, saturating at 2: the analyzer's negations plus the long form
    /// "-지 않- / -지 못하-", the copula "아니-" and "추천 못해", which the rule analyzer does not tag. An odd count
    /// flips the feedback ([`FeedbackSpan::negated`]); two cancel into a hedged feedback.
    pub negation_count: u8,
    /// A request auxiliary or request ending follows ("추천해 **줘**"): probably a request, not feedback.
    pub request: bool,
}

impl FeedbackSpan {
    /// An odd number of negators: the feedback flips its sign.
    pub fn negated(&self) -> bool {
        self.negation_count % 2 == 1
    }
}

/// One word of a feedback utterance with the flags `zena_verdict::feedback::FeedbackToken` takes (field for field;
/// this crate does not depend on zena-verdict).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeedbackWord {
    pub surface: String,
    pub lemma: String,
    pub is_auxiliary_request: bool,
    pub is_negation: bool,
    pub negated_in_clause: bool,
    pub is_subject_or_topic: bool,
}

/// The feedback of one utterance ([`RouterOutput::feedback`]): hand `utterance`, `words` and the context to
/// `zena_verdict::feedback::extract_pragmatic_feedback`, which decides the vote (requests, questions, adversative
/// clauses, retractions, sarcasm).
#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackCatch {
    /// The utterance as said (punctuation kept: "?" makes a question).
    pub utterance: String,
    pub t_ms: u64,
    pub words: Vec<FeedbackWord>,
    pub spans: Vec<FeedbackSpan>,
}

/// Negating adverbs (the rule analyzer tags them too; listed for the v1 router, which runs no analyzer).
const NEG_ADVERBS: &[&str] = &[
    "안", "못", "not", "never", "dont", "didnt", "isnt", "wasnt", "doesnt", "cant", "wont",
];
/// The long-form negation after a "-지" word ("나쁘지 않았어", "추천하지 못해").
const LONG_NEGATION: &[&str] = &["않", "못하", "못해", "못했", "못한", "못함"];
const FUSED_LONG_NEGATION: &[&str] = &["지않", "지못하", "지못해", "지못했", "지못한"];
/// The copula negation ("추천은 아니야", "나쁜 건 아니야").
const COPULA_NEGATION: &[&str] = &["아니", "아냐", "아닌", "아님"];
/// A negating adverb fused to the light verb after a noun ("추천 못해", "추천 안함").
const FUSED_NEGATED_LIGHT_VERBS: &[&str] = &[
    "못하", "못해", "못했", "못한", "못함", "안하", "안해", "안했", "안한", "안함", "안돼", "안됐", "안되", "안된",
];
const LIGHT_VERBS: &[&str] = &["하", "해", "했", "한", "할", "합", "되", "돼", "됐", "된", "될"];
const NOMINALIZERS: &[&str] = &["건", "거", "게", "것", "것은", "것도", "거는"];
/// Request auxiliaries after the evaluated word, and request endings fused to it.
const REQUEST_WORDS: &[&str] = &[
    "줘",
    "줘요",
    "주세요",
    "줄래",
    "줄래요",
    "주라",
    "달라",
    "봐",
    "봐줘",
    "볼래",
    "볼래요",
    "please",
];
const REQUEST_ENDINGS: &[&str] = &[
    "해줘",
    "해줘요",
    "해주세요",
    "해줄래",
    "해줄래요",
    "해달라",
    "해봐",
    "해볼래",
    "해주라",
];

/// English contractions split by normalisation ("don't" → "don" "t").
const CONTRACTED: &[&str] = &[
    "don", "didn", "doesn", "isn", "wasn", "aren", "weren", "can", "won", "couldn", "wouldn", "shouldn",
];

fn starts_any(s: &str, list: &[&str]) -> bool {
    list.iter().any(|p| s.starts_with(p))
}

/// A negating adverb before the predicate: "안", "못", "not", "never", "don't" (normalised to "don t").
fn negating_adverb(words: &[Word], i: usize) -> bool {
    let s = words[i].surface.as_str();
    NEG_ADVERBS.contains(&s) || (s == "t" && i > 0 && CONTRACTED.contains(&words[i - 1].surface.as_str()))
}

/// Whether `words[i]` negates: the analyzer's negation role, a negating adverb, the long form after a "-지" word,
/// the copula negation, or a negation fused into the word.
fn is_negator(words: &[Word], i: usize) -> bool {
    let s = words[i].surface.as_str();
    words[i].role == Role::Negation
        || negating_adverb(words, i)
        || (i > 0 && words[i - 1].surface.ends_with('지') && starts_any(s, LONG_NEGATION))
        || (i > 0 && (starts_any(s, COPULA_NEGATION) || starts_any(s, FUSED_NEGATED_LIGHT_VERBS)))
        || (s.chars().count() > 2 && FUSED_LONG_NEGATION.iter().any(|t| s.contains(t)))
}

/// A connective that ends a clause ("했는데", "와서", "가면").
fn ends_clause(s: &str) -> bool {
    s.chars().count() >= 2 && (s.ends_with('고') || s.ends_with('서') || s.ends_with("는데") || s.ends_with("지만") || s.ends_with('면'))
}

/// Negators acting on the predicate of `words[start..end]`, saturating at 2: inside the span, a negating adverb just
/// before it, and after it through light verbs and nominalisers ("추천 안 해", "나쁘지 않았어", "추천은 하지 않아",
/// "나쁜 건 아니야"); the analyzer's own scope counts when nothing else did.
fn feedback_negations(words: &[Word], start: usize, end: usize) -> u8 {
    let mut n = (start..end).filter(|&i| is_negator(words, i)).count();
    if start > 0 && negating_adverb(words, start - 1) {
        n += 1;
    }
    let mut j = end;
    while j < words.len() {
        let s = words[j].surface.as_str();
        if negating_adverb(words, j) {
            // "추천 안 해" negates the light verb of 추천하다; "별로 안 좋아" negates 좋다, not the span.
            if words.get(j + 1).is_some_and(|w| starts_any(&w.surface, LIGHT_VERBS)) {
                n += 1;
                j += 2;
                continue;
            }
            break;
        }
        if is_negator(words, j) {
            n += 1;
        } else if !(starts_any(s, LIGHT_VERBS) || NOMINALIZERS.contains(&s)) {
            break;
        }
        j += 1;
        if ends_clause(s) {
            break;
        }
    }
    if n == 0 && words[start..end].iter().any(|w| w.negated) {
        n = 1;
    }
    n.min(2) as u8
}

/// A request ending on the span's last word ("추천해줘") or a request word later in the utterance ("추천해 줘").
fn is_request(words: &[Word], end: usize) -> bool {
    let last = words[end - 1].surface.as_str();
    let fused = last.chars().count() > 2 && REQUEST_ENDINGS.iter().any(|e| last.ends_with(e));
    fused
        || words[end..]
            .iter()
            .any(|w| w.role == Role::Auxiliary || REQUEST_WORDS.contains(&w.surface.as_str()) || w.surface.starts_with("알려"))
}

/// The words of an utterance as [`FeedbackWord`]s.
fn feedback_words(words: &[Word]) -> Vec<FeedbackWord> {
    (0..words.len())
        .map(|i| {
            let w = &words[i];
            let negation = is_negator(words, i);
            // The "-지" word before a long-form negation, or the word before "아니-", is negated too.
            let negated_by_next = i + 1 < words.len() && is_negator(words, i + 1) && !negating_adverb(words, i + 1);
            let topic = w.lemma != w.surface && ["은", "는", "이", "가"].iter().any(|j| w.surface.ends_with(j));
            FeedbackWord {
                surface: w.surface.clone(),
                lemma: w.lemma.clone(),
                is_auxiliary_request: w.role == Role::Auxiliary || (i > 0 && REQUEST_WORDS.contains(&w.surface.as_str())),
                is_negation: negation,
                negated_in_clause: w.negated || negation || negated_by_next,
                is_subject_or_topic: matches!(w.deprel.as_str(), "nsubj" | "nsubj:pass" | "csubj" | "dislocated") || topic,
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub enum RouterInput {
    Utterance {
        text: String,
        t_ms: u64,
    },
    GameEvent {
        label: String,
        score: f32,
        polar: Polar,
        t_ms: u64,
    },
}

/// A domain caught in an utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Caught {
    /// Dictionary key (`group.name`).
    pub key: String,
    pub action: Action,
    /// The words that named the domain.
    pub span: String,
    /// The whole utterance.
    pub utterance: String,
    pub score: f32,
    pub hit: Hit,
}

#[derive(Debug, Clone, Default)]
pub struct RouterOutput {
    pub system: Vec<(SystemCommand, Option<WindowLayerMode>)>,
    /// Avatar asset commands (track switches, accessories), in the order they were said.
    pub assets: Vec<ZenaAssetArgs>,
    /// Dictionary expressions Zena plays without the LLM, best first (a command's own expression included).
    pub expressions: Vec<Caught>,
    /// Requests for Raon-1-nano (Lv3), best first.
    pub queries: Vec<Caught>,
    /// Alerts that survived NMS: (label, polar, score).
    pub events: Vec<(String, Polar, f32)>,
    pub decisions: Vec<Decision>,
    /// Utterances with `feedback.*` spans, negated ones included (no temporal NMS: each is a vote to judge).
    pub feedback: Vec<FeedbackCatch>,
}

/// How spans become domains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Selection {
    /// v1: best domain per span above `accept`, greedy overlap suppression ([`span_nms`]).
    Greedy { accept: f32 },
    /// v2: double-centred spans × domains with margin-gated 1:1 assignment ([`span_assign`]).
    Assign(AssignConfig),
}

/// The models and thresholds the router runs with.
#[derive(Clone)]
pub struct Models {
    pub embedder: SharedEmbedder,
    pub analyzer: Arc<dyn Analyzer>,
    pub selection: Selection,
    /// Floor of the temporal semantic NMS duplicate threshold (`clamp(μ + 3σ, base, 0.995)`) in this vector space.
    pub base_tau: f32,
    /// Score Dynamics ledger (`nms` track): decay shapes of chosen spans, confusion between near-tied domains.
    pub sds: Option<Arc<Mutex<zena_sds::Ledger>>>,
    /// A short description for logs ("granite + stanza", ...).
    pub label: String,
}

impl Models {
    /// The v1 router: hashed n-gram embedder, no analyzer, greedy selection.
    pub fn v1() -> Self {
        Self {
            embedder: SharedEmbedder::new(HashedNgramEmbedder::default()),
            analyzer: Arc::new(NoAnalyzer),
            selection: Selection::Greedy { accept: 0.6 },
            base_tau: 0.6,
            sds: None,
            label: "hash".into(),
        }
    }

    /// v1 scoring with the rule analyzer (auxiliaries, negation): what runs when no model is installed.
    pub fn rules() -> Self {
        Self {
            analyzer: Arc::new(RuleAnalyzer),
            label: "hash + rules".into(),
            ..Self::v1()
        }
    }

    /// NLP-NMS v2 with whatever models are installed: granite embeddings with double-centred assignment when the
    /// granite model loads, Stanza roles when its Korean model and ONNX Runtime load; otherwise [`Models::rules`].
    #[cfg(feature = "neural")]
    pub fn auto(ui_language: &str) -> Self {
        let analyzer: Arc<dyn Analyzer> = Self::stanza_analyzer(ui_language);
        let mut m = match crate::embed::GraniteEmbedder::load_default() {
            Ok(g) => Self {
                embedder: SharedEmbedder::new(g),
                analyzer,
                selection: Selection::Assign(GRANITE_ASSIGN),
                base_tau: GRANITE_BASE_TAU,
                sds: None,
                label: String::new(),
            },
            Err(e) => {
                eprintln!("[nlp] granite unavailable ({e:#}); hashed embedder");
                Self { analyzer, ..Self::rules() }
            }
        };
        m.label = format!("{} + {}", m.embedder.space(), m.analyzer.name());
        m
    }

    #[cfg(feature = "neural")]
    fn stanza_analyzer(ui_language: &str) -> Arc<dyn Analyzer> {
        #[cfg(feature = "stanza")]
        {
            let lang = premium_discount_nlp::stanza::lang_code(ui_language).unwrap_or("ko");
            match crate::analyze::StanzaAnalyzer::warm(lang) {
                Ok(()) => {
                    return Arc::new(crate::analyze::StanzaAnalyzer {
                        ui_language: ui_language.to_string(),
                    })
                }
                Err(e) => eprintln!("[nlp] Stanza unavailable ({e:#}); rule analyzer"),
            }
        }
        let _ = ui_language;
        Arc::new(RuleAnalyzer)
    }

    pub fn with_sds(mut self, ledger: zena_sds::Ledger) -> Self {
        self.sds = Some(Arc::new(Mutex::new(ledger)));
        self
    }
}

/// v2 thresholds in the granite space, calibrated on `data/eval/nlp_nms_ko.jsonl` (`tests/eval_ko.rs`
/// `calibrate_v2`, objective correct − 2 × false commands). Granite's CLS cosines sit in a narrow band (≈ 0.70–0.87
/// for short Korean phrases), so the margin on double-centred residuals does most of the work; the lexical bonus
/// lets shared words break semantic near-ties. Accept 0.62–0.70 score the same here (a plateau, not a spike).
pub const GRANITE_ASSIGN: AssignConfig = AssignConfig {
    max_len: 5,
    accept: 0.66,
    margin: 0.06,
    residual: 0.0,
    lexical_weight: 0.5,
    lexical_floor: 0.5,
};
/// Semantic NMS duplicate floor in the granite space (unrelated short phrases sit higher than in the hashed space).
pub const GRANITE_BASE_TAU: f32 = 0.85;

/// What one utterance yields before the temporal NMS.
#[derive(Debug, Clone, Default)]
pub struct Catch {
    pub normalized: Option<Normalized>,
    /// Kept spans: (bank index, span text, score, hit). A feedback span is here only when it is neither negated nor
    /// followed by a request (it then acts as its node says, e.g. Zena's thanks).
    pub hits: Vec<(usize, String, f32, Hit)>,
    /// Spans dropped because they were negated ("앱 끄지 마"); never a feedback span.
    pub negated: Vec<(usize, String)>,
    /// Every `feedback.*` span, negated or not.
    pub feedback: Vec<FeedbackSpan>,
}

pub struct NlpNmsRouter {
    pub bank: DomainBank<SharedEmbedder>,
    /// The hashed (lexical) bank beside a neural one, for hybrid scoring (`AssignConfig::lexical_weight`).
    pub lexical: Option<DomainBank<HashedNgramEmbedder>>,
    pub nms: SemanticNms,
    pub models: Models,
    next_id: u64,
}

impl Default for NlpNmsRouter {
    fn default() -> Self {
        Self::new(&Lexicon::bundled())
    }
}

impl NlpNmsRouter {
    /// The v1 router (see [`Models::v1`]).
    pub fn new(lex: &Lexicon) -> Self {
        Self::with_models(lex, Models::v1())
    }

    pub fn with_models(lex: &Lexicon, models: Models) -> Self {
        let mut nms = SemanticNms::default();
        nms.base_tau = models.base_tau;
        Self {
            bank: Self::bank(lex, &models),
            lexical: Self::lexical_bank(lex, &models),
            nms,
            models,
            next_id: 1,
        }
    }

    /// Dictionary phrases are embedded like the spans they are compared with: by content words when an analyzer
    /// runs, as written otherwise (v1).
    fn bank(lex: &Lexicon, models: &Models) -> DomainBank<SharedEmbedder> {
        if models.analyzer.name() == "none" {
            DomainBank::from_lexicon(lex, models.embedder.clone())
        } else {
            DomainBank::from_lexicon_content(lex, models.embedder.clone())
        }
    }

    fn lexical_bank(lex: &Lexicon, models: &Models) -> Option<DomainBank<HashedNgramEmbedder>> {
        matches!(models.selection, Selection::Assign(c) if c.lexical_weight > 0.0)
            .then(|| DomainBank::from_lexicon_content(lex, HashedNgramEmbedder::default()))
    }

    /// Swaps in an updated dictionary; the NMS memory of recent requests stays.
    pub fn reload(&mut self, lex: &Lexicon) {
        self.bank = Self::bank(lex, &self.models);
        self.lexical = Self::lexical_bank(lex, &self.models);
    }

    /// Change the selection (thresholds, hybrid weight) and rebuild the lexical bank if it is now needed.
    pub fn set_selection(&mut self, lex: &Lexicon, selection: Selection) {
        self.models.selection = selection;
        self.lexical = Self::lexical_bank(lex, &self.models);
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn sds_scope(&self) -> zena_sds::Scope {
        zena_sds::Scope::new("nms", self.models.embedder.space(), "")
    }

    /// One utterance → its domains (no temporal NMS, no state change except SDS observations).
    pub fn catch(&self, text: &str) -> Catch {
        let mut n = normalize(text);
        if n.words.is_empty() {
            return Catch::default();
        }
        self.models.analyzer.analyze(&mut n);
        let spans: Vec<SpanHit> = match self.models.selection {
            Selection::Greedy { accept } => span_nms(&n.words, &self.bank, 5, accept),
            Selection::Assign(cfg) => {
                let scope = self.sds_scope();
                let mut cfg = cfg;
                let ledger = self.models.sds.as_ref().map(|l| l.lock().unwrap());
                // SDS: once this space has enough observations, the margin follows its usual 1st–2nd gap.
                if let Some((_, _, m_mean, m_sd)) = ledger.as_ref().and_then(|l| l.decay_baseline(&scope, "nms.span")) {
                    cfg.margin = (m_mean - m_sd).clamp(cfg.margin * 0.5, cfg.margin * 2.0);
                }
                let keys: Vec<String> = (0..self.bank.len()).map(|i| self.bank.entry(i).key.clone()).collect();
                let allow = |f: usize, r: usize, kind: zena_sds::TieKind| {
                    kind != zena_sds::TieKind::Flat && ledger.as_ref().is_some_and(|l| l.tie_break_allows(&scope, &keys[f], &keys[r]))
                };
                let a = span_assign(&n.words, &self.bank, self.lexical.as_ref(), &cfg, allow);
                drop(ledger);
                if let Some(l) = &self.models.sds {
                    let mut l = l.lock().unwrap();
                    for c in &a.columns {
                        l.record_decay(&scope, "nms.span", c);
                    }
                    for &(f, r) in &a.held {
                        l.record_tie(&scope, &keys[f], &keys[r], 0.0);
                    }
                }
                a.hits
            }
        };
        let mut out = Catch::default();
        for h in spans {
            let negated = n.words[h.start..h.end].iter().any(|w| w.negated);
            let key = &self.bank.entry(h.index).key;
            if key.starts_with(FEEDBACK_PREFIX) {
                let negation_count = feedback_negations(&n.words, h.start, h.end);
                let request = is_request(&n.words, h.end);
                if negation_count == 0 && !request {
                    out.hits.push((h.index, h.text.clone(), h.score, h.hit));
                }
                out.feedback.push(FeedbackSpan {
                    key: key.clone(),
                    span: h.text,
                    start: h.start,
                    end: h.end,
                    score: h.score,
                    hit: h.hit,
                    negation_count,
                    request,
                });
            } else if negated {
                out.negated.push((h.index, h.text));
            } else {
                out.hits.push((h.index, h.text, h.score, h.hit));
            }
        }
        if out.hits.is_empty() && out.negated.is_empty() && out.feedback.is_empty() && n.addressed {
            if let Some(i) = self.bank.index_of(SMALL_TALK) {
                out.hits.push((i, n.text.clone(), 0.5, Hit::Vector));
            }
        }
        out.normalized = Some(n);
        out
    }

    pub fn route(&mut self, inputs: Vec<RouterInput>, now_ms: u64) -> RouterOutput {
        let mut items = Vec::new();
        let mut caught: HashMap<u64, Caught> = HashMap::new();
        let mut feedback = Vec::new();
        for inp in inputs {
            match inp {
                RouterInput::Utterance { text, t_ms } => {
                    let c = self.catch(&text);
                    let Some(n) = c.normalized else { continue };
                    for (i, span) in &c.negated {
                        eprintln!("[nlp] negated, not acted on: {} ('{span}')", self.bank.entry(*i).key);
                    }
                    if !c.feedback.is_empty() {
                        feedback.push(FeedbackCatch {
                            utterance: text.clone(),
                            t_ms,
                            words: feedback_words(&n.words),
                            spans: c.feedback,
                        });
                    }
                    let hits = c.hits;
                    let full_vec = self.bank.embedder.embed(&n.text);
                    for (index, span, score, hit) in hits {
                        let id = self.id();
                        let e = self.bank.entry(index);
                        // Commands and expressions dedupe on their own words; questions on the whole utterance.
                        let ask = matches!(e.payload, Action::Ask { .. });
                        let (t, v) = if ask {
                            (text.clone(), full_vec.clone())
                        } else {
                            (span.clone(), self.bank.embedder.embed(&span))
                        };
                        caught.insert(
                            id,
                            Caught {
                                key: e.key.clone(),
                                action: e.payload.clone(),
                                span,
                                utterance: text.clone(),
                                score,
                                hit,
                            },
                        );
                        items.push(Item {
                            id,
                            source: Source::User,
                            key: e.key.clone(),
                            is_command: e.payload.is_command(),
                            text: t,
                            vec: v,
                            score,
                            t_ms,
                            polar: None,
                        });
                    }
                }
                RouterInput::GameEvent { label, score, polar, t_ms } => {
                    let id = self.id();
                    let vec = self.bank.embedder.embed(&label);
                    items.push(Item {
                        id,
                        source: Source::Game,
                        key: label.clone(),
                        is_command: false,
                        text: label,
                        vec,
                        score,
                        t_ms,
                        polar: Some(polar),
                    });
                }
            }
        }
        let decisions = self.nms.process(items, now_ms);
        let mut out = RouterOutput::default();
        // Utterances that gave a command: they are not also a question unless one was caught exactly.
        let mut commanding: Vec<String> = Vec::new();
        for d in decisions.iter().filter(|d| d.kept) {
            match d.item.source {
                Source::Game => out.events.push((d.item.key.clone(), d.item.polar.unwrap(), d.item.score)),
                Source::User => {
                    let Some(c) = caught.remove(&d.item.id) else { continue };
                    match &c.action {
                        Action::System { command, mode, express } => {
                            out.system.push((*command, *mode));
                            commanding.push(c.utterance.clone());
                            if express.is_some() {
                                out.expressions.push(c);
                            }
                        }
                        Action::Asset { asset, express } => {
                            if !out.assets.contains(asset) {
                                out.assets.push(asset.clone());
                            }
                            commanding.push(c.utterance.clone());
                            if express.is_some() {
                                out.expressions.push(c);
                            }
                        }
                        Action::Express(_) => out.expressions.push(c),
                        Action::Ask { .. } => out.queries.push(c),
                        Action::None => {}
                    }
                }
            }
        }
        out.queries.retain(|q| !(q.hit == Hit::Vector && commanding.contains(&q.utterance)));
        // "안경 벗어" names the part and the removal: taking off wins over putting on.
        let removing = |a: &ZenaAssetArgs| matches!(a.action, AssetAction::RemovePart | AssetAction::ResetParts);
        if out.assets.iter().any(removing) {
            out.assets.retain(|a| a.action != AssetAction::EquipPart);
            out.expressions
                .retain(|c| !matches!(&c.action, Action::Asset { asset, .. } if asset.action == AssetAction::EquipPart));
        }
        out.expressions.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.queries.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.decisions = decisions;
        out.feedback = feedback;
        out
    }

    /// Spatial-text NMS: which detected region does `request` refer to?
    pub fn resolve_target(&self, request: &str, boxes: &mut [BoxCand]) -> Option<(usize, Polar)> {
        let q = self.bank.embedder.embed(request);
        select_target(boxes, &q, 0.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_burst_is_deduplicated() {
        let mut r = NlpNmsRouter::default();
        let p = |r, t| Polar { r, theta_deg: t };
        let out = r.route(
            vec![
                RouterInput::Utterance {
                    text: "제나야 방금 그 보스 패턴 어떻게 피해?".into(),
                    t_ms: 0,
                },
                RouterInput::Utterance {
                    text: "보스 패턴 어떻게 피해".into(),
                    t_ms: 10,
                },
                RouterInput::GameEvent {
                    label: "보스 패턴".into(),
                    score: 0.9,
                    polar: p(0.15, 30.0),
                    t_ms: 0,
                },
                RouterInput::GameEvent {
                    label: "몬스터 접근".into(),
                    score: 0.7,
                    polar: p(0.6, 200.0),
                    t_ms: 0,
                },
                RouterInput::GameEvent {
                    label: "몬스터 접근".into(),
                    score: 0.6,
                    polar: p(0.61, 201.0),
                    t_ms: 0,
                },
            ],
            20,
        );
        assert_eq!(out.queries.len(), 1, "{:?}", out.queries);
        assert_eq!(out.queries[0].key, "coach.game_tip");
        let labels: Vec<_> = out.events.iter().map(|e| e.0.as_str()).collect();
        assert_eq!(
            labels,
            vec!["몬스터 접근"],
            "boss alert is covered by the question; duplicate approach merged"
        );
    }

    #[test]
    fn commands_route_to_system_and_repeat_is_ignored() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 오늘 그만 켜둘게, 종료해 줘".into(),
                t_ms: 0,
            }],
            0,
        );
        assert_eq!(out.system, vec![(SystemCommand::ExitApp, None)]);
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "종료해".into(),
                t_ms: 800,
            }],
            800,
        );
        assert!(out.system.is_empty());
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "잠깐 숨어있어".into(),
                t_ms: 900,
            }],
            900,
        );
        assert_eq!(out.system, vec![(SystemCommand::HideAvatar, None)]);
    }

    #[test]
    fn avatar_asset_requests_route_to_assets() {
        use zena_core::tools::{AvatarPart, AvatarTrack};
        let say = |r: &mut NlpNmsRouter, text: &str, t: u64| {
            r.route(
                vec![RouterInput::Utterance {
                    text: text.into(),
                    t_ms: t,
                }],
                t,
            )
        };
        let mut r = NlpNmsRouter::default();
        let out = say(&mut r, "제나야, 로봇 모드로 변신해 줘", 0);
        assert_eq!(out.assets.len(), 1, "{:?}", out.decisions);
        assert_eq!(
            (out.assets[0].action, out.assets[0].track),
            (AssetAction::SwitchTrack, Some(AvatarTrack::Solidcraft))
        );
        assert!(out.expressions.iter().any(|c| c.key == "avatar.robot_mode"), "plays its line too");
        let out = say(&mut r, "안경이랑 헤드셋 써 봐", 5_000);
        let mut parts: Vec<_> = out.assets.iter().filter_map(|a| a.part).collect();
        parts.sort();
        assert_eq!(parts, vec![AvatarPart::Headset, AvatarPart::Glasses], "{:?}", out.assets);
        assert!(
            out.queries.is_empty(),
            "a command is not also a question for the LLM: {:?}",
            out.queries.iter().map(|q| (&q.key, &q.span, q.score, q.hit)).collect::<Vec<_>>()
        );
        // The same command amid game alerts (as in the live pipeline) is still no question.
        let out = r.route(
            vec![
                RouterInput::GameEvent {
                    label: "근접 위협".into(),
                    score: 0.9,
                    polar: Polar { r: 0.1, theta_deg: 30.0 },
                    t_ms: 7_000,
                },
                RouterInput::Utterance {
                    text: "제나야 리본 달아 봐".into(),
                    t_ms: 7_000,
                },
            ],
            7_000,
        );
        assert_eq!(
            out.assets.iter().filter_map(|a| a.part).collect::<Vec<_>>(),
            vec![AvatarPart::Ribbon]
        );
        assert!(
            out.queries.is_empty(),
            "{:?}",
            out.queries.iter().map(|q| &q.key).collect::<Vec<_>>()
        );
        let out = say(&mut r, "안경 벗어", 10_000);
        assert_eq!(
            out.assets.iter().map(|a| a.action).collect::<Vec<_>>(),
            vec![AssetAction::ResetParts],
            "taking off wins"
        );
    }

    /// v2 selection on the hashed space (no models needed): negated commands are dropped, SDS sees every chosen
    /// span, and once a level has its minimum observations the margin follows the ledger.
    #[test]
    fn v2_selection_drops_negations_and_feeds_sds() {
        let models = Models {
            analyzer: Arc::new(RuleAnalyzer),
            selection: Selection::Assign(AssignConfig {
                max_len: 5,
                accept: 0.6,
                margin: 0.0,
                residual: 0.0,
                lexical_weight: 0.0,
                lexical_floor: 0.5,
            }),
            ..Models::v1()
        }
        .with_sds(zena_sds::Ledger::in_memory("nms"));
        let r = NlpNmsRouter::with_models(&Lexicon::bundled(), models);
        let keys = |c: &Catch| c.hits.iter().map(|h| r.bank.entry(h.0).key.clone()).collect::<Vec<_>>();
        assert!(keys(&r.catch("앱 끄지 마")).is_empty());
        let c = r.catch("클릭 통과 하지 마");
        assert!(keys(&c).is_empty(), "{:?}", keys(&c));
        assert_eq!(
            c.negated.iter().map(|(i, _)| r.bank.entry(*i).key.as_str()).collect::<Vec<_>>(),
            ["system.ghost_mode"],
            "the command was caught and held back as negated"
        );
        assert_eq!(
            keys(&r.catch("클릭 통과시켜 줘")),
            ["system.ghost_mode"],
            "without the negation it acts"
        );
        assert_eq!(keys(&r.catch("고마워")), ["emotion.thanks"]);
        let scope = zena_sds::Scope::new("nms", "hash-ngram-v1", "");
        let ledger = r.models.sds.as_ref().unwrap();
        assert!(
            ledger.lock().unwrap().decay_baseline(&scope, "nms.span").is_none(),
            "too few observations"
        );
        for t in [
            "보스 패턴 어떻게 피해",
            "내 인벤토리에서 물약 어디 있어",
            "주식 올랐어",
            "너무 피곤해",
            "심심해",
            "무서워",
        ] {
            r.catch(t);
        }
        let l = ledger.lock().unwrap();
        assert!(l.observations() >= 6, "{}", l.report());
        assert!(l.decay_baseline(&scope, "nms.span").is_some(), "{}", l.report());
    }

    #[test]
    fn small_talk_when_addressed_without_intent() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 오늘 기분 어때".into(),
                t_ms: 0,
            }],
            0,
        );
        assert_eq!(out.queries.first().map(|q| q.key.as_str()), Some(SMALL_TALK));
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "음 그렇구나".into(),
                t_ms: 5000,
            }],
            5000,
        );
        assert!(out.queries.is_empty() && out.system.is_empty());
    }

    #[test]
    fn feelings_are_caught_into_dictionary_expressions() {
        use zena_core::tools::{BodyMotion, EmotionState};
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "와 보스 잡았다! 이겼어!".into(),
                t_ms: 0,
            }],
            0,
        );
        let e = out.expressions.first().expect("an expression");
        assert_eq!(e.key, "emotion.celebrate");
        let x = e.action.express().unwrap();
        assert_eq!((x.emotion, x.motion), (EmotionState::Excited, BodyMotion::Cheer));
        assert!(out.queries.is_empty(), "no LLM for a feeling the dictionary covers");
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "아 또 죽었어 짜증나".into(),
                t_ms: 5000,
            }],
            5000,
        );
        assert_eq!(out.expressions.first().map(|c| c.key.as_str()), Some("emotion.frustrated"));
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "오늘 주식 떡락했어".into(),
                t_ms: 9000,
            }],
            9000,
        );
        assert_eq!(out.expressions.first().map(|c| c.key.as_str()), Some("market.stock_down"));
        // The same thanks twice in a row is said once (semantic NMS over the span).
        let a = r.route(
            vec![RouterInput::Utterance {
                text: "고마워".into(),
                t_ms: 12000,
            }],
            12000,
        );
        let b = r.route(
            vec![RouterInput::Utterance {
                text: "고마워".into(),
                t_ms: 12500,
            }],
            12500,
        );
        assert_eq!(a.expressions.len(), 1);
        assert!(b.expressions.is_empty());
        // Quitting waves goodbye before the command.
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "제나야 종료해 줘".into(),
                t_ms: 20000,
            }],
            20000,
        );
        assert_eq!(out.system, vec![(SystemCommand::ExitApp, None)]);
        assert_eq!(
            out.expressions.first().and_then(|c| c.action.express()).map(|x| x.motion),
            Some(BodyMotion::WaveGoodbye)
        );
    }

    /// The bundled dictionary plus feedback nodes (it has none yet).
    fn feedback_lexicon() -> Lexicon {
        let mut root: serde_json::Value = serde_json::from_str(zena_lexicon::model::BUNDLED_JSON).unwrap();
        root["feedback"] = serde_json::json!({
            "recommend": {
                "semantic": "The user recommends what Zena suggested.",
                "exact_match": ["추천", "강추"],
                "express": {"emotion": "JOY", "motion": "NOD", "lines": ["기억해 둘게!"]}
            },
            "not_recommend": {
                "semantic": "The user advises against what Zena suggested.",
                "exact_match": ["비추", "별로", "나쁘", "나쁜"],
                "express": {"emotion": "WORRIED", "motion": "NOD", "lines": ["다음엔 다른 길로 가 보자."]}
            }
        });
        Lexicon::from_json(&root.to_string()).unwrap()
    }

    #[test]
    fn fc01_negated_feedback_is_kept_in_its_own_channel() {
        let lex = feedback_lexicon();
        for models in [Models::v1(), Models::rules()] {
            let label = models.label.clone();
            let r = NlpNmsRouter::with_models(&lex, models);
            let feedback_hit = |c: &Catch| c.hits.iter().any(|h| r.bank.entry(h.0).key.starts_with(FEEDBACK_PREFIX));
            let span = |c: &Catch, key: &str| {
                c.feedback
                    .iter()
                    .find(|f| f.key == key)
                    .cloned()
                    .unwrap_or_else(|| panic!("{label}: no {key} in {:?}", c.feedback))
            };

            // The long form "-지 않-" (untagged by the pinned rule analyzer): kept, one negator → a weak recommendation.
            let c = r.catch("우회하긴 했는데 나쁘지 않았어");
            let f = span(&c, "feedback.not_recommend");
            assert_eq!(
                (f.span.as_str(), f.negation_count, f.negated(), f.request),
                ("나쁘지", 1, true, false),
                "{label}"
            );
            assert!(c.negated.is_empty(), "{label}: not dropped as a negated command");
            assert!(!feedback_hit(&c), "{label}: a negated feedback does not act as its plain node");

            let c = r.catch("그거 추천 안 해");
            assert_eq!(span(&c, "feedback.recommend").negation_count, 1, "{label}: 추천 + 안 해");
            assert!(c.negated.is_empty() && !feedback_hit(&c), "{label}");
            assert_eq!(
                span(&r.catch("추천은 하지 않아"), "feedback.recommend").negation_count,
                1,
                "{label}"
            );
            assert_eq!(
                span(&r.catch("나쁜 건 아니야"), "feedback.not_recommend").negation_count,
                1,
                "{label}"
            );
            assert_eq!(
                span(&r.catch("별로 안 좋아"), "feedback.not_recommend").negation_count,
                0,
                "{label}: 안 negates 좋아"
            );

            // A request is kept for the extractor but not acted on as feedback.
            let c = r.catch("다른 공략 추천해 줘");
            let f = span(&c, "feedback.recommend");
            assert_eq!((f.request, f.negation_count), (true, 0), "{label}");
            assert!(!feedback_hit(&c), "{label}");

            // Plain feedback acts as its node says and is reported too.
            let c = r.catch("이거 추천");
            assert!(feedback_hit(&c), "{label}");
            assert_eq!(span(&c, "feedback.recommend").negation_count, 0, "{label}");
        }
        // Commands are held back as before (the rule analyzer marks the prohibition).
        let r = NlpNmsRouter::with_models(&lex, Models::rules());
        let c = r.catch("앱 끄지 마");
        assert!(c.feedback.is_empty() && c.hits.is_empty(), "{:?}", c.hits);
    }

    #[test]
    fn route_passes_feedback_through_with_its_words() {
        let mut r = NlpNmsRouter::with_models(&feedback_lexicon(), Models::rules());
        let text = "제나야 우회하긴 했는데 나쁘지 않았어";
        let out = r.route(
            vec![RouterInput::Utterance {
                text: text.into(),
                t_ms: 5,
            }],
            5,
        );
        assert_eq!(out.feedback.len(), 1, "{:?}", out.feedback);
        let fc = &out.feedback[0];
        assert_eq!((fc.utterance.as_str(), fc.t_ms), (text, 5));
        let surfaces: Vec<&str> = fc.words.iter().map(|w| w.surface.as_str()).collect();
        assert_eq!(surfaces, ["우회하긴", "했는데", "나쁘지", "않았어"]);
        let w = |s: &str| fc.words.iter().find(|w| w.surface == s).unwrap();
        assert!(w("않았어").is_negation && w("나쁘지").negated_in_clause);
        assert!(!w("우회하긴").negated_in_clause && !w("했는데").is_negation);
        assert!(
            out.queries.iter().all(|q| q.key != SMALL_TALK),
            "addressed feedback is no small talk"
        );
        assert!(
            out.expressions.iter().all(|e| !e.key.starts_with(FEEDBACK_PREFIX)),
            "no expression for a negated span"
        );

        let out = r.route(
            vec![RouterInput::Utterance {
                text: "이거 추천".into(),
                t_ms: 10_000,
            }],
            10_000,
        );
        assert!(
            out.expressions.iter().any(|e| e.key == "feedback.recommend"),
            "{:?}",
            out.expressions
        );
        assert_eq!(out.feedback[0].spans[0].negation_count, 0);
        let topic = feedback_words(&normalize("공략은 맞는데").words);
        assert!(topic[0].is_subject_or_topic && !topic[1].is_subject_or_topic);
        let req = feedback_words(&normalize("추천해 줘").words);
        assert!(req[1].is_auxiliary_request);
    }

    #[test]
    fn bundled_lexicon_populates_feedback_channel_out_of_the_box() {
        let mut r = NlpNmsRouter::with_models(&Lexicon::bundled(), Models::rules());
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "우회하긴 했는데 나쁘지 않았어".into(),
                t_ms: 10,
            }],
            10,
        );
        assert_eq!(out.feedback.len(), 1, "bundled lexicon now has feedback.* nodes");
        assert_eq!(out.feedback[0].spans[0].key, "feedback.not_recommend");
        assert!(out.feedback[0].spans[0].negated());
        let rec = r.route(
            vec![RouterInput::Utterance {
                text: "이거 추천".into(),
                t_ms: 10_000,
            }],
            10_000,
        );
        assert_eq!(rec.feedback.len(), 1);
        assert_eq!(rec.feedback[0].spans[0].key, "feedback.recommend");
        assert!(!rec.feedback[0].spans[0].negated());
    }

    #[test]
    fn a_reloaded_dictionary_catches_new_words() {
        let mut r = NlpNmsRouter::default();
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "개꿀잼".into(),
                t_ms: 0,
            }],
            0,
        );
        assert!(out.expressions.is_empty());
        let mut lex = Lexicon::bundled();
        let i = lex.nodes.iter().position(|n| n.key == "emotion.celebrate").unwrap();
        lex.nodes[i].exact_match.push("개꿀잼".into());
        r.reload(&lex);
        let out = r.route(
            vec![RouterInput::Utterance {
                text: "개꿀잼".into(),
                t_ms: 100,
            }],
            100,
        );
        assert_eq!(
            out.expressions.first().map(|c| (c.key.as_str(), c.hit)),
            Some(("emotion.celebrate", Hit::Exact))
        );
    }
}
