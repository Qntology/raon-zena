//! Incremental `<tool_call>` parsing while Raon-1-nano is still generating.
//!
//! The model writes `control_zena_avatar` parameters in schema order — gaze, emotion, motion, emote, then the
//! line. [`ToolCallStream`] turns the token deltas into [`StreamEvent`]s as soon as each piece is complete:
//!
//! * an expressive parameter closes (`emotion_state`, `body_motion`, `character_emote`, or `gaze_target` once one
//!   of those is known) → [`StreamEvent::Expression`]: Zena makes the face and moves before the line exists;
//! * `speech_text` is cut at sentence ends while it is being written → [`StreamEvent::Speech`] per sentence, for
//!   TTS and the speech bubble;
//! * `manage_zena_system` / `manage_avatar_asset` calls → [`StreamEvent::Call`] when their `</tool_call>` closes.
//!
//! A completion with no tool call is spoken as plain text by [`ToolCallStream::finish`]. The events of a whole
//! completion carry the same calls [`crate::tools::parse_model_output`] finds in it.

use crate::tools::{
    param_value, strip_think, to_call, truncate_speech, BodyMotion, EmotionState, ToolCall, ZenaAvatarArgs, MAX_SPEECH_CHARS,
};
use serde_json::{Map, Value};

const AVATAR_FN: &str = "control_zena_avatar";
const SPEECH: &str = "speech_text";
const TAGS: [&str; 7] = [
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<think>",
];

#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Gaze / emotion / motion / emote to play now (`speech_text` is always `None`).
    Expression(ZenaAvatarArgs),
    /// The next sentence of the line.
    Speech(String),
    /// A complete system or asset call.
    Call(ToolCall),
    /// A call that failed validation.
    Rejected { name: String, error: String },
}

#[derive(Debug, Clone, PartialEq, Default)]
enum St {
    #[default]
    Outside,
    InCall,
    InFunction,
    InParam {
        key: String,
        start: usize,
    },
    AfterFunction,
}

/// Feed it the text deltas of one completion, then call [`ToolCallStream::finish`].
#[derive(Debug, Default)]
pub struct ToolCallStream {
    buf: String,
    cursor: usize,
    st: St,
    function: Option<String>,
    params: Map<String, Value>,
    /// Start of the not yet spoken part of `speech_text` (byte offset into `buf`).
    speech_from: usize,
    spoken: String,
    spoken_chars: usize,
    expr_sent: Option<ZenaAvatarArgs>,
    saw_call: bool,
    finished: bool,
}

/// Where a sentence may end inside `text`: after `.!?~…。` followed by whitespace, or (with `final_`) the end.
/// A comma ends a chunk too once at least 20 characters are waiting, so long lines start speaking early.
fn chunk_end(text: &str, final_: bool) -> usize {
    if final_ {
        return text.len();
    }
    let mut end = 0;
    let mut chars = text.char_indices().peekable();
    let mut count = 0;
    while let Some((i, c)) = chars.next() {
        count += 1;
        let next_ws = chars.peek().is_some_and(|(_, n)| n.is_whitespace());
        let hard = matches!(c, '.' | '!' | '?' | '~' | '…' | '。');
        let soft = c == ',' && count >= 20;
        if (hard || soft) && next_ws {
            end = i + c.len_utf8();
        }
    }
    end
}

/// Length of `rest` that can be consumed without cutting a tag that may still be arriving.
fn safe_len(rest: &str) -> usize {
    let max = TAGS.iter().map(|t| t.len()).max().unwrap_or(0);
    let from = rest.len().saturating_sub(max);
    for (i, _) in rest.char_indices().filter(|(i, c)| *i >= from && *c == '<') {
        let tail = &rest[i..];
        if TAGS.iter().any(|t| t.len() > tail.len() && t.starts_with(tail)) {
            return i;
        }
    }
    rest.len()
}

impl ToolCallStream {
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything spoken so far (sentences joined by spaces).
    pub fn spoken(&self) -> &str {
        &self.spoken
    }

    /// Whether any tool call started.
    pub fn saw_call(&self) -> bool {
        self.saw_call
    }

    pub fn push(&mut self, delta: &str) -> Vec<StreamEvent> {
        let mut ev = Vec::new();
        if self.finished {
            return ev;
        }
        self.buf.push_str(delta);
        self.advance(&mut ev, false);
        ev
    }

    /// The completion ended (EOS or the token limit): flush what is left.
    pub fn finish(&mut self) -> Vec<StreamEvent> {
        let mut ev = Vec::new();
        if self.finished {
            return ev;
        }
        self.advance(&mut ev, true);
        if let St::InParam { key, start } = std::mem::take(&mut self.st) {
            let end = self.buf.len();
            self.close_param(&key, start, end, &mut ev);
        }
        if self.function.is_some() {
            self.finish_call(&mut ev);
        }
        if !self.saw_call {
            let text = strip_think(&self.buf).trim().to_string();
            if !text.is_empty() {
                let args = ZenaAvatarArgs {
                    body_motion: BodyMotion::Nod,
                    emotion_state: EmotionState::Neutral,
                    ..Default::default()
                };
                ev.push(StreamEvent::Expression(args));
                self.say(&text, &mut ev);
            }
        }
        self.finished = true;
        ev
    }

    fn advance(&mut self, ev: &mut Vec<StreamEvent>, at_end: bool) {
        loop {
            let rest = &self.buf[self.cursor..];
            match self.st.clone() {
                St::Outside => {
                    // Skip a think block if the model wrote one.
                    if let Some(t) = rest.find("<think>") {
                        if rest.find("<tool_call>").is_none_or(|c| t < c) {
                            match rest[t..].find("</think>") {
                                Some(e) => {
                                    self.cursor += t + e + "</think>".len();
                                    continue;
                                }
                                None => return,
                            }
                        }
                    }
                    match rest.find("<tool_call>") {
                        Some(i) => {
                            self.cursor += i + "<tool_call>".len();
                            self.saw_call = true;
                            self.st = St::InCall;
                        }
                        None => {
                            let n = if at_end { rest.len() } else { safe_len(rest) };
                            self.cursor += n;
                            return;
                        }
                    }
                }
                St::InCall => {
                    let f = rest.find("<function=");
                    let close = rest.find("</tool_call>");
                    match (f, close) {
                        (Some(f), c) if c.is_none_or(|c| f < c) => match rest[f..].find('>') {
                            Some(gt) => {
                                let name = rest[f + "<function=".len()..f + gt].trim().to_string();
                                self.function = Some(name);
                                self.params.clear();
                                self.expr_sent = None;
                                self.cursor += f + gt + 1;
                                self.st = St::InFunction;
                            }
                            None => return,
                        },
                        (_, Some(c)) => {
                            ev.push(StreamEvent::Rejected {
                                name: "<malformed>".into(),
                                error: "missing <function=".into(),
                            });
                            self.cursor += c + "</tool_call>".len();
                            self.st = St::Outside;
                        }
                        _ => return,
                    }
                }
                St::InFunction => {
                    let p = rest.find("<parameter=");
                    let ef = rest.find("</function>");
                    let ec = rest.find("</tool_call>");
                    let first = [p, ef, ec].into_iter().flatten().min();
                    match first {
                        Some(i) if Some(i) == p => match rest[i..].find('>') {
                            Some(gt) => {
                                let key = rest[i + "<parameter=".len()..i + gt].trim().to_string();
                                self.cursor += i + gt + 1;
                                if key == SPEECH {
                                    self.speech_from = self.cursor;
                                }
                                self.st = St::InParam { key, start: self.cursor };
                            }
                            None => return,
                        },
                        Some(i) if Some(i) == ef => {
                            self.finish_call(ev);
                            self.cursor += i + "</function>".len();
                            self.st = St::AfterFunction;
                        }
                        Some(i) => {
                            self.finish_call(ev);
                            self.cursor += i + "</tool_call>".len();
                            self.st = St::Outside;
                        }
                        None => return,
                    }
                }
                St::InParam { key, start } => {
                    // The value ends at `</parameter>`, or where the model went on without closing it.
                    let ends = [
                        rest.find("</parameter>").map(|i| (i, "</parameter>".len())),
                        rest.find("<parameter=").map(|i| (i, 0)),
                        rest.find("</function>").map(|i| (i, 0)),
                        rest.find("</tool_call>").map(|i| (i, 0)),
                    ];
                    match ends.into_iter().flatten().min_by_key(|(i, _)| *i) {
                        Some((i, skip)) => {
                            let end = self.cursor + i;
                            self.close_param(&key, start, end, ev);
                            self.cursor = end + skip;
                            self.st = St::InFunction;
                        }
                        None => {
                            if key == SPEECH {
                                let end = self.cursor + safe_len(rest);
                                self.stream_speech(end, false, ev);
                            }
                            return;
                        }
                    }
                }
                St::AfterFunction => match rest.find("</tool_call>") {
                    Some(i) => {
                        self.cursor += i + "</tool_call>".len();
                        self.st = St::Outside;
                    }
                    None if rest.contains("<tool_call>") => self.st = St::Outside,
                    None => return,
                },
            }
        }
    }

    fn close_param(&mut self, key: &str, start: usize, end: usize, ev: &mut Vec<StreamEvent>) {
        if key == SPEECH {
            self.stream_speech(end, true, ev);
            return;
        }
        let v = param_value(&self.buf[start..end]);
        self.params.insert(key.to_string(), v);
        if self.function.as_deref() == Some(AVATAR_FN) {
            self.emit_expression(ev);
        }
    }

    fn stream_speech(&mut self, end: usize, final_: bool, ev: &mut Vec<StreamEvent>) {
        if self.function.as_deref() != Some(AVATAR_FN) || end <= self.speech_from {
            return;
        }
        let pending = &self.buf[self.speech_from..end];
        let n = chunk_end(pending, final_);
        if n == 0 {
            return;
        }
        let chunk = pending[..n].trim().to_string();
        self.speech_from += n;
        if !chunk.is_empty() {
            // The line plays with the expression even if no expressive parameter came first.
            if self.expr_sent.is_none() {
                self.emit_expression_forced(ev);
            }
            self.say(&chunk, ev);
        }
    }

    /// Speak `text`, keeping the whole line within [`MAX_SPEECH_CHARS`].
    fn say(&mut self, text: &str, ev: &mut Vec<StreamEvent>) {
        let sep = usize::from(!self.spoken.is_empty());
        let room = MAX_SPEECH_CHARS.saturating_sub(self.spoken_chars + sep);
        // Later sentences that do not fit are dropped whole; only the first one is cut.
        let t = if text.chars().count() <= room {
            text.to_string()
        } else if sep == 0 {
            truncate_speech(text, room)
        } else {
            return;
        };
        if t.is_empty() {
            return;
        }
        if sep == 1 {
            self.spoken.push(' ');
        }
        self.spoken_chars += t.chars().count() + sep;
        self.spoken.push_str(&t);
        ev.push(StreamEvent::Speech(t));
    }

    fn current_expression(&self) -> Option<ZenaAvatarArgs> {
        let mut p = self.params.clone();
        p.remove(SPEECH);
        match to_call(AVATAR_FN, Value::Object(p)) {
            Ok(ToolCall::ControlZenaAvatar(a)) => Some(ZenaAvatarArgs { speech_text: None, ..a }),
            _ => None,
        }
    }

    fn emit_expression(&mut self, ev: &mut Vec<StreamEvent>) {
        // Gaze alone waits for the emotion or motion, so Zena does not snap to a neutral face first.
        if !["emotion_state", "body_motion", "character_emote"]
            .iter()
            .any(|k| self.params.contains_key(*k))
        {
            return;
        }
        self.emit_expression_forced(ev);
    }

    fn emit_expression_forced(&mut self, ev: &mut Vec<StreamEvent>) {
        if let Some(a) = self.current_expression() {
            if self.expr_sent.as_ref() != Some(&a) {
                self.expr_sent = Some(a.clone());
                ev.push(StreamEvent::Expression(a));
            }
        }
    }

    fn finish_call(&mut self, ev: &mut Vec<StreamEvent>) {
        let Some(name) = self.function.take() else { return };
        let mut params = std::mem::take(&mut self.params);
        if name == AVATAR_FN {
            params.remove(SPEECH);
            match to_call(&name, Value::Object(params)) {
                Ok(ToolCall::ControlZenaAvatar(a)) => {
                    let a = ZenaAvatarArgs { speech_text: None, ..a };
                    if self.expr_sent.as_ref() != Some(&a) {
                        ev.push(StreamEvent::Expression(a.clone()));
                        self.expr_sent = Some(a);
                    }
                }
                Ok(_) => {}
                Err(e) => ev.push(StreamEvent::Rejected { name, error: e }),
            }
        } else {
            match to_call(&name, Value::Object(params)) {
                Ok(c) => ev.push(StreamEvent::Call(c)),
                Err(e) => ev.push(StreamEvent::Rejected { name, error: e }),
            }
        }
        self.expr_sent = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{parse_model_output, CharacterEmote, SystemCommand};

    const OUT: &str = "<tool_call>\n<function=control_zena_avatar>\n<parameter=gaze_target>\n{\"angle_deg\": 45, \"distance_norm\": 0.3}\n</parameter>\n<parameter=emotion_state>\nSURPRISED_URGENT\n</parameter>\n<parameter=body_motion>\nLEAN_FORWARD_POINT\n</parameter>\n<parameter=speech_text>\n오른쪽 위에 보스야! 지금 바로 굴러서 피해, 그다음 반격!\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=manage_zena_system>\n<parameter=command>\nSET_TOPMOST_ON\n</parameter>\n</function>\n</tool_call>";

    /// Feed `text` in pieces of `step` characters (token-sized), collecting events with the byte offset at which
    /// each one appeared.
    fn feed(text: &str, step: usize) -> Vec<(usize, StreamEvent)> {
        let mut s = ToolCallStream::new();
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut at = 0;
        for piece in chars.chunks(step) {
            let d: String = piece.iter().collect();
            at += d.len();
            out.extend(s.push(&d).into_iter().map(|e| (at, e)));
        }
        out.extend(s.finish().into_iter().map(|e| (text.len(), e)));
        out
    }

    #[test]
    fn expression_comes_before_the_line_and_the_line_streams_by_sentence() {
        for step in [1, 2, 3, 7, 1000] {
            let ev = feed(OUT, step);
            let first_expr = ev.iter().position(|(_, e)| matches!(e, StreamEvent::Expression(_))).unwrap();
            let first_speech = ev.iter().position(|(_, e)| matches!(e, StreamEvent::Speech(_))).unwrap();
            assert!(first_expr < first_speech, "step {step}: {ev:?}");
            let speech: Vec<&str> = ev
                .iter()
                .filter_map(|(_, e)| match e {
                    StreamEvent::Speech(s) => Some(s.as_str()),
                    _ => None,
                })
                .collect();
            if step < 1000 {
                assert_eq!(
                    speech,
                    ["오른쪽 위에 보스야!", "지금 바로 굴러서 피해, 그다음 반격!"],
                    "step {step}"
                );
                // The first sentence is out before the parameter closes.
                let (at, _) = ev.iter().find(|(_, e)| matches!(e, StreamEvent::Speech(_))).unwrap();
                assert!(*at < OUT.find("</parameter>\n</function>").unwrap(), "step {step}");
            }
            let StreamEvent::Expression(last) = ev
                .iter()
                .rev()
                .find_map(|(_, e)| matches!(e, StreamEvent::Expression(_)).then_some(e))
                .unwrap()
            else {
                unreachable!()
            };
            assert_eq!(last.emotion_state, EmotionState::SurprisedUrgent);
            assert_eq!(last.body_motion, BodyMotion::LeanForwardPoint);
            assert_eq!(last.gaze_target.unwrap().angle_deg, 45.0);
            assert_eq!(last.emote(), Some(CharacterEmote::ThreatArrow3d));
            assert!(ev.iter().any(|(_, e)| matches!(
                e,
                StreamEvent::Call(ToolCall::ManageZenaSystem(s)) if s.command == SystemCommand::SetTopmostOn
            )));
        }
    }

    #[test]
    fn the_stream_agrees_with_the_batch_parser() {
        let batch = parse_model_output(OUT);
        let ToolCall::ControlZenaAvatar(a) = &batch.calls[0] else {
            panic!()
        };
        let mut s = ToolCallStream::new();
        let mut ev = s.push(OUT);
        ev.extend(s.finish());
        assert_eq!(s.spoken(), a.speech_text.as_deref().unwrap());
        let calls: Vec<&ToolCall> = ev
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Call(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls, vec![&batch.calls[1]]);
    }

    #[test]
    fn plain_text_is_spoken_at_the_end_and_truncated_lines_are_flushed() {
        let ev = feed("인벤토리는 I 키로 열 수 있어.", 2);
        assert!(matches!(&ev[0].1, StreamEvent::Expression(a) if a.body_motion == BodyMotion::Nod));
        assert_eq!(ev[1].1, StreamEvent::Speech("인벤토리는 I 키로 열 수 있어.".into()));
        // The token limit hit in the middle of the line.
        let cut = "<tool_call>\n<function=control_zena_avatar>\n<parameter=emotion_state>\nJOY\n</parameter>\n<parameter=speech_text>\n좋았어! 이대로 계속";
        let ev = feed(cut, 3);
        let speech: Vec<_> = ev
            .iter()
            .filter_map(|(_, e)| match e {
                StreamEvent::Speech(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(speech, ["좋았어!", "이대로 계속"]);
    }

    #[test]
    fn the_whole_line_stays_within_the_speech_limit() {
        let long = format!(
            "<tool_call>\n<function=control_zena_avatar>\n<parameter=emotion_state>\nFOCUSED\n</parameter>\n<parameter=speech_text>\n{}\n</parameter>\n</function>\n</tool_call>",
            "짧게 말할게. ".repeat(30)
        );
        let mut s = ToolCallStream::new();
        let mut n = 0;
        for c in long.chars() {
            for e in s.push(&c.to_string()) {
                if let StreamEvent::Speech(t) = e {
                    n += t.chars().count() + 1;
                }
            }
        }
        s.finish();
        assert!(n <= MAX_SPEECH_CHARS + 1, "{n}");
        assert!(s.spoken().chars().count() <= MAX_SPEECH_CHARS);
    }

    #[test]
    fn think_blocks_partial_tags_and_bad_calls() {
        let ev = feed(
            "<think>음...</think>\n<tool_call>\n<function=nuke_disk>\n</function>\n</tool_call>",
            1,
        );
        assert!(matches!(&ev[..], [(_, StreamEvent::Rejected { name, .. })] if name == "nuke_disk"));
        assert_eq!(safe_len("abc<tool_"), 3);
        assert_eq!(safe_len("abc<b>"), 6);
        assert_eq!(safe_len("값</param"), "값".len());
        // A gaze alone waits for the emotion.
        let ev = feed("<tool_call>\n<function=control_zena_avatar>\n<parameter=gaze_target>\n{\"angle_deg\": 90, \"distance_norm\": 0.5}\n</parameter>\n", 4);
        assert!(ev.is_empty() || matches!(&ev[..], [(_, StreamEvent::Expression(_))]), "{ev:?}");
    }
}
