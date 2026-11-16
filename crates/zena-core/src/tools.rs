//! Tool-call schema shared by Raon-1-nano, the avatar adapter and the window controller.
//!
//! Two wire formats are accepted:
//! * the JSON envelope of PRD §4.1 (`{"tool_calls":[{"name":…,"arguments":{…}}]}`), and
//! * Raon-1-nano's native chat-template format:
//!   `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>…</function>\n</tool_call>`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GazeTarget {
    /// Screen-space azimuth around the player, `0°` = right, `90°` = up.
    pub angle_deg: f32,
    /// Normalised distance 0 (player) .. 1 (screen corner).
    pub distance_norm: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EmotionState {
    Joy,
    Excited,
    Surprised,
    SurprisedUrgent,
    Angry,
    Sorrow,
    Worried,
    Focused,
    Relieved,
    #[serde(other)]
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BodyMotion {
    Nod,
    LeanForwardPoint,
    Cheer,
    Clap,
    Flinch,
    Think,
    HideCrouch,
    WaveGoodbye,
    #[serde(other)]
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SystemCommand {
    ExitApp,
    SetTopmostOn,
    SetTopmostOff,
    HideAvatar,
    ShowAvatar,
    #[serde(other)]
    KeepActive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WindowLayerMode {
    GhostClickThrough,
    InteractivePet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZenaAvatarArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gaze_target: Option<GazeTarget>,
    #[serde(default = "neutral")]
    pub emotion_state: EmotionState,
    #[serde(default = "idle")]
    pub body_motion: BodyMotion,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speech_text: Option<String>,
}

fn neutral() -> EmotionState {
    EmotionState::Neutral
}
fn idle() -> BodyMotion {
    BodyMotion::Idle
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZenaSystemArgs {
    pub command: SystemCommand,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_layer_mode: Option<WindowLayerMode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", content = "arguments", rename_all = "snake_case")]
pub enum ToolCall {
    ControlZenaAvatar(ZenaAvatarArgs),
    ManageZenaSystem(ZenaSystemArgs),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ToolCallEnvelope {
    pub tool_calls: Vec<ToolCall>,
}

/// Maximum speech length kept from a generated line (characters). Longer lines are cut at the
/// last sentence boundary so TTS latency stays bounded.
pub const MAX_SPEECH_CHARS: usize = 90;

impl ToolCall {
    /// Clamp/normalise model output into the ranges the renderer expects.
    pub fn sanitized(mut self) -> Self {
        if let ToolCall::ControlZenaAvatar(a) = &mut self {
            if let Some(g) = &mut a.gaze_target {
                let mut t = g.angle_deg % 360.0;
                if t < 0.0 {
                    t += 360.0;
                }
                g.angle_deg = if t.is_finite() { t } else { 0.0 };
                g.distance_norm = if g.distance_norm.is_finite() {
                    g.distance_norm.clamp(0.0, 1.0)
                } else {
                    0.5
                };
            }
            if let Some(s) = &mut a.speech_text {
                *s = truncate_speech(s.trim(), MAX_SPEECH_CHARS);
            }
            if a.speech_text.as_deref() == Some("") {
                a.speech_text = None;
            }
        }
        self
    }
}

fn truncate_speech(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    match cut.rfind(['.', '!', '?', '。', '~']) {
        Some(i) if i > 0 => cut[..i + cut[i..].chars().next().unwrap().len_utf8()].to_string(),
        _ => cut,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedOutput {
    /// Text the model wrote before its first tool call (optional reasoning/plain answer).
    pub preamble: String,
    pub calls: Vec<ToolCall>,
    /// Calls that were syntactically present but failed validation (name, message).
    pub rejected: Vec<(String, String)>,
}

/// Parse a model completion in either supported format.
pub fn parse_model_output(text: &str) -> ParsedOutput {
    let text = strip_think(text);
    if let Some(start) = text.find("<tool_call>") {
        let preamble = text[..start].trim().to_string();
        let mut calls = Vec::new();
        let mut rejected = Vec::new();
        let mut rest = &text[start..];
        while let Some(s) = rest.find("<tool_call>") {
            let body_start = s + "<tool_call>".len();
            let end = rest[body_start..].find("</tool_call>").map(|e| body_start + e);
            let body = &rest[body_start..end.unwrap_or(rest.len())];
            match parse_xml_call(body) {
                Ok((name, args)) => push_call(&name, args, &mut calls, &mut rejected),
                Err(e) => rejected.push(("<malformed>".into(), e)),
            }
            rest = match end {
                Some(e) => &rest[e + "</tool_call>".len()..],
                None => "",
            };
        }
        return ParsedOutput { preamble, calls, rejected };
    }
    // JSON envelope (possibly wrapped in prose or a code fence).
    if let Some(v) = first_json_object(text) {
        let mut calls = Vec::new();
        let mut rejected = Vec::new();
        let items: Vec<Value> = match v.get("tool_calls") {
            Some(Value::Array(a)) => a.clone(),
            _ if v.get("name").is_some() => vec![v.clone()],
            _ => vec![],
        };
        for it in items {
            let it = it.get("function").cloned().unwrap_or(it);
            let name = it.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let mut args = it.get("arguments").cloned().unwrap_or(Value::Object(Map::new()));
            if let Value::String(s) = &args {
                args = serde_json::from_str(s).unwrap_or(Value::Object(Map::new()));
            }
            push_call(&name, args, &mut calls, &mut rejected);
        }
        if !calls.is_empty() || !rejected.is_empty() {
            return ParsedOutput {
                preamble: String::new(),
                calls,
                rejected,
            };
        }
    }
    ParsedOutput {
        preamble: text.trim().to_string(),
        calls: vec![],
        rejected: vec![],
    }
}

fn strip_think(text: &str) -> &str {
    match text.find("</think>") {
        Some(i) => &text[i + "</think>".len()..],
        None => text,
    }
}

fn push_call(name: &str, args: Value, calls: &mut Vec<ToolCall>, rejected: &mut Vec<(String, String)>) {
    let v = json!({ "name": name, "arguments": args });
    match serde_json::from_value::<ToolCall>(v) {
        Ok(c) => calls.push(c.sanitized()),
        Err(e) => rejected.push((name.to_string(), e.to_string())),
    }
}

fn parse_xml_call(body: &str) -> Result<(String, Value), String> {
    let f = body.find("<function=").ok_or("missing <function=")?;
    let after = &body[f + "<function=".len()..];
    let gt = after.find('>').ok_or("unterminated <function=")?;
    let name = after[..gt].trim().to_string();
    let mut args = Map::new();
    let mut rest = &after[gt + 1..];
    while let Some(p) = rest.find("<parameter=") {
        let a = &rest[p + "<parameter=".len()..];
        let gt = a.find('>').ok_or("unterminated <parameter=")?;
        let key = a[..gt].trim().to_string();
        let v = &a[gt + 1..];
        let end = v.find("</parameter>").unwrap_or(v.len());
        let raw = v[..end].trim_matches(|c| c == '\n' || c == '\r');
        let val = serde_json::from_str::<Value>(raw.trim())
            .ok()
            .filter(|x| x.is_object() || x.is_array() || x.is_number() || x.is_boolean())
            .unwrap_or_else(|| Value::String(raw.trim().to_string()));
        args.insert(key, val);
        rest = if end < v.len() { &v[end + "</parameter>".len()..] } else { "" };
    }
    Ok((name, Value::Object(args)))
}

/// Extract the first balanced top-level `{…}` JSON object from free text.
pub fn first_json_object(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while let Some(off) = text[i..].find('{') {
        let start = i + off;
        let (mut depth, mut in_str, mut esc) = (0i32, false, false);
        for (j, &b) in bytes.iter().enumerate().skip(start) {
            if in_str {
                match (esc, b) {
                    (true, _) => esc = false,
                    (false, b'\\') => esc = true,
                    (false, b'"') => in_str = false,
                    _ => {}
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str(&text[start..=j]) {
                            return Some(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        i = start + 1;
    }
    None
}

/// Tool definitions injected into the system prompt (`tools` of the chat template).
pub fn tool_definitions() -> Value {
    json!([
      {
        "type": "function",
        "function": {
          "name": "control_zena_avatar",
          "description": "Drive Zena's gaze, facial emotion, body motion and an optional short spoken line.",
          "parameters": {
            "type": "object",
            "properties": {
              "gaze_target": {
                "type": "object",
                "description": "Where to look, around the player at screen centre.",
                "properties": {
                  "angle_deg": {"type": "number", "description": "0=right, 90=up, 180=left, 270=down"},
                  "distance_norm": {"type": "number", "description": "0=player, 1=screen corner"}
                },
                "required": ["angle_deg", "distance_norm"]
              },
              "emotion_state": {"type": "string", "enum": ["NEUTRAL","JOY","EXCITED","SURPRISED","SURPRISED_URGENT","ANGRY","SORROW","WORRIED","FOCUSED","RELIEVED"]},
              "body_motion": {"type": "string", "enum": ["IDLE","NOD","LEAN_FORWARD_POINT","CHEER","CLAP","FLINCH","THINK","HIDE_CROUCH","WAVE_GOODBYE"]},
              "speech_text": {"type": "string", "description": "One short coaching sentence (<= 90 chars)."}
            },
            "required": ["emotion_state", "body_motion"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "manage_zena_system",
          "description": "Change Zena's window/app state.",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {"type": "string", "enum": ["KEEP_ACTIVE","EXIT_APP","SET_TOPMOST_ON","SET_TOPMOST_OFF","HIDE_AVATAR","SHOW_AVATAR"]},
              "window_layer_mode": {"type": "string", "enum": ["GHOST_CLICK_THROUGH","INTERACTIVE_PET"]}
            },
            "required": ["command"]
          }
        }
      }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRD_JSON: &str = r#"{
      "tool_calls": [
        {"name": "control_zena_avatar", "arguments": {
            "gaze_target": { "angle_deg": 45, "distance_norm": 0.35 },
            "emotion_state": "SURPRISED_URGENT",
            "body_motion": "LEAN_FORWARD_POINT",
            "speech_text": "오른쪽 바로 코앞에 보스가 나타났어! 체력이 바닥이니까 당장 회피기 써!"}},
        {"name": "manage_zena_system", "arguments": {"command": "KEEP_ACTIVE", "window_layer_mode": "GHOST_CLICK_THROUGH"}}
      ]}"#;

    #[test]
    fn parses_prd_json_envelope() {
        let p = parse_model_output(PRD_JSON);
        assert_eq!(p.calls.len(), 2, "{:?}", p.rejected);
        match &p.calls[0] {
            ToolCall::ControlZenaAvatar(a) => {
                assert_eq!(a.emotion_state, EmotionState::SurprisedUrgent);
                assert_eq!(a.body_motion, BodyMotion::LeanForwardPoint);
                assert_eq!(a.gaze_target.unwrap().angle_deg, 45.0);
            }
            c => panic!("{c:?}"),
        }
        assert_eq!(
            p.calls[1],
            ToolCall::ManageZenaSystem(ZenaSystemArgs {
                command: SystemCommand::KeepActive,
                window_layer_mode: Some(WindowLayerMode::GhostClickThrough)
            })
        );
    }

    #[test]
    fn parses_raon_native_format() {
        let out = "<think></think>\n\n<tool_call>\n<function=control_zena_avatar>\n<parameter=gaze_target>\n{\"angle_deg\": -90, \"distance_norm\": 1.7}\n</parameter>\n<parameter=emotion_state>\nWORRIED\n</parameter>\n<parameter=body_motion>\nDANCE\n</parameter>\n<parameter=speech_text>\n아래쪽 조심해!\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=manage_zena_system>\n<parameter=command>\nEXIT_APP\n</parameter>\n</function>\n</tool_call>";
        let p = parse_model_output(out);
        assert_eq!(p.calls.len(), 2, "{:?}", p.rejected);
        let ToolCall::ControlZenaAvatar(a) = &p.calls[0] else { panic!() };
        let g = a.gaze_target.unwrap();
        assert_eq!(g.angle_deg, 270.0);
        assert_eq!(g.distance_norm, 1.0);
        assert_eq!(a.emotion_state, EmotionState::Worried);
        assert_eq!(a.body_motion, BodyMotion::Idle, "unknown motion falls back to IDLE");
        assert_eq!(a.speech_text.as_deref(), Some("아래쪽 조심해!"));
        assert!(matches!(
            p.calls[1],
            ToolCall::ManageZenaSystem(ZenaSystemArgs {
                command: SystemCommand::ExitApp,
                ..
            })
        ));
    }

    #[test]
    fn unknown_tool_is_rejected_and_plain_text_passes_through() {
        let p = parse_model_output("<tool_call>\n<function=format_disk>\n</function>\n</tool_call>");
        assert!(p.calls.is_empty());
        assert_eq!(p.rejected.len(), 1);
        let p = parse_model_output("그냥 대답이야.");
        assert_eq!(p.preamble, "그냥 대답이야.");
    }

    #[test]
    fn speech_is_truncated_at_sentence_boundary() {
        let long = "가".repeat(80) + "! " + &"나".repeat(40);
        let t = truncate_speech(&long, MAX_SPEECH_CHARS);
        assert!(t.ends_with('!'));
        assert!(t.chars().count() <= MAX_SPEECH_CHARS);
    }

    #[test]
    fn roundtrip_serialization_matches_prd_shape() {
        let p = parse_model_output(PRD_JSON);
        let env = ToolCallEnvelope { tool_calls: p.calls };
        let v = serde_json::to_value(&env).unwrap();
        assert_eq!(v["tool_calls"][0]["name"], "control_zena_avatar");
        assert_eq!(v["tool_calls"][1]["arguments"]["window_layer_mode"], "GHOST_CLICK_THROUGH");
        assert!(tool_definitions().as_array().unwrap().len() == 2);
    }
}
