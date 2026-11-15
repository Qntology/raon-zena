//! Lv3 generation backends.
//!
//! The production backend is Raon-1-nano running from the pyramid GGUF (RDA + MLA + Raon-mHC
//! forward pass with the fused kernels of `raon-quant`). That forward pass is not implemented
//! yet; [`TemplateBackend`] stands in for it and emits the *same wire format* the model's chat
//! template produces (`<tool_call><function=…><parameter=…>`), so prompt building, parsing,
//! routing and rendering are exercised end to end.

use zena_core::prompt::Situation;

pub trait LlmBackend: Send {
    fn name(&self) -> &'static str;
    /// Generate a completion for the fast-path prompt. `situation` is passed alongside the
    /// rendered prompt so non-neural backends can respond without parsing text.
    fn generate(&mut self, prompt: &str, situation: &Situation) -> String;
}

pub struct TemplateBackend;

impl LlmBackend for TemplateBackend {
    fn name(&self) -> &'static str {
        "template-stub"
    }

    fn generate(&mut self, _prompt: &str, s: &Situation) -> String {
        let (emotion, motion, line) = if let Some(q) = &s.user_query {
            if q.contains("물약") || q.contains("인벤") || q.contains("포션") {
                (
                    "FOCUSED",
                    "THINK",
                    "인벤토리는 I 키로 열 수 있어. 물약은 보통 소모품 칸에 있어!".to_string(),
                )
            } else {
                (
                    "FOCUSED",
                    "NOD",
                    "패턴 시작 모션이 보이면 옆으로 구르고, 끝나면 바로 반격해!".to_string(),
                )
            }
        } else if s.hp_ratio.is_some_and(|h| h < 0.3) {
            (
                "SURPRISED_URGENT",
                "LEAN_FORWARD_POINT",
                "체력이 위험해! 당장 회복하고 거리 벌려!".to_string(),
            )
        } else {
            (
                "SURPRISED_URGENT",
                "LEAN_FORWARD_POINT",
                format!("{} 조심해!", s.event_label.clone().unwrap_or_else(|| "적".into())),
            )
        };
        let gaze = s
            .polar
            .map(|p| format!("{{\"angle_deg\": {:.0}, \"distance_norm\": {:.2}}}", p.theta_deg, p.r))
            .unwrap_or_else(|| "{\"angle_deg\": 0, \"distance_norm\": 0.0}".into());
        format!(
            "<tool_call>\n<function=control_zena_avatar>\n<parameter=gaze_target>\n{gaze}\n</parameter>\n<parameter=emotion_state>\n{emotion}\n</parameter>\n<parameter=body_motion>\n{motion}\n</parameter>\n<parameter=speech_text>\n{line}\n</parameter>\n</function>\n</tool_call>"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zena_core::tools::{parse_model_output, ToolCall};

    #[test]
    fn stub_output_parses_with_the_real_parser() {
        let s = Situation {
            event_label: Some("보스".into()),
            polar: Some(zena_core::Polar { r: 0.1, theta_deg: 30.0 }),
            ..Default::default()
        };
        let out = TemplateBackend.generate("", &s);
        let p = parse_model_output(&out);
        assert!(matches!(p.calls.as_slice(), [ToolCall::ControlZenaAvatar(_)]), "{:?}", p.rejected);
    }
}
