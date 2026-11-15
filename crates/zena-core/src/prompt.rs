//! ChatML prompt builder for Raon-1-nano's non-thinking fast path.
//!
//! The released chat template opens the assistant turn with `<think>\n` unless
//! `enable_thinking=false`, in which case it emits an empty `<think></think>\n\n` block.
//! Pre-filling that empty block skips reasoning tokens entirely (Lv3 fast path).

use crate::geometry::{FlowDirection, Polar};
use crate::tools::tool_definitions;

pub const SYSTEM_PERSONA: &str = "너는 게임 화면 옆에 떠 있는 버튜버 컴패니언 '제나(Zena)'다. \
플레이어를 짧고 다정하게 코칭한다. 대사는 한 문장, 90자 이내. \
반드시 control_zena_avatar 도구로 시선·표정·동작·대사를 함께 지정하고, \
창/앱 제어 요청일 때만 manage_zena_system 을 호출한다.";

/// Compact, model-readable description of the situation for one Lv3 call.
#[derive(Debug, Clone, Default)]
pub struct Situation {
    pub event_label: Option<String>,
    pub polar: Option<Polar>,
    pub flow: Option<FlowDirection>,
    pub hp_ratio: Option<f32>,
    pub user_query: Option<String>,
    /// Retrieved memory / game-guide snippets (already NMS-filtered).
    pub context: Vec<String>,
}

fn direction_ko(theta: f32) -> &'static str {
    const NAMES: [&str; 8] = ["오른쪽", "오른쪽 위", "위", "왼쪽 위", "왼쪽", "왼쪽 아래", "아래", "오른쪽 아래"];
    NAMES[(((theta + 22.5) % 360.0) / 45.0) as usize % 8]
}

impl Situation {
    pub fn render(&self) -> String {
        let mut s = String::new();
        if let Some(q) = &self.user_query {
            s.push_str(&format!("[플레이어 질문] {q}\n"));
        }
        if let Some(l) = &self.event_label {
            s.push_str(&format!("[화면 이벤트] {l}"));
            if let Some(p) = self.polar {
                let dist = if p.r < 0.22 {
                    "바로 코앞"
                } else if p.r < 0.5 {
                    "중거리"
                } else {
                    "멀리"
                };
                s.push_str(&format!(
                    " / {} {} (angle_deg={:.0}, distance_norm={:.2})",
                    direction_ko(p.theta_deg),
                    dist,
                    p.theta_deg,
                    p.r
                ));
            }
            if let Some(f) = self.flow {
                s.push_str(match f {
                    FlowDirection::Inward => " / 접근 중",
                    FlowDirection::Outward => " / 플레이어 공격",
                    _ => "",
                });
            }
            s.push('\n');
        }
        if let Some(hp) = self.hp_ratio {
            s.push_str(&format!("[HP] {:.0}%\n", hp * 100.0));
        }
        for c in &self.context {
            s.push_str(&format!("[참고] {c}\n"));
        }
        s
    }
}

/// Full prompt string in Raon-1-nano's chat format, ending with the pre-filled empty think block.
pub fn build_fast_path_prompt(sit: &Situation) -> String {
    let mut p = String::from("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for t in tool_definitions().as_array().expect("array") {
        p.push('\n');
        p.push_str(&t.to_string());
    }
    p.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n");
    p.push_str(SYSTEM_PERSONA);
    p.push_str("<|im_end|>\n<|im_start|>user\n");
    p.push_str(sit.render().trim_end());
    p.push_str("<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n");
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_has_fast_path_prefill_and_tools() {
        let s = Situation {
            event_label: Some("보스 등장".into()),
            polar: Some(Polar { r: 0.1, theta_deg: 10.0 }),
            flow: Some(FlowDirection::Inward),
            hp_ratio: Some(0.12),
            ..Default::default()
        };
        let p = build_fast_path_prompt(&s);
        assert!(p.ends_with("<|im_start|>assistant\n<think></think>\n\n"));
        assert!(p.contains("control_zena_avatar"));
        assert!(p.contains("오른쪽 바로 코앞"));
        assert!(p.contains("[HP] 12%"));
        assert_eq!(direction_ko(350.0), "오른쪽");
        assert_eq!(direction_ko(135.0), "왼쪽 위");
    }
}
