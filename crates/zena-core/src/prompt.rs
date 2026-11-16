//! ChatML prompt builder for Raon-1-nano's non-thinking fast path.
//!
//! The released chat template opens the assistant turn with `<think>\n` unless
//! `enable_thinking=false`, in which case it emits an empty `<think></think>\n\n` block.
//! Pre-filling that empty block skips reasoning tokens entirely (Lv3 fast path).
//!
//! Multi-turn memory and the session summary go into the **user** turn ([`Situation::history`],
//! [`Situation::session`]), never into the system prompt: the backend prefills the system + tools prefix once and
//! clones its state per request, so that prefix must stay byte-identical for every request.

use crate::geometry::{FlowDirection, Polar};
use crate::tools::tool_definitions;
use std::collections::VecDeque;

/// One exchange remembered for the next prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DialogueTurn {
    /// What the player said (`None`: Zena spoke on her own).
    pub user: Option<String>,
    pub zena: String,
}

/// Sliding window of the last few exchanges.
#[derive(Debug, Clone)]
pub struct Dialogue {
    turns: VecDeque<DialogueTurn>,
    pub capacity: usize,
    /// Each remembered utterance is cut to this many characters.
    pub max_chars: usize,
}

impl Default for Dialogue {
    fn default() -> Self {
        Self::new(4)
    }
}

fn clip(s: &str, n: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

impl Dialogue {
    pub fn new(capacity: usize) -> Self {
        Self {
            turns: VecDeque::new(),
            capacity,
            max_chars: 60,
        }
    }

    pub fn push(&mut self, user: Option<&str>, zena: &str) {
        if zena.trim().is_empty() && user.is_none_or(|u| u.trim().is_empty()) {
            return;
        }
        self.turns.push_back(DialogueTurn {
            user: user.map(|u| clip(u, self.max_chars)).filter(|u| !u.is_empty()),
            zena: clip(zena, self.max_chars),
        });
        while self.turns.len() > self.capacity {
            self.turns.pop_front();
        }
    }

    pub fn turns(&self) -> Vec<DialogueTurn> {
        self.turns.iter().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }
}

/// What Zena knows about the play session, rendered as one compact block so her tone follows it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionContext {
    /// What the screen shows (the scene's description) and for how long.
    pub scene: Option<String>,
    pub scene_secs: u64,
    /// Defeats / victories in a row (screen events and what the player said).
    pub defeat_streak: u32,
    pub victory_streak: u32,
    /// 0 (stranger) .. 1 (close): praise, thanks and conversation raise it.
    pub affinity: f32,
    /// 0 (calm) .. 1 (frantic): combat and defeats raise it, it decays over time.
    pub tension: f32,
}

impl SessionContext {
    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if let Some(s) = &self.scene {
            let m = self.scene_secs / 60;
            parts.push(if m > 0 {
                format!("화면: {s} ({m}분째)")
            } else {
                format!("화면: {s}")
            });
        }
        if self.defeat_streak > 0 {
            parts.push(format!("연속 패배 {}회", self.defeat_streak));
        }
        if self.victory_streak > 0 {
            parts.push(format!("연속 승리 {}회", self.victory_streak));
        }
        parts.push(
            match self.affinity {
                a if a >= 0.66 => "친밀도 높음",
                a if a >= 0.33 => "친밀도 보통",
                _ => "친밀도 낮음",
            }
            .to_string(),
        );
        if self.tension >= 0.5 {
            parts.push("긴장도 높음".into());
        }
        let mut s = format!("[세션] {}\n", parts.join(" / "));
        if let Some(t) = self.tone() {
            s.push_str(&format!("[말투] {t}\n"));
        }
        s
    }

    /// The tone hint the summary implies.
    pub fn tone(&self) -> Option<&'static str> {
        if self.defeat_streak >= 3 {
            Some("계속 지고 있다. 다그치지 말고 먼저 위로하고, 쉬어 가자고 제안한다.")
        } else if self.tension >= 0.7 {
            Some("급박하다. 아주 짧고 단호하게 말한다.")
        } else if self.victory_streak >= 2 {
            Some("연승 중이다. 신나게 칭찬한다.")
        } else if self.affinity >= 0.66 {
            Some("친한 사이다. 편한 반말로 장난스럽게 말한다.")
        } else {
            None
        }
    }
}

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
    /// The last few exchanges, oldest first.
    pub history: Vec<DialogueTurn>,
    pub session: Option<SessionContext>,
}

fn direction_ko(theta: f32) -> &'static str {
    const NAMES: [&str; 8] = ["오른쪽", "오른쪽 위", "위", "왼쪽 위", "왼쪽", "왼쪽 아래", "아래", "오른쪽 아래"];
    NAMES[(((theta + 22.5) % 360.0) / 45.0) as usize % 8]
}

impl Situation {
    pub fn render(&self) -> String {
        let mut s = String::new();
        if !self.history.is_empty() {
            s.push_str("[최근 대화]\n");
            for t in &self.history {
                if let Some(u) = &t.user {
                    s.push_str(&format!("플레이어: {u}\n"));
                }
                s.push_str(&format!("제나: {}\n", t.zena));
            }
        }
        if let Some(c) = &self.session {
            s.push_str(&c.render());
        }
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

    #[test]
    fn history_and_session_go_into_the_user_turn_so_the_cached_prefix_never_changes() {
        let user_turn = "<|im_start|>user\n";
        let bare = build_fast_path_prompt(&Situation::default());
        let mut d = Dialogue::new(2);
        d.push(Some("보스 패턴 알려줘"), "내려찍기 전에 옆으로 굴러!");
        d.push(Some("고마워"), "천만에!");
        d.push(None, "슬슬 스트레칭 한 번 할까?");
        assert_eq!(d.len(), 2, "the window keeps the last two");
        let s = Situation {
            user_query: Some("또 졌어".into()),
            history: d.turns(),
            session: Some(SessionContext {
                scene: Some("게임 전투".into()),
                scene_secs: 754,
                defeat_streak: 3,
                affinity: 0.7,
                ..Default::default()
            }),
            ..Default::default()
        };
        let p = build_fast_path_prompt(&s);
        let cut = |p: &str| p[..p.find(user_turn).unwrap()].to_string();
        assert_eq!(cut(&p), cut(&bare), "system + tools prefix is byte-identical");
        let user = &p[p.find(user_turn).unwrap()..];
        assert!(user.contains("[최근 대화]\n플레이어: 고마워\n제나: 천만에!\n제나: 슬슬 스트레칭 한 번 할까?\n"));
        assert!(!user.contains("보스 패턴"), "dropped from the window");
        assert!(user.contains("[세션] 화면: 게임 전투 (12분째) / 연속 패배 3회 / 친밀도 높음"));
        assert!(user.contains("[말투] 계속 지고 있다"));
        assert!(user.contains("[플레이어 질문] 또 졌어"));
    }

    #[test]
    fn dialogue_clips_long_lines() {
        let mut d = Dialogue::new(4);
        d.push(Some(&"가".repeat(100)), "응");
        assert_eq!(d.turns()[0].user.as_ref().unwrap().chars().count(), 61);
        d.push(None, "  ");
        assert_eq!(d.len(), 1, "nothing said, nothing remembered");
    }
}
