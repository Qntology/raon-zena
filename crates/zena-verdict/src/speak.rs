//! What Zena may say about a recommendation: the local verdict (my own record) and the global one (other
//! users, premium-discount's server) side by side. They are never mixed into one number: "it worked for me but
//! the others say it does not" is the most useful thing to say, and a blend would hide it.
//!
//! | local | global | Zena says |
//! |---|---|---|
//! | judged | same tier | both agree |
//! | judged | other tier | both, explicitly |
//! | judged | unrated | the local verdict only |
//! | unrated | judged | "the others …" |
//! | unrated | unrated | facts only: never "recommended" / "not recommended" |

use crate::Tier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stance {
    Agree(Tier),
    Differ {
        local: Tier,
        global: Tier,
    },
    LocalOnly(Tier),
    GlobalOnly(Tier),
    /// Neither side has enough evidence.
    FactsOnly,
}

fn judged(t: Option<Tier>) -> Option<Tier> {
    t.filter(|t| *t != Tier::Unrated)
}

pub fn stance(local: Option<Tier>, global: Option<Tier>) -> Stance {
    match (judged(local), judged(global)) {
        (Some(l), Some(g)) if l == g => Stance::Agree(l),
        (Some(l), Some(g)) => Stance::Differ { local: l, global: g },
        (Some(l), None) => Stance::LocalOnly(l),
        (None, Some(g)) => Stance::GlobalOnly(g),
        (None, None) => Stance::FactsOnly,
    }
}

/// 추천 / 비추천 / 보통 (판정 전은 말하지 않으므로 빈 문자열)
pub fn word(t: Tier) -> &'static str {
    match t {
        Tier::Premium => "추천",
        Tier::Discount => "비추천",
        Tier::Neutral => "보통",
        Tier::Unrated => "",
    }
}

impl Stance {
    /// One context line for the model (Korean, as the rest of Zena's prompt), about the record `label`.
    pub fn line(self, label: &str) -> String {
        match self {
            Stance::Agree(t) => format!("{label}: 내 기록과 서버 평가 모두 {}", word(t)),
            Stance::Differ { local, global } => {
                format!("{label}: 내 기록에선 {}인데 서버에선 {} (둘 다 말할 것)", word(local), word(global))
            }
            Stance::LocalOnly(t) => format!("{label}: 내 기록으로는 {} (서버 판정 전)", word(t)),
            Stance::GlobalOnly(t) => format!("{label}: 다른 사용자들은 {} (내 기록은 판정 전)", word(t)),
            Stance::FactsOnly => format!("{label}: 판정 전 (추천·비추천이라고 말하지 말고 있었던 일만)"),
        }
    }

    /// The character emote the briefing opens with (`zena_core::tools::CharacterEmote` names).
    pub fn emote(self) -> Option<&'static str> {
        match self {
            Stance::Differ { .. } => Some("QUESTION"),
            Stance::Agree(t) | Stance::LocalOnly(t) | Stance::GlobalOnly(t) => match t {
                Tier::Premium => Some("STAR_EYES"),
                Tier::Discount => Some("SWEAT_PANIC"),
                _ => None,
            },
            Stance::FactsOnly => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_five_rows_of_the_table() {
        use Tier::*;
        assert_eq!(stance(Some(Premium), Some(Premium)), Stance::Agree(Premium));
        assert_eq!(
            stance(Some(Premium), Some(Discount)),
            Stance::Differ {
                local: Premium,
                global: Discount
            }
        );
        assert_eq!(stance(Some(Neutral), Some(Unrated)), Stance::LocalOnly(Neutral));
        assert_eq!(stance(None, Some(Discount)), Stance::GlobalOnly(Discount));
        assert_eq!(stance(Some(Unrated), None), Stance::FactsOnly);

        assert_eq!(
            stance(Some(Premium), Some(Premium)).line("빙결 공략"),
            "빙결 공략: 내 기록과 서버 평가 모두 추천"
        );
        assert!(stance(Some(Premium), Some(Discount)).line("x").contains("둘 다"));
        let facts = Stance::FactsOnly.line("구르기 공략");
        assert!(!facts.contains("추천이야") && facts.contains("판정 전"), "{facts}");

        assert_eq!(Stance::Agree(Premium).emote(), Some("STAR_EYES"));
        assert_eq!(Stance::GlobalOnly(Discount).emote(), Some("SWEAT_PANIC"));
        assert_eq!(
            Stance::Differ {
                local: Premium,
                global: Neutral
            }
            .emote(),
            Some("QUESTION")
        );
        assert_eq!(Stance::LocalOnly(Neutral).emote(), None);
    }
}
