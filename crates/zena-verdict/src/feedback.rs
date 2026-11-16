//! A spoken recommendation becomes an observation. The rules are deterministic on purpose: the observation
//! comes from the dictionary match and Stanza's negation scope (NLP-NMS v2), never from the model, so a
//! hallucinated "vote" cannot pollute a verdict.
//!
//! Unlike a command (NLP-NMS holds a negated command instead of running it), negated feedback flips its sign:
//!
//! | utterance | intent | negated | observation |
//! |---|---|---|---|
//! | "그 공략 추천" | recommend | no | +1 |
//! | "그거 추천 안 해" | recommend | yes | −1 |
//! | "별로였어" | not_recommend | no | −1 |
//! | "별로지 않았어" · "나쁘지 않았어" | not_recommend | yes | +1, half weight |

use crate::{Obs, Params};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeedbackIntent {
    /// `feedback.recommend` ("추천", "좋았어", "덕분에 깼다", "통했어")
    Recommend,
    /// `feedback.not_recommend` ("비추", "별로", "안 통해", "틀렸어")
    NotRecommend,
}

/// The observation for one piece of feedback about a record (`credit` from the target resolution, 0..1).
pub fn feedback_obs(intent: FeedbackIntent, negated: bool, credit: f64, at_ms: i64, params: &Params) -> Obs {
    let base = params.weight("feedback");
    let (sign, weight) = match (intent, negated) {
        (FeedbackIntent::Recommend, false) => (1, base),
        (FeedbackIntent::Recommend, true) => (-1, base),
        (FeedbackIntent::NotRecommend, false) => (-1, base),
        // A double negative ("not bad") is a weak recommendation.
        (FeedbackIntent::NotRecommend, true) => (1, base * 0.5),
    };
    Obs {
        kind: "feedback".into(),
        sign,
        weight,
        credit: credit.clamp(0.0, 1.0),
        at_ms,
    }
}

/// A manual ▲ / ▼ correction in the history view: the strongest local observation.
pub fn manual_obs(up: bool, at_ms: i64, params: &Params) -> Obs {
    Obs {
        kind: "manual".into(),
        sign: if up { 1 } else { -1 },
        weight: params.weight("manual"),
        credit: 1.0,
        at_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negation_flips_feedback_and_a_double_negative_is_weak() {
        let p = Params::v1();
        let o = |i, n| {
            let x = feedback_obs(i, n, 0.8, 7, &p);
            (x.sign, x.weight)
        };
        assert_eq!(o(FeedbackIntent::Recommend, false), (1, 1.0));
        assert_eq!(o(FeedbackIntent::Recommend, true), (-1, 1.0));
        assert_eq!(o(FeedbackIntent::NotRecommend, false), (-1, 1.0));
        assert_eq!(o(FeedbackIntent::NotRecommend, true), (1, 0.5));
        let x = feedback_obs(FeedbackIntent::Recommend, false, 1.7, 7, &p);
        assert_eq!((x.kind.as_str(), x.credit, x.at_ms), ("feedback", 1.0, 7), "credit is clamped");
        let m = manual_obs(false, 9, &p);
        assert_eq!((m.sign, m.weight, m.kind.as_str()), (-1, 2.0, "manual"));
    }
}
