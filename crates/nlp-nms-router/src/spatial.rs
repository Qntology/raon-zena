//! Spatial-text NMS: pick the screen region a natural-language request refers to.
//!
//! * Confidence of each candidate box = cosine similarity between the request embedding and
//!   the box's feature (a Raon-VisionEncoder ROI embedding, or the embedding of its label).
//! * Greedy suppression in descending confidence, with logis-center's crop rules: a lower box
//!   is a duplicate when it is identical, has IoU ≥ 0.90 ("twin"), or has IoU ≥ 0.85 and is
//!   covered by the higher one ("snap"); same-label boxes additionally merge at IoU ≥ `iou_same`.
//! * [`exclusive_assign`] ports `exclusive_assign_by_score`: one-to-one query↔box assignment by
//!   own score, then margin over the best qualifying rival.

use crate::embed::cosine;
use serde::Serialize;
use zena_core::geometry::{BBox, Polar};

pub const SNAP_IOU: f32 = 0.85;
pub const TWIN_IOU: f32 = 0.90;

#[derive(Debug, Clone, Serialize)]
pub struct BoxCand {
    pub bbox: BBox,
    pub label: String,
    #[serde(skip)]
    pub feature: Vec<f32>,
    pub polar: Polar,
    pub confidence: f32,
}

fn covers(a: &BBox, b: &BBox) -> bool {
    a.x0 <= b.x0 && a.y0 <= b.y0 && a.x1 >= b.x1 && a.y1 >= b.y1
}

/// Set each candidate's confidence from its similarity to `query`.
pub fn score_by_text(cands: &mut [BoxCand], query: &[f32]) {
    for c in cands {
        c.confidence = cosine(&c.feature, query);
    }
}

/// Indices of the surviving boxes, best first, and `(suppressed, by)` pairs.
pub fn greedy_nms(cands: &[BoxCand], iou_same: f32) -> (Vec<usize>, Vec<(usize, usize)>) {
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| cands[b].confidence.total_cmp(&cands[a].confidence));
    let mut kept: Vec<usize> = Vec::new();
    let mut sup = Vec::new();
    for i in order {
        let c = &cands[i];
        let hit = kept.iter().copied().find(|&k| {
            let o = &cands[k];
            let iou = o.bbox.iou(&c.bbox);
            let same = o.bbox == c.bbox;
            same || iou >= TWIN_IOU || (iou >= SNAP_IOU && covers(&o.bbox, &c.bbox)) || (o.label == c.label && iou >= iou_same)
        });
        match hit {
            Some(k) => sup.push((i, k)),
            None => kept.push(i),
        }
    }
    (kept, sup)
}

/// One-to-one assignment of rows (queries) to columns (boxes). A claim needs `own ≥ abs` and a
/// margin over the best *qualifying* rival row (`≥ abs`) of at least `margin`.
pub fn exclusive_assign(matrix: &[Vec<f32>], abs: f32, margin: f32) -> Vec<Option<(usize, f32, f32)>> {
    let rows = matrix.len();
    let cols = matrix.iter().map(Vec::len).max().unwrap_or(0);
    let get = |r: usize, c: usize| matrix.get(r).and_then(|x| x.get(c)).copied().unwrap_or(-1.0);
    let mut claims = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let own = get(r, c);
            if own < abs {
                continue;
            }
            let rival = (0..rows)
                .filter(|&o| o != r)
                .map(|o| get(o, c))
                .filter(|&s| s >= abs)
                .fold(f32::MIN, f32::max);
            let rival = if rival == f32::MIN { abs } else { rival };
            let m = own - rival;
            if m >= margin {
                claims.push((r, c, own, m));
            }
        }
    }
    claims.sort_by(|a, b| b.2.total_cmp(&a.2).then(b.3.total_cmp(&a.3)));
    let mut out = vec![None; rows];
    let mut taken = vec![false; cols];
    for (r, c, own, m) in claims {
        if out[r].is_none() && !taken[c] {
            out[r] = Some((c, own, m));
            taken[c] = true;
        }
    }
    out
}

/// Score by `query`, suppress duplicates, and return the single best target (index, polar).
pub fn select_target(cands: &mut [BoxCand], query: &[f32], min_conf: f32) -> Option<(usize, Polar)> {
    score_by_text(cands, query);
    let (kept, _) = greedy_nms(cands, 0.5);
    kept.into_iter()
        .find(|&i| cands[i].confidence >= min_conf)
        .map(|i| (i, cands[i].polar))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{HashedNgramEmbedder, TextEmbedder};

    fn cand(e: &HashedNgramEmbedder, label: &str, b: (f32, f32, f32, f32), theta: f32) -> BoxCand {
        BoxCand {
            bbox: BBox::new(b.0, b.1, b.2, b.3),
            label: label.into(),
            feature: e.embed(label),
            polar: Polar { r: 0.3, theta_deg: theta },
            confidence: 0.0,
        }
    }

    #[test]
    fn picks_the_box_the_request_names() {
        let e = HashedNgramEmbedder::default();
        let mut c = vec![
            cand(&e, "슬라임 몬스터", (0.0, 0.0, 50.0, 50.0), 10.0),
            cand(&e, "보스 몬스터", (100.0, 100.0, 200.0, 200.0), 45.0),
            cand(&e, "보스 몬스터", (102.0, 101.0, 201.0, 199.0), 46.0), // duplicate detection
            cand(&e, "체력 바", (0.0, 400.0, 300.0, 420.0), 250.0),
        ];
        let q = e.embed("저 보스 어디 있어?");
        let (i, p) = select_target(&mut c, &q, 0.2).unwrap();
        assert_eq!(c[i].label, "보스 몬스터");
        assert!((p.theta_deg - 45.0).abs() < 1.5);
        let (kept, sup) = greedy_nms(&c, 0.5);
        assert_eq!(kept.len(), 3);
        assert_eq!(sup.len(), 1);
    }

    #[test]
    fn exclusive_assignment_is_one_to_one() {
        // Two queries both like box 0; query 1 likes it more, so query 0 gets box 1.
        let m = vec![vec![0.7, 0.6], vec![0.9, 0.1]];
        let a = exclusive_assign(&m, 0.3, 0.0);
        assert_eq!(a[1].map(|x| x.0), Some(0));
        assert_eq!(a[0].map(|x| x.0), Some(1));
    }
}
