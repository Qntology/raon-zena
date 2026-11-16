//! Mergeable statistics (logis "6-1 records"). Every type merges entry by entry with one rule
//! (`merge` / `merge_from`), so a base ledger and a run can be combined per queried entry or per whole scope with
//! the same result. Rings keep the most recent values; `ring_len == 0` means unbounded.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::{TIE_SPECIFICITY_SLACK, TIE_WIN_RATE};

/// Running mean / variance (Welford) plus a ring of the latest values for drift.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(default)]
    pub ring: Vec<f64>,
}

impl Welford {
    /// Adds one value; non-finite values are ignored.
    pub fn push(&mut self, x: f64, ring_len: usize) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / (self.n as f64);
        self.m2 += d * (x - self.mean);
        self.ring.push(x);
        self.trim_ring(ring_len);
    }

    /// Chan et al. parallel merge; `other`'s ring is appended after ours (it is the newer side).
    pub fn merge(&mut self, other: &Welford, ring_len: usize) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean;
            self.m2 = other.m2;
            self.ring = other.ring.clone();
        } else {
            let na = self.n as f64;
            let nb = other.n as f64;
            let n = na + nb;
            let delta = other.mean - self.mean;
            self.mean += delta * (nb / n);
            self.m2 += other.m2 + delta * delta * (na * nb / n);
            self.n += other.n;
            self.ring.extend(other.ring.iter().copied());
        }
        self.trim_ring(ring_len);
    }

    /// Drops the oldest ring values beyond `ring_len` (no-op when `ring_len == 0`).
    pub fn trim_ring(&mut self, ring_len: usize) {
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }

    /// Sample variance (n − 1); 0 below two samples.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / ((self.n - 1) as f64)
        }
    }

    pub fn sd(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Mean of the ring; the overall mean when the ring is empty.
    pub fn recent_mean(&self) -> f64 {
        if self.ring.is_empty() {
            return self.mean;
        }
        self.ring.iter().sum::<f64>() / (self.ring.len() as f64)
    }

    /// `(recent mean − mean) / σ`: how far the latest values moved; 0 without spread or ring.
    pub fn drift_z(&self) -> f64 {
        let sd = self.sd();
        if sd <= 0.0 || self.ring.is_empty() {
            return 0.0;
        }
        (self.recent_mean() - self.mean) / sd
    }
}

/// Shape of one ranked score list: decisive (one clear winner) or flat.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecayShape {
    /// Finite scores used.
    pub n: usize,
    pub top1: f64,
    pub top2: f64,
    /// `top1 − top2`.
    pub margin: f64,
    /// Margin over the distance from top1 to the tail median, in [0, 1].
    pub top_gap_ratio: f64,
    /// Tail (2nd and below) σ over the full range, in [0, 1]; 1 when every score is equal.
    pub tail_flatness: f64,
    /// Softmax entropy (scaled by the tail σ) over ln n, in [0, 1]; 1 when the tail has no spread.
    pub entropy_norm: f64,
    /// Share of scores above 0.
    pub positive_ratio: f64,
}

/// Compresses a score list into its [`DecayShape`]; `None` below two finite scores.
pub fn decay_shape(scores: &[f32]) -> Option<DecayShape> {
    let mut v: Vec<f64> = scores.iter().filter(|s| s.is_finite()).map(|s| *s as f64).collect();
    if v.len() < 2 {
        return None;
    }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let top1 = v[0];
    let top2 = v[1];
    let margin = top1 - top2;
    let tail = &v[1..];
    let tail_median = tail[tail.len() / 2];
    let span = (top1 - tail_median).abs();
    let top_gap_ratio = if span > 1e-9 { (margin / span).clamp(0.0, 1.0) } else { 0.0 };
    let tail_mean = tail.iter().sum::<f64>() / (tail.len() as f64);
    let tail_var = tail.iter().map(|x| (x - tail_mean).powi(2)).sum::<f64>() / (tail.len() as f64);
    let full_range = (v[0] - v[n - 1]).abs();
    let tail_flatness = if full_range > 1e-9 {
        (tail_var.sqrt() / full_range).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let entropy_norm = {
        let scale = tail_var.sqrt();
        if scale <= 1e-9 {
            1.0
        } else {
            let mx = v[0];
            let exps: Vec<f64> = v.iter().map(|x| ((x - mx) / scale).exp()).collect();
            let sum = exps.iter().sum::<f64>().max(1e-12);
            let mut h = 0.0f64;
            for e in &exps {
                let p = e / sum;
                if p > 1e-12 {
                    h -= p * p.ln();
                }
            }
            let hmax = (n as f64).ln();
            if hmax > 1e-9 {
                (h / hmax).clamp(0.0, 1.0)
            } else {
                0.0
            }
        }
    };
    let positive_ratio = v.iter().filter(|x| **x > 0.0).count() as f64 / (n as f64);
    Some(DecayShape {
        n,
        top1,
        top2,
        margin,
        top_gap_ratio,
        tail_flatness,
        entropy_norm,
        positive_ratio,
    })
}

/// Distribution of [`DecayShape`]s on one axis: the scope's usual margin and flatness.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DecayStat {
    pub margin: Welford,
    pub top_gap_ratio: Welford,
    pub tail_flatness: Welford,
    pub entropy_norm: Welford,
    pub positive_ratio: Welford,
}

impl DecayStat {
    pub fn push(&mut self, shape: &DecayShape, ring: usize) {
        self.margin.push(shape.margin, ring);
        self.top_gap_ratio.push(shape.top_gap_ratio, ring);
        self.tail_flatness.push(shape.tail_flatness, ring);
        self.entropy_norm.push(shape.entropy_norm, ring);
        self.positive_ratio.push(shape.positive_ratio, ring);
    }

    pub fn merge_from(&mut self, v: &DecayStat, ring: usize) {
        self.margin.merge(&v.margin, ring);
        self.top_gap_ratio.merge(&v.top_gap_ratio, ring);
        self.tail_flatness.merge(&v.tail_flatness, ring);
        self.entropy_norm.merge(&v.entropy_norm, ring);
        self.positive_ratio.merge(&v.positive_ratio, ring);
    }

    fn trim_rings(&mut self, ring: usize) {
        for w in [
            &mut self.margin,
            &mut self.top_gap_ratio,
            &mut self.tail_flatness,
            &mut self.entropy_norm,
            &mut self.positive_ratio,
        ] {
            w.trim_ring(ring);
        }
    }
}

/// One unordered pair `a|b` (a ≤ b): how often the two met closely and who won. `ties` counts every meeting,
/// decided or not, so undecided ties lower both win rates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfusionStat {
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub margin: Welford,
}

impl ConfusionStat {
    pub fn merge_from(&mut self, v: &ConfusionStat, ring: usize) {
        self.ties += v.ties;
        self.a_wins += v.a_wins;
        self.b_wins += v.b_wins;
        self.margin.merge(&v.margin, ring);
    }
}

/// How often an item was considered and how often (and why) it was rejected; replaces logis `FieldRejectStat`
/// with free-form reject kinds (`demoted`, `format`, `outcome_failed`, …).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectStat {
    pub seen: u64,
    pub rejected: u64,
    #[serde(default)]
    pub kinds: BTreeMap<String, u64>,
}

impl RejectStat {
    pub fn merge_from(&mut self, v: &RejectStat, _ring: usize) {
        self.seen += v.seen;
        self.rejected += v.rejected;
        for (k, n) in &v.kinds {
            *self.kinds.entry(k.clone()).or_insert(0) += n;
        }
    }

    /// Rejected share `rejected / max(seen, rejected, 1)` in [0, 1] (logis `learned_specificity`).
    pub fn specificity(&self) -> f32 {
        let denom = self.seen.max(self.rejected).max(1) as f64;
        (self.rejected as f64 / denom) as f32
    }
}

/// Everything recorded under one scope key.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScopeStat {
    /// Per-axis baselines (`valence`, `credit.margin`, `align.cos`, …).
    #[serde(default)]
    pub baseline: HashMap<String, Welford>,
    /// Per-axis decay shapes.
    #[serde(default)]
    pub decay: HashMap<String, DecayStat>,
    /// Pair confusion ledger keyed `a|b` with a ≤ b.
    #[serde(default)]
    pub confusion: HashMap<String, ConfusionStat>,
    /// Per-item reject counts.
    #[serde(default)]
    pub reject: HashMap<String, RejectStat>,
    /// Transition counts keyed `from>to`.
    #[serde(default)]
    pub transition: HashMap<String, u64>,
    /// Last observation time (Unix ms).
    #[serde(default)]
    pub updated_ms: i64,
}

impl ScopeStat {
    /// Merges `other` into this scope entry by entry (same rules as the per-entry reads).
    pub fn absorb(&mut self, other: ScopeStat, ring: usize) {
        for (k, v) in other.baseline {
            self.baseline.entry(k).or_default().merge(&v, ring);
        }
        for (k, v) in other.decay {
            self.decay.entry(k).or_default().merge_from(&v, ring);
        }
        for (k, v) in other.confusion {
            self.confusion.entry(k).or_default().merge_from(&v, ring);
        }
        for (k, v) in other.reject {
            self.reject.entry(k).or_default().merge_from(&v, ring);
        }
        for (k, v) in other.transition {
            *self.transition.entry(k).or_insert(0) += v;
        }
        self.updated_ms = self.updated_ms.max(other.updated_ms);
    }

    pub fn is_empty(&self) -> bool {
        self.baseline.is_empty()
            && self.decay.is_empty()
            && self.confusion.is_empty()
            && self.reject.is_empty()
            && self.transition.is_empty()
    }

    /// Observation count used to rank scopes when a ledger must shed some.
    pub fn observations(&self) -> u64 {
        self.baseline.values().map(|w| w.n).sum::<u64>()
            + self.decay.values().map(|d| d.margin.n).sum::<u64>()
            + self.confusion.values().map(|c| c.ties).sum::<u64>()
            + self.reject.values().map(|r| r.seen.max(r.rejected)).sum::<u64>()
            + self.transition.values().sum::<u64>()
    }

    /// Re-applies a (changed) ring length to every ring in this scope.
    pub fn trim_rings(&mut self, ring: usize) {
        for w in self.baseline.values_mut() {
            w.trim_ring(ring);
        }
        for d in self.decay.values_mut() {
            d.trim_rings(ring);
        }
        for c in self.confusion.values_mut() {
            c.margin.trim_ring(ring);
        }
    }
}

/// Pure tie decision (logis `tie_verdict`): break the tie toward this item only if history names it the winner at
/// [`TIE_WIN_RATE`] or more and — when both specificities are known — it is not rejected more than
/// [`TIE_SPECIFICITY_SLACK`] more often than its rival. Unknown specificity (too few samples) checks the rate alone.
pub fn tie_verdict(history: Option<(bool, f32)>, spec_self: Option<f32>, spec_rival: Option<f32>) -> bool {
    let Some((self_wins, rate)) = history else { return false };
    if !self_wins || rate.is_nan() || rate < TIE_WIN_RATE {
        return false;
    }
    match (spec_self, spec_rival) {
        (Some(a), Some(b)) => a <= b + TIE_SPECIFICITY_SLACK,
        _ => true,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Deterministic values in [-5, 5) (Numerical Recipes LCG).
    pub(crate) fn lcg(seed: u64, n: usize) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 10.0 - 5.0
            })
            .collect()
    }

    fn two_pass(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var)
    }

    fn welford(xs: &[f64], ring: usize) -> Welford {
        let mut w = Welford::default();
        for x in xs {
            w.push(*x, ring);
        }
        w
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn welford_push_and_merge_match_two_pass() {
        let xs = lcg(42, 1000);
        let (mean, var) = two_pass(&xs);
        let w = welford(&xs, 0);
        assert_eq!(w.n, 1000);
        assert!(
            close(w.mean, mean) && close(w.variance(), var),
            "{} {} vs {} {}",
            w.mean,
            w.variance(),
            mean,
            var
        );

        for split in [1, 7, 500, 999] {
            let mut a = welford(&xs[..split], 0);
            a.merge(&welford(&xs[split..], 0), 0);
            assert_eq!(a.n, 1000);
            assert!(close(a.mean, mean) && close(a.variance(), var), "split {split}");
            assert_eq!(a.ring, xs, "ring keeps order across the merge");
        }
        let mut empty = Welford::default();
        empty.merge(&w, 0);
        assert_eq!(empty, w);
        let mut same = w.clone();
        same.merge(&Welford::default(), 0);
        assert_eq!(same, w);
        assert!(!Welford {
            n: 1,
            ..Default::default()
        }
        .variance()
        .is_nan());
    }

    #[test]
    fn ring_trims_to_the_latest() {
        let mut w = welford(&[1.0, 2.0, 3.0, 4.0, 5.0], 3);
        assert_eq!(w.ring, vec![3.0, 4.0, 5.0]);
        assert_eq!(w.n, 5);
        w.push(f64::NAN, 3);
        assert_eq!(w.n, 5, "non-finite ignored");
        w.merge(&welford(&[6.0, 7.0], 3), 3);
        assert_eq!(w.ring, vec![5.0, 6.0, 7.0]);
        let unbounded = welford(&[1.0, 2.0, 3.0, 4.0], 0);
        assert_eq!(unbounded.ring.len(), 4);
    }

    #[test]
    fn drift_z_follows_the_recent_values() {
        let mut up = welford(&[0.0; 20], 4);
        for _ in 0..4 {
            up.push(1.0, 4);
        }
        assert!(up.drift_z() > 1.0, "{}", up.drift_z());
        let mut down = welford(&[1.0; 20], 4);
        for _ in 0..4 {
            down.push(0.0, 4);
        }
        assert!(down.drift_z() < -1.0, "{}", down.drift_z());
        assert_eq!(welford(&[0.5; 10], 4).drift_z(), 0.0, "no spread, no drift");
        assert_eq!(Welford::default().recent_mean(), 0.0);
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn decay_shape_known_vectors() {
        // One clear winner over a uniform tail.
        let d = decay_shape(&[0.5, 1.0, 0.5, 0.5]).unwrap();
        assert_eq!((d.n, d.top1, d.top2, d.margin), (4, 1.0, 0.5, 0.5));
        assert_eq!(
            (d.top_gap_ratio, d.tail_flatness, d.entropy_norm, d.positive_ratio),
            (1.0, 0.0, 1.0, 1.0)
        );

        // [0.9, 0.8, 0.1]: tail [0.8, 0.1], median 0.1, tail σ 0.35, range 0.8.
        let d = decay_shape(&[0.1, 0.9, f32::NAN, 0.8]).unwrap();
        assert_eq!(d.n, 3);
        assert!(near(d.margin, 0.1) && near(d.top_gap_ratio, 0.125), "{d:?}");
        assert!(near(d.tail_flatness, 0.4375), "{d:?}");
        assert!(near(d.entropy_norm, 0.7813), "{d:?}");
        assert!(near(d.positive_ratio, 1.0));

        // All equal: no gap, fully flat.
        let d = decay_shape(&[0.3, 0.3, 0.3]).unwrap();
        assert_eq!((d.margin, d.top_gap_ratio, d.tail_flatness, d.entropy_norm), (0.0, 0.0, 1.0, 1.0));

        let d = decay_shape(&[1.0, -1.0]).unwrap();
        assert_eq!(d.positive_ratio, 0.5);
        assert!(decay_shape(&[1.0]).is_none());
        assert!(decay_shape(&[1.0, f32::INFINITY]).is_none());
    }

    /// Per-entry merge (the ledger's read rule) equals whole-scope absorb (logis `entry_merge_tests`).
    #[test]
    fn entry_merge_matches_whole_scope_absorb() {
        const RING: usize = 8;
        let mut base = ScopeStat::default();
        let mut run = ScopeStat::default();
        base.baseline.insert("both".into(), welford(&[0.1, 0.4, 0.2, 0.9, 1.3], RING));
        run.baseline
            .insert("both".into(), welford(&[0.7, 0.3, 0.8, 0.2, 0.6, 0.5, 0.4], RING));
        base.baseline.insert("base_only".into(), welford(&[1.0, 2.0, 4.0], RING));
        run.baseline.insert("run_only".into(), welford(&[3.0, 5.0], RING));
        for (st, salt) in [(&mut base, 1.0f64), (&mut run, 2.0)] {
            let d = st.decay.entry("axis".into()).or_default();
            for k in 0..6 {
                let x = salt * 0.1 + k as f64 * 0.07;
                d.margin.push(x, RING);
                d.top_gap_ratio.push(x * 0.5, RING);
                d.tail_flatness.push(1.0 - x, RING);
                d.entropy_norm.push(x * x, RING);
                d.positive_ratio.push(0.3 + x, RING);
            }
            let r = st.reject.entry("record-12".into()).or_default();
            r.seen += 3 + salt as u64;
            r.rejected += 1;
            *r.kinds.entry("demoted".into()).or_default() += salt as u64;
            let c = st.confusion.entry("a|b".into()).or_default();
            c.ties += 4;
            c.a_wins += salt as u64;
            c.b_wins += 4 - salt as u64;
            c.margin.push(0.05 * salt, RING);
            *st.transition.entry("x>y".into()).or_default() += salt as u64;
        }
        let mut merged = base.clone();
        merged.absorb(run.clone(), RING);

        let entry = |b: Option<&Welford>, r: Option<&Welford>| match (b, r) {
            (Some(b), Some(r)) => {
                let mut m = b.clone();
                m.merge(r, RING);
                m
            }
            (Some(e), None) | (None, Some(e)) => e.clone(),
            (None, None) => unreachable!(),
        };
        for k in ["both", "base_only", "run_only"] {
            assert_eq!(entry(base.baseline.get(k), run.baseline.get(k)), merged.baseline[k], "baseline {k}");
        }
        let mut d = base.decay["axis"].clone();
        d.merge_from(&run.decay["axis"], RING);
        assert_eq!(d, merged.decay["axis"]);
        let mut r = base.reject["record-12"].clone();
        r.merge_from(&run.reject["record-12"], RING);
        assert_eq!(r, merged.reject["record-12"]);
        assert_eq!((r.seen, r.rejected, r.kinds["demoted"]), (9, 2, 3));
        let mut c = base.confusion["a|b"].clone();
        c.merge_from(&run.confusion["a|b"], RING);
        assert_eq!(c, merged.confusion["a|b"]);
        assert_eq!(merged.transition["x>y"], 3);
    }

    #[test]
    fn specificity_and_tie_verdict() {
        let r = RejectStat {
            seen: 10,
            rejected: 3,
            ..Default::default()
        };
        assert!((r.specificity() - 0.3).abs() < 1e-6);
        assert_eq!(
            RejectStat {
                seen: 0,
                rejected: 2,
                ..Default::default()
            }
            .specificity(),
            1.0
        );
        assert_eq!(RejectStat::default().specificity(), 0.0);

        assert!(!tie_verdict(None, None, None));
        assert!(!tie_verdict(Some((false, 0.9)), None, None), "rival is the historical winner");
        assert!(!tie_verdict(Some((true, 0.6)), None, None), "bare majority is not enough");
        assert!(!tie_verdict(Some((true, f32::NAN)), None, None));
        assert!(tie_verdict(Some((true, TIE_WIN_RATE)), None, None));
        assert!(
            tie_verdict(Some((true, 0.8)), Some(0.5), None),
            "unknown rival specificity: rate alone"
        );
        assert!(tie_verdict(Some((true, 0.8)), Some(0.3), Some(0.2)), "within slack");
        assert!(
            !tie_verdict(Some((true, 0.8)), Some(0.4), Some(0.2)),
            "rejected 0.2 more often than the rival"
        );
    }
}
