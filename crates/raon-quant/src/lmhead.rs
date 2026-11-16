//! Exact decoding with a sliced output head.
//!
//! Slicing `lm_head` to `kept_ids` (PRD §3.4) is only safe if the arg-max (or top-k) never sits
//! in a dropped row. The reference implementation this project grew out of measured slicing and
//! rejected it for exactly that reason. Here the dropped ("cold") rows are clustered offline;
//! each cluster stores its centre `c` and radius `ρ = max ‖w − c‖`. For a hidden state `h`,
//! every cold logit obeys `w·h ≤ c·h + ρ‖h‖` (Cauchy–Schwarz), so after scoring the resident
//! `[K, D]` head we only re-score the clusters whose bound beats the current best — gathered row
//! by row from the on-disk table. The result equals a full `[V, D]` arg-max over the same
//! quantized rows.

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct ColdGuard {
    pub dim: usize,
    /// `[C, D]` cluster centres.
    pub centers: Vec<f32>,
    /// `[C]` radii (already inflated by a rounding margin).
    pub radius: Vec<f32>,
    /// Cold token ids grouped by cluster.
    pub members: Vec<u32>,
    /// `[C + 1]` offsets into `members`.
    pub offsets: Vec<u32>,
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        acc[i & 7] += x * y;
    }
    acc.iter().sum()
}

/// Lloyd k-means (Euclidean) over the decoded cold rows `rows` (`n × dim`), seeded with evenly
/// strided rows. Deterministic.
pub fn build_cold_guard(rows: &[f32], ids: &[u32], dim: usize, clusters: usize, iters: usize) -> ColdGuard {
    let n = ids.len();
    assert_eq!(rows.len(), n * dim);
    if n == 0 {
        return ColdGuard {
            dim,
            centers: vec![],
            radius: vec![],
            members: vec![],
            offsets: vec![0],
        };
    }
    let k = clusters.clamp(1, n);
    let mut centers: Vec<f32> = (0..k)
        .flat_map(|c| rows[(c * n / k) * dim..(c * n / k + 1) * dim].to_vec())
        .collect();
    let mut assign = vec![0u32; n];
    for it in 0..=iters {
        let cn: Vec<f32> = centers.chunks_exact(dim).map(|c| dot(c, c)).collect();
        assign.par_iter_mut().enumerate().for_each(|(i, a)| {
            let w = &rows[i * dim..(i + 1) * dim];
            let mut best = (f32::INFINITY, 0u32);
            for (c, cv) in centers.chunks_exact(dim).enumerate() {
                let d = cn[c] - 2.0 * dot(w, cv);
                if d < best.0 {
                    best = (d, c as u32);
                }
            }
            *a = best.1;
        });
        if it == iters {
            break;
        }
        let mut sums = vec![0f64; k * dim];
        let mut cnt = vec![0usize; k];
        for (i, &a) in assign.iter().enumerate() {
            cnt[a as usize] += 1;
            for (s, &v) in sums[a as usize * dim..(a as usize + 1) * dim]
                .iter_mut()
                .zip(&rows[i * dim..(i + 1) * dim])
            {
                *s += v as f64;
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                for d in 0..dim {
                    centers[c * dim + d] = (sums[c * dim + d] / cnt[c] as f64) as f32;
                }
            }
        }
    }
    let mut radius = vec![0f32; k];
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &a) in assign.iter().enumerate() {
        let (w, c) = (&rows[i * dim..(i + 1) * dim], &centers[a as usize * dim..(a as usize + 1) * dim]);
        let r = w.iter().zip(c).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>().sqrt() as f32;
        radius[a as usize] = radius[a as usize].max(r);
        buckets[a as usize].push(ids[i]);
    }
    for r in &mut radius {
        *r = *r * (1.0 + 1e-4) + 1e-6;
    }
    let mut members = Vec::with_capacity(n);
    let mut offsets = vec![0u32];
    for b in buckets {
        members.extend(b);
        offsets.push(members.len() as u32);
    }
    ColdGuard {
        dim,
        centers,
        radius,
        members,
        offsets,
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ArgmaxStats {
    pub clusters: usize,
    pub clusters_refined: usize,
    pub rows_refined: usize,
}

/// Exact top-1 over the full vocabulary.
///
/// * `kept_logits[i]` is the logit of token `kept_ids[i]` (from the resident sliced head);
/// * `cold_logit(id)` scores one dropped row from the on-disk table (same quantization).
pub fn exact_argmax(
    kept_logits: &[f32],
    kept_ids: &[u32],
    guard: &ColdGuard,
    h: &[f32],
    cold_logit: impl Fn(u32) -> f32,
) -> (u32, f32, ArgmaxStats) {
    let (mut best_id, mut best) = (u32::MAX, f32::NEG_INFINITY);
    for (&l, &id) in kept_logits.iter().zip(kept_ids) {
        if l > best {
            best = l;
            best_id = id;
        }
    }
    let hn = dot(h, h).sqrt();
    let c = guard.radius.len();
    let mut bounds: Vec<(f32, usize)> = (0..c)
        .map(|j| {
            let ch = dot(&guard.centers[j * guard.dim..(j + 1) * guard.dim], h);
            let ub = ch + guard.radius[j] * hn;
            // Margin for f32 rounding differences between the bound and the row kernels.
            (ub + 1e-3 * (ch.abs() + guard.radius[j] * hn) + 1e-4, j)
        })
        .collect();
    bounds.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut st = ArgmaxStats {
        clusters: c,
        ..Default::default()
    };
    for (ub, j) in bounds {
        if ub < best {
            break;
        }
        st.clusters_refined += 1;
        for &id in &guard.members[guard.offsets[j] as usize..guard.offsets[j + 1] as usize] {
            st.rows_refined += 1;
            let l = cold_logit(id);
            if l > best || (l == best && id < best_id) {
                best = l;
                best_id = id;
            }
        }
    }
    (best_id, best, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }

    #[test]
    fn sliced_argmax_equals_full_argmax() {
        let (v, d) = (3000usize, 64usize);
        let mut s = 42u64;
        let table: Vec<f32> = (0..v * d).map(|_| rng(&mut s)).collect();
        let kept: Vec<u32> = (0..v as u32).filter(|i| i % 3 != 0).collect();
        let cold: Vec<u32> = (0..v as u32).filter(|i| i % 3 == 0).collect();
        let cold_rows: Vec<f32> = cold
            .iter()
            .flat_map(|&i| table[i as usize * d..(i as usize + 1) * d].to_vec())
            .collect();
        let g = build_cold_guard(&cold_rows, &cold, d, 32, 4);
        assert_eq!(g.members.len(), cold.len());
        let mut refined = 0;
        for trial in 0..200 {
            let mut h: Vec<f32> = (0..d).map(|_| rng(&mut s)).collect();
            if trial % 2 == 0 {
                // Point h at a cold row so the true arg-max is outside the kept set.
                let t = cold[trial % cold.len()] as usize;
                h.iter_mut().zip(&table[t * d..(t + 1) * d]).for_each(|(a, b)| *a = 0.1 * *a + b);
            }
            let logit = |id: u32| dot(&table[id as usize * d..(id as usize + 1) * d], &h);
            let kept_logits: Vec<f32> = kept.iter().map(|&i| logit(i)).collect();
            let (id, l, st) = exact_argmax(&kept_logits, &kept, &g, &h, logit);
            let (full_id, full_l) = (0..v as u32)
                .map(|i| (i, logit(i)))
                .fold((0, f32::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            assert_eq!((id, l), (full_id, full_l), "trial {trial}");
            refined += st.rows_refined;
        }
        // The bound prunes: on average far fewer rows than the cold set are re-scored.
        assert!(refined / 200 < cold.len(), "{refined}");
    }
}
