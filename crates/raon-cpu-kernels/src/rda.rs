//! Zero-allocation SIMD RDA (per-key-channel decay gated delta rule) host scan for Raon-1-nano.
//!
//! For each head `hi in 0..h` in parallel (rayon) and each token `ti in 0..t`:
//!   1. `decay[k] = exp(la[ti, hi, k])`
//!   2. For each `v in 0..d`:
//!        `S[v, k] *= decay[k]`
//!        `pred[v] = sum_k S[v, k] * k_t[k]`
//!   3. For each `v in 0..d`:
//!        `e = beta[ti, hi] * (v_t[v] - pred[v])`
//!        `S[v, k] += e * k_t[k]`
//!        `out[ti, hi, v] = sum_k S[v, k] * q_t[k]`
//!
//! Eliminates all inner-loop heap allocations (`decay` per token, `outs` Vec per head) and uses
//! fused AVX2+FMA inner loops for `d = 128` (zero scalar tail).

use rayon::prelude::*;

struct SendPtr(*mut f32);
unsafe impl Sync for SendPtr {}
unsafe impl Send for SendPtr {}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn decay_and_dot_avx2(row: *mut f32, decay: *const f32, kt: *const f32, d: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= d {
        let r0 = _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), _mm256_loadu_ps(decay.add(i)));
        let r1 = _mm256_mul_ps(_mm256_loadu_ps(row.add(i + 8)), _mm256_loadu_ps(decay.add(i + 8)));
        _mm256_storeu_ps(row.add(i), r0);
        _mm256_storeu_ps(row.add(i + 8), r1);
        s0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(kt.add(i)), s0);
        s1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(kt.add(i + 8)), s1);
        i += 16;
    }
    let mut s = _mm256_add_ps(s0, s1);
    while i + 8 <= d {
        let r = _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), _mm256_loadu_ps(decay.add(i)));
        _mm256_storeu_ps(row.add(i), r);
        s = _mm256_fmadd_ps(r, _mm256_loadu_ps(kt.add(i)), s);
        i += 8;
    }
    let lo = _mm256_castps256_ps128(s);
    let hi = _mm256_extractf128_ps(s, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut acc = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < d {
        let rv = *row.add(i) * *decay.add(i);
        *row.add(i) = rv;
        acc = rv.mul_add(*kt.add(i), acc);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn update_and_dot_avx2(row: *mut f32, e: f32, kt: *const f32, qt: *const f32, d: usize) -> f32 {
    use std::arch::x86_64::*;
    let ev = _mm256_set1_ps(e);
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= d {
        let r0 = _mm256_fmadd_ps(ev, _mm256_loadu_ps(kt.add(i)), _mm256_loadu_ps(row.add(i)));
        let r1 = _mm256_fmadd_ps(ev, _mm256_loadu_ps(kt.add(i + 8)), _mm256_loadu_ps(row.add(i + 8)));
        _mm256_storeu_ps(row.add(i), r0);
        _mm256_storeu_ps(row.add(i + 8), r1);
        s0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(qt.add(i)), s0);
        s1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(qt.add(i + 8)), s1);
        i += 16;
    }
    let mut s = _mm256_add_ps(s0, s1);
    while i + 8 <= d {
        let r = _mm256_fmadd_ps(ev, _mm256_loadu_ps(kt.add(i)), _mm256_loadu_ps(row.add(i)));
        _mm256_storeu_ps(row.add(i), r);
        s = _mm256_fmadd_ps(r, _mm256_loadu_ps(qt.add(i)), s);
        i += 8;
    }
    let lo = _mm256_castps256_ps128(s);
    let hi = _mm256_extractf128_ps(s, 1);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sums = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let mut acc = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
    while i < d {
        let rv = e.mul_add(*kt.add(i), *row.add(i));
        *row.add(i) = rv;
        acc = rv.mul_add(*qt.add(i), acc);
        i += 1;
    }
    acc
}

/// Token-serial V-first RDA delta-rule scan parallelized across heads, writing directly to `out [t, h, d]`.
#[allow(clippy::too_many_arguments)]
pub fn rda_host_scan_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    la: &[f32],
    beta: &[f32],
    state: &mut [f32],
    t: usize,
    h: usize,
    d: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), t * h * d);
    assert_eq!(k.len(), t * h * d);
    assert_eq!(v.len(), t * h * d);
    assert_eq!(la.len(), t * h * d);
    assert_eq!(beta.len(), t * h);
    assert_eq!(state.len(), h * d * d);
    assert_eq!(out.len(), t * h * d);
    if t == 0 || h == 0 || d == 0 {
        return;
    }

    let avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            !crate::is_force_scalar()
                && std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    let hd = d * d;
    let out_ptr = SendPtr(out.as_mut_ptr());

    let run_head = |(hi, s): (usize, &mut [f32])| {
        let out_ptr = &out_ptr;
        let mut pred = vec![0f32; d];
        let mut decay = vec![0f32; d];
        for ti in 0..t {
            let base = (ti * h + hi) * d;
            let qt = &q[base..base + d];
            let kt = &k[base..base + d];
            let vt = &v[base..base + d];
            let at = &la[base..base + d];
            let bt = beta[ti * h + hi];
            for ki in 0..d {
                decay[ki] = at[ki].exp();
            }

            if avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                        pred[vi] = decay_and_dot_avx2(row.as_mut_ptr(), decay.as_ptr(), kt.as_ptr(), d);
                    }
                    let dst = std::slice::from_raw_parts_mut(out_ptr.0.add(base), d);
                    for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                        let e = bt * (vt[vi] - pred[vi]);
                        dst[vi] = update_and_dot_avx2(row.as_mut_ptr(), e, kt.as_ptr(), qt.as_ptr(), d);
                    }
                }
            } else {
                for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                    let mut acc = 0f32;
                    for ki in 0..d {
                        let rv = row[ki] * decay[ki];
                        row[ki] = rv;
                        acc = rv.mul_add(kt[ki], acc);
                    }
                    pred[vi] = acc;
                }
                let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.0.add(base), d) };
                for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                    let e = bt * (vt[vi] - pred[vi]);
                    let mut acc = 0f32;
                    for ki in 0..d {
                        let rv = e.mul_add(kt[ki], row[ki]);
                        row[ki] = rv;
                        acc = rv.mul_add(qt[ki], acc);
                    }
                    dst[vi] = acc;
                }
            }
        }
    };

    if h <= 1 || t * h * d * d < 16_384 {
        state.chunks_exact_mut(hd).enumerate().for_each(run_head);
    } else {
        state.par_chunks_exact_mut(hd).enumerate().for_each(run_head);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(n: usize, salt: u32) -> Vec<f32> {
        let mut x = 0x9E37_79B9u32 ^ salt.wrapping_mul(2_654_435_761);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn rda_simd_matches_scalar_reference_d128() {
        let (t, h, d) = (5usize, 16usize, 128usize);
        let q = det(t * h * d, 1);
        let k = det(t * h * d, 2);
        let v = det(t * h * d, 3);
        let la: Vec<f32> = det(t * h * d, 4).into_iter().map(|x| -x.abs() - 0.1).collect();
        let beta: Vec<f32> = det(t * h, 5).into_iter().map(|x| 0.5 + 0.4 * x).collect();
        let s_init = det(h * d * d, 6);

        let mut s_simd = s_init.clone();
        let mut out_simd = vec![0f32; t * h * d];
        rda_host_scan_f32(&q, &k, &v, &la, &beta, &mut s_simd, t, h, d, &mut out_simd);

        crate::set_force_scalar(true);
        let mut s_ref = s_init;
        let mut out_ref = vec![0f32; t * h * d];
        rda_host_scan_f32(&q, &k, &v, &la, &beta, &mut s_ref, t, h, d, &mut out_ref);
        crate::set_force_scalar(false);

        let max_o = out_simd
            .iter()
            .zip(&out_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_s = s_simd
            .iter()
            .zip(&s_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_o < 1e-4, "RDA out max diff {max_o}");
        assert!(max_s < 1e-4, "RDA state max diff {max_s}");
    }
}
