//! Block decoders and fused dequant-dot kernels (PRD §3.5).
//!
//! Weights stay in their packed block bytes (`q`, `d`, `scales`, `dmin`, …); a dot product
//! unpacks one block at a time into registers/stack and accumulates in **F32** — Tier 2 and the
//! normalisation tiers never accumulate in reduced precision (precision invariant).
//! Q4_K/Q5_K use the factored form `Σ_j d·sc_j·Σ(q·x) − dmin·m_j·Σx`, so the min term costs one
//! multiply per 32-element sub-block.

use crate::gguf::GgmlType;
use half::{bf16, f16};

#[inline]
fn f16_at(b: &[u8], i: usize) -> f32 {
    f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Decode one row of `ty` (`out.len()` = columns).
pub fn dequant_row(ty: GgmlType, b: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => out
            .iter_mut()
            .zip(b.chunks_exact(4))
            .for_each(|(o, c)| *o = f32::from_le_bytes(c.try_into().unwrap())),
        GgmlType::F16 => out
            .iter_mut()
            .zip(b.chunks_exact(2))
            .for_each(|(o, c)| *o = f16::from_le_bytes([c[0], c[1]]).to_f32()),
        GgmlType::BF16 => out
            .iter_mut()
            .zip(b.chunks_exact(2))
            .for_each(|(o, c)| *o = bf16::from_le_bytes([c[0], c[1]]).to_f32()),
        GgmlType::I32 => out
            .iter_mut()
            .zip(b.chunks_exact(4))
            .for_each(|(o, c)| *o = i32::from_le_bytes(c.try_into().unwrap()) as f32),
        GgmlType::Q8_0 => {
            for (blk, y) in b.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(blk, 0);
                for (yi, &q) in y.iter_mut().zip(&blk[2..]) {
                    *yi = d * (q as i8) as f32;
                }
            }
        }
        GgmlType::Q4K => {
            for (blk, y) in b.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
                let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                let sc = &blk[4..16];
                let qs = &blk[16..];
                for c in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * c, sc);
                    let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                    let (d1, mm1, d2, mm2) = (d * s1 as f32, dmin * m1 as f32, d * s2 as f32, dmin * m2 as f32);
                    let q = &qs[32 * c..32 * c + 32];
                    for l in 0..32 {
                        y[64 * c + l] = d1 * (q[l] & 0xF) as f32 - mm1;
                        y[64 * c + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
                    }
                }
            }
        }
        GgmlType::Q5K => {
            for (blk, y) in b.chunks_exact(176).zip(out.chunks_exact_mut(256)) {
                let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                let sc = &blk[4..16];
                let qh = &blk[16..48];
                let ql = &blk[48..];
                for c in 0..4 {
                    let (s1, m1) = scale_min_k4(2 * c, sc);
                    let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                    let (u1, u2) = (1u8 << (2 * c), 2u8 << (2 * c));
                    let q = &ql[32 * c..32 * c + 32];
                    for l in 0..32 {
                        let lo = (q[l] & 0xF) + if qh[l] & u1 != 0 { 16 } else { 0 };
                        let hi = (q[l] >> 4) + if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[64 * c + l] = d * s1 as f32 * lo as f32 - dmin * m1 as f32;
                        y[64 * c + 32 + l] = d * s2 as f32 * hi as f32 - dmin * m2 as f32;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for (blk, y) in b.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
                let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
                let d = f16_at(blk, 208);
                for h in 0..2 {
                    let (ql, qh, sc, y) = (&ql[64 * h..], &qh[32 * h..], &sc[8 * h..], &mut y[128 * h..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
    }
}

#[inline]
fn dot32(q: impl Iterator<Item = f32>, x: &[f32]) -> f32 {
    // Eight partial sums keep the loop vectorisable while staying in f32.
    let mut acc = [0f32; 8];
    for (i, (qv, &xv)) in q.zip(x).enumerate() {
        acc[i & 7] += qv * xv;
    }
    acc.iter().sum()
}

#[inline]
fn sum(x: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (i, &v) in x.iter().enumerate() {
        acc[i & 7] += v;
    }
    acc.iter().sum()
}

/// Fused dequant-dot of one packed row with an f32 activation vector (F32 accumulator).
pub fn dot_row(ty: GgmlType, b: &[u8], x: &[f32]) -> f32 {
    match ty {
        GgmlType::Q8_0 => {
            let mut acc = 0f32;
            for (blk, xb) in b.chunks_exact(34).zip(x.chunks_exact(32)) {
                acc += f16_at(blk, 0) * dot32(blk[2..].iter().map(|&q| (q as i8) as f32), xb);
            }
            acc
        }
        GgmlType::Q4K => {
            let mut acc = 0f32;
            for (blk, xs) in b.chunks_exact(144).zip(x.chunks_exact(256)) {
                let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                let (sc, qs) = (&blk[4..16], &blk[16..]);
                let (mut a, mut m) = (0f32, 0f32);
                for c in 0..4 {
                    let q = &qs[32 * c..32 * c + 32];
                    let (x1, x2) = (&xs[64 * c..64 * c + 32], &xs[64 * c + 32..64 * c + 64]);
                    let (s1, m1) = scale_min_k4(2 * c, sc);
                    let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                    a += s1 as f32 * dot32(q.iter().map(|v| (v & 0xF) as f32), x1)
                        + s2 as f32 * dot32(q.iter().map(|v| (v >> 4) as f32), x2);
                    m += m1 as f32 * sum(x1) + m2 as f32 * sum(x2);
                }
                acc += d * a - dmin * m;
            }
            acc
        }
        GgmlType::Q5K => {
            let mut acc = 0f32;
            for (blk, xs) in b.chunks_exact(176).zip(x.chunks_exact(256)) {
                let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
                let (sc, qh, ql) = (&blk[4..16], &blk[16..48], &blk[48..]);
                let (mut a, mut m) = (0f32, 0f32);
                for c in 0..4 {
                    let q = &ql[32 * c..32 * c + 32];
                    let (u1, u2) = (1u8 << (2 * c), 2u8 << (2 * c));
                    let (x1, x2) = (&xs[64 * c..64 * c + 32], &xs[64 * c + 32..64 * c + 64]);
                    let (s1, m1) = scale_min_k4(2 * c, sc);
                    let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                    let lo = q.iter().zip(qh).map(|(v, h)| ((v & 0xF) + if h & u1 != 0 { 16 } else { 0 }) as f32);
                    let hi = q.iter().zip(qh).map(|(v, h)| ((v >> 4) + if h & u2 != 0 { 16 } else { 0 }) as f32);
                    a += s1 as f32 * dot32(lo, x1) + s2 as f32 * dot32(hi, x2);
                    m += m1 as f32 * sum(x1) + m2 as f32 * sum(x2);
                }
                acc += d * a - dmin * m;
            }
            acc
        }
        GgmlType::Q6K => {
            let mut acc = 0f32;
            let mut tmp = [0f32; 256];
            for (blk, xs) in b.chunks_exact(210).zip(x.chunks_exact(256)) {
                // Q6_K's 16-element sub-blocks interleave across the two 128-halves; decode the
                // integer levels once, then apply one scale per 16 elements.
                let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
                for h in 0..2 {
                    for l in 0..32 {
                        let (a, b2, hh) = (ql[64 * h + l], ql[64 * h + l + 32], qh[32 * h + l]);
                        tmp[128 * h + l] = (((a & 0xF) | ((hh & 3) << 4)) as i32 - 32) as f32;
                        tmp[128 * h + l + 32] = (((b2 & 0xF) | (((hh >> 2) & 3) << 4)) as i32 - 32) as f32;
                        tmp[128 * h + l + 64] = (((a >> 4) | (((hh >> 4) & 3) << 4)) as i32 - 32) as f32;
                        tmp[128 * h + l + 96] = (((b2 >> 4) | (((hh >> 6) & 3) << 4)) as i32 - 32) as f32;
                    }
                }
                let mut a = 0f32;
                for (s, &scale) in sc.iter().enumerate() {
                    let r = 16 * s..16 * s + 16;
                    a += (scale as i8) as f32 * dot32(tmp[r.clone()].iter().copied(), &xs[r]);
                }
                acc += f16_at(blk, 208) * a;
            }
            acc
        }
        _ => {
            let mut row = vec![0f32; x.len()];
            dequant_row(ty, b, &mut row);
            dot32(row.into_iter(), x)
        }
    }
}

/// `y = W x` for a packed row-major matrix, rows split across threads.
pub fn matvec(ty: GgmlType, w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let rb = ty.row_bytes(cols);
    assert_eq!(w.len(), rows * rb);
    assert_eq!(x.len(), cols);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(rows.max(1));
    let per = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, ys) in y.chunks_mut(per).enumerate() {
            let w = &w[t * per * rb..];
            s.spawn(move || {
                for (i, yi) in ys.iter_mut().enumerate() {
                    *yi = dot_row(ty, &w[i * rb..(i + 1) * rb], x);
                }
            });
        }
    });
}

/// Result of the start-up kernel self-test.
#[derive(Debug, Clone, serde::Serialize)]
pub struct KernelCheck {
    pub ty: &'static str,
    pub max_abs_err: f32,
    pub ok: bool,
}

/// PRD §3.5 precision invariant: before enabling the native kernels, run them on a small
/// test tensor and compare against an F64 reference over the *decoded* weights. Each format
/// must agree within `tol` (default 1e-3).
pub fn verify_kernels(tol: f32) -> Vec<KernelCheck> {
    use crate::kquant::{encode, Format};
    let (rows, cols) = (48usize, 1024usize);
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..rows * cols)
        .map(|i| rnd() * 0.05 * if i % 97 == 0 { 6.0 } else { 1.0 })
        .collect();
    let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
    let mut out = Vec::new();
    for f in [Format::Q8_0, Format::Q6K, Format::Q5K, Format::Q4K] {
        let ty = f.ggml_type();
        let packed = encode(&w, cols, f, None).expect("encode");
        let mut y = vec![0f32; rows];
        matvec(ty, &packed, rows, cols, &x, &mut y);
        let rb = ty.row_bytes(cols);
        let mut deq = vec![0f32; cols];
        let mut max_err = 0f32;
        for r in 0..rows {
            dequant_row(ty, &packed[r * rb..(r + 1) * rb], &mut deq);
            let reference: f64 = deq.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
            max_err = max_err.max((y[r] as f64 - reference).abs() as f32);
        }
        out.push(KernelCheck {
            ty: ty.name(),
            max_abs_err: max_err,
            ok: max_err <= tol,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kquant::{encode, rel, Format};

    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (s >> 33) as f32 / (1u64 << 31) as f32 - 1.0;
                u * 0.02 * if i % 97 == 0 { 6.0 } else { 1.0 }
            })
            .collect()
    }

    #[test]
    fn roundtrip_error_ordering_matches_bit_width() {
        let cols = 1024;
        let x = data(32 * cols, 7);
        let mut errs = Vec::new();
        for f in [Format::Q4K, Format::Q5K, Format::Q6K, Format::Q8_0] {
            let ty = f.ggml_type();
            let b = encode(&x, cols, f, None).unwrap();
            assert_eq!(b.len(), 32 * ty.row_bytes(cols));
            let mut y = vec![0f32; x.len()];
            for (r, yr) in y.chunks_exact_mut(cols).enumerate() {
                dequant_row(ty, &b[r * ty.row_bytes(cols)..(r + 1) * ty.row_bytes(cols)], yr);
            }
            errs.push(rel(&y, &x));
        }
        // Same bounds the reference crate tests against candle's ggml decoder.
        assert!(errs[0] < 0.11 && errs[1] < 0.055 && errs[2] < 0.03 && errs[3] < 0.008, "{errs:?}");
        assert!(errs.windows(2).all(|w| w[0] > w[1]), "{errs:?}");
    }

    #[test]
    fn fused_dot_matches_decoded_reference() {
        for c in verify_kernels(1e-3) {
            assert!(c.ok, "{c:?}");
            assert!(c.max_abs_err < 1e-4, "{c:?}");
        }
    }
}
