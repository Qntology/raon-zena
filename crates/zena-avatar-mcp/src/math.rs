//! Small column-major matrix / quaternion helpers (`m[col][row]`, quaternions `[x, y, z, w]`).

pub type Mat4 = [[f32; 4]; 4];
pub type V3 = [f32; 3];

pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0; 4]; 4];
    for (c, col) in r.iter_mut().enumerate() {
        for (row, v) in col.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[k][row] * b[c][k]).sum();
        }
    }
    r
}

pub fn translation(t: V3) -> Mat4 {
    let mut m = IDENTITY;
    m[3] = [t[0], t[1], t[2], 1.0];
    m
}

pub fn scaling(s: V3) -> Mat4 {
    let mut m = IDENTITY;
    m[0][0] = s[0];
    m[1][1] = s[1];
    m[2][2] = s[2];
    m
}

pub fn quat_axis_angle(axis: V3, angle: f32) -> [f32; 4] {
    let a = normalize(axis);
    let (s, c) = (angle * 0.5).sin_cos();
    [a[0] * s, a[1] * s, a[2] * s, c]
}

pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

pub fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if l < 1e-12 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / l, q[1] / l, q[2] / l, q[3] / l]
    }
}

pub fn quat_to_mat(q: [f32; 4]) -> Mat4 {
    let [x, y, z, w] = quat_normalize(q);
    let (xx, yy, zz, xy, xz, yz, wx, wy, wz) = (x * x, y * y, z * z, x * y, x * z, y * z, w * x, w * y, w * z);
    [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0],
        [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0],
        [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

pub fn rotation(axis: V3, angle: f32) -> Mat4 {
    quat_to_mat(quat_axis_angle(axis, angle))
}

/// `T * R * S`.
pub fn trs(t: V3, r: [f32; 4], s: V3) -> Mat4 {
    let mut m = quat_to_mat(r);
    for (c, sc) in s.iter().enumerate() {
        for v in m[c].iter_mut().take(3) {
            *v *= sc;
        }
    }
    m[3] = [t[0], t[1], t[2], 1.0];
    m
}

pub fn transform_point(m: &Mat4, p: V3) -> V3 {
    let mut r = [0.0; 3];
    for (row, v) in r.iter_mut().enumerate() {
        *v = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2] + m[3][row];
    }
    r
}

pub fn transform_vector(m: &Mat4, p: V3) -> V3 {
    let mut r = [0.0; 3];
    for (row, v) in r.iter_mut().enumerate() {
        *v = m[0][row] * p[0] + m[1][row] * p[1] + m[2][row] * p[2];
    }
    r
}

fn col3(m: &Mat4, c: usize) -> V3 {
    [m[c][0], m[c][1], m[c][2]]
}

pub fn det3(m: &Mat4) -> f32 {
    dot(col3(m, 0), cross(col3(m, 1), col3(m, 2)))
}

/// Transforms a normal by the inverse-transpose of the upper 3×3 (unnormalised direction, sign-correct).
pub fn transform_normal(m: &Mat4, n: V3) -> V3 {
    let (a, b, c) = (col3(m, 0), col3(m, 1), col3(m, 2));
    let (bc, ca, ab) = (cross(b, c), cross(c, a), cross(a, b));
    let s = if det3(m) < 0.0 { -1.0 } else { 1.0 };
    normalize([
        s * (bc[0] * n[0] + ca[0] * n[1] + ab[0] * n[2]),
        s * (bc[1] * n[0] + ca[1] * n[1] + ab[1] * n[2]),
        s * (bc[2] * n[0] + ca[2] * n[1] + ab[2] * n[2]),
    ])
}

/// Inverse of an affine matrix (upper 3×3 + translation); identity if singular.
pub fn inverse_affine(m: &Mat4) -> Mat4 {
    let (a, b, c) = (col3(m, 0), col3(m, 1), col3(m, 2));
    let det = det3(m);
    if det.abs() < 1e-20 {
        return IDENTITY;
    }
    let rows = [cross(b, c), cross(c, a), cross(a, b)];
    let mut r = IDENTITY;
    for (row, rv) in rows.iter().enumerate() {
        for col in 0..3 {
            r[col][row] = rv[col] / det;
        }
    }
    let t = transform_vector(&r, [m[3][0], m[3][1], m[3][2]]);
    r[3] = [-t[0], -t[1], -t[2], 1.0];
    r
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn scale(a: V3, s: f32) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    let l = length(a);
    if l < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        scale(a, 1.0 / l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Mat4, b: &Mat4) -> bool {
        (0..4).all(|c| (0..4).all(|r| (a[c][r] - b[c][r]).abs() < 1e-5))
    }

    #[test]
    fn trs_inverse_and_quat_rotation() {
        let m = trs([1.0, 2.0, 3.0], quat_axis_angle([0.3, 1.0, 0.2], 0.7), [2.0, 0.5, 1.5]);
        assert!(close(&mat_mul(&m, &inverse_affine(&m)), &IDENTITY));
        let r = rotation([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
        let p = transform_point(&r, [1.0, 0.0, 0.0]);
        assert!((p[0]).abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6);
        let q = quat_mul(quat_axis_angle([0.0, 0.0, 1.0], 0.4), quat_axis_angle([0.0, 0.0, 1.0], 0.6));
        assert!(close(&quat_to_mat(q), &rotation([0.0, 0.0, 1.0], 1.0)));
        let n = transform_normal(&scaling([2.0, 1.0, 1.0]), normalize([1.0, 1.0, 0.0]));
        assert!(n[1] > n[0], "normals shrink along stretched axes: {n:?}");
    }
}
