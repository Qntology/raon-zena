//! Track A built-in parametric generator ("forge"): the in-process stand-in for SolidCraft.
//!
//! Units are metres, Y up, models face +Z (the viewer). Builders return a [`Geo`] (positions, vertex normals,
//! uvs, triangles) that is placed into a [`MeshData`] with a material per part:
//! [`rounded_box`], [`cylinder`], [`sphere`], [`torus_segment`], [`tube`], [`lathe`], [`extrude`] (with a
//! chamfer). On top of them:
//!
//! * [`emote_prop`] – 3D comic emote props sized to a ~0.25 m cube at the origin (head socket);
//! * [`part_mesh`]  – accessories placed around a head of radius ~0.5 centred at the origin;
//! * [`robot_zena`] – the SD toy-robot Zena body with VRM 1.0 preset morph targets on the face.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use zena_core::tools::{AvatarPart, CharacterEmote};

use crate::math::{self, Mat4, V3};
use crate::mesh::{Material, MeshData, MorphTarget};

/// Zena's accent purple (shared with the procedural avatar).
pub const ZENA_PURPLE: [f32; 3] = [0.30, 0.22, 0.45];
/// Zena's eye violet.
pub const ZENA_VIOLET: [f32; 3] = [0.42, 0.25, 0.70];

/// Raw triangle geometry from a builder.
#[derive(Debug, Clone, Default)]
pub struct Geo {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl Geo {
    fn vertex(&mut self, p: V3, n: V3, uv: [f32; 2]) -> u32 {
        self.positions.push(p);
        self.normals.push(math::normalize(n));
        self.uvs.push(uv);
        (self.positions.len() - 1) as u32
    }

    fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.indices.extend([a, b, c, a, c, d]);
    }

    pub fn merge(&mut self, o: &Geo) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&o.positions);
        self.normals.extend_from_slice(&o.normals);
        self.uvs.extend_from_slice(&o.uvs);
        self.indices.extend(o.indices.iter().map(|i| i + base));
    }

    /// A transformed copy (normals by the inverse-transpose; mirrors flip the winding).
    pub fn transformed(&self, m: &Mat4) -> Geo {
        let flip = math::det3(m) < 0.0;
        Geo {
            positions: self.positions.iter().map(|p| math::transform_point(m, *p)).collect(),
            normals: self.normals.iter().map(|n| math::transform_normal(m, *n)).collect(),
            uvs: self.uvs.clone(),
            indices: if flip {
                self.indices.chunks_exact(3).flat_map(|t| [t[0], t[2], t[1]]).collect()
            } else {
                self.indices.clone()
            },
        }
    }

    /// Orients every triangle to agree with its vertex normals and drops degenerate ones.
    fn oriented(mut self) -> Geo {
        let mut out = Vec::with_capacity(self.indices.len());
        for t in self.indices.chunks_exact(3) {
            let p = |i: u32| self.positions[i as usize];
            let g = math::cross(math::sub(p(t[1]), p(t[0])), math::sub(p(t[2]), p(t[0])));
            if math::dot(g, g) < 1e-24 {
                continue;
            }
            let n = math::add(
                math::add(self.normals[t[0] as usize], self.normals[t[1] as usize]),
                self.normals[t[2] as usize],
            );
            if math::dot(g, n) < 0.0 {
                out.extend([t[0], t[2], t[1]]);
            } else {
                out.extend_from_slice(t);
            }
        }
        self.indices = out;
        self
    }

    /// Adds this geometry to `mesh` with `material` under transform `m`; returns the base vertex.
    pub fn add_to(&self, mesh: &mut MeshData, material: usize, m: &Mat4) -> u32 {
        let g = self.transformed(m);
        mesh.push_triangles(material, &g.positions, &g.normals, &g.uvs, &g.indices)
    }
}

// ---------------------------------------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------------------------------------

/// UV sphere centred at the origin.
pub fn sphere(radius: f32, seg_u: u32, seg_v: u32) -> Geo {
    let (su, sv) = (seg_u.max(3), seg_v.max(2));
    let mut g = Geo::default();
    for j in 0..=sv {
        let th = PI * j as f32 / sv as f32;
        for i in 0..=su {
            let ph = TAU * i as f32 / su as f32;
            let n = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
            g.vertex(math::scale(n, radius), n, [i as f32 / su as f32, j as f32 / sv as f32]);
        }
    }
    let row = su + 1;
    for j in 0..sv {
        for i in 0..su {
            let a = j * row + i;
            g.quad(a, a + 1, a + 1 + row, a + row);
        }
    }
    g.oriented()
}

fn corner_samples(half: f32, inner: f32, seg: u32) -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..=seg {
        v.push(-half + (half - inner) * i as f32 / seg as f32);
    }
    for i in 0..=seg {
        v.push(inner + (half - inner) * i as f32 / seg as f32);
    }
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    v
}

/// Box of `size` centred at the origin with edges and corners rounded by `radius`.
pub fn rounded_box(size: [f32; 3], radius: f32, seg: u32) -> Geo {
    let half = size.map(|s| s.abs() * 0.5);
    let r = radius.clamp(1e-4, half.iter().copied().fold(f32::INFINITY, f32::min) * 0.999);
    let inner = half.map(|h| (h - r).max(0.0));
    let seg = seg.max(1);
    let mut g = Geo::default();
    for k in 0..3 {
        for s in [-1.0f32, 1.0] {
            let (u, v) = ((k + 1) % 3, (k + 2) % 3);
            let us = corner_samples(half[u], inner[u], seg);
            let vs = corner_samples(half[v], inner[v], seg);
            let base = g.positions.len() as u32;
            for &b in &vs {
                for &a in &us {
                    let mut q = [0.0; 3];
                    q[k] = s * half[k];
                    q[u] = a;
                    q[v] = b;
                    let c: V3 = std::array::from_fn(|i| q[i].clamp(-inner[i], inner[i]));
                    let dir = math::normalize(math::sub(q, c));
                    let p = math::add(c, math::scale(dir, r));
                    g.vertex(p, dir, [(a + half[u]) / (2.0 * half[u]), (b + half[v]) / (2.0 * half[v])]);
                }
            }
            let row = us.len() as u32;
            for j in 0..vs.len() as u32 - 1 {
                for i in 0..row - 1 {
                    let a = base + j * row + i;
                    g.quad(a, a + 1, a + 1 + row, a + row);
                }
            }
        }
    }
    g.oriented()
}

/// Cylinder / cone frustum along Y centred at the origin (`r_top == 0` makes a cone).
pub fn cylinder(r_bottom: f32, r_top: f32, height: f32, seg: u32, caps: bool) -> Geo {
    let seg = seg.max(3);
    let (y0, y1) = (-height * 0.5, height * 0.5);
    let mut g = Geo::default();
    for i in 0..=seg {
        let ph = TAU * i as f32 / seg as f32;
        let (c, s) = (ph.cos(), ph.sin());
        let n = [c * height, r_bottom - r_top, s * height];
        let u = i as f32 / seg as f32;
        g.vertex([r_bottom * c, y0, r_bottom * s], n, [u, 1.0]);
        g.vertex([r_top * c, y1, r_top * s], n, [u, 0.0]);
    }
    for i in 0..seg {
        let a = 2 * i;
        g.quad(a, a + 2, a + 3, a + 1);
    }
    if caps {
        for (y, r, ny) in [(y0, r_bottom, -1.0f32), (y1, r_top, 1.0)] {
            if r <= 1e-6 {
                continue;
            }
            let centre = g.vertex([0.0, y, 0.0], [0.0, ny, 0.0], [0.5, 0.5]);
            let first = g.positions.len() as u32;
            for i in 0..=seg {
                let ph = TAU * i as f32 / seg as f32;
                g.vertex(
                    [r * ph.cos(), y, r * ph.sin()],
                    [0.0, ny, 0.0],
                    [0.5 + 0.5 * ph.cos(), 0.5 + 0.5 * ph.sin()],
                );
            }
            for i in 0..seg {
                g.indices.extend([centre, first + i, first + i + 1]);
            }
        }
    }
    g.oriented()
}

/// A tube of per-point radius swept along `path` (parallel-transport frames). Open tubes get spherical caps.
pub fn tube_var(path: &[V3], radii: &[f32], seg: u32, closed: bool) -> Geo {
    let mut g = Geo::default();
    let n = path.len();
    if n < 2 {
        return g;
    }
    let seg = seg.max(3);
    let radius = |i: usize| radii.get(i).copied().or(radii.last().copied()).unwrap_or(0.01);
    let tangent = |i: usize| -> V3 {
        let (a, b) = if closed {
            (path[(i + n - 1) % n], path[(i + 1) % n])
        } else {
            (path[i.saturating_sub(1)], path[(i + 1).min(n - 1)])
        };
        math::normalize(math::sub(b, a))
    };
    let t0 = tangent(0);
    let up = if t0[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let mut nrm = math::normalize(math::cross(t0, up));
    for (i, centre) in path.iter().enumerate() {
        let t = tangent(i);
        nrm = math::normalize(math::sub(nrm, math::scale(t, math::dot(nrm, t))));
        let b = math::cross(t, nrm);
        for k in 0..=seg {
            let a = TAU * k as f32 / seg as f32;
            let d = math::add(math::scale(nrm, a.cos()), math::scale(b, a.sin()));
            g.vertex(
                math::add(*centre, math::scale(d, radius(i))),
                d,
                [k as f32 / seg as f32, i as f32 / (n - 1) as f32],
            );
        }
    }
    let row = seg + 1;
    let rings = if closed { n } else { n - 1 };
    for i in 0..rings {
        let (r0, r1) = (i as u32 * row, ((i + 1) % n) as u32 * row);
        for k in 0..seg {
            g.quad(r0 + k, r0 + k + 1, r1 + k + 1, r1 + k);
        }
    }
    if !closed {
        for (i, p) in [(0, path[0]), (n - 1, path[n - 1])] {
            g.merge(&sphere(radius(i), seg, seg / 2 + 1).transformed(&math::translation(p)));
        }
    }
    g.oriented()
}

/// Constant-radius [`tube_var`].
pub fn tube(path: &[V3], radius: f32, seg: u32, closed: bool) -> Geo {
    tube_var(path, &[radius], seg, closed)
}

/// Torus (segment) in the XY plane around +Z: ring radius `major`, tube radius `minor`, from angle `a0` to `a1`
/// (radians, 0 = +X, counter-clockwise). A full turn is closed; segments get rounded ends.
pub fn torus_segment(major: f32, minor: f32, a0: f32, a1: f32, seg_major: u32, seg_minor: u32) -> Geo {
    let full = (a1 - a0).abs() >= TAU - 1e-4;
    let steps = seg_major.max(3);
    let pts: Vec<V3> = if full {
        (0..steps)
            .map(|i| a0 + TAU * i as f32 / steps as f32)
            .map(|a| [major * a.cos(), major * a.sin(), 0.0])
            .collect()
    } else {
        (0..=steps)
            .map(|i| a0 + (a1 - a0) * i as f32 / steps as f32)
            .map(|a| [major * a.cos(), major * a.sin(), 0.0])
            .collect()
    };
    tube(&pts, minor, seg_minor, full)
}

/// Revolves a 2D profile `[radius, y]` (ordered bottom → top for outward normals) around Y.
pub fn lathe(profile: &[[f32; 2]], seg: u32) -> Geo {
    let mut g = Geo::default();
    let n = profile.len();
    if n < 2 {
        return g;
    }
    let seg = seg.max(3);
    let same = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-7;
    for (i, p) in profile.iter().enumerate() {
        let prev = profile[i.saturating_sub(1)];
        let next = profile[(i + 1).min(n - 1)];
        let (a, b) = if i > 0 && same(prev, *p) {
            (*p, next)
        } else if i + 1 < n && same(next, *p) {
            (prev, *p)
        } else {
            (prev, next)
        };
        let t = [b[0] - a[0], b[1] - a[1]];
        let n2 = [t[1], -t[0]];
        for k in 0..=seg {
            let ph = TAU * k as f32 / seg as f32;
            let (c, s) = (ph.cos(), ph.sin());
            g.vertex(
                [p[0] * c, p[1], p[0] * s],
                [n2[0] * c, n2[1], n2[0] * s],
                [k as f32 / seg as f32, i as f32 / (n - 1) as f32],
            );
        }
    }
    let row = seg + 1;
    for i in 0..n as u32 - 1 {
        for k in 0..seg {
            let a = i * row + k;
            g.quad(a, a + 1, a + 1 + row, a + row);
        }
    }
    g.oriented()
}

fn signed_area(p: &[[f32; 2]]) -> f32 {
    (0..p.len())
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % p.len()]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f32>()
        * 0.5
}

fn cross2(o: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Ear-clipping triangulation of a simple counter-clockwise polygon.
fn triangulate(p: &[[f32; 2]]) -> Vec<[u32; 3]> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let mut tris = Vec::new();
    while idx.len() > 3 {
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (a, b, c) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            if cross2(p[a], p[b], p[c]) <= 1e-12 {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                j != a
                    && j != b
                    && j != c
                    && cross2(p[a], p[b], p[j]) > 0.0
                    && cross2(p[b], p[c], p[j]) > 0.0
                    && cross2(p[c], p[a], p[j]) > 0.0
            });
            if !inside {
                tris.push([a as u32, b as u32, c as u32]);
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Degenerate input: drop a collinear vertex, else force an ear.
            match (0..m).find(|&k| cross2(p[idx[(k + m - 1) % m]], p[idx[k]], p[idx[(k + 1) % m]]).abs() <= 1e-12) {
                Some(k) => {
                    idx.remove(k);
                }
                None => {
                    tris.push([idx[m - 1] as u32, idx[0] as u32, idx[1] as u32]);
                    idx.remove(0);
                }
            }
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0] as u32, idx[1] as u32, idx[2] as u32]);
    }
    tris
}

fn edge_normal(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = (dx * dx + dy * dy).sqrt().max(1e-12);
    [dy / l, -dx / l]
}

/// Offsets a counter-clockwise polygon outward by `d` (inward when negative) with clamped mitres.
pub fn offset_polygon(p: &[[f32; 2]], d: f32) -> Vec<[f32; 2]> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
            let (n0, n1) = (edge_normal(a, b), edge_normal(b, c));
            let m = [n0[0] + n1[0], n0[1] + n1[1]];
            let l = (m[0] * m[0] + m[1] * m[1]).sqrt();
            if l < 1e-6 {
                return [b[0] + n1[0] * d, b[1] + n1[1] * d];
            }
            let m = [m[0] / l, m[1] / l];
            let len = d / (m[0] * n1[0] + m[1] * n1[1]).max(0.3);
            [b[0] + m[0] * len, b[1] + m[1] * len]
        })
        .collect()
}

/// Inset by `d` with mitres; if that folds over (an edge reverses direction, as at very sharp tips), the
/// polygon is instead scaled toward its centroid so the farthest vertex moves by `d`.
fn safe_inset(p: &[[f32; 2]], d: f32) -> Vec<[f32; 2]> {
    let off = offset_polygon(p, -d);
    let n = p.len();
    let folded = (0..n).any(|i| {
        let j = (i + 1) % n;
        let (e, f) = (
            [p[j][0] - p[i][0], p[j][1] - p[i][1]],
            [off[j][0] - off[i][0], off[j][1] - off[i][1]],
        );
        e[0] * f[0] + e[1] * f[1] <= 0.0
    });
    if !folded {
        return off;
    }
    let c = p.iter().fold([0.0f32; 2], |a, q| [a[0] + q[0] / n as f32, a[1] + q[1] / n as f32]);
    let r = p
        .iter()
        .map(|q| ((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).sqrt())
        .fold(1e-9f32, f32::max);
    let k = (1.0 - d / r).max(0.1);
    p.iter().map(|q| [c[0] + (q[0] - c[0]) * k, c[1] + (q[1] - c[1]) * k]).collect()
}

/// Rounds every corner of a polygon with arcs of `radius` (`steps` segments per corner).
pub fn rounded_polygon(pts: &[[f32; 2]], radius: f32, steps: u32) -> Vec<[f32; 2]> {
    let n = pts.len();
    let mut out = Vec::new();
    for i in 0..n {
        let (a, p, b) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
        let (la, lb) = (
            ((a[0] - p[0]).powi(2) + (a[1] - p[1]).powi(2)).sqrt(),
            ((b[0] - p[0]).powi(2) + (b[1] - p[1]).powi(2)).sqrt(),
        );
        if la < 1e-9 || lb < 1e-9 {
            continue;
        }
        let u = [(a[0] - p[0]) / la, (a[1] - p[1]) / la];
        let v = [(b[0] - p[0]) / lb, (b[1] - p[1]) / lb];
        let half = ((u[0] * v[0] + u[1] * v[1]).clamp(-1.0, 1.0)).acos() * 0.5;
        if !(1e-3..=FRAC_PI_2 - 1e-3).contains(&half) {
            out.push(p);
            continue;
        }
        let d = (radius / half.tan()).min(0.45 * la.min(lb));
        let r = d * half.tan();
        let bis = math::normalize([u[0] + v[0], u[1] + v[1], 0.0]);
        let c = [p[0] + bis[0] * r / half.sin(), p[1] + bis[1] * r / half.sin()];
        let s = [p[0] + u[0] * d, p[1] + u[1] * d];
        let e = [p[0] + v[0] * d, p[1] + v[1] * d];
        let a0 = (s[1] - c[1]).atan2(s[0] - c[0]);
        let a1 = (e[1] - c[1]).atan2(e[0] - c[0]);
        let mut delta = a1 - a0;
        while delta > PI {
            delta -= TAU;
        }
        while delta < -PI {
            delta += TAU;
        }
        for k in 0..=steps.max(1) {
            let t = a0 + delta * k as f32 / steps.max(1) as f32;
            out.push([c[0] + r * t.cos(), c[1] + r * t.sin()]);
        }
    }
    out
}

/// Extrudes a 2D polygon (XY) along Z, centred, `depth` thick, with a 45° `chamfer` on both cap edges.
/// Corners sharper than ~35° get crisp normals, gentler ones are smoothed.
pub fn extrude(polygon: &[[f32; 2]], depth: f32, chamfer: f32) -> Geo {
    let mut poly: Vec<[f32; 2]> = polygon.to_vec();
    poly.dedup_by(|a, b| (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-7);
    if poly.len() > 1 && (poly[0][0] - poly[poly.len() - 1][0]).abs() + (poly[0][1] - poly[poly.len() - 1][1]).abs() < 1e-7 {
        poly.pop();
    }
    let mut g = Geo::default();
    if poly.len() < 3 {
        return g;
    }
    if signed_area(&poly) < 0.0 {
        poly.reverse();
    }
    let n = poly.len();
    let c = chamfer.clamp(0.0, depth * 0.45);
    let inset = if c > 0.0 { safe_inset(&poly, c) } else { poly.clone() };
    let (z0, z1) = (-depth * 0.5, depth * 0.5);
    let tris = triangulate(&poly);
    for (z, nz) in [(z1, 1.0f32), (z0, -1.0)] {
        let base = g.positions.len() as u32;
        for q in &inset {
            g.vertex([q[0], q[1], z], [0.0, 0.0, nz], [q[0], q[1]]);
        }
        for t in &tris {
            g.indices.extend(t.iter().map(|i| i + base));
        }
    }
    // Side walls (and chamfer strips) per edge with smooth/crisp vertex normals.
    let en: Vec<[f32; 2]> = (0..n).map(|i| edge_normal(poly[i], poly[(i + 1) % n])).collect();
    let vn = |i: usize, e: usize| -> [f32; 2] {
        let prev = en[(i + n - 1) % n];
        let next = en[i % n];
        if prev[0] * next[0] + prev[1] * next[1] > 0.82 {
            let s = [prev[0] + next[0], prev[1] + next[1]];
            let l = (s[0] * s[0] + s[1] * s[1]).sqrt().max(1e-9);
            [s[0] / l, s[1] / l]
        } else {
            en[e]
        }
    };
    for e in 0..n {
        let (i, j) = (e, (e + 1) % n);
        let (ni, nj) = (vn(i, e), vn(j, e));
        let (wi, wj) = (poly[i], poly[j]);
        let side = |g: &mut Geo, pa: [f32; 2], pb: [f32; 2], za: f32, zb: f32, na: V3, nb: V3| {
            let a = g.vertex([pa[0], pa[1], za], na, [0.0, 0.0]);
            let b = g.vertex([pb[0], pb[1], za], nb, [1.0, 0.0]);
            let cc = g.vertex([pb[0], pb[1], zb], nb, [1.0, 1.0]);
            let d = g.vertex([pa[0], pa[1], zb], na, [0.0, 1.0]);
            g.quad(a, b, cc, d);
        };
        side(&mut g, wi, wj, z0 + c, z1 - c, [ni[0], ni[1], 0.0], [nj[0], nj[1], 0.0]);
        if c > 0.0 {
            let (ii, ij) = (inset[i], inset[j]);
            // Top chamfer: outer edge at z1 - c up to the inset cap edge at z1.
            let a = g.vertex([wi[0], wi[1], z1 - c], [ni[0], ni[1], 1.0], [0.0, 0.0]);
            let b = g.vertex([wj[0], wj[1], z1 - c], [nj[0], nj[1], 1.0], [1.0, 0.0]);
            let cc = g.vertex([ij[0], ij[1], z1], [nj[0], nj[1], 1.0], [1.0, 1.0]);
            let d = g.vertex([ii[0], ii[1], z1], [ni[0], ni[1], 1.0], [0.0, 1.0]);
            g.quad(a, b, cc, d);
            let a = g.vertex([ii[0], ii[1], z0], [ni[0], ni[1], -1.0], [0.0, 0.0]);
            let b = g.vertex([ij[0], ij[1], z0], [nj[0], nj[1], -1.0], [1.0, 0.0]);
            let cc = g.vertex([wj[0], wj[1], z0 + c], [nj[0], nj[1], -1.0], [1.0, 1.0]);
            let d = g.vertex([wi[0], wi[1], z0 + c], [ni[0], ni[1], -1.0], [0.0, 1.0]);
            g.quad(a, b, cc, d);
        }
    }
    g.oriented()
}

// ---------------------------------------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------------------------------------

fn bezier(p: [V3; 4], steps: u32) -> Vec<V3> {
    (0..=steps)
        .map(|i| {
            let t = i as f32 / steps as f32;
            let u = 1.0 - t;
            let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            std::array::from_fn(|k| w[0] * p[0][k] + w[1] * p[1][k] + w[2] * p[2][k] + w[3] * p[3][k])
        })
        .collect()
}

fn arc(centre: V3, r: f32, a0: f32, a1: f32, steps: u32) -> Vec<V3> {
    (0..=steps)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f32 / steps as f32;
            [centre[0] + r * a.cos(), centre[1] + r * a.sin(), centre[2]]
        })
        .collect()
}

/// Heart outline `width` wide, centred on its bounding box.
pub fn heart_polygon(width: f32, samples: usize) -> Vec<[f32; 2]> {
    // Two circular lobes (radius r, centres ±a) joined to the tip by tangent lines: no cusps, so insets and
    // chamfers stay well-formed (unlike the classic parametric heart curve).
    let r = width / 3.5;
    let (a, h) = (0.75 * r, 2.0 * r);
    let notch = (r * r - a * a).sqrt();
    let d = (a * a + h * h).sqrt();
    let tangent = (-h).atan2(-a) + (r / d).acos();
    let notch_angle = notch.atan2(-a);
    let n = (samples / 2).max(8);
    let mut pts = vec![[0.0, -h]];
    for i in 0..=n {
        let t = tangent + (notch_angle - tangent) * i as f32 / n as f32;
        pts.push([a + r * t.cos(), r * t.sin()]);
    }
    for i in 1..=n {
        let t = (PI - notch_angle) + (notch_angle - tangent) * i as f32 / n as f32;
        pts.push([-a + r * t.cos(), r * t.sin()]);
    }
    let cy = (r - h) * 0.5;
    let p: Vec<[f32; 2]> = pts.iter().map(|q| [q[0], q[1] - cy]).collect();
    rounded_polygon(&p, r * 0.15, 4)
}

/// A comic sparkle: `points` sharp spikes with concave edges.
pub fn sparkle_polygon(r_out: f32, r_in: f32, points: u32, samples: usize) -> Vec<[f32; 2]> {
    (0..samples)
        .map(|i| {
            let t = TAU * i as f32 / samples as f32 + FRAC_PI_2;
            let k = ((points as f32 * (t - FRAC_PI_2) * 0.5).cos()).abs().powi(3);
            let r = r_in + (r_out - r_in) * k;
            [r * t.cos(), r * t.sin()]
        })
        .collect()
}

/// Five-pointed star outline.
pub fn star_polygon(r_out: f32, r_in: f32) -> Vec<[f32; 2]> {
    (0..10)
        .map(|i| {
            let a = FRAC_PI_2 + PI * i as f32 / 5.0;
            let r = if i % 2 == 0 { r_out } else { r_in };
            [r * a.cos(), r * a.sin()]
        })
        .collect()
}

/// Teardrop lathe profile (round bottom, point on top), `height` tall, `width` wide at most.
fn teardrop(height: f32, width: f32, steps: u32) -> Geo {
    let profile: Vec<[f32; 2]> = (0..=steps)
        .map(|i| {
            let th = PI * (1.0 - i as f32 / steps as f32);
            let r = th.sin() * (th * 0.5).sin() / 0.77 * width * 0.5;
            [r.max(0.0), th.cos() * height * 0.5]
        })
        .collect();
    lathe(&profile, 20)
}

fn mat(name: &str, rgb: [f32; 3], emissive: f32) -> Material {
    Material::new(name, [rgb[0], rgb[1], rgb[2], 1.0]).with_emissive(rgb.map(|c| c * emissive))
}

fn t(p: V3) -> Mat4 {
    math::translation(p)
}

fn rz(a: f32) -> Mat4 {
    math::rotation([0.0, 0.0, 1.0], a)
}

fn compose(ms: &[Mat4]) -> Mat4 {
    ms.iter().fold(math::IDENTITY, |acc, m| math::mat_mul(&acc, m))
}

// ---------------------------------------------------------------------------------------------------------
// Emote props
// ---------------------------------------------------------------------------------------------------------

/// The 3D comic prop for an emote (≈0.25 m cube at the origin), or `None` when the emote has no prop form
/// (`CHIBI_CHEER`, `TEARS`, `STAR_EYES`, `DIZZY_SPIRAL`). `angle_deg` orients `THREAT_ARROW_3D` in the XY plane
/// (0 = right / +X, 90 = up / +Y).
pub fn emote_prop(emote: CharacterEmote, angle_deg: f32) -> Option<MeshData> {
    use CharacterEmote as E;
    Some(match emote {
        E::ComicAnger3d => anger_mark(),
        E::SparkleIdea => light_bulb(),
        E::Exclaim3d => exclamation(),
        E::ThreatArrow3d => arrow(angle_deg),
        E::SweatPanic => sweat_drops(),
        E::ChibiHeart => hearts(false),
        E::HeartEyes => hearts(true),
        E::Question => question_mark(),
        E::MusicNote => music_note(),
        E::ReliefSigh => relief_puff(),
        E::ChibiCheer | E::Tears | E::StarEyes | E::DizzySpiral => return None,
    })
}

/// 💢 — four curved bars around a cross-shaped gap, bulging toward the centre.
fn anger_mark() -> MeshData {
    let mut m = MeshData::new();
    let red = m.add_material(mat("anger_red", [0.92, 0.08, 0.12], 0.25));
    let (d, r, ext) = (0.09, 0.06, 0.25);
    for (sx, sy) in [(1.0f32, 1.0f32), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
        let ax = if sx > 0.0 { PI } else { 0.0 };
        let ay = if sy > 0.0 { -FRAC_PI_2 } else { FRAC_PI_2 };
        let mut delta = ay - ax;
        while delta > PI {
            delta -= TAU;
        }
        while delta <= -PI {
            delta += TAU;
        }
        let dir = delta.signum();
        let path = arc([sx * d, sy * d, 0.0], r, ax - dir * ext, ax + delta + dir * ext, 18);
        let radii: Vec<f32> = (0..path.len())
            .map(|i| 0.016 * (0.7 + 0.3 * (PI * i as f32 / (path.len() - 1) as f32).sin()))
            .collect();
        tube_var(&path, &radii, 12, false).add_to(&mut m, red, &math::IDENTITY);
    }
    m
}

/// 💡 — glass bulb with a screw base and three sparkles.
fn light_bulb() -> MeshData {
    let mut m = MeshData::new();
    let glass = m.add_material(Material::new("bulb_glass", [1.0, 0.86, 0.25, 1.0]).with_emissive([0.85, 0.65, 0.12]));
    let metal = m.add_material(mat("bulb_base", [0.62, 0.62, 0.68], 0.0));
    let spark = m.add_material(Material::new("sparkle", [1.0, 0.95, 0.55, 1.0]).with_emissive([1.0, 0.9, 0.4]));
    sphere(0.062, 28, 16).add_to(&mut m, glass, &t([0.0, 0.03, 0.0]));
    cylinder(0.033, 0.05, 0.045, 28, false).add_to(&mut m, glass, &t([0.0, -0.028, 0.0]));
    let mut profile = vec![[0.0, -0.112], [0.012, -0.11], [0.02, -0.103]];
    for k in 0..4 {
        let y = -0.098 + k as f32 * 0.012;
        profile.push([0.031, y]);
        profile.push([0.027, y + 0.006]);
    }
    profile.push([0.034, -0.05]);
    profile.push([0.0, -0.048]);
    lathe(&profile, 28).add_to(&mut m, metal, &math::IDENTITY);
    for (p, s) in [
        ([-0.088, 0.082, 0.0], 0.03),
        ([0.09, 0.078, 0.0], 0.022),
        ([0.095, -0.02, 0.0], 0.016),
    ] {
        extrude(&sparkle_polygon(s, s * 0.22, 4, 48), s * 0.35, s * 0.08).add_to(&mut m, spark, &t(p));
    }
    m
}

/// ❗ — tapered bar and dot.
fn exclamation() -> MeshData {
    let mut m = MeshData::new();
    let red = m.add_material(mat("exclaim_red", [0.95, 0.18, 0.10], 0.25));
    let orange = m.add_material(mat("exclaim_orange", [1.0, 0.55, 0.10], 0.25));
    cylinder(0.017, 0.03, 0.12, 24, false).add_to(&mut m, red, &t([0.0, 0.025, 0.0]));
    sphere(0.03, 24, 12).add_to(&mut m, red, &t([0.0, 0.085, 0.0]));
    sphere(0.017, 20, 10).add_to(&mut m, red, &t([0.0, -0.035, 0.0]));
    sphere(0.024, 24, 12).add_to(&mut m, orange, &t([0.0, -0.088, 0.0]));
    m
}

/// The coaching arrow outline pointing at +X (tip at x = 0.105).
pub fn arrow_polygon() -> Vec<[f32; 2]> {
    vec![
        [-0.105, -0.022],
        [0.02, -0.022],
        [0.02, -0.055],
        [0.105, 0.0],
        [0.02, 0.055],
        [0.02, 0.022],
        [-0.105, 0.022],
    ]
}

/// Thick extruded arrow with a white outline plate behind it, rotated to point at `angle_deg`.
fn arrow(angle_deg: f32) -> MeshData {
    let mut m = MeshData::new();
    let body = m.add_material(mat("arrow", [1.0, 0.32, 0.08], 0.3));
    let outline = m.add_material(mat("arrow_outline", [1.0, 1.0, 1.0], 0.25));
    let rot = rz(angle_deg.to_radians());
    let poly = arrow_polygon();
    extrude(&poly, 0.04, 0.006).add_to(&mut m, body, &rot);
    extrude(&offset_polygon(&poly, 0.009), 0.024, 0.004).add_to(&mut m, outline, &compose(&[rot, t([0.0, 0.0, -0.012])]));
    m
}

/// Three light-blue teardrops (panic sweat).
fn sweat_drops() -> MeshData {
    let mut m = MeshData::new();
    let water = m.add_material(Material::new("sweat", [0.55, 0.82, 1.0, 1.0]).with_emissive([0.05, 0.12, 0.2]));
    let shine = m.add_material(mat("sweat_shine", [1.0, 1.0, 1.0], 0.6));
    let drop = teardrop(0.1, 0.07, 24);
    for (p, s, a) in [
        ([0.03, -0.015, 0.0], 1.0, -0.35),
        ([-0.065, 0.05, 0.0], 0.6, 0.3),
        ([0.088, 0.078, 0.0], 0.4, -0.5),
    ] {
        let xf = compose(&[t(p), rz(a), math::scaling([s, s, s * 0.85])]);
        drop.add_to(&mut m, water, &xf);
        sphere(0.008, 12, 6).add_to(&mut m, shine, &compose(&[xf, t([-0.012, -0.022, 0.026])]));
    }
    m
}

/// One big puffy heart, or (`pair`) two smaller tilted hearts for heart eyes.
fn hearts(pair: bool) -> MeshData {
    let mut m = MeshData::new();
    let pink = m.add_material(mat("heart_pink", [1.0, 0.38, 0.6], 0.25));
    if pair {
        let h = extrude(&heart_polygon(0.1, 72), 0.035, 0.009);
        for s in [-1.0f32, 1.0] {
            h.add_to(&mut m, pink, &compose(&[t([s * 0.062, 0.0, 0.0]), rz(-s * 0.25)]));
        }
    } else {
        extrude(&heart_polygon(0.2, 96), 0.06, 0.015).add_to(&mut m, pink, &math::IDENTITY);
    }
    m
}

/// ❓ — a tube along the hook curve plus a dot.
fn question_mark() -> MeshData {
    let mut m = MeshData::new();
    let blue = m.add_material(mat("question", [0.45, 0.38, 0.95], 0.2));
    let mut path = arc([0.0, 0.045, 0.0], 0.042, 160f32.to_radians(), -55f32.to_radians(), 22);
    let p0 = *path.last().expect("arc");
    let a = -55f32.to_radians();
    let tan = [a.sin(), -a.cos(), 0.0];
    let curve = bezier(
        [p0, math::add(p0, math::scale(tan, 0.02)), [0.0, -0.008, 0.0], [0.0, -0.035, 0.0]],
        10,
    );
    path.extend(curve.into_iter().skip(1));
    tube(&path, 0.017, 14, false).add_to(&mut m, blue, &math::IDENTITY);
    sphere(0.021, 20, 10).add_to(&mut m, blue, &t([0.0, -0.085, 0.0]));
    m
}

/// ♪ — eighth note: tilted oval head, stem and a curled flag.
fn music_note() -> MeshData {
    let mut m = MeshData::new();
    let ink = m.add_material(mat("music_note", [0.30, 0.25, 0.75], 0.2));
    sphere(0.03, 24, 12).add_to(
        &mut m,
        ink,
        &compose(&[t([-0.035, -0.075, 0.0]), rz(0.4), math::scaling([1.3, 0.95, 0.8])]),
    );
    cylinder(0.008, 0.008, 0.155, 12, true).add_to(&mut m, ink, &t([0.002, 0.0075, 0.0]));
    let flag = bezier(
        [[0.002, 0.083, 0.0], [0.04, 0.07, 0.0], [0.075, 0.03, 0.0], [0.045, -0.008, 0.0]],
        16,
    );
    let radii: Vec<f32> = (0..flag.len())
        .map(|i| 0.011 - 0.005 * i as f32 / (flag.len() - 1) as f32)
        .collect();
    tube_var(&flag, &radii, 12, false).add_to(&mut m, ink, &math::IDENTITY);
    m
}

/// A puff of relief: a cluster of soft white spheres trailing off to the right.
fn relief_puff() -> MeshData {
    let mut m = MeshData::new();
    let white = m.add_material(mat("puff", [0.95, 0.97, 1.0], 0.15));
    for (p, r) in [
        ([-0.06, 0.0, 0.0], 0.04),
        ([-0.015, 0.02, 0.0], 0.05),
        ([0.04, 0.005, 0.0], 0.043),
        ([-0.02, -0.03, 0.01], 0.038),
        ([0.09, 0.03, 0.0], 0.022),
        ([0.112, 0.056, 0.0], 0.012),
    ] {
        sphere(r, 24, 12).add_to(&mut m, white, &t(p));
    }
    m
}

// ---------------------------------------------------------------------------------------------------------
// Accessory parts
// ---------------------------------------------------------------------------------------------------------

/// An accessory placed around a head of radius ~0.5 centred at the origin (face toward +Z).
pub fn part_mesh(part: AvatarPart) -> MeshData {
    match part {
        AvatarPart::Headset => headset(),
        AvatarPart::Glasses => glasses(),
        AvatarPart::CatEars => cat_ears(),
        AvatarPart::Ribbon => ribbon(),
    }
}

fn headset() -> MeshData {
    let mut m = MeshData::new();
    let band = m.add_material(mat("headset_band", [0.15, 0.15, 0.18], 0.0));
    let cup = m.add_material(mat("headset_cup", ZENA_PURPLE, 0.0));
    let cushion = m.add_material(mat("headset_cushion", [0.08, 0.08, 0.1], 0.0));
    let led = m.add_material(mat("headset_led", ZENA_VIOLET, 0.9));
    torus_segment(0.545, 0.028, 0.12, PI - 0.12, 40, 12).add_to(&mut m, band, &math::IDENTITY);
    let to_x = rz(FRAC_PI_2);
    for s in [-1.0f32, 1.0] {
        cylinder(0.13, 0.13, 0.09, 32, true).add_to(&mut m, cup, &compose(&[t([s * 0.565, 0.0, 0.0]), to_x]));
        cylinder(0.11, 0.115, 0.035, 32, true).add_to(&mut m, cushion, &compose(&[t([s * 0.505, 0.0, 0.0]), to_x]));
        cylinder(0.085, 0.085, 0.02, 32, true).add_to(&mut m, led, &compose(&[t([s * 0.615, 0.0, 0.0]), to_x]));
    }
    let boom = bezier(
        [[-0.6, -0.05, 0.04], [-0.58, -0.22, 0.25], [-0.4, -0.31, 0.42], [-0.2, -0.3, 0.48]],
        20,
    );
    tube(&boom, 0.012, 10, false).add_to(&mut m, band, &math::IDENTITY);
    sphere(0.03, 16, 8).add_to(&mut m, cushion, &t([-0.18, -0.3, 0.485]));
    m
}

fn glasses() -> MeshData {
    let mut m = MeshData::new();
    let frame = m.add_material(mat("glasses_frame", [0.16, 0.12, 0.22], 0.0));
    let lens = m.add_material(Material::new("glasses_lens", [0.7, 0.85, 1.0, 0.3]).blended());
    let face_z = math::rotation([1.0, 0.0, 0.0], FRAC_PI_2);
    for s in [-1.0f32, 1.0] {
        let c = [s * 0.18, 0.04, 0.5];
        torus_segment(0.12, 0.014, 0.0, TAU, 40, 10).add_to(&mut m, frame, &t(c));
        cylinder(0.112, 0.112, 0.006, 40, true).add_to(&mut m, lens, &compose(&[t(c), face_z]));
        let temple = bezier(
            [
                [s * 0.3, 0.06, 0.5],
                [s * 0.42, 0.065, 0.42],
                [s * 0.49, 0.06, 0.25],
                [s * 0.5, 0.05, 0.0],
            ],
            14,
        );
        tube(&temple, 0.011, 8, false).add_to(&mut m, frame, &math::IDENTITY);
    }
    let bridge = bezier(
        [[-0.062, 0.06, 0.5], [-0.03, 0.085, 0.51], [0.03, 0.085, 0.51], [0.062, 0.06, 0.5]],
        10,
    );
    tube(&bridge, 0.012, 8, false).add_to(&mut m, frame, &math::IDENTITY);
    m
}

fn cat_ears() -> MeshData {
    let mut m = MeshData::new();
    let fur = m.add_material(mat("cat_ear_fur", ZENA_PURPLE, 0.0));
    let inner = m.add_material(mat("cat_ear_inner", [1.0, 0.68, 0.78], 0.05));
    let outer = extrude(&rounded_polygon(&[[-0.16, 0.0], [0.16, 0.0], [0.025, 0.33]], 0.035, 5), 0.09, 0.025);
    let pink = extrude(
        &rounded_polygon(&[[-0.095, 0.045], [0.095, 0.045], [0.018, 0.24]], 0.025, 4),
        0.02,
        0.005,
    );
    for s in [-1.0f32, 1.0] {
        let xf = compose(&[
            t([s * 0.26, 0.36, 0.0]),
            math::rotation([0.0, 1.0, 0.0], s * 0.3),
            rz(-s * 0.35),
            math::scaling([s, 1.0, 1.0]),
        ]);
        outer.add_to(&mut m, fur, &xf);
        pink.add_to(&mut m, inner, &compose(&[xf, t([0.0, 0.0, 0.05])]));
    }
    m
}

fn ribbon() -> MeshData {
    let mut m = MeshData::new();
    let cloth = m.add_material(mat("ribbon", [0.92, 0.30, 0.52], 0.05));
    let knot = m.add_material(mat("ribbon_knot", [0.78, 0.2, 0.42], 0.05));
    let dir = math::normalize([0.55, 0.78, 0.25]);
    let anchor = math::scale(dir, 0.52);
    let place = compose(&[
        t(anchor),
        math::rotation([0.0, 1.0, 0.0], 0.35),
        rz(-0.55),
        math::scaling([1.5, 1.5, 1.5]),
    ]);
    let wing = extrude(
        &rounded_polygon(&[[0.0, 0.025], [0.17, 0.095], [0.17, -0.085], [0.0, -0.025]], 0.03, 5),
        0.045,
        0.012,
    );
    let tail = extrude(
        &rounded_polygon(&[[-0.02, -0.02], [0.03, -0.02], [0.07, -0.16], [0.03, -0.15]], 0.01, 3),
        0.02,
        0.004,
    );
    for s in [-1.0f32, 1.0] {
        let mirror = math::scaling([s, 1.0, 1.0]);
        wing.add_to(&mut m, cloth, &compose(&[place, mirror, rz(0.12)]));
        tail.add_to(&mut m, cloth, &compose(&[place, t([0.0, 0.0, -0.01]), mirror]));
    }
    sphere(0.045, 20, 10).add_to(&mut m, knot, &compose(&[place, math::scaling([1.0, 1.0, 0.8])]));
    m
}

// ---------------------------------------------------------------------------------------------------------
// Robot Zena (Track A body)
// ---------------------------------------------------------------------------------------------------------

/// VRM 1.0 preset expression names carried by [`robot_zena`]'s morph targets, in target order.
pub const ROBOT_MORPHS: [&str; 11] = [
    "blink",
    "happy",
    "angry",
    "sad",
    "surprised",
    "relaxed",
    "aa",
    "ih",
    "ou",
    "ee",
    "oh",
];

const STRIP_COLS: usize = 13;
const FACE_Z: (f32, f32) = (0.448, 0.47);
const EYE_X: f32 = 0.19;
const EYE_Y: f32 = 0.03;
const MOUTH_Y: f32 = -0.17;

/// One face feature as `(x, y_bottom, y_top)` columns, left → right.
type Strip = Vec<(f32, f32, f32)>;

fn strip(half_w: f32, cx: f32, cy: f32, f: impl Fn(f32) -> (f32, f32)) -> Strip {
    (0..STRIP_COLS)
        .map(|i| {
            let u = -1.0 + 2.0 * i as f32 / (STRIP_COLS - 1) as f32;
            let (b, tp) = f(u);
            (cx + half_w * u, cy + b, cy + tp.max(b))
        })
        .collect()
}

/// A slab between `z0` and `z1` over a strip; the topology depends only on the column count, so every
/// expression variant lines up vertex for vertex.
fn strip_geo(cols: &Strip, z0: f32, z1: f32) -> Geo {
    let m = cols.len() as u32;
    let mut g = Geo::default();
    for (z, nz) in [(z1, 1.0f32), (z0, -1.0)] {
        let base = g.positions.len() as u32;
        for &(x, b, tp) in cols {
            g.vertex([x, b, z], [0.0, 0.0, nz], [0.0, 0.0]);
            g.vertex([x, tp, z], [0.0, 0.0, nz], [0.0, 1.0]);
        }
        for i in 0..m - 1 {
            let (b0, t0, b1, t1) = (base + 2 * i, base + 2 * i + 1, base + 2 * i + 2, base + 2 * i + 3);
            if nz > 0.0 {
                g.quad(b0, b1, t1, t0);
            } else {
                g.quad(b0, t0, t1, b1);
            }
        }
    }
    // Counter-clockwise outline: bottom left → right, then top right → left.
    let mut outline: Vec<[f32; 2]> = cols.iter().map(|c| [c.0, c.1]).collect();
    outline.extend(cols.iter().rev().map(|c| [c.0, c.2]));
    let k = outline.len();
    let cx = cols.iter().map(|c| c.0).sum::<f32>() / cols.len() as f32;
    for e in 0..k {
        let (p, q) = (outline[e], outline[(e + 1) % k]);
        let n = if (q[0] - p[0]).abs() + (q[1] - p[1]).abs() < 1e-9 {
            [(p[0] - cx).signum(), 0.0, 0.0]
        } else {
            let n2 = edge_normal(p, q);
            [n2[0], n2[1], 0.0]
        };
        let a = g.vertex([p[0], p[1], z0], n, [0.0, 0.0]);
        let b = g.vertex([q[0], q[1], z0], n, [1.0, 0.0]);
        let c = g.vertex([q[0], q[1], z1], n, [1.0, 1.0]);
        let d = g.vertex([p[0], p[1], z1], n, [0.0, 1.0]);
        g.quad(a, b, c, d);
    }
    g
}

fn ellipse_top(u: f32) -> f32 {
    (1.0 - u * u).max(0.0).sqrt()
}

/// Eye shape for an expression; `side` is +1 for the viewer-left eye (its inner corner is +X), −1 otherwise.
fn eye_strip(expr: &str, side: f32) -> Strip {
    let cx = -side * EYE_X;
    let (w, h) = (0.065, 0.095);
    match expr {
        "blink" => strip(w * 1.05, cx, EYE_Y - 0.03, |u| (-0.012 * ellipse_top(u), 0.012 * ellipse_top(u))),
        "happy" => strip(w * 1.1, cx, EYE_Y, |u| {
            let c = -0.03 + 0.07 * (1.0 - u * u);
            let th = 0.022 * ellipse_top(u) + 0.004;
            (c - th, c + th)
        }),
        "angry" | "sad" => {
            let tilt = if expr == "angry" { -0.45 } else { 0.45 };
            strip(w, cx, if expr == "sad" { EYE_Y - 0.01 } else { EYE_Y }, |u| {
                let v = u * side;
                let e = h * ellipse_top(u);
                let lid = h * (0.5 + tilt * v);
                (-e, e.min(lid).max(-e + 0.006 * ellipse_top(u)))
            })
        }
        "surprised" => strip(w * 1.12, cx, EYE_Y + 0.005, |u| {
            (-1.12 * h * ellipse_top(u), 1.12 * h * ellipse_top(u))
        }),
        "relaxed" => strip(w, cx, EYE_Y, |u| {
            let e = h * ellipse_top(u);
            (-0.9 * e, e.min(0.2 * h))
        }),
        _ => strip(w, cx, EYE_Y, |u| (-h * ellipse_top(u), h * ellipse_top(u))),
    }
}

fn mouth_strip(expr: &str) -> Strip {
    let line = |half_w: f32, curve: f32, th: f32| {
        strip(half_w, 0.0, MOUTH_Y, move |u| {
            let c = curve * (u * u - 0.5);
            let t = th * ellipse_top(u) + 0.004;
            (c - t, c + t)
        })
    };
    let oval = |half_w: f32, top: f32, bottom: f32| strip(half_w, 0.0, MOUTH_Y, move |u| (-bottom * ellipse_top(u), top * ellipse_top(u)));
    match expr {
        "happy" => line(0.08, 0.05, 0.012),
        "angry" => line(0.07, -0.04, 0.01),
        "sad" => line(0.06, -0.035, 0.009),
        "surprised" => oval(0.04, 0.05, 0.05),
        "relaxed" => line(0.06, 0.025, 0.008),
        "aa" => oval(0.065, 0.03, 0.075),
        "ih" => oval(0.09, 0.018, 0.018),
        "ou" => oval(0.03, 0.035, 0.035),
        "ee" => oval(0.085, 0.025, 0.035),
        "oh" => oval(0.045, 0.045, 0.05),
        _ => line(0.05, 0.0, 0.008),
    }
}

fn face_geo(expr: &str) -> Geo {
    let mut g = strip_geo(&eye_strip(expr, 1.0), FACE_Z.0, FACE_Z.1);
    g.merge(&strip_geo(&eye_strip(expr, -1.0), FACE_Z.0, FACE_Z.1));
    g.merge(&strip_geo(&mouth_strip(expr), FACE_Z.0, FACE_Z.1));
    g
}

/// Track A body: an SD toy-robot Zena (head ~1 m wide centred at the origin, so [`part_mesh`] accessories fit),
/// with a dark visor, glowing violet eyes and mouth carrying the [`ROBOT_MORPHS`] morph targets, an antenna,
/// and a small body with arms and legs below. Unskinned.
pub fn robot_zena() -> MeshData {
    let mut m = MeshData::new();
    let shell = m.add_material(mat("robot_shell", [0.93, 0.93, 0.96], 0.0));
    let accent = m.add_material(mat("robot_accent", ZENA_PURPLE, 0.0));
    let visor = m.add_material(mat("robot_visor", [0.05, 0.04, 0.09], 0.0));
    let glow = m.add_material(
        Material::new("robot_eye_glow", [ZENA_VIOLET[0], ZENA_VIOLET[1], ZENA_VIOLET[2], 1.0]).with_emissive([0.55, 0.35, 0.95]),
    );
    let joint = m.add_material(mat("robot_joint", [0.55, 0.55, 0.62], 0.0));
    let blush = m.add_material(mat("robot_blush", [1.0, 0.5, 0.7], 0.35));
    let to_x = rz(FRAC_PI_2);
    let to_z = math::rotation([1.0, 0.0, 0.0], FRAC_PI_2);

    // Head.
    rounded_box([1.0, 0.82, 0.86], 0.2, 6).add_to(&mut m, shell, &math::IDENTITY);
    for s in [-1.0f32, 1.0] {
        cylinder(0.15, 0.15, 0.1, 32, true).add_to(&mut m, accent, &compose(&[t([s * 0.53, -0.02, 0.0]), to_x]));
        cylinder(0.07, 0.07, 0.03, 24, true).add_to(&mut m, glow, &compose(&[t([s * 0.59, -0.02, 0.0]), to_x]));
    }
    cylinder(0.06, 0.05, 0.05, 24, true).add_to(&mut m, joint, &t([0.0, 0.425, 0.0]));
    cylinder(0.018, 0.018, 0.2, 12, false).add_to(&mut m, joint, &t([0.0, 0.53, 0.0]));
    sphere(0.055, 20, 10).add_to(&mut m, glow, &t([0.0, 0.64, 0.0]));
    rounded_box([0.8, 0.52, 0.08], 0.07, 4).add_to(&mut m, visor, &t([0.0, -0.03, 0.41]));
    for s in [-1.0f32, 1.0] {
        cylinder(0.045, 0.045, 0.008, 24, true).add_to(
            &mut m,
            blush,
            &compose(&[t([s * 0.3, -0.13, 0.452]), math::scaling([1.0, 0.6, 1.0]), to_z]),
        );
    }

    // Body.
    cylinder(0.09, 0.09, 0.12, 24, false).add_to(&mut m, joint, &t([0.0, -0.46, 0.0]));
    rounded_box([0.56, 0.44, 0.42], 0.14, 5).add_to(&mut m, shell, &t([0.0, -0.72, 0.0]));
    rounded_box([0.26, 0.16, 0.04], 0.035, 3).add_to(&mut m, accent, &t([0.0, -0.66, 0.205]));
    extrude(&heart_polygon(0.09, 48), 0.02, 0.005).add_to(&mut m, glow, &t([0.0, -0.66, 0.23]));
    for s in [-1.0f32, 1.0] {
        sphere(0.075, 20, 10).add_to(&mut m, joint, &t([s * 0.31, -0.58, 0.0]));
        rounded_box([0.13, 0.32, 0.14], 0.06, 4).add_to(&mut m, shell, &compose(&[t([s * 0.36, -0.72, 0.0]), rz(s * 0.35)]));
        sphere(0.085, 20, 10).add_to(&mut m, accent, &t([s * 0.43, -0.9, 0.0]));
        rounded_box([0.16, 0.18, 0.18], 0.06, 4).add_to(&mut m, joint, &t([s * 0.13, -1.0, 0.0]));
        rounded_box([0.22, 0.09, 0.3], 0.045, 4).add_to(&mut m, accent, &t([s * 0.14, -1.11, 0.04]));
    }

    m.append(&robot_face(), math::IDENTITY);
    m
}

/// Just the robot's glowing eyes and mouth (on the visor front, z ≈ 0.45–0.47) with the [`ROBOT_MORPHS`]
/// targets — grafted onto SolidCraft-made bodies, which carry no morph targets.
pub fn robot_face() -> MeshData {
    let mut m = MeshData::new();
    let glow = m.add_material(
        Material::new("robot_face_glow", [ZENA_VIOLET[0], ZENA_VIOLET[1], ZENA_VIOLET[2], 1.0]).with_emissive([0.55, 0.35, 0.95]),
    );
    let base = face_geo("neutral");
    let first = m.push_triangles(glow, &base.positions, &base.normals, &base.uvs, &base.indices);
    for (k, name) in ROBOT_MORPHS.iter().enumerate() {
        let v = face_geo(name);
        let deltas = v
            .positions
            .iter()
            .zip(&base.positions)
            .enumerate()
            .filter_map(|(i, (a, b))| {
                let d = math::sub(*a, *b);
                (math::dot(d, d) > 1e-12).then_some((first + i as u32, d))
            })
            .collect();
        m.morphs.push(MorphTarget {
            name: (*name).to_string(),
            mesh: 0,
            target: k,
            deltas,
        });
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{load_glb, write_glb};

    fn check(m: &MeshData, half_extent: f32, what: &str) {
        m.validate().unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(m.vertex_count() > 0 && m.triangle_count() > 0, "{what} is empty");
        let (mn, mx) = m.bounds();
        for k in 0..3 {
            assert!(
                mn[k] >= -half_extent && mx[k] <= half_extent,
                "{what} bounds {mn:?}..{mx:?} exceed ±{half_extent}"
            );
        }
        assert!(
            m.normals.iter().all(|n| (math::length(*n) - 1.0).abs() < 1e-3),
            "{what} has non-unit normals"
        );
    }

    /// Fraction of triangles whose winding agrees with their vertex normals (outward-facing).
    fn outward_fraction(m: &MeshData) -> f32 {
        let mut good = 0;
        for t in m.indices.chunks_exact(3) {
            let p = |i: u32| m.positions[i as usize];
            let g = math::cross(math::sub(p(t[1]), p(t[0])), math::sub(p(t[2]), p(t[0])));
            if math::dot(g, m.normals[t[0] as usize]) >= 0.0 {
                good += 1;
            }
        }
        good as f32 / m.triangle_count() as f32
    }

    #[test]
    fn builders_produce_closed_outward_geometry() {
        let mut m = MeshData::new();
        let mat = m.add_material(Material::default());
        for g in [
            sphere(1.0, 16, 8),
            rounded_box([1.0, 2.0, 0.5], 0.1, 3),
            cylinder(0.5, 0.2, 1.0, 16, true),
            torus_segment(1.0, 0.2, 0.0, PI, 16, 8),
            lathe(&[[0.0, 0.0], [1.0, 0.5], [0.0, 1.0]], 12),
            extrude(&heart_polygon(1.0, 48), 0.3, 0.05),
            extrude(&arrow_polygon(), 0.04, 0.0),
        ] {
            let mut one = MeshData::new();
            let k = one.add_material(Material::default());
            g.add_to(&mut one, k, &math::IDENTITY);
            assert!(outward_fraction(&one) > 0.999);
            g.add_to(&mut m, mat, &math::IDENTITY);
        }
        m.validate().unwrap();
        let (mn, mx) = rounded_box([1.0, 2.0, 0.5], 0.1, 3)
            .positions
            .iter()
            .fold(([f32::MAX; 3], [f32::MIN; 3]), |(a, b), p| {
                (std::array::from_fn(|k| a[k].min(p[k])), std::array::from_fn(|k| b[k].max(p[k])))
            });
        assert!((mx[1] - 1.0).abs() < 1e-5 && (mn[2] + 0.25).abs() < 1e-5);
        assert_eq!(triangulate(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.2], [0.0, 1.0]]).len(), 3);
    }

    #[test]
    fn every_prop_part_and_body_is_valid() {
        let mut props = 0;
        for e in CharacterEmote::ALL {
            if let Some(m) = emote_prop(e, 30.0) {
                check(&m, 0.14, e.name());
                props += 1;
            }
        }
        assert_eq!(props, 10);
        for e in [
            CharacterEmote::ComicAnger3d,
            CharacterEmote::SparkleIdea,
            CharacterEmote::Exclaim3d,
            CharacterEmote::ThreatArrow3d,
        ] {
            assert!(emote_prop(e, 0.0).is_some(), "every Prop-form emote has a prop");
        }
        assert!(emote_prop(CharacterEmote::ChibiCheer, 0.0).is_none());
        for p in AvatarPart::ALL {
            let m = part_mesh(p);
            check(&m, 0.75, p.name());
        }
        let r = robot_zena();
        check(&r, 1.25, "robot");
        assert!(outward_fraction(&r) > 0.99);
    }

    #[test]
    fn arrow_points_along_angle() {
        for angle in [0.0f32, 45.0, 90.0, 180.0, 270.0, 333.0] {
            let m = emote_prop(CharacterEmote::ThreatArrow3d, angle).unwrap();
            let d = [angle.to_radians().cos(), angle.to_radians().sin(), 0.0];
            let tip = m
                .positions
                .iter()
                .copied()
                .max_by(|a, b| math::dot(*a, d).total_cmp(&math::dot(*b, d)))
                .unwrap();
            let tip_angle = tip[1].atan2(tip[0]).to_degrees().rem_euclid(360.0);
            let diff = (tip_angle - angle.rem_euclid(360.0) + 540.0).rem_euclid(360.0) - 180.0;
            assert!(diff.abs() < 3.0, "angle {angle}: tip at {tip_angle}");
            assert!(math::dot(tip, d) > 0.1);
            let tail = m.positions.iter().map(|p| math::dot(*p, d)).fold(f32::MAX, f32::min);
            assert!(tail < -0.09, "the shaft trails behind");
        }
    }

    #[test]
    fn robot_has_vrm_morphs_that_move_the_face() {
        let r = robot_zena();
        for name in ROBOT_MORPHS {
            let i = r.morph_index(name).unwrap_or_else(|| panic!("missing {name}"));
            assert!(!r.morphs[i].deltas.is_empty(), "{name} moves nothing");
        }
        // Blink squashes the eyes: their vertical extent shrinks.
        let eye_extent = |p: &[[f32; 3]]| {
            let ys: Vec<f32> = p
                .iter()
                .filter(|v| v[2] > 0.46 && v[1] > -0.1 && v[0] > 0.1 && v[0] < 0.3)
                .map(|v| v[1])
                .collect();
            ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min)
        };
        let mut out = Vec::new();
        r.apply_morphs(&[(r.morph_index("blink").unwrap(), 1.0)], &mut out);
        assert!(eye_extent(&out) < 0.4 * eye_extent(&r.positions));
        // aa opens the mouth.
        let mouth_h = |p: &[[f32; 3]]| {
            let ys: Vec<f32> = p
                .iter()
                .filter(|v| v[2] > 0.46 && v[1] < -0.1 && v[0].abs() < 0.1)
                .map(|v| v[1])
                .collect();
            ys.iter().copied().fold(f32::MIN, f32::max) - ys.iter().copied().fold(f32::MAX, f32::min)
        };
        r.apply_morphs(&[(r.morph_index("aa").unwrap(), 1.0)], &mut out);
        assert!(mouth_h(&out) > 3.0 * mouth_h(&r.positions));
    }

    #[test]
    fn glb_round_trip_preserves_vertices_and_morph_names() {
        let mut meshes = vec![("robot".to_string(), robot_zena())];
        meshes.extend(AvatarPart::ALL.map(|p| (p.name().to_string(), part_mesh(p))));
        meshes.extend(
            CharacterEmote::ALL
                .iter()
                .filter_map(|e| emote_prop(*e, 45.0).map(|m| (e.name().to_string(), m))),
        );
        for (name, m) in meshes {
            let back = load_glb(&write_glb(&m)).unwrap_or_else(|e| panic!("{name}: {e}"));
            back.validate().unwrap();
            assert_eq!(back.vertex_count(), m.vertex_count(), "{name}");
            assert_eq!(back.triangle_count(), m.triangle_count(), "{name}");
            assert_eq!(back.materials.len(), m.materials.len(), "{name}");
            let names: Vec<&str> = back.morphs.iter().map(|x| x.name.as_str()).collect();
            let want: Vec<&str> = m.morphs.iter().map(|x| x.name.as_str()).collect();
            assert_eq!(names, want, "{name}");
            for (a, b) in back.morphs.iter().zip(&m.morphs) {
                assert_eq!(a.deltas.len(), b.deltas.len(), "{name}/{}", a.name);
            }
        }
    }
}
