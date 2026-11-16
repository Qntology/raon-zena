//! Visual sanity check: writes every chibi atlas, stamp contact sheets and software-rasterised renders of every
//! forge mesh to `<target dir>/preview/` (e.g. `target/mcp-agent/preview/`).

use std::path::{Path, PathBuf};

use image::{Rgba, RgbaImage};
use zena_avatar_mcp::chibi::ChibiKit;
use zena_avatar_mcp::mesh::{load_glb, write_glb, AlphaMode, Material, MeshData, IDENTITY};
use zena_avatar_mcp::{forge, AvatarPart, CharacterEmote};

fn preview_dir() -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).parent().expect("target dir").join("preview");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn checker(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        if (x / 10 + y / 10) % 2 == 0 {
            Rgba([200, 204, 210, 255])
        } else {
            Rgba([170, 175, 182, 255])
        }
    })
}

fn grid(images: &[RgbaImage], cols: u32) -> RgbaImage {
    let (w, h) = (images[0].width(), images[0].height());
    let rows = (images.len() as u32).div_ceil(cols);
    let mut sheet = checker(cols * w, rows * h);
    for (i, img) in images.iter().enumerate() {
        let (cx, cy) = ((i as u32 % cols) * w, (i as u32 / cols) * h);
        image::imageops::overlay(&mut sheet, img, cx as i64, cy as i64);
        for k in 0..w {
            sheet.put_pixel(cx + k, cy, Rgba([120, 120, 130, 255]));
        }
        for k in 0..h {
            sheet.put_pixel(cx, cy + k, Rgba([120, 120, 130, 255]));
        }
    }
    sheet
}

/// A tiny z-buffered rasteriser: yaw/pitch orbit, Lambert + ambient + emissive, back-face culling for
/// single-sided materials (so inverted triangles show up as holes), alpha-blended materials drawn last.
fn render(mesh: &MeshData, morphs: &[(usize, f32)], size: u32, yaw: f32, pitch: f32) -> RgbaImage {
    let mut pos = Vec::new();
    mesh.apply_morphs(morphs, &mut pos);
    let (mn, mx) = mesh.bounds();
    let c = [(mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5];
    let radius = pos
        .iter()
        .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt())
        .fold(1e-6f32, f32::max);
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let rot = |p: [f32; 3]| -> [f32; 3] {
        let (x, y, z) = (cy * p[0] + sy * p[2], p[1], -sy * p[0] + cy * p[2]);
        [x, cp * y - sp * z, sp * y + cp * z]
    };
    let scale = size as f32 * 0.46 / radius;
    let view: Vec<[f32; 3]> = pos.iter().map(|p| rot([p[0] - c[0], p[1] - c[1], p[2] - c[2]])).collect();
    let scr: Vec<[f32; 3]> = view
        .iter()
        .map(|v| [size as f32 * 0.5 + v[0] * scale, size as f32 * 0.5 - v[1] * scale, v[2]])
        .collect();
    let nrm: Vec<[f32; 3]> = mesh.normals.iter().map(|n| rot(*n)).collect();
    let l = {
        let v = [-0.4f32, 0.6, 0.7];
        let k = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / k, v[1] / k, v[2] / k]
    };
    let n = size as usize;
    let mut z = vec![f32::MIN; n * n];
    let mut img = RgbaImage::from_pixel(size, size, Rgba([0, 0, 0, 0]));
    let mut col = vec![[0.0f32; 4]; n * n];
    for pass_blend in [false, true] {
        for prim in &mesh.primitives {
            let m: &Material = &mesh.materials[prim.material];
            if (m.alpha_mode == AlphaMode::Blend) != pass_blend {
                continue;
            }
            let idx = &mesh.indices[prim.first_index as usize..(prim.first_index + prim.index_count) as usize];
            for t in idx.chunks_exact(3) {
                let (a, b, cc) = (scr[t[0] as usize], scr[t[1] as usize], scr[t[2] as usize]);
                let area = (b[0] - a[0]) * (cc[1] - a[1]) - (b[1] - a[1]) * (cc[0] - a[0]);
                if area.abs() < 1e-9 || (!m.double_sided && area > 0.0) {
                    continue;
                }
                let x0 = a[0].min(b[0]).min(cc[0]).floor().max(0.0) as usize;
                let x1 = (a[0].max(b[0]).max(cc[0]).ceil().max(0.0) as usize).min(n);
                let y0 = a[1].min(b[1]).min(cc[1]).floor().max(0.0) as usize;
                let y1 = (a[1].max(b[1]).max(cc[1]).ceil().max(0.0) as usize).min(n);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                        let w0 = ((b[0] - fx) * (cc[1] - fy) - (b[1] - fy) * (cc[0] - fx)) / area;
                        let w1 = ((cc[0] - fx) * (a[1] - fy) - (cc[1] - fy) * (a[0] - fx)) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let depth = w0 * a[2] + w1 * b[2] + w2 * cc[2];
                        let k = py * n + px;
                        if depth <= z[k] {
                            continue;
                        }
                        let nn: [f32; 3] =
                            std::array::from_fn(|i| w0 * nrm[t[0] as usize][i] + w1 * nrm[t[1] as usize][i] + w2 * nrm[t[2] as usize][i]);
                        let len = (nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]).sqrt().max(1e-6);
                        let mut d = (nn[0] * l[0] + nn[1] * l[1] + nn[2] * l[2]) / len;
                        if area > 0.0 {
                            d = -d;
                        }
                        let shade = 0.32 + 0.68 * d.max(0.0);
                        let rgb: [f32; 3] = std::array::from_fn(|i| (m.base_color[i] * shade + m.emissive[i] * 0.6).min(1.0));
                        if pass_blend {
                            let al = m.base_color[3];
                            for i in 0..3 {
                                col[k][i] = rgb[i] * al + col[k][i] * (1.0 - al);
                            }
                            col[k][3] = al + col[k][3] * (1.0 - al);
                        } else {
                            z[k] = depth;
                            col[k] = [rgb[0], rgb[1], rgb[2], 1.0];
                        }
                    }
                }
            }
        }
    }
    for (k, c) in col.iter().enumerate() {
        if c[3] > 0.0 {
            let q = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 1.4) * 255.0) as u8;
            img.put_pixel(
                (k % n) as u32,
                (k / n) as u32,
                Rgba([q(c[0] / c[3].max(1e-6) * c[3].min(1.0)), q(c[1]), q(c[2]), (c[3] * 255.0) as u8]),
            );
        }
    }
    img
}

fn with_head(part: &MeshData) -> MeshData {
    let mut m = MeshData::new();
    let skin = m.add_material(Material::new("head", [0.85, 0.82, 0.8, 1.0]));
    forge::sphere(0.47, 40, 20).add_to(&mut m, skin, &IDENTITY);
    m.append(part, IDENTITY);
    m
}

#[test]
fn write_previews() {
    let dir = preview_dir();
    let kit = ChibiKit::builtin();
    for e in CharacterEmote::ALL {
        kit.write_atlas(e, 8, &dir.join("atlases")).unwrap();
    }
    let mut stamps: Vec<RgbaImage> = CharacterEmote::ALL.iter().map(|e| kit.compose(*e, 0, 8)).collect();
    stamps.push(kit.neutral());
    grid(&stamps, 5).save(dir.join("chibi_contact_sheet.png")).unwrap();
    let mut frames = Vec::new();
    for e in [
        CharacterEmote::ChibiCheer,
        CharacterEmote::ComicAnger3d,
        CharacterEmote::Tears,
        CharacterEmote::ChibiHeart,
    ] {
        frames.extend((0..8).map(|f| kit.compose(e, f, 8)));
    }
    grid(&frames, 8).save(dir.join("chibi_frames.png")).unwrap();

    // 3D: every prop (plus extra arrow angles), every part on a head, the robot and its morphs — rendered from
    // the glb round trip so the writer and reader are exercised too.
    let rt = |m: &MeshData| load_glb(&write_glb(m)).unwrap();
    let (yaw, pitch) = (0.45f32, 0.3f32);
    let mut tiles = Vec::new();
    for e in CharacterEmote::ALL {
        if let Some(m) = forge::emote_prop(e, 30.0) {
            tiles.push(render(&rt(&m), &[], 200, yaw, pitch));
        }
    }
    for a in [90.0f32, 225.0] {
        tiles.push(render(
            &rt(&forge::emote_prop(CharacterEmote::ThreatArrow3d, a).unwrap()),
            &[],
            200,
            0.0,
            0.0,
        ));
    }
    for p in AvatarPart::ALL {
        tiles.push(render(&with_head(&rt(&forge::part_mesh(p))), &[], 200, yaw, pitch));
    }
    let robot = rt(&forge::robot_zena());
    tiles.push(render(&robot, &[], 200, 0.0, 0.05));
    tiles.push(render(&robot, &[], 200, 0.7, 0.25));
    for name in ["happy", "blink", "aa", "angry", "surprised", "sad"] {
        let i = robot.morph_index(name).unwrap();
        let face_zoom = render(&robot, &[(i, 1.0)], 400, 0.0, 0.0);
        tiles.push(
            image::imageops::crop_imm(&face_zoom, 80, 40, 240, 240)
                .to_image()
                .pipe(|t| image::imageops::resize(&t, 200, 200, image::imageops::FilterType::Triangle)),
        );
    }
    grid(&tiles, 6).save(dir.join("forge_sheet.png")).unwrap();
    println!("previews written to {}", dir.display());
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}
