//! What the overlay draws each frame, in three passes on one device:
//!
//! 1. **Body** — the selected track: the procedural face, a mesh body (Track A robot `.glb`, Track B-2 `.vrm` with
//!    expression / viseme / blink morphs and head-look skinning) or the 2D layered body (Track B-1), plus the
//!    accessories it wears.
//! 2. **Emote & props** — Zena's own emote at her head: a 3D prop (anger mark, light bulb, coaching arrow pointing
//!    at the threat …), a comic face morph on a VRM, or her SD chibi stamp, popping in on a spring.
//! 3. **Speech bubble** — the current line (streamed sentences append) and, while typing, the chat box.
//!
//! Assets arrive from the asset service (decoded off the UI thread) through [`Scene::load`]; until a track's body
//! is loaded the procedural face stands in, and an emote without its prop falls back to its stamp.

use super::mesh::{GpuMesh, MeshPipeline, MeshPose};
use super::procedural::Procedural;
use super::sprite::{Placement, Sprite, SpritePipeline};
use super::text::{BoxStyle, TextRaster};
use super::{mul, ortho_frame, quat_yaw_pitch, rot_x, rot_y, scale, translate, FrameParams, Mat4, IDENTITY};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use zena_avatar_mcp::mesh::MeshData;
use zena_avatar_mcp::vrm::{self, VrmInfo, VrmVersion};
use zena_core::avatar::{to_vrm, VrmVersion as CoreVrm};
use zena_core::tools::{AvatarPart, AvatarTrack, CharacterEmote, EmoteForm, ZenaAvatarArgs};

/// Prop angles are bucketed so the coaching arrow is generated at most 24 times.
pub fn angle_bucket(angle_deg: f32) -> i32 {
    (angle_deg.rem_euclid(360.0) / 15.0).round() as i32 % 24 * 15
}

/// A decoded asset ready for upload.
pub enum Loaded {
    Stamp {
        emote: CharacterEmote,
        w: u32,
        h: u32,
        rgba: Vec<u8>,
        frames: u32,
    },
    Prop {
        emote: CharacterEmote,
        angle: i32,
        mesh: MeshData,
    },
    Part {
        part: AvatarPart,
        mesh: MeshData,
    },
    Body2d {
        w: u32,
        h: u32,
        rgba: Vec<u8>,
    },
    BodyMesh {
        track: AvatarTrack,
        mesh: MeshData,
        vrm: Option<VrmInfo>,
    },
}

/// An asset the scene would draw but does not have.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Missing {
    Stamp(CharacterEmote),
    Prop(CharacterEmote, i32),
    Part(AvatarPart),
    Body(AvatarTrack),
}

/// The emote on show.
#[derive(Debug, Clone, Copy)]
pub struct EmoteShow {
    pub emote: CharacterEmote,
    /// Seconds since it appeared.
    pub age: f32,
    /// 1 → 0 while it fades out.
    pub fade: f32,
    /// Where the coaching arrow points (gaze angle).
    pub angle_deg: f32,
}

use crate::browser_uia::NavBadge;
use crate::pd_bridge::DrawerItem;
use premium_discount_verdict::Tier;
use zena_verdict::UserPin;

/// Clickable `[▲]` / `[▼]` hit regions for one Stage 2 drawer row (window pixel coordinates).
#[derive(Debug, Clone)]
pub struct DrawerHitRow {
    pub record_id: String,
    pub up_rect: [f32; 4],
    pub down_rect: [f32; 4],
}

/// Interactive hit-test layout computed during the latest frame (window pixel coordinates `[x0, y0, x1, y1]`).
#[derive(Debug, Clone, Default)]
pub struct OverlayLayout {
    pub pill_rect: Option<[f32; 4]>,
    pub rejection_rect: Option<[f32; 4]>,
    pub drawer_rows: Vec<DrawerHitRow>,
    pub badge_rects: Vec<(u8, bool, [f32; 4])>,
}

impl OverlayLayout {
    pub fn hit_rect(r: [f32; 4], x: f32, y: f32) -> bool {
        x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]
    }
}

/// Everything one frame needs.
#[derive(Clone, Copy)]
pub struct Stage<'a> {
    pub params: FrameParams,
    pub track: AvatarTrack,
    pub parts: &'a [AvatarPart],
    pub emote: Option<EmoteShow>,
    /// The line on show and its opacity.
    pub bubble: Option<(&'a str, f32)>,
    /// The chat box: committed text, IME preedit, caret visible.
    pub input: Option<(&'a str, &'a str, bool)>,
    /// Stage 0 status pill: label text and status dot/border color.
    pub sync_pill: Option<(&'a str, [f32; 4])>,
    /// Stage 1 unacknowledged rejection alert chip.
    pub rejection_chip: Option<&'a str>,
    /// Stage 2 history/verdict drawer rows (non-blended local vs global + ▲/▼ pin controls).
    pub drawer: Option<&'a [DrawerItem]>,
    /// 1–9 numbered navigation badges for the active screen/browser tab.
    pub badges: &'a [NavBadge],
    /// Drop-note banner shown while authoring a note for a dragged/dropped target window.
    pub drop_hint: Option<&'a str>,
}

enum Body {
    Mesh { gpu: GpuMesh, vrm: Option<VrmInfo> },
    Sprite(Sprite),
}

struct Stamp {
    sprite: Sprite,
    frames: u32,
}

pub struct Scene {
    procedural: Procedural,
    sprites: SpritePipeline,
    meshes: MeshPipeline,
    text: TextRaster,
    stamps: HashMap<CharacterEmote, Stamp>,
    props: HashMap<(CharacterEmote, i32), GpuMesh>,
    parts: HashMap<AvatarPart, GpuMesh>,
    bodies: HashMap<AvatarTrack, Body>,
    bubble: Option<(String, Sprite)>,
    input: Option<(String, Sprite)>,
    pill: Option<(String, Sprite)>,
    rejection: Option<(String, Sprite)>,
    drawer_rows: Vec<(String, Sprite)>,
    badge_sprites: Vec<(String, Sprite)>,
    drop_hint: Option<(String, Sprite)>,
    layout: OverlayLayout,
    /// Assets that failed to load (not asked for again).
    pub failed: HashSet<Missing>,
}

/// Morph weights by VRM 1.0 expression name for a mesh body (`zena-avatar-mcp` maps VRM 0.x presets onto the 1.0
/// names, and the forge robot uses them too).
fn body_expressions(p: &FrameParams, emote: Option<&EmoteShow>) -> Vec<(String, f32)> {
    let args = ZenaAvatarArgs {
        emotion_state: p.emotion,
        body_motion: p.motion,
        ..Default::default()
    };
    let ver = CoreVrm::V1;
    let mut w: Vec<(String, f32)> = to_vrm(&args, None, ver)
        .expressions
        .into_iter()
        .map(|(n, v)| (n.to_lowercase(), v))
        .collect();
    if let Some(name) = p.viseme.vrm_name(ver) {
        w.push((name.to_lowercase(), p.mouth_open.clamp(0.0, 1.0)));
    }
    let blink = (p.time * 0.23).fract() > 0.97 || p.eye_open < 0.3;
    if blink {
        w.push(("blink".into(), 1.0));
    }
    if let Some(e) = emote {
        if let Some(m) = e.emote.comic_morph() {
            w.push((m.to_string(), e.fade));
        }
    }
    w
}

/// The body motion as a pose: a lift (hop, crouch) and a head pitch nudge (nod, lean, think).
pub fn posed(mut p: FrameParams) -> FrameParams {
    use zena_core::tools::BodyMotion as M;
    let t = p.motion_t;
    let decay = |len: f32| (1.0 - t / len).max(0.0);
    let (lift, nod) = match p.motion {
        M::Cheer | M::Clap if t < 1.2 => (0.045 * (t * 9.0).sin().abs() * decay(1.2), 0.0),
        M::Nod if t < 0.9 => (0.0, -0.35 * (t * 10.5).sin().max(0.0) * decay(0.9)),
        M::Flinch if t < 0.5 => (-0.025 * decay(0.5), 0.15 * decay(0.5)),
        M::HideCrouch => (-0.14, -0.2),
        M::LeanForwardPoint => (-0.01, -0.12),
        M::Think => (0.0, 0.12),
        M::WaveGoodbye if t < 2.5 => (0.012 * (t * 6.0).sin(), 0.0),
        _ => (0.0, 0.0),
    };
    p.lift = lift;
    p.head[1] = (p.head[1] + nod).clamp(-1.2, 1.2);
    p
}

/// Pop-in on a spring, then a gentle float.
fn spring(age: f32) -> f32 {
    (1.0 - (-10.0 * age).exp()) * (1.0 + 0.22 * (-5.0 * age).exp() * (16.0 * age).sin())
}

impl Scene {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, font: Option<&Path>) -> Self {
        let text = TextRaster::load(font);
        if text.has_font() {
            eprintln!("[overlay] bubble font: {}", text.source);
        } else {
            eprintln!("[overlay] no usable font found: lines are logged but the speech bubble stays hidden (--font)");
        }
        Self {
            procedural: Procedural::new(device, format),
            sprites: SpritePipeline::new(device, format),
            meshes: MeshPipeline::new(device, format),
            text,
            stamps: HashMap::new(),
            props: HashMap::new(),
            parts: HashMap::new(),
            bodies: HashMap::new(),
            bubble: None,
            input: None,
            pill: None,
            rejection: None,
            drawer_rows: Vec::new(),
            badge_sprites: Vec::new(),
            drop_hint: None,
            layout: OverlayLayout::default(),
            failed: HashSet::new(),
        }
    }

    pub fn layout(&self) -> &OverlayLayout {
        &self.layout
    }

    pub fn load(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, asset: Loaded) {
        match asset {
            Loaded::Stamp { emote, w, h, rgba, frames } => {
                let sprite = self.sprites.sprite(device, queue, w, h, &rgba);
                self.stamps.insert(
                    emote,
                    Stamp {
                        sprite,
                        frames: frames.max(1),
                    },
                );
            }
            Loaded::Prop { emote, angle, mesh } => {
                let gpu = self.meshes.upload(device, queue, Arc::new(mesh));
                self.props.insert((emote, angle), gpu);
            }
            Loaded::Part { part, mesh } => {
                let gpu = self.meshes.upload(device, queue, Arc::new(mesh));
                self.parts.insert(part, gpu);
            }
            Loaded::Body2d { w, h, rgba } => {
                let s = self.sprites.sprite(device, queue, w, h, &rgba);
                self.bodies.insert(AvatarTrack::Subculture2d, Body::Sprite(s));
            }
            Loaded::BodyMesh { track, mesh, vrm } => {
                let gpu = self.meshes.upload(device, queue, Arc::new(mesh));
                self.bodies.insert(track, Body::Mesh { gpu, vrm });
            }
        }
    }

    /// What `stage` would show but is not loaded (and has not failed).
    pub fn missing(&self, stage: &Stage) -> Vec<Missing> {
        let mut out = Vec::new();
        if stage.track != AvatarTrack::Procedural && !self.bodies.contains_key(&stage.track) {
            out.push(Missing::Body(stage.track));
        }
        for p in stage.parts {
            if !self.parts.contains_key(p) {
                out.push(Missing::Part(*p));
            }
        }
        if let Some(e) = &stage.emote {
            if !self.stamps.contains_key(&e.emote) {
                out.push(Missing::Stamp(e.emote));
            }
            if e.emote.form() == EmoteForm::Prop {
                let a = self.prop_angle(e);
                if !self.props.contains_key(&(e.emote, a)) {
                    out.push(Missing::Prop(e.emote, a));
                }
            }
        }
        out.retain(|m| !self.failed.contains(m));
        out
    }

    fn prop_angle(&self, e: &EmoteShow) -> i32 {
        if e.emote == CharacterEmote::ThreatArrow3d {
            angle_bucket(e.angle_deg)
        } else {
            0
        }
    }

    /// NDC centre of the head and model-units → NDC scale for accessories on the current body.
    fn head_frame(&self, stage: &Stage) -> (Mat4, Mat4) {
        let p = &stage.params;
        let aspect = p.aspect();
        match self.bodies.get(&stage.track) {
            Some(Body::Mesh { gpu, vrm }) => {
                let (lo, hi) = gpu.data.bounds();
                let (vp, model) = body_view(stage, lo, hi, vrm.as_ref());
                // Head: a VRM's humanoid head bone (≈ 0.11 m radius, centre a little above the bone), else the
                // top of the model; the forge robot's 1 m head is centred at the origin, the accessories' frame.
                let bone = vrm.as_ref().and_then(|info| {
                    let node = info.humanoid.iter().find(|(b, _)| b == "head").map(|(_, n)| *n)?;
                    let w = gpu.data.skeleton.as_ref()?.world_matrices(&[]);
                    w.get(node).map(|m| [m[3][0], m[3][1] + 0.08, m[3][2]])
                });
                let (c, r) = match (vrm.is_some(), bone) {
                    (true, Some(c)) => (c, 0.11),
                    (true, None) => ([0.0, hi[1] - 0.12, 0.0], 0.11),
                    (false, _) => ([0.0, 0.0, 0.0], 0.5),
                };
                // A VRM turns its head by skinning, so the accessories follow with the same turn; the robot's
                // model matrix already turns it.
                let rot = if vrm.is_some() { head_rot(p) } else { IDENTITY };
                let head = mul(
                    &model,
                    &mul(&translate(c[0], c[1], c[2]), &mul(&rot, &scale(r / 0.5, r / 0.5, r / 0.5))),
                );
                (vp, head)
            }
            _ => {
                let bob = (p.time * 2.0).sin() * 0.01;
                let centre = [p.head[0] * 0.08, (0.28 + p.head[1] * 0.06 + bob) / aspect + p.lift];
                (ortho_frame(centre, 1.24, aspect, 2.0), head_rot(p))
            }
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        stage: &Stage,
    ) {
        let posed_stage = Stage {
            params: posed(stage.params),
            ..*stage
        };
        let stage = &posed_stage;
        let p = stage.params;
        let aspect = p.aspect();
        let px = p.width as f32 / zena_core::window::VIEWPORT_W as f32; // DPI scale

        // ---- prepare: body
        let mut body_kind = 0; // 0 procedural, 1 mesh, 2 sprite
        let mut comic_on_body = false;
        match self.bodies.get_mut(&stage.track) {
            Some(Body::Mesh { gpu, vrm }) if stage.track != AvatarTrack::Procedural => {
                let (lo, hi) = gpu.data.bounds();
                let (vp, model) = body_view(stage, lo, hi, vrm.as_ref());
                let v0 = vrm.as_ref().is_some_and(|v| v.version == VrmVersion::V0);
                let names = body_expressions(&p, stage.emote.as_ref());
                let weights: Vec<(usize, f32)> = match vrm {
                    Some(info) => {
                        let named: Vec<(&str, f32)> = names.iter().map(|(n, w)| (n.as_str(), *w)).collect();
                        if let Some(e) = &stage.emote {
                            comic_on_body = vrm::emote_expression(info, e.emote).is_some();
                        }
                        vrm::expression_morphs(info, &gpu.data, &named)
                    }
                    None => names.iter().filter_map(|(n, w)| gpu.data.morph_index(n).map(|i| (i, *w))).collect(),
                };
                gpu.set_morphs(queue, &weights);
                if let (Some(info), Some(sk)) = (vrm.as_ref(), gpu.data.skeleton.as_ref()) {
                    let sign = if v0 { -1.0 } else { 1.0 };
                    let (yaw, pitch) = (p.look.yaw_deg.to_radians() * sign, -p.look.pitch_deg.to_radians() * sign);
                    let mut extra = Vec::new();
                    for (bone, share) in [("neck", 0.35f32), ("head", 0.45)] {
                        if let Some(node) = info.humanoid.iter().find(|(b, _)| b == bone).map(|(_, n)| *n) {
                            extra.push((node, quat_yaw_pitch(yaw * share, pitch * share)));
                        }
                    }
                    gpu.set_joints(queue, &sk.joint_matrices(&extra));
                }
                gpu.set_pose(
                    queue,
                    &MeshPose {
                        view_proj: vp,
                        model,
                        alpha: p.alpha,
                        time: p.time,
                    },
                );
                body_kind = 1;
            }
            Some(Body::Sprite(s)) if stage.track == AvatarTrack::Subculture2d => {
                let bob = (p.time * 2.0).sin() * 0.012;
                let squash = 1.0 + 0.025 * p.mouth_open;
                let hw = 0.92;
                let hh = hw * s.height as f32 / s.width.max(1) as f32 / aspect;
                s.place(
                    queue,
                    &Placement {
                        centre: [p.head[0] * 0.04, -0.08 + bob + p.head[1] * 0.02 + p.lift],
                        half: [hw / squash.sqrt(), hh * squash],
                        uv0: [0.0, 0.0],
                        uv1: [1.0, 1.0],
                        angle: -p.head[0] * 0.06,
                        alpha: p.alpha,
                    },
                    aspect,
                );
                body_kind = 2;
            }
            _ => {}
        }
        if body_kind == 0 {
            self.procedural.prepare(queue, &p);
        }

        // ---- accessories on the head
        let (head_vp, head_model) = self.head_frame(stage);
        for part in stage.parts {
            if let Some(g) = self.parts.get(part) {
                g.set_pose(
                    queue,
                    &MeshPose {
                        view_proj: head_vp,
                        model: head_model,
                        alpha: p.alpha,
                        time: p.time,
                    },
                );
            }
        }

        // ---- emote: a 3D prop, a comic morph on the VRM, or the SD stamp
        let mut show_prop: Option<(CharacterEmote, i32)> = None;
        let mut show_stamp: Option<CharacterEmote> = None;
        if let Some(e) = stage.emote.filter(|e| e.fade > 0.01) {
            let s = spring(e.age) * e.fade.sqrt();
            let a = self.prop_angle(&e);
            let anchor = [0.56, 0.62];
            if e.emote.form() == EmoteForm::Prop && self.props.contains_key(&(e.emote, a)) {
                let g = &self.props[&(e.emote, a)];
                let (centre, model) = if e.emote == CharacterEmote::ThreatArrow3d {
                    // Around the head toward the threat, nudging outward.
                    let t = e.angle_deg.to_radians();
                    let push = 0.04 * (p.time * 6.0).sin();
                    let c = [0.0 + (0.62 + push) * t.cos(), (0.28 + (0.62 + push) * t.sin()) / aspect];
                    (c, scale(s, s, s))
                } else {
                    let wob = (p.time * 1.6).sin() * 0.5;
                    let float = (p.time * 2.4).sin() * 0.02;
                    ([anchor[0], anchor[1] + float], mul(&rot_y(wob), &scale(s, s, s)))
                };
                g.set_pose(
                    queue,
                    &MeshPose {
                        view_proj: ortho_frame(centre, 2.4, aspect, 1.0),
                        model,
                        alpha: p.alpha * e.fade,
                        time: p.time,
                    },
                );
                show_prop = Some((e.emote, a));
            } else if !(comic_on_body && e.emote.form() == EmoteForm::Morph) {
                if let Some(st) = self.stamps.get(&e.emote) {
                    let f = ((e.age * 8.0) as u32) % st.frames;
                    let fw = 1.0 / st.frames as f32;
                    let size = 118.0 * px;
                    let mut pl = Placement::pixels(0.0, 0.0, size * s, size * s, p.width, p.height);
                    pl.centre = [anchor[0], anchor[1] + (p.time * 2.4).sin() * 0.015];
                    pl.uv0 = [f as f32 * fw, 0.0];
                    pl.uv1 = [(f + 1) as f32 * fw, 1.0];
                    pl.angle = (p.time * 3.0).sin() * 0.08;
                    pl.alpha = p.alpha * e.fade;
                    st.sprite.place(queue, &pl, aspect);
                    show_stamp = Some(e.emote);
                }
            }
        }

        // ---- pass 3 prep: Stage 0 pill, Stage 1 rejection chip, Stage 2 drawer, 1–9 badges, bubble & chat
        self.layout = OverlayLayout::default();
        let width = (p.width as f32 - 16.0 * px).max(32.0) as u32;
        let mut y_top = 8.0 * px;

        let mut show_pill = false;
        if let Some((label, color)) = stage.sync_pill.filter(|(t, _)| !t.trim().is_empty()) {
            let key = format!("{label}|{:.2},{:.2},{:.2}", color[0], color[1], color[2]);
            let pill_w = (p.width as f32 * 0.56).clamp(96.0, width as f32) as u32;
            if self.pill.as_ref().is_none_or(|(k, _)| *k != key) {
                self.pill = self
                    .text
                    .render_box(label, "", false, pill_w, &BoxStyle::pill(px, color))
                    .map(|img| (key, self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
            }
            if let Some((_, s)) = &self.pill {
                let (sw, sh) = (s.width as f32, s.height as f32);
                let x0 = 8.0 * px;
                s.place(
                    queue,
                    &Placement::pixels(x0 + sw / 2.0, y_top + sh / 2.0, sw, sh, p.width, p.height),
                    aspect,
                );
                self.layout.pill_rect = Some([x0, y_top, x0 + sw, y_top + sh]);
                y_top += sh + 4.0 * px;
                show_pill = true;
            }
        }

        let mut show_rejection = false;
        if let Some(msg) = stage.rejection_chip.filter(|t| !t.trim().is_empty()) {
            if self.rejection.as_ref().is_none_or(|(k, _)| k != msg) {
                self.rejection = self
                    .text
                    .render_box(msg, "", false, width, &BoxStyle::rejection(px))
                    .map(|img| (msg.to_string(), self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
            }
            if let Some((_, s)) = &self.rejection {
                let (sw, sh) = (s.width as f32, s.height as f32);
                let x0 = (p.width as f32 - sw) / 2.0;
                s.place(
                    queue,
                    &Placement::pixels(x0 + sw / 2.0, y_top + sh / 2.0, sw, sh, p.width, p.height),
                    aspect,
                );
                self.layout.rejection_rect = Some([x0, y_top, x0 + sw, y_top + sh]);
                y_top += sh + 4.0 * px;
                show_rejection = true;
            }
        }

        let mut show_drop_hint = false;
        if let Some(hint) = stage.drop_hint.filter(|t| !t.trim().is_empty()) {
            if self.drop_hint.as_ref().is_none_or(|(k, _)| k != hint) {
                self.drop_hint = self
                    .text
                    .render_box(hint, "", false, width, &BoxStyle::pill(px, [0.42, 0.85, 0.62, 1.0]))
                    .map(|img| (hint.to_string(), self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
            }
            if let Some((_, s)) = &self.drop_hint {
                let (sw, sh) = (s.width as f32, s.height as f32);
                s.place(
                    queue,
                    &Placement::pixels(p.width as f32 / 2.0, y_top + sh / 2.0, sw, sh, p.width, p.height),
                    aspect,
                );
                y_top += sh + 4.0 * px;
                show_drop_hint = true;
            }
        }

        // Stage 2 history/verdict drawer rows (up to 4 visible rows with clickable ▲/▼ areas).
        let mut drawer_draw_count = 0usize;
        if let Some(items) = stage.drawer {
            let max_rows = items.len().min(4);
            let mut keys: Vec<String> = Vec::with_capacity(max_rows);
            for item in &items[..max_rows] {
                let local_chip = tier_chip("나", item.pair.local_tier, item.pair.local_wl);
                let global_chip = tier_chip("전체", item.pair.global_tier, item.pair.global_up_down);
                let up_mark = if item.pinned == Some(UserPin::Promoted) { "【▲】" } else { "[▲]" };
                let down_mark = if item.pinned == Some(UserPin::Demoted) { "【▼】" } else { "[▼]" };
                keys.push(format!(
                    "{}\n{local_chip} · {global_chip}  {up_mark} {down_mark}",
                    item.label
                ));
            }
            let cache_matches = self.drawer_rows.len() == keys.len()
                && self.drawer_rows.iter().zip(&keys).all(|((k, _), want)| k == want);
            if !cache_matches {
                self.drawer_rows.clear();
                for k in &keys {
                    if let Some(img) = self.text.render_box(k, "", false, width, &BoxStyle::drawer(px)) {
                        self.drawer_rows
                            .push((k.clone(), self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
                    }
                }
            }
            let x0 = (p.width as f32 - width as f32) / 2.0;
            for (idx, (_, s)) in self.drawer_rows.iter().enumerate() {
                let (sw, sh) = (s.width as f32, s.height as f32);
                s.place(
                    queue,
                    &Placement::pixels(x0 + sw / 2.0, y_top + sh / 2.0, sw, sh, p.width, p.height),
                    aspect,
                );
                let btn_w = 36.0 * px;
                let down_x1 = x0 + sw - 6.0 * px;
                let down_x0 = (down_x1 - btn_w).max(x0);
                let up_x1 = (down_x0 - 4.0 * px).max(x0);
                let up_x0 = (up_x1 - btn_w).max(x0);
                let btn_y0 = y_top + sh * 0.45;
                let btn_y1 = y_top + sh;
                if let Some(item) = items.get(idx) {
                    self.layout.drawer_rows.push(DrawerHitRow {
                        record_id: item.record_id.clone(),
                        up_rect: [up_x0, btn_y0, up_x1, btn_y1],
                        down_rect: [down_x0, btn_y0, down_x1, btn_y1],
                    });
                }
                y_top += sh + 3.0 * px;
                drawer_draw_count += 1;
            }
        }

        // 1–9 numbered navigation badges (stacked cleanly along the right edge when drawer is closed).
        let mut badge_draw_count = 0usize;
        if !stage.badges.is_empty() && drawer_draw_count == 0 {
            let max_badges = stage.badges.len().min(9);
            let badge_w = (124.0 * px).min(p.width as f32 * 0.44) as u32;
            let keys: Vec<(String, [f32; 4], bool)> = stage.badges[..max_badges]
                .iter()
                .map(|b| {
                    let lock = if b.commit_guarded { "🔒 " } else { "" };
                    let short: String = b.label.chars().take(10).collect();
                    let label = format!("#{} {lock}{short}", b.number);
                    let border = tier_border(b.tier);
                    (label, border, b.commit_guarded)
                })
                .collect();
            let cache_matches = self.badge_sprites.len() == keys.len()
                && self
                    .badge_sprites
                    .iter()
                    .zip(&keys)
                    .all(|((k, _), (want, _, cg))| *k == format!("{want}|{cg}"));
            if !cache_matches {
                self.badge_sprites.clear();
                for (label, border, cg) in &keys {
                    if let Some(img) =
                        self.text
                            .render_box(label, "", false, badge_w, &BoxStyle::badge(px, *border, *cg))
                    {
                        self.badge_sprites.push((
                            format!("{label}|{cg}"),
                            self.sprites.sprite(device, queue, img.w, img.h, &img.rgba),
                        ));
                    }
                }
            }
            let mut by = 8.0 * px;
            for (idx, (_, s)) in self.badge_sprites.iter().enumerate() {
                let (sw, sh) = (s.width as f32, s.height as f32);
                let x0 = p.width as f32 - sw - 8.0 * px;
                s.place(
                    queue,
                    &Placement::pixels(x0 + sw / 2.0, by + sh / 2.0, sw, sh, p.width, p.height),
                    aspect,
                );
                if let Some(b) = stage.badges.get(idx) {
                    self.layout
                        .badge_rects
                        .push((b.number, b.commit_guarded, [x0, by, x0 + sw, by + sh]));
                }
                by += sh + 3.0 * px;
                badge_draw_count += 1;
            }
        }

        let mut y_bottom = p.height as f32 - 8.0 * px;
        let mut show_input = false;
        if let Some((text, pre, caret)) = stage.input {
            let key = format!("{text}\u{1}{pre}\u{1}{caret}");
            if self.input.as_ref().is_none_or(|(k, _)| *k != key) {
                self.input = self
                    .text
                    .render_box(text, pre, caret, width, &BoxStyle::input(px))
                    .map(|img| (key, self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
            }
            if let Some((_, s)) = &self.input {
                let h = s.height as f32;
                s.place(
                    queue,
                    &Placement::pixels(p.width as f32 / 2.0, y_bottom - h / 2.0, s.width as f32, h, p.width, p.height),
                    aspect,
                );
                y_bottom -= h + 6.0 * px;
                show_input = true;
            }
        }
        let mut show_bubble = false;
        if let Some((text, alpha)) = stage.bubble.filter(|(t, a)| !t.trim().is_empty() && *a > 0.01) {
            if self.bubble.as_ref().is_none_or(|(k, _)| k != text) {
                self.bubble = self
                    .text
                    .render_box(text, "", false, width, &BoxStyle::bubble(px))
                    .map(|img| (text.to_string(), self.sprites.sprite(device, queue, img.w, img.h, &img.rgba)));
            }
            if let Some((_, s)) = &self.bubble {
                let h = s.height as f32;
                let mut pl = Placement::pixels(p.width as f32 / 2.0, y_bottom - h / 2.0, s.width as f32, h, p.width, p.height);
                pl.alpha = alpha * p.alpha.max(0.6);
                s.place(queue, &pl, aspect);
                show_bubble = true;
            }
        }

        // ---- pass 1: body (+ accessories)
        {
            let mut rp = begin(enc, color, depth, true, "pass1.body");
            match (body_kind, self.bodies.get(&stage.track)) {
                (1, Some(Body::Mesh { gpu, .. })) => self.meshes.draw(&mut rp, gpu),
                (2, Some(Body::Sprite(s))) => self.sprites.draw(&mut rp, s),
                _ => self.procedural.draw(&mut rp),
            }
            for part in stage.parts {
                if let Some(g) = self.parts.get(part) {
                    self.meshes.draw(&mut rp, g);
                }
            }
        }
        // ---- pass 2: emote & props
        {
            let mut rp = begin(enc, color, depth, false, "pass2.emote");
            if let Some(k) = show_prop {
                self.meshes.draw(&mut rp, &self.props[&k]);
            }
            if let Some(e) = show_stamp {
                self.sprites.draw(&mut rp, &self.stamps[&e].sprite);
            }
        }
        // ---- pass 3: Stage 0/1/2 overlay chrome, 1–9 badges, speech bubble + chat box
        let has_ui = show_pill
            || show_rejection
            || show_drop_hint
            || drawer_draw_count > 0
            || badge_draw_count > 0
            || show_bubble
            || show_input;
        if has_ui {
            let mut rp = begin(enc, color, depth, false, "pass3.bubble");
            if let (true, Some((_, s))) = (show_pill, &self.pill) {
                self.sprites.draw(&mut rp, s);
            }
            if let (true, Some((_, s))) = (show_rejection, &self.rejection) {
                self.sprites.draw(&mut rp, s);
            }
            if let (true, Some((_, s))) = (show_drop_hint, &self.drop_hint) {
                self.sprites.draw(&mut rp, s);
            }
            for (_, s) in self.drawer_rows.iter().take(drawer_draw_count) {
                self.sprites.draw(&mut rp, s);
            }
            for (_, s) in self.badge_sprites.iter().take(badge_draw_count) {
                self.sprites.draw(&mut rp, s);
            }
            if let (true, Some((_, s))) = (show_bubble, &self.bubble) {
                self.sprites.draw(&mut rp, s);
            }
            if let (true, Some((_, s))) = (show_input, &self.input) {
                self.sprites.draw(&mut rp, s);
            }
        }
    }
}

fn tier_chip(who: &str, tier: Tier, counts: Option<(u32, u32)>) -> String {
    let word = match tier {
        Tier::Premium => "추천",
        Tier::Discount => "비추",
        Tier::Neutral => "보통",
        Tier::Unrated => "판정전",
    };
    match counts {
        Some((a, b)) => format!("{who}:{word}({a}/{b})"),
        None => format!("{who}:{word}"),
    }
}

fn tier_border(tier: Option<Tier>) -> [f32; 4] {
    match tier {
        Some(Tier::Premium) => [0.28, 0.85, 0.56, 1.0],
        Some(Tier::Discount) => [0.95, 0.36, 0.38, 1.0],
        Some(Tier::Neutral) => [0.92, 0.76, 0.34, 1.0],
        _ => [0.60, 0.46, 0.94, 1.0],
    }
}

/// Head yaw / pitch for an unskinned head (procedural face accessories, robot).
fn head_rot(p: &FrameParams) -> Mat4 {
    mul(&rot_y(p.head[0] * 0.45), &rot_x(-p.head[1] * 0.3))
}

/// View-projection and model matrix framing a mesh body in the viewport.
fn body_view(stage: &Stage, lo: [f32; 3], hi: [f32; 3], vrm: Option<&VrmInfo>) -> (Mat4, Mat4) {
    let p = &stage.params;
    let aspect = p.aspect();
    let (w, h) = ((hi[0] - lo[0]).max(1e-3), (hi[1] - lo[1]).max(1e-3));
    let depth = ((hi[2] - lo[2]).max(w).max(h)).max(0.1);
    let bob = (p.time * 2.0).sin() * 0.01;
    match vrm {
        Some(info) => {
            // Bust shot: the top 0.85 m.
            let span = 0.85f32.min(h);
            let k = (1.85 * aspect / span).min(1.8 / w.max(0.4));
            let cy = 0.95 - (hi[1] + 0.05) * k / aspect + bob + p.lift;
            let facing = if info.version == VrmVersion::V0 {
                rot_y(std::f32::consts::PI)
            } else {
                IDENTITY
            };
            let lean = rot_x(-p.head[1] * 0.02);
            (ortho_frame([0.0, cy], k, aspect, depth), mul(&lean, &facing))
        }
        None => {
            // The whole toy turns about its head (the forge / SolidCraft robot's origin).
            let k = (1.7 * aspect / h).min(1.8 / w);
            let mid = (lo[1] + hi[1]) * 0.5;
            let cy = 0.02 - mid * k / aspect + bob + p.lift;
            (ortho_frame([0.0, cy], k, aspect, depth), head_rot(p))
        }
    }
}

fn begin<'a>(
    enc: &'a mut wgpu::CommandEncoder,
    color: &'a wgpu::TextureView,
    depth: &'a wgpu::TextureView,
    clear_color: bool,
    label: &'static str,
) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: color,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: if clear_color {
                    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
                } else {
                    wgpu::LoadOp::Load
                },
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_angles_bucket_to_fifteen_degrees() {
        assert_eq!(angle_bucket(0.0), 0);
        assert_eq!(angle_bucket(7.0), 0);
        assert_eq!(angle_bucket(8.0), 15);
        assert_eq!(angle_bucket(359.0), 0);
        assert_eq!(angle_bucket(-90.0), 270);
        assert!(spring(0.0).abs() < 1e-6 && (spring(3.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn body_expressions_follow_emotion_viseme_blink_and_comic_morph() {
        let p = FrameParams {
            emotion: zena_core::tools::EmotionState::Joy,
            viseme: zena_core::avatar::Viseme::A,
            mouth_open: 0.7,
            ..Default::default()
        };
        let e = EmoteShow {
            emote: CharacterEmote::HeartEyes,
            age: 0.5,
            fade: 1.0,
            angle_deg: 0.0,
        };
        let w = body_expressions(&p, Some(&e));
        assert!(w.contains(&("happy".into(), 1.0)));
        assert!(w.contains(&("aa".into(), 0.7)));
        assert!(w.contains(&("heart_eyes".into(), 1.0)));
        let robot = zena_avatar_mcp::forge::robot_zena();
        for (name, _) in body_expressions(&p, None) {
            assert!(robot.morph_index(&name).is_some(), "the forge robot has a '{name}' morph");
        }
    }
}
