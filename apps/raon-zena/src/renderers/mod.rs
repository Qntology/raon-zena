//! The overlay's renderers, all on the one wgpu device of the micro-viewport:
//!
//! * [`procedural`] – the built-in face (no assets);
//! * [`mesh`]       – glTF / VRM meshes with toon shading, morph targets and GPU skinning (SolidCraft / forge
//!   robot body, VRM body, 3D emote props, accessories);
//! * [`sprite`]     – textured quads (SD chibi stamps, the 2D layered body, the speech bubble, the chat box);
//! * [`text`]       – CPU text layout and rasterization for the bubble and the chat box (system Korean font);
//! * [`scene`]      – the three passes: body → emote & props → speech bubble.

pub mod mesh;
pub mod procedural;
pub mod scene;
pub mod sprite;
pub mod text;

use zena_core::avatar::{LookAt, Viseme};
use zena_core::tools::{BodyMotion, EmotionState};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Column-major 4×4 matrix (`m[col][row]`), as WGSL `mat4x4<f32>` expects.
pub type Mat4 = [[f32; 4]; 4];

/// Per-frame values every pass reads (computed once by the overlay).
#[derive(Debug, Clone, Copy)]
pub struct FrameParams {
    /// Eyeballs in [-1, 1] (ParamEyeBallX/Y).
    pub gaze: [f32; 2],
    /// Head yaw / pitch in [-1, 1].
    pub head: [f32; 2],
    pub mouth_open: f32,
    pub mouth_form: f32,
    pub brow: f32,
    pub eye_open: f32,
    /// Hover fade.
    pub alpha: f32,
    pub time: f32,
    pub width: u32,
    pub height: u32,
    pub emotion: EmotionState,
    pub motion: BodyMotion,
    pub viseme: Viseme,
    pub look: LookAt,
    /// Seconds since the current motion started.
    pub motion_t: f32,
    /// Vertical body offset in NDC from the motion (hop, crouch), set by the scene.
    pub lift: f32,
}

impl Default for FrameParams {
    fn default() -> Self {
        Self {
            gaze: [0.0; 2],
            head: [0.0; 2],
            mouth_open: 0.0,
            mouth_form: 0.2,
            brow: 0.0,
            eye_open: 1.0,
            alpha: 1.0,
            time: 0.0,
            width: zena_core::window::VIEWPORT_W,
            height: zena_core::window::VIEWPORT_H,
            emotion: EmotionState::Neutral,
            motion: BodyMotion::Idle,
            viseme: Viseme::Rest,
            look: LookAt::default(),
            motion_t: 10.0,
            lift: 0.0,
        }
    }
}

impl FrameParams {
    /// Viewport height / width (the procedural shader's uv stretch).
    pub fn aspect(&self) -> f32 {
        self.height.max(1) as f32 / self.width.max(1) as f32
    }
}

pub fn depth_state(write: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(if write {
            wgpu::CompareFunction::Less
        } else {
            wgpu::CompareFunction::Always
        }),
        stencil: Default::default(),
        bias: Default::default(),
    }
}

pub fn depth_texture(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub fn texture_entries(tex: u32, sampler: u32) -> [wgpu::BindGroupLayoutEntry; 2] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: tex,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: sampler,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ]
}

pub fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// An RGBA8 sRGB texture from straight-alpha pixels, premultiplied on upload when `premultiply` (sprites; mesh
/// textures stay straight because the mesh shader applies alpha itself).
pub fn rgba_texture(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, straight: &[u8], premultiply: bool) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rgba"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    write_rgba(queue, &tex, w, h, straight, premultiply);
    tex
}

pub fn write_rgba(queue: &wgpu::Queue, tex: &wgpu::Texture, w: u32, h: u32, straight: &[u8], premultiply: bool) {
    let mut pm = straight.to_vec();
    if premultiply {
        for px in pm.chunks_exact_mut(4) {
            let a = px[3] as u32;
            for c in &mut px[..3] {
                *c = ((*c as u32 * a + 127) / 255) as u8;
            }
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pm,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

// ------------------------------------------------------------------------------------------
// Small matrix helpers (column-major)
// ------------------------------------------------------------------------------------------

pub const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[0.0; 4]; 4];
    for (c, col) in m.iter_mut().enumerate() {
        for (r, v) in col.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    m
}

pub fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = IDENTITY;
    m[3] = [x, y, z, 1.0];
    m
}

pub fn scale(x: f32, y: f32, z: f32) -> Mat4 {
    [[x, 0.0, 0.0, 0.0], [0.0, y, 0.0, 0.0], [0.0, 0.0, z, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

pub fn rot_y(rad: f32) -> Mat4 {
    let (s, c) = rad.sin_cos();
    [[c, 0.0, -s, 0.0], [0.0, 1.0, 0.0, 0.0], [s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

pub fn rot_x(rad: f32) -> Mat4 {
    let (s, c) = rad.sin_cos();
    [[1.0, 0.0, 0.0, 0.0], [0.0, c, s, 0.0], [0.0, -s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn rot_z(rad: f32) -> Mat4 {
    let (s, c) = rad.sin_cos();
    [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

/// Model units → NDC: `(x, y)` at `centre + units · k` with the viewport's aspect corrected, `z` (toward the
/// viewer) into depth 0..1 around 0.5.
pub fn ortho_frame(centre: [f32; 2], k: f32, aspect: f32, depth_range: f32) -> Mat4 {
    let mut m = scale(k, k / aspect, -0.5 / depth_range.max(1e-3));
    m[3] = [centre[0], centre[1], 0.5, 1.0];
    m
}

/// Quaternion (xyzw) for yaw (about +Y) then pitch (about +X).
pub fn quat_yaw_pitch(yaw: f32, pitch: f32) -> [f32; 4] {
    let (sy, cy) = (yaw * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    // q = q_yaw * q_pitch
    [cy * sp, sy * cp, -sy * sp, cy * cp]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(m: &Mat4, p: [f32; 3]) -> [f32; 3] {
        let mut o = [0.0; 3];
        for (r, v) in o.iter_mut().enumerate() {
            *v = m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r];
        }
        o
    }

    #[test]
    fn matrices_compose_like_column_major_glsl() {
        let m = mul(&translate(1.0, 2.0, 3.0), &scale(2.0, 2.0, 2.0));
        assert_eq!(apply(&m, [1.0, 1.0, 1.0]), [3.0, 4.0, 5.0]);
        let r = apply(&rot_z(std::f32::consts::FRAC_PI_2), [1.0, 0.0, 0.0]);
        assert!((r[0]).abs() < 1e-6 && (r[1] - 1.0).abs() < 1e-6);
        let r = apply(&rot_y(std::f32::consts::FRAC_PI_2), [0.0, 0.0, 1.0]);
        assert!(
            (r[0] - 1.0).abs() < 1e-6,
            "+Z turns to +X: a positive yaw looks screen-right ({r:?})"
        );
        let f = ortho_frame([0.5, 0.0], 1.0, 2.0, 1.0);
        let p = apply(&f, [0.25, 0.5, 1.0]);
        assert_eq!(p, [0.75, 0.25, 0.0], "toward the viewer is nearer");
        let q = quat_yaw_pitch(0.0, 0.0);
        assert_eq!(q, [0.0, 0.0, 0.0, 1.0]);
    }
}
