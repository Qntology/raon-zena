//! Textured quads: SD chibi emote stamps (frame strips), the 2D layered body, the speech bubble and the chat box.

use super::{depth_state, linear_sampler, rgba_texture, texture_entries, uniform_entry};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Uniforms {
    centre: [f32; 2],
    half: [f32; 2],
    uv0: [f32; 2],
    uv1: [f32; 2],
    angle: f32,
    alpha: f32,
    aspect: f32,
    _pad: f32,
}

/// Where and how one sprite is drawn this frame.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    /// Centre in NDC.
    pub centre: [f32; 2],
    /// Half size in NDC.
    pub half: [f32; 2],
    /// Texture rectangle (`uv0` top-left, `uv1` bottom-right).
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub angle: f32,
    pub alpha: f32,
}

impl Placement {
    /// A quad of `w × h` pixels centred at pixel `(cx, cy)` (y down) in a `vw × vh` viewport.
    pub fn pixels(cx: f32, cy: f32, w: f32, h: f32, vw: u32, vh: u32) -> Self {
        Self {
            centre: [cx / vw as f32 * 2.0 - 1.0, 1.0 - cy / vh as f32 * 2.0],
            half: [w / vw as f32, h / vh as f32],
            uv0: [0.0, 0.0],
            uv1: [1.0, 1.0],
            angle: 0.0,
            alpha: 1.0,
        }
    }
}

pub struct SpritePipeline {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// One texture with its own uniforms.
pub struct Sprite {
    pub width: u32,
    pub height: u32,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl SpritePipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sprite"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sprite.wgsl").into()),
        });
        let [t, s] = texture_entries(1, 2);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprite"),
            entries: &[uniform_entry(0), t, s],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprite"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(depth_state(false)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            bgl,
            sampler: linear_sampler(device),
        }
    }

    /// A sprite from straight-alpha RGBA8 pixels.
    pub fn sprite(&self, device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32, rgba: &[u8]) -> Sprite {
        let tex = rgba_texture(device, queue, w, h, rgba, true);
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite.u"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = tex.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        Sprite {
            width: w,
            height: h,
            ubuf,
            bind,
        }
    }

    pub fn draw(&self, rp: &mut wgpu::RenderPass<'_>, s: &Sprite) {
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &s.bind, &[]);
        rp.draw(0..4, 0..1);
    }
}

impl Sprite {
    pub fn place(&self, queue: &wgpu::Queue, p: &Placement, aspect: f32) {
        let u = Uniforms {
            centre: p.centre,
            half: p.half,
            uv0: p.uv0,
            uv1: p.uv1,
            angle: p.angle,
            alpha: p.alpha,
            aspect,
            _pad: 0.0,
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }
}
