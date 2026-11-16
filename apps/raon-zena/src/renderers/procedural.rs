//! The built-in procedural face (`avatar.wgsl`): eyes follow the gaze, brows and mouth the emotion, the mouth
//! opens with the speech envelope. Needs no assets.

use super::{depth_state, uniform_entry, FrameParams};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Uniforms {
    gaze: [f32; 2],
    head: [f32; 2],
    mouth_open: f32,
    mouth_form: f32,
    brow: f32,
    eye_open: f32,
    alpha: f32,
    time: f32,
    aspect: f32,
    _pad: f32,
}

pub struct Procedural {
    pipeline: wgpu::RenderPipeline,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl Procedural {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("avatar"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../avatar.wgsl").into()),
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("procedural.u"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[uniform_entry(0)],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("procedural"),
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_state(false)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, ubuf, bind }
    }

    pub fn prepare(&self, queue: &wgpu::Queue, p: &FrameParams) {
        let u = Uniforms {
            gaze: p.gaze,
            // The shader moves the face by 0.06 uv per head unit; the motion's lift rides on that.
            head: [p.head[0], p.head[1] + p.lift * p.aspect() / 0.06],
            mouth_open: p.mouth_open,
            mouth_form: p.mouth_form,
            brow: p.brow,
            eye_open: p.eye_open,
            alpha: p.alpha,
            time: p.time,
            aspect: p.aspect(),
            _pad: 0.0,
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }

    pub fn draw(&self, rp: &mut wgpu::RenderPass<'_>) {
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.bind, &[]);
        rp.draw(0..3, 0..1);
    }
}
