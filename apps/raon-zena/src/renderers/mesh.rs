//! glTF / VRM meshes on the overlay's device: the SolidCraft / forge robot body (Track A), the VRM body
//! (Track B-2), 3D emote props and accessories. Toon shading, alpha mask / blend, morph targets blended on the CPU
//! (expressions, visemes, comic morphs), GPU skinning from joint matrices (head look on a VRM).

use super::{depth_state, linear_sampler, rgba_texture, texture_entries, uniform_entry, Mat4, DEPTH_FORMAT, IDENTITY};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use zena_avatar_mcp::mesh::{AlphaMode, MeshData};

/// Joint matrices the storage buffer holds (larger skins are cut).
pub const MAX_JOINTS: usize = 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: Mat4,
    model: Mat4,
    light: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialU {
    base: [f32; 4],
    emissive: [f32; 4],
    flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Attr {
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
    joint: [u32; 4],
    weight: [f32; 4],
}

pub struct MeshPipeline {
    opaque: wgpu::RenderPipeline,
    blend: wgpu::RenderPipeline,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

struct GpuMaterial {
    bind: wgpu::BindGroup,
    blend: bool,
}

/// One mesh resident on the device.
pub struct GpuMesh {
    pub data: Arc<MeshData>,
    positions: wgpu::Buffer,
    attrs: wgpu::Buffer,
    index: wgpu::Buffer,
    ubuf: wgpu::Buffer,
    joints: wgpu::Buffer,
    bind0: wgpu::BindGroup,
    materials: Vec<GpuMaterial>,
    applied: Vec<(usize, f32)>,
    scratch: Vec<[f32; 3]>,
    pub skinned: bool,
}

/// How a mesh is placed this frame.
#[derive(Debug, Clone, Copy)]
pub struct MeshPose {
    pub view_proj: Mat4,
    pub model: Mat4,
    pub alpha: f32,
    pub time: f32,
}

impl MeshPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh.wgsl").into()),
        });
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh.0"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let [t, s] = texture_entries(1, 2);
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh.1"),
            entries: &[uniform_entry(0), t, s],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            ..Default::default()
        });
        let buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Attr>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Uint32x4, 5 => Float32x4],
            }),
        ];
        let make = |blend: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if blend { "mesh.blend" } else { "mesh.opaque" }),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &buffers,
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
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(if blend {
                    wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::Less),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }
                } else {
                    depth_state(true)
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        Self {
            opaque: make(false),
            blend: make(true),
            bgl0,
            bgl1,
            sampler: linear_sampler(device),
        }
    }

    pub fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue, data: Arc<MeshData>) -> GpuMesh {
        let n = data.positions.len();
        let skinned = data.skeleton.is_some() && data.joints.len() == n && data.weights.len() == n;
        let attrs: Vec<Attr> = (0..n)
            .map(|i| Attr {
                normal: data.normals.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                uv: data.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                color: data.colors.get(i).copied().unwrap_or([1.0; 4]),
                joint: if skinned {
                    data.joints[i].map(|j| (j as u32).min(MAX_JOINTS as u32 - 1))
                } else {
                    [0; 4]
                },
                weight: if skinned { data.weights[i] } else { [1.0, 0.0, 0.0, 0.0] },
            })
            .collect();
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh.pos"),
            contents: bytemuck::cast_slice(&data.positions),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let attrs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh.attr"),
            contents: bytemuck::cast_slice(&attrs),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh.index"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh.u"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let joints = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh.joints"),
            contents: bytemuck::cast_slice(&vec![IDENTITY; MAX_JOINTS]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bind0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh.0"),
            layout: &self.bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: joints.as_entire_binding(),
                },
            ],
        });
        let views: Vec<wgpu::TextureView> = data
            .textures
            .iter()
            .map(|t| rgba_texture(device, queue, t.width, t.height, &t.rgba8, false).create_view(&Default::default()))
            .collect();
        let white = rgba_texture(device, queue, 1, 1, &[255; 4], false).create_view(&Default::default());
        let mut materials: Vec<GpuMaterial> = data
            .materials
            .iter()
            .map(|m| {
                let mode = match m.alpha_mode {
                    AlphaMode::Opaque => 0.0,
                    AlphaMode::Mask => 1.0,
                    AlphaMode::Blend => 2.0,
                };
                let u = MaterialU {
                    base: m.base_color,
                    emissive: [m.emissive[0], m.emissive[1], m.emissive[2], m.alpha_cutoff],
                    flags: [mode, 0.0, 0.0, 0.0],
                };
                let mbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh.material"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let view = m.texture.and_then(|t| views.get(t)).unwrap_or(&white);
                GpuMaterial {
                    bind: self.material_bind(device, &mbuf, view),
                    blend: matches!(m.alpha_mode, AlphaMode::Blend),
                }
            })
            .collect();
        if materials.is_empty() {
            let u = MaterialU {
                base: [1.0; 4],
                emissive: [0.0, 0.0, 0.0, 0.5],
                flags: [0.0; 4],
            };
            let mbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh.material"),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            materials.push(GpuMaterial {
                bind: self.material_bind(device, &mbuf, &white),
                blend: false,
            });
        }
        GpuMesh {
            scratch: Vec::with_capacity(n),
            data,
            positions,
            attrs,
            index,
            ubuf,
            joints,
            bind0,
            materials,
            applied: Vec::new(),
            skinned,
        }
    }

    fn material_bind(&self, device: &wgpu::Device, buf: &wgpu::Buffer, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh.1"),
            layout: &self.bgl1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Opaque and masked primitives first, then blended ones.
    pub fn draw(&self, rp: &mut wgpu::RenderPass<'_>, m: &GpuMesh) {
        rp.set_bind_group(0, &m.bind0, &[]);
        rp.set_vertex_buffer(0, m.positions.slice(..));
        rp.set_vertex_buffer(1, m.attrs.slice(..));
        rp.set_index_buffer(m.index.slice(..), wgpu::IndexFormat::Uint32);
        for pass_blend in [false, true] {
            rp.set_pipeline(if pass_blend { &self.blend } else { &self.opaque });
            for p in &m.data.primitives {
                let mat = &m.materials[p.material.min(m.materials.len() - 1)];
                if mat.blend != pass_blend || p.index_count == 0 {
                    continue;
                }
                rp.set_bind_group(1, &mat.bind, &[]);
                rp.draw_indexed(p.first_index..p.first_index + p.index_count, 0, 0..1);
            }
        }
    }
}

impl GpuMesh {
    /// Blend morph targets (`(index into data.morphs, weight)`) into the positions when they changed.
    pub fn set_morphs(&mut self, queue: &wgpu::Queue, weights: &[(usize, f32)]) {
        let mut w: Vec<(usize, f32)> = weights.iter().copied().filter(|(_, v)| *v > 0.01).collect();
        w.sort_by_key(|x| x.0);
        let same = w.len() == self.applied.len() && w.iter().zip(&self.applied).all(|(a, b)| a.0 == b.0 && (a.1 - b.1).abs() < 0.02);
        if same || self.data.morphs.is_empty() {
            return;
        }
        self.data.apply_morphs(&w, &mut self.scratch);
        queue.write_buffer(&self.positions, 0, bytemuck::cast_slice(&self.scratch));
        self.applied = w;
    }

    /// Joint matrices (skinned meshes only).
    pub fn set_joints(&self, queue: &wgpu::Queue, mats: &[Mat4]) {
        if self.skinned && !mats.is_empty() {
            let n = mats.len().min(MAX_JOINTS);
            queue.write_buffer(&self.joints, 0, bytemuck::cast_slice(&mats[..n]));
        }
    }

    pub fn set_pose(&self, queue: &wgpu::Queue, pose: &MeshPose) {
        let l = [-0.45f32, 0.75, 0.55];
        let u = Uniforms {
            view_proj: pose.view_proj,
            model: pose.model,
            light: [l[0], l[1], l[2], 0.18],
            params: [pose.alpha, pose.time, if self.skinned { 1.0 } else { 0.0 }, 0.0],
        };
        queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }
}
