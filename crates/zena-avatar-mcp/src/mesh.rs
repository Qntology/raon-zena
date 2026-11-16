//! Renderer-ready mesh data plus a glTF 2.0 binary writer and a `.glb` / `.vrm` reader.
//!
//! [`MeshData`] is one shared vertex buffer (positions, normals, uvs, colours, optional skin) and one
//! triangle-list index buffer split into [`Primitive`]s, one per material draw. Morph targets are sparse
//! position deltas; `MorphTarget::mesh/target` keep the source glTF mesh and target index so VRM
//! expression binds can be resolved ([`crate::vrm::expression_morphs`]).
//!
//! Units are whatever the source uses (the forge writes metres, Y up, +Z facing the viewer). Matrices are
//! column-major `[[f32; 4]; 4]` with `m[col][row]`, as WGSL `mat4x4<f32>` expects. Quaternions are `[x, y, z, w]`.
//!
//! [`load_glb`] walks the default scene: unskinned meshes are baked into world space; skinned meshes stay in
//! bind space with `JOINTS_0` / `WEIGHTS_0` and a [`Skeleton`] (all glTF nodes, so VRM humanoid node indices
//! stay valid). When a file mixes both, unskinned meshes are rigidly bound to their own node (an extra joint
//! whose inverse bind is the node's rest world transform) so they follow bone animation.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use image::ImageEncoder;
use serde_json::{json, Value};

use crate::math::{self, Mat4};

/// The identity matrix (for [`MeshData::append`]).
pub const IDENTITY: [[f32; 4]; 4] = math::IDENTITY;

/// `translation * rotation(quaternion xyzw) * scale` as a column-major matrix.
pub fn trs_matrix(translation: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [[f32; 4]; 4] {
    math::trs(translation, rotation, scale)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    pub name: String,
    /// Linear RGBA factor.
    pub base_color: [f32; 4],
    /// Index into [`MeshData::textures`] (base colour texture, sampled with `TEXCOORD_0`).
    pub texture: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
    pub emissive: [f32; 3],
}

impl Default for Material {
    fn default() -> Self {
        Material {
            name: "default".into(),
            base_color: [1.0; 4],
            texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            emissive: [0.0; 3],
        }
    }
}

impl Material {
    pub fn new(name: &str, base_color: [f32; 4]) -> Material {
        Material {
            name: name.into(),
            base_color,
            ..Default::default()
        }
    }

    pub fn with_emissive(mut self, emissive: [f32; 3]) -> Material {
        self.emissive = emissive;
        self
    }

    /// Alpha-blended and double-sided (glass, lenses).
    pub fn blended(mut self) -> Material {
        self.alpha_mode = AlphaMode::Blend;
        self.double_sided = true;
        self
    }
}

/// A decoded RGBA8 image (straight alpha, sRGB as stored in the file).
#[derive(Debug, Clone, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// One draw: `index_count` indices starting at `first_index`, drawn with `material`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primitive {
    pub first_index: u32,
    pub index_count: u32,
    pub material: usize,
}

/// A sparse position morph target.
#[derive(Debug, Clone, PartialEq)]
pub struct MorphTarget {
    pub name: String,
    /// Source glTF mesh index.
    pub mesh: usize,
    /// Target index within that glTF mesh.
    pub target: usize,
    /// `(vertex, delta)` pairs; near-zero deltas are skipped.
    pub deltas: Vec<(u32, [f32; 3])>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SkelNode {
    pub name: String,
    pub parent: Option<usize>,
    pub translation: [f32; 3],
    /// Rest rotation, quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// Node hierarchy plus the joints referenced by [`MeshData::joints`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Skeleton {
    /// Every node of the source file (indices match glTF / VRM node indices).
    pub nodes: Vec<SkelNode>,
    /// Joint slot → node index.
    pub joints: Vec<usize>,
    /// Joint slot → inverse bind matrix.
    pub inverse_bind: Vec<[[f32; 4]; 4]>,
}

impl Skeleton {
    /// Node by exact name, else case-insensitive.
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.name == name)
            .or_else(|| self.nodes.iter().position(|n| n.name.eq_ignore_ascii_case(name)))
    }

    /// World matrices of every node; `extra_rotation` quaternions are multiplied onto the rest rotation
    /// (`rest * extra`, i.e. applied in the node's own rest frame).
    pub fn world_matrices(&self, extra_rotation: &[(usize, [f32; 4])]) -> Vec<[[f32; 4]; 4]> {
        let local: Vec<Mat4> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let mut r = n.rotation;
                for (node, q) in extra_rotation {
                    if *node == i {
                        r = math::quat_mul(r, *q);
                    }
                }
                math::trs(n.translation, r, n.scale)
            })
            .collect();
        let mut world: Vec<Option<Mat4>> = vec![None; self.nodes.len()];
        for i in 0..self.nodes.len() {
            // Walk up to the first computed ancestor, then fill back down (cycle-safe via the depth bound).
            let mut chain = vec![i];
            let mut cur = i;
            while world[cur].is_none() {
                match self.nodes[cur].parent {
                    Some(p) if p < self.nodes.len() && chain.len() <= self.nodes.len() => {
                        chain.push(p);
                        cur = p;
                    }
                    _ => break,
                }
            }
            let mut acc = match world[cur] {
                Some(m) => {
                    chain.pop();
                    m
                }
                None => math::IDENTITY,
            };
            for &n in chain.iter().rev() {
                acc = math::mat_mul(&acc, &local[n]);
                world[n] = Some(acc);
            }
        }
        world.into_iter().map(|m| m.unwrap_or(math::IDENTITY)).collect()
    }

    /// Joint matrices (column-major, as wgpu/WGSL `mat4x4` expects) with extra local rotations (quaternion xyzw)
    /// multiplied onto the rest rotation of the given nodes: `world(joint) * inverse_bind(joint)`.
    pub fn joint_matrices(&self, extra_rotation: &[(usize, [f32; 4])]) -> Vec<[[f32; 4]; 4]> {
        let world = self.world_matrices(extra_rotation);
        self.joints
            .iter()
            .enumerate()
            .map(|(j, &node)| {
                let w = world.get(node).copied().unwrap_or(math::IDENTITY);
                let ib = self.inverse_bind.get(j).copied().unwrap_or(math::IDENTITY);
                math::mat_mul(&w, &ib)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Same length as `positions`, zeros if absent.
    pub uvs: Vec<[f32; 2]>,
    /// Per-vertex linear RGBA multiplier, ones if absent.
    pub colors: Vec<[f32; 4]>,
    /// Empty if not skinned.
    pub joints: Vec<[u16; 4]>,
    /// Empty if not skinned.
    pub weights: Vec<[f32; 4]>,
    /// Triangle list.
    pub indices: Vec<u32>,
    pub primitives: Vec<Primitive>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub morphs: Vec<MorphTarget>,
    pub skeleton: Option<Skeleton>,
}

impl MeshData {
    pub fn new() -> MeshData {
        MeshData::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_skinned(&self) -> bool {
        !self.joints.is_empty()
    }

    pub fn add_material(&mut self, m: Material) -> usize {
        self.materials.push(m);
        self.materials.len() - 1
    }

    /// Appends raw triangles drawn with `material`; returns the base vertex. `uvs` may be empty.
    /// Extends the last primitive when it uses the same material and ends at the current index count.
    pub fn push_triangles(
        &mut self,
        material: usize,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        uvs: &[[f32; 2]],
        indices: &[u32],
    ) -> u32 {
        let base = self.positions.len() as u32;
        let n = positions.len();
        self.positions.extend_from_slice(positions);
        if normals.len() == n {
            self.normals.extend_from_slice(normals);
        } else {
            self.normals.extend(std::iter::repeat_n([0.0, 1.0, 0.0], n));
        }
        if uvs.len() == n {
            self.uvs.extend_from_slice(uvs);
        } else {
            self.uvs.extend(std::iter::repeat_n([0.0, 0.0], n));
        }
        self.colors.extend(std::iter::repeat_n([1.0; 4], n));
        if self.is_skinned() {
            self.joints.extend(std::iter::repeat_n([0; 4], n));
            self.weights.extend(std::iter::repeat_n([1.0, 0.0, 0.0, 0.0], n));
        }
        self.push_indices(material, indices.iter().map(|i| i + base));
        base
    }

    fn push_indices(&mut self, material: usize, indices: impl IntoIterator<Item = u32>) {
        let first = self.indices.len() as u32;
        self.indices.extend(indices);
        let count = self.indices.len() as u32 - first;
        if count == 0 {
            return;
        }
        match self.primitives.last_mut() {
            Some(p) if p.material == material && p.first_index + p.index_count == first => p.index_count += count,
            _ => self.primitives.push(Primitive {
                first_index: first,
                index_count: count,
                material,
            }),
        }
    }

    /// Axis-aligned bounds `(min, max)`; zeros when empty.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        if self.positions.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut mn = [f32::INFINITY; 3];
        let mut mx = [f32::NEG_INFINITY; 3];
        for p in &self.positions {
            for k in 0..3 {
                mn[k] = mn[k].min(p[k]);
                mx[k] = mx[k].max(p[k]);
            }
        }
        (mn, mx)
    }

    /// Merges `other` transformed by `transform`, remapping materials, textures and morph targets.
    /// Morph `mesh` indices of `other` are offset past this mesh's so `(mesh, target)` pairs stay distinct;
    /// mirroring transforms flip the triangle winding. If either side is skinned the result is too (the other
    /// side's vertices bind to joint 0); `other`'s skeleton is used only when this mesh has none.
    pub fn append(&mut self, other: &MeshData, transform: [[f32; 4]; 4]) {
        let vo = self.positions.len() as u32;
        let mo = self.materials.len();
        let to = self.textures.len();
        let n = other.positions.len();
        let skinned = self.is_skinned() || other.is_skinned();
        if skinned && !self.is_skinned() {
            self.joints = vec![[0; 4]; self.positions.len()];
            self.weights = vec![[1.0, 0.0, 0.0, 0.0]; self.positions.len()];
        }
        self.positions
            .extend(other.positions.iter().map(|p| math::transform_point(&transform, *p)));
        for i in 0..n {
            let nm = other.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]);
            self.normals.push(math::transform_normal(&transform, nm));
            self.uvs.push(other.uvs.get(i).copied().unwrap_or([0.0; 2]));
            self.colors.push(other.colors.get(i).copied().unwrap_or([1.0; 4]));
            if skinned {
                if other.is_skinned() {
                    self.joints.push(other.joints.get(i).copied().unwrap_or([0; 4]));
                    self.weights.push(other.weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]));
                } else {
                    self.joints.push([0; 4]);
                    self.weights.push([1.0, 0.0, 0.0, 0.0]);
                }
            }
        }
        let flip = math::det3(&transform) < 0.0;
        let io = self.indices.len() as u32;
        for tri in other.indices.chunks_exact(3) {
            if flip {
                self.indices.extend([tri[0] + vo, tri[2] + vo, tri[1] + vo]);
            } else {
                self.indices.extend([tri[0] + vo, tri[1] + vo, tri[2] + vo]);
            }
        }
        let prims: Vec<Primitive> = if other.primitives.is_empty() && !other.indices.is_empty() {
            vec![Primitive {
                first_index: 0,
                index_count: (other.indices.len() / 3 * 3) as u32,
                material: 0,
            }]
        } else {
            other.primitives.clone()
        };
        for p in prims {
            self.primitives.push(Primitive {
                first_index: p.first_index + io,
                index_count: p.index_count,
                material: p.material + mo,
            });
        }
        if other.materials.is_empty() && !other.indices.is_empty() {
            self.materials.push(Material::default());
        }
        self.materials.extend(other.materials.iter().map(|m| Material {
            texture: m.texture.map(|t| t + to),
            ..m.clone()
        }));
        self.textures.extend(other.textures.iter().cloned());
        let mesh_off = self.morphs.iter().map(|m| m.mesh + 1).max().unwrap_or(0);
        self.morphs.extend(other.morphs.iter().map(|m| {
            MorphTarget {
                name: m.name.clone(),
                mesh: m.mesh + mesh_off,
                target: m.target,
                deltas: m
                    .deltas
                    .iter()
                    .map(|(v, d)| (v + vo, math::transform_vector(&transform, *d)))
                    .collect(),
            }
        }));
        if self.skeleton.is_none() {
            self.skeleton = other.skeleton.clone();
        }
    }

    /// First morph target with this name (case-insensitive).
    pub fn morph_index(&self, name: &str) -> Option<usize> {
        self.morphs.iter().position(|m| m.name.eq_ignore_ascii_case(name))
    }

    /// `out = positions + Σ weight · delta` for `(morph index, weight)` pairs.
    pub fn apply_morphs(&self, weights: &[(usize, f32)], out_positions: &mut Vec<[f32; 3]>) {
        out_positions.clear();
        out_positions.extend_from_slice(&self.positions);
        for &(m, w) in weights {
            let Some(morph) = self.morphs.get(m) else { continue };
            if w == 0.0 {
                continue;
            }
            for (v, d) in &morph.deltas {
                if let Some(p) = out_positions.get_mut(*v as usize) {
                    p[0] += w * d[0];
                    p[1] += w * d[1];
                    p[2] += w * d[2];
                }
            }
        }
    }

    /// Checks array lengths, index/primitive/material/texture ranges and finiteness.
    pub fn validate(&self) -> Result<()> {
        let n = self.positions.len();
        if self.normals.len() != n || self.uvs.len() != n || self.colors.len() != n {
            bail!(
                "attribute length mismatch: positions {n}, normals {}, uvs {}, colors {}",
                self.normals.len(),
                self.uvs.len(),
                self.colors.len()
            );
        }
        if (!self.joints.is_empty() || !self.weights.is_empty()) && (self.joints.len() != n || self.weights.len() != n) {
            bail!("skin length mismatch");
        }
        if !self.indices.len().is_multiple_of(3) {
            bail!("index count {} is not a multiple of 3", self.indices.len());
        }
        if let Some(i) = self.indices.iter().find(|&&i| i as usize >= n) {
            bail!("index {i} out of range ({n} vertices)");
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !self.positions.iter().all(|p| finite(p)) || !self.normals.iter().all(|p| finite(p)) {
            bail!("non-finite position or normal");
        }
        for p in &self.primitives {
            if (p.first_index + p.index_count) as usize > self.indices.len() || !p.index_count.is_multiple_of(3) {
                bail!("primitive {p:?} out of range");
            }
            if p.material >= self.materials.len() {
                bail!("primitive material {} out of range", p.material);
            }
        }
        for m in &self.materials {
            if m.texture.is_some_and(|t| t >= self.textures.len()) {
                bail!("material {} texture out of range", m.name);
            }
        }
        for m in &self.morphs {
            if m.deltas.iter().any(|(v, d)| *v as usize >= n || !finite(d)) {
                bail!("morph {} has an invalid delta", m.name);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------
// GLB container
// ---------------------------------------------------------------------------------------------------------

const GLB_MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// Splits a `.glb` / `.vrm` into its JSON chunk and optional BIN chunk.
pub fn split_glb(bytes: &[u8]) -> Result<(Value, Option<&[u8]>)> {
    let u32_at = |o: usize| -> Result<u32> {
        let b = bytes.get(o..o + 4).context("truncated GLB")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    if bytes.get(0..4) != Some(GLB_MAGIC.as_slice()) {
        bail!("not a binary glTF (bad magic)");
    }
    let total = (u32_at(8)? as usize).min(bytes.len());
    let (mut json, mut bin) = (None, None);
    let mut off = 12;
    while off + 8 <= total {
        let len = u32_at(off)? as usize;
        let ty = u32_at(off + 4)?;
        let data = bytes.get(off + 8..off + 8 + len).context("truncated GLB chunk")?;
        match ty {
            CHUNK_JSON if json.is_none() => json = Some(serde_json::from_slice::<Value>(data).context("GLB JSON chunk")?),
            CHUNK_BIN if bin.is_none() => bin = Some(data),
            _ => {}
        }
        off += 8 + len.div_ceil(4) * 4;
    }
    Ok((json.context("GLB has no JSON chunk")?, bin))
}

/// Packs a glTF JSON document and BIN payload into a `.glb`.
pub fn pack_glb(json: &Value, bin: &[u8]) -> Vec<u8> {
    let mut js = serde_json::to_vec(json).expect("serialisable JSON");
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let mut bn = bin.to_vec();
    while !bn.len().is_multiple_of(4) {
        bn.push(0);
    }
    let total = 12 + 8 + js.len() + if bn.is_empty() { 0 } else { 8 + bn.len() };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(GLB_MAGIC);
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&js);
    if !bn.is_empty() {
        out.extend_from_slice(&(bn.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&bn);
    }
    out
}

#[derive(Default)]
struct BinWriter {
    data: Vec<u8>,
    views: Vec<Value>,
}

impl BinWriter {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
        let mut v = json!({"buffer": 0, "byteOffset": self.data.len(), "byteLength": bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.data.extend_from_slice(bytes);
        self.views.push(v);
        self.views.len() - 1
    }
}

fn f32_bytes<'a>(it: impl IntoIterator<Item = &'a f32>) -> Vec<u8> {
    it.into_iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn u32_bytes<'a>(it: impl IntoIterator<Item = &'a u32>) -> Vec<u8> {
    it.into_iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn alpha_mode_str(m: AlphaMode) -> &'static str {
    match m {
        AlphaMode::Opaque => "OPAQUE",
        AlphaMode::Mask => "MASK",
        AlphaMode::Blend => "BLEND",
    }
}

fn encode_png(t: &Texture) -> Vec<u8> {
    let mut png = Vec::new();
    let ok = t.rgba8.len() == (t.width as usize) * (t.height as usize) * 4
        && t.width > 0
        && image::codecs::png::PngEncoder::new(&mut png)
            .write_image(&t.rgba8, t.width, t.height, image::ExtendedColorType::Rgba8)
            .is_ok();
    if !ok {
        png.clear();
        let _ = image::codecs::png::PngEncoder::new(&mut png).write_image(&[255; 4], 1, 1, image::ExtendedColorType::Rgba8);
    }
    png
}

/// glTF 2.0 binary: one node, one mesh with one primitive per [`Primitive`] (all sharing the vertex
/// accessors), PBR materials (`baseColorFactor`, `emissiveFactor`, `alphaMode`, `doubleSided`, metallic 0),
/// `POSITION` (min/max) / `NORMAL` / `TEXCOORD_0` / `COLOR_0` (only when not all ones), sparse morph targets
/// named in `mesh.extras.targetNames`, and textures as embedded PNG. Skinning is not written.
pub fn write_glb(mesh: &MeshData) -> Vec<u8> {
    let mut bin = BinWriter::default();
    let mut accessors: Vec<Value> = Vec::new();
    let n = mesh.positions.len();
    let mut root = json!({
        "asset": {"version": "2.0", "generator": concat!("zena-avatar-mcp ", env!("CARGO_PKG_VERSION"))},
        "scene": 0,
        "scenes": [{"nodes": []}],
    });

    // Materials (a default one when primitives need it).
    let mut materials: Vec<Material> = mesh.materials.clone();
    if materials.is_empty() {
        materials.push(Material::default());
    }
    let clamp01 = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    let mut textures_json = Vec::new();
    let mut images_json = Vec::new();
    for (i, t) in mesh.textures.iter().enumerate() {
        let view = bin.view(&encode_png(t), None);
        images_json.push(json!({"bufferView": view, "mimeType": "image/png"}));
        textures_json.push(json!({"sampler": 0, "source": i}));
    }
    let materials_json: Vec<Value> = materials
        .iter()
        .map(|m| {
            let mut pbr = json!({
                "baseColorFactor": m.base_color.map(clamp01),
                "metallicFactor": 0.0,
                "roughnessFactor": 0.6,
            });
            if let Some(t) = m.texture.filter(|t| *t < mesh.textures.len()) {
                pbr["baseColorTexture"] = json!({"index": t});
            }
            let mut v = json!({
                "name": m.name,
                "pbrMetallicRoughness": pbr,
                "emissiveFactor": m.emissive.map(clamp01),
                "alphaMode": alpha_mode_str(m.alpha_mode),
                "doubleSided": m.double_sided,
            });
            if m.alpha_mode == AlphaMode::Mask {
                v["alphaCutoff"] = json!(m.alpha_cutoff);
            }
            v
        })
        .collect();

    if n > 0 {
        let (mn, mx) = mesh.bounds();
        let pos_view = bin.view(&f32_bytes(mesh.positions.iter().flatten()), Some(34962));
        accessors.push(json!({"bufferView": pos_view, "componentType": 5126, "count": n, "type": "VEC3", "min": mn, "max": mx}));
        let mut attributes = json!({"POSITION": 0});
        let normals: Vec<[f32; 3]> = (0..n).map(|i| mesh.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0])).collect();
        let nv = bin.view(&f32_bytes(normals.iter().flatten()), Some(34962));
        accessors.push(json!({"bufferView": nv, "componentType": 5126, "count": n, "type": "VEC3"}));
        attributes["NORMAL"] = json!(accessors.len() - 1);
        let uvs: Vec<[f32; 2]> = (0..n).map(|i| mesh.uvs.get(i).copied().unwrap_or([0.0; 2])).collect();
        let uv = bin.view(&f32_bytes(uvs.iter().flatten()), Some(34962));
        accessors.push(json!({"bufferView": uv, "componentType": 5126, "count": n, "type": "VEC2"}));
        attributes["TEXCOORD_0"] = json!(accessors.len() - 1);
        if mesh.colors.len() == n && mesh.colors.iter().any(|c| *c != [1.0; 4]) {
            let cv = bin.view(&f32_bytes(mesh.colors.iter().flatten()), Some(34962));
            accessors.push(json!({"bufferView": cv, "componentType": 5126, "count": n, "type": "VEC4"}));
            attributes["COLOR_0"] = json!(accessors.len() - 1);
        }

        // Morph targets as sparse accessors (zero-filled accessor when a target has no deltas).
        let mut targets = Vec::new();
        for m in &mesh.morphs {
            let mut acc: std::collections::BTreeMap<u32, [f32; 3]> = std::collections::BTreeMap::new();
            for (v, d) in &m.deltas {
                if (*v as usize) < n {
                    let e = acc.entry(*v).or_insert([0.0; 3]);
                    for k in 0..3 {
                        e[k] += d[k];
                    }
                }
            }
            let (mut dmn, mut dmx) = if acc.len() < n {
                ([0.0f32; 3], [0.0f32; 3])
            } else {
                ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3])
            };
            for d in acc.values() {
                for k in 0..3 {
                    dmn[k] = dmn[k].min(d[k]);
                    dmx[k] = dmx[k].max(d[k]);
                }
            }
            let mut a = json!({"componentType": 5126, "count": n, "type": "VEC3", "min": dmn, "max": dmx});
            if !acc.is_empty() {
                let idx: Vec<u32> = acc.keys().copied().collect();
                let vals: Vec<f32> = acc.values().flatten().copied().collect();
                let iv = bin.view(&u32_bytes(&idx), None);
                let vv = bin.view(&f32_bytes(&vals), None);
                a["sparse"] = json!({
                    "count": idx.len(),
                    "indices": {"bufferView": iv, "componentType": 5125},
                    "values": {"bufferView": vv},
                });
            }
            accessors.push(a);
            targets.push(json!({"POSITION": accessors.len() - 1}));
        }

        let iview = bin.view(&u32_bytes(&mesh.indices), Some(34963));
        let prims: Vec<Primitive> = if mesh.primitives.is_empty() {
            vec![Primitive {
                first_index: 0,
                index_count: (mesh.indices.len() / 3 * 3) as u32,
                material: 0,
            }]
        } else {
            mesh.primitives.clone()
        };
        let mut prims_json = Vec::new();
        for p in prims {
            if p.index_count == 0 || (p.first_index + p.index_count) as usize > mesh.indices.len() {
                continue;
            }
            accessors.push(json!({
                "bufferView": iview, "byteOffset": p.first_index * 4, "componentType": 5125,
                "count": p.index_count, "type": "SCALAR",
            }));
            let mut pj = json!({
                "attributes": attributes,
                "indices": accessors.len() - 1,
                "material": p.material.min(materials.len() - 1),
                "mode": 4,
            });
            if !targets.is_empty() {
                pj["targets"] = json!(targets);
            }
            prims_json.push(pj);
        }
        if !prims_json.is_empty() {
            let mut mj = json!({"name": "zena", "primitives": prims_json});
            if !mesh.morphs.is_empty() {
                mj["weights"] = json!(vec![0.0; mesh.morphs.len()]);
                mj["extras"] = json!({"targetNames": mesh.morphs.iter().map(|m| m.name.clone()).collect::<Vec<_>>()});
            }
            root["meshes"] = json!([mj]);
            root["nodes"] = json!([{"name": "zena", "mesh": 0}]);
            root["scenes"] = json!([{"nodes": [0]}]);
        }
    }
    root["materials"] = json!(materials_json);
    if !images_json.is_empty() {
        root["images"] = json!(images_json);
        root["textures"] = json!(textures_json);
        root["samplers"] = json!([{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}]);
    }
    if !accessors.is_empty() {
        root["accessors"] = json!(accessors);
    }
    while !bin.data.len().is_multiple_of(4) {
        bin.data.push(0);
    }
    if !bin.data.is_empty() {
        root["bufferViews"] = json!(bin.views);
        root["buffers"] = json!([{"byteLength": bin.data.len()}]);
    }
    pack_glb(&root, &bin.data)
}

// ---------------------------------------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------------------------------------

/// Reads `.glb` and `.vrm` (see the module docs for how nodes, skins and morph targets are mapped).
pub fn load_glb(bytes: &[u8]) -> Result<MeshData> {
    match std::panic::catch_unwind(|| load_glb_inner(bytes)) {
        Ok(r) => r,
        Err(_) => bail!("malformed glTF (reader panicked)"),
    }
}

fn target_names(extras: &gltf::json::Extras) -> Option<Vec<String>> {
    let raw = extras.as_ref()?;
    let v: Value = serde_json::from_str(raw.get()).ok()?;
    let names = v.get("targetNames")?.as_array()?;
    Some(names.iter().map(|x| x.as_str().unwrap_or_default().to_string()).collect())
}

struct JointTable {
    joints: Vec<usize>,
    inverse_bind: Vec<Mat4>,
}

impl JointTable {
    fn slot(&mut self, node: usize, ib: Mat4) -> u16 {
        let same = |a: &Mat4| (0..4).all(|c| (0..4).all(|r| (a[c][r] - ib[c][r]).abs() < 1e-4));
        if let Some(i) = (0..self.joints.len()).find(|&i| self.joints[i] == node && same(&self.inverse_bind[i])) {
            return i as u16;
        }
        self.joints.push(node);
        self.inverse_bind.push(ib);
        (self.joints.len() - 1) as u16
    }
}

fn load_glb_inner(bytes: &[u8]) -> Result<MeshData> {
    let gltf = gltf::Gltf::from_slice_without_validation(bytes).context("parsing glTF")?;
    let blob = gltf.blob.as_deref();
    let doc = &gltf.document;
    let buffers: Vec<Option<&[u8]>> = doc
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        })
        .collect();
    let get = |b: gltf::Buffer| buffers.get(b.index()).copied().flatten();

    let n_nodes = doc.nodes().len();
    let mut parent = vec![None; n_nodes];
    for n in doc.nodes() {
        for c in n.children() {
            if c.index() < n_nodes && c.index() != n.index() {
                parent[c.index()] = Some(n.index());
            }
        }
    }
    let skel_nodes: Vec<SkelNode> = doc
        .nodes()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            SkelNode {
                name: n.name().map(str::to_string).unwrap_or_else(|| format!("node_{}", n.index())),
                parent: parent[n.index()],
                translation: t,
                rotation: r,
                scale: s,
            }
        })
        .collect();
    // Exact world matrices from the file's own local transforms (matrix or TRS).
    let local: Vec<Mat4> = doc.nodes().map(|n| n.transform().matrix()).collect();
    let mut world = vec![math::IDENTITY; n_nodes];

    // Traversal order from the default (or first) scene; parentless nodes when there is no scene.
    let roots: Vec<usize> = match doc.default_scene().or_else(|| doc.scenes().next()) {
        Some(s) => s.nodes().map(|n| n.index()).collect(),
        None => (0..n_nodes).filter(|&i| parent[i].is_none()).collect(),
    };
    let mut order = Vec::new();
    let mut seen = vec![false; n_nodes];
    let mut stack: Vec<(usize, Mat4)> = roots.iter().rev().map(|&r| (r, math::IDENTITY)).collect();
    while let Some((i, pw)) = stack.pop() {
        if i >= n_nodes || seen[i] {
            continue;
        }
        seen[i] = true;
        world[i] = math::mat_mul(&pw, &local[i]);
        order.push(i);
        let node = doc.nodes().nth(i).expect("node index");
        let children: Vec<usize> = node.children().map(|c| c.index()).collect();
        for c in children.into_iter().rev() {
            stack.push((c, world[i]));
        }
    }
    let nodes: Vec<gltf::Node> = doc.nodes().collect();
    let any_skinned = order.iter().any(|&i| nodes[i].mesh().is_some() && nodes[i].skin().is_some());

    let mut out = MeshData::new();
    let mut joints = JointTable {
        joints: Vec::new(),
        inverse_bind: Vec::new(),
    };
    let mut skin_remap: HashMap<usize, Vec<u16>> = HashMap::new();
    let mut material_map: HashMap<Option<usize>, usize> = HashMap::new();
    let mut texture_map: HashMap<usize, Option<usize>> = HashMap::new();

    for &ni in &order {
        let node = &nodes[ni];
        let Some(mesh) = node.mesh() else { continue };
        let w = world[ni];
        let remap = node.skin().map(|skin| {
            skin_remap
                .entry(skin.index())
                .or_insert_with(|| {
                    let ibms: Vec<Mat4> = skin
                        .reader(get)
                        .read_inverse_bind_matrices()
                        .map(|it| it.collect())
                        .unwrap_or_default();
                    skin.joints()
                        .enumerate()
                        .map(|(j, jn)| joints.slot(jn.index(), ibms.get(j).copied().unwrap_or(math::IDENTITY)))
                        .collect()
                })
                .clone()
        });
        let rigid_slot = (any_skinned && remap.is_none()).then(|| joints.slot(ni, math::inverse_affine(&w)));
        let mesh_names = target_names(mesh.extras());
        let mut inst_morphs: Vec<MorphTarget> = Vec::new();
        let mut shared: HashMap<String, u32> = HashMap::new();

        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(get);
            let Some(pos_iter) = reader.read_positions() else { continue };
            let mut pos: Vec<[f32; 3]> = pos_iter.collect();
            let n = pos.len();
            if n == 0 {
                continue;
            }
            let mut nrm: Option<Vec<[f32; 3]>> = reader.read_normals().map(|i| i.collect()).filter(|v: &Vec<_>| v.len() == n);
            let mut uv: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect())
                .filter(|v: &Vec<_>| v.len() == n)
                .unwrap_or_else(|| vec![[0.0; 2]; n]);
            let mut col: Vec<[f32; 4]> = reader
                .read_colors(0)
                .map(|c| c.into_rgba_f32().collect())
                .filter(|v: &Vec<_>| v.len() == n)
                .unwrap_or_else(|| vec![[1.0; 4]; n]);
            let skin_data = remap.as_ref().and_then(|_| {
                let j: Vec<[u16; 4]> = reader.read_joints(0)?.into_u16().collect();
                let wts: Vec<[f32; 4]> = reader.read_weights(0)?.into_f32().collect();
                (j.len() == n && wts.len() == n).then_some((j, wts))
            });
            let mut idx: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..n as u32).collect());
            let mut tri: Vec<u32> = Vec::with_capacity(idx.len());
            for t in idx.chunks_exact(3) {
                if t.iter().all(|&i| (i as usize) < n) {
                    tri.extend_from_slice(t);
                }
            }
            idx = tri;
            let mut targets: Vec<Vec<[f32; 3]>> = reader
                .read_morph_targets()
                .map(|(p, _, _)| p.map(|it| it.collect::<Vec<_>>()).filter(|v| v.len() == n).unwrap_or_default())
                .collect();
            let (mut joints_v, mut weights_v) = match &skin_data {
                Some((j, w)) => (j.clone(), w.clone()),
                None => (Vec::new(), Vec::new()),
            };

            // Primitives sharing the same vertex accessors (common in VRM exports) share one vertex block.
            let key = {
                let mut attrs: Vec<String> = prim.attributes().map(|(s, a)| format!("{s:?}={}", a.index())).collect();
                attrs.sort();
                let tg: Vec<String> = prim
                    .morph_targets()
                    .map(|t| t.positions().map(|a| a.index().to_string()).unwrap_or_default())
                    .collect();
                format!("{}|{}", attrs.join(","), tg.join(","))
            };
            let had_normals = nrm.is_some();
            let reuse = if had_normals { shared.get(&key).copied() } else { None };

            let base = match reuse {
                Some(b) => b,
                None => {
                    if nrm.is_none() {
                        // Flat normals: one vertex per triangle corner.
                        let ex = |v: &Vec<[f32; 3]>| idx.iter().map(|&i| v[i as usize]).collect::<Vec<_>>();
                        pos = ex(&pos);
                        uv = idx.iter().map(|&i| uv[i as usize]).collect();
                        col = idx.iter().map(|&i| col[i as usize]).collect();
                        if !joints_v.is_empty() {
                            joints_v = idx.iter().map(|&i| joints_v[i as usize]).collect();
                            weights_v = idx.iter().map(|&i| weights_v[i as usize]).collect();
                        }
                        targets = targets.iter().map(|t| if t.is_empty() { Vec::new() } else { ex(t) }).collect();
                        let mut fn_ = Vec::with_capacity(pos.len());
                        for t in pos.chunks_exact(3) {
                            let fnm = math::normalize(math::cross(math::sub(t[1], t[0]), math::sub(t[2], t[0])));
                            fn_.extend([fnm; 3]);
                        }
                        nrm = Some(fn_);
                        idx = (0..pos.len() as u32).collect();
                    }
                    let mut nrm = nrm.take().unwrap_or_default();
                    let rigid = skin_data.is_none();
                    if rigid {
                        for p in &mut pos {
                            *p = math::transform_point(&w, *p);
                        }
                        for q in &mut nrm {
                            *q = math::transform_normal(&w, *q);
                        }
                        for t in &mut targets {
                            for d in t.iter_mut() {
                                *d = math::transform_vector(&w, *d);
                            }
                        }
                    }
                    let base = out.positions.len() as u32;
                    out.positions.extend_from_slice(&pos);
                    out.normals.extend_from_slice(&nrm);
                    out.uvs.extend_from_slice(&uv);
                    out.colors.extend_from_slice(&col);
                    if any_skinned {
                        if let (Some(remap), false) = (&remap, joints_v.is_empty()) {
                            for (j, wt) in joints_v.iter().zip(&weights_v) {
                                let sum: f32 = wt.iter().sum();
                                let wn = if sum > 1e-6 { wt.map(|x| x / sum) } else { [1.0, 0.0, 0.0, 0.0] };
                                out.joints.push(j.map(|x| remap.get(x as usize).copied().unwrap_or(0)));
                                out.weights.push(wn);
                            }
                        } else {
                            let slot = rigid_slot.unwrap_or_else(|| joints.slot(ni, math::inverse_affine(&w)));
                            out.joints.extend(std::iter::repeat_n([slot, 0, 0, 0], pos.len()));
                            out.weights.extend(std::iter::repeat_n([1.0, 0.0, 0.0, 0.0], pos.len()));
                        }
                    }
                    // Morph deltas (merged across primitives of one mesh instance).
                    let prim_names = target_names(prim.extras());
                    for (t, d) in targets.iter().enumerate() {
                        while inst_morphs.len() <= t {
                            let k = inst_morphs.len();
                            let name = mesh_names
                                .as_ref()
                                .and_then(|v| v.get(k))
                                .or_else(|| prim_names.as_ref().and_then(|v| v.get(k)))
                                .cloned()
                                .unwrap_or_else(|| format!("morph_{k}"));
                            inst_morphs.push(MorphTarget {
                                name,
                                mesh: mesh.index(),
                                target: k,
                                deltas: Vec::new(),
                            });
                        }
                        for (i, dv) in d.iter().enumerate() {
                            if dv.iter().any(|c| c.abs() > 1e-7) {
                                inst_morphs[t].deltas.push((base + i as u32, *dv));
                            }
                        }
                    }
                    if had_normals {
                        shared.insert(key, base);
                    }
                    base
                }
            };

            let flip = skin_data.is_none() && math::det3(&w) < 0.0;
            let mat = material_index(&prim.material(), &mut out, &mut material_map, &mut texture_map, &get);
            let indices: Vec<u32> = idx
                .chunks_exact(3)
                .flat_map(|t| if flip { [t[0], t[2], t[1]] } else { [t[0], t[1], t[2]] })
                .map(|i| i + base)
                .collect();
            out.push_indices(mat, indices);
        }
        out.morphs.extend(inst_morphs);
    }
    if any_skinned {
        out.skeleton = Some(Skeleton {
            nodes: skel_nodes,
            joints: joints.joints,
            inverse_bind: joints.inverse_bind,
        });
    }
    Ok(out)
}

fn material_index<'a, 's, F>(
    m: &gltf::Material<'a>,
    out: &mut MeshData,
    cache: &mut HashMap<Option<usize>, usize>,
    textures: &mut HashMap<usize, Option<usize>>,
    get: &F,
) -> usize
where
    F: Fn(gltf::Buffer<'a>) -> Option<&'s [u8]>,
{
    if let Some(&i) = cache.get(&m.index()) {
        return i;
    }
    let pbr = m.pbr_metallic_roughness();
    let texture = pbr.base_color_texture().and_then(|info| {
        let img = info.texture().source();
        *textures.entry(img.index()).or_insert_with(|| {
            let decoded = match img.source() {
                gltf::image::Source::View { view, .. } => get(view.buffer())
                    .and_then(|data| data.get(view.offset()..view.offset() + view.length()))
                    .and_then(|bytes| image::load_from_memory(bytes).ok()),
                gltf::image::Source::Uri { .. } => None,
            };
            decoded.map(|d| {
                let rgba = d.to_rgba8();
                out.textures.push(Texture {
                    width: rgba.width(),
                    height: rgba.height(),
                    rgba8: rgba.into_raw(),
                });
                out.textures.len() - 1
            })
        })
    });
    let mat = Material {
        name: m
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| m.index().map(|i| format!("material_{i}")).unwrap_or_else(|| "default".into())),
        base_color: pbr.base_color_factor(),
        texture,
        alpha_mode: match m.alpha_mode() {
            gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
        double_sided: m.double_sided(),
        emissive: m.emissive_factor(),
    };
    let i = out.add_material(mat);
    cache.insert(m.index(), i);
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube (24 vertices) with the top face in a second material and a "lift" morph on the top face.
    fn test_box() -> MeshData {
        let mut m = MeshData::new();
        let red = m.add_material(Material::new("red", [1.0, 0.0, 0.0, 1.0]));
        let blue = m.add_material(Material::new("blue", [0.0, 0.0, 1.0, 1.0]).with_emissive([0.1, 0.1, 0.5]));
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ];
        let mut top_base = 0;
        for (fi, (nrm, a, b)) in faces.iter().enumerate() {
            let c = |s: f32, t: f32| -> [f32; 3] { std::array::from_fn(|k| 0.5 * nrm[k] + s * 0.5 * a[k] + t * 0.5 * b[k]) };
            let pos = [c(-1.0, -1.0), c(1.0, -1.0), c(1.0, 1.0), c(-1.0, 1.0)];
            let mat = if fi == 5 { blue } else { red };
            let base = m.push_triangles(
                mat,
                &pos,
                &[*nrm; 4],
                &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                &[0, 1, 2, 0, 2, 3],
            );
            if fi == 5 {
                top_base = base;
            }
        }
        m.morphs.push(MorphTarget {
            name: "lift".into(),
            mesh: 0,
            target: 0,
            deltas: (0..4).map(|i| (top_base + i, [0.0, 0.5, 0.0])).collect(),
        });
        m
    }

    #[test]
    fn glb_round_trip_keeps_geometry_materials_and_morphs() {
        let m = test_box();
        m.validate().unwrap();
        let glb = write_glb(&m);
        let back = load_glb(&glb).unwrap();
        back.validate().unwrap();
        assert_eq!(back.positions.len(), 24);
        assert_eq!(back.indices.len(), 36);
        assert_eq!(back.materials.len(), 2);
        assert_eq!(back.primitives.len(), 2, "one draw per material");
        assert_eq!(back.materials[1].name, "blue");
        assert_eq!(back.materials[1].emissive, [0.1, 0.1, 0.5]);
        assert_eq!(back.morph_index("LIFT"), Some(0));
        assert_eq!(back.morphs[0].deltas.len(), 4);
        let mut out = Vec::new();
        back.apply_morphs(&[(0, 1.0)], &mut out);
        assert!((back.bounds().1[1] - 0.5).abs() < 1e-6);
        assert!((out.iter().map(|p| p[1]).fold(f32::MIN, f32::max) - 1.0).abs() < 1e-6);
        assert!(back.skeleton.is_none());
        let (json, _) = split_glb(&glb).unwrap();
        assert_eq!(json["meshes"][0]["extras"]["targetNames"][0], "lift");
    }

    #[test]
    fn textures_survive_the_round_trip() {
        let mut m = test_box();
        m.textures.push(Texture {
            width: 2,
            height: 1,
            rgba8: vec![255, 0, 0, 255, 0, 255, 0, 128],
        });
        m.materials[0].texture = Some(0);
        let back = load_glb(&write_glb(&m)).unwrap();
        assert_eq!(back.textures.len(), 1);
        assert_eq!(back.textures[0].rgba8, vec![255, 0, 0, 255, 0, 255, 0, 128]);
        assert_eq!(back.materials[0].texture, Some(0));
    }

    #[test]
    fn append_transforms_and_remaps() {
        let mut a = test_box();
        let b = test_box();
        a.append(
            &b,
            math::mat_mul(&math::translation([3.0, 0.0, 0.0]), &math::scaling([2.0, 2.0, 2.0])),
        );
        a.validate().unwrap();
        assert_eq!(a.positions.len(), 48);
        assert_eq!(a.materials.len(), 4);
        assert_eq!(a.primitives.last().unwrap().material, 3);
        let (mn, mx) = a.bounds();
        assert_eq!(mn, [-0.5, -1.0, -1.0]);
        assert_eq!(mx, [4.0, 1.0, 1.0]);
        assert_eq!(a.morphs.len(), 2);
        assert_eq!(a.morphs[1].mesh, 1);
        assert!(
            a.morphs[1].deltas.iter().all(|(v, d)| *v >= 24 && d[1] == 1.0),
            "deltas are offset and scaled"
        );
        // A mirror keeps faces pointing outward (winding flipped).
        let mut c = MeshData::new();
        c.append(&b, math::scaling([-1.0, 1.0, 1.0]));
        let t = &c.indices[0..3];
        let p: Vec<[f32; 3]> = t.iter().map(|&i| c.positions[i as usize]).collect();
        let geo = math::cross(math::sub(p[1], p[0]), math::sub(p[2], p[0]));
        assert!(math::dot(geo, c.normals[t[0] as usize]) > 0.0);
    }

    fn two_bone_skeleton() -> Skeleton {
        let nodes = vec![
            SkelNode {
                name: "root".into(),
                parent: None,
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
            SkelNode {
                name: "Arm".into(),
                parent: Some(0),
                translation: [1.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        ];
        let mut sk = Skeleton {
            nodes,
            joints: vec![0, 1],
            inverse_bind: Vec::new(),
        };
        sk.inverse_bind = sk.world_matrices(&[]).iter().map(math::inverse_affine).collect();
        sk
    }

    #[test]
    fn joint_matrices_identity_at_rest_and_rotation_moves_vertices() {
        let sk = two_bone_skeleton();
        assert_eq!(sk.node_index("arm"), Some(1));
        for m in sk.joint_matrices(&[]) {
            for c in 0..4 {
                for r in 0..4 {
                    assert!((m[c][r] - IDENTITY[c][r]).abs() < 1e-6);
                }
            }
        }
        // Rotate the arm 90° about +Z: a vertex at (2, 0, 0) bound to it swings to (1, 1, 0).
        let q = math::quat_axis_angle([0.0, 0.0, 1.0], std::f32::consts::FRAC_PI_2);
        let jm = sk.joint_matrices(&[(1, q)]);
        let p = math::transform_point(&jm[1], [2.0, 0.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-5 && (p[1] - 1.0).abs() < 1e-5 && p[2].abs() < 1e-5, "{p:?}");
        let root = math::transform_point(&jm[0], [2.0, 0.0, 0.0]);
        assert!((root[0] - 2.0).abs() < 1e-6, "the parent joint is unaffected");
    }

    #[test]
    fn skinned_gltf_loads_in_bind_space() {
        // Hand-built skinned triangle: two joints, the mesh node itself is offset (and must be ignored).
        let mut bin = BinWriter::default();
        let pos: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let joints: [u8; 12] = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];
        let weights: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let ibm: Vec<f32> = [math::IDENTITY, math::translation([-1.0, 0.0, 0.0])]
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let pv = bin.view(&f32_bytes(&pos), None);
        let jv = bin.view(&joints, None);
        let wv = bin.view(&f32_bytes(&weights), None);
        let iv = bin.view(&f32_bytes(&ibm), None);
        let doc = json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0, 3]}],
            "nodes": [
                {"name": "Hips", "children": [1]},
                {"name": "Arm", "translation": [1.0, 0.0, 0.0]},
                {"name": "unused"},
                {"name": "body", "mesh": 0, "skin": 0, "translation": [5.0, 5.0, 5.0]}
            ],
            "skins": [{"joints": [0, 1], "inverseBindMatrices": 3}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
            "accessors": [
                {"bufferView": pv, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]},
                {"bufferView": jv, "componentType": 5121, "count": 3, "type": "VEC4"},
                {"bufferView": wv, "componentType": 5126, "count": 3, "type": "VEC4"},
                {"bufferView": iv, "componentType": 5126, "count": 2, "type": "MAT4"}
            ],
            "bufferViews": bin.views,
            "buffers": [{"byteLength": bin.data.len()}],
        });
        let m = load_glb(&pack_glb(&doc, &bin.data)).unwrap();
        m.validate().unwrap();
        assert_eq!(
            m.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            "bind space, node offset ignored"
        );
        assert_eq!(m.joints[1], [1, 0, 0, 0]);
        assert_eq!(m.weights[2], [1.0, 0.0, 0.0, 0.0], "weights are normalised");
        let sk = m.skeleton.as_ref().unwrap();
        assert_eq!(sk.nodes.len(), 4);
        assert_eq!(sk.nodes[1].parent, Some(0));
        assert_eq!(sk.joints, vec![0, 1]);
        assert!(m.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6), "flat normals generated");
        let jm = sk.joint_matrices(&[(1, math::quat_axis_angle([0.0, 0.0, 1.0], std::f32::consts::PI))]);
        let p = math::transform_point(&jm[1], m.positions[2]);
        assert!((p[0] - 1.0).abs() < 1e-5 && (p[1] + 1.0).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(load_glb(b"definitely not a model").is_err());
        assert!(split_glb(b"glTF\x02\0\0\0").is_err());
        let empty = write_glb(&MeshData::new());
        assert_eq!(load_glb(&empty).unwrap().positions.len(), 0);
    }
}
