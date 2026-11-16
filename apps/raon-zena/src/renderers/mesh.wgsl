// Toon-shaded meshes ("MToon-lite"): two-band ramp, rim light, emissive, alpha mask / blend, GPU skinning.
// Positions come from their own buffer (morph targets are blended on the CPU and re-uploaded).

struct U {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    light: vec4<f32>,    // xyz: direction toward the light (world), w: ambient floor
    params: vec4<f32>,   // x: alpha (hover fade), y: time, z: skinned (1/0), w: unused
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> joints: array<mat4x4<f32>>;

struct M {
    base: vec4<f32>,
    emissive: vec4<f32>, // rgb emissive, w: alpha cutoff
    flags: vec4<f32>,    // x: alpha mode (0 opaque, 1 mask, 2 blend)
};
@group(1) @binding(0) var<uniform> m: M;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) joint: vec4<u32>,
    @location(5) weight: vec4<f32>,
};

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs(v: VIn) -> VOut {
    var skin = mat4x4<f32>(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
    if (u.params.z > 0.5) {
        skin = joints[v.joint.x] * v.weight.x + joints[v.joint.y] * v.weight.y
             + joints[v.joint.z] * v.weight.z + joints[v.joint.w] * v.weight.w;
    }
    let world = u.model * skin * vec4(v.pos, 1.0);
    var o: VOut;
    o.pos = u.view_proj * world;
    o.normal = normalize((u.model * skin * vec4(v.normal, 0.0)).xyz);
    o.uv = v.uv;
    o.color = v.color;
    return o;
}

@fragment
fn fs(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let base = m.base * textureSample(tex, samp, i.uv) * i.color;
    if (m.flags.x > 0.5 && m.flags.x < 1.5 && base.a < m.emissive.w) {
        discard;
    }
    var n = normalize(i.normal);
    if (!front) {
        n = -n;
    }
    let l = normalize(u.light.xyz);
    let lit = smoothstep(-0.05, 0.12, dot(n, l));
    let shade = mix(0.62, 1.0, lit);
    let rim = pow(1.0 - max(n.z, 0.0), 3.0) * 0.22;
    let rgb = base.rgb * (u.light.w + (1.0 - u.light.w) * shade) + base.rgb * rim + m.emissive.rgb;
    var a = u.params.x;
    if (m.flags.x > 1.5) {
        a = a * base.a;
    }
    return vec4(rgb * a, a);
}
