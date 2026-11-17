// Toon-shaded meshes ("MToon-lite"): half-Lambert soft ramp into a violet shade colour, Fresnel rim light, emissive,
// alpha mask / blend, GPU skinning.
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

// Tone of the toon ramp. The shaded side is the base colour times a warm violet (Zena's palette, `ZENA_PURPLE`
// lifted to a multiplier) instead of a grey, so faces and hair shade into colour rather than mud; its brightness
// matches the old grey floor (ambient + 62 % of the rest).
const SHADE_TINT: vec3<f32> = vec3(0.80, 0.66, 0.90);
const SHADE_TINT_LUMA: f32 = 0.7071; // Rec. 709 luma of SHADE_TINT
// Half-Lambert ramp edges: the lit/shaded border spans N·L from about -0.4 to +0.3, so noses, cheeks and low-poly
// faces get a soft gradient instead of a hard polygon edge.
const RAMP_LO: f32 = 0.30;
const RAMP_HI: f32 = 0.65;
// Fresnel rim: a faint lilac-white backlight around the silhouette (stronger on the shaded side) so the character
// stays readable over dark game scenes. The camera is orthographic along -z, so the view vector is +z.
const RIM_COLOR: vec3<f32> = vec3(0.92, 0.86, 1.0);
const RIM_POWER: f32 = 3.0;
const RIM_STRENGTH: f32 = 0.38;

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
    let half_lambert = dot(n, l) * 0.5 + 0.5;
    let lit = smoothstep(RAMP_LO, RAMP_HI, half_lambert);
    let shade_floor = u.light.w + (1.0 - u.light.w) * 0.62;
    let shaded = base.rgb * SHADE_TINT * (shade_floor / SHADE_TINT_LUMA);
    var rgb = mix(shaded, base.rgb, lit);
    let fresnel = pow(1.0 - clamp(n.z, 0.0, 1.0), RIM_POWER);
    rgb = rgb + mix(base.rgb, RIM_COLOR, 0.6) * fresnel * RIM_STRENGTH * mix(1.0, 0.55, lit);
    rgb = rgb + m.emissive.rgb;
    var a = u.params.x;
    if (m.flags.x > 1.5) {
        a = a * base.a;
    }
    return vec4(min(rgb, vec3(1.0)) * a, a);
}
