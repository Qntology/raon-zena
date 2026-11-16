// Textured quad in NDC with rotation; premultiplied-alpha textures (sRGB, premultiplied on upload).

struct U {
    centre: vec2<f32>,   // NDC
    half: vec2<f32>,     // half size in NDC
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    angle: f32,          // radians, counter-clockwise
    alpha: f32,
    aspect: f32,         // viewport height / width (keeps rotation round)
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var corner = array<vec2<f32>, 4>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    let c = corner[i];
    // Rotate in a square space (x in NDC, y scaled by the aspect), then back.
    let local = vec2(c.x * u.half.x, c.y * u.half.y * u.aspect);
    let s = sin(u.angle);
    let k = cos(u.angle);
    let r = vec2(local.x * k - local.y * s, local.x * s + local.y * k);
    var o: VOut;
    o.pos = vec4(u.centre + vec2(r.x, r.y / u.aspect), 0.0, 1.0);
    let t = c * 0.5 + 0.5;
    o.uv = vec2(mix(u.uv0.x, u.uv1.x, t.x), mix(u.uv1.y, u.uv0.y, t.y));
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, i.uv) * u.alpha;
}
