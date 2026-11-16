// Procedural placeholder avatar for Raon Zena: a face whose eyes follow the gaze target, whose
// brows/mouth follow the emotion, and whose mouth opens with the TTS RMS. It exercises the full
// tool-call → avatar-parameter path until a Live2D / VRM renderer is plugged in.

struct U {
    gaze: vec2<f32>,      // eyeball x/y in [-1, 1] (ParamEyeBallX/Y)
    head: vec2<f32>,      // head yaw/pitch in [-1, 1]
    mouth_open: f32,      // ParamMouthOpenY
    mouth_form: f32,      // ParamMouthForm (-1 pout .. +1 smile)
    brow: f32,            // -1 worried .. +1 angry
    eye_open: f32,        // 0..1.4 (surprise widens)
    alpha: f32,           // window hover fade
    time: f32,
    aspect: f32,          // viewport height / width
    _pad: f32,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    var o: VOut;
    o.pos = vec4(p[i], 0.0, 1.0);
    o.uv = vec2(p[i].x, p[i].y * u.aspect);
    return o;
}

fn ellipse(p: vec2<f32>, c: vec2<f32>, r: vec2<f32>) -> f32 {
    let d = (p - c) / r;
    return length(d) - 1.0;
}

fn fill(d: f32) -> f32 { return clamp(0.5 - d * 60.0, 0.0, 1.0); }

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let bob = sin(u.time * 2.0) * 0.01;
    let p = i.uv - vec2(u.head.x * 0.08, u.head.y * 0.06 + 0.25 + bob);
    var col = vec3(0.0);
    var a = 0.0;
    // hair + face
    let hair = fill(ellipse(p, vec2(0.0, 0.08), vec2(0.66, 0.7)));
    col = mix(col, vec3(0.30, 0.22, 0.45), hair); a = max(a, hair);
    let face = fill(ellipse(p, vec2(0.0, -0.02), vec2(0.52, 0.58)));
    col = mix(col, vec3(1.0, 0.89, 0.82), face); a = max(a, face);
    // eyes
    for (var s = -1.0; s <= 1.0; s += 2.0) {
        let ec = vec2(0.2 * s, 0.05);
        let open = 0.11 * clamp(u.eye_open, 0.1, 1.4) * (select(1.0, 0.1, fract(u.time * 0.23) > 0.97));
        let white = fill(ellipse(p, ec, vec2(0.12, open)));
        col = mix(col, vec3(1.0), white);
        let pc = ec + vec2(u.gaze.x * 0.055, u.gaze.y * 0.04 * clamp(u.eye_open, 0.2, 1.0));
        let pupil = fill(ellipse(p, pc, vec2(0.055, min(0.065, open)))) * white;
        col = mix(col, vec3(0.42, 0.25, 0.70), pupil);
        // brows tilt: angry slants inward-down, worried inward-up
        let bx = p.x - 0.2 * s;
        let by = p.y - (0.22 - u.brow * 0.06 * bx * s * 8.0);
        let brow = fill(ellipse(vec2(bx, by), vec2(0.0, 0.0), vec2(0.13, 0.018)));
        col = mix(col, vec3(0.25, 0.17, 0.35), brow);
    }
    // mouth: curve by form, height by open
    let mx = p.x;
    let curve = -u.mouth_form * 0.12 * (1.0 - (mx / 0.14) * (mx / 0.14));
    let mouth = fill(ellipse(vec2(mx, p.y + 0.26 + curve), vec2(0.0, 0.0), vec2(0.14 - 0.04 * u.mouth_open, 0.012 + 0.09 * u.mouth_open)));
    col = mix(col, vec3(0.65, 0.2, 0.3), mouth * face);
    // cheeks
    for (var s = -1.0; s <= 1.0; s += 2.0) {
        let ch = fill(ellipse(p, vec2(0.3 * s, -0.12), vec2(0.07, 0.035))) * 0.35;
        col = mix(col, vec3(1.0, 0.6, 0.65), ch * face);
    }
    let alpha = a * u.alpha;
    return vec4(col * alpha, alpha); // premultiplied
}
