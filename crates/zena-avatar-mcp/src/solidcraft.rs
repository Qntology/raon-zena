//! Track A bridge to SolidCraft's headless MCP server (`solidcraft-cli mcp --allow-path DIR`).
//!
//! Each asset is one session round: `new_design` → `batch {commands, rollback: true}` →
//! `export {path, format: "glb"}` (binary glTF in metres, Y up, body colours kept), then the file is checked
//! with [`crate::mesh::load_glb`]. The robot body additionally gets the forge's morphing face grafted on
//! ([`crate::forge::robot_face`]) because SolidCraft exports carry no morph targets.
//!
//! **Unverified value formats.** SolidCraft's `docs/mcp.md` names the batch commands and their parameters but
//! not the value syntax, so the built-in scripts are a best guess: lengths in millimetres, points / vectors as
//! `[x, y, z]` arrays, sketch planes `"XY" | "XZ" | "YZ"`, sketch points as `[u, v]`, colours `"#RRGGBB"`,
//! axes `"X" | "Y" | "Z"`, angles in degrees, `operation: "new_body"`, bodies named Fusion-style `Body1, Body2…`
//! in creation order and edge selectors `"<body>/edges"`. Modelling is Z-up with the viewer at −Y (the glb
//! exporter's Y-up conversion then makes −Y the +Z front). To correct any of this without recompiling, drop a
//! `<template_dir>/<name>.json` (`prop_<EMOTE>`, `part_<PART>`, `robot`) holding a command array (or
//! `{"commands": [...]}`) with `{{angle_deg}}`-style placeholders — see [`load_template`]. On connect the bridge
//! calls `list_commands` once and logs which needed commands are missing; any failure is returned as `Err` so
//! callers fall back to the built-in [`crate::forge`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use zena_core::tools::{AvatarPart, CharacterEmote};

use crate::protocol::{result_json, McpClient};

/// Batch commands the built-in scripts use (all documented in SolidCraft's `docs/mcp.md`).
pub const NEEDED_COMMANDS: [&str; 9] = [
    "sketch.create",
    "sketch.rectangle.two_point",
    "sketch.circle.center",
    "sketch.finish",
    "solid.box",
    "solid.extrude",
    "solid.fillet",
    "solid.move",
    "appearance.assign",
];

const TOOL_TIMEOUT: Duration = Duration::from_secs(120);

/// A live SolidCraft MCP session.
pub struct SolidCraftBridge {
    client: McpClient,
    allow_dir: PathBuf,
    template_dir: Option<PathBuf>,
    missing: Vec<String>,
}

/// Every string that looks like a command id in a `list_commands` result (arrays of strings, `id` / `command`
/// / `name` fields, or dotted object keys).
fn command_ids(v: &Value, out: &mut HashSet<String>) {
    match v {
        Value::String(s) if s.contains('.') && !s.contains(' ') => {
            out.insert(s.clone());
        }
        Value::Array(a) => a.iter().for_each(|x| command_ids(x, out)),
        Value::Object(o) => {
            for (k, x) in o {
                if k.contains('.') && !k.contains(' ') {
                    out.insert(k.clone());
                }
                match (k.as_str(), x) {
                    ("id" | "command" | "name", Value::String(s)) => {
                        out.insert(s.clone());
                    }
                    (_, Value::Array(_) | Value::Object(_)) => command_ids(x, out),
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn display_path(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

impl SolidCraftBridge {
    /// Spawns `exe mcp --allow-path allow_dir`, checks the session tools and logs missing batch commands.
    pub fn connect(exe: &Path, allow_dir: &Path, template_dir: Option<&Path>) -> Result<Self> {
        std::fs::create_dir_all(allow_dir).with_context(|| format!("creating {}", allow_dir.display()))?;
        let allow_dir = allow_dir.canonicalize()?;
        let args = vec!["mcp".to_string(), "--allow-path".to_string(), display_path(&allow_dir)];
        let mut client = McpClient::spawn(exe, &args)?;
        let tools = client.list_tools()?;
        for t in ["new_design", "batch", "export"] {
            if !tools.iter().any(|x| x == t) {
                bail!("{} does not offer the `{t}` tool (has: {})", client.server_name(), tools.join(", "));
            }
        }
        let mut missing = Vec::new();
        if tools.iter().any(|x| x == "list_commands") {
            match client.call_tool("list_commands", json!({}), Duration::from_secs(30)) {
                Ok(r) => {
                    let mut ids = HashSet::new();
                    command_ids(&result_json(&r), &mut ids);
                    if ids.is_empty() {
                        eprintln!("[solidcraft] could not read list_commands output; assuming every command exists");
                    } else {
                        missing = NEEDED_COMMANDS
                            .iter()
                            .filter(|c| !ids.contains(**c))
                            .map(|c| c.to_string())
                            .collect();
                    }
                }
                Err(e) => eprintln!("[solidcraft] list_commands failed: {e:#}"),
            }
        }
        if !missing.is_empty() {
            eprintln!(
                "[solidcraft] missing batch commands: {} (templates or the built-in forge cover them)",
                missing.join(", ")
            );
        }
        Ok(SolidCraftBridge {
            client,
            allow_dir,
            template_dir: template_dir.map(Path::to_path_buf),
            missing,
        })
    }

    /// Needed batch commands `list_commands` did not report.
    pub fn missing_commands(&self) -> &[String] {
        &self.missing
    }

    pub fn server_name(&self) -> &str {
        self.client.server_name()
    }

    fn script(&self, name: &str, vars: &[(&str, f64)], builtin: impl FnOnce() -> Option<Vec<Value>>) -> Result<Vec<Value>> {
        if let Some(dir) = &self.template_dir {
            if let Some(s) = load_template(dir, name, vars) {
                return Ok(s);
            }
        }
        let s = builtin().with_context(|| format!("no SolidCraft script for {name}"))?;
        let used: HashSet<&str> = s.iter().filter_map(|c| c.get("command").and_then(Value::as_str)).collect();
        let lacking: Vec<&String> = self.missing.iter().filter(|m| used.contains(m.as_str())).collect();
        if !lacking.is_empty() {
            bail!("SolidCraft lacks {lacking:?} needed by the built-in {name} script");
        }
        Ok(s)
    }

    fn output_path(&self, out: &Path, default_name: &str) -> Result<PathBuf> {
        let out = if out.is_dir() { out.join(default_name) } else { out.to_path_buf() };
        let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let parent = parent.canonicalize()?;
        if !parent.starts_with(&self.allow_dir) {
            bail!(
                "{} is outside the SolidCraft allow path {}",
                out.display(),
                self.allow_dir.display()
            );
        }
        Ok(parent.join(out.file_name().context("output file name")?))
    }

    fn run(&mut self, design: &str, commands: Vec<Value>, out: &Path) -> Result<PathBuf> {
        let out = self.output_path(out, &format!("{design}.glb"))?;
        let _ = std::fs::remove_file(&out);
        self.client.call_tool("new_design", json!({"name": design}), TOOL_TIMEOUT)?;
        self.client
            .call_tool("batch", json!({"commands": commands, "rollback": true}), TOOL_TIMEOUT)?;
        self.client
            .call_tool("export", json!({"path": display_path(&out), "format": "glb"}), TOOL_TIMEOUT)?;
        let bytes = std::fs::read(&out).with_context(|| format!("SolidCraft reported success but {} is missing", out.display()))?;
        let mesh = crate::mesh::load_glb(&bytes).context("SolidCraft export does not parse")?;
        if mesh.positions.is_empty() {
            bail!("SolidCraft exported an empty model");
        }
        Ok(out)
    }

    /// Builds an emote prop; `out` is a `.glb` path (or a directory for `prop_<EMOTE>.glb`) inside the allow dir.
    pub fn forge_prop(&mut self, emote: CharacterEmote, angle_deg: f32, out: &Path) -> Result<PathBuf> {
        let a = angle_deg as f64;
        let vars = [("angle_deg", a), ("neg_angle_deg", -a), ("angle_rad", a.to_radians())];
        let name = format!("prop_{}", emote.name());
        let s = self.script(&name, &vars, || prop_script(emote, angle_deg))?;
        self.run(&name, s, out)
    }

    pub fn forge_part(&mut self, part: AvatarPart, out: &Path) -> Result<PathBuf> {
        let name = format!("part_{}", part.name());
        let s = self.script(&name, &[], || Some(part_script(part)))?;
        self.run(&name, s, out)
    }

    /// The Track A robot body, with the forge face (and its VRM-named morph targets) grafted on.
    pub fn forge_robot(&mut self, out: &Path) -> Result<PathBuf> {
        let s = self.script("robot", &[], || Some(robot_script()))?;
        let path = self.run("robot", s, out)?;
        let mut body = crate::mesh::load_glb(&std::fs::read(&path)?)?;
        body.append(&crate::forge::robot_face(), crate::mesh::IDENTITY);
        std::fs::write(&path, crate::mesh::write_glb(&body))?;
        Ok(path)
    }
}

/// Loads `<dir>/<name>.json` (a command array or `{"commands": [...]}`), substituting `{{var}}` placeholders —
/// a quoted `"{{var}}"` becomes a JSON number, an embedded one is replaced in place. `None` when the file is
/// absent or malformed (malformed files are logged).
pub fn load_template(dir: &Path, name: &str, vars: &[(&str, f64)]) -> Option<Vec<Value>> {
    let path = dir.join(format!("{name}.json"));
    let mut text = std::fs::read_to_string(&path).ok()?;
    for (k, v) in vars {
        let num = if v.is_finite() { format!("{v}") } else { "0".into() };
        text = text
            .replace(&format!("\"{{{{{k}}}}}\""), &num)
            .replace(&format!("{{{{{k}}}}}"), &num);
    }
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[solidcraft] template {} is not valid JSON: {e}", path.display());
            return None;
        }
    };
    let cmds = match parsed {
        Value::Array(a) => a,
        Value::Object(mut o) => match o.remove("commands") {
            Some(Value::Array(a)) => a,
            _ => return None,
        },
        _ => return None,
    };
    let mut out = Vec::with_capacity(cmds.len());
    for c in cmds {
        let Some(command) = c.get("command").and_then(Value::as_str) else {
            eprintln!("[solidcraft] template {} has an element without `command`", path.display());
            return None;
        };
        let params = c.get("params").cloned().unwrap_or_else(|| json!({}));
        out.push(json!({"command": command, "params": params}));
    }
    Some(out)
}

/// `SOLIDCRAFT_CLI`, then `solidcraft-cli(.exe)` on `PATH`, then `C:\Program Files\SolidCraft\solidcraft-cli.exe`.
pub fn find_solidcraft() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SOLIDCRAFT_CLI").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let names: &[&str] = if cfg!(windows) {
        &["solidcraft-cli.exe", "solidcraft-cli"]
    } else {
        &["solidcraft-cli"]
    };
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            for n in names {
                let p = dir.join(n);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    let pf = Path::new(r"C:\Program Files\SolidCraft\solidcraft-cli.exe");
    pf.is_file().then(|| pf.to_path_buf())
}

// ---------------------------------------------------------------------------------------------------------
// Script builders (forge coordinates in metres, converted to Z-up millimetres)
// ---------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Script {
    cmds: Vec<Value>,
    bodies: usize,
}

fn mm(v: f32) -> f64 {
    ((v as f64) * 1000.0 * 100.0).round() / 100.0
}

/// glTF (Y up, +Z front, metres) → CAD (Z up, −Y front, millimetres).
fn cad(p: [f32; 3]) -> [f64; 3] {
    [mm(p[0]), mm(-p[2]), mm(p[1])]
}

impl Script {
    fn cmd(&mut self, command: &str, params: Value) {
        self.cmds.push(json!({"command": command, "params": params}));
    }

    fn new_body(&mut self, color: &str) -> String {
        self.bodies += 1;
        let b = format!("Body{}", self.bodies);
        self.cmd("appearance.assign", json!({"bodies": [b], "color": color}));
        b
    }

    fn fillet(&mut self, body: &str, radius: f32) {
        if radius > 0.0 {
            self.cmd("solid.fillet", json!({"edges": [format!("{body}/edges")], "radius": mm(radius)}));
        }
    }

    /// Axis-aligned box at a forge-space centre / size.
    fn block(&mut self, centre: [f32; 3], size: [f32; 3], color: &str, fillet: f32) -> String {
        let c = cad(centre);
        let s = [mm(size[0]), mm(size[2]), mm(size[1])];
        let corner = [c[0] - s[0] / 2.0, c[1] - s[1] / 2.0, c[2] - s[2] / 2.0];
        self.cmd(
            "solid.box",
            json!({"corner": corner, "length": s[0], "width": s[1], "height": s[2], "operation": "new_body", "targets": []}),
        );
        let b = self.new_body(color);
        self.fillet(&b, fillet);
        b
    }

    /// A disc facing the viewer: a circle on the front (XZ) plane extruded `depth`.
    fn disc(&mut self, centre: [f32; 2], radius: f32, depth: f32, color: &str) -> String {
        self.cmd("sketch.create", json!({"plane": "XZ"}));
        self.cmd(
            "sketch.circle.center",
            json!({"center": [mm(centre[0]), mm(centre[1])], "radius": mm(radius)}),
        );
        self.cmd("sketch.finish", json!({}));
        self.cmd("solid.extrude", json!({"distance": mm(depth), "operation": "new_body"}));
        let b = self.new_body(color);
        self.fillet(&b, depth * 0.3);
        b
    }

    /// A flat plate facing the viewer: a rectangle on the front plane extruded `depth`.
    fn plate(&mut self, p0: [f32; 2], p1: [f32; 2], depth: f32, color: &str) -> String {
        self.cmd("sketch.create", json!({"plane": "XZ"}));
        self.cmd(
            "sketch.rectangle.two_point",
            json!({"p0": [mm(p0[0]), mm(p0[1])], "p1": [mm(p1[0]), mm(p1[1])]}),
        );
        self.cmd("sketch.finish", json!({}));
        self.cmd("solid.extrude", json!({"distance": mm(depth), "operation": "new_body"}));
        self.new_body(color)
    }

    /// Rotates bodies about the CAD axis through the origin; `view_deg` is counter-clockwise as seen by the viewer
    /// (about the −Y view axis, i.e. `angle = −view_deg` about +Y).
    fn spin(&mut self, bodies: &[String], view_deg: f32) {
        if !bodies.is_empty() && view_deg.abs() > 1e-3 {
            self.cmd("solid.move", json!({"bodies": bodies, "axis": "Y", "angle": -(view_deg as f64)}));
        }
    }

    fn heart(&mut self, centre: [f32; 2], scale: f32, color: &str) -> Vec<String> {
        let [cx, cy] = centre;
        let mut b = vec![
            self.disc([cx - 0.045 * scale, cy + 0.025 * scale], 0.05 * scale, 0.05 * scale, color),
            self.disc([cx + 0.045 * scale, cy + 0.025 * scale], 0.05 * scale, 0.05 * scale, color),
        ];
        for (i, w) in [0.19f32, 0.15, 0.11, 0.07, 0.03].iter().enumerate() {
            let y = cy + (0.01 - 0.019 * i as f32) * scale;
            b.push(self.block([cx, y, 0.0], [w * scale, 0.02 * scale, 0.05 * scale], color, 0.004 * scale));
        }
        b
    }
}

/// Batch commands for an emote prop (`None` for emotes without a prop form, as in [`crate::forge::emote_prop`]).
pub fn prop_script(emote: CharacterEmote, angle_deg: f32) -> Option<Vec<Value>> {
    use CharacterEmote as E;
    let mut s = Script::default();
    match emote {
        E::ComicAnger3d => {
            for (sx, sy) in [(1.0f32, 1.0f32), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)] {
                s.block([sx * 0.045, sy * 0.08, 0.0], [0.024, 0.07, 0.03], "#EB141F", 0.006);
                s.block([sx * 0.08, sy * 0.045, 0.0], [0.07, 0.024, 0.03], "#EB141F", 0.006);
            }
        }
        E::SparkleIdea => {
            s.disc([0.0, 0.03], 0.062, 0.05, "#FFDB40");
            s.block([0.0, -0.035, 0.0], [0.07, 0.04, 0.05], "#FFDB40", 0.008);
            for k in 0..4 {
                let w = if k % 2 == 0 { 0.062 } else { 0.054 };
                s.block([0.0, -0.06 - 0.012 * k as f32, 0.0], [w, 0.01, w], "#9E9EAD", 0.002);
            }
            for (c, r) in [([-0.088f32, 0.082f32], 0.03f32), ([0.09, 0.078], 0.022), ([0.095, -0.02], 0.016)] {
                s.block([c[0], c[1], 0.0], [2.0 * r, 0.3 * r, 0.3 * r], "#FFF28C", 0.0);
                s.block([c[0], c[1], 0.0], [0.3 * r, 2.0 * r, 0.3 * r], "#FFF28C", 0.0);
            }
        }
        E::Exclaim3d => {
            let bar = s.plate([-0.025, -0.035], [0.025, 0.095], 0.05, "#F22E1A");
            s.fillet(&bar, 0.012);
            s.disc([0.0, -0.085], 0.026, 0.05, "#FF8C1A");
        }
        E::ThreatArrow3d => {
            let mut bodies = Vec::new();
            let mut part = |s: &mut Script, c: [f32; 2], size: [f32; 2]| {
                bodies.push(s.block([c[0], c[1], 0.0], [size[0], size[1], 0.04], "#FF5214", 0.004));
                bodies.push(s.block([c[0], c[1], -0.016], [size[0] + 0.016, size[1] + 0.016, 0.02], "#FFFFFF", 0.0));
            };
            part(&mut s, [-0.0425, 0.0], [0.125, 0.044]);
            for (k, h) in [0.055f32, 0.041, 0.027, 0.013].iter().enumerate() {
                part(&mut s, [0.0306 + 0.0212 * k as f32, 0.0], [0.0212, 2.0 * h]);
            }
            s.spin(&bodies, angle_deg);
        }
        E::SweatPanic => {
            for (c, k) in [([0.03f32, -0.03f32], 1.0f32), ([-0.065, 0.045], 0.6), ([0.088, 0.075], 0.4)] {
                s.disc(c, 0.035 * k, 0.04 * k, "#8CD1FF");
                s.block([c[0], c[1] + 0.035 * k, 0.0], [0.03 * k, 0.05 * k, 0.03 * k], "#8CD1FF", 0.012 * k);
            }
        }
        E::ChibiHeart => {
            s.heart([0.0, 0.0], 1.0, "#FF6199");
        }
        E::HeartEyes => {
            s.heart([-0.062, 0.0], 0.5, "#FF6199");
            s.heart([0.062, 0.0], 0.5, "#FF6199");
        }
        E::Question => {
            for (c, size) in [
                ([0.0f32, 0.085f32], [0.07f32, 0.022f32]),
                ([-0.035, 0.06], [0.022, 0.03]),
                ([0.035, 0.055], [0.022, 0.06]),
                ([0.015, 0.025], [0.05, 0.022]),
                ([0.0, -0.005], [0.022, 0.04]),
            ] {
                s.block([c[0], c[1], 0.0], [size[0], size[1], 0.035], "#7361F2", 0.006);
            }
            s.disc([0.0, -0.075], 0.02, 0.035, "#7361F2");
        }
        E::MusicNote => {
            s.disc([-0.03, -0.07], 0.032, 0.03, "#4D40BF");
            s.block([0.0, 0.0, 0.0], [0.016, 0.16, 0.02], "#4D40BF", 0.0);
            s.block([0.025, 0.07, 0.0], [0.05, 0.016, 0.02], "#4D40BF", 0.0);
            s.block([0.045, 0.045, 0.0], [0.016, 0.04, 0.02], "#4D40BF", 0.0);
        }
        E::ReliefSigh => {
            for (c, r) in [
                ([-0.06f32, 0.0f32], 0.04f32),
                ([-0.015, 0.02], 0.05),
                ([0.04, 0.005], 0.043),
                ([0.09, 0.03], 0.022),
            ] {
                s.disc(c, r, 0.06, "#F2F7FF");
            }
        }
        E::ChibiCheer | E::Tears | E::StarEyes | E::DizzySpiral => return None,
    }
    Some(s.cmds)
}

/// Batch commands for an accessory around a head of radius ~0.5 at the origin.
pub fn part_script(part: AvatarPart) -> Vec<Value> {
    let mut s = Script::default();
    match part {
        AvatarPart::Headset => {
            s.block([0.0, 0.545, 0.0], [0.62, 0.05, 0.06], "#26262E", 0.02);
            for x in [-1.0f32, 1.0] {
                s.block([x * 0.545, 0.3, 0.0], [0.05, 0.5, 0.06], "#26262E", 0.02);
                s.block([x * 0.565, 0.0, 0.0], [0.09, 0.26, 0.26], "#4D3873", 0.04);
                s.block([x * 0.62, 0.0, 0.0], [0.02, 0.17, 0.17], "#6B40B3", 0.008);
            }
            s.block([-0.56, -0.2, 0.2], [0.02, 0.02, 0.36], "#26262E", 0.0);
            s.block([-0.38, -0.29, 0.44], [0.36, 0.02, 0.02], "#26262E", 0.0);
            s.block([-0.18, -0.29, 0.46], [0.06, 0.06, 0.06], "#141419", 0.025);
        }
        AvatarPart::Glasses => {
            for x in [-1.0f32, 1.0] {
                let c = [x * 0.18, 0.04];
                s.block([c[0], c[1] + 0.085, 0.5], [0.24, 0.022, 0.022], "#291F38", 0.006);
                s.block([c[0], c[1] - 0.085, 0.5], [0.24, 0.022, 0.022], "#291F38", 0.006);
                s.block([c[0] - 0.12, c[1], 0.5], [0.022, 0.19, 0.022], "#291F38", 0.006);
                s.block([c[0] + 0.12, c[1], 0.5], [0.022, 0.19, 0.022], "#291F38", 0.006);
                s.block([c[0], c[1], 0.5], [0.22, 0.15, 0.004], "#B3D9FF", 0.0);
                s.block([x * 0.31, 0.06, 0.25], [0.02, 0.02, 0.5], "#291F38", 0.0);
            }
            s.block([0.0, 0.07, 0.5], [0.12, 0.02, 0.02], "#291F38", 0.006);
        }
        AvatarPart::CatEars => {
            for view_deg in [35.0f32, -35.0] {
                let mut b = Vec::new();
                for (k, w) in [0.24f32, 0.16, 0.08].iter().enumerate() {
                    b.push(s.block([0.0, 0.43 + 0.07 * k as f32, 0.0], [*w, 0.07, 0.07], "#4D3873", 0.01));
                }
                b.push(s.block([0.0, 0.46, 0.04], [0.12, 0.06, 0.02], "#FFADC7", 0.005));
                s.spin(&b, view_deg);
            }
        }
        AvatarPart::Ribbon => {
            let b = vec![
                s.block([0.0, 0.53, 0.12], [0.07, 0.07, 0.06], "#C7366B", 0.02),
                s.block([-0.1, 0.53, 0.12], [0.14, 0.12, 0.045], "#EB4D85", 0.025),
                s.block([0.1, 0.53, 0.12], [0.14, 0.12, 0.045], "#EB4D85", 0.025),
            ];
            s.spin(&b, -30.0);
        }
    }
    s.cmds
}

/// Batch commands for the Track A robot body (head ~1 m wide at the origin). The eyes and mouth are not
/// modelled here — [`SolidCraftBridge::forge_robot`] grafts on the forge face with its morph targets.
pub fn robot_script() -> Vec<Value> {
    let mut s = Script::default();
    let (white, purple, violet, dark, grey) = ("#EDEDF5", "#4D3873", "#6B40B3", "#0D0A17", "#8C8C9E");
    s.block([0.0, 0.0, 0.0], [1.0, 0.82, 0.86], white, 0.2);
    for x in [-1.0f32, 1.0] {
        s.block([x * 0.53, -0.02, 0.0], [0.1, 0.3, 0.3], purple, 0.045);
    }
    s.block([0.0, 0.425, 0.0], [0.11, 0.05, 0.11], grey, 0.01);
    s.block([0.0, 0.53, 0.0], [0.036, 0.2, 0.036], grey, 0.0);
    s.block([0.0, 0.64, 0.0], [0.11, 0.11, 0.11], violet, 0.05);
    s.block([0.0, -0.03, 0.41], [0.8, 0.52, 0.08], dark, 0.035);
    s.block([0.0, -0.46, 0.0], [0.18, 0.12, 0.18], grey, 0.0);
    s.block([0.0, -0.72, 0.0], [0.56, 0.44, 0.42], white, 0.14);
    s.block([0.0, -0.66, 0.205], [0.26, 0.16, 0.04], purple, 0.015);
    for x in [-1.0f32, 1.0] {
        s.block([x * 0.36, -0.72, 0.0], [0.13, 0.32, 0.14], white, 0.05);
        s.block([x * 0.38, -0.92, 0.0], [0.17, 0.17, 0.17], purple, 0.08);
        s.block([x * 0.13, -1.0, 0.0], [0.16, 0.18, 0.18], grey, 0.05);
        s.block([x * 0.14, -1.11, 0.04], [0.22, 0.09, 0.3], purple, 0.04);
    }
    s.cmds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zena-mcp-test-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn assert_well_formed(cmds: &[Value], what: &str) {
        assert!(!cmds.is_empty(), "{what}");
        for c in cmds {
            let name = c["command"].as_str().unwrap_or_else(|| panic!("{what}: {c}"));
            assert!(NEEDED_COMMANDS.contains(&name), "{what}: undocumented command {name}");
            assert!(c["params"].is_object(), "{what}: {c}");
            assert_eq!(c.as_object().unwrap().len(), 2, "{what}: only command + params");
        }
        // Every body gets a colour.
        let bodies = cmds
            .iter()
            .filter(|c| c["command"] == "solid.box" || c["command"] == "solid.extrude")
            .count();
        let colours = cmds.iter().filter(|c| c["command"] == "appearance.assign").count();
        assert_eq!(bodies, colours, "{what}");
    }

    #[test]
    fn scripts_are_well_formed() {
        let mut props = 0;
        for e in CharacterEmote::ALL {
            let s = prop_script(e, 30.0);
            assert_eq!(s.is_some(), crate::forge::emote_prop(e, 30.0).is_some(), "{}", e.name());
            if let Some(s) = s {
                assert_well_formed(&s, e.name());
                props += 1;
            }
        }
        assert_eq!(props, 10);
        for p in AvatarPart::ALL {
            assert_well_formed(&part_script(p), p.name());
        }
        assert_well_formed(&robot_script(), "robot");
        let arrow = prop_script(CharacterEmote::ThreatArrow3d, 30.0).unwrap();
        let spin = arrow.iter().find(|c| c["command"] == "solid.move").unwrap();
        assert_eq!(spin["params"]["angle"], json!(-30.0));
        assert_eq!(spin["params"]["bodies"].as_array().unwrap().len(), 10);
        let first_box = robot_script()[0].clone();
        assert_eq!(
            first_box["params"]["corner"],
            json!([-500.0, -430.0, -410.0]),
            "mm, Z up, centred head"
        );
    }

    #[test]
    fn templates_substitute_placeholders() {
        let dir = temp("templates");
        std::fs::write(
            dir.join("prop_THREAT_ARROW_3D.json"),
            r#"{"commands": [
                {"command": "solid.box", "params": {"corner": [0, 0, 0], "length": 10, "width": 10, "height": 10}},
                {"command": "solid.move", "params": {"bodies": ["Body1"], "axis": "Y", "angle": "{{neg_angle_deg}}", "note": "a={{angle_deg}}"}},
                {"command": "sketch.finish"}
            ]}"#,
        )
        .unwrap();
        std::fs::write(dir.join("broken.json"), "[{\"params\": {}}]").unwrap();
        let s = load_template(&dir, "prop_THREAT_ARROW_3D", &[("angle_deg", 45.0), ("neg_angle_deg", -45.0)]).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(
            s[1]["params"]["angle"].as_f64(),
            Some(-45.0),
            "a quoted placeholder becomes a number"
        );
        assert_eq!(s[1]["params"]["note"], "a=45");
        assert_eq!(s[2]["params"], json!({}));
        assert!(load_template(&dir, "missing", &[]).is_none());
        assert!(load_template(&dir, "broken", &[]).is_none());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn command_ids_are_found_in_any_shape() {
        let mut ids = HashSet::new();
        command_ids(
            &json!({"commands": [{"id": "solid.box", "description": "a box"}, "sketch.finish"], "groups": {"solid.fillet": {}}}),
            &mut ids,
        );
        assert!(ids.contains("solid.box") && ids.contains("sketch.finish") && ids.contains("solid.fillet"));
    }

    #[test]
    fn missing_solidcraft_is_an_error() {
        let dir = temp("nosc");
        assert!(SolidCraftBridge::connect(&dir.join("solidcraft-cli-missing.exe"), &dir, None).is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
