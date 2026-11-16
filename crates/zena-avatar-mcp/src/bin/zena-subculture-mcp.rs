//! `zena-subculture-mcp --allow-path DIR [--chibi-kit DIR]` — Raon Zena's subculture (Track B) assets and the
//! built-in Track A forge as MCP tools over stdio (newline-delimited JSON-RPC 2.0).
//!
//! Tools: `list_emotes`, `compose_chibi_emote`, `forge_prop`, `robot_body`, `vrm_expressions`, `emote_morphs`.
//! Every `out_dir` must resolve (after canonicalisation, relative paths against the allow dir) inside
//! `--allow-path`; anything else is refused.

use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use zena_avatar_mcp::chibi::ChibiKit;
use zena_avatar_mcp::mesh::{write_glb, MeshData};
use zena_avatar_mcp::protocol::{serve, ToolHandler};
use zena_avatar_mcp::vrm::{emote_expression, VrmInfo, VrmVersion};
use zena_avatar_mcp::{forge, AvatarPart, CharacterEmote, EmoteForm};

struct Server {
    allow: PathBuf,
    kit_dir: Option<PathBuf>,
    kit: Option<ChibiKit>,
}

fn display(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

fn form_name(f: EmoteForm) -> &'static str {
    match f {
        EmoteForm::Stamp => "stamp",
        EmoteForm::Prop => "prop",
        EmoteForm::Morph => "morph",
    }
}

fn emote_arg(args: &Value) -> Result<CharacterEmote> {
    let s = args.get("emote").and_then(Value::as_str).context("missing `emote`")?;
    CharacterEmote::parse(s).with_context(|| format!("unknown emote {s}"))
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key).and_then(Value::as_str).with_context(|| format!("missing `{key}`"))
}

/// Resolves `out_dir` inside the allow dir (refusing `..` escapes and symlinked exits) and creates it.
fn resolve_inside(allow: &Path, out_dir: &str) -> Result<PathBuf> {
    let p = Path::new(out_dir);
    let p = if p.is_absolute() { p.to_path_buf() } else { allow.join(p) };
    let mut existing = p.as_path();
    let mut rest: Vec<&std::ffi::OsStr> = Vec::new();
    while !existing.exists() {
        match (existing.file_name(), existing.parent()) {
            (Some(name), Some(parent)) => {
                rest.push(name);
                existing = parent;
            }
            _ => bail!("invalid out_dir {out_dir}"),
        }
    }
    let mut full = existing.canonicalize().with_context(|| format!("resolving {out_dir}"))?;
    for name in rest.iter().rev() {
        if Path::new(name).components().any(|c| !matches!(c, Component::Normal(_))) {
            bail!("invalid out_dir {out_dir}");
        }
        full.push(name);
    }
    if !full.starts_with(allow) {
        bail!("out_dir {out_dir} is outside the allowed folder {}", display(allow));
    }
    std::fs::create_dir_all(&full).with_context(|| format!("creating {}", display(&full)))?;
    Ok(full)
}

fn emote_enum() -> Vec<&'static str> {
    CharacterEmote::ALL.iter().map(|e| e.name()).collect()
}

fn write_mesh(dir: &Path, name: &str, mesh: &MeshData) -> Result<Value> {
    let path = dir.join(name);
    std::fs::write(&path, write_glb(mesh)).with_context(|| format!("writing {}", display(&path)))?;
    Ok(json!({
        "path": display(&path),
        "vertices": mesh.vertex_count(),
        "triangles": mesh.triangle_count(),
        "materials": mesh.materials.len(),
        "morphs": mesh.morphs.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
    }))
}

fn read_vrm(path: &str) -> Result<VrmInfo> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    VrmInfo::from_glb(&bytes)?.with_context(|| format!("{path} has no VRM extension"))
}

impl Server {
    fn kit(&mut self) -> &ChibiKit {
        if self.kit.is_none() {
            let kit = match &self.kit_dir {
                Some(d) => ChibiKit::load(d).unwrap_or_else(|e| {
                    eprintln!("chibi kit {}: {e:#}; using the built-in kit", d.display());
                    ChibiKit::builtin()
                }),
                None => ChibiKit::builtin(),
            };
            self.kit = Some(kit);
        }
        self.kit.as_ref().expect("kit just set")
    }
}

impl ToolHandler for Server {
    fn tools(&self) -> Vec<Value> {
        let emotes = emote_enum();
        let parts: Vec<&str> = AvatarPart::ALL.iter().map(|p| p.name()).collect();
        let out_dir =
            json!({"type": "string", "description": "Output folder (inside --allow-path; relative paths are resolved against it)"});
        vec![
            json!({
                "name": "list_emotes",
                "description": "Zena's character emotes with their form (stamp / prop / morph) and comic VRM morph.",
                "inputSchema": {"type": "object", "properties": {}},
            }),
            json!({
                "name": "compose_chibi_emote",
                "description": "Compose an animated SD (chibi) stamp atlas (frames side by side) as chibi_<EMOTE>.png.",
                "inputSchema": {"type": "object", "properties": {
                    "emote": {"type": "string", "enum": emotes},
                    "frames": {"type": "integer", "minimum": 1, "maximum": 64, "default": 8},
                    "out_dir": out_dir,
                }, "required": ["emote", "out_dir"]},
            }),
            json!({
                "name": "forge_prop",
                "description": "Generate a 3D comic emote prop or an accessory part as binary glTF (metres, Y up, +Z front).",
                "inputSchema": {"type": "object", "properties": {
                    "emote": {"type": "string", "enum": emotes},
                    "part": {"type": "string", "enum": parts},
                    "angle_deg": {"type": "number", "description": "THREAT_ARROW_3D direction: 0 = right, 90 = up"},
                    "out_dir": out_dir,
                }, "required": ["out_dir"]},
            }),
            json!({
                "name": "robot_body",
                "description": "Generate the Track A SD toy-robot Zena body (binary glTF with VRM-named face morphs).",
                "inputSchema": {"type": "object", "properties": {"out_dir": out_dir}, "required": ["out_dir"]},
            }),
            json!({
                "name": "vrm_expressions",
                "description": "List a VRM model's expressions (VRM 1.0 names) and humanoid bones.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            }),
            json!({
                "name": "emote_morphs",
                "description": "The VRM expression that drives an emote's comic face morph on a model.",
                "inputSchema": {"type": "object", "properties": {
                    "emote": {"type": "string", "enum": emotes},
                    "path": {"type": "string"},
                }, "required": ["emote", "path"]},
            }),
        ]
    }

    fn call(&mut self, name: &str, args: &Value) -> Result<Value> {
        match name {
            "list_emotes" => Ok(json!({"emotes": CharacterEmote::ALL.iter().map(|e| json!({
                "name": e.name(),
                "form": form_name(e.form()),
                "comic_morph": e.comic_morph(),
                "prop": forge::emote_prop(*e, 0.0).is_some(),
            })).collect::<Vec<_>>()})),
            "compose_chibi_emote" => {
                let emote = emote_arg(args)?;
                let frames = args.get("frames").and_then(Value::as_u64).unwrap_or(8).clamp(1, 64) as usize;
                let dir = resolve_inside(&self.allow, str_arg(args, "out_dir")?)?;
                let kit = self.kit();
                let path = kit.write_atlas(emote, frames, &dir)?;
                let (w, h) = kit.size();
                Ok(json!({"path": display(&path), "frames": frames, "width": w * frames as u32, "height": h, "frame_width": w}))
            }
            "forge_prop" => {
                let dir = resolve_inside(&self.allow, str_arg(args, "out_dir")?)?;
                let angle = args.get("angle_deg").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                if let Some(e) = args.get("emote").and_then(Value::as_str) {
                    let emote = CharacterEmote::parse(e).with_context(|| format!("unknown emote {e}"))?;
                    let mesh = forge::emote_prop(emote, angle).with_context(|| format!("{e} has no 3D prop"))?;
                    let file = if emote == CharacterEmote::ThreatArrow3d {
                        format!("prop_{}_a{:03}.glb", emote.name(), angle.round().rem_euclid(360.0) as u32)
                    } else {
                        format!("prop_{}.glb", emote.name())
                    };
                    write_mesh(&dir, &file, &mesh)
                } else if let Some(p) = args.get("part").and_then(Value::as_str) {
                    let part = AvatarPart::ALL
                        .into_iter()
                        .find(|x| x.name().eq_ignore_ascii_case(p))
                        .with_context(|| format!("unknown part {p}"))?;
                    write_mesh(&dir, &format!("part_{}.glb", part.name()), &forge::part_mesh(part))
                } else {
                    bail!("forge_prop needs `emote` or `part`")
                }
            }
            "robot_body" => {
                let dir = resolve_inside(&self.allow, str_arg(args, "out_dir")?)?;
                write_mesh(&dir, "robot_zena.glb", &forge::robot_zena())
            }
            "vrm_expressions" => {
                let info = read_vrm(str_arg(args, "path")?)?;
                Ok(json!({
                    "version": if info.version == VrmVersion::V0 { "0.x" } else { "1.0" },
                    "expressions": info.expressions.iter().map(|e| json!({"name": e.name, "binds": e.binds.len()})).collect::<Vec<_>>(),
                    "humanoid": info.humanoid.iter().map(|(b, n)| (b.clone(), json!(n))).collect::<serde_json::Map<_, _>>(),
                }))
            }
            "emote_morphs" => {
                let emote = emote_arg(args)?;
                let info = read_vrm(str_arg(args, "path")?)?;
                Ok(json!({
                    "emote": emote.name(),
                    "comic_morph": emote.comic_morph(),
                    "expression": emote_expression(&info, emote),
                }))
            }
            other => bail!("unknown tool {other}"),
        }
    }
}

fn main() {
    let mut allow: Option<PathBuf> = None;
    let mut kit_dir: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--allow-path" => allow = args.next().map(PathBuf::from),
            "--chibi-kit" => kit_dir = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                eprintln!("usage: zena-subculture-mcp --allow-path DIR [--chibi-kit DIR]");
                return;
            }
            other => eprintln!("zena-subculture-mcp: ignoring unknown argument {other}"),
        }
    }
    let Some(allow) = allow else {
        eprintln!("usage: zena-subculture-mcp --allow-path DIR [--chibi-kit DIR]");
        std::process::exit(2);
    };
    let allow = match std::fs::create_dir_all(&allow).and_then(|_| allow.canonicalize()) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("zena-subculture-mcp: --allow-path {}: {e}", allow.display());
            std::process::exit(2);
        }
    };
    let mut server = Server { allow, kit_dir, kit: None };
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    if let Err(e) = serve(&mut server, stdin, stdout, "zena-subculture-mcp") {
        eprintln!("zena-subculture-mcp: {e:#}");
        std::process::exit(1);
    }
}
