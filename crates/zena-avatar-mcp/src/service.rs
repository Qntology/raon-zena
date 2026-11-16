//! [`AssetService`]: the background thread the overlay app asks for avatar assets.
//!
//! Requests are handled one at a time on the service thread. Track A meshes (props, parts, the robot body)
//! come from SolidCraft when [`ServiceConfig::solidcraft`] is set and it connects, else from the in-process
//! [`crate::forge`]; stamps come from the `zena-subculture-mcp` server when [`ServiceConfig::subculture_mcp`] is
//! set, else from the in-process [`crate::chibi`] kit. Both connections are made lazily, once; a failed
//! connection is logged once and the in-process path is used from then on. Results are cached by file name in
//! [`ServiceConfig::cache_dir`] (prop angles bucketed to 15°): a cached file is returned immediately with
//! `via == "cache"`. SolidCraft templates are read from `<cache_dir>/solidcraft_templates` when it exists.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::json;
use zena_core::tools::{AvatarPart, AvatarTrack, CharacterEmote};

use crate::chibi::ChibiKit;
use crate::mesh::{load_glb, write_glb, MeshData};
use crate::protocol::{result_json, McpClient};
use crate::solidcraft::SolidCraftBridge;

#[derive(Debug, Clone, PartialEq)]
pub enum AssetRequest {
    Body(AvatarTrack),
    EmoteStamp(CharacterEmote),
    EmoteProp { emote: CharacterEmote, angle_deg: f32 },
    Part(AvatarPart),
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssetReady {
    pub request: AssetRequest,
    pub path: PathBuf,
    /// `"solidcraft"`, `"zena-subculture-mcp"`, `"in-process"` or `"cache"`.
    pub via: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceConfig {
    pub cache_dir: PathBuf,
    /// `solidcraft-cli` executable (see [`crate::solidcraft::find_solidcraft`]); `None` = forge only.
    pub solidcraft: Option<PathBuf>,
    /// `zena-subculture-mcp` executable; `None` = in-process chibi compositor.
    pub subculture_mcp: Option<PathBuf>,
    /// The Track B-2 `.vrm` model.
    pub vrm_path: Option<PathBuf>,
    /// A `kit.json` directory for Track B-1 (else the built-in kit).
    pub chibi_kit: Option<PathBuf>,
    /// Frames per stamp atlas (8).
    pub stamp_frames: usize,
}

impl ServiceConfig {
    /// In-process only, 8-frame stamps.
    pub fn new(cache_dir: impl Into<PathBuf>) -> ServiceConfig {
        ServiceConfig {
            cache_dir: cache_dir.into(),
            solidcraft: None,
            subculture_mcp: None,
            vrm_path: None,
            chibi_kit: None,
            stamp_frames: 8,
        }
    }
}

/// Handle to the asset thread; dropping it stops the thread after the queued requests.
pub struct AssetService {
    tx: Sender<AssetRequest>,
}

impl AssetService {
    /// Starts the service thread. `on_ready` / `on_error` run on that thread.
    pub fn spawn(
        cfg: ServiceConfig,
        on_ready: impl Fn(AssetReady) + Send + 'static,
        on_error: impl Fn(AssetRequest, String) + Send + 'static,
    ) -> AssetService {
        let (tx, rx) = mpsc::channel::<AssetRequest>();
        // MCP servers resolve `out_dir` against their own allow path: hand them absolute paths only.
        let mut cfg = cfg;
        if let Ok(abs) = std::path::absolute(&cfg.cache_dir) {
            cfg.cache_dir = abs;
        }
        let spawned = std::thread::Builder::new().name("zena-assets".into()).spawn(move || {
            let mut w = Worker::new(cfg);
            for req in rx {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| w.handle(&req)));
                match outcome {
                    Ok(Ok((path, via))) => on_ready(AssetReady {
                        request: req,
                        path,
                        via: via.to_string(),
                    }),
                    Ok(Err(e)) => on_error(req, format!("{e:#}")),
                    Err(_) => on_error(req, "asset generation panicked".into()),
                }
            }
        });
        if let Err(e) = spawned {
            eprintln!("[asset-service] could not start the asset thread: {e}");
        }
        AssetService { tx }
    }

    /// Queues a request (answered through `on_ready` / `on_error`).
    pub fn request(&self, r: AssetRequest) {
        if self.tx.send(r).is_err() {
            eprintln!("[asset-service] the asset thread has stopped");
        }
    }
}

/// Angle bucket (degrees, 0..360, 15° steps) used for the arrow prop cache.
pub fn angle_bucket(angle_deg: f32) -> u32 {
    if !angle_deg.is_finite() {
        return 0;
    }
    ((angle_deg / 15.0).round() * 15.0).rem_euclid(360.0) as u32 % 360
}

/// Cache file name for a prop (`prop_<EMOTE>.glb`, the arrow adds `_a<bucket>`).
pub fn prop_file_name(emote: CharacterEmote, angle_deg: f32) -> String {
    if emote == CharacterEmote::ThreatArrow3d {
        format!("prop_{}_a{:03}.glb", emote.name(), angle_bucket(angle_deg))
    } else {
        format!("prop_{}.glb", emote.name())
    }
}

fn cached(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0)
}

/// Writes via a temporary file so a crash never leaves a truncated cache entry.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    Ok(())
}

struct Worker {
    cfg: ServiceConfig,
    solidcraft: Option<SolidCraftBridge>,
    solidcraft_tried: bool,
    solidcraft_failures: u32,
    subculture: Option<McpClient>,
    subculture_tried: bool,
    kit: Option<ChibiKit>,
}

const VIA_SOLIDCRAFT: &str = "solidcraft";
const VIA_SUBCULTURE: &str = "zena-subculture-mcp";
const VIA_IN_PROCESS: &str = "in-process";
const VIA_CACHE: &str = "cache";

impl Worker {
    fn new(cfg: ServiceConfig) -> Worker {
        Worker {
            cfg,
            solidcraft: None,
            solidcraft_tried: false,
            solidcraft_failures: 0,
            subculture: None,
            subculture_tried: false,
            kit: None,
        }
    }

    fn handle(&mut self, req: &AssetRequest) -> Result<(PathBuf, &'static str)> {
        std::fs::create_dir_all(&self.cfg.cache_dir).with_context(|| format!("creating {}", self.cfg.cache_dir.display()))?;
        match *req {
            AssetRequest::Body(AvatarTrack::Procedural) => bail!("procedural needs no asset"),
            AssetRequest::Body(AvatarTrack::Solidcraft) => {
                self.mesh_asset("robot_zena.glb", |sc, out| sc.forge_robot(out), crate::forge::robot_zena)
            }
            AssetRequest::Body(AvatarTrack::Subculture3d) => {
                let path = self.cfg.vrm_path.clone().context("no VRM model configured (vrm_path)")?;
                let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                let mesh = load_glb(&bytes).with_context(|| format!("loading {}", path.display()))?;
                if mesh.positions.is_empty() {
                    bail!("{} has no geometry", path.display());
                }
                if crate::vrm::VrmInfo::from_glb(&bytes)?.is_none() {
                    eprintln!("[asset-service] {} has no VRM extension (no expressions)", path.display());
                }
                Ok((path, VIA_IN_PROCESS))
            }
            AssetRequest::Body(AvatarTrack::Subculture2d) => {
                let path = self.cfg.cache_dir.join("body_2d.png");
                if cached(&path) {
                    return Ok((path, VIA_CACHE));
                }
                let img = self.kit().neutral();
                let mut png = Vec::new();
                img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
                write_atomic(&path, &png)?;
                Ok((path, VIA_IN_PROCESS))
            }
            AssetRequest::EmoteStamp(emote) => self.stamp(emote),
            AssetRequest::EmoteProp { emote, angle_deg } => {
                let bucket = angle_bucket(angle_deg) as f32;
                if crate::forge::emote_prop(emote, bucket).is_none() {
                    bail!("{} has no 3D prop (use its stamp)", emote.name());
                }
                let name = prop_file_name(emote, angle_deg);
                self.mesh_asset(
                    &name,
                    |sc, out| sc.forge_prop(emote, bucket, out),
                    || crate::forge::emote_prop(emote, bucket).unwrap_or_default(),
                )
            }
            AssetRequest::Part(part) => {
                let name = format!("part_{}.glb", part.name());
                self.mesh_asset(&name, |sc, out| sc.forge_part(part, out), || crate::forge::part_mesh(part))
            }
        }
    }

    fn kit(&mut self) -> &ChibiKit {
        if self.kit.is_none() {
            let kit = match &self.cfg.chibi_kit {
                Some(dir) => ChibiKit::load(dir).unwrap_or_else(|e| {
                    eprintln!("[asset-service] chibi kit {}: {e:#}; using the built-in kit", dir.display());
                    ChibiKit::builtin()
                }),
                None => ChibiKit::builtin(),
            };
            self.kit = Some(kit);
        }
        self.kit.as_ref().expect("kit just set")
    }

    fn solidcraft(&mut self) -> Option<&mut SolidCraftBridge> {
        if !self.solidcraft_tried {
            self.solidcraft_tried = true;
            if let Some(exe) = self.cfg.solidcraft.clone() {
                let templates = self.cfg.cache_dir.join("solidcraft_templates");
                let templates = templates.is_dir().then_some(templates);
                match SolidCraftBridge::connect(&exe, &self.cfg.cache_dir, templates.as_deref()) {
                    Ok(b) => {
                        eprintln!("[asset-service] connected to {}", b.server_name());
                        self.solidcraft = Some(b);
                    }
                    Err(e) => eprintln!("[asset-service] SolidCraft unavailable ({e:#}); using the built-in forge"),
                }
            }
        }
        self.solidcraft.as_mut()
    }

    fn mesh_asset(
        &mut self,
        file: &str,
        via_solidcraft: impl FnOnce(&mut SolidCraftBridge, &Path) -> Result<PathBuf>,
        forge: impl FnOnce() -> MeshData,
    ) -> Result<(PathBuf, &'static str)> {
        let path = self.cfg.cache_dir.join(file);
        if cached(&path) {
            return Ok((path, VIA_CACHE));
        }
        if let Some(sc) = self.solidcraft() {
            match via_solidcraft(sc, &path) {
                Ok(p) => {
                    self.solidcraft_failures = 0;
                    return Ok((p, VIA_SOLIDCRAFT));
                }
                Err(e) => {
                    eprintln!("[asset-service] SolidCraft failed for {file} ({e:#}); using the built-in forge");
                    self.solidcraft_failures += 1;
                    if self.solidcraft_failures >= 3 {
                        eprintln!("[asset-service] disconnecting SolidCraft after repeated failures");
                        self.solidcraft = None;
                    }
                }
            }
        }
        let mesh = forge();
        if mesh.positions.is_empty() {
            bail!("the forge produced no geometry for {file}");
        }
        write_atomic(&path, &write_glb(&mesh))?;
        Ok((path, VIA_IN_PROCESS))
    }

    fn subculture(&mut self) -> Option<&mut McpClient> {
        if !self.subculture_tried {
            self.subculture_tried = true;
            if let Some(exe) = self.cfg.subculture_mcp.clone() {
                let mut args = vec!["--allow-path".to_string(), self.cfg.cache_dir.display().to_string()];
                if let Some(k) = &self.cfg.chibi_kit {
                    args.push("--chibi-kit".into());
                    args.push(k.display().to_string());
                }
                match McpClient::spawn(&exe, &args) {
                    Ok(c) => self.subculture = Some(c),
                    Err(e) => eprintln!("[asset-service] {} unavailable ({e:#}); composing stamps in-process", exe.display()),
                }
            }
        }
        self.subculture.as_mut()
    }

    fn stamp(&mut self, emote: CharacterEmote) -> Result<(PathBuf, &'static str)> {
        let frames = self.cfg.stamp_frames.max(1);
        let dir = self.cfg.cache_dir.join(format!("stamps_{frames}f"));
        let path = dir.join(format!("chibi_{}.png", emote.name()));
        if cached(&path) {
            return Ok((path, VIA_CACHE));
        }
        std::fs::create_dir_all(&dir)?;
        if let Some(client) = self.subculture() {
            let args = json!({"emote": emote.name(), "frames": frames, "out_dir": dir.display().to_string()});
            match client.call_tool("compose_chibi_emote", args, Duration::from_secs(60)) {
                Ok(r) => {
                    // The server writes into our cache folder; prefer our own spelling of the path.
                    if cached(&path) {
                        return Ok((path, VIA_SUBCULTURE));
                    }
                    let p = result_json(&r).get("path").and_then(|p| p.as_str()).map(PathBuf::from);
                    match p {
                        Some(p) if cached(&p) => return Ok((p, VIA_SUBCULTURE)),
                        _ => eprintln!("[asset-service] compose_chibi_emote returned no file; composing in-process"),
                    }
                }
                Err(e) => eprintln!("[asset-service] compose_chibi_emote failed ({e:#}); composing in-process"),
            }
        }
        let img = self.kit().atlas(emote, frames);
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
        write_atomic(&path, &png)?;
        Ok((path, VIA_IN_PROCESS))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn buckets_and_names() {
        assert_eq!(angle_bucket(44.0), 45);
        assert_eq!(angle_bucket(-10.0), 345);
        assert_eq!(angle_bucket(359.0), 0);
        assert_eq!(angle_bucket(f32::NAN), 0);
        assert_eq!(prop_file_name(CharacterEmote::ThreatArrow3d, 91.0), "prop_THREAT_ARROW_3D_a090.glb");
        assert_eq!(prop_file_name(CharacterEmote::Exclaim3d, 91.0), "prop_EXCLAIM_3D.glb");
    }

    #[test]
    fn in_process_service_produces_and_caches_every_asset() {
        let dir = std::env::temp_dir().join(format!("zena-mcp-test-{}-service", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (tx, rx) = mpsc::channel::<std::result::Result<AssetReady, (AssetRequest, String)>>();
        let tx2 = tx.clone();
        let svc = AssetService::spawn(
            ServiceConfig::new(&dir),
            move |r| {
                let _ = tx.send(Ok(r));
            },
            move |q, e| {
                let _ = tx2.send(Err((q, e)));
            },
        );
        let mut wanted: Vec<AssetRequest> = CharacterEmote::ALL.iter().map(|e| AssetRequest::EmoteStamp(*e)).collect();
        for e in CharacterEmote::ALL {
            if crate::forge::emote_prop(e, 0.0).is_some() {
                wanted.push(AssetRequest::EmoteProp { emote: e, angle_deg: 40.0 });
            }
        }
        wanted.extend(AvatarPart::ALL.map(AssetRequest::Part));
        wanted.push(AssetRequest::Body(AvatarTrack::Solidcraft));
        wanted.push(AssetRequest::Body(AvatarTrack::Subculture2d));
        let failing = [
            AssetRequest::Body(AvatarTrack::Procedural),
            AssetRequest::Body(AvatarTrack::Subculture3d),
            AssetRequest::EmoteProp {
                emote: CharacterEmote::ChibiCheer,
                angle_deg: 0.0,
            },
        ];
        for r in wanted.iter().chain(&failing) {
            svc.request(r.clone());
        }
        let mut ready = Vec::new();
        let mut errors = Vec::new();
        while ready.len() + errors.len() < wanted.len() + failing.len() {
            match rx.recv_timeout(Duration::from_secs(120)).expect("service answered") {
                Ok(r) => ready.push(r),
                Err(e) => errors.push(e),
            }
        }
        assert_eq!(errors.len(), 3, "{errors:?}");
        assert!(errors.iter().any(|(_, e)| e.contains("procedural needs no asset")));
        for r in &ready {
            assert_eq!(r.via, "in-process", "{r:?}");
            assert!(cached(&r.path), "{r:?}");
        }
        assert_eq!(ready.len(), wanted.len());
        let arrow = ready
            .iter()
            .find(|r| {
                matches!(
                    r.request,
                    AssetRequest::EmoteProp {
                        emote: CharacterEmote::ThreatArrow3d,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(arrow.path.ends_with("prop_THREAT_ARROW_3D_a045.glb"));
        let robot = load_glb(
            &std::fs::read(
                &ready
                    .iter()
                    .find(|r| r.request == AssetRequest::Body(AvatarTrack::Solidcraft))
                    .unwrap()
                    .path,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(robot.morph_index("blink").is_some());

        // A second request for the same (bucketed) asset is answered from the cache.
        let t = Instant::now();
        svc.request(AssetRequest::EmoteProp {
            emote: CharacterEmote::ThreatArrow3d,
            angle_deg: 47.0,
        });
        let again = rx.recv_timeout(Duration::from_secs(30)).unwrap().unwrap();
        assert_eq!(again.via, "cache");
        assert_eq!(again.path, arrow.path);
        assert!(t.elapsed() < Duration::from_secs(5));
        drop(svc);
        let _ = std::fs::remove_dir_all(dir);
    }
}
