//! The overlay's link to the dual-track asset pipeline (`zena-avatar-mcp`): what the scene misses becomes an
//! [`AssetRequest`] for the [`AssetService`] thread — which drives SolidCraft's MCP server (Track A) and the
//! `zena-subculture-mcp` server (Track B) in the background, or builds the asset in-process when they are not
//! available — and the files it produces are decoded there, off the UI thread, into [`Loaded`] assets.

use crate::renderers::scene::{angle_bucket, Loaded, Missing};
use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use zena_avatar_mcp::mesh::load_glb;
use zena_avatar_mcp::service::{AssetReady, AssetRequest, AssetService, ServiceConfig};
use zena_avatar_mcp::vrm::VrmInfo;
use zena_core::tools::AvatarTrack;

/// Where decoded assets (or failures) go — the overlay's event loop.
pub trait AssetLink: Send + Sync + 'static {
    fn ready(&self, l: Loaded);
    fn failed(&self, m: Missing);
}

pub struct Assets {
    svc: AssetService,
}

/// The per-user asset cache (`%LOCALAPPDATA%\RaonZena\assets`).
pub fn default_cache() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("RaonZena").join("assets")
}

/// `zena-subculture-mcp` next to this executable, if it was built.
pub fn sibling_server() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let p = exe.with_file_name(format!("zena-subculture-mcp{}", std::env::consts::EXE_SUFFIX));
    p.is_file().then_some(p)
}

pub fn request_of(m: &Missing) -> AssetRequest {
    match m {
        Missing::Stamp(e) => AssetRequest::EmoteStamp(*e),
        Missing::Prop(e, a) => AssetRequest::EmoteProp {
            emote: *e,
            angle_deg: *a as f32,
        },
        Missing::Part(p) => AssetRequest::Part(*p),
        Missing::Body(t) => AssetRequest::Body(*t),
    }
}

pub fn missing_of(r: &AssetRequest) -> Missing {
    match r {
        AssetRequest::EmoteStamp(e) => Missing::Stamp(*e),
        AssetRequest::EmoteProp { emote, angle_deg } => Missing::Prop(*emote, angle_bucket(*angle_deg)),
        AssetRequest::Part(p) => Missing::Part(*p),
        AssetRequest::Body(t) => Missing::Body(*t),
    }
}

fn png(path: &std::path::Path) -> Result<(u32, u32, Vec<u8>)> {
    let img = image::open(path)
        .with_context(|| format!("decoding {}", path.display()))?
        .to_rgba8();
    Ok((img.width(), img.height(), img.into_raw()))
}

/// Decode a produced file into an asset the scene can upload (`stamp_frames`: frames per stamp strip).
pub fn decode(r: &AssetReady, stamp_frames: usize) -> Result<Loaded> {
    let glb = || -> Result<Vec<u8>> { std::fs::read(&r.path).with_context(|| format!("reading {}", r.path.display())) };
    Ok(match &r.request {
        AssetRequest::EmoteStamp(e) => {
            let (w, h, rgba) = png(&r.path)?;
            // A horizontal strip of `stamp_frames` frames.
            Loaded::Stamp {
                emote: *e,
                frames: stamp_frames.clamp(1, w.max(1) as usize) as u32,
                w,
                h,
                rgba,
            }
        }
        AssetRequest::EmoteProp { emote, angle_deg } => Loaded::Prop {
            emote: *emote,
            angle: angle_bucket(*angle_deg),
            mesh: load_glb(&glb()?)?,
        },
        AssetRequest::Part(p) => Loaded::Part {
            part: *p,
            mesh: load_glb(&glb()?)?,
        },
        AssetRequest::Body(AvatarTrack::Subculture2d) => {
            let (w, h, rgba) = png(&r.path)?;
            Loaded::Body2d { w, h, rgba }
        }
        AssetRequest::Body(track) => {
            let bytes = glb()?;
            let vrm = VrmInfo::from_glb(&bytes)?;
            Loaded::BodyMesh {
                track: *track,
                mesh: load_glb(&bytes)?,
                vrm,
            }
        }
    })
}

impl Assets {
    pub fn spawn(cfg: ServiceConfig, link: impl AssetLink) -> Self {
        let link: Arc<dyn AssetLink> = Arc::new(link);
        let (ok, err) = (link.clone(), link);
        let frames = cfg.stamp_frames;
        let svc = AssetService::spawn(
            cfg,
            move |r: AssetReady| match decode(&r, frames) {
                Ok(l) => {
                    eprintln!("[assets] {:?} via {} ({})", r.request, r.via, r.path.display());
                    ok.ready(l)
                }
                Err(e) => {
                    eprintln!("[assets] {:?}: {e:#}", r.request);
                    ok.failed(missing_of(&r.request))
                }
            },
            move |req: AssetRequest, e: String| {
                eprintln!("[assets] {req:?}: {e}");
                err.failed(missing_of(&req))
            },
        );
        Self { svc }
    }

    pub fn request(&self, m: &Missing) {
        self.svc.request(request_of(m));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{channel, Sender};
    use std::sync::Mutex;
    use std::time::Duration;
    use zena_core::tools::{AvatarPart, CharacterEmote};

    struct Collect(Mutex<Sender<Result<Loaded, Missing>>>);
    impl AssetLink for Collect {
        fn ready(&self, l: Loaded) {
            let _ = self.0.lock().unwrap().send(Ok(l));
        }
        fn failed(&self, m: Missing) {
            let _ = self.0.lock().unwrap().send(Err(m));
        }
    }

    #[test]
    fn requests_round_trip_and_the_in_process_service_feeds_the_scene() {
        for m in [
            Missing::Stamp(CharacterEmote::ChibiCheer),
            Missing::Prop(CharacterEmote::ThreatArrow3d, 45),
            Missing::Part(AvatarPart::Glasses),
            Missing::Body(AvatarTrack::Solidcraft),
        ] {
            assert_eq!(missing_of(&request_of(&m)), m);
        }
        let dir = std::env::temp_dir().join(format!("zena-assets-test-{}", std::process::id()));
        let (tx, rx) = channel();
        let cfg = ServiceConfig {
            cache_dir: dir.clone(),
            solidcraft: None,
            subculture_mcp: None,
            vrm_path: None,
            chibi_kit: None,
            stamp_frames: 4,
        };
        let a = Assets::spawn(cfg, Collect(Mutex::new(tx)));
        a.request(&Missing::Stamp(CharacterEmote::SparkleIdea));
        a.request(&Missing::Prop(CharacterEmote::ComicAnger3d, 0));
        a.request(&Missing::Part(AvatarPart::Headset));
        a.request(&Missing::Body(AvatarTrack::Solidcraft));
        a.request(&Missing::Body(AvatarTrack::Subculture3d));
        let mut got = Vec::new();
        for _ in 0..5 {
            got.push(rx.recv_timeout(Duration::from_secs(30)).expect("an answer per request"));
        }
        assert!(got.iter().any(|r| matches!(
            r,
            Ok(Loaded::Stamp {
                emote: CharacterEmote::SparkleIdea,
                frames: 4,
                ..
            })
        )));
        assert!(got
            .iter()
            .any(|r| matches!(r, Ok(Loaded::Prop { mesh, .. }) if !mesh.indices.is_empty())));
        assert!(got.iter().any(|r| matches!(
            r,
            Ok(Loaded::Part {
                part: AvatarPart::Headset,
                ..
            })
        )));
        assert!(got.iter().any(|r| matches!(
            r,
            Ok(Loaded::BodyMesh { track: AvatarTrack::Solidcraft, mesh, vrm: None }) if !mesh.morphs.is_empty()
        )));
        assert!(
            got.iter().any(|r| matches!(r, Err(Missing::Body(AvatarTrack::Subculture3d)))),
            "no VRM configured: the 3D track reports it is unavailable"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
