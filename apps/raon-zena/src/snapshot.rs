//! `raon-zena --snapshot DIR`: render the overlay's three passes headless — no window, an offscreen target on the
//! same kind of wgpu device — for a few scripted moments (each avatar track, 2D stamps and 3D props, accessories,
//! the streamed bubble and the chat box) and write one PNG per moment plus a contact sheet. Assets come from the
//! real asset service (MCP servers when present, otherwise in-process), so this checks the whole asset path too.
//! With `--vrm model.vrm` two Track B-2 moments render that model (the VRM body under the toon shader).

use crate::assets::{decode, default_cache, sibling_server};
use crate::renderers::scene::{EmoteShow, Loaded, Scene, Stage};
use crate::renderers::{depth_texture, FrameParams};
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::mpsc::channel;
use std::time::Duration;
use zena_avatar_mcp::service::{AssetRequest, AssetService, ServiceConfig};
use zena_core::avatar::{LookAt, Viseme};
use zena_core::tools::{AvatarPart, AvatarTrack, BodyMotion, CharacterEmote, EmotionState};
use zena_core::window::{VIEWPORT_H, VIEWPORT_W};

struct Moment {
    name: &'static str,
    track: AvatarTrack,
    parts: Vec<AvatarPart>,
    emotion: EmotionState,
    motion: BodyMotion,
    emote: Option<(CharacterEmote, f32)>,
    viseme: Viseme,
    mouth: f32,
    look: LookAt,
    bubble: Option<&'static str>,
    input: Option<(&'static str, &'static str)>,
}

fn moments(vrm: bool) -> Vec<Moment> {
    let base = |name| Moment {
        name,
        track: AvatarTrack::Procedural,
        parts: vec![],
        emotion: EmotionState::Neutral,
        motion: BodyMotion::Idle,
        emote: None,
        viseme: Viseme::Rest,
        mouth: 0.0,
        look: LookAt::default(),
        bubble: None,
        input: None,
    };
    let mut list = vec![
        Moment {
            emotion: EmotionState::Excited,
            motion: BodyMotion::Cheer,
            emote: Some((CharacterEmote::ChibiCheer, 0.0)),
            viseme: Viseme::A,
            mouth: 0.6,
            bubble: Some("해냈다! 최고야! 보스 패턴을 완벽하게 피했어."),
            ..base("1_procedural_stamp_bubble")
        },
        Moment {
            emotion: EmotionState::Angry,
            emote: Some((CharacterEmote::ComicAnger3d, 0.0)),
            parts: vec![AvatarPart::Glasses, AvatarPart::Headset],
            ..base("2_procedural_prop_parts")
        },
        Moment {
            track: AvatarTrack::Solidcraft,
            emotion: EmotionState::SurprisedUrgent,
            motion: BodyMotion::LeanForwardPoint,
            emote: Some((CharacterEmote::ThreatArrow3d, 30.0)),
            viseme: Viseme::O,
            mouth: 0.7,
            look: LookAt {
                yaw_deg: 25.0,
                pitch_deg: 14.0,
            },
            bubble: Some("오른쪽 위에 보스야! 지금 바로 굴러서 피해!"),
            ..base("3_trackA_robot_arrow")
        },
        Moment {
            track: AvatarTrack::Solidcraft,
            emotion: EmotionState::Joy,
            parts: vec![AvatarPart::CatEars, AvatarPart::Ribbon],
            emote: Some((CharacterEmote::SparkleIdea, 0.0)),
            ..base("4_trackA_robot_parts")
        },
        Moment {
            track: AvatarTrack::Subculture2d,
            emotion: EmotionState::Focused,
            emote: Some((CharacterEmote::HeartEyes, 0.0)),
            bubble: Some("패턴 시작 모션이 보이면 옆으로 굴러!"),
            input: Some(("제나야 보스 패턴 ", "알")),
            ..base("5_trackB1_2d_chat")
        },
    ];
    if vrm {
        list.push(Moment {
            track: AvatarTrack::Subculture3d,
            emotion: EmotionState::Joy,
            viseme: Viseme::A,
            mouth: 0.4,
            look: LookAt {
                yaw_deg: 18.0,
                pitch_deg: 6.0,
            },
            ..base("6_trackB2_vrm_joy")
        });
        list.push(Moment {
            track: AvatarTrack::Subculture3d,
            emotion: EmotionState::Neutral,
            look: LookAt {
                yaw_deg: -30.0,
                pitch_deg: -4.0,
            },
            ..base("7_trackB2_vrm_side")
        });
    }
    list
}

/// Over a dark/light checker so transparency is visible.
fn over_checker(w: u32, h: u32, premul: &[u8]) -> image::RgbaImage {
    let mut img = image::RgbaImage::new(w, h);
    for (i, px) in img.pixels_mut().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        let bg = if (x / 16 + y / 16) % 2 == 0 { 58u8 } else { 74 };
        let s = &premul[i * 4..i * 4 + 4];
        let a = s[3] as u32;
        let c = |v: u8| (v as u32 + bg as u32 * (255 - a) / 255).min(255) as u8;
        *px = image::Rgba([c(s[0]), c(s[1]), c(s[2]), 255]);
    }
    img
}

pub fn run(dir: &Path, font: Option<&Path>, vrm: Option<&Path>) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    #[allow(unused_mut)]
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    #[cfg(windows)]
    if std::env::var_os("WGPU_BACKEND").is_none() {
        // The overlay's backend (and no probing of other Vulkan drivers).
        desc.backends = wgpu::Backends::DX12;
    }
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))?;
    let info = adapter.get_info();
    eprintln!("[snapshot] {} ({:?})", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("raon-zena-snapshot"),
        ..Default::default()
    }))?;
    let (w, h) = (VIEWPORT_W, VIEWPORT_H);
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let depth = depth_texture(&device, w, h);
    let mut scene = Scene::new(&device, format, font);

    // Every asset the moments need, through the asset service.
    let mut requests: Vec<AssetRequest> = vec![
        AssetRequest::Body(AvatarTrack::Solidcraft),
        AssetRequest::Body(AvatarTrack::Subculture2d),
    ];
    if vrm.is_some() {
        requests.push(AssetRequest::Body(AvatarTrack::Subculture3d));
    }
    for m in moments(vrm.is_some()) {
        for p in &m.parts {
            requests.push(AssetRequest::Part(*p));
        }
        if let Some((e, a)) = m.emote {
            requests.push(AssetRequest::EmoteStamp(e));
            if e.form() == zena_core::tools::EmoteForm::Prop {
                requests.push(AssetRequest::EmoteProp {
                    emote: e,
                    angle_deg: if e == CharacterEmote::ThreatArrow3d {
                        crate::renderers::scene::angle_bucket(a) as f32
                    } else {
                        0.0
                    },
                });
            }
        }
    }
    let mut unique: Vec<AssetRequest> = Vec::new();
    for r in requests {
        if !unique.contains(&r) {
            unique.push(r);
        }
    }
    let requests = unique;
    let (tx, rx) = channel::<Result<Loaded, String>>();
    let (tx2, tx3) = (tx.clone(), tx);
    let svc = AssetService::spawn(
        ServiceConfig {
            cache_dir: default_cache(),
            solidcraft: zena_avatar_mcp::solidcraft::find_solidcraft(),
            subculture_mcp: sibling_server(),
            vrm_path: vrm.map(Path::to_path_buf),
            chibi_kit: None,
            catalog: None,
            stamp_frames: 8,
        },
        move |r| {
            eprintln!("[snapshot] {:?} via {}", r.request, r.via);
            let _ = tx2.send(decode(&r, 8).map_err(|e| format!("{e:#}")));
        },
        move |r, e| {
            let _ = tx3.send(Err(format!("{r:?}: {e}")));
        },
    );
    for r in &requests {
        svc.request(r.clone());
    }
    for _ in 0..requests.len() {
        match rx.recv_timeout(Duration::from_secs(60)).context("waiting for the asset service")? {
            Ok(l) => scene.load(&device, &queue, l),
            Err(e) => eprintln!("[snapshot] asset failed: {e}"),
        }
    }

    let row = (w * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let moments = moments(vrm.is_some());
    let mut sheet = image::RgbaImage::new(w * moments.len() as u32, h);
    for (i, m) in moments.iter().enumerate() {
        let params = FrameParams {
            emotion: m.emotion,
            motion: m.motion,
            viseme: m.viseme,
            mouth_open: m.mouth,
            look: m.look,
            head: [m.look.yaw_deg / 35.0 * 0.6, m.look.pitch_deg / 25.0 * 0.6],
            gaze: [m.look.yaw_deg / 35.0, m.look.pitch_deg / 25.0],
            brow: if m.emotion == EmotionState::Angry { 1.0 } else { 0.0 },
            eye_open: if matches!(m.emotion, EmotionState::SurprisedUrgent) {
                1.3
            } else {
                1.0
            },
            mouth_form: if matches!(m.emotion, EmotionState::Joy | EmotionState::Excited) {
                0.8
            } else {
                0.1
            },
            time: 1.3,
            motion_t: 0.17,
            width: w,
            height: h,
            ..Default::default()
        };
        let stage = Stage {
            params,
            track: m.track,
            parts: &m.parts,
            emote: m.emote.map(|(e, a)| EmoteShow {
                emote: e,
                age: 0.8,
                fade: 1.0,
                angle_deg: a,
            }),
            bubble: m.bubble.map(|b| (b, 1.0)),
            input: m.input.map(|(t, p)| (t, p, true)),
            sync_pill: None,
            rejection_chip: None,
            drawer: None,
            badges: &[],
            drop_hint: None,
        };
        let missing = scene.missing(&stage);
        if !missing.is_empty() {
            eprintln!("[snapshot] {}: not loaded {missing:?} (fallbacks shown)", m.name);
        }
        let mut enc = device.create_command_encoder(&Default::default());
        scene.render(&device, &queue, &mut enc, &view, &depth, &stage);
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = readback.get_mapped_range(..)?;
            for y in 0..h {
                let s = (y * row) as usize;
                px.extend_from_slice(&data[s..s + (w * 4) as usize]);
            }
        }
        readback.unmap();
        let covered = px.chunks(4).filter(|p| p[3] > 0).count();
        let img = over_checker(w, h, &px);
        image::imageops::replace(&mut sheet, &img, (i as u32 * w) as i64, 0);
        let path = dir.join(format!("{}.png", m.name));
        img.save(&path)?;
        eprintln!(
            "[snapshot] {} → {} ({:.0}% covered)",
            m.name,
            path.display(),
            covered as f32 * 100.0 / (w * h) as f32
        );
    }
    let path = dir.join("contact_sheet.png");
    sheet.save(&path)?;
    eprintln!("[snapshot] contact sheet → {}", path.display());
    Ok(())
}
