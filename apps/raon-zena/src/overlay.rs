//! The 350×450 transparent micro-viewport: winit window + one wgpu device, the three-pass scene
//! ([`crate::renderers::scene`]: body → emote & props → Stage 0/1/2 chrome, 1–9 badges, speech bubble
//! & chat box), dynamic FPS, Win32 layering, privacy etiquette, click-vs-drag drop handling, and the
//! graceful shutdown sequence.
//!
//! Streamed answers arrive as [`AvatarCmd::Expression`] (the face and motion first) and [`AvatarCmd::Speak`]
//! sentences that queue after what Zena is saying and append to the bubble.

use crate::assets::{AssetLink, Assets};
use crate::browser_uia::NavBadgeOverlay;
use crate::pd_bridge::OverlaySyncView;
use crate::pipeline::{AvatarCmd, OverlayAction};
use crate::renderers::scene::{EmoteShow, Loaded, Missing, OverlayLayout, Scene, Stage};
use crate::renderers::{depth_texture, FrameParams};
use premium_discount_schema::envelope;
use premium_discount_window::stage::DRAG_THRESHOLD;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, Ime, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId, WindowLevel};
use zena_core::avatar::{
    live2d_gaze_params, live2d_mouth, look_at_event, AvatarPlacement, LookAt, Viseme,
};
use zena_core::geometry::{BBox, Polar, ScreenGeometry};
use zena_core::ingress_guard::zeroize_string;
use zena_core::nav_target::NavTargetProjector;
use zena_core::reaction::ReactionLevel;
use zena_core::tools::{
    AssetAction, AvatarPart, AvatarTrack, BodyMotion, CharacterEmote, EmotionState, SystemCommand,
    WindowLayerMode, ZenaAssetArgs, ZenaAvatarArgs,
};
use zena_core::window::{
    apply_command, Hotkey, LayerFlags, PrivacyAction, PrivacyEtiquette, ShutdownPhase,
    WindowEffect, WindowState, HOVER_ALPHA, VIEWPORT_H, VIEWPORT_W,
};
use zena_verdict::UserPin;

#[derive(Debug)]
pub enum UserEvent {
    Cmd(AvatarCmd),
    #[cfg_attr(not(windows), allow(dead_code))]
    Hotkey(Hotkey),
    #[cfg_attr(not(windows), allow(dead_code))]
    DrawerToggle,
    /// An asset decoded by the asset service.
    Asset(Box<Loaded>),
    /// An asset that could not be produced.
    AssetFailed(Missing),
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Loaded::Stamp { emote, frames, .. } => write!(f, "Stamp({emote:?}, {frames} frames)"),
            Loaded::Prop { emote, angle, .. } => write!(f, "Prop({emote:?}, {angle}°)"),
            Loaded::Part { part, .. } => write!(f, "Part({part:?})"),
            Loaded::Body2d { w, h, .. } => write!(f, "Body2d({w}x{h})"),
            Loaded::BodyMesh { track, mesh, vrm } => {
                write!(
                    f,
                    "BodyMesh({track:?}, {} vertices, vrm {})",
                    mesh.positions.len(),
                    vrm.is_some()
                )
            }
        }
    }
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    depth: wgpu::TextureView,
    scene: Scene,
}

/// Smoothed animation state (targets set by commands, current values eased per frame).
struct Anim {
    gaze: Option<Polar>,
    eye: [f32; 2],
    head: [f32; 2],
    emotion: EmotionState,
    motion: BodyMotion,
    motion_start: Instant,
    level: ReactionLevel,
    speech_start: Instant,
    speech_ms: u32,
    visemes: Vec<(u32, Viseme)>,
    expression_until: Instant,
    emote: Option<(CharacterEmote, Instant)>,
    look: LookAt,
}

/// The line on show (streamed sentences append while the answer lasts).
#[derive(Default)]
struct Bubble {
    text: String,
    /// The next sentence starts a new bubble (a new answer began).
    fresh: bool,
}

/// The in-overlay chat / drop-note box.
#[derive(Default)]
struct Chat {
    open: bool,
    text: String,
    preedit: String,
    /// When set, submitting sends `OverlayAction::DropNote` targeting this `cc` instead of chat.
    drop_target_cc: Option<String>,
}

pub struct Overlay {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    state: WindowState,
    privacy: PrivacyEtiquette,
    shutdown: ShutdownPhase,
    anim: Anim,
    bubble: Bubble,
    chat: Chat,
    sync_view: OverlaySyncView,
    drawer_open: bool,
    nav_badges: NavBadgeOverlay,
    step_aside_occluded: bool,
    cursor_pos: (f64, f64),
    press_pos: Option<(f64, f64)>,
    dragging: bool,
    track: AvatarTrack,
    parts: Vec<AvatarPart>,
    assets: Option<Assets>,
    requested: HashSet<Missing>,
    pending: Vec<Loaded>,
    font: Option<std::path::PathBuf>,
    t0: Instant,
    screen: ScreenGeometry,
    utterances: Sender<String>,
    actions: Option<Sender<OverlayAction>>,
    ask_epoch: Arc<AtomicU64>,
    pub capture_running: Arc<AtomicBool>,
}

const BUBBLE_HOLD_MS: u64 = 2500;
const EMOTE_FADE_S: f32 = 0.45;

impl Overlay {
    pub fn new(
        capture_running: Arc<AtomicBool>,
        screen: ScreenGeometry,
        utterances: Sender<String>,
    ) -> Self {
        let now = Instant::now();
        Self {
            window: None,
            gfx: None,
            state: WindowState::default(),
            privacy: PrivacyEtiquette::default(),
            shutdown: ShutdownPhase::Running,
            anim: Anim {
                gaze: None,
                eye: [0.0; 2],
                head: [0.0; 2],
                emotion: EmotionState::Neutral,
                motion: BodyMotion::Idle,
                motion_start: now,
                level: ReactionLevel::Lv1Silent,
                speech_start: now,
                speech_ms: 0,
                visemes: vec![],
                expression_until: now,
                emote: None,
                look: LookAt::default(),
            },
            bubble: Bubble::default(),
            chat: Chat::default(),
            sync_view: OverlaySyncView::default(),
            drawer_open: false,
            nav_badges: NavBadgeOverlay::default(),
            step_aside_occluded: false,
            cursor_pos: (0.0, 0.0),
            press_pos: None,
            dragging: false,
            track: AvatarTrack::Procedural,
            parts: Vec::new(),
            assets: None,
            requested: HashSet::new(),
            pending: Vec::new(),
            font: None,
            t0: now,
            screen,
            utterances,
            actions: None,
            ask_epoch: Arc::new(AtomicU64::new(0)),
            capture_running,
        }
    }

    pub fn with_pipeline_links(
        mut self,
        actions: Sender<OverlayAction>,
        ask_epoch: Arc<AtomicU64>,
    ) -> Self {
        self.actions = Some(actions);
        self.ask_epoch = ask_epoch;
        self
    }

    /// The asset service (track bodies, emote stamps and props, accessories) and the body track to start with.
    pub fn with_assets(mut self, assets: Assets, track: AvatarTrack) -> Self {
        let mut first: Vec<Missing> = CharacterEmote::ALL
            .iter()
            .map(|e| Missing::Stamp(*e))
            .collect();
        first.extend(
            CharacterEmote::ALL
                .iter()
                .filter(|e| {
                    e.form() == zena_core::tools::EmoteForm::Prop
                        && **e != CharacterEmote::ThreatArrow3d
                })
                .map(|e| Missing::Prop(*e, 0)),
        );
        if track != AvatarTrack::Procedural {
            first.push(Missing::Body(track));
        }
        for m in first {
            assets.request(&m);
            self.requested.insert(m);
        }
        self.assets = Some(assets);
        self.track = track;
        self
    }

    pub fn with_font(mut self, font: Option<std::path::PathBuf>) -> Self {
        self.font = font;
        self
    }

    fn send_action(&self, act: OverlayAction) {
        if let Some(tx) = &self.actions {
            let _ = tx.send(act);
        }
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn speaking(&self) -> bool {
        self.anim.speech_start.elapsed() < Duration::from_millis(self.anim.speech_ms as u64)
    }

    fn clear_speech(&mut self) {
        self.bubble.text.clear();
        self.bubble.fresh = true;
        self.anim.speech_ms = 0;
        self.anim.visemes.clear();
    }

    fn apply_privacy_actions(&mut self, acts: Vec<PrivacyAction>) {
        for a in acts {
            match a {
                PrivacyAction::CancelGeneration => {
                    self.ask_epoch.fetch_add(1, Ordering::AcqRel);
                }
                PrivacyAction::ClearSpeech => {
                    self.clear_speech();
                }
                PrivacyAction::HideCrouch => {
                    let crouch = PrivacyEtiquette::lock_reaction();
                    self.express(&crouch, ReactionLevel::Lv1Silent);
                }
                PrivacyAction::ApplyStyle => {
                    self.platform_style();
                }
                PrivacyAction::ZeroizeInput => {
                    zeroize_string(&mut self.chat.text);
                    zeroize_string(&mut self.chat.preedit);
                }
                PrivacyAction::Resume => {
                    self.anim.emotion = EmotionState::Neutral;
                    self.anim.motion = BodyMotion::Idle;
                    self.anim.motion_start = Instant::now();
                }
            }
        }
    }

    fn set_privacy_locked(&mut self, locked: bool) {
        let now = self.now_ms();
        let acts = self.privacy.set_locked(locked, now);
        if locked {
            self.drawer_open = false;
            self.nav_badges = NavBadgeOverlay::default();
            self.step_aside_occluded = false;
        }
        self.apply_privacy_actions(acts);
    }

    /// Gaze, emotion, motion and emote (no speech).
    fn express(&mut self, args: &ZenaAvatarArgs, level: ReactionLevel) {
        if let Some(g) = args.gaze_target {
            self.anim.gaze = Some(Polar {
                r: g.distance_norm,
                theta_deg: g.angle_deg,
            });
        }
        if args.body_motion != self.anim.motion || args.body_motion != BodyMotion::Idle {
            self.anim.motion_start = Instant::now();
        }
        self.anim.emotion = args.emotion_state;
        self.anim.motion = args.body_motion;
        self.anim.level = self.anim.level.max(level);
        let now = Instant::now();
        let showing = self.anim.emote.is_some() && now < self.anim.expression_until;
        match args.emote() {
            Some(e) if showing && self.anim.emote.map(|x| x.0) == Some(e) => {}
            Some(e) => self.anim.emote = Some((e, now)),
            None => {}
        }
        let speech_end = self.anim.speech_start + Duration::from_millis(self.anim.speech_ms as u64);
        self.anim.expression_until = speech_end.max(now) + Duration::from_millis(1800);
    }

    /// A whole reaction: replaces what Zena was saying.
    fn act(
        &mut self,
        args: &ZenaAvatarArgs,
        level: ReactionLevel,
        visemes: Vec<(u32, Viseme)>,
        speech_ms: u32,
    ) {
        self.anim.level = level;
        self.anim.visemes = visemes;
        self.anim.speech_ms = speech_ms;
        self.anim.speech_start = Instant::now();
        self.express(args, level);
        if let Some(t) = &args.speech_text {
            eprintln!("[zena:{level:?}] {t}");
            self.bubble.text = t.clone();
            self.bubble.fresh = false;
        }
    }

    /// A streamed sentence: queued after the current speech.
    fn speak(&mut self, text: &str, level: ReactionLevel, visemes: Vec<(u32, Viseme)>, ms: u32) {
        eprintln!("[zena:{level:?}] {text}");
        if self.speaking() {
            let off = self.anim.speech_ms;
            if self
                .anim
                .visemes
                .last()
                .is_some_and(|(t, v)| *v == Viseme::Rest && *t >= off)
            {
                self.anim.visemes.pop();
            }
            self.anim
                .visemes
                .extend(visemes.into_iter().map(|(t, v)| (t + off, v)));
            self.anim.speech_ms += ms;
        } else {
            self.anim.speech_start = Instant::now();
            self.anim.visemes = visemes;
            self.anim.speech_ms = ms;
        }
        if self.bubble.fresh || self.bubble.text.is_empty() {
            self.bubble.text = text.to_string();
            self.bubble.fresh = false;
        } else {
            self.bubble.text.push(' ');
            self.bubble.text.push_str(text);
        }
        self.anim.level = self.anim.level.max(level);
        let end = self.anim.speech_start + Duration::from_millis(self.anim.speech_ms as u64);
        self.anim.expression_until = self.anim.expression_until.max(end + Duration::from_millis(1800));
    }

    fn asset(&mut self, a: &ZenaAssetArgs) {
        match a.action {
            AssetAction::SwitchTrack => {
                if let Some(t) = a.track {
                    eprintln!("[avatar] track → {t:?}");
                    self.track = t;
                }
            }
            AssetAction::EquipPart => {
                if let Some(p) = a.part.filter(|p| !self.parts.contains(p)) {
                    eprintln!("[avatar] equip {p:?}");
                    self.parts.push(p);
                    self.parts.sort();
                }
            }
            AssetAction::RemovePart => {
                if let Some(p) = a.part {
                    self.parts.retain(|x| *x != p);
                }
            }
            AssetAction::ResetParts => self.parts.clear(),
        }
    }

    fn system(&mut self, cmd: SystemCommand, mode: Option<WindowLayerMode>) {
        for fx in apply_command(&mut self.state, cmd, mode) {
            match fx {
                WindowEffect::BeginShutdown => {
                    if self.shutdown == ShutdownPhase::Running {
                        self.shutdown = ShutdownPhase::begin(self.now_ms());
                        // PS-12: cut capture at millisecond 0 of Farewell (`!allows_capture()`).
                        if !self.shutdown.allows_capture() {
                            self.capture_running.store(false, Ordering::SeqCst);
                        }
                        self.send_action(OverlayAction::ShutdownStarted);
                        let line = "오늘도 수고했어! 또 봐~";
                        let bye = ZenaAvatarArgs {
                            emotion_state: EmotionState::Joy,
                            body_motion: BodyMotion::WaveGoodbye,
                            character_emote: Some(CharacterEmote::ChibiHeart),
                            speech_text: Some(line.into()),
                            ..Default::default()
                        };
                        let ms = 1800;
                        self.act(
                            &bye,
                            ReactionLevel::Lv3Coaching,
                            zena_core::avatar::viseme_timeline(line, ms),
                            ms,
                        );
                    }
                }
                WindowEffect::ApplyStyle => self.platform_style(),
                WindowEffect::SetTopmost(on) => {
                    #[cfg(windows)]
                    if let Some(h) = self.hwnd() {
                        crate::win32::set_topmost(h, on);
                    }
                    #[cfg(not(windows))]
                    if let Some(w) = &self.window {
                        w.set_window_level(if on {
                            WindowLevel::AlwaysOnTop
                        } else {
                            WindowLevel::Normal
                        });
                    }
                }
                WindowEffect::Show(v) => {
                    self.send_action(OverlayAction::AvatarHidden(!v));
                    #[cfg(windows)]
                    if let Some(h) = self.hwnd() {
                        crate::win32::show(h, v);
                    }
                    #[cfg(not(windows))]
                    if let Some(w) = &self.window {
                        w.set_visible(v);
                    }
                }
                WindowEffect::None => {}
            }
        }
    }

    fn update_nav_badges(&mut self, mut ov: NavBadgeOverlay) {
        let mut occluded = false;
        let mut arrow_deg = None;
        if let Some(w) = &self.window {
            let pos = w.outer_position().unwrap_or(PhysicalPosition::new(0, 0));
            let overlay_box = BBox::new(
                pos.x as f32,
                pos.y as f32,
                pos.x as f32 + VIEWPORT_W as f32,
                pos.y as f32 + VIEWPORT_H as f32,
            );
            let place = AvatarPlacement {
                head_x: pos.x as f32 + VIEWPORT_W as f32 / 2.0,
                head_y: pos.y as f32 + VIEWPORT_H as f32 * 0.3,
                viewer_depth_px: 900.0,
            };
            let proj = NavTargetProjector {
                screen: self.screen,
                overlay: overlay_box,
                place,
            };
            let win_box = ov
                .window_rect_px
                .map(|r| BBox::new(r[0], r[1], r[2], r[3]));
            for b in &ov.badges {
                let rel = BBox::new(b.rect.x, b.rect.y, b.rect.x + b.rect.w, b.rect.y + b.rect.h);
                let pt = proj.project(b.number, win_box, rel);
                if b.number == 1 {
                    arrow_deg = Some(pt.arrow_angle_deg);
                }
                if pt.occluded_by_overlay {
                    occluded = true;
                }
            }
        }
        ov.occluded_by_overlay = occluded;
        ov.primary_arrow_deg = arrow_deg;
        let changed = self.step_aside_occluded != occluded;
        self.step_aside_occluded = occluded;
        self.nav_badges = ov;
        if changed {
            self.platform_style();
        }
    }

    fn toggle_drawer(&mut self) {
        if self.privacy.is_locked() {
            return;
        }
        self.drawer_open = !self.drawer_open;
        if self.drawer_open {
            self.send_action(OverlayAction::AckRejections);
            self.send_action(OverlayAction::RefreshDrawer);
        }
        self.platform_style();
    }

    /// Open / close the chat (or drop-note) box: takes focus & IME while open; zeroizes buffers on close (PS-12).
    fn set_chat(&mut self, open: bool) {
        if self.chat.open == open {
            return;
        }
        self.chat.open = open;
        self.state.typing = open;
        self.send_action(OverlayAction::TypingChanged(open));
        if !open {
            self.chat.drop_target_cc = None;
            let acts = self.privacy.on_chat_closed();
            self.apply_privacy_actions(acts);
        } else {
            zeroize_string(&mut self.chat.preedit);
        }
        self.platform_style();
        if let Some(w) = &self.window {
            w.set_ime_allowed(open);
            if open {
                w.focus_window();
            }
        }
        #[cfg(windows)]
        if open {
            if let Some(h) = self.hwnd() {
                crate::win32::focus(h);
            }
        }
    }

    fn open_drop_note_for(&mut self, cc: String) {
        if self.privacy.is_locked() {
            return;
        }
        self.chat.drop_target_cc = Some(cc);
        self.set_chat(true);
    }

    fn submit_chat(&mut self) {
        let mut text = std::mem::take(&mut self.chat.text);
        let drop_cc = self.chat.drop_target_cc.take();
        if !text.trim().is_empty() {
            if let Some(cc) = drop_cc {
                self.send_action(OverlayAction::DropNote {
                    cc,
                    to: None,
                    text: text.clone(),
                });
            } else {
                let _ = self.utterances.send(text.clone());
            }
        }
        zeroize_string(&mut text);
        self.set_chat(false);
    }

    fn handle_click(&mut self, x: f32, y: f32) {
        let layout = self
            .gfx
            .as_ref()
            .map(|g| g.scene.layout().clone())
            .unwrap_or_default();

        // 1. Stage 2 drawer [▲] / [▼] buttons (highest priority while drawer is open).
        if self.drawer_open {
            for row in &layout.drawer_rows {
                if OverlayLayout::hit_rect(row.up_rect, x, y) {
                    self.send_action(OverlayAction::TogglePin {
                        record_id: row.record_id.clone(),
                        pin: UserPin::Promoted,
                    });
                    return;
                }
                if OverlayLayout::hit_rect(row.down_rect, x, y) {
                    self.send_action(OverlayAction::TogglePin {
                        record_id: row.record_id.clone(),
                        pin: UserPin::Demoted,
                    });
                    return;
                }
            }
        }

        // 2. Stage 1 unseen rejection chip -> open drawer + acknowledge rejections (E-16).
        if layout
            .rejection_rect
            .is_some_and(|r| OverlayLayout::hit_rect(r, x, y))
        {
            self.drawer_open = true;
            self.send_action(OverlayAction::AckRejections);
            self.send_action(OverlayAction::RefreshDrawer);
            self.platform_style();
            return;
        }

        // 3. Stage 0 status pill -> toggle Stage 2 drawer.
        if layout
            .pill_rect
            .is_some_and(|r| OverlayLayout::hit_rect(r, x, y))
        {
            self.toggle_drawer();
            return;
        }

        // 4. Numbered 1–9 badges (respects CommitGuard: final payment buttons focus-only!).
        for &(num, commit_guarded, rect) in &layout.badge_rects {
            if OverlayLayout::hit_rect(rect, x, y) {
                self.send_action(OverlayAction::BadgePicked {
                    number: num,
                    commit_guarded,
                });
                return;
            }
        }
    }

    fn detect_drop_target_cc(&self) -> Option<String> {
        #[cfg(windows)]
        {
            let (cx, cy) = crate::win32::cursor_pos_px()?;
            let win = premium_discount_window::exclude_self(
                premium_discount_window::window_at(cx, cy),
            )?;
            let cc = envelope::normalize_cc(&win.app_uri())
                .map(|c| c.cc)
                .unwrap_or_else(|_| win.exe.to_ascii_lowercase());
            (!cc.is_empty()).then_some(cc)
        }
        #[cfg(not(windows))]
        {
            None
        }
    }

    #[cfg(windows)]
    fn hwnd(&self) -> Option<windows::Win32::Foundation::HWND> {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let w = self.window.as_ref()?;
        match w.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => Some(crate::win32::hwnd_from_raw(h.hwnd.get())),
            _ => None,
        }
    }

    fn effective_ex_style(&self) -> u32 {
        if self.privacy.is_locked() {
            return self.privacy.ex_style(&self.state);
        }
        if self.step_aside_occluded && !self.state.typing && !self.drawer_open {
            return LayerFlags {
                click_through: true,
                topmost: self.state.topmost,
                typing: false,
            }
            .ex_style();
        }
        if self.drawer_open {
            // Drawer open accepts clicks on ▲/▼ and rejection chips even if normally GhostClickThrough.
            return LayerFlags {
                click_through: false,
                topmost: self.state.topmost,
                typing: self.state.typing,
            }
            .ex_style();
        }
        self.state.ex_style()
    }

    fn platform_style(&self) {
        let ex = self.effective_ex_style();
        #[cfg(windows)]
        if let Some(h) = self.hwnd() {
            crate::win32::apply_style_ex(h, ex, self.state.topmost);
        }
        #[cfg(not(windows))]
        if let Some(w) = &self.window {
            let interactive = (ex & zena_core::window::WS_EX_TRANSPARENT) == 0;
            let _ = w.set_cursor_hittest(interactive);
        }
    }

    fn frame_params(&mut self, width: u32, height: u32) -> FrameParams {
        let t = self.t0.elapsed().as_secs_f32();
        let (mut tx, mut ty, mut hx, mut hy) = (0.0, 0.0, 0.0, 0.0);
        if let (Some(p), Some(w)) = (self.anim.gaze, &self.window) {
            let pos = w.outer_position().unwrap_or(PhysicalPosition::new(0, 0));
            let place = AvatarPlacement {
                head_x: pos.x as f32 + VIEWPORT_W as f32 / 2.0,
                head_y: pos.y as f32 + VIEWPORT_H as f32 * 0.3,
                viewer_depth_px: 900.0,
            };
            let look = look_at_event(&self.screen, &place, p);
            self.anim.look = look;
            let prm = live2d_gaze_params(look);
            (hx, hy, tx, ty) = (prm[0].1 / 30.0, prm[1].1 / 30.0, prm[3].1, prm[4].1);
        }
        let k = 0.25;
        self.anim.eye[0] += (tx - self.anim.eye[0]) * k;
        self.anim.eye[1] += (ty - self.anim.eye[1]) * k;
        self.anim.head[0] += (hx - self.anim.head[0]) * k * 0.5;
        self.anim.head[1] += (hy - self.anim.head[1]) * k * 0.5;
        if Instant::now() > self.anim.expression_until && !self.privacy.is_locked() {
            self.anim.emotion = EmotionState::Neutral;
            self.anim.motion = BodyMotion::Idle;
        }
        let (brow, eye_open, base_form) = match self.anim.emotion {
            EmotionState::Joy | EmotionState::Excited => (0.0, 0.9, 0.8),
            EmotionState::Surprised | EmotionState::SurprisedUrgent => (-0.3, 1.35, 0.0),
            EmotionState::Angry => (1.0, 0.9, -0.3),
            EmotionState::Sorrow | EmotionState::Worried => (-1.0, 0.85, -0.4),
            EmotionState::Focused => (0.4, 0.95, 0.0),
            EmotionState::Relieved => (-0.2, 0.8, 0.5),
            EmotionState::Neutral => (0.0, 1.0, 0.2),
        };
        let (mut mouth_open, mut mouth_form, mut viseme) = (0.0, base_form, Viseme::Rest);
        if self.speaking() {
            let el = self.anim.speech_start.elapsed().as_millis() as u32;
            viseme = self
                .anim
                .visemes
                .iter()
                .rev()
                .find(|(s, _)| *s <= el)
                .map(|x| x.1)
                .unwrap_or(Viseme::Rest);
            let rms = 0.25 + 0.2 * (t * 17.0).sin().abs();
            let m = live2d_mouth(rms, viseme);
            mouth_open = m[0].1;
            mouth_form = 0.5 * base_form + 0.5 * m[1].1;
        }
        let hover = {
            #[cfg(windows)]
            {
                self.hwnd().map(crate::win32::cursor_over).unwrap_or(false)
            }
            #[cfg(not(windows))]
            {
                false
            }
        };
        let alpha_u8 = if self.step_aside_occluded && !self.state.typing && !self.drawer_open {
            HOVER_ALPHA
        } else {
            self.privacy.alpha(&self.state, hover)
        };
        FrameParams {
            gaze: self.anim.eye,
            head: self.anim.head,
            mouth_open,
            mouth_form,
            brow,
            eye_open,
            alpha: alpha_u8 as f32 / 255.0,
            time: t,
            width,
            height,
            emotion: self.anim.emotion,
            motion: self.anim.motion,
            viseme,
            look: self.anim.look,
            motion_t: self.anim.motion_start.elapsed().as_secs_f32(),
            lift: 0.0,
        }
    }

    fn emote_show(&self) -> Option<EmoteShow> {
        let (e, since) = self.anim.emote?;
        let now = Instant::now();
        let left = self
            .anim
            .expression_until
            .saturating_duration_since(now)
            .as_secs_f32();
        let fade = (left / EMOTE_FADE_S).clamp(0.0, 1.0);
        if fade <= 0.0 {
            return None;
        }
        let angle_deg = if let Some(a) = self.nav_badges.primary_arrow_deg {
            a
        } else {
            let l = self.anim.look;
            if l.yaw_deg.abs() + l.pitch_deg.abs() > 1.0 {
                l.pitch_deg.atan2(l.yaw_deg).to_degrees()
            } else {
                self.anim.gaze.map(|g| g.theta_deg).unwrap_or(0.0)
            }
        };
        Some(EmoteShow {
            emote: e,
            age: since.elapsed().as_secs_f32(),
            fade,
            angle_deg,
        })
    }

    fn bubble_alpha(&self) -> f32 {
        if self.bubble.text.is_empty() {
            return 0.0;
        }
        let end = self.anim.speech_start
            + Duration::from_millis(self.anim.speech_ms as u64 + BUBBLE_HOLD_MS);
        let left = end.saturating_duration_since(Instant::now()).as_secs_f32();
        (left / 0.5).clamp(0.0, 1.0)
    }

    fn pill_info(&self) -> Option<(String, [f32; 4])> {
        if self.privacy.is_locked() {
            return Some(("● 보안 보호 중".into(), [0.95, 0.68, 0.24, 1.0]));
        }
        match self.sync_view.status_dot.as_deref() {
            Some("attention") => {
                let msg = if self.sync_view.auth_required {
                    "● 로그인 필요"
                } else {
                    "● 거부 알림 확인"
                };
                Some((msg.into(), [0.95, 0.36, 0.38, 1.0]))
            }
            Some("queued") => Some((
                format!("● 대기 {}건", self.sync_view.queued_count.max(1)),
                [0.95, 0.78, 0.28, 1.0],
            )),
            Some("pending") => Some((
                format!("● 검토 {}건", self.sync_view.pending_count.max(1)),
                [0.45, 0.72, 0.98, 1.0],
            )),
            Some("stored") => Some(("● 동기화 완료 · 서랍".into(), [0.28, 0.85, 0.56, 1.0])),
            _ if !self.sync_view.drawer_items.is_empty() || self.drawer_open => {
                Some(("● 기록 서랍 (Ctrl+Shift+D)".into(), [0.62, 0.48, 0.96, 1.0]))
            }
            _ => None,
        }
    }

    fn rejection_text(&self) -> Option<String> {
        if self.privacy.is_locked() || self.sync_view.unacked_rejections.is_empty() {
            return None;
        }
        let latest = self.sync_view.unacked_rejections.last()?;
        let count = self.sync_view.unacked_rejections.len();
        if count > 1 {
            Some(format!("거부 알림 {count}건 · {} (눌러서 확인)", latest.label_ko))
        } else {
            Some(format!("거부됨: {} (눌러서 확인)", latest.label_ko))
        }
    }

    fn render(&mut self) {
        let Some((w, h)) = self
            .gfx
            .as_ref()
            .map(|g| (g.config.width, g.config.height))
        else {
            return;
        };
        for asset in std::mem::take(&mut self.pending) {
            if let Some(g) = &mut self.gfx {
                g.scene.load(&g.device, &g.queue, asset);
            }
        }
        let params = self.frame_params(w, h);
        let emote = self.emote_show();
        let bubble_alpha = self.bubble_alpha();
        if bubble_alpha <= 0.0 && !self.speaking() {
            self.bubble.fresh = true;
        }
        let caret = (self.t0.elapsed().as_millis() / 500) % 2 == 0;
        let pill = self.pill_info();
        let rejection = self.rejection_text();
        let drop_hint = self
            .chat
            .drop_target_cc
            .as_ref()
            .map(|cc| format!("드롭 메모 대상: {cc} (Enter 저장 · Esc 취소)"));
        let drawer_slice = (self.drawer_open && !self.privacy.is_locked())
            .then_some(self.sync_view.drawer_items.as_slice());
        let badges_slice = if self.privacy.is_locked() {
            &[][..]
        } else {
            self.nav_badges.badges.as_slice()
        };
        let Some(g) = &mut self.gfx else { return };
        let stage = Stage {
            params,
            track: self.track,
            parts: &self.parts,
            emote,
            bubble: (bubble_alpha > 0.0).then_some((self.bubble.text.as_str(), bubble_alpha)),
            input: self
                .chat
                .open
                .then_some((self.chat.text.as_str(), self.chat.preedit.as_str(), caret)),
            sync_pill: pill.as_ref().map(|(t, c)| (t.as_str(), *c)),
            rejection_chip: rejection.as_deref(),
            drawer: drawer_slice,
            badges: badges_slice,
            drop_hint: drop_hint.as_deref(),
        };
        if let Some(a) = &self.assets {
            for m in g.scene.missing(&stage) {
                if self.requested.insert(m.clone()) {
                    a.request(&m);
                }
            }
        }
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) => f,
            wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
                g.surface.configure(&g.device, &g.config);
                f
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            _ => {
                g.surface.configure(&g.device, &g.config);
                return;
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("zena"),
            });
        g.scene
            .render(&g.device, &g.queue, &mut enc, &view, &g.depth, &stage);
        g.queue.submit([enc.finish()]);
        g.queue.present(frame);
    }

    fn init_gfx(window: Arc<Window>, font: Option<&std::path::Path>) -> anyhow::Result<Gfx> {
        #[allow(unused_mut)]
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        #[cfg(windows)]
        {
            desc.backends = wgpu::Backends::DX12;
            desc.backend_options.dx12.presentation_system =
                wgpu::Dx12SwapchainKind::DxgiFromVisual;
        }
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("raon-zena"),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("surface unsupported"))?;
        let caps = surface.get_capabilities(&adapter);
        config.alpha_mode = [
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::PostMultiplied,
        ]
        .into_iter()
        .find(|m| caps.alpha_modes.contains(m))
        .unwrap_or(caps.alpha_modes[0]);
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        eprintln!(
            "[overlay] {} ({:?}), {}x{} {:?}",
            adapter.get_info().name,
            adapter.get_info().backend,
            config.width,
            config.height,
            config.format
        );
        let depth = depth_texture(&device, config.width, config.height);
        let scene = Scene::new(&device, config.format, font);
        Ok(Gfx {
            surface,
            device,
            queue,
            config,
            depth,
            scene,
        })
    }

    fn on_key(&mut self, ev: &winit::event::KeyEvent) {
        if ev.state != ElementState::Pressed {
            return;
        }
        if !self.chat.open {
            match &ev.logical_key {
                Key::Named(NamedKey::Escape) if self.drawer_open => {
                    self.drawer_open = false;
                    self.platform_style();
                }
                Key::Named(NamedKey::Escape) if self.speaking() => {
                    self.clear_speech();
                    self.send_action(OverlayAction::DismissCoaching);
                }
                Key::Character(ch) if ch.len() == 1 => {
                    if let Some(d) = ch.chars().next().and_then(|c| c.to_digit(10)) {
                        if (1..=9).contains(&d) {
                            let num = d as u8;
                            if let Some((badge, dispatch)) = self.nav_badges.dispatch_for(num) {
                                self.send_action(OverlayAction::BadgePicked {
                                    number: badge.number,
                                    commit_guarded: dispatch
                                        == zena_core::brief::SlotDispatch::FocusOnly,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
            return;
        }
        match &ev.logical_key {
            Key::Named(NamedKey::Enter) if self.chat.preedit.is_empty() => self.submit_chat(),
            Key::Named(NamedKey::Escape) => self.set_chat(false),
            Key::Named(NamedKey::Backspace) if self.chat.preedit.is_empty() => {
                self.chat.text.pop();
            }
            _ => {
                if let Some(t) = &ev.text {
                    if self.chat.preedit.is_empty() {
                        self.chat.text.extend(t.chars().filter(|c| !c.is_control()));
                    }
                }
            }
        }
    }
}

impl ApplicationHandler<UserEvent> for Overlay {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Raon Zena")
            .with_inner_size(LogicalSize::new(VIEWPORT_W, VIEWPORT_H))
            .with_transparent(true)
            .with_decorations(false)
            .with_resizable(false)
            .with_window_level(WindowLevel::AlwaysOnTop);
        #[cfg(windows)]
        let attrs = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attrs.with_skip_taskbar(true)
        };
        let window = Arc::new(el.create_window(attrs).expect("create window"));
        match Self::init_gfx(window.clone(), self.font.as_deref()) {
            Ok(g) => self.gfx = Some(g),
            Err(e) => eprintln!("GPU init failed: {e:#}"),
        }
        self.window = Some(window);
        self.platform_style();
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::Cmd(AvatarCmd::Gaze(g)) => {
                if self.privacy.allows_screen_reactions() {
                    self.anim.gaze = Some(Polar {
                        r: g.distance_norm,
                        theta_deg: g.angle_deg,
                    });
                }
            }
            UserEvent::Cmd(AvatarCmd::PreThink { .. }) => {
                // 0 ms pre-think pose while the 600 ms RAG race runs (CR-05).
                let think = ZenaAvatarArgs {
                    emotion_state: EmotionState::Focused,
                    body_motion: BodyMotion::Think,
                    ..Default::default()
                };
                self.express(&think, ReactionLevel::Lv2Cached);
            }
            UserEvent::Cmd(AvatarCmd::Act {
                args,
                level,
                visemes,
                speech_ms,
            }) => self.act(&args, level, visemes, speech_ms),
            UserEvent::Cmd(AvatarCmd::Expression { args, level }) => {
                if !self.speaking() {
                    self.bubble.fresh = true;
                }
                self.express(&args, level)
            }
            UserEvent::Cmd(AvatarCmd::Speak {
                text,
                level,
                visemes,
                speech_ms,
            }) => self.speak(&text, level, visemes, speech_ms),
            UserEvent::Cmd(AvatarCmd::System(c, m)) => self.system(c, m),
            UserEvent::Cmd(AvatarCmd::Asset(a)) => self.asset(&a),
            UserEvent::Cmd(AvatarCmd::PrivacyLock { locked }) => self.set_privacy_locked(locked),
            UserEvent::Cmd(AvatarCmd::ClearSpeech) => self.clear_speech(),
            UserEvent::Cmd(AvatarCmd::NavBadges(ov)) => self.update_nav_badges(ov),
            UserEvent::Cmd(AvatarCmd::SyncStatus(view)) => self.sync_view = view,
            UserEvent::Hotkey(Hotkey::Chat) => {
                let open = !self.chat.open;
                self.set_chat(open);
            }
            UserEvent::Hotkey(h) => {
                let (c, m) = h.command(&self.state);
                self.system(c, m);
            }
            UserEvent::DrawerToggle => self.toggle_drawer(),
            UserEvent::Asset(a) => self.pending.push(*a),
            UserEvent::AssetFailed(m) => {
                eprintln!("[assets] {m:?} unavailable");
                if let Some(g) = &mut self.gfx {
                    g.scene.failed.insert(m);
                }
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => self.system(SystemCommand::ExitApp, None),
            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = s.width.max(1);
                    g.config.height = s.height.max(1);
                    g.surface.configure(&g.device, &g.config);
                    g.depth = depth_texture(&g.device, g.config.width, g.config.height);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x, position.y);
                if let Some((px, py)) = self.press_pos {
                    let dist = (position.x - px).hypot(position.y - py);
                    if !self.dragging
                        && dist >= DRAG_THRESHOLD
                        && self.state.mode == WindowLayerMode::InteractivePet
                        && !self.privacy.is_locked()
                    {
                        self.dragging = true;
                        self.send_action(OverlayAction::DraggingChanged(true));
                        if let Some(w) = &self.window {
                            let _ = w.drag_window();
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !self.privacy.is_locked() {
                    self.press_pos = Some(self.cursor_pos);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                let was_dragging = std::mem::take(&mut self.dragging);
                let press = self.press_pos.take();
                if was_dragging {
                    self.send_action(OverlayAction::DraggingChanged(false));
                    if let Some(cc) = self.detect_drop_target_cc() {
                        self.open_drop_note_for(cc);
                    }
                } else if press.is_some() {
                    let (cx, cy) = self.cursor_pos;
                    self.handle_click(cx as f32, cy as f32);
                }
            }
            WindowEvent::DroppedFile(path) => {
                if !self.privacy.is_locked() {
                    let text = std::fs::read_to_string(&path)
                        .map(|s| s.chars().take(1024).collect::<String>())
                        .unwrap_or_else(|_| {
                            path.file_name()
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_default()
                        });
                    if !text.trim().is_empty() {
                        let cc = self.detect_drop_target_cc().unwrap_or_default();
                        self.send_action(OverlayAction::DropNote { cc, to: None, text });
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => self.on_key(&event),
            WindowEvent::Ime(Ime::Preedit(text, _)) if self.chat.open => self.chat.preedit = text,
            WindowEvent::Ime(Ime::Commit(text)) if self.chat.open => {
                zeroize_string(&mut self.chat.preedit);
                self.chat.text.push_str(&text);
            }
            WindowEvent::Focused(false) if self.chat.open => self.set_chat(false),
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
        if self.chat.open || self.drawer_open {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
        if self.shutdown == ShutdownPhase::Exit {
            el.exit();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let next = self.shutdown.step(self.now_ms());
        if next != self.shutdown {
            match next {
                ShutdownPhase::StopCapture => self.capture_running.store(false, Ordering::SeqCst),
                ShutdownPhase::ReleaseResources => {
                    self.gfx = None;
                }
                ShutdownPhase::Exit => {
                    el.exit();
                    return;
                }
                _ => {}
            }
            self.shutdown = next;
        }
        let animating = self.emote_show().is_some() || self.bubble_alpha() > 0.0;
        let level = if self.speaking() || self.chat.open {
            ReactionLevel::Lv3Coaching
        } else if animating {
            ReactionLevel::Lv2Cached
        } else {
            self.anim.level.min(ReactionLevel::Lv2Cached)
        };
        match self
            .privacy
            .frame_interval(level, self.speaking(), self.state.visible)
        {
            Some(dt) => {
                el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + dt));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            None => el.set_control_flow(ControlFlow::Wait),
        }
        if self.shutdown != ShutdownPhase::Running {
            el.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(16),
            ));
        }
        if !animating && !self.speaking() {
            self.anim.level = ReactionLevel::Lv1Silent;
        }
    }
}

/// Forwards what the asset service produced to the UI thread.
pub struct OverlayAssets(pub winit::event_loop::EventLoopProxy<UserEvent>);

impl AssetLink for OverlayAssets {
    fn ready(&self, l: Loaded) {
        let _ = self.0.send_event(UserEvent::Asset(Box::new(l)));
    }
    fn failed(&self, m: Missing) {
        let _ = self.0.send_event(UserEvent::AssetFailed(m));
    }
}
