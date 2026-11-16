//! The 350×450 transparent micro-viewport: winit window + one wgpu device, procedural avatar,
//! dynamic FPS, Win32 layering and the graceful shutdown sequence.

use crate::pipeline::AvatarCmd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowId, WindowLevel};
use zena_core::avatar::{live2d_gaze_params, live2d_mouth, look_at_event, AvatarPlacement, Viseme};
use zena_core::geometry::{Polar, ScreenGeometry};
use zena_core::reaction::ReactionLevel;
use zena_core::tools::{BodyMotion, EmotionState, SystemCommand, WindowLayerMode, ZenaAvatarArgs};
use zena_core::window::{apply_command, frame_interval, Hotkey, ShutdownPhase, WindowEffect, WindowState, VIEWPORT_H, VIEWPORT_W};

#[derive(Debug, Clone)]
pub enum UserEvent {
    Cmd(AvatarCmd),
    #[cfg_attr(not(windows), allow(dead_code))]
    Hotkey(Hotkey),
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Uniforms {
    gaze: [f32; 2],
    head: [f32; 2],
    mouth_open: f32,
    mouth_form: f32,
    brow: f32,
    eye_open: f32,
    alpha: f32,
    time: f32,
    _pad: [f32; 2],
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    ubuf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// Smoothed animation state (targets set by commands, current values eased per frame).
struct Anim {
    gaze: Option<Polar>,
    eye: [f32; 2],
    head: [f32; 2],
    emotion: EmotionState,
    motion: BodyMotion,
    level: ReactionLevel,
    speech_start: Instant,
    speech_ms: u32,
    visemes: Vec<(u32, Viseme)>,
    expression_until: Instant,
}

pub struct Overlay {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    state: WindowState,
    shutdown: ShutdownPhase,
    anim: Anim,
    t0: Instant,
    screen: ScreenGeometry,
    pub capture_running: Arc<AtomicBool>,
}

impl Overlay {
    pub fn new(capture_running: Arc<AtomicBool>, screen: ScreenGeometry) -> Self {
        let now = Instant::now();
        Self {
            window: None,
            gfx: None,
            state: WindowState::default(),
            shutdown: ShutdownPhase::Running,
            anim: Anim {
                gaze: None,
                eye: [0.0; 2],
                head: [0.0; 2],
                emotion: EmotionState::Neutral,
                motion: BodyMotion::Idle,
                level: ReactionLevel::Lv1Silent,
                speech_start: now,
                speech_ms: 0,
                visemes: vec![],
                expression_until: now,
            },
            t0: now,
            screen,
            capture_running,
        }
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn speaking(&self) -> bool {
        self.anim.speech_start.elapsed() < Duration::from_millis(self.anim.speech_ms as u64)
    }

    fn act(&mut self, args: &ZenaAvatarArgs, level: ReactionLevel, visemes: Vec<(u32, Viseme)>, speech_ms: u32) {
        if let Some(g) = args.gaze_target {
            self.anim.gaze = Some(Polar {
                r: g.distance_norm,
                theta_deg: g.angle_deg,
            });
        }
        self.anim.emotion = args.emotion_state;
        self.anim.motion = args.body_motion;
        self.anim.level = level;
        self.anim.visemes = visemes;
        self.anim.speech_ms = speech_ms;
        self.anim.speech_start = Instant::now();
        self.anim.expression_until = Instant::now() + Duration::from_millis(speech_ms as u64 + 1200);
        if let Some(t) = &args.speech_text {
            // TTS backend hook (Raon-OpenTTS); until then the line is logged.
            eprintln!("[zena:{level:?}] {t}");
        }
    }

    fn system(&mut self, cmd: SystemCommand, mode: Option<WindowLayerMode>) {
        for fx in apply_command(&mut self.state, cmd, mode) {
            match fx {
                WindowEffect::BeginShutdown => {
                    if self.shutdown == ShutdownPhase::Running {
                        self.shutdown = ShutdownPhase::Farewell { started_ms: self.now_ms() };
                        let bye = ZenaAvatarArgs {
                            gaze_target: None,
                            emotion_state: EmotionState::Joy,
                            body_motion: BodyMotion::WaveGoodbye,
                            speech_text: Some("오늘도 수고했어! 또 봐~".into()),
                        };
                        let ms = 1800;
                        self.act(
                            &bye,
                            ReactionLevel::Lv3Coaching,
                            zena_core::avatar::viseme_timeline("오늘도 수고했어! 또 봐~", ms),
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
                        w.set_window_level(if on { WindowLevel::AlwaysOnTop } else { WindowLevel::Normal });
                    }
                }
                WindowEffect::Show(v) => {
                    // SW_HIDE: no rendering load; capture + coaching keep running.
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

    #[cfg(windows)]
    fn hwnd(&self) -> Option<windows::Win32::Foundation::HWND> {
        use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
        let w = self.window.as_ref()?;
        match w.window_handle().ok()?.as_raw() {
            RawWindowHandle::Win32(h) => Some(crate::win32::hwnd_from_raw(h.hwnd.get())),
            _ => None,
        }
    }

    fn platform_style(&self) {
        #[cfg(windows)]
        if let Some(h) = self.hwnd() {
            crate::win32::apply_style(h, &self.state);
        }
        #[cfg(not(windows))]
        if let Some(w) = &self.window {
            // Best effort elsewhere: click-through via winit where supported.
            let _ = w.set_cursor_hittest(self.state.mode == WindowLayerMode::InteractivePet);
        }
    }

    fn uniforms(&mut self) -> Uniforms {
        let t = self.t0.elapsed().as_secs_f32();
        // Gaze: from the avatar's head on the desktop toward the event.
        let (mut tx, mut ty, mut hx, mut hy) = (0.0, 0.0, 0.0, 0.0);
        if let (Some(p), Some(w)) = (self.anim.gaze, &self.window) {
            let pos = w.outer_position().unwrap_or(PhysicalPosition::new(0, 0));
            let place = AvatarPlacement {
                head_x: pos.x as f32 + VIEWPORT_W as f32 / 2.0,
                head_y: pos.y as f32 + VIEWPORT_H as f32 * 0.3,
                viewer_depth_px: 900.0,
            };
            let look = look_at_event(&self.screen, &place, p);
            let prm = live2d_gaze_params(look);
            (hx, hy, tx, ty) = (prm[0].1 / 30.0, prm[1].1 / 30.0, prm[3].1, prm[4].1);
        }
        let k = 0.25;
        self.anim.eye[0] += (tx - self.anim.eye[0]) * k;
        self.anim.eye[1] += (ty - self.anim.eye[1]) * k;
        self.anim.head[0] += (hx - self.anim.head[0]) * k * 0.5;
        self.anim.head[1] += (hy - self.anim.head[1]) * k * 0.5;
        if Instant::now() > self.anim.expression_until {
            self.anim.emotion = EmotionState::Neutral;
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
        let (mut mouth_open, mut mouth_form) = (0.0, base_form);
        if self.speaking() {
            let el = self.anim.speech_start.elapsed().as_millis() as u32;
            let v = self
                .anim
                .visemes
                .iter()
                .rev()
                .find(|(s, _)| *s <= el)
                .map(|x| x.1)
                .unwrap_or(Viseme::Rest);
            // Stand-in for the TTS RMS envelope.
            let rms = 0.25 + 0.2 * (t * 17.0).sin().abs();
            let m = live2d_mouth(rms, v);
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
        // Hover fade (30 %) is applied in the shader so the layered window keeps alpha 255.
        let alpha = self.state.alpha(hover);
        Uniforms {
            gaze: self.anim.eye,
            head: self.anim.head,
            mouth_open,
            mouth_form,
            brow,
            eye_open,
            alpha: alpha as f32 / 255.0,
            time: t,
            _pad: [0.0; 2],
        }
    }

    fn render(&mut self) {
        let u = self.uniforms();
        let Some(g) = &self.gfx else { return };
        g.queue.write_buffer(&g.ubuf, 0, bytemuck::bytes_of(&u));
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
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = g
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zena") });
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("avatar"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&g.pipeline);
            rp.set_bind_group(0, &g.bind, &[]);
            rp.draw(0..3, 0..1);
        }
        g.queue.submit([enc.finish()]);
        g.queue.present(frame);
    }

    fn init_gfx(window: Arc<Window>) -> anyhow::Result<Gfx> {
        #[allow(unused_mut)]
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        #[cfg(windows)]
        {
            // Per-pixel transparency on Windows needs a DirectComposition-backed swap chain;
            // capture, sensor, vision and avatar all share this single DX12 device.
            desc.backends = wgpu::Backends::DX12;
            desc.backend_options.dx12.presentation_system = wgpu::Dx12SwapchainKind::DxgiFromVisual;
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
        config.alpha_mode = [wgpu::CompositeAlphaMode::PreMultiplied, wgpu::CompositeAlphaMode::PostMultiplied]
            .into_iter()
            .find(|m| caps.alpha_modes.contains(m))
            .unwrap_or(caps.alpha_modes[0]);
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("avatar"),
            source: wgpu::ShaderSource::Wgsl(include_str!("avatar.wgsl").into()),
        });
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("u"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ubuf.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("avatar"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Ok(Gfx {
            surface,
            device,
            queue,
            config,
            pipeline,
            ubuf,
            bind,
        })
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
        match Self::init_gfx(window.clone()) {
            Ok(g) => self.gfx = Some(g),
            Err(e) => eprintln!("GPU init failed: {e:#}"),
        }
        self.window = Some(window);
        self.platform_style();
    }

    fn user_event(&mut self, _el: &ActiveEventLoop, ev: UserEvent) {
        match ev {
            UserEvent::Cmd(AvatarCmd::Gaze(g)) => {
                self.anim.gaze = Some(Polar {
                    r: g.distance_norm,
                    theta_deg: g.angle_deg,
                })
            }
            UserEvent::Cmd(AvatarCmd::Act {
                args,
                level,
                visemes,
                speech_ms,
            }) => self.act(&args, level, visemes, speech_ms),
            UserEvent::Cmd(AvatarCmd::System(c, m)) => self.system(c, m),
            UserEvent::Hotkey(h) => {
                let (c, m) = h.command(&self.state);
                self.system(c, m);
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
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.state.mode == WindowLayerMode::InteractivePet {
                    if let Some(w) = &self.window {
                        let _ = w.drag_window();
                    }
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
        if self.shutdown == ShutdownPhase::Exit {
            el.exit();
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        // Shutdown sequence: farewell → stop capture → release GPU/mmaps → exit.
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
        let level = if self.speaking() {
            ReactionLevel::Lv3Coaching
        } else {
            self.anim.level.min(ReactionLevel::Lv2Cached)
        };
        match frame_interval(level, self.speaking(), self.state.visible) {
            Some(dt) => {
                el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + dt));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            None => el.set_control_flow(ControlFlow::Wait),
        }
        if self.shutdown != ShutdownPhase::Running {
            el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(16)));
        }
    }
}
