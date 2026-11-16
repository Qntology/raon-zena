//! Window/app control that does not need Win32 to be *computed*: extended-style bit math,
//! system-command effects, graceful shutdown sequencing, hotkeys and FPS throttling.
//! The Windows client (`apps/raon-zena`) applies the results with `SetWindowLongPtrW`,
//! `SetWindowPos` and `ShowWindow`.

use crate::reaction::ReactionLevel;
use crate::tools::{SystemCommand, WindowLayerMode};
use std::time::Duration;

// Win32 extended window styles (winuser.h).
pub const WS_EX_TOPMOST: u32 = 0x0000_0008;
pub const WS_EX_TRANSPARENT: u32 = 0x0000_0020;
pub const WS_EX_TOOLWINDOW: u32 = 0x0000_0080;
pub const WS_EX_LAYERED: u32 = 0x0008_0000;
pub const WS_EX_NOACTIVATE: u32 = 0x0800_0000;

/// Micro-viewport: the swap chain only covers the character, not the monitor.
pub const VIEWPORT_W: u32 = 350;
pub const VIEWPORT_H: u32 = 450;

/// Opacity (0..=255, for `SetLayeredWindowAttributes`) when the cursor hovers the avatar in
/// ghost mode, so the game underneath stays visible.
pub const HOVER_ALPHA: u8 = (255.0 * 0.30) as u8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowState {
    pub mode: WindowLayerMode,
    pub topmost: bool,
    pub visible: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            mode: WindowLayerMode::GhostClickThrough,
            topmost: true,
            visible: true,
        }
    }
}

impl WindowState {
    /// The extended style to apply for the current state.
    pub fn ex_style(&self) -> u32 {
        let mut s = WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
        if self.topmost {
            s |= WS_EX_TOPMOST;
        }
        if self.mode == WindowLayerMode::GhostClickThrough {
            s |= WS_EX_TRANSPARENT;
        }
        s
    }

    /// Opacity for the current cursor position.
    pub fn alpha(&self, cursor_over_avatar: bool) -> u8 {
        if self.mode == WindowLayerMode::GhostClickThrough && cursor_over_avatar {
            HOVER_ALPHA
        } else {
            255
        }
    }
}

/// What the platform layer must do after a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowEffect {
    None,
    ApplyStyle,
    SetTopmost(bool),
    Show(bool),
    BeginShutdown,
}

pub fn apply_command(st: &mut WindowState, cmd: SystemCommand, mode: Option<WindowLayerMode>) -> Vec<WindowEffect> {
    let mut fx = Vec::new();
    if let Some(m) = mode {
        if m != st.mode {
            st.mode = m;
            fx.push(WindowEffect::ApplyStyle);
        }
    }
    match cmd {
        SystemCommand::KeepActive => {}
        SystemCommand::ExitApp => fx.push(WindowEffect::BeginShutdown),
        SystemCommand::SetTopmostOn | SystemCommand::SetTopmostOff => {
            let on = cmd == SystemCommand::SetTopmostOn;
            if st.topmost != on {
                st.topmost = on;
                fx.push(WindowEffect::SetTopmost(on));
            }
        }
        SystemCommand::HideAvatar | SystemCommand::ShowAvatar => {
            let vis = cmd == SystemCommand::ShowAvatar;
            if st.visible != vis {
                st.visible = vis;
                fx.push(WindowEffect::Show(vis));
            }
        }
    }
    if fx.is_empty() {
        fx.push(WindowEffect::None);
    }
    fx
}

/// Global hotkeys (`RegisterHotKey`): MOD_CONTROL|MOD_SHIFT + key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hotkey {
    Quit,
    HideToggle,
    InteractiveToggle,
}

impl Hotkey {
    pub const ALL: [Hotkey; 3] = [Hotkey::Quit, Hotkey::HideToggle, Hotkey::InteractiveToggle];
    /// Virtual-key code (`'Q'`, `'H'`, `'L'`).
    pub fn vk(self) -> u32 {
        match self {
            Hotkey::Quit => b'Q' as u32,
            Hotkey::HideToggle => b'H' as u32,
            Hotkey::InteractiveToggle => b'L' as u32,
        }
    }
    pub fn command(self, st: &WindowState) -> (SystemCommand, Option<WindowLayerMode>) {
        match self {
            Hotkey::Quit => (SystemCommand::ExitApp, None),
            Hotkey::HideToggle => (
                if st.visible {
                    SystemCommand::HideAvatar
                } else {
                    SystemCommand::ShowAvatar
                },
                None,
            ),
            Hotkey::InteractiveToggle => (
                SystemCommand::KeepActive,
                Some(match st.mode {
                    WindowLayerMode::GhostClickThrough => WindowLayerMode::InteractivePet,
                    WindowLayerMode::InteractivePet => WindowLayerMode::GhostClickThrough,
                }),
            ),
        }
    }
}

/// Graceful shutdown (PRD §5.3): farewell wave 2.5 s → stop capture → release GPU/mmap → exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownPhase {
    Running,
    Farewell { started_ms: u64 },
    StopCapture,
    ReleaseResources,
    Exit,
}

pub const FAREWELL_MS: u64 = 2500;

impl ShutdownPhase {
    /// Advance one step; `Farewell` waits for the motion to finish, the others progress
    /// as soon as the platform layer reports the previous step done.
    pub fn step(self, now_ms: u64) -> ShutdownPhase {
        match self {
            ShutdownPhase::Running => ShutdownPhase::Running,
            ShutdownPhase::Farewell { started_ms } if now_ms.saturating_sub(started_ms) < FAREWELL_MS => self,
            ShutdownPhase::Farewell { .. } => ShutdownPhase::StopCapture,
            ShutdownPhase::StopCapture => ShutdownPhase::ReleaseResources,
            ShutdownPhase::ReleaseResources | ShutdownPhase::Exit => ShutdownPhase::Exit,
        }
    }
}

/// Dynamic frame pacing (`ControlFlow::WaitUntil`): 60 FPS while speaking/moving (Lv3),
/// 24 FPS for Lv2 cues, 15 FPS idle; hidden windows only wake for input.
pub fn frame_interval(level: ReactionLevel, speaking: bool, visible: bool) -> Option<Duration> {
    if !visible {
        return None;
    }
    let fps = if speaking || level == ReactionLevel::Lv3Coaching {
        60
    } else if level == ReactionLevel::Lv2Cached {
        24
    } else {
        15
    };
    Some(Duration::from_micros(1_000_000 / fps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_mode_flags_match_prd() {
        let st = WindowState::default();
        assert_eq!(
            st.ex_style(),
            WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        );
        let pet = WindowState {
            mode: WindowLayerMode::InteractivePet,
            ..st
        };
        assert_eq!(pet.ex_style() & WS_EX_TRANSPARENT, 0);
        assert_eq!(st.alpha(true), 76);
        assert_eq!(pet.alpha(true), 255);
    }

    #[test]
    fn commands_produce_effects() {
        let mut st = WindowState::default();
        assert_eq!(
            apply_command(&mut st, SystemCommand::SetTopmostOff, None),
            vec![WindowEffect::SetTopmost(false)]
        );
        assert_eq!(apply_command(&mut st, SystemCommand::SetTopmostOff, None), vec![WindowEffect::None]);
        assert_eq!(
            apply_command(&mut st, SystemCommand::HideAvatar, None),
            vec![WindowEffect::Show(false)]
        );
        let (c, m) = Hotkey::InteractiveToggle.command(&st);
        assert_eq!(apply_command(&mut st, c, m), vec![WindowEffect::ApplyStyle]);
        assert_eq!(st.mode, WindowLayerMode::InteractivePet);
        assert_eq!(
            apply_command(&mut st, SystemCommand::ExitApp, None),
            vec![WindowEffect::BeginShutdown]
        );
    }

    #[test]
    fn shutdown_sequence() {
        let mut p = ShutdownPhase::Farewell { started_ms: 1000 };
        p = p.step(2000);
        assert!(matches!(p, ShutdownPhase::Farewell { .. }));
        p = p.step(3600);
        assert_eq!(p, ShutdownPhase::StopCapture);
        p = p.step(3600);
        assert_eq!(p, ShutdownPhase::ReleaseResources);
        assert_eq!(p.step(3600), ShutdownPhase::Exit);
    }

    #[test]
    fn fps_throttle() {
        assert_eq!(
            frame_interval(ReactionLevel::Lv3Coaching, false, true),
            Some(Duration::from_micros(16_666))
        );
        assert_eq!(
            frame_interval(ReactionLevel::Lv1Silent, false, true),
            Some(Duration::from_micros(66_666))
        );
        assert_eq!(frame_interval(ReactionLevel::Lv1Silent, false, false), None);
    }
}
