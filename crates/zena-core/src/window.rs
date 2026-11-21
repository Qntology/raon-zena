//! Window/app control that does not need Win32 to be *computed*: extended-style bit math,
//! system-command effects, graceful shutdown sequencing, hotkeys and FPS throttling. The style bits, the micro
//! viewport, the hover alpha and the frame budget are premium-discount's `premium-discount-window` (shared with
//! premium-discount-app's overlay); this module maps Zena's modes and reaction levels onto them.
//! The Windows client (`apps/raon-zena`) applies the results with `SetWindowLongPtrW`,
//! `SetWindowPos` and `ShowWindow`.

use crate::reaction::ReactionLevel;
use crate::tools::{BodyMotion, EmotionState, SystemCommand, WindowLayerMode, ZenaAvatarArgs};
use std::time::Duration;

// Win32 extended window styles (winuser.h), the micro-viewport (the swap chain only covers the character) and the
// hover opacity (ghost mode, so the game underneath stays visible).
pub use premium_discount_window::{
    Activity, LayerFlags, HOVER_ALPHA, VIEWPORT_H, VIEWPORT_W, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    WS_EX_TRANSPARENT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowState {
    pub mode: WindowLayerMode,
    pub topmost: bool,
    pub visible: bool,
    /// The in-overlay chat box is open: the window takes focus and keyboard / IME input until it closes.
    pub typing: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            mode: WindowLayerMode::GhostClickThrough,
            topmost: true,
            visible: true,
            typing: false,
        }
    }
}

impl WindowState {
    /// The layer flags of this state (ghost mode = click-through).
    pub fn flags(&self) -> LayerFlags {
        LayerFlags {
            click_through: self.mode == WindowLayerMode::GhostClickThrough,
            topmost: self.topmost,
            typing: self.typing,
        }
    }

    /// The extended style to apply for the current state.
    pub fn ex_style(&self) -> u32 {
        self.flags().ex_style()
    }

    /// Opacity for the current cursor position.
    pub fn alpha(&self, cursor_over_avatar: bool) -> u8 {
        self.flags().alpha(cursor_over_avatar)
    }

    /// The extended style with the privacy etiquette: while a privacy lock is on, even pet mode is forced
    /// click-through and non-activating (the mode itself is kept, so the lock's end restores it); only an open
    /// chat box (`typing`, BlindAssist) still takes focus.
    pub fn effective_ex_style(&self, privacy_locked: bool) -> u32 {
        if privacy_locked && !self.typing {
            LayerFlags {
                click_through: true,
                topmost: self.topmost,
                typing: false,
            }
            .ex_style()
        } else {
            self.ex_style()
        }
    }

    /// Opacity with the privacy etiquette: while locked (and not typing) always [`HOVER_ALPHA`] (76), wherever the
    /// cursor is, so a virtual keypad behind Zena stays visible.
    pub fn effective_alpha(&self, cursor_over_avatar: bool, privacy_locked: bool) -> u8 {
        if privacy_locked && !self.typing {
            HOVER_ALPHA
        } else {
            self.alpha(cursor_over_avatar)
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
    /// Open / close the chat box (handled by the overlay itself, not a system command).
    Chat,
}

impl Hotkey {
    pub const ALL: [Hotkey; 4] = [Hotkey::Quit, Hotkey::HideToggle, Hotkey::InteractiveToggle, Hotkey::Chat];
    /// Virtual-key code (`'Q'`, `'H'`, `'L'`, `'T'`).
    pub fn vk(self) -> u32 {
        match self {
            Hotkey::Quit => b'Q' as u32,
            Hotkey::HideToggle => b'H' as u32,
            Hotkey::InteractiveToggle => b'L' as u32,
            Hotkey::Chat => b'T' as u32,
        }
    }
    pub fn command(self, st: &WindowState) -> (SystemCommand, Option<WindowLayerMode>) {
        match self {
            Hotkey::Quit => (SystemCommand::ExitApp, None),
            Hotkey::Chat => (SystemCommand::KeepActive, None),
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
    /// The farewell starting now (what [`WindowEffect::BeginShutdown`] begins).
    pub fn begin(now_ms: u64) -> ShutdownPhase {
        ShutdownPhase::Farewell { started_ms: now_ms }
    }

    /// Screen capture may run only while `Running`: it is gated off at the first millisecond of the farewell,
    /// not after the 2.5 s wave (`StopCapture` then only joins the already idle capture thread).
    pub const fn allows_capture(self) -> bool {
        matches!(self, ShutdownPhase::Running)
    }

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
    let activity = if speaking || level == ReactionLevel::Lv3Coaching {
        Activity::Busy
    } else if level == ReactionLevel::Lv2Cached {
        Activity::Light
    } else {
        Activity::Idle
    };
    premium_discount_window::frame_interval(activity, visible)
}

/// What the overlay does when a privacy lock starts or ends (privacy_shield_v1 PS-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivacyAction {
    /// Bump the ask epoch: the running generation / stream stops at once.
    CancelGeneration,
    /// Clear the speech bubble and TTS now (no 2.5 s linger): `speaking = false`.
    ClearSpeech,
    /// Play the silent HIDE_CROUCH reaction ([`PrivacyEtiquette::lock_reaction`], Lv1).
    HideCrouch,
    /// Re-apply the window style and alpha ([`PrivacyEtiquette::ex_style`] / [`PrivacyEtiquette::alpha`]).
    ApplyStyle,
    /// Zero the chat box's input line and IME preedit (`ingress_guard::zeroize_string`).
    ZeroizeInput,
    /// The lock ended: back to the normal pose.
    Resume,
}

/// The overlay's manners while a privacy lock is on: click-through at alpha 76 whatever the mode, no screen
/// reactions, 15 FPS, a silent crouch. Pure state; the app applies the actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrivacyEtiquette {
    locked_since_ms: Option<u64>,
}

impl PrivacyEtiquette {
    pub fn is_locked(&self) -> bool {
        self.locked_since_ms.is_some()
    }

    pub fn locked_since(&self) -> Option<u64> {
        self.locked_since_ms
    }

    /// Edge-triggered: the actions to run when the lock state changes (none when it does not).
    pub fn set_locked(&mut self, locked: bool, now_ms: u64) -> Vec<PrivacyAction> {
        match (self.is_locked(), locked) {
            (false, true) => {
                self.locked_since_ms = Some(now_ms);
                vec![
                    PrivacyAction::CancelGeneration,
                    PrivacyAction::ClearSpeech,
                    PrivacyAction::HideCrouch,
                    PrivacyAction::ApplyStyle,
                ]
            }
            (true, false) => {
                self.locked_since_ms = None;
                vec![PrivacyAction::ApplyStyle, PrivacyAction::Resume]
            }
            _ => Vec::new(),
        }
    }

    /// The chat box closed (sent, Esc or focus lost): its buffers are zeroed, and under a lock the window goes
    /// back to click-through.
    pub fn on_chat_closed(&self) -> Vec<PrivacyAction> {
        if self.is_locked() {
            vec![PrivacyAction::ZeroizeInput, PrivacyAction::ApplyStyle]
        } else {
            vec![PrivacyAction::ZeroizeInput]
        }
    }

    pub fn ex_style(&self, st: &WindowState) -> u32 {
        st.effective_ex_style(self.is_locked())
    }

    pub fn alpha(&self, st: &WindowState, cursor_over_avatar: bool) -> u8 {
        st.effective_alpha(cursor_over_avatar, self.is_locked())
    }

    /// Frame pacing: under the lock screen reactions never raise it above idle (15 FPS); a BlindAssist answer
    /// the player asked for (`speaking`) may still lip-sync.
    pub fn frame_interval(&self, level: ReactionLevel, speaking: bool, visible: bool) -> Option<Duration> {
        frame_interval(if self.is_locked() { ReactionLevel::Lv1Silent } else { level }, speaking, visible)
    }

    /// Whether screen-driven reactions and proactive lines may run.
    pub fn allows_screen_reactions(&self) -> bool {
        !self.is_locked()
    }

    /// The silent reaction when the lock starts.
    pub fn lock_reaction() -> ZenaAvatarArgs {
        ZenaAvatarArgs {
            emotion_state: EmotionState::Neutral,
            body_motion: BodyMotion::HideCrouch,
            ..Default::default()
        }
    }
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
        let typing = WindowState { typing: true, ..st };
        assert_eq!(
            typing.ex_style() & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE),
            0,
            "the chat box takes focus"
        );
        assert_eq!(typing.alpha(true), 255);
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

    #[test]
    fn privacy_lock_overrides_style_and_alpha_except_while_typing() {
        let ghost = WindowState::default();
        let pet = WindowState {
            mode: WindowLayerMode::InteractivePet,
            ..ghost
        };
        let typing = WindowState { typing: true, ..pet };
        // Unlocked: exactly the plain style and alpha.
        for st in [ghost, pet, typing] {
            assert_eq!(st.effective_ex_style(false), st.ex_style());
            assert_eq!(st.effective_alpha(true, false), st.alpha(true));
            assert_eq!(st.effective_alpha(false, false), st.alpha(false));
        }
        assert_eq!(ghost.alpha(false), 255, "ghost mode is opaque away from the cursor normally");
        // Locked: click-through + no-activate + 76 for ghost and pet, cursor or not.
        for st in [ghost, pet] {
            let s = st.effective_ex_style(true);
            assert_eq!(s & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE), WS_EX_TRANSPARENT | WS_EX_NOACTIVATE);
            assert_eq!(s & WS_EX_TOPMOST, WS_EX_TOPMOST);
            assert_eq!(st.effective_alpha(false, true), HOVER_ALPHA);
            assert_eq!(st.effective_alpha(true, true), HOVER_ALPHA);
        }
        // BlindAssist: the chat box the player opened still takes keyboard / IME focus.
        assert_eq!(typing.effective_ex_style(true) & (WS_EX_TRANSPARENT | WS_EX_NOACTIVATE), 0);
        assert_eq!(typing.effective_alpha(false, true), 255);
    }

    #[test]
    fn privacy_etiquette_transitions_and_pacing() {
        let mut e = PrivacyEtiquette::default();
        assert!(e.allows_screen_reactions());
        assert!(e.set_locked(false, 0).is_empty(), "no edge, no action");
        let on = e.set_locked(true, 100);
        assert_eq!(on.first(), Some(&PrivacyAction::CancelGeneration), "the stream stops first");
        assert_eq!(e.locked_since(), Some(100));
        assert!(e.set_locked(true, 200).is_empty());
        assert!(!e.allows_screen_reactions());
        assert_eq!(
            e.frame_interval(ReactionLevel::Lv3Coaching, false, true),
            Some(Duration::from_micros(66_666))
        );
        assert_eq!(
            e.frame_interval(ReactionLevel::Lv2Cached, true, true),
            Some(Duration::from_micros(16_666)),
            "a BlindAssist answer may lip-sync"
        );
        assert_eq!(e.on_chat_closed(), vec![PrivacyAction::ZeroizeInput, PrivacyAction::ApplyStyle]);
        let r = PrivacyEtiquette::lock_reaction();
        assert_eq!((r.body_motion, r.speech_text), (BodyMotion::HideCrouch, None));
        assert_eq!(e.set_locked(false, 300), vec![PrivacyAction::ApplyStyle, PrivacyAction::Resume]);
        assert_eq!(e.on_chat_closed(), vec![PrivacyAction::ZeroizeInput]);
        assert_eq!(
            e.frame_interval(ReactionLevel::Lv3Coaching, false, true),
            Some(Duration::from_micros(16_666))
        );
    }

    #[test]
    fn capture_stops_at_the_start_of_the_farewell_and_the_sequence_is_unchanged() {
        let mut st = WindowState::default();
        assert_eq!(
            apply_command(&mut st, SystemCommand::ExitApp, None),
            vec![WindowEffect::BeginShutdown]
        );
        let mut p = ShutdownPhase::begin(1_000);
        assert!(ShutdownPhase::Running.allows_capture());
        assert!(!p.allows_capture(), "0 ms into the farewell");
        for now in [1_000, 2_000, 3_499] {
            p = p.step(now);
            assert!(matches!(p, ShutdownPhase::Farewell { .. }) && !p.allows_capture());
        }
        p = p.step(3_500);
        assert_eq!(p, ShutdownPhase::StopCapture);
        assert!(!p.allows_capture() && !ShutdownPhase::ReleaseResources.allows_capture() && !ShutdownPhase::Exit.allows_capture());
    }
}
