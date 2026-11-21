//! Win32 window control: extended styles, topmost, visibility, layered alpha, cursor hover,
//! drop-target cursor lookup, and global hotkeys. All style math comes from `zena_core::window`.

#![cfg(windows)]

use std::ffi::c_void;
use windows::Win32::Foundation::{COLORREF, HWND, POINT, RECT};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    RegisterHotKey, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::*;
use zena_core::window::Hotkey;

const HOTKEY_DRAWER_ID: i32 = 100;

/// Global hotkey events delivered to the overlay event loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Core(Hotkey),
    /// `Ctrl+Shift+D`: toggle the Stage 2 history / verdict drawer.
    DrawerToggle,
}

pub fn hwnd_from_raw(raw: isize) -> HWND {
    HWND(raw as *mut c_void)
}

/// Applies an arbitrary computed extended window style (e.g. `PrivacyEtiquette::ex_style` or
/// self-occlusion click-through step-aside) and topmost z-order.
pub fn apply_style_ex(hwnd: HWND, ex_style: u32, topmost: bool) {
    unsafe {
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex_style as isize);
        // A WS_EX_LAYERED window stays invisible until its layered attributes are set once.
        set_alpha(hwnd, 255);
        let after = if topmost { HWND_TOPMOST } else { HWND_NOTOPMOST };
        let _ = SetWindowPos(
            hwnd,
            Some(after),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED,
        );
    }
}

pub fn set_topmost(hwnd: HWND, on: bool) {
    unsafe {
        let after = if on { HWND_TOPMOST } else { HWND_NOTOPMOST };
        let _ = SetWindowPos(
            hwnd,
            Some(after),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

pub fn show(hwnd: HWND, visible: bool) {
    unsafe {
        let _ = ShowWindow(hwnd, if visible { SW_SHOWNOACTIVATE } else { SW_HIDE });
    }
}

/// Bring the overlay to the foreground for typing (the hotkey that asked for it grants the right to).
pub fn focus(hwnd: HWND) {
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOW);
        let _ = SetForegroundWindow(hwnd);
    }
}

pub fn set_alpha(hwnd: HWND, alpha: u8) {
    unsafe {
        let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), alpha, LWA_ALPHA);
    }
}

/// Current desktop cursor position in physical pixels (for drop-target `window_at`).
pub fn cursor_pos_px() -> Option<(i32, i32)> {
    unsafe {
        let mut p = POINT::default();
        GetCursorPos(&mut p).ok()?;
        Some((p.x, p.y))
    }
}

pub fn cursor_over(hwnd: HWND) -> bool {
    unsafe {
        let mut p = POINT::default();
        let mut r = RECT::default();
        if GetCursorPos(&mut p).is_err() || GetWindowRect(hwnd, &mut r).is_err() {
            return false;
        }
        p.x >= r.left && p.x < r.right && p.y >= r.top && p.y < r.bottom
    }
}

/// Register `Ctrl+Shift+{Q,H,L,T}` plus `Ctrl+Shift+D` (Stage 2 drawer) on a dedicated thread.
pub fn spawn_hotkeys(send: impl Fn(HotkeyEvent) + Send + 'static) {
    std::thread::spawn(move || unsafe {
        for (i, h) in Hotkey::ALL.iter().enumerate() {
            let _ = RegisterHotKey(
                None,
                i as i32 + 1,
                MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT,
                h.vk(),
            );
        }
        let _ = RegisterHotKey(
            None,
            HOTKEY_DRAWER_ID,
            MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT,
            b'D' as u32,
        );
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            if msg.message == WM_HOTKEY {
                let id = msg.wParam.0 as i32;
                if id == HOTKEY_DRAWER_ID {
                    send(HotkeyEvent::DrawerToggle);
                } else if let Some(h) = Hotkey::ALL.get(msg.wParam.0.wrapping_sub(1)) {
                    send(HotkeyEvent::Core(*h));
                }
            }
        }
    });
}
