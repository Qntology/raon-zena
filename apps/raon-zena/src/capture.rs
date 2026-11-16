//! Frame sources for the Stage 1 sensor.
//!
//! * [`SyntheticSource`] – deterministic moving shapes (development on any OS, tests).
//! * `DxgiSource` (Windows) – DXGI Desktop Duplication of the primary output, read back and
//!   decimated on the CPU. The zero-copy path (sharing the duplicated D3D11 texture with the
//!   wgpu DX12 device and running `event_diff.wgsl` on it) is the next step; see docs/ARCHITECTURE.md.

pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Tightly packed BGRA8.
    pub bgra: Vec<u8>,
}

pub trait FrameSource: Send {
    /// Block until the next frame (or timeout); `None` when no new frame is available.
    fn next_frame(&mut self) -> anyhow::Result<Option<Frame>>;
    fn size(&self) -> (usize, usize);
}

/// A dim background with a bright "monster" spiralling toward the centre and an occasional
/// outward "skill" burst from the centre.
pub struct SyntheticSource {
    pub w: usize,
    pub h: usize,
    t: u32,
}

impl SyntheticSource {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, t: 0 }
    }
}

impl FrameSource for SyntheticSource {
    fn next_frame(&mut self) -> anyhow::Result<Option<Frame>> {
        std::thread::sleep(std::time::Duration::from_millis(33));
        self.t += 1;
        let (w, h) = (self.w, self.h);
        let mut f = vec![24u8; w * h * 4];
        let phase = (self.t % 150) as f32 / 150.0;
        let ang = phase * std::f32::consts::TAU * 0.75;
        let rad = (1.0 - phase) * 0.45 * h as f32;
        let (cx, cy) = (w as f32 / 2.0 + rad * ang.cos(), h as f32 / 2.0 - rad * ang.sin());
        let s = (h / 12) as isize;
        let mut plot = |x0: isize, y0: isize, half: isize, v: u8| {
            for y in (y0 - half).max(0)..(y0 + half).min(h as isize) {
                for x in (x0 - half).max(0)..(x0 + half).min(w as isize) {
                    let i = (y as usize * w + x as usize) * 4;
                    f[i..i + 3].fill(v);
                }
            }
        };
        plot(cx as isize, cy as isize, s, 220);
        if self.t % 90 < 10 {
            let r = (self.t % 90) as isize * (h as isize / 30);
            plot(w as isize / 2 + r, h as isize / 2, s / 2, 255);
        }
        Ok(Some(Frame {
            width: w,
            height: h,
            bgra: f,
        }))
    }
    fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
}

#[cfg(windows)]
pub use dxgi::DxgiSource;

#[cfg(windows)]
mod dxgi {
    use super::{Frame, FrameSource};
    use anyhow::{Context, Result};
    use windows::core::Interface;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
    use windows::Win32::Graphics::Direct3D11::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    pub struct DxgiSource {
        device: ID3D11Device,
        ctx: ID3D11DeviceContext,
        dup: IDXGIOutputDuplication,
        staging: Option<ID3D11Texture2D>,
        src_w: usize,
        src_h: usize,
        /// Keep every `step`-th pixel in both axes (1080p / 4 → 480×270).
        step: usize,
    }

    // The COM objects are only used from the capture thread that owns the source.
    unsafe impl Send for DxgiSource {}

    impl DxgiSource {
        pub fn new(step: usize) -> Result<Self> {
            unsafe {
                let mut device = None;
                let mut ctx = None;
                D3D11CreateDevice(
                    None,
                    D3D_DRIVER_TYPE_HARDWARE,
                    HMODULE::default(),
                    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                    None,
                    D3D11_SDK_VERSION,
                    Some(&mut device),
                    None,
                    Some(&mut ctx),
                )?;
                let device: ID3D11Device = device.context("no D3D11 device")?;
                let ctx = ctx.context("no D3D11 context")?;
                let adapter = device.cast::<IDXGIDevice>()?.GetAdapter()?;
                let output: IDXGIOutput1 = adapter.EnumOutputs(0)?.cast()?;
                let desc = output.GetDesc()?;
                let r = desc.DesktopCoordinates;
                let dup = output.DuplicateOutput(&device)?;
                Ok(Self {
                    device,
                    ctx,
                    dup,
                    staging: None,
                    src_w: (r.right - r.left) as usize,
                    src_h: (r.bottom - r.top) as usize,
                    step: step.max(1),
                })
            }
        }
    }

    impl FrameSource for DxgiSource {
        fn size(&self) -> (usize, usize) {
            (self.src_w / self.step, self.src_h / self.step)
        }

        fn next_frame(&mut self) -> Result<Option<Frame>> {
            unsafe {
                let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
                let mut res: Option<IDXGIResource> = None;
                match self.dup.AcquireNextFrame(50, &mut info, &mut res) {
                    Ok(()) => {}
                    Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => return Ok(None),
                    Err(e) => return Err(e.into()),
                }
                let tex: ID3D11Texture2D = res.context("no frame resource")?.cast()?;
                if self.staging.is_none() {
                    let mut d = D3D11_TEXTURE2D_DESC::default();
                    tex.GetDesc(&mut d);
                    d.Usage = D3D11_USAGE_STAGING;
                    d.BindFlags = 0;
                    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ.0 as u32;
                    d.MiscFlags = 0;
                    d.MipLevels = 1;
                    d.ArraySize = 1;
                    d.SampleDesc = DXGI_SAMPLE_DESC { Count: 1, Quality: 0 };
                    let mut st = None;
                    self.device.CreateTexture2D(&d, None, Some(&mut st))?;
                    self.staging = st;
                }
                let staging = self.staging.as_ref().unwrap();
                self.ctx.CopyResource(staging, &tex);
                let mut m = D3D11_MAPPED_SUBRESOURCE::default();
                self.ctx.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut m))?;
                let (w, h) = self.size();
                let mut out = vec![0u8; w * h * 4];
                let base = m.pData as *const u8;
                for y in 0..h {
                    let row = base.add(y * self.step * m.RowPitch as usize);
                    for x in 0..w {
                        let s = row.add(x * self.step * 4);
                        std::ptr::copy_nonoverlapping(s, out.as_mut_ptr().add((y * w + x) * 4), 4);
                    }
                }
                self.ctx.Unmap(staging, 0);
                self.dup.ReleaseFrame()?;
                Ok(Some(Frame {
                    width: w,
                    height: h,
                    bgra: out,
                }))
            }
        }
    }
}
