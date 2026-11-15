//! KRAFTON Raon Zena — overlay client.
//!
//! ```text
//! raon-zena [--synthetic] [--model out/raon-1-nano.pyramid.gguf] [--vision out/raon-ve-vision.pyramid.gguf]
//! ```
//! Type a line on stdin to talk to Zena while it runs (voice input is not wired yet).

mod capture;
mod llm;
mod overlay;
mod pipeline;
#[cfg(windows)]
mod win32;

use anyhow::{bail, Result};
use raon_quant::gguf::GgufFile;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use winit::event_loop::EventLoop;
use zena_core::geometry::ScreenGeometry;

struct Args {
    synthetic: bool,
    model: Option<PathBuf>,
    vision: Option<PathBuf>,
}

fn args() -> Args {
    let mut a = Args {
        synthetic: !cfg!(windows),
        model: None,
        vision: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(x) = it.next() {
        match x.as_str() {
            "--synthetic" => a.synthetic = true,
            "--model" => a.model = it.next().map(PathBuf::from),
            "--vision" => a.vision = it.next().map(PathBuf::from),
            _ => eprintln!("ignoring argument {x}"),
        }
    }
    a
}

/// Map a pyramid GGUF and report what would be resident vs gathered from disk.
fn open_model(path: &std::path::Path) -> Result<GgufFile> {
    let g = GgufFile::open(path)?;
    let meta: std::collections::BTreeMap<String, serde_json_lite::Meta> =
        serde_json_lite::parse(g.kv_str("raon.tensor_meta").unwrap_or("{}"));
    let (mut dev, mut host) = (0usize, 0usize);
    for (n, t) in &g.tensors {
        match meta.get(n).map(|m| m.residency.as_str()) {
            Some("Device") => dev += t.nbytes(),
            _ => host += t.nbytes(),
        }
    }
    eprintln!(
        "{}: arch={} resident {:.0} MiB, disk-gather {:.0} MiB (mmap)",
        path.display(),
        g.kv_str("general.architecture").unwrap_or("?"),
        dev as f64 / 1048576.0,
        host as f64 / 1048576.0
    );
    Ok(g)
}

/// Tiny residency extractor so the client does not need a JSON dependency.
mod serde_json_lite {
    pub struct Meta {
        pub residency: String,
    }
    pub fn parse(s: &str) -> std::collections::BTreeMap<String, Meta> {
        let mut out = std::collections::BTreeMap::new();
        for part in s.split("},\"") {
            let name = part.trim_start_matches("{\"").split('"').next().unwrap_or("").to_string();
            if let Some(i) = part.find("\"residency\":\"") {
                let r = part[i + 13..].split('"').next().unwrap_or("").to_string();
                out.insert(name, Meta { residency: r });
            }
        }
        out
    }
}

fn main() -> Result<()> {
    let a = args();
    // PRD §3.5: verify the fused kernels before enabling them.
    let checks = raon_quant::kernels::verify_kernels(1e-3);
    if checks.iter().any(|c| !c.ok) {
        bail!("kernel self-test failed: {checks:?}");
    }
    let _models: Vec<GgufFile> = [a.model.as_ref(), a.vision.as_ref()]
        .into_iter()
        .flatten()
        .map(|p| open_model(p))
        .collect::<Result<_>>()?;

    let (src, screen): (Box<dyn capture::FrameSource>, ScreenGeometry) = {
        #[cfg(windows)]
        {
            if a.synthetic {
                (Box::new(capture::SyntheticSource::new(480, 270)), ScreenGeometry::new(1920, 1080))
            } else {
                let s = capture::DxgiSource::new(4)?;
                let (w, h) = capture::FrameSource::size(&s);
                (Box::new(s), ScreenGeometry::new((w * 4) as u32, (h * 4) as u32))
            }
        }
        #[cfg(not(windows))]
        {
            let _ = a.synthetic;
            (Box::new(capture::SyntheticSource::new(480, 270)), ScreenGeometry::new(1920, 1080))
        }
    };

    let event_loop = EventLoop::<overlay::UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let running = Arc::new(AtomicBool::new(true));

    let (utx, urx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if utx.send(line).is_err() {
                break;
            }
        }
    });
    let (ctx, crx) = mpsc::channel::<pipeline::AvatarCmd>();
    {
        let running = running.clone();
        std::thread::spawn(move || {
            let r = pipeline::run(src, Box::new(llm::TemplateBackend), urx, ctx, pipeline::PipelineConfig::default());
            if running.load(Ordering::SeqCst) {
                if let Err(e) = r {
                    eprintln!("pipeline stopped: {e:#}");
                }
            }
        });
    }
    {
        let proxy = proxy.clone();
        std::thread::spawn(move || {
            for cmd in crx {
                if proxy.send_event(overlay::UserEvent::Cmd(cmd)).is_err() {
                    break;
                }
            }
        });
    }
    #[cfg(windows)]
    {
        let proxy = proxy.clone();
        win32::spawn_hotkeys(move |h| {
            let _ = proxy.send_event(overlay::UserEvent::Hotkey(h));
        });
    }
    let mut app = overlay::Overlay::new(running, screen);
    event_loop.run_app(&mut app)?;
    Ok(())
}
