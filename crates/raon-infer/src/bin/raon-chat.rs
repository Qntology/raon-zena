//! `raon-chat`: run Raon-1-nano on CPU / CUDA / ROCm / Vulkan / Metal.
//!
//! ```text
//! raon-chat --gguf out/raon-1-nano.pyramid.gguf --tokenizer models/KRAFTON/Raon-1-nano/tokenizer.json \
//!           --device auto --prompt "제나야, 보스 패턴 어떻게 피해?"
//! raon-chat --safetensors models/KRAFTON/Raon-1-nano ...      # exact original weights
//! raon-chat ... --raw "text" --dump out.f32                   # [T, V] logits for validation
//! ```

use anyhow::{bail, Result};
use raon_infer::device::{compiled_backends, describe, select_device, Backend};
use raon_infer::generate::{chat_prompt, Generator};
use std::io::Write;
use std::path::PathBuf;

struct Args {
    gguf: Option<PathBuf>,
    safetensors: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    device: Backend,
    prompt: Option<String>,
    raw: Option<String>,
    system: Option<String>,
    max_new: usize,
    dump: Option<PathBuf>,
    exact: Option<bool>,
    head: raon_infer::head::HeadMode,
}

fn parse() -> Result<Args> {
    let mut a = Args {
        gguf: None,
        safetensors: None,
        tokenizer: None,
        device: Backend::Auto,
        prompt: None,
        raw: None,
        system: None,
        max_new: 64,
        dump: None,
        exact: None,
        head: raon_infer::head::HeadMode::Full,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow::anyhow!("{k} needs a value"));
        match k.as_str() {
            "--gguf" => a.gguf = Some(v()?.into()),
            "--safetensors" => a.safetensors = Some(v()?.into()),
            "--tokenizer" => a.tokenizer = Some(v()?.into()),
            "--device" => a.device = v()?.parse()?,
            "--prompt" => a.prompt = Some(v()?),
            "--raw" => a.raw = Some(v()?),
            "--system" => a.system = Some(v()?),
            "--max-new" => a.max_new = v()?.parse()?,
            "--dump" => a.dump = Some(v()?.into()),
            "--head" => a.head = v()?.parse()?,
            "--exact-matmul" => a.exact = Some(true),
            "--kquant-matmul" => a.exact = Some(false),
            "-h" | "--help" => {
                println!("raon-chat (--gguf F | --safetensors DIR) [--tokenizer F] [--device auto|cpu|cuda|rocm|vulkan|metal] (--prompt TEXT | --raw TEXT) [--system TEXT] [--max-new N] [--head full|kept|guarded] [--dump F.f32] [--exact-matmul|--kquant-matmul]\nbuilt with: {}", compiled_backends().join(", "));
                std::process::exit(0);
            }
            _ => bail!("unknown argument {k}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let a = parse()?;
    let device = select_device(a.device, 0)?;
    let t = std::time::Instant::now();
    let gen = match (&a.gguf, &a.safetensors) {
        (Some(g), _) => {
            let tok = a.tokenizer.clone().unwrap_or_else(|| g.with_file_name("tokenizer.json"));
            Generator::load_gguf(g, &tok, device.clone(), a.exact, a.head)?
        }
        (None, Some(d)) => {
            let tok = a.tokenizer.clone().unwrap_or_else(|| d.join("tokenizer.json"));
            Generator::load_safetensors(d, &tok, device.clone())?
        }
        _ => bail!("pass --gguf or --safetensors"),
    };
    eprintln!(
        "[raon-chat] device={} exact_matmul={} head={:?} loaded in {:.1}s",
        describe(&device),
        gen.model.w.exact,
        gen.head.mode,
        t.elapsed().as_secs_f32()
    );
    let text = match (&a.raw, &a.prompt) {
        (Some(r), _) => r.clone(),
        (None, Some(p)) => chat_prompt(a.system.as_deref(), p),
        _ => bail!("pass --prompt or --raw"),
    };
    let ids = gen.encode(&text)?;
    if let Some(path) = &a.dump {
        // Validation: full-vocabulary logits for every position + the hidden states.
        let mut st = gen.model.new_state();
        let h = gen.model.forward(&ids, &mut st)?;
        let rows = h.to_vec2::<f32>()?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        for r in &rows {
            for x in gen.head.full_logits(&gen.model.w, r)? {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        let mut hf = std::io::BufWriter::new(std::fs::File::create(path.with_extension("hidden.f32"))?);
        for r in &rows {
            for x in r {
                hf.write_all(&x.to_le_bytes())?;
            }
        }
        std::fs::write(path.with_extension("ids.json"), serde_json::to_string(&ids)?)?;
        eprintln!("[raon-chat] dumped {} positions to {}", rows.len(), path.display());
        return Ok(());
    }
    let (_, stats) = gen.generate(&ids, a.max_new, |s| {
        print!("{s}");
        let _ = std::io::stdout().flush();
    })?;
    println!();
    eprintln!("[raon-chat] {}", serde_json::to_string(&stats)?);
    Ok(())
}
