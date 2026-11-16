//! End-to-end check of the Lv3 fast path with the real model: Zena's system prompt + tool
//! definitions (prefilled once and cached), a situation, greedy generation, tool-call parsing.
//!
//! cargo run --release -p raon-infer --example zena_fastpath -- GGUF TOKENIZER [device]
use raon_infer::generate::Generator;
use raon_infer::head::HeadMode;
use std::time::Instant;
use zena_core::geometry::{FlowDirection, Polar};
use zena_core::prompt::{build_fast_path_prompt, Situation};
use zena_core::tools::parse_model_output;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dev = raon_infer::device::select_device(
        a.get(3)
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(raon_infer::device::Backend::Auto),
        0,
    )?;
    let g = Generator::load_gguf(a[1].as_ref(), a[2].as_ref(), dev.clone(), None, HeadMode::Full)?;
    let full = build_fast_path_prompt(&Situation::default());
    let cut = full.find("<|im_start|>user\n").unwrap();
    let t = Instant::now();
    let prefix_ids = g.encode(&full[..cut])?;
    let prefix = g.prefix_state(&prefix_ids)?;
    println!(
        "device {} | cached prefix {} tokens in {:.1}s",
        raon_infer::device::describe(&dev),
        prefix_ids.len(),
        t.elapsed().as_secs_f32()
    );
    let cases = [
        Situation {
            event_label: Some("보스 돌진".into()),
            polar: Some(Polar { r: 0.12, theta_deg: 20.0 }),
            flow: Some(FlowDirection::Inward),
            hp_ratio: Some(0.18),
            ..Default::default()
        },
        Situation {
            user_query: Some("제나야 내 인벤토리에서 물약 어디 있어?".into()),
            ..Default::default()
        },
        Situation {
            user_query: Some("제나야 오늘 그만할게, 앱 종료해 줘".into()),
            ..Default::default()
        },
    ];
    for s in cases {
        let p = build_fast_path_prompt(&s);
        let ids = g.encode(&p[cut..])?;
        let (text, st) = g.generate_from(prefix.clone(), &ids, 120, |_| {})?;
        let parsed = parse_model_output(&text);
        println!("\n--- situation:\n{}--- completion ({} prompt tok, {} new, prefill {:.0} ms, {:.2} tok/s):\n{}\n--- parsed: {:?}  rejected: {:?}", s.render(), st.prompt_tokens, st.new_tokens, st.prefill_ms, st.decode_tok_per_s, text.trim(), parsed.calls, parsed.rejected);
    }
    Ok(())
}
