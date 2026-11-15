//! Lv3 generation backends.
//!
//! * [`RaonNanoBackend`] — KRAFTON Raon-1-nano from the pyramid GGUF through `raon-infer`
//!   (candle + the candle-rocm-vulkan fork: CPU / CUDA / ROCm / Vulkan / Metal). The constant
//!   part of the prompt (persona + tool definitions) is prefilled once and its recurrent/KV
//!   state is cloned for every request, so only the situation lines are prefilled per call.
//! * [`TemplateBackend`] — a rule-based stand-in emitting the same `<tool_call>` wire format,
//!   used when no model is configured.
//!
//! [`LlmWorker`] runs a backend on its own thread so generation never blocks the sensor loop.

use raon_infer::device::{select_device, Backend};
use raon_infer::generate::Generator;
use raon_infer::head::HeadMode;
use raon_infer::model::State;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use zena_core::prompt::{build_fast_path_prompt, Situation};

pub trait LlmBackend: Send {
    fn name(&self) -> &'static str;
    /// Generate a completion for the fast-path prompt. `situation` is passed alongside the
    /// rendered prompt so non-neural backends can respond without parsing text.
    fn generate(&mut self, prompt: &str, situation: &Situation) -> String;
}

const USER_TURN: &str = "<|im_start|>user\n";

pub struct RaonNanoBackend {
    gen: Generator,
    prefix: State,
    pub max_new: usize,
}

impl RaonNanoBackend {
    /// `model` is either a pyramid GGUF written by `raon-quant`, or the original Hugging Face
    /// safetensors directory, quantized while loading with `preset` (`prd`, `zena`, `q8`) and an
    /// optional imatrix. With a block cache (default) the blocks are built from the originals on
    /// the first run and mapped on later runs, like logis-center's runtime towers.
    pub fn load(
        model: &Path,
        tokenizer: &Path,
        device: Backend,
        head: HeadMode,
        preset: &str,
        imatrix: Option<&Path>,
        cache: &raon_infer::weights::BlockCache,
    ) -> anyhow::Result<Self> {
        let dev = select_device(device, 0)?;
        let t = std::time::Instant::now();
        let gen = if model.is_dir() {
            let cfg = raon_quant::pyramid::PyramidConfig::preset(preset)?;
            let g = Generator::load_safetensors_quantized(model, tokenizer, dev.clone(), preset, &cfg, imatrix, cache, head)?;
            if let Some(q) = &g.model.w.quant {
                let s = q.stats.lock().unwrap();
                eprintln!(
                    "[llm] load-time pyramid '{preset}': cache {} ({} of {} tensors from it), {:.0} MiB of blocks, total {:.1}s",
                    s.cache,
                    s.cached_tensors,
                    s.tensors,
                    s.bytes as f64 / 1048576.0,
                    t.elapsed().as_secs_f32()
                );
            }
            g
        } else {
            Generator::load_gguf(model, tokenizer, dev.clone(), None, head)?
        };
        let full = build_fast_path_prompt(&Situation::default());
        let cut = full.find(USER_TURN).ok_or_else(|| anyhow::anyhow!("prompt has no user turn"))?;
        let t = std::time::Instant::now();
        let prefix_ids = gen.encode(&full[..cut])?;
        let prefix = gen.prefix_state(&prefix_ids)?;
        eprintln!(
            "[llm] Raon-1-nano on {} (head {:?}); cached {}-token system/tools prefix in {:.1}s",
            raon_infer::device::describe(&dev),
            gen.head.mode,
            prefix_ids.len(),
            t.elapsed().as_secs_f32()
        );
        Ok(Self { gen, prefix, max_new: 96 })
    }
}

impl LlmBackend for RaonNanoBackend {
    fn name(&self) -> &'static str {
        "raon-1-nano"
    }

    fn generate(&mut self, prompt: &str, _situation: &Situation) -> String {
        // Special tokens are atomic, so splitting right before the user turn tokenizes exactly
        // like the whole prompt.
        let suffix = prompt.find(USER_TURN).map(|i| &prompt[i..]).unwrap_or(prompt);
        let run = || -> anyhow::Result<String> {
            let ids = self.gen.encode(suffix)?;
            let (text, stats) = self.gen.generate_from(self.prefix.clone(), &ids, self.max_new, |_| {})?;
            eprintln!(
                "[llm] {} prompt tok, {} new tok, prefill {:.0} ms, {:.1} tok/s",
                stats.prompt_tokens, stats.new_tokens, stats.prefill_ms, stats.decode_tok_per_s
            );
            Ok(text)
        };
        run().unwrap_or_else(|e| {
            eprintln!("[llm] generation failed: {e:#}");
            String::new()
        })
    }
}

pub struct TemplateBackend;

impl LlmBackend for TemplateBackend {
    fn name(&self) -> &'static str {
        "template-stub"
    }

    fn generate(&mut self, _prompt: &str, s: &Situation) -> String {
        let (emotion, motion, line) = if let Some(q) = &s.user_query {
            if q.contains("물약") || q.contains("인벤") || q.contains("포션") {
                (
                    "FOCUSED",
                    "THINK",
                    "인벤토리는 I 키로 열 수 있어. 물약은 보통 소모품 칸에 있어!".to_string(),
                )
            } else {
                (
                    "FOCUSED",
                    "NOD",
                    "패턴 시작 모션이 보이면 옆으로 구르고, 끝나면 바로 반격해!".to_string(),
                )
            }
        } else if s.hp_ratio.is_some_and(|h| h < 0.3) {
            (
                "SURPRISED_URGENT",
                "LEAN_FORWARD_POINT",
                "체력이 위험해! 당장 회복하고 거리 벌려!".to_string(),
            )
        } else {
            (
                "SURPRISED_URGENT",
                "LEAN_FORWARD_POINT",
                format!("{} 조심해!", s.event_label.clone().unwrap_or_else(|| "적".into())),
            )
        };
        let gaze = s
            .polar
            .map(|p| format!("{{\"angle_deg\": {:.0}, \"distance_norm\": {:.2}}}", p.theta_deg, p.r))
            .unwrap_or_else(|| "{\"angle_deg\": 0, \"distance_norm\": 0.0}".into());
        format!(
            "<tool_call>\n<function=control_zena_avatar>\n<parameter=gaze_target>\n{gaze}\n</parameter>\n<parameter=emotion_state>\n{emotion}\n</parameter>\n<parameter=body_motion>\n{motion}\n</parameter>\n<parameter=speech_text>\n{line}\n</parameter>\n</function>\n</tool_call>"
        )
    }
}

/// A request for the LLM thread.
pub struct LlmJob {
    pub prompt: String,
    pub situation: Situation,
    pub level: zena_core::reaction::ReactionLevel,
}

/// Runs a backend on its own thread; at most one job in flight (new Lv3 requests are dropped
/// while busy — the reaction router has already rate-limited them).
pub struct LlmWorker {
    tx: SyncSender<LlmJob>,
    pub name: &'static str,
}

impl LlmWorker {
    pub fn spawn(mut backend: Box<dyn LlmBackend>, on_done: impl Fn(LlmJob, String) + Send + 'static) -> Self {
        let name = backend.name();
        let (tx, rx): (SyncSender<LlmJob>, Receiver<LlmJob>) = sync_channel(0);
        std::thread::Builder::new()
            .name("zena-llm".into())
            .spawn(move || {
                for job in rx {
                    let out = backend.generate(&job.prompt, &job.situation);
                    on_done(job, out);
                }
            })
            .expect("spawn llm thread");
        Self { tx, name }
    }

    /// `false` when the worker is busy (or gone).
    pub fn try_submit(&self, job: LlmJob) -> bool {
        !matches!(self.tx.try_send(job), Err(TrySendError::Full(_) | TrySendError::Disconnected(_)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zena_core::tools::{parse_model_output, ToolCall};

    #[test]
    fn stub_output_parses_with_the_real_parser() {
        let s = Situation {
            event_label: Some("보스".into()),
            polar: Some(zena_core::Polar { r: 0.1, theta_deg: 30.0 }),
            ..Default::default()
        };
        let out = TemplateBackend.generate("", &s);
        let p = parse_model_output(&out);
        assert!(matches!(p.calls.as_slice(), [ToolCall::ControlZenaAvatar(_)]), "{:?}", p.rejected);
    }

    #[test]
    fn worker_runs_jobs_off_thread() {
        let (tx, rx) = std::sync::mpsc::channel();
        let w = LlmWorker::spawn(Box::new(TemplateBackend), move |_, out| tx.send(out).unwrap());
        // sync_channel(0): a send only succeeds when the worker is waiting; retry briefly.
        let mut ok = false;
        for _ in 0..100 {
            if w.try_submit(LlmJob {
                prompt: String::new(),
                situation: Situation::default(),
                level: zena_core::reaction::ReactionLevel::Lv3Coaching,
            }) {
                ok = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(ok);
        assert!(rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap().contains("<tool_call>"));
    }
}
