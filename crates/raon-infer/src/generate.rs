//! Greedy generation with Raon-1-nano's chat format (non-thinking fast path) and timing stats.

use crate::head::{Head, HeadMode};
use crate::model::{Model, State};
use crate::weights::Weights;
use anyhow::{anyhow, Result};
use candle_core::Device;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub decode_tok_per_s: f64,
    pub guard_rows_refined: usize,
}

pub struct Generator {
    pub model: Model,
    pub head: Head,
    pub tok: Tokenizer,
}

/// `user` turn → ChatML prompt that pre-fills an empty think block (Lv3 fast path).
pub fn chat_prompt(system: Option<&str>, user: &str) -> String {
    let mut p = String::new();
    if let Some(s) = system {
        p.push_str(&format!("<|im_start|>system\n{s}<|im_end|>\n"));
    }
    p.push_str(&format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n"
    ));
    p
}

impl Generator {
    pub fn load_gguf(gguf: &Path, tokenizer: &Path, device: Device, exact: Option<bool>, head: HeadMode) -> Result<Self> {
        let exact = exact.unwrap_or_else(|| crate::device::exact_matmul_native(&device));
        let w = Weights::open_gguf(gguf, device, exact)?;
        Self::from_weights(w, tokenizer, head)
    }

    /// Original safetensors with pyramid quantization applied while loading (no GGUF).
    #[allow(clippy::too_many_arguments)]
    pub fn load_safetensors_quantized(
        dir: &Path,
        tokenizer: &Path,
        device: Device,
        preset: &str,
        cfg: &raon_quant::pyramid::PyramidConfig,
        imatrix: Option<&Path>,
        cache: &crate::weights::BlockCache,
        head: HeadMode,
    ) -> Result<Self> {
        Self::from_weights(
            Weights::open_safetensors_quantized(dir, device, preset, cfg, imatrix, cache)?,
            tokenizer,
            head,
        )
    }

    pub fn load_safetensors(dir: &Path, tokenizer: &Path, device: Device) -> Result<Self> {
        Self::from_weights(Weights::open_safetensors(dir, device)?, tokenizer, HeadMode::Full)
    }

    fn from_weights(w: Weights, tokenizer: &Path, head: HeadMode) -> Result<Self> {
        let head = Head::load(&w, head)?;
        let model = Model::load(w)?;
        let tok = Tokenizer::from_file(tokenizer).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Self { model, head, tok })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, false).map_err(|e| anyhow!("{e}"))
    }

    /// Run a fixed prefix (system prompt + tool definitions) once; the returned state is cloned
    /// for every request so only the new suffix is prefilled.
    pub fn prefix_state(&self, prefix: &[u32]) -> Result<State> {
        let mut st = self.model.new_state();
        if !prefix.is_empty() {
            self.model.forward(prefix, &mut st)?;
        }
        Ok(st)
    }

    /// Greedy decode until EOS / `max_new`, streaming text deltas through `on_text`.
    pub fn generate(&self, prompt: &[u32], max_new: usize, on_text: impl FnMut(&str)) -> Result<(String, GenStats)> {
        self.generate_from(self.model.new_state(), prompt, max_new, on_text)
    }

    /// Like [`Self::generate`], continuing from `st` (e.g. a cloned [`Self::prefix_state`]).
    pub fn generate_from(
        &self,
        st: State,
        prompt: &[u32],
        max_new: usize,
        on_text: impl FnMut(&str),
    ) -> Result<(String, GenStats)> {
        self.generate_from_cancellable(st, prompt, max_new, on_text, || false)
    }

    /// Like [`Self::generate_from`], polling `is_cancelled` before prefill and before each decode
    /// token so rapid re-asks or privacy locks abort immediately without burning remaining tokens.
    pub fn generate_from_cancellable(
        &self,
        mut st: State,
        prompt: &[u32],
        max_new: usize,
        mut on_text: impl FnMut(&str),
        is_cancelled: impl Fn() -> bool,
    ) -> Result<(String, GenStats)> {
        if prompt.is_empty() {
            anyhow::bail!("empty prompt");
        }
        let mut stats = GenStats {
            prompt_tokens: prompt.len(),
            ..Default::default()
        };
        if is_cancelled() {
            return Ok((String::new(), stats));
        }
        let t0 = Instant::now();
        let h = self.model.forward(prompt, &mut st)?;
        if is_cancelled() {
            stats.prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
            return Ok((String::new(), stats));
        }
        let last = h.narrow(0, prompt.len() - 1, 1)?;
        let (mut next, _, s) = self.head.argmax(&self.model.w, &last)?;
        stats.guard_rows_refined += s.rows_refined;
        stats.prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let mut out: Vec<u32> = Vec::new();
        let mut shown = 0usize;
        while out.len() < max_new && !self.model.cfg.eos.contains(&next) {
            if is_cancelled() {
                break;
            }
            out.push(next);
            let text = self.decode(&out)?;
            if text.len() > shown && text.is_char_boundary(shown) && !text.ends_with('\u{FFFD}') {
                on_text(&text[shown..]);
                shown = text.len();
            }
            if out.len() == max_new || is_cancelled() {
                break;
            }
            let h = self.model.forward(&[next], &mut st)?;
            let (n, _, s) = self.head.argmax(&self.model.w, &h)?;
            stats.guard_rows_refined += s.rows_refined;
            next = n;
        }
        stats.new_tokens = out.len();
        stats.decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        stats.decode_tok_per_s = if stats.decode_ms > 0.0 {
            out.len().saturating_sub(1) as f64 / (stats.decode_ms / 1e3)
        } else {
            0.0
        };
        Ok((self.decode(&out)?, stats))
    }
}
