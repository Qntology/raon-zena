//! Raon-VisionEncoder's text tower (open_clip `TextTransformer`: 27 blocks, full bidirectional attention over the 64
//! positions with no padding mask, GELU-tanh MLP, `ln_final`, the **last** position pooled, `text_projection` with bias)
//! and its tokenizer (the SigLIP2 Gemma tokenizer of `timm/ViT-SO400M-16-SigLIP2-256`: no BOS, EOS appended, `<pad>` = 0
//! to 64, after open_clip's `canonicalize` cleaning: `_` → space, ASCII punctuation removed, lowercased).
//!
//! Zena only needs it when the word dictionary changes: it turns the vision prompts into anchors that are stored in
//! LanceDB, so at run time only the vision tower is loaded. The token table (256 000 × 1152) stays in the mmap; the
//! rows of the tokens in use are gathered from it.

use crate::tower::{attend, mlp_width, unit, Loader, Proj, WeightFormat};
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::LayerNorm;
use raon_quant::safetensors::SafeTensors;
use std::path::Path;

pub const CONTEXT: usize = 64;
pub const PAD_ID: u32 = 0;
pub const EOS_ID: u32 = 1;

/// open_clip `canonicalize_text` after `basic_clean` (without ftfy): HTML entities unescaped, `_` → space, ASCII
/// punctuation removed, lowercase, whitespace collapsed.
pub fn canonicalize(text: &str) -> String {
    let unescaped = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    let kept: String = unescaped.replace('_', " ").chars().filter(|c| !c.is_ascii_punctuation()).collect();
    kept.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

pub struct SiglipTokenizer {
    inner: tokenizers::Tokenizer,
}

impl SiglipTokenizer {
    pub fn load(path: &Path) -> Result<SiglipTokenizer> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
        Ok(SiglipTokenizer { inner })
    }

    /// `CONTEXT` ids: the canonical text's tokens, EOS, then padding (truncated to keep the EOS last).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(canonicalize(text), false).map_err(|e| anyhow!("{e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().iter().copied().take(CONTEXT - 1).collect();
        ids.push(EOS_ID);
        ids.resize(CONTEXT, PAD_ID);
        Ok(ids)
    }
}

struct TextBlock {
    ln_1: LayerNorm,
    in_proj: Proj,
    out_proj: Proj,
    ln_2: LayerNorm,
    c_fc: Proj,
    c_proj: Proj,
    heads: usize,
    hd: usize,
}

impl TextBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, d) = x.dims3()?;
        let qkv = self
            .in_proj
            .forward(&self.ln_1.forward(x)?)?
            .reshape((b, n, 3, self.heads, self.hd))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (qkv.get(0)?.contiguous()?, qkv.get(1)?.contiguous()?, qkv.get(2)?.contiguous()?);
        let o = attend(&q, &k, &v, self.hd)?.transpose(1, 2)?.reshape((b, n, d))?;
        let x = (x + self.out_proj.forward(&o)?)?;
        let m = self.c_proj.forward(&self.c_fc.forward(&self.ln_2.forward(&x)?)?.gelu()?)?;
        x + m
    }
}

pub struct TextTower {
    st: SafeTensors,
    pos: Tensor,
    blocks: Vec<TextBlock>,
    ln_final: LayerNorm,
    proj: Proj,
    device: Device,
    dtype: DType,
    hidden: usize,
}

impl TextTower {
    /// Loads from the Raon-VisionEncoder checkpoint (`model.safetensors`); the token table is gathered per use.
    pub fn load(checkpoint: &Path, device: &Device, format: WeightFormat) -> Result<TextTower> {
        let st = SafeTensors::open(&[checkpoint.to_path_buf()]).with_context(|| checkpoint.display().to_string())?;
        let (d, heads, layers, eps) = (1152usize, 16usize, 27usize, 1e-6);
        let blocks;
        let (pos, ln_final, proj);
        {
            let l = Loader {
                st: &st,
                device: device.clone(),
                format,
            };
            let m = mlp_width(4304, format);
            blocks = (0..layers)
                .map(|i| {
                    let p = format!("text.transformer.resblocks.{i}");
                    Ok(TextBlock {
                        ln_1: l.norm(&format!("{p}.ln_1"), d, eps)?,
                        in_proj: l.proj(
                            &format!("{p}.attn.in_proj_weight"),
                            Some(&format!("{p}.attn.in_proj_bias")),
                            3 * d,
                            d,
                        )?,
                        out_proj: l.proj(&format!("{p}.attn.out_proj.weight"), Some(&format!("{p}.attn.out_proj.bias")), d, d)?,
                        ln_2: l.norm(&format!("{p}.ln_2"), d, eps)?,
                        c_fc: l.proj(&format!("{p}.mlp.c_fc.weight"), Some(&format!("{p}.mlp.c_fc.bias")), m, d)?,
                        c_proj: l.proj(&format!("{p}.mlp.c_proj.weight"), Some(&format!("{p}.mlp.c_proj.bias")), d, m)?,
                        heads,
                        hd: d / heads,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            pos = l.tensor("text.positional_embedding", &[CONTEXT, d])?;
            ln_final = l.norm("text.ln_final", d, eps)?;
            proj = l.proj("text.text_projection.weight", Some("text.text_projection.bias"), d, d)?;
        }
        Ok(TextTower {
            st,
            pos,
            blocks,
            ln_final,
            proj,
            device: device.clone(),
            dtype: format.dtype(),
            hidden: d,
        })
    }

    /// Unit text embeddings of `rows` token rows (`ids.len() == rows * CONTEXT`).
    pub fn embed_ids(&self, ids: &[u32], rows: usize) -> Result<Vec<Vec<f32>>> {
        let d = self.hidden;
        let mut emb = Vec::with_capacity(ids.len() * d);
        self.st.gather_f32("text.token_embedding.weight", ids, &mut emb)?;
        let x = Tensor::from_vec(emb, (rows, CONTEXT, d), &self.device)?.to_dtype(self.dtype)?;
        let mut x = x.broadcast_add(&self.pos.unsqueeze(0)?)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let x = self.ln_final.forward(&x)?;
        let last = x.narrow(1, CONTEXT - 1, 1)?.squeeze(1)?;
        let out: Vec<Vec<f32>> = self.proj.forward(&last)?.to_dtype(DType::F32)?.to_vec2()?;
        Ok(out.into_iter().map(unit).collect())
    }

    /// Unit embeddings of texts, `batch` rows at a time.
    pub fn embed(&self, tok: &SiglipTokenizer, texts: &[String], batch: usize) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch.max(1)) {
            let mut ids = Vec::with_capacity(chunk.len() * CONTEXT);
            for t in chunk {
                ids.extend(tok.encode(t)?);
            }
            out.extend(self.embed_ids(&ids, chunk.len())?);
        }
        Ok(out)
    }

    /// logit scale (already `exp`'d) and bias of the checkpoint, for SigLIP's pair probability.
    pub fn logit_scale_bias(&self) -> Result<(f32, f32)> {
        Ok((self.st.all_f32("logit_scale")?[0].exp(), self.st.all_f32("logit_bias")?[0]))
    }
}

/// The unit mean of unit vectors (prompt ensembling, as noto-sandbox's `VisionBank` does).
pub fn mean_unit(vs: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vs.first()?;
    let mut m = vec![0f32; first.len()];
    for v in vs {
        for (a, b) in m.iter_mut().zip(v) {
            *a += b;
        }
    }
    Some(unit(m))
}

/// The prompt-ensembled anchor of every phrase: each phrase under every template, embedded, averaged.
pub fn compute_anchors(
    text: &TextTower,
    tok: &SiglipTokenizer,
    prompting: &zena_lexicon::VisionPrompting,
    phrases: &[String],
) -> Result<Vec<(String, Vec<f32>)>> {
    let forms: Vec<Vec<String>> = phrases.iter().map(|p| prompting.forms(p)).collect();
    let flat: Vec<String> = forms.iter().flatten().cloned().collect();
    let vecs = text.embed(tok, &flat, 16)?;
    let mut out = Vec::with_capacity(phrases.len());
    let mut i = 0;
    for (p, f) in phrases.iter().zip(&forms) {
        let v = mean_unit(&vecs[i..i + f.len()]).ok_or_else(|| anyhow!("no prompt for '{p}'"))?;
        i += f.len();
        out.push((p.clone(), v));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_is_open_clips() {
        assert_eq!(canonicalize("A screenshot of a Stock_Chart."), "a screenshot of a stock chart");
        assert_eq!(canonicalize("  {} on a  monitor! "), "on a monitor");
        assert_eq!(canonicalize("rock &amp; roll"), "rock roll");
    }
}
