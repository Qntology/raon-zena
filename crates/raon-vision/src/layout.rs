//! Two checkpoint layouts of the same SigLIP2 So400m/16 NaFlex network, read through one set of names.
//!
//! * [`Layout::RaonVe`]: KRAFTON `Raon-VisionEncoder` (timm / open_clip names: `visual.trunk.*`, `text.*`, fused
//!   `attn.qkv`, MAP head `attn_pool`, the patch projection as a conv `[1152, 3, 16, 16]`).
//! * [`Layout::Siglip2Hf`]: `google/siglip2-so400m-patch16-naflex` (Hugging Face transformers names:
//!   `vision_model.*`, `text_model.*`, separate `self_attn.{q,k,v}_proj`, MAP head `head.*` with a packed
//!   `in_proj`, the patch projection as a linear map `[1152, 768]` over `(p_h, p_w, c)` pixels).
//!
//! The towers ask for Raon-VE names; [`Weights`] answers them from either file. For the Hugging Face file it
//! concatenates the q / k / v rows into the fused `qkv` / `in_proj` the forward passes use, splits the head's
//! packed `in_proj` into the latent query and the key-value projection, and reorders the patch projection's
//! columns to the channel-major `(c, p_h, p_w)` patches `zena_sensor::naflex` produces. Everything downstream —
//! Q8_0 padding, the sparse passes, the MAP head's per-patch path — is the same code for both.
//!
//! What still differs between the two checkpoints is outside the weights: pixel normalisation and resampling
//! ([`crate::preprocess_for`]) and the position-table resize ([`Layout::pos_antialias`]).

use anyhow::{bail, Result};
use raon_quant::safetensors::SafeTensors;

pub const HIDDEN: usize = 1152;
pub const PATCH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RaonVe,
    Siglip2Hf,
}

/// A name the Hugging Face checkpoint always has (and Raon-VE never).
const HF_MARKER: &str = "vision_model.embeddings.patch_embedding.weight";

impl Layout {
    pub fn detect(st: &SafeTensors) -> Layout {
        if st.tensors.contains_key(HF_MARKER) {
            Layout::Siglip2Hf
        } else {
            Layout::RaonVe
        }
    }

    /// The family tag of fingerprints and reports.
    pub fn tag(self) -> &'static str {
        match self {
            Layout::RaonVe => "raon-ve",
            Layout::Siglip2Hf => "siglip2-hf",
        }
    }

    /// Whether the 16 × 16 position table is shrunk with an antialiased bilinear filter. transformers'
    /// `Siglip2VisionEmbeddings.resize_positional_embeddings` passes `antialias=True`; timm's NaFlex 1-D mode
    /// (Raon-VE) does not. The two only differ on a grid side below 16.
    pub fn pos_antialias(self) -> bool {
        self == Layout::Siglip2Hf
    }
}

/// Where a Raon-VE name comes from in a Hugging Face file.
#[derive(Debug, PartialEq)]
enum Source {
    Same(String),
    /// Rows of these tensors, one after the other (q, k, v → qkv).
    Concat(Vec<String>),
    /// Rows `[start, start + len)` of a tensor (the head's packed in_proj).
    Rows(String, usize, usize),
    /// The patch projection with its columns reordered from `(p_h, p_w, c)` to `(c, p_h, p_w)`.
    PatchColumns(String),
}

fn layer_tail(rest: &str) -> Option<(usize, &str)> {
    let (i, tail) = rest.split_once('.')?;
    Some((i.parse().ok()?, tail))
}

fn hf_source(name: &str) -> Option<Source> {
    use Source::*;
    let same = |s: String| Some(Same(s));
    let qkv = |p: &str, what: &str| Some(Concat(["q", "k", "v"].iter().map(|x| format!("{p}.self_attn.{x}_proj.{what}")).collect()));
    if let Some(rest) = name.strip_prefix("visual.trunk.blocks.") {
        let (i, tail) = layer_tail(rest)?;
        let p = format!("vision_model.encoder.layers.{i}");
        return match tail {
            "norm1.weight" | "norm1.bias" => same(format!("{p}.layer_norm1.{}", &tail[6..])),
            "norm2.weight" | "norm2.bias" => same(format!("{p}.layer_norm2.{}", &tail[6..])),
            "attn.qkv.weight" => qkv(&p, "weight"),
            "attn.qkv.bias" => qkv(&p, "bias"),
            "attn.proj.weight" | "attn.proj.bias" => same(format!("{p}.self_attn.out_proj.{}", &tail[10..])),
            t if t.starts_with("mlp.fc") => same(format!("{p}.{t}")),
            _ => None,
        };
    }
    if let Some(tail) = name.strip_prefix("visual.trunk.attn_pool.") {
        let h = "vision_model.head";
        let d = HIDDEN;
        return match tail {
            "latent" => same(format!("{h}.probe")),
            "q.weight" => Some(Rows(format!("{h}.attention.in_proj_weight"), 0, d)),
            "q.bias" => Some(Rows(format!("{h}.attention.in_proj_bias"), 0, d)),
            "kv.weight" => Some(Rows(format!("{h}.attention.in_proj_weight"), d, 2 * d)),
            "kv.bias" => Some(Rows(format!("{h}.attention.in_proj_bias"), d, 2 * d)),
            "proj.weight" | "proj.bias" => same(format!("{h}.attention.out_proj.{}", &tail[5..])),
            "norm.weight" | "norm.bias" => same(format!("{h}.layernorm.{}", &tail[5..])),
            t if t.starts_with("mlp.fc") => same(format!("{h}.{t}")),
            _ => None,
        };
    }
    if let Some(rest) = name.strip_prefix("text.transformer.resblocks.") {
        let (i, tail) = layer_tail(rest)?;
        let p = format!("text_model.encoder.layers.{i}");
        return match tail {
            "ln_1.weight" | "ln_1.bias" => same(format!("{p}.layer_norm1.{}", &tail[5..])),
            "ln_2.weight" | "ln_2.bias" => same(format!("{p}.layer_norm2.{}", &tail[5..])),
            "attn.in_proj_weight" => qkv(&p, "weight"),
            "attn.in_proj_bias" => qkv(&p, "bias"),
            "attn.out_proj.weight" | "attn.out_proj.bias" => same(format!("{p}.self_attn.out_proj.{}", &tail[14..])),
            "mlp.c_fc.weight" | "mlp.c_fc.bias" => same(format!("{p}.mlp.fc1.{}", &tail[9..])),
            "mlp.c_proj.weight" | "mlp.c_proj.bias" => same(format!("{p}.mlp.fc2.{}", &tail[11..])),
            _ => None,
        };
    }
    match name {
        "visual.trunk.patch_embed.proj.weight" => Some(PatchColumns("vision_model.embeddings.patch_embedding.weight".into())),
        "visual.trunk.patch_embed.proj.bias" => same("vision_model.embeddings.patch_embedding.bias".into()),
        "visual.trunk.pos_embed" => same("vision_model.embeddings.position_embedding.weight".into()),
        "visual.trunk.norm.weight" => same("vision_model.post_layernorm.weight".into()),
        "visual.trunk.norm.bias" => same("vision_model.post_layernorm.bias".into()),
        "text.token_embedding.weight" => same("text_model.embeddings.token_embedding.weight".into()),
        "text.positional_embedding" => same("text_model.embeddings.position_embedding.weight".into()),
        "text.ln_final.weight" => same("text_model.final_layer_norm.weight".into()),
        "text.ln_final.bias" => same("text_model.final_layer_norm.bias".into()),
        "text.text_projection.weight" => same("text_model.head.weight".into()),
        "text.text_projection.bias" => same("text_model.head.bias".into()),
        "logit_scale" | "logit_bias" => same(name.into()),
        _ => None,
    }
}

/// `W[o, (y·P + x)·3 + c]` → `W[o, c·P² + y·P + x]` for every output row.
pub fn patch_columns_to_channel_major(w: &[f32], rows: usize) -> Vec<f32> {
    let n = 3 * PATCH * PATCH;
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let (src, dst) = (&w[r * n..(r + 1) * n], &mut out[r * n..(r + 1) * n]);
        for y in 0..PATCH {
            for x in 0..PATCH {
                for c in 0..3 {
                    dst[c * PATCH * PATCH + y * PATCH + x] = src[(y * PATCH + x) * 3 + c];
                }
            }
        }
    }
    out
}

/// A checkpoint read under Raon-VE names.
pub struct Weights<'a> {
    pub st: &'a SafeTensors,
    pub layout: Layout,
}

impl<'a> Weights<'a> {
    pub fn new(st: &'a SafeTensors) -> Weights<'a> {
        Weights {
            st,
            layout: Layout::detect(st),
        }
    }

    fn source(&self, name: &str) -> Result<Source> {
        match self.layout {
            Layout::RaonVe => Ok(Source::Same(name.to_string())),
            Layout::Siglip2Hf => match hf_source(name) {
                Some(s) => Ok(s),
                None => bail!("{name} has no counterpart in a Hugging Face SigLIP2 checkpoint"),
            },
        }
    }

    /// Shape as the towers expect it (rows of a concatenation add up, a row slice has its own row count).
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(match self.source(name)? {
            Source::Same(n) | Source::PatchColumns(n) => self.st.info(&n)?.shape.clone(),
            Source::Concat(parts) => {
                let mut shape = self.st.info(&parts[0])?.shape.clone();
                shape[0] = parts.iter().map(|p| self.st.info(p).map(|i| i.shape[0])).sum::<Result<usize>>()?;
                shape
            }
            Source::Rows(n, _, len) => {
                let mut shape = self.st.info(&n)?.shape.clone();
                shape[0] = len;
                shape
            }
        })
    }

    pub fn all_f32(&self, name: &str) -> Result<Vec<f32>> {
        match self.source(name)? {
            Source::Same(n) => self.st.all_f32(&n),
            Source::Concat(parts) => {
                let mut out = Vec::new();
                for p in &parts {
                    out.extend(self.st.all_f32(p)?);
                }
                Ok(out)
            }
            Source::Rows(n, start, len) => {
                let info = self.st.info(&n)?;
                let mut out = Vec::new();
                if info.shape.len() == 1 {
                    let all = self.st.all_f32(&n)?;
                    if start + len > all.len() {
                        bail!("{n}: rows {start}..{} of {}", start + len, all.len());
                    }
                    out.extend_from_slice(&all[start..start + len]);
                } else {
                    self.st.rows_f32(&n, start, start + len, &mut out)?;
                }
                Ok(out)
            }
            Source::PatchColumns(n) => {
                let info = self.st.info(&n)?;
                if info.shape.len() != 2 || info.shape[1] != 3 * PATCH * PATCH {
                    bail!("{n}: expected [out, {}], got {:?}", 3 * PATCH * PATCH, info.shape);
                }
                Ok(patch_columns_to_channel_major(&self.st.all_f32(&n)?, info.shape[0]))
            }
        }
    }

    /// Selected rows of a table (the text token embedding).
    pub fn gather_f32(&self, name: &str, ids: &[u32], out: &mut Vec<f32>) -> Result<()> {
        match self.source(name)? {
            Source::Same(n) => self.st.gather_f32(&n, ids, out),
            other => bail!("{name}: cannot gather rows from {other:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tower_name_has_a_hugging_face_source() {
        let mut names: Vec<String> = vec![
            "visual.trunk.patch_embed.proj.weight".into(),
            "visual.trunk.patch_embed.proj.bias".into(),
            "visual.trunk.pos_embed".into(),
            "visual.trunk.norm.weight".into(),
            "text.token_embedding.weight".into(),
            "text.positional_embedding".into(),
            "text.ln_final.bias".into(),
            "text.text_projection.weight".into(),
            "logit_scale".into(),
        ];
        for t in ["norm1.weight", "norm2.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.bias"] {
            names.push(format!("visual.trunk.blocks.26.{t}"));
        }
        for t in ["latent", "q.weight", "q.bias", "kv.weight", "kv.bias", "proj.bias", "norm.weight", "mlp.fc2.weight"] {
            names.push(format!("visual.trunk.attn_pool.{t}"));
        }
        for t in ["ln_1.weight", "ln_2.bias", "attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.bias", "mlp.c_fc.weight", "mlp.c_proj.bias"] {
            names.push(format!("text.transformer.resblocks.0.{t}"));
        }
        for n in &names {
            assert!(hf_source(n).is_some(), "{n}");
        }
        assert_eq!(
            hf_source("visual.trunk.blocks.3.attn.qkv.bias"),
            Some(Source::Concat(vec![
                "vision_model.encoder.layers.3.self_attn.q_proj.bias".into(),
                "vision_model.encoder.layers.3.self_attn.k_proj.bias".into(),
                "vision_model.encoder.layers.3.self_attn.v_proj.bias".into(),
            ]))
        );
        assert_eq!(
            hf_source("visual.trunk.attn_pool.kv.weight"),
            Some(Source::Rows("vision_model.head.attention.in_proj_weight".into(), HIDDEN, 2 * HIDDEN))
        );
        assert_eq!(
            hf_source("text.transformer.resblocks.5.mlp.c_proj.weight"),
            Some(Source::Same("text_model.encoder.layers.5.mlp.fc2.weight".into()))
        );
        assert_eq!(
            hf_source("visual.trunk.blocks.0.attn.proj.bias"),
            Some(Source::Same("vision_model.encoder.layers.0.self_attn.out_proj.bias".into()))
        );
        assert!(hf_source("visual.trunk.blocks.x.norm1.weight").is_none());
    }

    #[test]
    fn patch_columns_follow_the_pixels() {
        // Row 0 numbers each (y, x, c) input column by its HF position; after the reorder, column c·256 + y·16 + x
        // must hold HF column (y·16 + x)·3 + c.
        let n = 3 * PATCH * PATCH;
        let hf: Vec<f32> = (0..2 * n).map(|i| i as f32).collect();
        let raon = patch_columns_to_channel_major(&hf, 2);
        for (y, x, c) in [(0, 0, 0), (0, 1, 2), (5, 7, 1), (15, 15, 2)] {
            assert_eq!(raon[c * 256 + y * 16 + x], ((y * 16 + x) * 3 + c) as f32);
            assert_eq!(raon[n + c * 256 + y * 16 + x], (n + (y * 16 + x) * 3 + c) as f32);
        }
        // A patch flattened both ways gives the same dot product with the two weight layouts.
        let pixel = |y: usize, x: usize, c: usize| ((y * 31 + x * 7 + c * 3) % 11) as f32 - 5.0;
        let mut hf_patch = vec![0f32; n];
        let mut raon_patch = vec![0f32; n];
        for y in 0..PATCH {
            for x in 0..PATCH {
                for c in 0..3 {
                    hf_patch[(y * PATCH + x) * 3 + c] = pixel(y, x, c);
                    raon_patch[c * PATCH * PATCH + y * PATCH + x] = pixel(y, x, c);
                }
            }
        }
        let dot = |w: &[f32], p: &[f32]| w.iter().zip(p).map(|(a, b)| a * b).sum::<f32>();
        assert_eq!(dot(&hf[n..], &hf_patch), dot(&raon[n..], &raon_patch));
    }
}
