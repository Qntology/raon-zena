//! Weight sources: the pyramid GGUF written by `raon-quant` (production) or the original
//! safetensors (exact BF16 → F32, used to validate the forward pass against the reference).

use anyhow::{bail, Context, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Tensor};
use raon_quant::gguf::{GgmlType, GgufFile};
use raon_quant::kernels::dequant_row;
use raon_quant::safetensors::SafeTensors;
use std::path::Path;

pub enum Source {
    Gguf(GgufFile),
    SafeTensors(SafeTensors),
}

pub struct Weights {
    pub src: Source,
    pub device: Device,
    /// Use `QMatMul::forward_exact` (no activation quantization).
    pub exact: bool,
    /// For a GGUF with a sliced vocabulary: the kept ids (ascending).
    pub kept_ids: Option<Vec<u32>>,
    pub config_json: String,
    /// Diagnostics (`RAON_MIX_GGUF` + `RAON_MIX_PATTERN=a,b`): with original safetensors weights,
    /// take the tensors whose name contains one of the patterns from a pyramid GGUF instead
    /// (dequantized) — attributes quantization error to tensor groups.
    mix: Option<(GgufFile, Vec<String>)>,
}

/// Importance-matrix collection (PRD §3.3): Σx² per input column of every linear layer, over a
/// calibration corpus run through the *original* weights. Written in llama.cpp's legacy
/// `imatrix.dat` format keyed by the original tensor names, which `raon-quant --imatrix` reads.
pub mod imatrix {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static ACC: Mutex<BTreeMap<String, (Vec<f64>, u64)>> = Mutex::new(BTreeMap::new());

    pub fn record(name: &str, x: &candle_core::Tensor) -> candle_core::Result<()> {
        if !ON.load(Ordering::Relaxed) {
            return Ok(());
        }
        let cols = x.dim(candle_core::D::Minus1)?;
        let v = x.flatten_all()?.to_vec1::<f32>()?;
        let mut acc = ACC.lock().unwrap();
        let e = acc.entry(name.to_string()).or_insert_with(|| (vec![0f64; cols], 0));
        for row in v.chunks_exact(cols) {
            for (a, &b) in e.0.iter_mut().zip(row) {
                *a += (b as f64) * (b as f64);
            }
            e.1 += 1;
        }
        Ok(())
    }

    /// Write `Σx²/rows` per column (ncall = 1).
    pub fn write(path: &std::path::Path) -> std::io::Result<usize> {
        let acc = ACC.lock().unwrap();
        let mut b = Vec::new();
        b.extend((acc.len() as i32).to_le_bytes());
        for (name, (sum, n)) in acc.iter() {
            b.extend((name.len() as i32).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(1i32.to_le_bytes());
            b.extend((sum.len() as i32).to_le_bytes());
            for s in sum {
                b.extend(((s / (*n).max(1) as f64) as f32).to_le_bytes());
            }
        }
        std::fs::write(path, b)?;
        Ok(acc.len())
    }
}

/// A linear layer `y = x Wᵀ` whose weight stays in its stored (quantized) form on the device.
pub struct Linear {
    inner: QMatMul,
    exact: bool,
    pub name: String,
    pub out_features: usize,
    pub in_features: usize,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        imatrix::record(&self.name, &x)?;
        if self.exact {
            self.inner.forward_exact(&x)
        } else {
            candle_core::Module::forward(&self.inner, &x)
        }
    }
}

fn ggml_dtype(t: GgmlType) -> Result<GgmlDType> {
    Ok(match t {
        GgmlType::F32 => GgmlDType::F32,
        GgmlType::F16 => GgmlDType::F16,
        GgmlType::BF16 => GgmlDType::BF16,
        GgmlType::Q8_0 => GgmlDType::Q8_0,
        GgmlType::Q4K => GgmlDType::Q4K,
        GgmlType::Q5K => GgmlDType::Q5K,
        GgmlType::Q6K => GgmlDType::Q6K,
        GgmlType::I32 => bail!("I32 tensors are not weights"),
    })
}

impl Weights {
    pub fn open_gguf(path: &Path, device: Device, exact: bool) -> Result<Self> {
        let g = GgufFile::open(path)?;
        if g.kv_str("general.architecture") != Some("raon1_nano") {
            bail!("{} is not a raon1_nano GGUF", path.display());
        }
        let config_json = g
            .kv_str("raon.source_config")
            .context("GGUF has no raon.source_config")?
            .to_string();
        let kept_ids = if g.tensors.contains_key("raon.vocab.kept_ids") {
            Some(g.i32_values("raon.vocab.kept_ids")?.into_iter().map(|x| x as u32).collect())
        } else {
            None
        };
        Ok(Self {
            src: Source::Gguf(g),
            device,
            exact,
            kept_ids,
            config_json,
            mix: None,
        })
    }

    pub fn open_safetensors(dir: &Path, device: Device) -> Result<Self> {
        let st = SafeTensors::open_model_dir(dir, &[])?;
        let config_json = std::fs::read_to_string(dir.join("config.json"))?;
        let mix = match (std::env::var("RAON_MIX_GGUF"), std::env::var("RAON_MIX_PATTERN")) {
            (Ok(g), Ok(p)) => {
                let pats: Vec<String> = p.split(',').filter(|x| !x.is_empty()).map(String::from).collect();
                eprintln!("[raon-infer] mixing GGUF tensors matching {pats:?} from {g}");
                Some((GgufFile::open(Path::new(&g))?, pats))
            }
            _ => None,
        };
        Ok(Self {
            src: Source::SafeTensors(st),
            device,
            exact: true,
            kept_ids: None,
            config_json,
            mix,
        })
    }

    fn mixed(&self, name: &str) -> Option<&GgufFile> {
        let (g, pats) = self.mix.as_ref()?;
        pats.iter().any(|p| name.contains(p.as_str())).then_some(g)
    }

    pub fn gguf(&self) -> Option<&GgufFile> {
        match &self.src {
            Source::Gguf(g) => Some(g),
            _ => None,
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match &self.src {
            Source::Gguf(g) => g.tensors.contains_key(name),
            Source::SafeTensors(s) => s.tensors.contains_key(name),
        }
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(match &self.src {
            Source::Gguf(g) => g.tensor(name)?.shape.clone(),
            Source::SafeTensors(s) => s.info(name)?.shape.clone(),
        })
    }

    /// Any tensor, decoded to f32 on the host.
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>> {
        if let Some(g) = self.mixed(name).filter(|g| g.tensors.contains_key(name)) {
            let t = g.tensor(name)?;
            let mut v = g.f32_values(name)?;
            // Undo zero padding (vision-style padded shapes do not occur in the nano GGUF).
            v.truncate(t.shape.iter().product());
            return Ok(v);
        }
        match &self.src {
            Source::Gguf(g) => g.f32_values(name),
            Source::SafeTensors(s) => s.all_f32(name),
        }
    }

    /// Any tensor as an f32 device tensor with `shape` (defaults to the stored shape).
    pub fn tensor(&self, name: &str, shape: Option<&[usize]>) -> Result<Tensor> {
        let v = self.host_f32(name).with_context(|| format!("load {name}"))?;
        let shape = match shape {
            Some(s) => s.to_vec(),
            None => {
                let s = self.shape(name)?;
                if s.is_empty() {
                    vec![1]
                } else {
                    s
                }
            }
        };
        Ok(Tensor::from_vec(v, shape, &self.device)?)
    }

    pub fn scalar(&self, name: &str) -> Result<f32> {
        Ok(self.host_f32(name)?[0])
    }

    pub fn linear(&self, name: &str) -> Result<Linear> {
        let shape = self.shape(name)?;
        if shape.len() != 2 {
            bail!("{name}: expected a matrix, got {shape:?}");
        }
        let (out_features, in_features) = (shape[0], shape[1]);
        let inner = match &self.src {
            Source::Gguf(g) => {
                let t = g.tensor(name)?;
                match t.ty {
                    GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => QMatMul::Tensor(self.tensor(name, None)?),
                    ty => {
                        let q = qtensor_from_ggml(ggml_dtype(ty)?, g.tensor_bytes(name)?, shape.clone(), &self.device)?;
                        QMatMul::from_qtensor(q)?
                    }
                }
            }
            Source::SafeTensors(_) => QMatMul::Tensor(self.tensor(name, None)?),
        };
        // Dense weights run a plain f32 matmul; `exact` only matters for quantized ones.
        Ok(Linear {
            inner,
            exact: self.exact,
            name: name.to_string(),
            out_features,
            in_features,
        })
    }

    /// Rows of a `[V, cols]` table as an f32 `[ids.len(), cols]` device tensor, gathered from the
    /// mmap (never resident as a whole).
    pub fn rows(&self, name: &str, ids: &[u32]) -> Result<Tensor> {
        if let Some(g) = self.mixed(name) {
            if name == "model.per_layer_inputs.table" {
                return self.gguf_ple_rows(g, ids);
            }
            let (ty, cols) = (g.tensor(name)?.ty, g.tensor(name)?.shape[1]);
            let mut out = vec![0f32; ids.len() * cols];
            for (i, &id) in ids.iter().enumerate() {
                dequant_row(ty, g.row_bytes(name, id as usize)?, &mut out[i * cols..(i + 1) * cols]);
            }
            return Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?);
        }
        let cols = *self.shape(name)?.last().unwrap();
        let mut out = vec![0f32; ids.len() * cols];
        match &self.src {
            Source::Gguf(g) => {
                let ty = g.tensor(name)?.ty;
                for (i, &id) in ids.iter().enumerate() {
                    dequant_row(ty, g.row_bytes(name, id as usize)?, &mut out[i * cols..(i + 1) * cols]);
                }
            }
            Source::SafeTensors(s) => {
                out.clear();
                s.gather_f32(name, ids, &mut out)?;
            }
        }
        Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?)
    }

    fn gguf_ple_rows(&self, g: &GgufFile, ids: &[u32]) -> Result<Tensor> {
        let kept: Vec<u32> = g.i32_values("raon.vocab.kept_ids")?.into_iter().map(|x| x as u32).collect();
        self.split_ple_rows(g, &kept, ids)
    }

    /// Per-layer-embedding table rows; in a sliced GGUF the table is split into `.kept` / `.cold`.
    pub fn ple_rows(&self, ids: &[u32]) -> Result<Tensor> {
        const T: &str = "model.per_layer_inputs.table";
        match (&self.src, &self.kept_ids) {
            (Source::Gguf(g), Some(kept)) if g.tensors.contains_key("model.per_layer_inputs.table.kept") => {
                self.split_ple_rows(g, kept, ids)
            }
            _ => self.rows(T, ids),
        }
    }

    fn split_ple_rows(&self, g: &GgufFile, kept: &[u32], ids: &[u32]) -> Result<Tensor> {
        {
            {
                let cols = g.tensor("model.per_layer_inputs.table.kept")?.shape[1];
                let mut out = vec![0f32; ids.len() * cols];
                for (i, &id) in ids.iter().enumerate() {
                    let (name, row) = match kept.binary_search(&id) {
                        Ok(r) => ("model.per_layer_inputs.table.kept", r),
                        // Cold rows are the complement in ascending order.
                        Err(rank) => ("model.per_layer_inputs.table.cold", id as usize - rank),
                    };
                    let ty = g.tensor(name)?.ty;
                    dequant_row(ty, g.row_bytes(name, row)?, &mut out[i * cols..(i + 1) * cols]);
                }
                Ok(Tensor::from_vec(out, (ids.len(), cols), &self.device)?)
            }
        }
    }
}
