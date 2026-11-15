//! Zero-copy safetensors reader over `mmap` (single file or sharded with an index).

use anyhow::{bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    file: usize,
    begin: usize,
    end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
    /// Rows/cols of the tensor viewed as a matrix (last dim = cols).
    pub fn rows_cols(&self) -> (usize, usize) {
        match self.shape.len() {
            0 => (1, 1),
            _ => {
                let cols = *self.shape.last().unwrap();
                (self.numel() / cols.max(1), cols)
            }
        }
    }
}

pub struct SafeTensors {
    maps: Vec<Mmap>,
    data_start: Vec<usize>,
    pub paths: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Open one or more `.safetensors` files (tensor names must be unique across them).
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let mut st = SafeTensors {
            maps: vec![],
            data_start: vec![],
            paths: vec![],
            tensors: BTreeMap::new(),
        };
        for p in paths {
            st.add(p)?;
        }
        Ok(st)
    }

    /// Open every shard of a Hugging Face model directory (`model.safetensors.index.json` or
    /// `model.safetensors`), plus any extra files.
    pub fn open_model_dir(dir: &Path, extra: &[PathBuf]) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<PathBuf> = if index.exists() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
            let mut f: Vec<String> = v["weight_map"]
                .as_object()
                .context("weight_map")?
                .values()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
            f.sort();
            f.dedup();
            f.into_iter().map(|n| dir.join(n)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };
        files.extend(extra.iter().cloned());
        Self::open(&files)
    }

    fn add(&mut self, path: &Path) -> Result<()> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is opened read-only and not modified while mapped.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < 8 {
            bail!("{} is too small", path.display());
        }
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if n > 100 << 20 || 8 + n > map.len() {
            bail!("{}: bad header length {n}", path.display());
        }
        let header: BTreeMap<String, Value> = serde_json::from_slice(&map[8..8 + n])?;
        let fi = self.maps.len();
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype = match v["dtype"].as_str() {
                Some("F32") => Dtype::F32,
                Some("F16") => Dtype::F16,
                Some("BF16") => Dtype::BF16,
                other => bail!("{name}: unsupported dtype {other:?} (only original F32/F16/BF16 weights are accepted)"),
            };
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let off = v["data_offsets"].as_array().context("offsets")?;
            let (begin, end) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
            let info = TensorInfo {
                dtype,
                shape,
                file: fi,
                begin,
                end,
            };
            if end - begin != info.numel() * dtype.size() || 8 + n + end > map.len() {
                bail!("{name}: byte range does not match its shape");
            }
            if self.tensors.insert(name.clone(), info).is_some() {
                bail!("duplicate tensor {name}");
            }
        }
        self.data_start.push(8 + n);
        self.maps.push(map);
        self.paths.push(path.to_path_buf());
        Ok(())
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("missing tensor {name}"))
    }

    /// Raw little-endian bytes of a tensor, borrowed from the map.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name)?;
        let s = self.data_start[t.file];
        Ok(&self.maps[t.file][s + t.begin..s + t.end])
    }

    /// Bytes of one row (zero-copy).
    pub fn row_bytes(&self, name: &str, row: usize) -> Result<&[u8]> {
        let t = self.info(name)?;
        let (rows, cols) = t.rows_cols();
        if row >= rows {
            bail!("{name}: row {row} out of range {rows}");
        }
        let w = cols * t.dtype.size();
        Ok(&self.bytes(name)?[row * w..(row + 1) * w])
    }

    /// Decode rows `[r0, r1)` to f32, appending to `out`.
    pub fn rows_f32(&self, name: &str, r0: usize, r1: usize, out: &mut Vec<f32>) -> Result<()> {
        let t = self.info(name)?;
        let (_, cols) = t.rows_cols();
        let w = cols * t.dtype.size();
        let b = &self.bytes(name)?[r0 * w..r1 * w];
        decode_into(t.dtype, b, out);
        Ok(())
    }

    /// Decode selected rows (`ids`) to f32, appending to `out`.
    pub fn gather_f32(&self, name: &str, ids: &[u32], out: &mut Vec<f32>) -> Result<()> {
        let dt = self.info(name)?.dtype;
        for &id in ids {
            decode_into(dt, self.row_bytes(name, id as usize)?, out);
        }
        Ok(())
    }

    pub fn all_f32(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.info(name)?;
        let mut v = Vec::with_capacity(t.numel());
        decode_into(t.dtype, self.bytes(name)?, &mut v);
        Ok(v)
    }
}

pub fn decode_into(dt: Dtype, b: &[u8], out: &mut Vec<f32>) {
    match dt {
        Dtype::F32 => out.extend(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()))),
        Dtype::F16 => out.extend(b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())),
        Dtype::BF16 => out.extend(b.chunks_exact(2).map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())),
    }
}

/// Write a minimal safetensors file (used by tests and examples).
pub fn write_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Result<()> {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    let mut off = 0usize;
    for (name, dt, shape, data) in tensors {
        let dts = match dt {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
        };
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dts, "shape": shape, "data_offsets": [off, off + data.len()]}),
        );
        off += data.len();
    }
    let mut h = serde_json::to_vec(&Value::Object(header))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut f = std::io::BufWriter::new(File::create(path)?);
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    for t in tensors {
        f.write_all(&t.3)?;
    }
    Ok(())
}
