//! GGUF v3 writer (streaming, crash-safe) and zero-copy reader.
//!
//! Dims are stored ggml-style (`ne[0]` = innermost/columns), i.e. reversed with respect to the
//! row-major shapes used everywhere else in this crate.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

pub const ALIGN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    I32 = 26,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            26 => GgmlType::I32,
            30 => GgmlType::BF16,
            _ => bail!("unsupported ggml type {id}"),
        })
    }
    pub fn block_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::I32 => 1,
            GgmlType::Q8_0 => 32,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K => 256,
        }
    }
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 | GgmlType::BF16 => 2,
            GgmlType::Q8_0 => 34,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
        }
    }
    pub fn row_bytes(self, cols: usize) -> usize {
        debug_assert_eq!(cols % self.block_size(), 0);
        cols / self.block_size() * self.type_size()
    }
    pub fn bpw(self) -> f32 {
        self.type_size() as f32 * 8.0 / self.block_size() as f32
    }
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::BF16 => "BF16",
            GgmlType::I32 => "I32",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kv {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    /// Any other array type, kept opaque.
    Other,
}

impl Kv {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Kv::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Kv::U32(v) => Some(v as u64),
            Kv::U64(v) => Some(v),
            Kv::I32(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorDecl {
    pub name: String,
    pub ty: GgmlType,
    /// Row-major shape (outermost first).
    pub shape: Vec<usize>,
}

impl TensorDecl {
    pub fn nbytes(&self) -> usize {
        let cols = *self.shape.last().unwrap_or(&1);
        let rows: usize = self.shape[..self.shape.len().saturating_sub(1)].iter().product();
        rows.max(1) * self.ty.row_bytes(cols)
    }
}

fn pad(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

fn put_str(w: &mut impl Write, s: &str) -> std::io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn put_kv(w: &mut impl Write, key: &str, v: &Kv) -> std::io::Result<()> {
    put_str(w, key)?;
    match v {
        Kv::U32(x) => {
            w.write_all(&4u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Kv::I32(x) => {
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Kv::F32(x) => {
            w.write_all(&6u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Kv::Bool(x) => {
            w.write_all(&7u32.to_le_bytes())?;
            w.write_all(&[*x as u8])
        }
        Kv::Str(s) => {
            w.write_all(&8u32.to_le_bytes())?;
            put_str(w, s)
        }
        Kv::U64(x) => {
            w.write_all(&10u32.to_le_bytes())?;
            w.write_all(&x.to_le_bytes())
        }
        Kv::ArrStr(a) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&8u32.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            a.iter().try_for_each(|s| put_str(w, s))
        }
        Kv::ArrI32(a) => {
            w.write_all(&9u32.to_le_bytes())?;
            w.write_all(&5u32.to_le_bytes())?;
            w.write_all(&(a.len() as u64).to_le_bytes())?;
            a.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        Kv::Other => unreachable!("Other is read-only"),
    }
}

/// Streaming writer: declare every tensor up front, then feed their bytes in order.
pub struct GgufWriter {
    w: BufWriter<File>,
    part: PathBuf,
    dest: PathBuf,
    decls: Vec<TensorDecl>,
    cur: usize,
    written: usize,
}

impl GgufWriter {
    pub fn create(dest: &Path, kvs: &[(String, Kv)], decls: Vec<TensorDecl>) -> Result<Self> {
        let part = dest.with_extension("gguf.part");
        let mut w = BufWriter::with_capacity(8 << 20, File::create(&part).with_context(|| format!("create {}", part.display()))?);
        w.write_all(b"GGUF")?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&(decls.len() as u64).to_le_bytes())?;
        w.write_all(&((kvs.len() + 1) as u64).to_le_bytes())?;
        put_kv(&mut w, "general.alignment", &Kv::U32(ALIGN as u32))?;
        for (k, v) in kvs {
            put_kv(&mut w, k, v)?;
        }
        let mut off = 0usize;
        for d in &decls {
            put_str(&mut w, &d.name)?;
            w.write_all(&(d.shape.len().max(1) as u32).to_le_bytes())?;
            if d.shape.is_empty() {
                w.write_all(&1u64.to_le_bytes())?;
            }
            for &n in d.shape.iter().rev() {
                w.write_all(&(n as u64).to_le_bytes())?;
            }
            w.write_all(&(d.ty as u32).to_le_bytes())?;
            w.write_all(&(off as u64).to_le_bytes())?;
            off = pad(off + d.nbytes());
        }
        let pos = w.stream_position()? as usize;
        w.write_all(&vec![0u8; pad(pos) - pos])?;
        Ok(Self {
            w,
            part,
            dest: dest.to_path_buf(),
            decls,
            cur: 0,
            written: 0,
        })
    }

    pub fn decl(&self, i: usize) -> &TensorDecl {
        &self.decls[i]
    }

    /// Append bytes of the current tensor; moves on automatically when it is complete.
    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let need = self.decls.get(self.cur).context("all tensors already written")?.nbytes();
        if self.written + bytes.len() > need {
            bail!("{}: got more than {need} bytes", self.decls[self.cur].name);
        }
        self.w.write_all(bytes)?;
        self.written += bytes.len();
        if self.written == need {
            self.w.write_all(&vec![0u8; pad(need) - need])?;
            self.cur += 1;
            self.written = 0;
        }
        Ok(())
    }

    pub fn current(&self) -> Option<&TensorDecl> {
        self.decls.get(self.cur)
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        if self.cur != self.decls.len() {
            bail!("only {}/{} tensors written", self.cur, self.decls.len());
        }
        self.w.flush()?;
        self.w.get_ref().sync_all()?;
        drop(self.w);
        std::fs::rename(&self.part, &self.dest)?;
        Ok(self.dest)
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub ty: GgmlType,
    /// Row-major shape.
    pub shape: Vec<usize>,
    pub offset: usize,
}

impl GgufTensor {
    pub fn rows_cols(&self) -> (usize, usize) {
        let cols = *self.shape.last().unwrap_or(&1);
        (self.shape.iter().product::<usize>() / cols.max(1), cols)
    }
    pub fn nbytes(&self) -> usize {
        let (r, c) = self.rows_cols();
        r * self.ty.row_bytes(c)
    }
}

/// Memory-mapped GGUF file; tensor/row accessors borrow directly from the map.
pub struct GgufFile {
    map: Mmap,
    data_start: usize,
    pub kvs: BTreeMap<String, Kv>,
    pub tensors: BTreeMap<String, GgufTensor>,
    pub order: Vec<String>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("truncated GGUF");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar_size(t: u32) -> Option<usize> {
        Some(match t {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            _ => return None,
        })
    }
    fn value(&mut self, t: u32) -> Result<Kv> {
        Ok(match t {
            4 => Kv::U32(self.u32()?),
            5 => Kv::I32(self.u32()? as i32),
            6 => Kv::F32(f32::from_bits(self.u32()?)),
            7 => Kv::Bool(self.take(1)?[0] != 0),
            8 => Kv::Str(self.string()?),
            10 => Kv::U64(self.u64()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                match et {
                    8 => Kv::ArrStr((0..n).map(|_| self.string()).collect::<Result<_>>()?),
                    5 => Kv::ArrI32((0..n).map(|_| self.u32().map(|v| v as i32)).collect::<Result<_>>()?),
                    _ => {
                        let s = Self::scalar_size(et).context("nested array")?;
                        self.take(n * s)?;
                        Kv::Other
                    }
                }
            }
            _ => {
                let s = Self::scalar_size(t).context("bad kv type")?;
                self.take(s)?;
                Kv::Other
            }
        })
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only map of a file we do not modify.
        let map = unsafe { Mmap::map(&f)? };
        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != b"GGUF" {
            bail!("not a GGUF file");
        }
        let ver = c.u32()?;
        if ver != 3 {
            bail!("GGUF version {ver} unsupported");
        }
        let nt = c.u64()? as usize;
        let nk = c.u64()? as usize;
        let mut kvs = BTreeMap::new();
        for _ in 0..nk {
            let k = c.string()?;
            let t = c.u32()?;
            kvs.insert(k, c.value(t)?);
        }
        let align = kvs.get("general.alignment").and_then(Kv::as_u64).unwrap_or(ALIGN as u64) as usize;
        let mut tensors = BTreeMap::new();
        let mut order = Vec::new();
        for _ in 0..nt {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims: Vec<usize> = (0..nd).map(|_| c.u64().map(|v| v as usize)).collect::<Result<_>>()?;
            dims.reverse();
            let ty = GgmlType::from_id(c.u32()?)?;
            let offset = c.u64()? as usize;
            order.push(name.clone());
            tensors.insert(name, GgufTensor { ty, shape: dims, offset });
        }
        let data_start = c.p.div_ceil(align) * align;
        let g = GgufFile {
            map,
            data_start,
            kvs,
            tensors,
            order,
        };
        for (n, t) in &g.tensors {
            if g.data_start + t.offset + t.nbytes() > g.map.len() {
                bail!("{n}: data out of bounds");
            }
        }
        Ok(g)
    }

    pub fn tensor(&self, name: &str) -> Result<&GgufTensor> {
        self.tensors.get(name).with_context(|| format!("missing tensor {name}"))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        let s = self.data_start + t.offset;
        Ok(&self.map[s..s + t.nbytes()])
    }

    /// Zero-copy row gather: the quantized bytes of one row, straight from the page cache.
    pub fn row_bytes(&self, name: &str, row: usize) -> Result<&[u8]> {
        let t = self.tensor(name)?;
        let (rows, cols) = t.rows_cols();
        if row >= rows {
            bail!("{name}: row {row} >= {rows}");
        }
        let rb = t.ty.row_bytes(cols);
        let s = self.data_start + t.offset + row * rb;
        Ok(&self.map[s..s + rb])
    }

    pub fn i32_values(&self, name: &str) -> Result<Vec<i32>> {
        let t = self.tensor(name)?;
        if t.ty != GgmlType::I32 {
            bail!("{name} is not I32");
        }
        Ok(self
            .tensor_bytes(name)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    pub fn f32_values(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.tensor(name)?;
        let (r, c) = t.rows_cols();
        let mut out = vec![0f32; r * c];
        for i in 0..r {
            crate::kernels::dequant_row(t.ty, self.row_bytes(name, i)?, &mut out[i * c..(i + 1) * c]);
        }
        Ok(out)
    }

    pub fn kv_str(&self, key: &str) -> Option<&str> {
        self.kvs.get(key).and_then(Kv::as_str)
    }

    pub fn file_len(&self) -> usize {
        self.map.len()
    }
}
