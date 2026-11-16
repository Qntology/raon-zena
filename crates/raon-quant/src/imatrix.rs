//! Reader for llama.cpp's legacy binary importance matrix (`imatrix.dat`):
//! `i32 n_entries`, then per entry `i32 name_len, name, i32 ncall, i32 nval, f32[nval]`
//! (Σ activation² per input column). Values are divided by `ncall` → mean squared activation σ².

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::Path;

pub fn read_imatrix(path: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let b = std::fs::read(path)?;
    let mut p = 0usize;
    let i32_at = |p: &mut usize| -> Result<i32> {
        if *p + 4 > b.len() {
            bail!("truncated imatrix");
        }
        let v = i32::from_le_bytes(b[*p..*p + 4].try_into()?);
        *p += 4;
        Ok(v)
    };
    let n = i32_at(&mut p)?;
    let mut out = HashMap::new();
    for _ in 0..n {
        let len = i32_at(&mut p)? as usize;
        let name = String::from_utf8_lossy(&b[p..p + len]).into_owned();
        p += len;
        let ncall = i32_at(&mut p)?.max(1) as f32;
        let nval = i32_at(&mut p)? as usize;
        if p + 4 * nval > b.len() {
            bail!("truncated imatrix entry {name}");
        }
        let v = b[p..p + 4 * nval]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()) / ncall)
            .collect();
        p += 4 * nval;
        out.insert(name, v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_legacy_format() {
        let mut b = Vec::new();
        b.extend(1i32.to_le_bytes());
        b.extend(3i32.to_le_bytes());
        b.extend(b"abc");
        b.extend(2i32.to_le_bytes());
        b.extend(2i32.to_le_bytes());
        b.extend(4f32.to_le_bytes());
        b.extend(8f32.to_le_bytes());
        let p = std::env::temp_dir().join(format!("imat-{}.dat", std::process::id()));
        std::fs::write(&p, b).unwrap();
        let m = super::read_imatrix(&p).unwrap();
        assert_eq!(m["abc"], vec![2.0, 4.0]);
        std::fs::remove_file(p).ok();
    }
}
