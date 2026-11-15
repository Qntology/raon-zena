//! Raon-1-nano forward pass on candle, mirroring the released `modeling_raon1_nano.py`:
//! 24 decoder layers on a 4-stream Raon-mHC residual — 18 RDA (gated delta rule with per-key-
//! channel decay and a write gate) + 6 NoPE gated MLA — a SwiGLU MLP per layer, the per-layer
//! embedding (PLE) added after each MLP, and a final RMSNorm over the stream mean.
//!
//! Activations are f32. Every GEMM goes through [`crate::weights::Linear`], i.e. the quantized
//! blocks resident on the device (CUDA / ROCm / Vulkan / Metal / CPU).

use crate::weights::{Linear, Weights};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops::{rms_norm, sigmoid, softmax_last_dim};
use rayon::prelude::*;

/// Opt-in phase profiler (`RAON_PROFILE=1`): accumulated milliseconds per phase.
pub mod profile {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static ACC: Mutex<[f64; 6]> = Mutex::new([0.0; 6]);
    pub const NAMES: [&str; 6] = ["mhc_route", "rda", "mla", "mlp", "ple_gate", "mix"];
    pub fn enabled() -> bool {
        ON.load(Ordering::Relaxed)
    }
    pub fn add(i: usize, t: Instant) {
        if enabled() {
            ACC.lock().unwrap()[i] += t.elapsed().as_secs_f64() * 1e3;
        }
    }
    pub fn take() -> Vec<(&'static str, f64)> {
        let mut a = ACC.lock().unwrap();
        let out = NAMES.iter().zip(a.iter()).map(|(n, v)| (*n, *v)).collect();
        *a = [0.0; 6];
        out
    }
}
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub heads: usize,
    pub inter: usize,
    pub eps: f64,
    pub mla_eps: f64,
    pub sel_eps: f64,
    pub post_factor: f64,
    pub streams: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_nope: usize,
    pub qk_pe: usize,
    pub v_dim: usize,
    pub rda_heads: usize,
    pub rda_head_dim: usize,
    pub conv_k: usize,
    pub gate_lower_bound: Option<f64>,
    pub ple_dim: usize,
    pub rda_layer: Vec<bool>,
    pub eos: Vec<u32>,
}

impl Config {
    pub fn from_json(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)?;
        let u = |k: &str, d: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(d);
        let f = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        let layers = u("num_hidden_layers", 24);
        let rda = &v["rda_config"];
        let rda_layers: Vec<usize> = rda["rda_layers"]
            .as_array()
            .context("rda_config.rda_layers")?
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let head_dim = u("head_dim", 128);
        Ok(Config {
            hidden: u("hidden_size", 1536),
            layers,
            vocab: u("vocab_size", 259840),
            heads: u("num_attention_heads", 32),
            inter: u("intermediate_size", 6144),
            eps: f("rms_norm_eps", 1e-6),
            mla_eps: f("mla_latent_norm_eps", 1e-6),
            sel_eps: v["raon_mhc_selector_eps"].as_f64().unwrap_or(1e-5),
            post_factor: v["raon_mhc_post_factor"].as_f64().unwrap_or(2.0),
            streams: v["raon_mhc_streams"].as_u64().map(|x| x as usize).unwrap_or(4),
            q_lora: u("q_lora_rank", 512),
            kv_lora: u("kv_lora_rank", 512),
            qk_nope: u("qk_nope_head_dim", 128),
            qk_pe: u("qk_rope_head_dim", 64),
            v_dim: u("v_head_dim", 128),
            rda_heads: rda["num_heads"].as_u64().map(|x| x as usize).unwrap_or(16),
            rda_head_dim: rda["head_dim"].as_u64().map(|x| x as usize).unwrap_or(head_dim),
            conv_k: rda["conv_kernel_size"].as_u64().map(|x| x as usize).unwrap_or(4),
            gate_lower_bound: if rda.get("gate_lower_bound").is_some_and(|g| g.is_null()) {
                None
            } else {
                Some(rda["gate_lower_bound"].as_f64().unwrap_or(-5.0))
            },
            ple_dim: u("per_layer_embedding_dim", 256),
            rda_layer: (0..layers).map(|i| rda_layers.contains(&(i + 1))).collect(),
            eos: vec![259202, 259200],
        })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    rms_norm(&x.contiguous()?, w, eps as f32)
}

// ---------------------------------------------------------------------------------------------
// Raon-mHC
// ---------------------------------------------------------------------------------------------

/// Helmert basis `Q ∈ R^{n×(n-1)}`, `QᵀQ = I`, `Qᵀ1 = 0`.
fn helmert(n: usize) -> Vec<f32> {
    let mut q = vec![0f32; n * (n - 1)];
    for k in 1..n {
        let s = ((k * (k + 1)) as f64).sqrt();
        for i in 0..k {
            q[i * (n - 1) + k - 1] = (1.0 / s) as f32;
        }
        q[k * (n - 1) + k - 1] = (-(k as f64) / s) as f32;
    }
    q
}

struct MhcConsts {
    q: Tensor,   // [n, n-1]
    b: Tensor,   // [n, n]  (I + 11ᵀ/n)/2
    eye: Tensor, // [n, n]
}

struct Mhc {
    sel: Tensor, // [2n + (n-1)², n·C]
    alpha_pre: f64,
    alpha_post: f64,
    alpha_z: f64,
    b_pre: Tensor,
    b_post: Tensor,
    b_z: Tensor,
    gamma: f64,
}

impl Mhc {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        let sel = Tensor::cat(
            &[
                w.tensor(&format!("{p}.phi_pre"), None)?,
                w.tensor(&format!("{p}.phi_post"), None)?,
                w.tensor(&format!("{p}.res.phi_z"), None)?,
            ],
            0,
        )?;
        let eta = w.scalar(&format!("{p}.res.eta"))? as f64;
        Ok(Self {
            sel: sel.t()?.contiguous()?,
            alpha_pre: w.scalar(&format!("{p}.alpha_pre"))? as f64,
            alpha_post: w.scalar(&format!("{p}.alpha_post"))? as f64,
            alpha_z: w.scalar(&format!("{p}.res.alpha_z"))? as f64,
            b_pre: w.tensor(&format!("{p}.b_pre"), None)?,
            b_post: w.tensor(&format!("{p}.b_post"), None)?,
            b_z: w.tensor(&format!("{p}.res.b_z"), None)?,
            gamma: 1.0 / (1.0 + (-eta).exp()),
        })
    }

    /// `stream [T, n, C]` → (`h_pre [T, n]`, `h_post [T, n]`, `h_res [T, n, n]`).
    fn route(&self, stream: &Tensor, k: &MhcConsts, cfg: &Config) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let (t, n, c) = stream.dims3()?;
        let flat = stream.reshape((t, n * c))?;
        let inv = flat.sqr()?.mean_keepdim(1)?.affine(1.0, cfg.sel_eps)?.sqrt()?.recip()?;
        let z = flat.matmul(&self.sel)?.broadcast_mul(&inv)?;
        let h_pre = sigmoid(&z.narrow(1, 0, n)?.affine(self.alpha_pre, 0.0)?.broadcast_add(&self.b_pre)?)?;
        let h_post = (sigmoid(&z.narrow(1, n, n)?.affine(self.alpha_post, 0.0)?.broadcast_add(&self.b_post)?)? * cfg.post_factor)?;
        let m = n - 1;
        let tz = z
            .narrow(1, 2 * n, m * m)?
            .affine(self.alpha_z, 0.0)?
            .broadcast_add(&self.b_z)?
            .reshape((t, m, m))?;
        let q = k.q.unsqueeze(0)?.broadcast_as((t, n, m))?.contiguous()?;
        let e = q.matmul(&tz)?.matmul(&q.transpose(1, 2)?.contiguous()?)?; // [T, n, n]
        let rho = e.neg()?.broadcast_div(&k.b)?.reshape((t, n * n))?.max_keepdim(1)?.relu()?; // [T, 1]
        let phi = e.broadcast_div(&(rho.affine(1.0, 1.0)?.unsqueeze(2)?))?.broadcast_add(&k.b)?;
        let h_res = (phi * self.gamma)?.broadcast_add(&(&k.eye * (1.0 - self.gamma))?)?;
        Ok((h_pre, h_post, h_res))
    }
}

fn aggregate(stream: &Tensor, h_pre: &Tensor) -> candle_core::Result<Tensor> {
    stream.broadcast_mul(&h_pre.unsqueeze(2)?)?.sum(1)
}

fn mix(stream: &Tensor, branch: &Tensor, h_post: &Tensor, h_res: &Tensor) -> candle_core::Result<Tensor> {
    h_res
        .matmul(&stream.contiguous()?)?
        .broadcast_add(&h_post.unsqueeze(2)?.broadcast_mul(&branch.unsqueeze(1)?)?)
}

// ---------------------------------------------------------------------------------------------
// RDA
// ---------------------------------------------------------------------------------------------

/// Recurrent state of one RDA layer: `[H, V, K]` f32, kept where the scan runs.
#[derive(Clone)]
pub enum RecState {
    Empty,
    Host(Vec<f32>),
    Dev(Tensor),
}

#[derive(Clone)]
pub struct RdaState {
    /// `[3·proj, K-1]` raw (pre-conv) q|k|v rows.
    pub conv: Option<Tensor>,
    pub rec: RecState,
}

struct Rda {
    q: Linear,
    k: Linear,
    v: Linear,
    g: Linear,
    b: Linear,
    w: Linear,
    fa: Linear,
    fb: Linear,
    conv: [Tensor; 3], // [C, K]
    a_exp: Tensor,     // [1, H, 1]
    dt_bias: Tensor,   // [H, D]
    o_norm: Tensor,
    o: Linear,
}

/// Token-serial V-first delta rule on the host (rayon over heads): fp32, exact reference order.
#[allow(clippy::too_many_arguments)]
fn host_scan(q: &[f32], k: &[f32], v: &[f32], la: &[f32], beta: &[f32], state: &mut [f32], t: usize, h: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * h * d];
    let hd = d * d;
    let outs: Vec<Vec<f32>> = state
        .par_chunks_mut(hd)
        .enumerate()
        .map(|(hi, s)| {
            let mut o = vec![0f32; t * d];
            let mut pred = vec![0f32; d];
            for ti in 0..t {
                let base = (ti * h + hi) * d;
                let (qt, kt, vt, at) = (&q[base..base + d], &k[base..base + d], &v[base..base + d], &la[base..base + d]);
                let bt = beta[ti * h + hi];
                let decay: Vec<f32> = at.iter().map(|x| x.exp()).collect();
                // S[v][k] *= exp(la[k]); pred[v] = Σ_k S[v][k] k[k]
                for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                    let mut acc = 0f32;
                    for ki in 0..d {
                        row[ki] *= decay[ki];
                        acc += row[ki] * kt[ki];
                    }
                    pred[vi] = acc;
                }
                // S += β (v − pred) kᵀ; out = S q
                for (vi, row) in s.chunks_exact_mut(d).enumerate() {
                    let e = bt * (vt[vi] - pred[vi]);
                    let mut acc = 0f32;
                    for ki in 0..d {
                        row[ki] += e * kt[ki];
                        acc += row[ki] * qt[ki];
                    }
                    o[ti * d + vi] = acc;
                }
            }
            o
        })
        .collect();
    for (hi, o) in outs.into_iter().enumerate() {
        for ti in 0..t {
            out[(ti * h + hi) * d..(ti * h + hi + 1) * d].copy_from_slice(&o[ti * d..(ti + 1) * d]);
        }
    }
    out
}

fn l2norm(x: &Tensor) -> candle_core::Result<Tensor> {
    // torch F.normalize: x / max(‖x‖₂, eps)
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-6)?;
    x.broadcast_div(&n)
}

fn short_conv(x: &Tensor, w: &Tensor, hist: Option<Tensor>, k: usize) -> candle_core::Result<(Tensor, Tensor)> {
    let (t, c) = x.dims2()?;
    let sig = x.t()?.contiguous()?; // [C, T]
    let hist = match hist {
        Some(h) => h,
        None => Tensor::zeros((c, k - 1), DType::F32, x.device())?,
    };
    let padded = Tensor::cat(&[&hist, &sig], 1)?; // [C, T+K-1]
    let mut out = padded.narrow(1, 0, t)?.broadcast_mul(&w.narrow(1, 0, 1)?)?;
    for j in 1..k {
        out = (out + padded.narrow(1, j, t)?.broadcast_mul(&w.narrow(1, j, 1)?)?)?;
    }
    let new_hist = padded.narrow(1, t, k - 1)?.contiguous()?;
    Ok((out.silu()?.t()?.contiguous()?, new_hist))
}

impl Rda {
    fn load(w: &Weights, p: &str, cfg: &Config) -> Result<Self> {
        let h = cfg.rda_heads;
        let a: Vec<f32> = w.host_f32(&format!("{p}.A_log"))?.iter().map(|x| x.exp()).collect();
        Ok(Self {
            q: w.linear(&format!("{p}.q_proj.weight"))?,
            k: w.linear(&format!("{p}.k_proj.weight"))?,
            v: w.linear(&format!("{p}.v_proj.weight"))?,
            g: w.linear(&format!("{p}.g_proj.weight"))?,
            b: w.linear(&format!("{p}.b_proj.weight"))?,
            w: w.linear(&format!("{p}.rda_w_proj.weight"))?,
            fa: w.linear(&format!("{p}.f_a_proj.weight"))?,
            fb: w.linear(&format!("{p}.f_b_proj.weight"))?,
            conv: [
                w.tensor(&format!("{p}.q_conv1d.weight"), None)?,
                w.tensor(&format!("{p}.k_conv1d.weight"), None)?,
                w.tensor(&format!("{p}.v_conv1d.weight"), None)?,
            ],
            a_exp: Tensor::from_vec(a, (1, h, 1), &w.device)?,
            dt_bias: w.tensor(&format!("{p}.dt_bias"), Some(&[h, cfg.rda_head_dim]))?,
            o_norm: w.tensor(&format!("{p}.o_norm.weight"), None)?,
            o: w.linear(&format!("{p}.o_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor, st: &mut RdaState, cfg: &Config) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let (h, d) = (cfg.rda_heads, cfg.rda_head_dim);
        let proj = h * d;
        let (q, k, v) = (self.q.forward(x)?, self.k.forward(x)?, self.v.forward(x)?);
        let gate = self.g.forward(x)?;
        let beta_raw = self.b.forward(x)?;
        let delta = self.w.forward(x)?;
        let decay_raw = self.fb.forward(&self.fa.forward(x)?)?;

        let hist = |i: usize| st.conv.as_ref().map(|c| c.narrow(0, i * proj, proj)).transpose();
        let (q, qh) = short_conv(&q, &self.conv[0], hist(0)?, cfg.conv_k)?;
        let (k, kh) = short_conv(&k, &self.conv[1], hist(1)?, cfg.conv_k)?;
        let (v, vh) = short_conv(&v, &self.conv[2], hist(2)?, cfg.conv_k)?;
        st.conv = Some(Tensor::cat(&[qh, kh, vh], 0)?);

        let q = (l2norm(&q.reshape((t, h, d))?)? * (d as f64).powf(-0.5))?;
        let k = l2norm(&k.reshape((t, h, d))?)?;
        let scaled = decay_raw
            .reshape((t, h, d))?
            .broadcast_add(&self.dt_bias)?
            .broadcast_mul(&self.a_exp)?;
        let log_alpha = match cfg.gate_lower_bound {
            Some(lb) => (sigmoid(&scaled)? * lb)?,
            // -softplus(x)
            None => (scaled.exp()?.affine(1.0, 1.0)?.log()? * -1.0)?,
        };
        let beta = sigmoid(&beta_raw)?; // [T, H]
                                        // write gate sigmoid(b + δ) / erase gate sigmoid(b), realised as a value scale
        let ratio = sigmoid(&(&beta_raw + &delta)?)?.div(&beta)?;
        let v = v.reshape((t, h, d))?.broadcast_mul(&ratio.unsqueeze(2)?)?;

        let out = if t == 1 && !x.device().is_cpu() {
            self.device_step(&q, &k, &v, &log_alpha, &beta, st, h, d)?
        } else {
            let mut s = match std::mem::replace(&mut st.rec, RecState::Empty) {
                RecState::Empty => vec![0f32; h * d * d],
                RecState::Host(s) => s,
                RecState::Dev(t) => t.flatten_all()?.to_vec1::<f32>()?,
            };
            let f = |x: &Tensor| x.flatten_all().and_then(|x| x.to_vec1::<f32>());
            let o = host_scan(&f(&q)?, &f(&k)?, &f(&v)?, &f(&log_alpha)?, &f(&beta)?, &mut s, t, h, d);
            st.rec = if x.device().is_cpu() {
                RecState::Host(s)
            } else {
                RecState::Dev(Tensor::from_vec(s, (h, d, d), x.device())?)
            };
            Tensor::from_vec(o, (t, h, d), x.device())?
        };
        let normed = rms(&out, &self.o_norm, cfg.eps)?;
        let gated = normed.mul(&sigmoid(&gate.reshape((t, h, d))?)?)?;
        Ok(self.o.forward(&gated.reshape((t, proj))?)?)
    }

    /// One decode token entirely on the device (no host round trip).
    #[allow(clippy::too_many_arguments)]
    fn device_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        la: &Tensor,
        beta: &Tensor,
        st: &mut RdaState,
        h: usize,
        d: usize,
    ) -> Result<Tensor> {
        let dev = q.device();
        let s = match std::mem::replace(&mut st.rec, RecState::Empty) {
            RecState::Empty => Tensor::zeros((h, d, d), DType::F32, dev)?,
            RecState::Host(s) => Tensor::from_vec(s, (h, d, d), dev)?,
            RecState::Dev(t) => t,
        };
        let (q, k, v, la) = (
            q.reshape((h, d, 1))?,
            k.reshape((h, d, 1))?,
            v.reshape((h, d, 1))?,
            la.reshape((h, 1, d))?,
        );
        let s = s.broadcast_mul(&la.exp()?)?; // decay along K
        let pred = s.matmul(&k)?; // [H, V, 1]
        let err = (v - pred)?.broadcast_mul(&beta.reshape((h, 1, 1))?)?;
        let s = (s + err.matmul(&k.transpose(1, 2)?)?)?;
        let out = s.matmul(&q)?.reshape((1, h, d))?;
        st.rec = RecState::Dev(s);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// MLA (NoPE, gated)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct KvState {
    pub k: Option<Tensor>, // [H, L, 192]
    pub v: Option<Tensor>, // [H, L, 128]
}

struct Mla {
    q_a: Linear,
    q_a_norm: Tensor,
    q_b: Linear,
    kv_a: Linear,
    kv_a_norm: Tensor,
    kv_b: Linear,
    g: Linear,
    o: Linear,
}

impl Mla {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            q_a: w.linear(&format!("{p}.q_a_proj.weight"))?,
            q_a_norm: w.tensor(&format!("{p}.q_a_layernorm.weight"), None)?,
            q_b: w.linear(&format!("{p}.q_b_proj.weight"))?,
            kv_a: w.linear(&format!("{p}.kv_a_proj_with_mqa.weight"))?,
            kv_a_norm: w.tensor(&format!("{p}.kv_a_layernorm.weight"), None)?,
            kv_b: w.linear(&format!("{p}.kv_b_proj.weight"))?,
            g: w.linear(&format!("{p}.g_proj.weight"))?,
            o: w.linear(&format!("{p}.o_proj.weight"))?,
        })
    }

    fn forward(&self, x: &Tensor, st: &mut KvState, cfg: &Config) -> Result<Tensor> {
        let (t, _) = x.dims2()?;
        let (h, dn, dp, dv) = (cfg.heads, cfg.qk_nope, cfg.qk_pe, cfg.v_dim);
        let dq = dn + dp;
        let q = self
            .q_b
            .forward(&rms(&self.q_a.forward(x)?, &self.q_a_norm, cfg.mla_eps)?)?
            .reshape((t, h, dq))?
            .transpose(0, 1)?
            .contiguous()?;
        let kv_full = self.kv_a.forward(x)?;
        let lat = kv_full.narrow(1, 0, cfg.kv_lora)?;
        let pe = kv_full.narrow(1, cfg.kv_lora, dp)?;
        let kv = self
            .kv_b
            .forward(&rms(&lat, &self.kv_a_norm, cfg.mla_eps)?)?
            .reshape((t, h, dn + dv))?;
        let key = Tensor::cat(&[kv.narrow(2, 0, dn)?, pe.unsqueeze(1)?.broadcast_as((t, h, dp))?.contiguous()?], 2)?
            .transpose(0, 1)?
            .contiguous()?;
        let val = kv.narrow(2, dn, dv)?.transpose(0, 1)?.contiguous()?;
        let key = match st.k.take() {
            Some(p) => Tensor::cat(&[p, key], 1)?,
            None => key,
        };
        let val = match st.v.take() {
            Some(p) => Tensor::cat(&[p, val], 1)?,
            None => val,
        };
        st.k = Some(key.clone());
        st.v = Some(val.clone());
        let l = key.dim(1)?;
        let scale = (dq as f64).powf(-0.5);
        let mut scores = (q.matmul(&key.t()?.contiguous()?)? * scale)?; // [H, T, L]
        if t > 1 {
            let off = l - t;
            let mask: Vec<f32> = (0..t)
                .flat_map(|i| (0..l).map(move |j| if j > i + off { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            scores = scores.broadcast_add(&Tensor::from_vec(mask, (1, t, l), x.device())?)?;
        }
        let ctx = softmax_last_dim(&scores)?.matmul(&val)?; // [H, T, dv]
        let ctx = ctx.transpose(0, 1)?.contiguous()?.reshape((t, h * dv))?;
        let ctx = ctx.mul(&sigmoid(&self.g.forward(x)?)?)?;
        Ok(self.o.forward(&ctx)?)
    }
}

// ---------------------------------------------------------------------------------------------
// Layer / model
// ---------------------------------------------------------------------------------------------

enum Attn {
    Rda(Box<Rda>),
    Mla(Box<Mla>),
}

#[derive(Clone)]
pub enum LayerState {
    Rda(RdaState),
    Mla(KvState),
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            gate: w.linear(&format!("{p}.gate_proj.weight"))?,
            up: w.linear(&format!("{p}.up_proj.weight"))?,
            down: w.linear(&format!("{p}.down_proj.weight"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let a = self.gate.forward(x)?.silu()?.mul(&self.up.forward(x)?)?;
        Ok(self.down.forward(&a)?)
    }
}

struct PleGate {
    gate: Linear,
    proj: Linear,
    norm: Tensor,
}

struct PleTimer(Instant);
impl Drop for PleTimer {
    fn drop(&mut self) {
        profile::add(4, self.0);
    }
}

struct Layer {
    in_norm: Tensor,
    post_norm: Tensor,
    attn: Attn,
    attn_mhc: Mhc,
    mlp_mhc: Mhc,
    mlp: Mlp,
    ple: Option<PleGate>,
}

impl Layer {
    fn load(w: &Weights, i: usize, cfg: &Config, rda: bool) -> Result<Self> {
        let p = format!("model.layers.{i}");
        let ple = if w.has(&format!("{p}.per_layer_input_gate.gate")) {
            Some(PleGate {
                gate: w.linear(&format!("{p}.per_layer_input_gate.gate"))?,
                proj: w.linear(&format!("{p}.per_layer_input_gate.projection"))?,
                norm: w.tensor(&format!("{p}.per_layer_input_gate.output_norm.weight"), None)?,
            })
        } else {
            None
        };
        Ok(Self {
            in_norm: w.tensor(&format!("{p}.input_layernorm.weight"), None)?,
            post_norm: w.tensor(&format!("{p}.post_attention_layernorm.weight"), None)?,
            attn: if rda {
                Attn::Rda(Box::new(Rda::load(w, &format!("{p}.self_attn"), cfg)?))
            } else {
                Attn::Mla(Box::new(Mla::load(w, &format!("{p}.self_attn"))?))
            },
            attn_mhc: Mhc::load(w, &format!("{p}.self_attn_raon_mhc"))?,
            mlp_mhc: Mhc::load(w, &format!("{p}.mlp_raon_mhc"))?,
            mlp: Mlp::load(w, &format!("{p}.mlp"))?,
            ple,
        })
    }

    fn forward(&self, stream: &Tensor, st: &mut LayerState, ple_in: Option<&Tensor>, k: &MhcConsts, cfg: &Config) -> Result<Tensor> {
        let t = Instant::now();
        let (h_pre, h_post, h_res) = self.attn_mhc.route(stream, k, cfg)?;
        let x = rms(&aggregate(stream, &h_pre)?, &self.in_norm, cfg.eps)?;
        profile::add(0, t);
        let t = Instant::now();
        let out = match (&self.attn, st) {
            (Attn::Rda(a), LayerState::Rda(s)) => {
                let o = a.forward(&x, s, cfg)?;
                profile::add(1, t);
                o
            }
            (Attn::Mla(a), LayerState::Mla(s)) => {
                let o = a.forward(&x, s, cfg)?;
                profile::add(2, t);
                o
            }
            _ => anyhow::bail!("layer state does not match the layer type"),
        };
        let t = Instant::now();
        let stream = mix(stream, &out, &h_post, &h_res)?;
        profile::add(5, t);
        let t = Instant::now();
        let (h_pre, h_post, h_res) = self.mlp_mhc.route(&stream, k, cfg)?;
        let x = rms(&aggregate(&stream, &h_pre)?, &self.post_norm, cfg.eps)?;
        profile::add(0, t);
        let t = Instant::now();
        let out = self.mlp.forward(&x)?;
        profile::add(3, t);
        let t = Instant::now();
        let mut stream = mix(&stream, &out, &h_post, &h_res)?;
        profile::add(5, t);
        let t = Instant::now();
        let _g = PleTimer(t);
        if let (Some(p), Some(li)) = (&self.ple, ple_in) {
            let hidden = stream.mean(1)?;
            let gated = p.gate.forward(&hidden)?.gelu()?.mul(li)?;
            let delta = rms(&p.proj.forward(&gated)?, &p.norm, cfg.eps)?;
            stream = stream.broadcast_add(&delta.unsqueeze(1)?)?;
        }
        Ok(stream)
    }
}

#[derive(Clone)]
pub struct State {
    pub layers: Vec<LayerState>,
    pub pos: usize,
}

pub struct Model {
    pub cfg: Config,
    pub w: Weights,
    layers: Vec<Layer>,
    norm: Tensor,
    ple_proj: Option<Linear>,
    ple_norm: Option<Tensor>,
    consts: MhcConsts,
}

impl Model {
    pub fn load(mut w: Weights) -> Result<Self> {
        let cfg = Config::from_json(&w.config_json)?;
        let n = cfg.streams;
        let dev = w.device.clone();
        let eye: Vec<f32> = (0..n * n).map(|i| if i / n == i % n { 1.0 } else { 0.0 }).collect();
        let b: Vec<f32> = eye.iter().map(|e| 0.5 * (e + 1.0 / n as f32)).collect();
        let consts = MhcConsts {
            q: Tensor::from_vec(helmert(n), (n, n - 1), &dev)?,
            b: Tensor::from_vec(b, (n, n), &dev)?,
            eye: Tensor::from_vec(eye, (n, n), &dev)?,
        };
        let layers = (0..cfg.layers)
            .map(|i| Layer::load(&w, i, &cfg, cfg.rda_layer[i]).with_context(|| format!("layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let (ple_proj, ple_norm) = if w.has("model.per_layer_inputs.embedding_projection") {
            (
                Some(w.linear("model.per_layer_inputs.embedding_projection")?),
                Some(w.tensor("model.per_layer_inputs.projection_norm.weight", None)?),
            )
        } else {
            (None, None)
        };
        let norm = w.tensor("model.norm.weight", None)?;
        // Weights are on the device now; re-map the sources so only rows gathered later
        // (embeddings, PLE, head scan) stay resident.
        w.release_mapped_pages()?;
        Ok(Self {
            cfg,
            w,
            layers,
            norm,
            ple_proj,
            ple_norm,
            consts,
        })
    }

    pub fn device(&self) -> &Device {
        &self.w.device
    }

    pub fn new_state(&self) -> State {
        let layers = self
            .cfg
            .rda_layer
            .iter()
            .map(|&r| {
                if r {
                    LayerState::Rda(RdaState {
                        conv: None,
                        rec: RecState::Empty,
                    })
                } else {
                    LayerState::Mla(KvState::default())
                }
            })
            .collect();
        State { layers, pos: 0 }
    }

    pub fn embed(&self, ids: &[u32]) -> Result<Tensor> {
        self.w.rows("model.embed_tokens.weight", ids)
    }

    /// `p = (RMSNorm(W_mp e) + √d · T[token]) / √2` → `[T, layers, d]`.
    fn ple_inputs(&self, ids: &[u32], e: &Tensor) -> Result<Option<Tensor>> {
        let (Some(proj), Some(norm)) = (&self.ple_proj, &self.ple_norm) else {
            return Ok(None);
        };
        let (t, l, d) = (ids.len(), self.cfg.layers, self.cfg.ple_dim);
        let projected = rms(&proj.forward(e)?.reshape((t, l, d))?, norm, self.cfg.eps)?;
        let rows = (self.w.ple_rows(ids)?.reshape((t, l, d))? * (d as f64).sqrt())?;
        Ok(Some(((projected + rows)? * std::f64::consts::FRAC_1_SQRT_2)?))
    }

    /// Run `ids` through the model, updating `st`; returns the final normed hidden states `[T, C]`.
    pub fn forward(&self, ids: &[u32], st: &mut State) -> Result<Tensor> {
        let e = self.embed(ids)?;
        self.forward_embeds(ids, &e, st)
    }

    pub fn forward_embeds(&self, ids: &[u32], e: &Tensor, st: &mut State) -> Result<Tensor> {
        let (t, c) = e.dims2()?;
        let ple = self.ple_inputs(ids, e)?;
        let mut stream = e.unsqueeze(1)?.broadcast_as((t, self.cfg.streams, c))?.contiguous()?;
        for (i, layer) in self.layers.iter().enumerate() {
            let li = ple.as_ref().map(|p| p.narrow(1, i, 1).and_then(|x| x.squeeze(1))).transpose()?;
            stream = layer
                .forward(&stream, &mut st.layers[i], li.as_ref(), &self.consts, &self.cfg)
                .with_context(|| format!("layer {i}"))?;
        }
        st.pos += t;
        Ok(rms(&stream.mean(1)?, &self.norm, self.cfg.eps)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helmert_is_orthonormal_and_orthogonal_to_ones() {
        let n = 4;
        let q = helmert(n);
        for a in 0..n - 1 {
            let col_sum: f32 = (0..n).map(|i| q[i * (n - 1) + a]).sum();
            assert!(col_sum.abs() < 1e-6);
            for b in 0..n - 1 {
                let dot: f32 = (0..n).map(|i| q[i * (n - 1) + a] * q[i * (n - 1) + b]).sum();
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn host_scan_matches_device_step_on_cpu_tensors() {
        // Run one token through host_scan and through the tensor formulation; they must agree.
        let (h, d) = (2usize, 4usize);
        let mk = |s: u64| {
            (0..h * d)
                .map(|i| (((i as u64 * 2654435761 + s) % 1000) as f32 / 1000.0) - 0.5)
                .collect::<Vec<f32>>()
        };
        let (q, k, v, la) = (mk(1), mk(2), mk(3), mk(4).iter().map(|x| -x.abs()).collect::<Vec<f32>>());
        let beta = vec![0.3f32, 0.8];
        let mut s0: Vec<f32> = (0..h * d * d).map(|i| (i as f32 * 0.01).sin()).collect();
        let dev = Device::Cpu;
        let st = Tensor::from_vec(s0.clone(), (h, d, d), &dev).unwrap();
        let out = host_scan(&q, &k, &v, &la, &beta, &mut s0, 1, h, d);
        let t = |x: &Vec<f32>, s: (usize, usize, usize)| Tensor::from_vec(x.clone(), s, &dev).unwrap();
        let s1 = st.broadcast_mul(&t(&la, (h, 1, d)).exp().unwrap()).unwrap();
        let pred = s1.matmul(&t(&k, (h, d, 1))).unwrap();
        let err = (t(&v, (h, d, 1)) - pred)
            .unwrap()
            .broadcast_mul(&Tensor::from_vec(beta.clone(), (h, 1, 1), &dev).unwrap())
            .unwrap();
        let s2 = (s1 + err.matmul(&t(&k, (h, 1, d))).unwrap()).unwrap();
        let o = s2
            .matmul(&t(&q, (h, d, 1)))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in out.iter().zip(&o) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let s2v = s2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in s0.iter().zip(&s2v) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
