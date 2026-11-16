//! One screen reading: the patch grid through the vision tower (full, or perforated with the minesweeper opening),
//! the pooled embedding scored against the dictionary's screen domains, and where on the screen the winning domain is
//! (the peak of its heat over the cells, for Zena's gaze). Only scores and one cell position leave this module.

use crate::bank::{AnchorBank, VisionScore};
use crate::perforate::{cells_to_open, fill_holes, first_cells, seeds, PerfReport, PerforatedCfg};
use crate::tower::{SparsePass, VisionTower};
use anyhow::Result;
use zena_sensor::naflex::PatchGrid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenConfig {
    /// NaFlex budget of the screen grid.
    pub max_patches: usize,
    /// `None`: the full pass.
    pub perforation: Option<PerforatedCfg>,
    /// Labels whose heat seeds the minesweeper opening (the best domains of the first pass).
    pub seed_labels: usize,
}

impl Default for ScreenConfig {
    fn default() -> Self {
        ScreenConfig {
            max_patches: 256,
            perforation: None,
            seed_labels: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScreenReading {
    /// `(grid_h, grid_w)`.
    pub grid: (usize, usize),
    pub scores: Vec<VisionScore>,
    /// Centre of the winning domain's hottest cell, in 0..1 screen coordinates.
    pub peak: Option<(f32, f32)>,
    pub report: PerfReport,
    pub encode_ms: f32,
    /// The pooled embedding (unit).
    pub pooled: Vec<f32>,
}

pub struct ScreenEncoder {
    pub tower: VisionTower,
    pub cfg: ScreenConfig,
}

impl ScreenEncoder {
    pub fn new(tower: VisionTower, cfg: ScreenConfig) -> ScreenEncoder {
        ScreenEncoder { tower, cfg }
    }

    /// The pass over `grid` this encoder is configured for.
    pub fn encode(&self, grid: &PatchGrid, bank: &AnchorBank) -> Result<(SparsePass, PerfReport)> {
        encode_pass(&self.tower, &self.cfg, grid, bank)
    }

    /// Encodes, scores and locates.
    pub fn read(&self, grid: &PatchGrid, bank: &AnchorBank) -> Result<ScreenReading> {
        read_pass(&self.tower, &self.cfg, grid, bank)
    }
}

/// Contrast heats of `labels` on the cells a pass evaluated (NaN on the others).
fn label_heats(tower: &VisionTower, pass: &SparsePass, bank: &AnchorBank, labels: &[usize], n: usize) -> Result<Vec<Vec<f32>>> {
    let dense = tower.dense(pass)?;
    let mut out = vec![vec![f32::NAN; n]; labels.len()];
    for (c, v) in pass.cells.iter().zip(&dense) {
        for (k, h) in bank.contrast(labels, v).into_iter().enumerate() {
            out[k][*c as usize] = h;
        }
    }
    Ok(out)
}

/// The pass over `grid` that `cfg` asks for: full, or perforated with the minesweeper opening.
pub fn encode_pass(tower: &VisionTower, cfg: &ScreenConfig, grid: &PatchGrid, bank: &AnchorBank) -> Result<(SparsePass, PerfReport)> {
    let n = grid.cells();
    let (cols, rows) = (grid.grid_w as u32, grid.grid_h as u32);
    let Some(perf) = cfg.perforation else {
        return Ok((
            tower.forward_full(grid)?,
            PerfReport {
                cells: n,
                first: n,
                tokens_run: n,
                ..PerfReport::default()
            },
        ));
    };
    let first = first_cells(cols, rows, perf.pattern);
    let mut pass = tower.forward_sparse(grid, &first)?;
    let mut rep = PerfReport {
        cells: n,
        first: first.len(),
        tokens_run: first.len(),
        ..PerfReport::default()
    };
    if let (Some(x), false) = (perf.expand, bank.is_empty()) {
        let pooled = tower.pooled(&pass)?;
        let labels: Vec<usize> = bank
            .score(&pooled)
            .iter()
            .take(cfg.seed_labels)
            .filter_map(|s| bank.index_of(&s.key))
            .collect();
        let mut heats = label_heats(tower, &pass, bank, &labels, n)?;
        let seed_cells = seeds(&heats, cols, rows, &x);
        rep.seeds = seed_cells.len();
        for round in 0..x.rounds {
            let mut open = vec![false; n];
            for &c in &pass.cells {
                open[c as usize] = true;
            }
            let add = cells_to_open(&heats, &open, &seed_cells, cols, rows, round, &x);
            if add.is_empty() {
                break;
            }
            let more = tower.extend_sparse(grid, &pass, &add)?;
            pass = tower.merge_sparse(&pass, &more)?;
            rep.opened += add.len();
            rep.tokens_run += add.len();
            heats = label_heats(tower, &pass, bank, &labels, n)?;
        }
    }
    Ok((pass, rep))
}

/// Encodes, scores and locates.
pub fn read_pass(tower: &VisionTower, cfg: &ScreenConfig, grid: &PatchGrid, bank: &AnchorBank) -> Result<ScreenReading> {
    let t0 = std::time::Instant::now();
    let (pass, report) = encode_pass(tower, cfg, grid, bank)?;
    let pooled = tower.pooled(&pass)?;
    let scores = bank.score(&pooled);
    let n = grid.cells();
    let (cols, rows) = (grid.grid_w as u32, grid.grid_h as u32);
    let top: Vec<usize> = scores
        .iter()
        .take(cfg.seed_labels.max(1))
        .filter_map(|s| bank.index_of(&s.key))
        .collect();
    let peak = match top.first() {
        Some(_) => {
            let mut h = label_heats(tower, &pass, bank, &top, n)?.remove(0);
            let mut open = vec![false; n];
            for &c in &pass.cells {
                open[c as usize] = true;
            }
            fill_holes(&mut h, &open, cols, rows);
            h.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(c, _)| {
                let (c, r) = (c as u32 % cols, c as u32 / cols);
                ((c as f32 + 0.5) / cols as f32, (r as f32 + 0.5) / rows as f32)
            })
        }
        None => None,
    };
    Ok(ScreenReading {
        grid: (grid.grid_h, grid.grid_w),
        scores,
        peak,
        report,
        encode_ms: t0.elapsed().as_secs_f32() * 1000.0,
        pooled,
    })
}
