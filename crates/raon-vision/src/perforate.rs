//! Perforated passes (port of noto-sandbox `noto-vision/src/perforate.rs`, Apache-2.0).
//!
//! Only some patches of the grid go through SigLIP2 in the first pass — a checkerboard, or a fovea — each with its own
//! cell's position embedding (holes are left out of the encoder, not blacked out). Around the strongest label peaks the
//! closed neighbours are then opened like a minesweeper board: they attend to every token computed so far
//! (`VisionTower::extend_sparse`). The pattern and the opening rule are fixed functions of the grid and the heat — no
//! random drop, the same frame gives the same cells. Holes never opened take the lowest heat of their evaluated
//! neighbours, so a hole never makes a peak of its own.
//!
//! noto's measurements (docs/research/ml-perforated.md there) are the reason this is an option, not the default:
//! holes change the features of the remaining tokens, so a perforated pass is not the full pass; on CPU the
//! checkerboard saved ~41 % of the encoder time and the fovea ~29 %, fovea + opening only +14 % (it costs more than it
//! saves), and on a GPU the gain was at most ~11 %. Zena therefore runs the full pass on a GPU and offers the
//! perforated pass for CPU-only machines, where the screen watcher's budget is tight.

use serde::{Deserialize, Serialize};

/// Which cells the first pass evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerfPattern {
    /// Every other cell (half the grid).
    Checker,
    /// A screen fovea: the middle of the screen dense, a checkerboard around it, a quarter of the outermost rows
    /// (title bars, task bars).
    Fovea,
    /// Every cell (the full pass through the perforated path; for checks).
    All,
}

/// The minesweeper step: seeds are each label's 3 × 3 local maxima of the first pass's heat at or above `seed_min`
/// (at most `seeds_per_label` per label, strongest first); every closed cell within one cell of a seed opens, and for
/// `rounds` − 1 more rounds the closed neighbours of opened cells whose heat reaches `grow_min` open too, never farther
/// than `radius` cells from their seed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpandCfg {
    pub seed_min: f32,
    pub grow_min: f32,
    pub radius: u32,
    pub seeds_per_label: usize,
    pub rounds: u32,
}

impl Default for ExpandCfg {
    fn default() -> Self {
        ExpandCfg {
            seed_min: 0.15,
            grow_min: 0.30,
            radius: 2,
            seeds_per_label: 2,
            rounds: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerforatedCfg {
    pub pattern: PerfPattern,
    #[serde(default)]
    pub expand: Option<ExpandCfg>,
}

impl PerforatedCfg {
    /// `checker`, `fovea`, `all`, each optionally with `-x` (the default minesweeper opening).
    pub fn parse(s: &str) -> Option<PerforatedCfg> {
        let (p, x) = match s.strip_suffix("-x") {
            Some(p) => (p, true),
            None => (s, false),
        };
        let pattern = match p {
            "checker" => PerfPattern::Checker,
            "fovea" => PerfPattern::Fovea,
            "all" => PerfPattern::All,
            _ => return None,
        };
        Some(PerforatedCfg {
            pattern,
            expand: x.then(ExpandCfg::default),
        })
    }

    pub fn tag(&self) -> String {
        let p = match self.pattern {
            PerfPattern::Checker => "checker",
            PerfPattern::Fovea => "fovea",
            PerfPattern::All => "all",
        };
        if self.expand.is_some() {
            format!("{p}-x")
        } else {
            p.to_string()
        }
    }
}

/// The cells (row-major) the first pass evaluates.
pub fn first_cells(cols: u32, rows: u32, pattern: PerfPattern) -> Vec<u32> {
    let mut out = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let checker = (c + r) % 2 == 0;
            let keep = match pattern {
                PerfPattern::Checker => checker,
                PerfPattern::All => true,
                PerfPattern::Fovea => {
                    let x = (c as f32 + 0.5) / cols as f32;
                    let y = (r as f32 + 0.5) / rows as f32;
                    if y < 1.0 / 12.0 || y > 11.0 / 12.0 {
                        c % 2 == 0 && r % 2 == 0
                    } else if (1.0 / 6.0..5.0 / 6.0).contains(&x) && (1.0 / 6.0..5.0 / 6.0).contains(&y) {
                        true
                    } else {
                        checker
                    }
                }
            };
            if keep {
                out.push(r * cols + c);
            }
        }
    }
    out
}

fn neighbours(c: u32, r: u32, cols: u32, rows: u32, reach: i32) -> impl Iterator<Item = (u32, u32)> {
    (-reach..=reach)
        .flat_map(move |dr| (-reach..=reach).map(move |dc| (dc, dr)))
        .filter_map(move |(dc, dr)| {
            let (cc, rr) = (c as i32 + dc, r as i32 + dr);
            ((dc, dr) != (0, 0) && cc >= 0 && rr >= 0 && cc < cols as i32 && rr < rows as i32).then_some((cc as u32, rr as u32))
        })
}

/// The seeds of the minesweeper step: per label (`heats[label][cell]`, NaN on closed cells), the evaluated cells that
/// are 3 × 3 local maxima among evaluated cells with heat ≥ `seed_min`, strongest first, at most `seeds_per_label`.
pub fn seeds(heats: &[Vec<f32>], cols: u32, rows: u32, cfg: &ExpandCfg) -> Vec<u32> {
    let mut out = Vec::new();
    for h in heats {
        let mut local: Vec<(u32, f32)> = (0..rows)
            .flat_map(|r| (0..cols).map(move |c| (c, r)))
            .filter_map(|(c, r)| {
                let v = h[(r * cols + c) as usize];
                let peak = v.is_finite()
                    && v >= cfg.seed_min
                    && neighbours(c, r, cols, rows, 1).all(|(cc, rr)| !(h[(rr * cols + cc) as usize] > v));
                peak.then_some((r * cols + c, v))
            })
            .collect();
        local.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.extend(local.into_iter().take(cfg.seeds_per_label).map(|(i, _)| i));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The closed cells to open next: those within one cell of a seed (`round` 0), or (later rounds) of an evaluated cell
/// within `radius` of a seed whose heat for any label reaches `grow_min`.
pub fn cells_to_open(heats: &[Vec<f32>], open: &[bool], seed_cells: &[u32], cols: u32, rows: u32, round: u32, cfg: &ExpandCfg) -> Vec<u32> {
    let mut want = vec![false; open.len()];
    let near_seed = |c: u32, r: u32| {
        seed_cells
            .iter()
            .any(|&s| (c as i32 - (s % cols) as i32).abs().max((r as i32 - (s / cols) as i32).abs()) <= cfg.radius as i32)
    };
    if round == 0 {
        for &s in seed_cells {
            for (cc, rr) in neighbours(s % cols, s / cols, cols, rows, 1) {
                want[(rr * cols + cc) as usize] = true;
            }
        }
    } else {
        for r in 0..rows {
            for c in 0..cols {
                let i = (r * cols + c) as usize;
                if !open[i] || !near_seed(c, r) || !heats.iter().any(|h| h[i] >= cfg.grow_min) {
                    continue;
                }
                for (cc, rr) in neighbours(c, r, cols, rows, 1) {
                    if near_seed(cc, rr) {
                        want[(rr * cols + cc) as usize] = true;
                    }
                }
            }
        }
    }
    (0..open.len()).filter(|&i| want[i] && !open[i]).map(|i| i as u32).collect()
}

/// Fills the closed cells of a heat grid with the lowest heat of their evaluated 8-neighbours (else the lowest
/// evaluated heat), so that no peak grows out of a hole.
pub fn fill_holes(values: &mut [f32], open: &[bool], cols: u32, rows: u32) {
    let floor = values
        .iter()
        .zip(open)
        .filter(|(_, o)| **o)
        .map(|(v, _)| *v)
        .fold(f32::INFINITY, f32::min);
    let floor = if floor.is_finite() { floor } else { 0.0 };
    let filled: Vec<f32> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (c, r)))
        .map(|(c, r)| {
            let i = (r * cols + c) as usize;
            if open[i] {
                return values[i];
            }
            neighbours(c, r, cols, rows, 1)
                .map(|(cc, rr)| (rr * cols + cc) as usize)
                .filter(|&j| open[j])
                .map(|j| values[j])
                .fold(f32::INFINITY, f32::min)
        })
        .map(|v| if v.is_finite() { v } else { floor })
        .collect();
    values.copy_from_slice(&filled);
}

/// What one pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfReport {
    pub cells: usize,
    pub first: usize,
    pub seeds: usize,
    pub opened: usize,
    /// Tokens that went through the encoder over all passes.
    pub tokens_run: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_checkerboard_keeps_half_and_the_fovea_the_middle() {
        assert_eq!(first_cells(18, 12, PerfPattern::Checker).len(), 108);
        assert_eq!(first_cells(21, 12, PerfPattern::All).len(), 252);
        let f = first_cells(21, 12, PerfPattern::Fovea);
        assert!((140..200).contains(&f.len()), "{} of 252", f.len());
        let at = |c: u32, r: u32| f.contains(&(r * 21 + c));
        assert!((4..17).all(|c| at(c, 5) && at(c, 6)), "the middle of the screen is dense");
        assert!(!at(1, 0) && !at(3, 0), "a quarter of the top row");
    }

    #[test]
    fn a_seed_opens_its_ring_and_strong_neighbours_grow_it_within_the_radius() {
        let (cols, rows) = (6u32, 5u32);
        let n = (cols * rows) as usize;
        let first = first_cells(cols, rows, PerfPattern::Checker);
        let mut open = vec![false; n];
        for &c in &first {
            open[c as usize] = true;
        }
        let mut h = vec![f32::NAN; n];
        for &c in &first {
            h[c as usize] = 0.05;
        }
        h[2 * 6 + 2] = 0.8;
        h[6 + 1] = 0.4;
        let cfg = ExpandCfg::default();
        let s = seeds(&[h.clone()], cols, rows, &cfg);
        assert_eq!(s, vec![14]);
        let ring = cells_to_open(&[h.clone()], &open, &s, cols, rows, 0, &cfg);
        assert_eq!(ring, vec![8, 13, 15, 20], "the closed cells of the 3 × 3 ring");
        for &c in &ring {
            open[c as usize] = true;
            h[c as usize] = 0.5;
        }
        let grow = cells_to_open(&[h.clone()], &open, &s, cols, rows, 1, &cfg);
        assert!(!grow.is_empty() && grow.iter().all(|&c| !open[c as usize]));
        assert!(
            grow.iter()
                .all(|&c| ((c % 6) as i32 - 2).abs().max(((c / 6) as i32 - 2).abs()) <= 2),
            "within the radius: {grow:?}"
        );
        let quiet: Vec<f32> = h.iter().map(|v| v.min(0.1)).collect();
        assert!(seeds(&[quiet], cols, rows, &cfg).is_empty());
    }

    #[test]
    fn holes_take_their_lowest_evaluated_neighbour() {
        let open = vec![true, false, true, true];
        let mut v = vec![0.9, f32::NAN, 0.2, 0.5];
        fill_holes(&mut v, &open, 2, 2);
        assert_eq!(v, vec![0.9, 0.2, 0.2, 0.5]);
        let mut none = vec![f32::NAN; 4];
        fill_holes(&mut none, &[false; 4], 2, 2);
        assert_eq!(none, vec![0.0; 4]);
    }

    #[test]
    fn tags_parse_back() {
        for t in ["checker", "fovea", "all", "checker-x", "fovea-x"] {
            assert_eq!(PerforatedCfg::parse(t).unwrap().tag(), t);
        }
        assert!(PerforatedCfg::parse("ring").is_none());
    }
}
