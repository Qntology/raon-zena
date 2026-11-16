//! Statistics isolation units. A [`Scope`] is `track | primary | secondary` (e.g. `drop | app://eldenring.exe |
//! boss-malenia`); observations are stored under the full (secondary) key and reads back off through the primary
//! and global keys. [`MinObs`] is the per-track sample floor of each level.

use serde::{Deserialize, Serialize};

/// Key of observations recorded without a track (logis "unscoped"); the last backoff level.
pub const UNSCOPED_KEY: &str = "unscoped||";

/// Ring length of the unscoped key (logis constant).
pub const UNSCOPED_RING: usize = 12;

/// Where a statistic belongs: record kind (`drop`, `web`, `coach`, `nms`), domain root, topic.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Scope {
    pub track: String,
    pub primary: String,
    pub secondary: String,
}

impl Scope {
    /// Trims all three parts and lowercases `primary` / `secondary` (logis `enter_scope`). Keys are joined with `|`,
    /// so parts should not contain it.
    pub fn new(track: &str, primary: &str, secondary: &str) -> Self {
        Self {
            track: track.trim().to_string(),
            primary: primary.trim().to_lowercase(),
            secondary: secondary.trim().to_lowercase(),
        }
    }

    /// No track: observations go to [`UNSCOPED_KEY`] and reads return `None`.
    pub fn is_empty(&self) -> bool {
        self.track.is_empty()
    }

    /// `track|primary|secondary`. With an empty secondary this equals [`Scope::key_primary`].
    pub fn key_secondary(&self) -> String {
        format!("{}|{}|{}", self.track, self.primary, self.secondary)
    }

    /// `track|primary|`.
    pub fn key_primary(&self) -> String {
        format!("{}|{}|", self.track, self.primary)
    }

    /// `track||`.
    pub fn key_global(&self) -> String {
        format!("{}||", self.track)
    }
}

/// Minimum observations before a level may answer. `0` disables that level (logis `need == 0 → skip`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinObs {
    pub secondary: u64,
    pub primary: u64,
    pub global: u64,
}

impl MinObs {
    pub const fn new(secondary: u64, primary: u64, global: u64) -> Self {
        Self {
            secondary,
            primary,
            global,
        }
    }

    /// Recent-ring length of the track's Welfords: the primary floor (logis `ring_len = min_obs().1`). `0` = unbounded.
    pub fn ring_len(&self) -> usize {
        self.primary as usize
    }
}

/// `(6, 3, 10)`: the floor of a track nobody configured (the design's `drop` default).
impl Default for MinObs {
    fn default() -> Self {
        Self::new(6, 3, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normalises_and_keys_nest() {
        let s = Scope::new(" drop ", " App://EldenRing.exe ", "Boss-Malenia ");
        assert_eq!(s.track, "drop");
        assert_eq!(s.key_secondary(), "drop|app://eldenring.exe|boss-malenia");
        assert_eq!(s.key_primary(), "drop|app://eldenring.exe|");
        assert_eq!(s.key_global(), "drop||");
        assert_eq!(
            Scope::new("drop", "x", "").key_secondary(),
            Scope::new("drop", "x", "").key_primary()
        );
        assert!(Scope::new("  ", "x", "y").is_empty());
    }

    #[test]
    fn min_obs_default_and_ring() {
        let m = MinObs::default();
        assert_eq!((m.secondary, m.primary, m.global), (6, 3, 10));
        assert_eq!(m.ring_len(), 3);
        assert_eq!(MinObs::new(8, 4, 12).ring_len(), 4);
    }
}
