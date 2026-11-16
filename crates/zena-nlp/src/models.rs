//! Where the text models live on this machine.
//!
//! Order for each model: an explicit environment variable, Raon Zena's own model folder
//! (`%LOCALAPPDATA%\RaonZena\models\…`, where the in-app downloader puts them), then an existing Qntology/logis-center
//! install of the same files (`%LOCALAPPDATA%\logis-center\models\…`), so a machine that already has them does not
//! download them twice.

use std::path::{Path, PathBuf};

/// The granite embedding model folder name (Hugging Face repository name).
pub const GRANITE_DIR: &str = "granite-embedding-97m-multilingual-r2";

/// Per-user data folder (`%LOCALAPPDATA%`, `$XDG_DATA_HOME`, `~/.local/share`).
fn data_local() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_DATA_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))
}

/// Raon Zena's model folder.
pub fn raon_models_dir() -> Option<PathBuf> {
    data_local().map(|d| d.join("RaonZena").join("models"))
}

/// Model folders to search, best first.
pub fn search_roots() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = std::env::var_os("RAON_MODELS_DIR") {
        v.push(PathBuf::from(d));
    }
    if let Some(d) = raon_models_dir() {
        v.push(d);
    }
    if let Some(d) = data_local() {
        v.push(d.join("logis-center").join("models"));
    }
    v
}

fn first_with(candidates: impl IntoIterator<Item = PathBuf>, marker: &str) -> Option<PathBuf> {
    candidates.into_iter().find(|d| d.join(marker).is_file())
}

/// The granite folder (has `model.safetensors`): `RAON_GRANITE_DIR`, then the search roots.
pub fn granite_dir() -> Option<PathBuf> {
    let env = std::env::var_os("RAON_GRANITE_DIR").map(PathBuf::from);
    first_with(
        env.into_iter().chain(search_roots().into_iter().map(|r| r.join(GRANITE_DIR))),
        "model.safetensors",
    )
}

/// The Stanza folder for language `lang` (has `vocab.json`): `RAON_STANZA_DIR/<lang>`, then `<root>/stanza/<lang>`.
pub fn stanza_dir(lang: &str) -> Option<PathBuf> {
    let env = std::env::var_os("RAON_STANZA_DIR").map(|d| PathBuf::from(d).join(lang));
    first_with(
        env.into_iter()
            .chain(search_roots().into_iter().map(|r| r.join("stanza").join(lang))),
        "vocab.json",
    )
}

/// Whether `dir` holds a granite model.
pub fn is_granite_dir(dir: &Path) -> bool {
    ["config.json", "model.safetensors", "tokenizer.json"]
        .iter()
        .all(|f| dir.join(f).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_are_ordered_and_lookups_need_their_marker_file() {
        let roots = search_roots();
        assert!(!roots.is_empty());
        let raon = raon_models_dir().unwrap();
        assert!(roots.iter().position(|r| *r == raon) < roots.iter().position(|r| r.ends_with("logis-center/models")));
        if let Some(d) = granite_dir() {
            assert!(is_granite_dir(&d), "{}", d.display());
        }
        assert!(stanza_dir("no-such-language").is_none());
    }
}
