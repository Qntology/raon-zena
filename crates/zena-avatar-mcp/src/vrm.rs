//! Track B-2: VRM 0.x / 1.0 metadata read straight from the GLB JSON chunk.
//!
//! * VRM 0.x — `extensions.VRM.blendShapeMaster.blendShapeGroups[{name, presetName, binds[{mesh, index,
//!   weight 0..100}]}]` and `humanoid.humanBones[{bone, node}]`.
//! * VRM 1.0 — `extensions.VRMC_vrm.expressions.{preset, custom}.<name>.morphTargetBinds[{node, index, weight
//!   0..1}]` (binds reference *nodes*, resolved to meshes via `nodes[node].mesh`) and
//!   `humanoid.humanBones.<bone>.node`.
//!
//! Expression names are lowercase. VRM 0.x presets are normalised to the VRM 1.0 names (`joy` → `happy`,
//! `sorrow` → `sad`, `fun` → `relaxed`, `a/i/u/e/o` → `aa/ih/ou/ee/oh`, `blink_l` → `blinkleft` …) so callers
//! can use one vocabulary; custom groups keep their own (lowercased) names. Bind weights are 0..1.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde_json::Value;
use zena_core::tools::CharacterEmote;

use crate::mesh::MeshData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrmVersion {
    /// VRM 0.x (`extensions.VRM`).
    V0,
    /// VRM 1.0 (`extensions.VRMC_vrm`).
    V1,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VrmExpression {
    /// Lowercase preset (VRM 1.0 vocabulary) or custom name.
    pub name: String,
    /// `(glTF mesh, morph target index, weight 0..1)`.
    pub binds: Vec<(usize, usize, f32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VrmInfo {
    pub version: VrmVersion,
    pub expressions: Vec<VrmExpression>,
    /// Humanoid bone name → node index.
    pub humanoid: Vec<(String, usize)>,
}

/// VRM 0.x preset → VRM 1.0 expression name.
fn v0_preset(p: &str) -> String {
    let p = p.to_ascii_lowercase();
    match p.as_str() {
        "a" => "aa",
        "i" => "ih",
        "u" => "ou",
        "e" => "ee",
        "o" => "oh",
        "joy" => "happy",
        "sorrow" => "sad",
        "fun" => "relaxed",
        "blink_l" => "blinkleft",
        "blink_r" => "blinkright",
        other => other,
    }
    .to_string()
}

fn idx(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

impl VrmInfo {
    /// Parses a `.vrm` / `.glb` (or `.gltf` JSON); `Ok(None)` for plain glTF without a VRM extension.
    pub fn from_glb(bytes: &[u8]) -> Result<Option<VrmInfo>> {
        let doc = if bytes.starts_with(b"glTF") {
            crate::mesh::split_glb(bytes)?.0
        } else {
            serde_json::from_slice(bytes).context("neither GLB nor glTF JSON")?
        };
        Ok(VrmInfo::from_json(&doc))
    }

    /// Reads the VRM extension of a parsed glTF document.
    pub fn from_json(doc: &Value) -> Option<VrmInfo> {
        if let Some(v1) = doc.pointer("/extensions/VRMC_vrm") {
            let node_mesh = |node: usize| {
                doc.pointer(&format!("/nodes/{node}/mesh"))
                    .and_then(Value::as_u64)
                    .map(|m| m as usize)
            };
            let mut expressions = Vec::new();
            for group in ["preset", "custom"] {
                let Some(map) = v1.pointer(&format!("/expressions/{group}")).and_then(Value::as_object) else {
                    continue;
                };
                for (name, e) in map {
                    let binds = e
                        .get("morphTargetBinds")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter_map(|b| {
                            let mesh = node_mesh(idx(b, "node")?)?;
                            let w = b.get("weight").and_then(Value::as_f64).unwrap_or(1.0) as f32;
                            Some((mesh, idx(b, "index")?, w.clamp(0.0, 1.0)))
                        })
                        .collect();
                    expressions.push(VrmExpression {
                        name: name.to_ascii_lowercase(),
                        binds,
                    });
                }
            }
            let humanoid = v1
                .pointer("/humanoid/humanBones")
                .and_then(Value::as_object)
                .into_iter()
                .flatten()
                .filter_map(|(bone, b)| Some((bone.clone(), idx(b, "node")?)))
                .collect();
            return Some(VrmInfo {
                version: VrmVersion::V1,
                expressions,
                humanoid,
            });
        }
        let v0 = doc.pointer("/extensions/VRM")?;
        let expressions = v0
            .pointer("/blendShapeMaster/blendShapeGroups")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|g| {
                let preset = g.get("presetName").and_then(Value::as_str).unwrap_or("");
                let name = if preset.is_empty() || preset.eq_ignore_ascii_case("unknown") {
                    g.get("name").and_then(Value::as_str).unwrap_or("").to_ascii_lowercase()
                } else {
                    v0_preset(preset)
                };
                let binds = g
                    .get("binds")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|b| {
                        let w = b.get("weight").and_then(Value::as_f64).unwrap_or(100.0) as f32 / 100.0;
                        Some((idx(b, "mesh")?, idx(b, "index")?, w.clamp(0.0, 1.0)))
                    })
                    .collect();
                VrmExpression { name, binds }
            })
            .collect();
        let humanoid = v0
            .pointer("/humanoid/humanBones")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|b| Some((b.get("bone")?.as_str()?.to_string(), idx(b, "node")?)))
            .collect();
        Some(VrmInfo {
            version: VrmVersion::V0,
            expressions,
            humanoid,
        })
    }

    /// Expression by name (case-insensitive; VRM 0.x preset names are accepted too).
    pub fn expression(&self, name: &str) -> Option<&VrmExpression> {
        let lower = name.to_ascii_lowercase();
        let alias = v0_preset(&lower);
        self.expressions
            .iter()
            .find(|e| e.name == lower)
            .or_else(|| self.expressions.iter().find(|e| e.name == alias))
    }

    /// Node of a humanoid bone (case-insensitive, e.g. `head`, `leftUpperArm`).
    pub fn bone_node(&self, bone: &str) -> Option<usize> {
        self.humanoid.iter().find(|(b, _)| b.eq_ignore_ascii_case(bone)).map(|(_, n)| *n)
    }
}

fn synonyms(comic: &str) -> &'static [&'static str] {
    match comic {
        "heart_eyes" => &["heart_eyes", "hearteyes", "heart", "love"],
        "star_eyes" => &["star_eyes", "stareyes", "star", "kirakira"],
        "dizzy_spiral" => &["dizzy_spiral", "dizzy", "guruguru", "spiral", "@_@"],
        "tears" => &["tears", "cry", "namida"],
        "angry_mark" => &["angry_mark", "anger", "ikari"],
        "sweat" => &["sweat", "ase"],
        _ => &[],
    }
}

/// The model's expression for an emote's comic morph ([`CharacterEmote::comic_morph`]): an exact
/// (case-insensitive) synonym match first, then a substring match — short alphanumeric synonyms (≤ 3 chars,
/// e.g. `ase`, `cry`) must match a whole `_`/`-`/space-separated token so `base` or `crystal` do not count.
pub fn emote_expression(info: &VrmInfo, emote: CharacterEmote) -> Option<String> {
    let comic = emote.comic_morph()?;
    let syn = synonyms(comic);
    for s in syn {
        if let Some(e) = info.expressions.iter().find(|e| e.name.eq_ignore_ascii_case(s)) {
            return Some(e.name.clone());
        }
    }
    for s in syn {
        let short = s.len() <= 3 && s.chars().all(|c| c.is_ascii_alphanumeric());
        let hit = info.expressions.iter().find(|e| {
            let n = e.name.to_ascii_lowercase();
            if short {
                n.split(|c: char| !c.is_ascii_alphanumeric()).any(|t| t == *s)
            } else {
                n.contains(s)
            }
        });
        if let Some(e) = hit {
            return Some(e.name.clone());
        }
    }
    None
}

/// Maps expression weights to `(index into mesh.morphs, weight)` by matching each bind's glTF mesh / target
/// against [`crate::mesh::MorphTarget::mesh`] / `target`; contributions are summed and clamped to 0..1,
/// zero results dropped, sorted by morph index.
pub fn expression_morphs(info: &VrmInfo, mesh: &MeshData, weights: &[(&str, f32)]) -> Vec<(usize, f32)> {
    let mut acc: BTreeMap<usize, f32> = BTreeMap::new();
    for (name, w) in weights {
        let Some(e) = info.expression(name) else { continue };
        for &(m, t, bw) in &e.binds {
            for (k, morph) in mesh.morphs.iter().enumerate() {
                if morph.mesh == m && morph.target == t {
                    *acc.entry(k).or_insert(0.0) += w * bw;
                }
            }
        }
    }
    acc.into_iter()
        .map(|(k, v)| (k, v.clamp(0.0, 1.0)))
        .filter(|(_, v)| *v > 0.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{load_glb, pack_glb, split_glb, write_glb};
    use serde_json::json;

    /// The robot (one glTF mesh, 11 morph targets) re-packed with a VRM extension.
    fn synthetic_vrm(ext_name: &str, ext: Value) -> Vec<u8> {
        let glb = write_glb(&crate::forge::robot_zena());
        let (mut doc, bin) = split_glb(&glb).unwrap();
        let bin = bin.unwrap().to_vec();
        doc["extensionsUsed"] = json!([ext_name]);
        doc["extensions"] = json!({ ext_name: ext });
        pack_glb(&doc, &bin)
    }

    #[test]
    fn vrm0_blend_shapes_and_humanoid() {
        let bytes = synthetic_vrm(
            "VRM",
            json!({
                "blendShapeMaster": {"blendShapeGroups": [
                    {"name": "Joy", "presetName": "joy", "binds": [{"mesh": 0, "index": 1, "weight": 100}]},
                    {"name": "A", "presetName": "a", "binds": [{"mesh": 0, "index": 6, "weight": 100}]},
                    {"name": "HeartEyes", "presetName": "unknown", "binds": [{"mesh": 0, "index": 0, "weight": 50}]},
                    {"name": "Base_Ase", "presetName": "unknown", "binds": []},
                    {"name": "Release", "presetName": "unknown", "binds": []}
                ]},
                "humanoid": {"humanBones": [{"bone": "head", "node": 0}, {"bone": "hips", "node": 0}]}
            }),
        );
        let info = VrmInfo::from_glb(&bytes).unwrap().unwrap();
        assert_eq!(info.version, VrmVersion::V0);
        let names: Vec<&str> = info.expressions.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["happy", "aa", "hearteyes", "base_ase", "release"]);
        assert_eq!(info.bone_node("Head"), Some(0));
        assert_eq!(info.expression("JOY").unwrap().name, "happy", "0.x preset names still resolve");
        assert_eq!(emote_expression(&info, CharacterEmote::HeartEyes).as_deref(), Some("hearteyes"));
        assert_eq!(
            emote_expression(&info, CharacterEmote::SweatPanic).as_deref(),
            Some("base_ase"),
            "token match"
        );
        assert_eq!(emote_expression(&info, CharacterEmote::StarEyes), None);
        assert_eq!(emote_expression(&info, CharacterEmote::ChibiCheer), None, "no comic morph");

        let mesh = load_glb(&bytes).unwrap();
        assert_eq!(mesh.morphs.len(), 11);
        let w = expression_morphs(&info, &mesh, &[("happy", 1.0), ("HeartEyes", 1.0), ("missing", 1.0)]);
        assert_eq!(w, vec![(0, 0.5), (1, 1.0)]);
        assert_eq!(mesh.morphs[1].name, "happy");
    }

    #[test]
    fn vrm1_expressions_resolve_nodes_to_meshes() {
        let bytes = synthetic_vrm(
            "VRMC_vrm",
            json!({
                "specVersion": "1.0",
                "humanoid": {"humanBones": {"head": {"node": 0}, "neck": {"node": 0}}},
                "expressions": {
                    "preset": {
                        "happy": {"morphTargetBinds": [{"node": 0, "index": 1, "weight": 1.0}]},
                        "blink": {"morphTargetBinds": [{"node": 0, "index": 0, "weight": 1.0}, {"node": 7, "index": 0, "weight": 1.0}]}
                    },
                    "custom": {"Star_Eyes": {"morphTargetBinds": [{"node": 0, "index": 2, "weight": 0.8}]}}
                }
            }),
        );
        let info = VrmInfo::from_glb(&bytes).unwrap().unwrap();
        assert_eq!(info.version, VrmVersion::V1);
        assert_eq!(info.humanoid.len(), 2);
        assert_eq!(
            info.expression("blink").unwrap().binds,
            vec![(0, 0, 1.0)],
            "a bind to a missing node is dropped"
        );
        assert_eq!(emote_expression(&info, CharacterEmote::StarEyes).as_deref(), Some("star_eyes"));
        let mesh = load_glb(&bytes).unwrap();
        let w = expression_morphs(&info, &mesh, &[("star_eyes", 1.0), ("happy", 0.5)]);
        assert_eq!(w, vec![(1, 0.5), (2, 0.8)]);
        let w = expression_morphs(&info, &mesh, &[("happy", 0.8), ("happy", 0.8)]);
        assert_eq!(w, vec![(1, 1.0)], "summed and clamped");
    }

    #[test]
    fn plain_gltf_is_not_a_vrm() {
        let glb = write_glb(&crate::forge::part_mesh(zena_core::tools::AvatarPart::Glasses));
        assert_eq!(VrmInfo::from_glb(&glb).unwrap(), None);
        assert!(VrmInfo::from_glb(b"garbage").is_err());
    }
}
