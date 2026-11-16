//! # zena-avatar-mcp
//!
//! Headless dual-track avatar asset pipeline for Raon Zena — no CAD or editor UI, everything is driven
//! through MCP (JSON-RPC 2.0 over stdio) or called in-process:
//!
//! * [`protocol`]   – a minimal MCP client ([`protocol::McpClient`]) and server loop ([`protocol::serve`]).
//! * [`mesh`]       – [`mesh::MeshData`] (what the overlay uploads to wgpu), a glTF 2.0 binary writer, a
//!   `.glb` / `.vrm` reader and skinning math.
//! * [`forge`]      – Track A built-in parametric generator: 3D comic emote props, accessory parts and the
//!   SD toy-robot Zena with VRM-named morph targets.
//! * [`solidcraft`] – Track A bridge to SolidCraft's MCP server (`solidcraft-cli mcp`), falling back to [`forge`].
//! * [`chibi`]      – Track B-1 SD (chibi) stamp compositor from layered PNG parts, with a procedural kit.
//! * [`vrm`]        – Track B-2 VRM 0.x / 1.0 expression and humanoid metadata.
//! * [`service`]    – [`service::AssetService`], the background thread the overlay app asks for assets.
//!
//! The `zena-subculture-mcp` binary exposes [`chibi`], [`forge`] and [`vrm`] as MCP tools.

pub mod chibi;
pub mod forge;
mod math;
pub mod mesh;
pub mod protocol;
pub mod service;
pub mod solidcraft;
pub mod vrm;

pub use zena_core::tools::{AvatarPart, AvatarTrack, CharacterEmote, EmoteForm};
