//! Client ↔ server: the crate's `McpClient` driving its own `zena-subculture-mcp` binary, and the
//! `AssetService` routing stamps through it.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use serde_json::json;
use zena_avatar_mcp::mesh::{load_glb, pack_glb, split_glb, write_glb};
use zena_avatar_mcp::protocol::{result_json, McpClient};
use zena_avatar_mcp::service::{AssetRequest, AssetService, ServiceConfig};
use zena_avatar_mcp::{forge, AvatarPart, CharacterEmote};

const T: Duration = Duration::from_secs(60);

fn temp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("zena-mcp-test-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn server_exe() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_zena-subculture-mcp"))
}

#[test]
fn client_drives_the_subculture_server() {
    let allow = temp("server");
    let cache = temp("server-cache");
    let mut c = McpClient::spawn(
        server_exe(),
        &["--allow-path".into(), allow.display().to_string(), "--cache-dir".into(), cache.display().to_string()],
    )
    .unwrap();
    assert_eq!(c.server_name(), "zena-subculture-mcp");
    let tools = c.list_tools().unwrap();
    assert_eq!(
        tools,
        [
            "list_emotes",
            "compose_chibi_emote",
            "forge_prop",
            "robot_body",
            "vrm_expressions",
            "emote_morphs",
            "list_avatar_models",
            "fetch_avatar_model"
        ]
    );

    // The catalog: pinned models, nothing cached yet; the built-in kit needs no download.
    let models = result_json(&c.call_tool("list_avatar_models", json!({}), T).unwrap());
    let list = models["models"].as_array().unwrap();
    let default_vrm = list.iter().find(|m| m["kind"] == "vrm" && m["default"] == true).unwrap();
    assert_eq!(default_vrm["id"], "vrm1-twist-sample");
    assert_eq!(default_vrm["cached"], false);
    assert_eq!(default_vrm["files"][0]["sha256"].as_str().unwrap().len(), 64);
    let kits = result_json(&c.call_tool("list_avatar_models", json!({"kind": "chibi_kit"}), T).unwrap());
    assert!(kits["models"].as_array().unwrap().iter().all(|m| m["kind"] == "chibi_kit"));
    assert!(c.call_tool("fetch_avatar_model", json!({"id": "builtin-chibi"}), T).is_err());
    assert!(c.call_tool("fetch_avatar_model", json!({"id": "nope"}), T).is_err());

    let emotes = result_json(&c.call_tool("list_emotes", json!({}), T).unwrap());
    let list = emotes["emotes"].as_array().unwrap();
    assert_eq!(list.len(), 14);
    assert_eq!(list[2]["name"], "COMIC_ANGER_3D");
    assert_eq!(list[2]["form"], "prop");
    assert_eq!(list[9]["comic_morph"], "star_eyes");

    let out = allow.join("stamps");
    let r = c
        .call_tool(
            "compose_chibi_emote",
            json!({"emote": "CHIBI_CHEER", "frames": 4, "out_dir": out.display().to_string()}),
            T,
        )
        .unwrap();
    let v = result_json(&r);
    assert_eq!(v["width"], 640);
    assert_eq!(v["frames"], 4);
    let png = PathBuf::from(v["path"].as_str().unwrap());
    assert!(png.ends_with("chibi_CHIBI_CHEER.png"));
    assert_eq!(image::open(&png).unwrap().width(), 640);

    // Relative out_dir resolves inside the allow path.
    let v = result_json(
        &c.call_tool(
            "forge_prop",
            json!({"emote": "THREAT_ARROW_3D", "angle_deg": 90, "out_dir": "props"}),
            T,
        )
        .unwrap(),
    );
    let glb = PathBuf::from(v["path"].as_str().unwrap());
    assert!(glb.ends_with("prop_THREAT_ARROW_3D_a090.glb"));
    assert_eq!(
        load_glb(&std::fs::read(&glb).unwrap()).unwrap().vertex_count() as u64,
        v["vertices"].as_u64().unwrap()
    );
    let v = result_json(
        &c.call_tool("forge_prop", json!({"part": "cat_ears", "out_dir": "props"}), T)
            .unwrap(),
    );
    assert!(v["path"].as_str().unwrap().ends_with("part_CAT_EARS.glb"));
    let v = result_json(&c.call_tool("robot_body", json!({"out_dir": "body"}), T).unwrap());
    assert_eq!(v["morphs"].as_array().unwrap().len(), 11);

    // Escapes are refused; tool errors come back as Err with the server's message.
    let outside = allow
        .parent()
        .unwrap()
        .join(format!("zena-mcp-test-{}-outside", std::process::id()));
    for bad in [json!("../escape"), json!(outside.display().to_string())] {
        let err = c
            .call_tool("compose_chibi_emote", json!({"emote": "TEARS", "out_dir": bad}), T)
            .unwrap_err();
        assert!(format!("{err:#}").contains("outside the allowed folder"), "{err:#}");
    }
    assert!(!outside.exists());
    assert!(c
        .call_tool("forge_prop", json!({"emote": "CHIBI_CHEER", "out_dir": "props"}), T)
        .is_err());
    assert!(c.call_tool("no_such_tool", json!({}), T).is_err());

    // VRM tools on a synthetic VRM 1.0 model.
    let (mut doc, bin) = split_glb(&write_glb(&forge::robot_zena()))
        .map(|(d, b)| (d, b.unwrap().to_vec()))
        .unwrap();
    doc["extensions"] = json!({"VRMC_vrm": {"specVersion": "1.0",
        "humanoid": {"humanBones": {"head": {"node": 0}}},
        "expressions": {"preset": {"happy": {"morphTargetBinds": [{"node": 0, "index": 1, "weight": 1.0}]}},
                        "custom": {"Namida": {"morphTargetBinds": [{"node": 0, "index": 3, "weight": 1.0}]}}}}});
    let vrm = allow.join("zena.vrm");
    std::fs::write(&vrm, pack_glb(&doc, &bin)).unwrap();
    let v = result_json(
        &c.call_tool("vrm_expressions", json!({"path": vrm.display().to_string()}), T)
            .unwrap(),
    );
    assert_eq!(v["version"], "1.0");
    assert_eq!(v["humanoid"]["head"], 0);
    assert_eq!(v["expressions"].as_array().unwrap().len(), 2);
    let v = result_json(
        &c.call_tool("emote_morphs", json!({"emote": "TEARS", "path": vrm.display().to_string()}), T)
            .unwrap(),
    );
    assert_eq!(v["expression"], "namida");
    let plain = allow.join("plain.glb");
    std::fs::write(&plain, write_glb(&forge::part_mesh(AvatarPart::Glasses))).unwrap();
    assert!(c
        .call_tool("vrm_expressions", json!({"path": plain.display().to_string()}), T)
        .is_err());

    drop(c);
    let _ = std::fs::remove_dir_all(allow);
}

#[test]
fn service_routes_stamps_through_the_server_and_falls_back_without_solidcraft() {
    let cache = temp("service-mcp");
    let mut cfg = ServiceConfig::new(&cache);
    cfg.subculture_mcp = Some(server_exe().to_path_buf());
    cfg.solidcraft = Some(cache.join("solidcraft-cli-not-installed.exe"));
    cfg.stamp_frames = 3;
    let (tx, rx) = mpsc::channel();
    let tx2 = tx.clone();
    let svc = AssetService::spawn(
        cfg,
        move |r| {
            let _ = tx.send(Ok(r));
        },
        move |q, e| {
            let _ = tx2.send(Err((q, e)));
        },
    );
    svc.request(AssetRequest::EmoteStamp(CharacterEmote::Tears));
    svc.request(AssetRequest::Part(AvatarPart::Headset));
    svc.request(AssetRequest::EmoteStamp(CharacterEmote::Tears));
    let mut got = Vec::new();
    for _ in 0..3 {
        got.push(rx.recv_timeout(T).unwrap().unwrap());
    }
    assert_eq!(got[0].via, "zena-subculture-mcp", "{:?}", got[0]);
    assert_eq!(image::open(&got[0].path).unwrap().width(), 3 * 160);
    assert_eq!(got[1].via, "in-process", "SolidCraft is absent, the forge answers");
    assert_eq!(got[2].via, "cache");
    assert_eq!(got[2].path, got[0].path);
    drop(svc);
    let _ = std::fs::remove_dir_all(cache);
}
