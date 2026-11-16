#!/usr/bin/env python3
"""Reference outputs of the official Raon-VisionEncoder (PyTorch, timm) for `cargo run --example ve_xcheck`.

Writes into OUT_DIR:
  images.json + <name>.rgb      synthetic test screens (raw RGB8) and their sizes
  <name>.<budget>.patches.f32   the official processor's valid patches (cells x 768, channel-major)
  <name>.<budget>.json          grid shape and the image embedding (unit)
  texts.json                    prompts, their token ids and text embeddings (unit)

usage: python ve_reference.py MODEL_DIR TOKENIZER_DIR OUT_DIR
"""
import json, os, sys, importlib
import numpy as np
import torch
from PIL import Image, ImageDraw

model_dir, tok_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, model_dir)
torch.manual_seed(0)


def screens():
    """A few 480x270 'screens' drawn with PIL (no external assets)."""
    W, H = 480, 270
    s = {}
    # stock chart: dark background, candlesticks
    im = Image.new("RGB", (W, H), (18, 20, 28)); d = ImageDraw.Draw(im)
    rng = np.random.default_rng(1); p = 150.0
    for i in range(40):
        o = p; p = p + rng.normal(0, 6); c = p
        x = 20 + i * 11; hi, lo = max(o, c) + rng.uniform(1, 6), min(o, c) - rng.uniform(1, 6)
        col = (220, 60, 60) if c > o else (60, 110, 230)
        d.line([(x + 3, H - hi), (x + 3, H - lo)], fill=col, width=1)
        d.rectangle([x, H - max(o, c), x + 6, H - min(o, c)], fill=col)
    for y in range(30, H, 40): d.line([(0, y), (W, y)], fill=(40, 44, 56))
    s["chart"] = im
    # code editor: dark theme, coloured monospace-ish lines
    im = Image.new("RGB", (W, H), (30, 30, 30)); d = ImageDraw.Draw(im)
    cols = [(86, 156, 214), (206, 145, 120), (78, 201, 176), (220, 220, 170), (156, 220, 254)]
    for i in range(16):
        x = 10 + (i % 4) * 16
        d.text((x, 6 + i * 16), "fn main() { let x = foo(bar, 42); }"[: 18 + (i * 7) % 20], fill=cols[i % 5])
    s["code"] = im
    # victory screen: gold banner text on a bright background
    im = Image.new("RGB", (W, H), (40, 90, 160)); d = ImageDraw.Draw(im)
    d.rectangle([90, 90, 390, 180], fill=(250, 210, 60)); d.text((200, 125), "VICTORY", fill=(120, 40, 0))
    s["victory"] = im
    # smooth gradient (no content)
    a = np.zeros((H, W, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, W)[None, :]; a[..., 1] = np.linspace(0, 255, H)[:, None]; a[..., 2] = 128
    s["gradient"] = Image.fromarray(a)
    return s


from transformers import AutoModel
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).eval().float()
mod = sys.modules[type(model).__module__]
tokmod = importlib.import_module("raon_vision_encoder.tokenizer")
tok = tokmod.HFTokenizer(tok_dir, context_length=64, clean="canonicalize")
proc = mod.RaonVEProcessor(patch_size=16, tokenizer=tok)

meta = {}
for name, im in screens().items():
    a = np.asarray(im, dtype=np.uint8)
    a.tofile(os.path.join(out, f"{name}.rgb"))
    meta[name] = {"w": im.width, "h": im.height}
    for budget in (256, 144):
        inp = proc(images=im, max_num_patches=budget)
        gh, gw = inp["spatial_shapes"][0].tolist()
        n = gh * gw
        inp["pixel_values"][0, :n].numpy().astype(np.float32).tofile(os.path.join(out, f"{name}.{budget}.patches.f32"))
        with torch.no_grad():
            f = model.encode_image(**inp)[0].numpy()
        json.dump({"grid": [gh, gw], "embedding": f.tolist()}, open(os.path.join(out, f"{name}.{budget}.json"), "w"))
json.dump(meta, open(os.path.join(out, "images.json"), "w"))

texts = ["a screenshot of a stock price candlestick chart.", "a computer screen showing source code in a code editor.",
         "a victory screen in a video game", "A Game_Over screen!", "a photo of a cat"]
ids = tok(texts)
with torch.no_grad():
    tf = model.encode_text(ids).numpy()
json.dump({"texts": texts, "ids": ids.tolist(), "embeddings": tf.tolist(),
           "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(model.logit_bias)},
          open(os.path.join(out, "texts.json"), "w"))
print("wrote", out, list(meta))
