#!/usr/bin/env python3
"""Reference outputs of google/siglip2-so400m-patch16-naflex (PyTorch, transformers) for `ve_xcheck`.

The same files `ve_reference.py` writes for Raon-VisionEncoder, so one Rust cross-check reads both:
  images.json + <name>.rgb      the same synthetic test screens (raw RGB8) and their sizes
  <name>.<budget>.patches.f32   the official processor's valid patches (cells x 768), reordered from the
                                processor's (p_h, p_w, c) to the channel-major (c, p_h, p_w) raon-vision uses
  <name>.<budget>.json          grid shape and the image embedding (unit)
  texts.json                    prompts, token ids (the checkpoint's tokenizer after raon-vision's text
                                cleaning), text embeddings (unit), logit scale / bias

The model is built from its config with transformers' Siglip2Model and the checkpoint loaded strictly, so a name
or shape the Rust mapping relies on cannot be silently missing.

usage: python siglip2_reference.py MODEL_DIR OUT_DIR
"""
import html, json, os, re, string, sys
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

model_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def screens():
    """The 480x270 'screens' of ve_reference.py (same drawing code, so the two references share inputs)."""
    W, H = 480, 270
    s = {}
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
    im = Image.new("RGB", (W, H), (30, 30, 30)); d = ImageDraw.Draw(im)
    cols = [(86, 156, 214), (206, 145, 120), (78, 201, 176), (220, 220, 170), (156, 220, 254)]
    for i in range(16):
        x = 10 + (i % 4) * 16
        d.text((x, 6 + i * 16), "fn main() { let x = foo(bar, 42); }"[: 18 + (i * 7) % 20], fill=cols[i % 5])
    s["code"] = im
    im = Image.new("RGB", (W, H), (40, 90, 160)); d = ImageDraw.Draw(im)
    d.rectangle([90, 90, 390, 180], fill=(250, 210, 60)); d.text((200, 125), "VICTORY", fill=(120, 40, 0))
    s["victory"] = im
    a = np.zeros((H, W, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, W)[None, :]; a[..., 1] = np.linspace(0, 255, H)[:, None]; a[..., 2] = 128
    s["gradient"] = Image.fromarray(a)
    return s


def canonicalize(text):
    """raon-vision's `text::canonicalize` (open_clip's): entities, `_` → space, ASCII punctuation out, lowercase."""
    t = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    t = "".join(ch for ch in t.replace("_", " ") if ch not in string.punctuation)
    return " ".join(t.lower().split())


from transformers import Siglip2Config, Siglip2Model, PreTrainedTokenizerFast

cfg = Siglip2Config(
    text_config=dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
                     vocab_size=256000, max_position_embeddings=64, projection_size=1152),
    vision_config=dict(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16,
                       num_channels=3, num_patches=256, patch_size=16),
)
model = Siglip2Model(cfg).eval()
state = load_file(os.path.join(model_dir, "model.safetensors"))
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [m for m in missing if not m.endswith("position_ids")]
assert not missing and not unexpected, (missing[:5], unexpected[:5])
del state

pre_cfg = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
try:
    from transformers import Siglip2ImageProcessorFast as Proc
    proc = Proc(**{k: v for k, v in pre_cfg.items() if k not in ("image_processor_type", "processor_class")})
except Exception as e:  # torchvision missing: the PIL processor
    print("fast processor unavailable:", e)
    from transformers import Siglip2ImageProcessor as Proc
    proc = Proc(**{k: v for k, v in pre_cfg.items() if k not in ("image_processor_type", "processor_class")})
print("processor:", type(proc).__name__)


def unit(v):
    v = np.asarray(v, np.float64)
    return (v / np.linalg.norm(v)).tolist()


def features(x):
    return x.pooler_output if hasattr(x, "pooler_output") else x


meta = {}
for name, im in screens().items():
    a = np.asarray(im, dtype=np.uint8)
    a.tofile(os.path.join(out, f"{name}.rgb"))
    meta[name] = {"w": im.width, "h": im.height}
    for budget in (256, 144):
        inp = proc(images=im, max_num_patches=budget, return_tensors="pt")
        gh, gw = inp["spatial_shapes"][0].tolist()
        n = gh * gw
        pv = inp["pixel_values"][0, :n].numpy().astype(np.float32)  # (n, 768) as (p_h, p_w, c)
        pv.reshape(n, 16, 16, 3).transpose(0, 3, 1, 2).reshape(n, 768).tofile(
            os.path.join(out, f"{name}.{budget}.patches.f32"))
        with torch.no_grad():
            f = features(model.get_image_features(pixel_values=inp["pixel_values"],
                                                  pixel_attention_mask=inp["pixel_attention_mask"],
                                                  spatial_shapes=inp["spatial_shapes"]))[0].numpy()
        json.dump({"grid": [gh, gw], "embedding": unit(f)}, open(os.path.join(out, f"{name}.{budget}.json"), "w"))
json.dump(meta, open(os.path.join(out, "images.json"), "w"))

tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(model_dir, "tokenizer.json"), pad_token="<pad>", eos_token="<eos>")
texts = ["a screenshot of a stock price candlestick chart.", "a computer screen showing source code in a code editor.",
         "a victory screen in a video game", "A Game_Over screen!", "a photo of a cat", "확인 버튼이 있는 로그인 화면"]
enc = tok([canonicalize(t) for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
ids = enc["input_ids"]
with torch.no_grad():
    tf = features(model.get_text_features(input_ids=ids)).numpy()
json.dump({"texts": texts, "ids": ids.tolist(), "embeddings": [unit(v) for v in tf],
           "logit_scale": float(model.logit_scale.exp()), "logit_bias": float(model.logit_bias)},
          open(os.path.join(out, "texts.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", out, list(meta))
