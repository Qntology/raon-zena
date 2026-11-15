#!/usr/bin/env python3
# Usage: python3 -I scripts/xcheck_gguf.py OUT.gguf ORIGINAL.safetensors [MORE.safetensors ...]
# Requires: pip install gguf numpy
"""Cross-check a raon-quant GGUF with llama.cpp's independent gguf-py reader/dequantizer
against the original safetensors values."""
import json, struct, sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

gguf_path, st_path = sys.argv[1], sys.argv[2:]
hdr = {}
for p in st_path:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n)); h.pop("__metadata__", None)
        for k, v in h.items(): hdr[k] = (p, 8 + n, v)

def load(name):
    p, base, v = hdr[name]
    a, b = v["data_offsets"]
    raw = np.memmap(p, dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
    if v["dtype"] == "F32": x = raw.view(np.float32)
    elif v["dtype"] == "BF16": x = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else: x = raw.view(np.float16).astype(np.float32)
    return x.reshape(v["shape"])

r = GGUFReader(gguf_path)
meta = json.loads(bytes(r.fields["raon.tensor_meta"].parts[-1]).decode())
seen = {}
for t in r.tensors:
    ty = t.tensor_type.name
    if ty in seen and seen[ty] >= 3: continue
    m = meta.get(t.name)
    if not m or "src" not in m or m["src"] not in hdr: continue
    src = load(m["src"])
    if src.ndim != 2 and ty not in ("F32", "BF16"): continue
    deq = dequantize(t.data, t.tensor_type).reshape(-1)
    lg = m["logical"]
    if src.ndim == 2 and len(lg) == 2:
        if lg[0] != src.shape[0]:  # row-selected (kept/cold) tensors are skipped here
            continue
        deq = deq.reshape(-1, int(t.shape[0]))[: lg[0], : lg[1]]
    else:
        deq = deq.reshape(-1)[: src.size].reshape(src.shape)
    rel = float(np.linalg.norm(deq - src) / max(np.linalg.norm(src), 1e-30))
    print(f"{ty:5} {t.name:55} gguf-py rel_err={rel:.5f}")
    seen[ty] = seen.get(ty, 0) + 1
