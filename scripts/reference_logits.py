#!/usr/bin/env python3
"""Run the OFFICIAL KRAFTON/Raon-1-nano PyTorch implementation (trust_remote_code) in float32 on
the CPU and dump hidden states / logits for the token ids written by `raon-chat --dump`.

    python3 scripts/reference_logits.py MODEL_DIR IDS.json OUT_PREFIX

Memory: the 1.6B per-layer-embedding table stays BF16 (its rows are only gathered; bf16 -> f32
is exact), everything else is cast to float32 — about 9.5 GB of RAM.
Requires: torch, transformers>=5 (the model card's Transformers quickstart versions).
"""
import json, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM

model_dir, ids_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
ids = json.load(open(ids_path))
t0 = time.time()
m = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True)
for name, p in m.named_parameters():
    if not name.endswith("per_layer_inputs.table"):
        p.data = p.data.float()
m.eval()
print(f"loaded in {time.time() - t0:.0f}s; lm_head tied: {m.lm_head.weight is m.model.embed_tokens.weight}", flush=True)
with torch.no_grad():
    t0 = time.time()
    o = m.model(input_ids=torch.tensor([ids]), use_cache=False)
    h = o.last_hidden_state[0].float()
    logits = m.lm_head(h).float()
    print(f"forward {len(ids)} tokens in {time.time() - t0:.1f}s", flush=True)
h.numpy().astype(np.float32).tofile(out + ".hidden.f32")
logits.numpy().astype(np.float32).tofile(out + ".f32")
print("argmax:", logits.argmax(-1).tolist())
