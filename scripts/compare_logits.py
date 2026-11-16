#!/usr/bin/env python3
"""Compare dumped logits/hidden states: max |diff|, KL(ref || other), top-1 / top-5 agreement.

    python3 scripts/compare_logits.py REF_PREFIX OTHER_PREFIX [OTHER_PREFIX ...]
"""
import sys
import numpy as np

V, C = 259840, 1536

def load(p):
    lg = np.fromfile(p + ".f32", dtype=np.float32).reshape(-1, V)
    hd = np.fromfile(p + ".hidden.f32", dtype=np.float32).reshape(-1, C)
    return lg, hd

def logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))

ref_l, ref_h = load(sys.argv[1])
lr = logsoftmax(ref_l.astype(np.float64))
for p in sys.argv[2:]:
    l, h = load(p)
    lo = logsoftmax(l.astype(np.float64))
    kl = (np.exp(lr) * (lr - lo)).sum(-1)
    top1 = (ref_l.argmax(-1) == l.argmax(-1)).mean()
    t5r = np.argsort(-ref_l, -1)[:, :5]
    t5o = np.argsort(-l, -1)[:, :5]
    top5 = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(t5r, t5o)])
    hrel = np.linalg.norm(h - ref_h, axis=-1) / np.linalg.norm(ref_h, axis=-1)
    print(f"{p}\n  positions={len(l)}  hidden rel err: mean {hrel.mean():.2e} max {hrel.max():.2e}\n"
          f"  logits max|diff| {np.abs(l - ref_l).max():.4f}  KL(ref||x): mean {kl.mean():.2e} max {kl.max():.2e}\n"
          f"  top-1 agreement {top1 * 100:.1f}%  top-5 overlap {top5 * 100:.1f}%")
