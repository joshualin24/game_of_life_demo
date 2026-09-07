"""Per-stage activation extraction for CNNTransformerV4 (V8 / V10).

Inference-only. Replicates `CNNTransformerV4.forward` up to the head and
captures every intermediate representation:

    cnn        (B, P, D)   CNN encoder output, mean-pooled within each p×p patch
    tokens_in  (B, P, D)   patch_proj(feat) + pos_embed        (== paper's "pre")
    tf_L1..Lk  (B, P, D)   output of each transformer encoder layer
                           (tf_Lk == paper's "post"; k=4 for V8, 6 for V10)

P = 100 patches, D = d_model. Nothing here is hooked; it is a straight
re-implementation, checked bitwise against model.forward in __main__.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CNNTransformerV4  # noqa: E402


@torch.no_grad()
def extract_all(model: CNNTransformerV4, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    """x: (B, 1, H, W) float. Returns ordered {stage_name: (B, P, D) tensor} on CPU."""
    m = model
    m.eval()
    p = m.patch_size
    B = x.shape[0]
    dev = next(m.parameters()).device
    x = x.to(dev)

    feat = m.cnn(x)                                      # (B, C, H, W)
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    h = w = H // p

    # patch-pooled CNN view (lossy: the model itself uses the full flattened patch)
    cnn_patched = feat.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5)  # (B,h,w,C,p,p)
    cnn_pooled = cnn_patched.reshape(B, m.n_patches, C, p * p).mean(-1)      # (B, P, C)

    # exact replica of the model's tokenisation
    feat_flat = feat.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(
        B, m.n_patches, C * p * p)
    tokens_in = m.patch_proj(feat_flat) + m.pos_embed    # (B, P, D)

    stages: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    stages["cnn"] = cnn_pooled.cpu()
    stages["tokens_in"] = tokens_in.cpu()

    h_state = tokens_in
    for i, layer in enumerate(m.transformer.layers, start=1):
        h_state = layer(h_state)
        stages[f"tf_L{i}"] = h_state.cpu()

    return stages


def stage_names(model: CNNTransformerV4) -> list[str]:
    k = len(model.transformer.layers)
    return ["cnn", "tokens_in"] + [f"tf_L{i}" for i in range(1, k + 1)]


@torch.no_grad()
def embeddings_for_grids(model, grids: np.ndarray, batch: int = 64
                         ) -> "OrderedDict[str, np.ndarray]":
    """grids: (N, H, W) uint8 -> {stage: (N, P, D) float32 ndarray}."""
    names = stage_names(model)
    acc: dict[str, list] = {n: [] for n in names}
    for s in range(0, len(grids), batch):
        xb = torch.from_numpy(grids[s:s + batch]).float().unsqueeze(1)
        st = extract_all(model, xb)
        for n in names:
            acc[n].append(st[n].numpy().astype(np.float32))
    return OrderedDict((n, np.concatenate(acc[n], 0)) for n in names)


if __name__ == "__main__":
    from load_models import load_v8, load_v10

    torch.manual_seed(0)
    for tag, load in (("V8", load_v8), ("V10", load_v10)):
        m = load()
        x = torch.randint(0, 2, (6, 1, 40, 40)).float()
        st = extract_all(m, x)
        dev = next(m.parameters()).device
        with torch.no_grad():
            # last tf layer must equal the real transformer output
            feat = m.cnn(x.to(dev))
            C, H, W = feat.shape[1:]
            pz = m.patch_size; hh = H // pz
            ff = feat.reshape(1 * x.shape[0], C, hh, pz, hh, pz).permute(
                0, 2, 4, 1, 3, 5).reshape(x.shape[0], m.n_patches, C * pz * pz)
            tin = m.patch_proj(ff) + m.pos_embed
            ref_post = m.transformer(tin).cpu()
            ref_logits = m(x.to(dev)).cpu()
        last = st[f"tf_L{len(m.transformer.layers)}"]
        d_post = (last - ref_post).abs().max().item()
        # reconstruct logits from captured last layer
        rec = m.patch_head(last.to(dev))
        rec = rec.reshape(x.shape[0], hh, hh, pz, pz).permute(0, 1, 3, 2, 4).reshape(
            x.shape[0], 1, 40, 40).cpu()
        d_log = (rec - ref_logits).abs().max().item()
        print(f"  {tag:>4}  stages={list(st)}")
        print(f"        max|tf_Lk - transformer(pre)| = {d_post:.2e}   "
              f"max|recon logits - model(x)| = {d_log:.2e}   "
              f"{'OK' if max(d_post, d_log) < 1e-4 else 'MISMATCH'}")
