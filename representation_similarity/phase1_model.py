"""Phase 1 — hybrid symmetry adapter on frozen V8 tf_L4.

Components:
  Adapter   z = h(tf_L4)          per-token residual MLP, (B,100,64) -> (B,100,64)
  Decoder   d(z) -> next-state logits (B,1,40,40)   (fresh; mirror of patch_head)
  SymParams learned transformation model on the pooled readout psi = mean_tok(z):
              translations : psi(t.x) - psi(x) ~= delta(t)   (periodic, delta(0)=0)
              D4           : psi(r.x)          ~= rho(r) psi(x)   (7 learned 64x64)

D4 elements are indexed 0..7; the Cayley table is computed from the actual grid
ops so it can't drift from `symmetry.d4_ops`.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from symmetry import d4_ops

D4_NAMES = list(d4_ops().keys())            # ['e','r90','r180','r270','flip_v','flip_h','transp','atransp']
N_D4 = len(D4_NAMES)


# ── D4 Cayley table from the real grid ops ──────────────────────────────────
def _build_cayley() -> np.ndarray:
    ops = d4_ops()
    fns = [ops[n] for n in D4_NAMES]
    probe = np.arange(40 * 40).reshape(1, 40, 40)          # asymmetric marker
    ref = [f(probe) for f in fns]
    table = np.zeros((N_D4, N_D4), dtype=np.int64)
    for i in range(N_D4):
        for j in range(N_D4):
            comp = fns[i](fns[j](probe))                   # apply j then i
            table[i, j] = next(k for k in range(N_D4)
                               if np.array_equal(comp, ref[k]))
    return table

CAYLEY = _build_cayley()                                   # CAYLEY[i,j] = index of (op_i ∘ op_j)


def d4_inverse() -> np.ndarray:
    inv = np.zeros(N_D4, dtype=np.int64)
    for i in range(N_D4):
        inv[i] = next(j for j in range(N_D4) if CAYLEY[i, j] == 0)
    return inv

D4_INV = d4_inverse()


# ── frozen V8 tf_L4 extraction (on-device, no grad) ────────────────────────
@torch.no_grad()
def v8_tfL4(model, x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,40,40) float on model device -> (B,100,64) tf_L4 tokens."""
    m = model
    p = m.patch_size
    B = x.shape[0]
    feat = m.cnn(x)
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    h = w = H // p
    feat = feat.reshape(B, C, h, p, w, p).permute(0, 2, 4, 1, 3, 5).reshape(
        B, m.n_patches, C * p * p)
    tok = m.patch_proj(feat) + m.pos_embed
    for layer in m.transformer.layers:
        tok = layer(tok)
    return tok


# ── Adapter ────────────────────────────────────────────────────────────────
class Adapter(nn.Module):
    def __init__(self, d_model: int = 64, hidden: int = 128, depth: int = 2):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_model)
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                nn.Linear(d_model, hidden), nn.GELU(),
                nn.Linear(hidden, d_model),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:      # (B,100,64)
        z = self.in_norm(tok)
        for blk in self.blocks:
            z = z + blk(z)
        return self.out_norm(z)


# ── Decoder (mirror of patch_head) ─────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, d_model: int = 64, patch: int = 4, grid: int = 40):
        super().__init__()
        self.patch, self.grid = patch, grid
        self.h = grid // patch
        self.head = nn.Linear(d_model, patch * patch)

    def forward(self, z: torch.Tensor) -> torch.Tensor:        # (B,100,64) -> (B,1,40,40)
        B = z.shape[0]
        p, hh = self.patch, self.h
        logits = self.head(z).reshape(B, hh, hh, p, p)
        logits = logits.permute(0, 1, 3, 2, 4).reshape(B, 1, self.grid, self.grid)
        return logits


# ── Symmetry parameters ───────────────────────────────────────────────────
class SymParams(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        # translation displacement: periodic in (dr%4, dc%4), zero at (0,0)
        self.delta_lin = nn.Linear(4, d_model, bias=False)
        nn.init.zeros_(self.delta_lin.weight)
        # D4 operators rho(r) for r != e  (index 1..7); rho(e) = I
        eye = torch.eye(d_model).unsqueeze(0).repeat(N_D4 - 1, 1, 1)
        self.rho_ne = nn.Parameter(eye.clone())

    def _tfeat(self, dr, dc, device):
        a = 2 * math.pi * (torch.as_tensor(dr, device=device, dtype=torch.float) % 4) / 4
        b = 2 * math.pi * (torch.as_tensor(dc, device=device, dtype=torch.float) % 4) / 4
        f = torch.stack([torch.sin(a), torch.cos(a), torch.sin(b), torch.cos(b)], -1)
        f0 = torch.tensor([0., 1., 0., 1.], device=device)
        return f - f0

    def delta(self, dr, dc, device) -> torch.Tensor:
        return self.delta_lin(self._tfeat(dr, dc, device))         # (...,64)

    def rho(self, r_idx: int, device) -> torch.Tensor:
        if r_idx == 0:
            return torch.eye(self.rho_ne.shape[-1], device=device)
        return self.rho_ne[r_idx - 1]

    def rho_all(self, device) -> torch.Tensor:
        I = torch.eye(self.rho_ne.shape[-1], device=device).unsqueeze(0)
        return torch.cat([I, self.rho_ne], 0)                      # (8,64,64)


# ── grid transforms as torch ops (batched, toroidal) ──────────────────────
def torch_translate(x: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    return torch.roll(x, shifts=(dr, dc), dims=(-2, -1))

_D4_TORCH = {
    "e":      lambda x: x,
    "r90":    lambda x: torch.rot90(x, 1, dims=(-2, -1)),
    "r180":   lambda x: torch.rot90(x, 2, dims=(-2, -1)),
    "r270":   lambda x: torch.rot90(x, 3, dims=(-2, -1)),
    "flip_v": lambda x: torch.flip(x, dims=(-2,)),
    "flip_h": lambda x: torch.flip(x, dims=(-1,)),
    "transp": lambda x: x.transpose(-2, -1),
    "atransp": lambda x: torch.flip(x.transpose(-2, -1), dims=(-2, -1)),
}
def torch_d4(x: torch.Tensor, r_idx: int) -> torch.Tensor:
    return _D4_TORCH[D4_NAMES[r_idx]](x)


def gol_step_torch(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,40,40) {0,1} float, toroidal. -> next state, same shape."""
    k = torch.ones(1, 1, 3, 3, device=x.device, dtype=x.dtype)
    k[0, 0, 1, 1] = 0
    xp = F.pad(x, (1, 1, 1, 1), mode="circular")
    n = F.conv2d(xp, k)
    return ((n == 3) | ((x == 1) & (n == 2))).to(x.dtype)


if __name__ == "__main__":
    print("D4 names:", D4_NAMES)
    print("Cayley table:\n", CAYLEY)
    print("inverses:", D4_INV.tolist())
    # sanity: group closure + identity row/col
    assert (CAYLEY[0] == np.arange(N_D4)).all() and (CAYLEY[:, 0] == np.arange(N_D4)).all()
    assert sorted(D4_INV.tolist()) == list(range(N_D4))
    sp = SymParams()
    d0 = sp.delta(torch.tensor([0, 4]), torch.tensor([0, 8]), "cpu")
    print("delta at 4-cell offsets (should be ~0):", d0.abs().max().item())
    print("ok")
