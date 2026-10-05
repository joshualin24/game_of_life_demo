"""Model 4 — a controlled ablation. Model 2 (extremes + fresh shift per
rotation candidate) and Model 3 (quadrupole/cubic moments + single
upfront shift) differ in TWO dimensions at once, so the perturbation
comparison between them was never a clean test of "extremes vs moments"
alone. Model 4 isolates that one variable: it reuses Model 3's translation
mechanism EXACTLY (`canonical_shift` -- shift once, before rotation,
verbatim) but scores rotation/reflection with bounding-box EXTREMES
(Model 1/2's criterion) instead of moments. This is the fair baseline
Model 3 should have been compared against from the start.

Pipeline: identical to Model 3's (shift first, then rotate; undo rotate
first, then unshift) -- see ../model3/net.py's docstring for the full
derivation of why a single upfront shift is valid here (round(), not
floor(); pivoting on the grid's rotation center, not the origin).

    feat        = poly(IsotropicConv2d(1, 2m))(x)
    feat        = poly(Conv2d(2m, m, k=1))(feat)
    t*(x),g*(x) = canonical_transform(x)     -- shift computed first, rotation scored on the shifted grid
    feat_t      = t*(x) . feat               -- shift first
    feat_can    = g*(x) . feat_t             -- then rotate (the "last layer"), scored by EXTREMES
    logits_can  = head(feat_can)
    logits_g    = g*(x)^-1 . logits_can      -- undo rotation first
    logits      = t*(x)^-1 . logits_g        -- then undo shift

IsotropicConv2d, PolyActivation, apply_d4_batched, apply_shift_batched,
canonical_shift are copied verbatim from ../model3/net.py. `_extents` is
copied verbatim from ../model1/net.py (the bounding-box criterion, not
moments). D4 group machinery still comes from ../../v1/adapter.py.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))       # representation_similarity/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))  # v1/
from adapter import D4_NAMES, N_D4, CAYLEY, D4_INV, torch_d4  # noqa: E402

_IDX = {name: i for i, name in enumerate(D4_NAMES)}
_CAYLEY_T = torch.as_tensor(CAYLEY, dtype=torch.long)
_D4_INV_T = torch.as_tensor(D4_INV, dtype=torch.long)


# ── isotropic (D4-symmetric) 3x3 conv [copied verbatim] ───────────────────
class IsotropicConv2d(nn.Module):
    """3x3 circular conv whose kernel has only 3 distinct weights per (in,out)
    pair: center, the 4 orthogonal (edge) neighbors (shared), and the 4
    diagonal (corner) neighbors (shared). Exactly D4-equivariant."""

    _ORTH = [(0, 1), (1, 0), (1, 2), (2, 1)]
    _DIAG = [(0, 0), (0, 2), (2, 0), (2, 2)]

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.w_c = nn.Parameter(torch.empty(out_ch, in_ch))
        self.w_e = nn.Parameter(torch.empty(out_ch, in_ch))
        self.w_d = nn.Parameter(torch.empty(out_ch, in_ch))
        for w in (self.w_c, self.w_e, self.w_d):
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def _kernel(self) -> torch.Tensor:
        k = torch.zeros(self.out_ch, self.in_ch, 3, 3,
                         device=self.w_c.device, dtype=self.w_c.dtype)
        k[:, :, 1, 1] = self.w_c
        for i, j in self._ORTH:
            k[:, :, i, j] = self.w_e
        for i, j in self._DIAG:
            k[:, :, i, j] = self.w_d
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, (1, 1, 1, 1), mode="circular")
        return F.conv2d(xp, self._kernel(), bias=self.bias)


# ── learnable per-channel 2nd-degree polynomial activation [copied verbatim] ──
class PolyActivation(nn.Module):
    """phi(x) = w0 + w1*x + w2*x^2, per channel. Original source:
    ../../../poly_activation_verify/model.py."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.w1 = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.w2 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 + self.w1 * x + self.w2 * x * x


def apply_d4_batched(x: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W), gidx: (B,) long. Applies a per-sample D4 element by
    computing all 8 transforms of the batch and gathering the right one."""
    outs = torch.stack([torch_d4(x, r) for r in range(N_D4)], dim=0)  # (8,B,C,H,W)
    idx = gidx.view(1, -1, 1, 1, 1).expand(1, -1, *x.shape[1:])
    return torch.gather(outs, 0, idx).squeeze(0)


def apply_shift_batched(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W); shift: (B,2) long (dr,dc), possibly different per
    sample. Equivalent to torch.roll(x[b], shifts=shift[b], dims=(-2,-1))
    for each sample b, done via gather -- toroidal."""
    B, C, H, W = x.shape
    device = x.device
    dr, dc = shift[:, 0], shift[:, 1]
    row_idx = (torch.arange(H, device=device).view(1, H, 1) - dr.view(B, 1, 1)) % H
    col_idx = (torch.arange(W, device=device).view(1, 1, W) - dc.view(B, 1, 1)) % W
    row_idx = row_idx.expand(B, H, W).unsqueeze(1).expand(B, C, H, W)
    x = torch.gather(x, 2, row_idx)
    col_idx = col_idx.expand(B, H, W).unsqueeze(1).expand(B, C, H, W)
    x = torch.gather(x, 3, col_idx)
    return x


def canonical_shift(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) in {0,1}. Returns (B,2) long: the ONE integer shift that
    recenters x's alive-cell centroid onto the grid's own rotation pivot
    C = ((H-1)/2, (W-1)/2). Copied verbatim from ../model3/net.py -- see
    that file's docstring for the full derivation of why round() (not
    floor()) pivoting on C (not the origin) makes a single upfront shift
    exactly consistent with every D4 operation: shift(g.x) == g.shift(x)
    exactly, because every D4 op reduces to component-wise negation/
    permutation of the centroid-offset vector, and round() with a
    sign-symmetric tie rule commutes with that exactly.

    Empty grids get shift (0,0).
    """
    with torch.no_grad():
        B, _, H, W = x.shape
        device = x.device
        rows = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
        cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
        mask = x[:, 0] > 0.5
        cnt = mask.sum(dim=(1, 2)).clamp(min=1)
        r_mean = (mask * rows).sum(dim=(1, 2)) / cnt
        c_mean = (mask * cols).sum(dim=(1, 2)) / cnt
        empty = mask.sum(dim=(1, 2)) == 0
        r_mean = torch.where(empty, torch.zeros_like(r_mean), r_mean)
        c_mean = torch.where(empty, torch.zeros_like(c_mean), c_mean)
        center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
        dr = torch.round(center_r - r_mean).long()
        dc = torch.round(center_c - c_mean).long()
        return torch.stack([dr, dc], dim=1)


# ── EXTREME-based orientation scoring [copied verbatim from ../model1/net.py] ─
def _extents(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns x_max, x_min, y_max, y_min: each (B,),
    the column/row offsets of alive cells from their own centroid -- the
    bounding-box criterion Model 1/2 used (contrast with Model 3's
    `_moments`, a sum over all cells instead of the two extreme ones)."""
    B, _, H, W = x.shape
    device = x.device
    rows = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    mask = x[:, 0] > 0.5
    cnt = mask.sum(dim=(1, 2)).clamp(min=1)
    r_mean = (mask * rows).sum(dim=(1, 2)) / cnt
    c_mean = (mask * cols).sum(dim=(1, 2)) / cnt
    r_off = rows - r_mean.view(B, 1, 1)
    c_off = cols - c_mean.view(B, 1, 1)

    big = torch.finfo(torch.float32).max / 2
    y_max = torch.where(mask, r_off, torch.full_like(r_off, -big)).amax(dim=(1, 2))
    y_min = torch.where(mask, r_off, torch.full_like(r_off, big)).amin(dim=(1, 2))
    x_max = torch.where(mask, c_off, torch.full_like(c_off, -big)).amax(dim=(1, 2))
    x_min = torch.where(mask, c_off, torch.full_like(c_off, big)).amin(dim=(1, 2))

    empty = mask.sum(dim=(1, 2)) == 0
    zero = torch.zeros_like(x_max)
    x_max, x_min, y_max, y_min = (torch.where(empty, zero, v) for v in (x_max, x_min, y_max, y_min))
    return x_max, x_min, y_max, y_min


def canonical_transform(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns (gidx, shift): shift FIRST (computed
    ONCE from x via canonical_shift), THEN gidx is chosen by scoring the 8
    D4 rotations of the ALREADY-SHIFTED grid -- identical control flow to
    ../model3/net.py::canonical_transform.

    Per-candidate SCORE, lexicographic (Model 1/2's extreme-based
    criterion, NOT Model 3's moments -- this is the one thing that changed):

        (x_max - x_min >= y_max - y_min,   ["wide" by extent]
         x_max >= |x_min|,                  ["extends right" by extreme]
         y_max >= |y_min|,                  ["extends down" by extreme]
         alive-cell coords of the candidate, relative to its own bounding box)  ["exact tiebreak"]
    """
    B = x.shape[0]
    device = x.device
    shift = canonical_shift(x)             # ONCE, from the original x
    x_shifted = apply_shift_batched(x, shift)

    keys_per_g = []
    for g in range(N_D4):
        xg = torch_d4(x_shifted, g)         # rotate the ALREADY-shifted grid
        x_max, x_min, y_max, y_min = _extents(xg)
        wide = (x_max - x_min >= y_max - y_min)
        right = (x_max >= x_min.abs())
        down = (y_max >= y_min.abs())
        mask_np = (xg[:, 0] > 0.5).cpu().numpy()
        rel_keys = []
        for b in range(B):
            rs, cs = mask_np[b].nonzero()
            if len(rs) == 0:
                rel_keys.append(())
            else:
                r0, c0 = int(rs.min()), int(cs.min())
                rel_keys.append(tuple(sorted(zip((rs - r0).tolist(), (cs - c0).tolist()))))
        keys_per_g.append(list(zip(wide.tolist(), right.tolist(), down.tolist(), rel_keys)))
    gidx = torch.empty(B, dtype=torch.long)
    for b in range(B):
        gidx[b] = max(range(N_D4), key=lambda g: keys_per_g[g][b])
    return gidx.to(device), shift.to(device)


class ToyCNNModel4(nn.Module):
    def __init__(self, m: int = 4):
        super().__init__()
        self.m = m
        self.iso1 = IsotropicConv2d(1, 2 * m)
        self.act1 = PolyActivation(2 * m)
        self.conv2 = nn.Conv2d(2 * m, m, kernel_size=1)
        self.act2 = PolyActivation(m)
        self.head = nn.Conv2d(m, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.act1(self.iso1(x))
        feat = self.act2(self.conv2(feat))               # (B,m,H,W) -- pre-canonicalization

        with torch.no_grad():
            gidx, shift = canonical_transform(x)           # (B,), (B,2); no grad
        feat_t = apply_shift_batched(feat, shift)          # shift first
        feat_can = apply_d4_batched(feat_t, gidx)          # then rotate (the "last layer")

        logits_can = self.head(feat_can)

        gidx_inv = _D4_INV_T.to(gidx.device)[gidx]
        logits_g = apply_d4_batched(logits_can, gidx_inv)   # undo rotation first
        logits = apply_shift_batched(logits_g, -shift)      # then undo shift

        if return_features:
            return logits, dict(feat=feat, feat_can=feat_can, gidx=gidx, shift=shift)
        return logits

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
