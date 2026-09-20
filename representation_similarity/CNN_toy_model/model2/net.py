"""Model 2 — Model 1's backbone, extended with JOINT translation + D4
canonicalization (Model 1 only canonicalized D4).

Motivation (see ../model1/notes.md and the design discussion that led here):
Model 1's D4-only canonicalization gives an exactly invariant `feat_can` for
ASYMMETRIC patterns, but leaves a real residual gap for patterns with a
genuine D4 sub-symmetry (block, pulsar, ...): when such a shape sits off
the grid's own rotation center, rotating the whole grid moves it even
though its local shape looks unchanged, and D4 canonicalization alone has
nothing to detect/correct (it only ever sees "already canonical" for a
symmetric shape, so it applies identity every time, leaving the position
shift uncorrected). Adding explicit translation canonicalization fixes
this at the root, and -- unlike D4 -- translation has NO discrete-tie
ambiguity: a shape's centroid is a single real point, not one of 8
discrete candidates that a symmetric shape can make indistinguishable.

IMPORTANT, found via testing (not obvious up front): translation and D4
corrections can NOT be computed independently from x and then composed --
floor(centroid) does not commute with rotation (a reflection flips a
coordinate's sign, so "floor after rotating" isn't "rotate the floored
value"), so an independently-chosen shift and rotation don't compose to an
exactly invariant result even for ASYMMETRIC shapes (this broke glider/
lwss, which Model 1 got exactly right, during development). The fix,
`canonical_transform`: for each of the 8 D4 candidates, compute ITS OWN
recentering shift fresh, form the full (rotate-then-recenter) candidate,
and pick the winner by scoring those -- see its docstring for why this is
exactly correct where the naive version wasn't.

Pipeline:
    feat        = poly(IsotropicConv2d(1, 2m))(x)
    feat        = poly(Conv2d(2m, m, k=1))(feat)
    g*(x),t*(x) = canonical_transform(x)     -- joint D4 + translation choice
    feat_g      = g*(x) . feat               -- rotate first
    feat_can    = t*(x) . feat_g             -- then recenter (the "last layer")
    logits_can  = head(feat_can)
    logits_g    = t*(x)^-1 . logits_can      -- undo translation first
    logits      = g*(x)^-1 . logits_g        -- then undo rotation (reverse order)

IsotropicConv2d, PolyActivation, `_extents`, `canonical_gidx`, and
`apply_d4_batched` are copied verbatim from ../model1/net.py (same
self-containment convention used throughout this repo -- see e.g. that
file copying PolyActivation from ../../../poly_activation_verify/model.py).
D4 group machinery (D4_NAMES, CAYLEY, D4_INV, torch_d4) is still reused
from ../../v1/adapter.py directly, same as Model 1.
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


# ── isotropic (D4-symmetric) 3x3 conv [copied verbatim from ../model1/net.py] ─
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


# ── D4 pose canonicalization [copied verbatim from ../model1/net.py] ─────────
def _extents(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns x_max, x_min, y_max, y_min: each (B,),
    the column/row offsets of alive cells from their own centroid."""
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


def canonical_gidx(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) in {0,1}. Returns (B,) long: the D4 element g*(x) that
    brings x closest to the canonical pose, chosen as an argmax over the
    group orbit with an exact, translation-invariant tiebreak. See
    ../model1/net.py::canonical_gidx for the full derivation and the two
    bugs (fixed-order composition; position-dependent tiebreak) this design
    avoids."""
    B = x.shape[0]
    device = x.device
    keys_per_g = []
    for g in range(N_D4):
        xg = torch_d4(x, g)
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
    return gidx.to(device)


def apply_d4_batched(x: torch.Tensor, gidx: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W), gidx: (B,) long. Applies a per-sample D4 element by
    computing all 8 transforms of the batch and gathering the right one."""
    outs = torch.stack([torch_d4(x, r) for r in range(N_D4)], dim=0)  # (8,B,C,H,W)
    idx = gidx.view(1, -1, 1, 1, 1).expand(1, -1, *x.shape[1:])
    return torch.gather(outs, 0, idx).squeeze(0)


# ── NEW: joint translation + D4 canonicalization ──────────────────────────
def _centroid_floor_shift(x: torch.Tensor) -> torch.Tensor:
    """x: (B,1,H,W) in {0,1}. Returns (B,2) long: -floor(centroid(x)), the
    shift that recenters x's alive-cell centroid near the grid origin.
    Empty grids get shift (0,0). This alone is exactly translation-
    consistent (floor(centroid+d) == floor(centroid)+d for integer d), but
    -- important, and the source of a real bug caught by testing -- it is
    NOT rotation-consistent: floor does not commute with the sign flips a
    reflection introduces, so computing this once on the original x and
    composing it with a separately-chosen rotation does NOT give an exactly
    invariant result (verified: broke even glider/lwss, which Model 1 got
    exactly right). canonical_transform below computes this fresh for each
    of the 8 rotated candidates instead of composing a single upfront value,
    which avoids the problem entirely -- see its docstring.
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
        dr = -torch.floor(r_mean).long()
        dc = -torch.floor(c_mean).long()
        return torch.stack([dr, dc], dim=1)


def canonical_transform(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns (gidx, shift): the D4 element and the
    (dr,dc) integer translation that JOINTLY canonicalize x, to be applied
    as "rotate by gidx, THEN shift by shift" (and undone in the reverse
    order: unshift, then unrotate).

    Computed as an argmax over 8 CANDIDATES, one per D4 element g: form
    x_g = g.x, then recenter IT with its own _centroid_floor_shift (fresh,
    not reused from x) to get the full candidate g_shift.(g.x); score that
    full candidate with the same wide/right/down + exact-relative-
    coordinate-tiebreak rule canonical_gidx used, and take the g that wins.

    This is the fix for the bug in the first version of this function
    (computing a shift once from x and a rotation once from x, independently,
    then composing them): floor(centroid) does not commute with rotation
    (reflections flip coordinate signs, so "floor after rotating" isn't
    "rotate the floored value"), so composing two independently-chosen
    corrections isn't exactly invariant, even for asymmetric shapes.
    Recomputing the shift FRESH for each candidate, then choosing among the
    already-fully-formed candidates, sidesteps the commutation question
    entirely: the 8 full candidates for x and for g0.x are the exact same
    SET (just relabeled), so the argmax picks the identical winning
    candidate either way -- the same "argmax over the orbit" principle
    that made canonical_gidx correct, now over the richer orbit.
    """
    B = x.shape[0]
    device = x.device
    keys_per_g, shifts_per_g = [], []
    for g in range(N_D4):
        xg = torch_d4(x, g)
        shift_g = _centroid_floor_shift(xg)                # (B,2) -- fresh, not composed
        xg_full = apply_shift_batched(xg, shift_g)          # the FULL candidate: rotate then recenter
        x_max, x_min, y_max, y_min = _extents(xg_full)
        wide = (x_max - x_min >= y_max - y_min)
        right = (x_max >= x_min.abs())
        down = (y_max >= y_min.abs())
        mask_np = (xg_full[:, 0] > 0.5).cpu().numpy()
        rel_keys = []
        for b in range(B):
            rs, cs = mask_np[b].nonzero()
            if len(rs) == 0:
                rel_keys.append(())
            else:
                r0, c0 = int(rs.min()), int(cs.min())
                rel_keys.append(tuple(sorted(zip((rs - r0).tolist(), (cs - c0).tolist()))))
        keys_per_g.append(list(zip(wide.tolist(), right.tolist(), down.tolist(), rel_keys)))
        shifts_per_g.append(shift_g)
    gidx = torch.empty(B, dtype=torch.long)
    shift = torch.empty(B, 2, dtype=torch.long)
    for b in range(B):
        best_g = max(range(N_D4), key=lambda g: keys_per_g[g][b])
        gidx[b] = best_g
        shift[b] = shifts_per_g[best_g][b]
    return gidx.to(device), shift.to(device)


def apply_shift_batched(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """x: (B,C,H,W); shift: (B,2) long (dr,dc), possibly different per
    sample. Equivalent to torch.roll(x[b], shifts=shift[b], dims=(-2,-1))
    for each sample b, done via gather (torch.roll has no per-sample-shift
    form) -- toroidal, matching the model's circular padding everywhere
    else."""
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


class ToyCNNModel2(nn.Module):
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
        # Apply in the SAME order the candidates were built in
        # canonical_transform: rotate first, then recenter.
        feat_g = apply_d4_batched(feat, gidx)
        feat_can = apply_shift_batched(feat_g, shift)      # (B,m,H,W) -- canonical pose AND position

        # Same rule as Model 1: head must be a function of feat_can alone.
        # shift/gidx are not invariant across the orbit (only feat_can is),
        # so feeding them into the head would reintroduce the exact bug
        # Model 1's sanity_check.py caught (head output depending on which
        # frame x started in).
        logits_can = self.head(feat_can)

        # Undo in the REVERSE order: unshift, then unrotate.
        gidx_inv = _D4_INV_T.to(gidx.device)[gidx]
        logits_g = apply_shift_batched(logits_can, -shift)
        logits = apply_d4_batched(logits_g, gidx_inv)

        if return_features:
            return logits, dict(feat=feat, feat_can=feat_can, gidx=gidx, shift=shift)
        return logits

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
