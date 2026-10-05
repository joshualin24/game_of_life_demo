"""Model 3 — same backbone as Model 1/2, but BOTH canonicalization steps
changed from Model 2's: translation is now computed once and applied
BEFORE rotation (not a fresh shift per rotation candidate), and the
ROTATION/REFLECTION criterion is replaced:
Model 1/2 picked orientation from the bounding-box EXTREMES (max/min
offset), which are dominated by a single cell and are exactly what made
`canonical_transform`'s argmax flip discontinuously under single-cell
perturbation (see ../model2/notes.md's perturbation-sensitivity section,
and the user's motivating idea for this model). Model 3 instead scores
orientation using the QUADRUPOLE MOMENT (second moment) and a signed cubic
(third) moment -- both SUMS over every alive cell, so a single flipped
cell changes them by a bounded, small increment instead of potentially
replacing which single cell is the extreme.

Quadrupole alone is not enough: as a symmetric rank-2 tensor it's blind to
point inversion and handedness (Q_rr, Q_cc, Q_rc are all invariant under
negating every coordinate, i.e. under r180), so it only pins down
orientation mod 180 degrees -- the same "double angle" blind spot a
principal-axis/PCA decomposition always has. The cubic moment (odd order,
so NOT identically zero around the centroid the way the first moment is by
definition) resolves the remaining sign ambiguity, while still being a sum
over all cells rather than an extreme.

Translation canonicalization is ALSO different from Model 2, per explicit
request: computed ONCE, directly from x, and applied BEFORE rotation is
even decided (Model 2 needed a fresh shift recomputed for each of the 8
rotation candidates -- see `canonical_shift`'s docstring for why that's
no longer necessary here: using round() instead of floor(), and pivoting
on the grid's own rotation center instead of the origin, makes a single
upfront shift exactly consistent with every D4 operation).

Pipeline (shift first, then rotate; undo rotate first, then unshift):
    feat        = poly(IsotropicConv2d(1, 2m))(x)
    feat        = poly(Conv2d(2m, m, k=1))(feat)
    t*(x),g*(x) = canonical_transform(x)     -- shift computed first, rotation scored on the shifted grid
    feat_t      = t*(x) . feat               -- shift first
    feat_can    = g*(x) . feat_t             -- then rotate (the "last layer")
    logits_can  = head(feat_can)
    logits_g    = g*(x)^-1 . logits_can      -- undo rotation first
    logits      = t*(x)^-1 . logits_g        -- then undo shift

IsotropicConv2d, PolyActivation, apply_d4_batched, apply_shift_batched are
copied verbatim from ../model2/net.py (same self-containment convention
used throughout this repo). D4 group machinery still comes from
../../v1/adapter.py.
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


# ── isotropic (D4-symmetric) 3x3 conv [copied verbatim from ../model2/net.py] ─
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
    C = ((H-1)/2, (W-1)/2) -- the fixed point torch_d4's rot90/flip/
    transpose all pivot around (verified earlier: a shape placed exactly at
    C is unaffected by any D4 op). Computed ONCE, directly from x, and
    applied BEFORE rotation -- unlike Model 2's `_centroid_floor_shift`
    (and this model's own first version), which had to be recomputed fresh
    per rotation candidate because it used floor() and targeted the
    origin/wrap-seam. Two changes fix both problems at once:

    1. round(), not floor(). Every D4 operation acts on the centroid-
       relative offset (centroid - C) by pure component-wise NEGATION
       and/or PERMUTATION (e.g. r90 swaps the two components and negates
       one; flip_h negates one component; never any interpolation between
       them). round() with a sign-symmetric tie rule (round-half-to-even,
       torch's default) commutes EXACTLY with negation and permutation
       applied component-wise -- verified directly: round(flip_c(v)) ==
       flip_c(round(v)) for v=(2.3,-1.7), whereas floor(flip_c(v)) !=
       flip_c(floor(v)) (floor is NOT symmetric under negation, which is
       exactly why the old version needed a fresh shift per candidate).
       So shift(g.x) == g.shift(x) exactly here, as long as g pivots about
       the same point C the shift targets.
    2. Target C (grid center), not the origin. C is the actual fixed point
       of the rotations being used, which is what makes point 1 valid --
       round() commuting with negation isn't enough by itself unless the
       rotation and the recentering share a pivot. Targeting C instead of
       the origin also moves the toroidal wrap seam to the far side of the
       grid (C + H/2) instead of placing it exactly where content gets
       recentered to -- the origin IS the wrap seam, which is what caused
       the measurement artifact found in the previous version (compact
       objects measured as split across row/col 0 after recentering, with
       naive non-toroidal distances inflating the moments computed on them).

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


# ── NEW: moment-based (not extreme-based) orientation scoring ─────────────
def _moments(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns (Q_rr, Q_cc, Q_rc, M3_r, M3_c): each
    (B,), the 2nd (quadrupole) and 3rd moments of alive-cell offsets from
    their own centroid. Unlike _extents (Model 1/2), every one of these is
    a SUM over all alive cells -- a single flipped cell changes it by one
    bounded term, not by replacing which cell the statistic even looks at."""
    B, _, H, W = x.shape
    device = x.device
    rows = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    mask = (x[:, 0] > 0.5).float()
    cnt = mask.sum(dim=(1, 2)).clamp(min=1)
    r_mean = (mask * rows).sum(dim=(1, 2)) / cnt
    c_mean = (mask * cols).sum(dim=(1, 2)) / cnt
    r_off = (rows - r_mean.view(B, 1, 1)) * mask
    c_off = (cols - c_mean.view(B, 1, 1)) * mask

    Q_rr = (r_off * r_off).sum(dim=(1, 2))
    Q_cc = (c_off * c_off).sum(dim=(1, 2))
    Q_rc = (r_off * c_off).sum(dim=(1, 2))
    M3_r = (r_off ** 3).sum(dim=(1, 2))
    M3_c = (c_off ** 3).sum(dim=(1, 2))
    return Q_rr, Q_cc, Q_rc, M3_r, M3_c


def canonical_transform(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns (gidx, shift): shift FIRST (computed
    ONCE from x via canonical_shift, pivoting on the grid's own rotation
    center), THEN gidx is chosen by scoring the 8 D4 rotations of the
    ALREADY-SHIFTED grid -- the order explicitly requested: translate the
    distribution's center of mass to canonical position first, then use
    the quadrupole (and cubic) moment to pick rotation/reflection.

    This is safe (exactly consistent across the full translation+D4 orbit)
    specifically because canonical_shift's round()-based, grid-center-
    pivoted shift commutes exactly with every D4 operation -- see that
    function's docstring. That's what makes computing the shift ONCE, up
    front, valid here, unlike the extremes/floor-based version in Model 2
    (and this model's own first attempt), which needed a fresh shift
    recomputed for each of the 8 candidates to avoid an inconsistent result.

    Per-candidate SCORE (evaluated on the 8 rotations of the single shifted
    grid), lexicographic:

        (Q_cc >= Q_rr,     ["wide" by spread, not by extent]
         M3_c >= 0,         ["skews right" by cubic moment, not by extreme]
         M3_r >= 0,         ["skews down" by cubic moment]
         alive-cell coords of the candidate, relative to its own bounding box)  ["exact tiebreak"]

    Q (quadrupole) alone only resolves orientation mod 180 degrees -- it's a
    symmetric rank-2 tensor, invariant under negating every coordinate, so
    it can't distinguish r180-related candidates (same blind spot a PCA/
    principal-axis angle always has). M3_c, M3_r (odd order, so not
    identically zero about the centroid the way the 1st moment always is)
    break that remaining tie. The exact tiebreak is the same final fallback
    as Model 1/2, for shapes with real D4 sub-symmetry where even M3 is
    identically zero by symmetry.
    """
    B = x.shape[0]
    device = x.device
    shift = canonical_shift(x)             # ONCE, from the original x
    x_shifted = apply_shift_batched(x, shift)

    keys_per_g = []
    for g in range(N_D4):
        xg = torch_d4(x_shifted, g)         # rotate the ALREADY-shifted grid
        Q_rr, Q_cc, Q_rc, M3_r, M3_c = _moments(xg)
        wide = (Q_cc >= Q_rr)
        right = (M3_c >= 0)
        down = (M3_r >= 0)
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


class ToyCNNModel3(nn.Module):
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
        # Apply in the order canonical_transform computed them in: shift
        # (to the grid-center pivot) FIRST, then rotate.
        feat_t = apply_shift_batched(feat, shift)
        feat_can = apply_d4_batched(feat_t, gidx)          # (B,m,H,W) -- canonical pose AND position

        # Same rule as Model 1/2: head must be a function of feat_can alone.
        logits_can = self.head(feat_can)

        # Undo in the REVERSE order: unrotate first, then unshift.
        gidx_inv = _D4_INV_T.to(gidx.device)[gidx]
        logits_g = apply_d4_batched(logits_can, gidx_inv)
        logits = apply_shift_batched(logits_g, -shift)

        if return_features:
            return logits, dict(feat=feat, feat_can=feat_can, gidx=gidx, shift=shift)
        return logits

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
