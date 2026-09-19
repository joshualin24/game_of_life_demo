"""Model 1 — isotropic-conv toy CNN with explicit D4 pose canonicalization.

Backbone (exactly D4 + toroidal-translation equivariant by construction):
    feat = poly(IsotropicConv2d(1, 2m))(x)     -- 3x3, circular padding
    feat = poly(Conv2d(2m, m, k=1))(feat)      -- 1x1, pointwise -> trivially equivariant

IsotropicConv2d has only 3 free weights per (in,out) channel pair (center /
shared-orthogonal-neighbor / shared-diagonal-neighbor) instead of a generic
3x3 kernel's 9. Permuting the 8 Moore neighbors by any D4 element permutes
the "orthogonal" and "diagonal" neighbor sets each onto themselves, so this
layer commutes exactly with D4 grid transforms; circular padding gives exact
toroidal-translation equivariance. Composition of equivariant maps is
equivariant, so the whole backbone above is exactly (D4 x translation)
equivariant -- this is what makes the canonicalization step below deliver a
real guarantee (see CNN_toy_model/model1/notes.md for the reasoning and the
generic-kernel counterexample that motivated this).

Pose canonicalization (the "internal layer" nodes from the design discussion):
    g*(x)   = deterministic D4 element from the bounding box of alive cells,
              chosen by argmax over the group orbit (see canonical_gidx).
              No gradient; pure function of x.
    feat_can   = g*(x) . feat        -- canonical-pose features
    logits_can = head(feat_can)      -- shared 1x1 head (head sees ONLY the
                                         canonical-frame features -- see the
                                         note in ToyCNNModel1.forward for why
                                         g*(x) itself must not also be fed
                                         into the head)
    logits     = g*(x)^-1 . logits_can   -- restore original frame

D4 group machinery (D4_NAMES, CAYLEY, D4_INV, torch_d4) is reused from
../../v1/adapter.py so the Cayley table / inverse table can't drift from the
tested implementation there.
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


# ── isotropic (D4-symmetric) 3x3 conv ──────────────────────────────────────
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


# ── learnable per-channel 2nd-degree polynomial activation ────────────────
class PolyActivation(nn.Module):
    """phi(x) = w0 + w1*x + w2*x^2, per channel. Verbatim design from
    ../../../poly_activation_verify/model.py (verified there to reliably
    reach 100% train accuracy on minimal GoL CNNs, unlike ReLU at the same
    param count)."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.w1 = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.w2 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w0 + self.w1 * x + self.w2 * x * x


# ── deterministic pose code from the bounding box of alive cells ──────────
def _extents(x: torch.Tensor):
    """x: (B,1,H,W) in {0,1}. Returns x_max, x_min, y_max, y_min: each (B,),
    the column/row offsets of alive cells from their own centroid. Empty
    grids get all-zero extents (harmless: they tie every candidate below)."""
    B, _, H, W = x.shape
    device = x.device
    rows = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    cols = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    mask = x[:, 0] > 0.5                                   # (B,H,W)
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
    brings x closest to the canonical pose (wide, extends right, extends
    down), chosen as g*(x) = argmax_{g in D4} key(g.x), where key is the
    lexicographic tuple

        (extent_x >= extent_y,   ["wide"]
         x_max >= |x_min|,        ["extends right"]
         y_max >= |y_min|,        ["extends down"]
         alive-cell coords of g.x, relative to their own bounding box)  ["exact tiebreak"]

    The first 3 booleans are the criteria from the design discussion. They
    tie far more often than one might expect -- e.g. any pattern with a
    square bounding box (most small GoL patterns, including the glider) has
    extent_x == extent_y under every D4 element, and two *different*
    elements (one rotation, one reflection) can satisfy all 3 booleans
    identically. Breaking such ties by "first g in a fixed enumeration
    order" does NOT satisfy the equivariant-labeling property (picking
    argmin of an arbitrary index doesn't commute with reparametrizing the
    group orbit) -- that was the bug in the first version of this function.

    The 4th key must ALSO be translation-invariant (relative to the shape's
    own bounding box, not absolute grid position) -- comparing g.x's raw
    absolute pixel content works fine for D4 ties but silently breaks
    translation-equivariance: two placements of the same tied shape at
    different positions could then resolve the tie to different elements,
    since absolute pixel content shifts with position even though the
    "which g resolves the tie" question shouldn't. Using the sorted set of
    alive-cell coordinates relative to their own bounding-box corner fixes
    this (it's still an exact function of g.x alone, so the equivariant-
    labeling property is preserved) and remains fully deterministic and
    collision-free except for x with an actual D4 sub-symmetry (multiple
    g.x identical up to translation) -- in which case any tied choice is
    equally valid.

    Implemented as an explicit per-sample lexicographic comparison (exact,
    no float tie-break fragility) rather than a packed numeric score.
    """
    B = x.shape[0]
    device = x.device
    keys_per_g = []
    for g in range(N_D4):
        xg = torch_d4(x, g)
        x_max, x_min, y_max, y_min = _extents(xg)
        wide = (x_max - x_min >= y_max - y_min)
        right = (x_max >= x_min.abs())
        down = (y_max >= y_min.abs())
        mask_np = (xg[:, 0] > 0.5).cpu().numpy()      # (B,H,W)
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
    """x: (B,C,H,W), gidx: (B,) long in [0, N_D4). Applies a (possibly
    different) D4 element to each sample. Computes all 8 transforms of the
    whole batch (cheap: rot90/flip are index permutations) and gathers the
    per-sample one -- simplest way to vectorize a heterogeneous-per-sample
    group action."""
    outs = torch.stack([torch_d4(x, r) for r in range(N_D4)], dim=0)  # (8,B,C,H,W)
    idx = gidx.view(1, -1, 1, 1, 1).expand(1, -1, *x.shape[1:])
    return torch.gather(outs, 0, idx).squeeze(0)


class ToyCNNModel1(nn.Module):
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
            gidx = canonical_gidx(x)                       # (B,), no grad
            onehot = F.one_hot(gidx, N_D4).float()          # (B,N_D4) -- for analysis only, see note below
        feat_can = apply_d4_batched(feat, gidx)            # (B,m,H,W) -- canonical-pose features

        # NOTE: onehot(gidx) must NOT be fed into `head`. gidx(x) itself is
        # NOT invariant across the D4 orbit (only feat_can is -- that's the
        # whole point of canonicalizing); feeding it into the head would make
        # the head's output depend on which orientation x started in, which
        # breaks the exact-equivariance guarantee (caught by sanity_check.py
        # check 2 during development). The head must be a function of
        # feat_can alone. `onehot`/`gidx` are still returned for analysis
        # (representation-similarity study) via return_features.
        logits_can = self.head(feat_can)

        gidx_inv = _D4_INV_T.to(gidx.device)[gidx]
        logits = apply_d4_batched(logits_can, gidx_inv)

        if return_features:
            return logits, dict(feat=feat, feat_can=feat_can, gidx=gidx, onehot=onehot)
        return logits

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
