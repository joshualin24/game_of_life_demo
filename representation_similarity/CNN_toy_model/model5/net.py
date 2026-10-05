"""Model 5 -- LEARNED canonicalization instead of hand-coded.

Models 1-4 computed the canonicalizing translation/rotation deterministically
from the input (centroid + bounding-box extremes or quadrupole/cubic
moments), giving exact, provable invariance/equivariance guarantees but
zero learning involved in the canonicalization itself. Model 5 asks: can a
network LEARN to produce the same thing?

Two parts, deliberately different in architecture because they need
different receptive fields:

  Part A (`encode`'s backbone):  the same local, D4-equivariant-by-
    construction isotropic-conv stack as Models 1-4 (IsotropicConv2d +
    poly activation). Its raw output `feat` is used AS-IS -- no
    canonicalization is applied inside Part A. Trained so that, once
    canonicalized by SOME pose (teacher's or its own), it reproduces
    Model 4's `feat_can`.
  Part B (`self.pose`, `PosePredictor`): a SEPARATE small conv + global-
    pool + MLP stack that sees the input with two extra coordinate
    channels appended (CoordConv-style). Needs a GLOBAL receptive field --
    unlike Part A, which only ever needs a 3x3 neighborhood to compute the
    GoL rule, predicting "where is the centroid" or "which orientation"
    requires seeing the whole grid.

    v1 of this pooling (plain global SUM over all H*W positions) collapsed:
    gidx_acc/shift_mae were frozen at their epoch-1 values for all 80
    epochs of training, and the 8-way rotation head only ever output 2 of
    8 classes. Root cause, confirmed by direct measurement: the coordinate
    channels are injected at EVERY position, dead or alive, so a plain sum
    over 1600 mostly-dead positions is dominated by that position-only
    background -- for sparse grids, 94-99.8% of the pooled vector's norm
    came from dead cells, not from the actual alive pattern. Fixed here by
    masking the pool to alive cells only and normalizing by alive-cell
    count, mirroring the hand-coded centroid/extent formulas in
    ../model2/net.py, ../model3/net.py, ../model4/net.py (e.g.
    `r_mean = (mask*rows).sum()/cnt`) -- this is a MASKED mean over the
    alive subset, not the whole-grid mean-pooling this project has
    previously warned against (averaging over ALL positions, which would
    dilute position information on a mostly-empty grid). An explicit
    `log1p(alive_count)` feature is concatenated back in afterward, since
    count may still be legitimately useful and this way it's an explicit,
    bounded input rather than an implicit, unbounded one dominating by
    magnitude.

Trained against Model 4 as a FROZEN teacher (see ../model5/train.py) with
three losses: L_pose (Part B's predicted shift/rotation vs. Model 4's
actual computed values -- direct supervision), L_feat (feat_can, built
from Part A's output canonicalized by the current pose, vs. Model 4's
feat_can), and L_task (the real t+1 prediction loss, end to end).

IMPORTANT, unlike Models 1-4: there is no exact equivariance guarantee
here. Part B's pose prediction is a learned, imperfect function -- the
model's symmetry behavior is only as good as Part B learned to make it,
not an architectural certainty. sanity_check.py reflects this: it reports
measured equivariance error instead of asserting it must be ~0.

IsotropicConv2d, PolyActivation, apply_d4_batched, apply_shift_batched are
copied verbatim from ../model4/net.py. D4 group machinery still comes from
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

GRID = 40


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


# ── Part B: learned pose predictor, GLOBAL receptive field ────────────────
class PosePredictor(nn.Module):
    """Predicts the canonicalizing (shift, D4 rotation/reflection) from the
    FULL input grid. Unlike Part A, this must NOT be local-only: deciding
    "where is the centroid" or "which orientation" is a global question.

    Coordinate channels (row-index, col-index, normalized to roughly
    [-0.5, 0.5]) are concatenated to the input so the network has direct
    access to position -- without them, a translation-equivariant conv
    stack has no way to produce a translation-COVARIANT shift prediction
    (it could only ever be translation-invariant via plain pooling, which
    is exactly wrong for a quantity that must track position).

    Pooling is a MASKED mean: `(h * x).sum(spatial) / alive_count`, i.e.
    average pooling restricted to alive cells only, not a plain whole-grid
    mean (see the module docstring above for why a plain sum collapsed
    training, and why this isn't the "mean-pooling trap" this project
    otherwise warns against). `log1p(alive_count)` is concatenated back in
    as an explicit extra feature.
    """

    def __init__(self, grid: int = GRID, channels: int = 16, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.act = nn.GELU()
        self.fc1 = nn.Linear(channels + 1, hidden)
        self.shift_head = nn.Linear(hidden, 2)
        self.gidx_head = nn.Linear(hidden, N_D4)

        rows = (torch.arange(grid).float() / (grid - 1) - 0.5).view(1, 1, grid, 1).expand(1, 1, grid, grid)
        cols = (torch.arange(grid).float() / (grid - 1) - 0.5).view(1, 1, 1, grid).expand(1, 1, grid, grid)
        self.register_buffer("coord_r", rows.clone())
        self.register_buffer("coord_c", cols.clone())

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        coord_r = self.coord_r.expand(B, -1, -1, -1)
        coord_c = self.coord_c.expand(B, -1, -1, -1)
        h = torch.cat([x, coord_r, coord_c], dim=1)
        h = self.act(self.conv1(h))
        h = self.act(self.conv2(h))
        cnt = x.sum(dim=(2, 3)).clamp(min=1.0)          # (B,1) alive-cell count
        pooled = (h * x).sum(dim=(2, 3)) / cnt           # (B,channels) masked MEAN pool, alive cells only
        log_cnt = torch.log1p(cnt)                        # (B,1) explicit, bounded count feature
        pooled = torch.cat([pooled, log_cnt], dim=1)      # (B,channels+1)
        h2 = self.act(self.fc1(pooled))
        shift_pred = self.shift_head(h2)        # (B,2) real-valued
        gidx_logits = self.gidx_head(h2)        # (B,N_D4)
        return shift_pred, gidx_logits


class ToyCNNModel5(nn.Module):
    def __init__(self, m: int = 4, pose_channels: int = 16, pose_hidden: int = 32):
        super().__init__()
        self.m = m
        # Part A: local backbone, same as Models 1-4, no canonicalization inside.
        self.iso1 = IsotropicConv2d(1, 2 * m)
        self.act1 = PolyActivation(2 * m)
        self.conv2 = nn.Conv2d(2 * m, m, kernel_size=1)
        self.act2 = PolyActivation(m)
        self.head = nn.Conv2d(m, 1, kernel_size=1)
        # Part B: global pose predictor.
        self.pose = PosePredictor(channels=pose_channels, hidden=pose_hidden)

    def encode(self, x: torch.Tensor):
        """Part A + Part B, no combination yet -- returns (feat, shift_pred,
        gidx_logits). Exposed separately from `forward` so training can mix
        teacher-forced and self-predicted pose (scheduled sampling) without
        recomputing the backbone twice."""
        feat = self.act1(self.iso1(x))
        feat = self.act2(self.conv2(feat))      # (B,m,H,W) -- Part A output, NOT canonicalized
        shift_pred, gidx_logits = self.pose(x)   # Part B output
        return feat, shift_pred, gidx_logits

    def combine(self, feat: torch.Tensor, shift_used: torch.Tensor, gidx_used: torch.Tensor):
        """Apply a GIVEN pose (shift_used, gidx_used -- teacher's, the
        model's own rounded/argmax prediction, or a scheduled-sampling mix
        of the two) to feat and produce the final t+1 logits. Same
        geometric pipeline as Model 3/4: shift first, then rotate; undo
        rotate first, then unshift."""
        feat_t = apply_shift_batched(feat, shift_used)
        feat_can = apply_d4_batched(feat_t, gidx_used)
        logits_can = self.head(feat_can)
        gidx_inv = _D4_INV_T.to(gidx_used.device)[gidx_used]
        logits_g = apply_d4_batched(logits_can, gidx_inv)
        logits = apply_shift_batched(logits_g, -shift_used)
        return logits, feat_can

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """Standalone inference: uses the model's OWN predicted pose
        (rounded shift, argmax rotation) -- no teacher, no hand-coded
        canonicalization anywhere. This is the actual, fully self-contained
        model behavior, and what should be used for evaluation."""
        feat, shift_pred, gidx_logits = self.encode(x)
        with torch.no_grad():
            shift_used = torch.round(shift_pred).long()
            gidx_used = gidx_logits.argmax(dim=1)
        logits, feat_can = self.combine(feat, shift_used, gidx_used)
        if return_aux:
            return logits, dict(feat=feat, feat_can=feat_can, shift_pred=shift_pred,
                                 gidx_logits=gidx_logits, shift_used=shift_used, gidx_used=gidx_used)
        return logits

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
