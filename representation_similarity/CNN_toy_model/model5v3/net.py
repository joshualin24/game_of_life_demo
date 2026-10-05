"""Model 5v3 -- learned canonicalization, Part A and Part B SHARE a trunk
instead of being fully separate networks.

Model 5 (v1/v2, see ../model5/) built Part A (dynamics) and Part B (pose)
as two completely independent conv stacks, both reading raw `x` but never
sharing a single layer, and gradient-isolated on top of that: `L_task`/
`L_feat` flow back through `combine()` into Part A only (the pose values
`combine()` actually uses are discretized under `no_grad()`), so the
*only* signal Part B ever received was `L_pose`. That was a deliberate
choice to cleanly isolate each part's contribution, and it worked (see
../model5/notes.md) -- but the user asked for something architecturally
different: Part A and Part B "allowed to talk to each other" while
staying "functionally distinct", similar in spirit to Model 4's
(IsotropicConv2d + PolyActivation) backbone.

v3's architecture: a SHARED TRUNK --

    trunk = PolyActivation(2m)(IsotropicConv2d(1, 2m)(x))      # (B,2m,H,W)

-- feeding BOTH functionally distinct heads:

  Dynamics head (`feat`, Part A's role): exactly Model 4's tail --
    PolyActivation(m)(Conv2d(2m,m,1)(trunk)) -- "the last layer of the
    dynamics part" the task requires to reproduce Model 4's feat_can.

  Pose head (`pose`, Part B's role): reads the trunk's OWN features
    (not raw x) plus two appended coordinate channels, then the same
    global conv + masked-mean-pool + MLP design validated in Model 5v2
    (../model5/notes.md's fix for the dead-cell-background collapse) --
    predicts (shift, D4 element).

Why the trunk (not raw x) is shared, and why coordinate channels are
appended ONLY on the pose head's branch, not the trunk itself:
IsotropicConv2d is exactly D4-equivariant BY CONSTRUCTION, for ANY
number of input/output channels, as long as every input channel
transforms the same way under a spatial relabeling that `x` does.
Appending fixed coordinate-index channels to the trunk's OWN input would
break that: `model(g.x)`'s trunk would concatinate `g.x` with the SAME
untransformed coordinate buffers, not `g.(coord_r, coord_c)`, so the
trunk could no longer promise `trunk(g.x) == g.trunk(x)` exactly. Putting
coordinate channels on the pose branch's own input (after the trunk,
where there is no equivariance promise to protect) keeps that one
genuinely-exact architectural guarantee intact while still letting both
heads consume the IDENTICAL shared trunk features.

The actual "talking" this buys, vs. v1/v2: `trunk`'s weights
(`iso1`/`act1`) now receive gradient contributions from BOTH sides --
`L_task`/`L_feat` via the dynamics head (same as Model 4's own
backbone), AND `L_pose` via the pose head (whose conv layers operate
directly on `trunk`'s output with autograd intact, no detaching). The
`round()`/`argmax()` discretization of the pose head's OWN output
remains under `no_grad()` (an unavoidable consequence of those being
non-differentiable ops, not a design choice to isolate) -- but the trunk
itself is now genuinely shaped by both tasks at once, and the two heads
remain cleanly distinguishable: `encode()` returns `feat` (dynamics) and
`(shift_pred, gidx_logits)` (pose) as before, just computed from a common
shared trunk instead of two independent ones.

IsotropicConv2d, PolyActivation, apply_d4_batched, apply_shift_batched
are copied verbatim from ../model5/net.py (originally ../model4/net.py).
D4 group machinery still comes from ../../v1/adapter.py.
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


# ── Part B: pose head, forked from the SHARED TRUNK (not raw x) ───────────
class PoseHead(nn.Module):
    """Predicts (shift, D4 rotation/reflection) from the shared trunk's
    output (`trunk_ch` channels, exactly D4-equivariant by construction)
    plus two appended coordinate channels (needed for the same reason
    ../model5/net.py's PosePredictor needed them -- a translation-
    equivariant conv stack has no other way to access absolute position).

    Pooling: masked MEAN over alive cells only, exactly the ../model5/net.py
    v2 fix (`(h*x).sum(spatial)/count` + `log1p(count)`) -- NOT the plain
    global sum that collapsed in v1 (see ../model5/notes.md).
    """

    def __init__(self, trunk_ch: int, grid: int = GRID, channels: int = 16, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(trunk_ch + 2, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="circular")
        self.act = nn.GELU()
        self.fc1 = nn.Linear(channels + 1, hidden)
        self.shift_head = nn.Linear(hidden, 2)
        self.gidx_head = nn.Linear(hidden, N_D4)

        rows = (torch.arange(grid).float() / (grid - 1) - 0.5).view(1, 1, grid, 1).expand(1, 1, grid, grid)
        cols = (torch.arange(grid).float() / (grid - 1) - 0.5).view(1, 1, 1, grid).expand(1, 1, grid, grid)
        self.register_buffer("coord_r", rows.clone())
        self.register_buffer("coord_c", cols.clone())

    def forward(self, trunk: torch.Tensor, x: torch.Tensor):
        """trunk: (B,trunk_ch,H,W) -- the SHARED trunk's own output, with
        autograd intact (NOT detached: this is how L_pose's gradient
        reaches the trunk). x: (B,1,H,W) -- original input, used only to
        mask the pool to alive cells (never seen directly by this head's
        convs)."""
        B = trunk.shape[0]
        coord_r = self.coord_r.expand(B, -1, -1, -1)
        coord_c = self.coord_c.expand(B, -1, -1, -1)
        h = torch.cat([trunk, coord_r, coord_c], dim=1)
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


class ToyCNNModel5v3(nn.Module):
    def __init__(self, m: int = 4, pose_channels: int = 16, pose_hidden: int = 32):
        super().__init__()
        self.m = m
        # SHARED TRUNK -- exactly Model 4's first layer, consumed by BOTH heads.
        self.iso1 = IsotropicConv2d(1, 2 * m)
        self.act1 = PolyActivation(2 * m)
        # Dynamics head (Part A's role) -- exactly Model 4's tail.
        self.conv2 = nn.Conv2d(2 * m, m, kernel_size=1)
        self.act2 = PolyActivation(m)
        self.head = nn.Conv2d(m, 1, kernel_size=1)
        # Pose head (Part B's role) -- forked from the trunk, not raw x.
        self.pose = PoseHead(trunk_ch=2 * m, channels=pose_channels, hidden=pose_hidden)

    def encode(self, x: torch.Tensor):
        """Shared trunk, then both functionally distinct heads -- returns
        (feat, shift_pred, gidx_logits). `trunk` is NOT detached before
        reaching `self.pose`: this is the whole point of v3 (gradients
        from L_pose reach `iso1`/`act1`, same as L_task/L_feat do via
        `feat`)."""
        trunk = self.act1(self.iso1(x))               # (B,2m,H,W) -- SHARED
        feat = self.act2(self.conv2(trunk))            # (B,m,H,W) -- dynamics head output, NOT canonicalized
        shift_pred, gidx_logits = self.pose(trunk, x)   # pose head output, reads the SAME trunk
        return feat, shift_pred, gidx_logits

    def combine(self, feat: torch.Tensor, shift_used: torch.Tensor, gidx_used: torch.Tensor):
        """Apply a GIVEN pose (teacher's, the model's own rounded/argmax
        prediction, or a scheduled-sampling mix) to feat and produce the
        final t+1 logits. Same geometric pipeline as Models 3/4/5: shift
        first, then rotate; undo rotate first, then unshift."""
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
        canonicalization anywhere."""
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
