"""Verify Model 1's equivariance guarantees BEFORE training. These are meant
to hold at random init (structural/architectural properties, not learned
ones), so failures in the HARD checks mean a bug, not an undertrained model.

HARD checks (must pass -- these are the actual design guarantee):
  A. Full-model exact D4 equivariance:        model(g.x) == g.model(x)
  B. Full-model exact translation equivariance: model(t.x) == t.model(x)
  C. Pose labeling is correct for ASYMMETRIC patterns (glider, lwss):
         CAYLEY[g*(g.x), g] == g*(x)   for every D4 element g
     (canonicalizing g.x and then re-applying g must land on the same
     canonical element as canonicalizing x directly.)

DIAGNOSTIC only (informational, not a failure -- see the printed notes):
  D. Pose-labeling "mismatch rate" on the full (mostly symmetric) named
     pattern set. Patterns with an actual D4 sub-symmetry (block, beehive,
     blinker, pulsar, ...) have MULTIPLE equally-valid canonical choices, so
     no deterministic tiebreak can label them "consistently" across the
     orbit -- this is a mathematical fact about symmetric orbits (like
     there being no continuous choice of square root), not a fixable bug.
  E. Canonical-frame feature invariance feat_can(x) == feat_can(g.x): this
     does NOT hold in general even for asymmetric shapes placed off-center,
     because a D4 grid-transform rotates about the GRID's center, which
     moves an off-center shape to a different absolute position --
     canonicalization here only fixes ORIENTATION, not position. This is
     fine: it does not threaten check A. Full-grid equivariance of the
     model holds regardless of check D/E, because `head` is a 1x1 conv
     (pointwise), which commutes with ANY spatial relabeling of pixels --
     so the canonicalize -> head -> restore sandwich cancels out correctly
     no matter what g*(x) is, consistently labeled or not. (Proved and
     empirically confirmed during development: check A/B passed on batches
     where check D showed >50% "mismatches" from symmetric-pattern ties.)
     Canonicalization would start to matter for correctness the moment the
     head gets real spatial extent (e.g. a k>1 conv) -- worth remembering
     for later models.

Run: python sanity_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel1, canonical_gidx, _CAYLEY_T  # noqa: E402
from data import make_batch, _BITMAPS, GRID, MARGIN  # noqa: E402
from adapter import D4_NAMES, N_D4, torch_translate, torch_d4  # noqa: E402

TOL = 1e-4


def _place(name: str, top: int, left: int) -> torch.Tensor:
    bmp = _BITMAPS[name]
    g = np.zeros((GRID, GRID), dtype=np.float32)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return torch.from_numpy(g).unsqueeze(0).unsqueeze(0)


def check_pose_labeling(x: torch.Tensor, label: str) -> tuple[int, int]:
    """Returns (mismatches, total) over all 8 D4 ops for this batch."""
    g0 = canonical_gidx(x)
    mismatches = 0
    for g in range(N_D4):
        xg = torch_d4(x, g)
        gg = canonical_gidx(xg)
        lhs = _CAYLEY_T[gg, torch.full_like(gg, g)]
        mismatches += (lhs != g0).sum().item()
    total = N_D4 * len(g0)
    print(f"  [{label}] pose-labeling mismatches: {mismatches}/{total}")
    return mismatches, total


def check_model_d4_equivariance(model, x: torch.Tensor) -> bool:
    ok = True
    y0 = model(x)
    for g in range(N_D4):
        xg = torch_d4(x, g)
        yg = model(xg)
        expected = torch_d4(y0, g)
        err = (yg - expected).abs().max().item()
        if err > TOL:
            ok = False
            print(f"  [FAIL] model D4-equivariance under {D4_NAMES[g]}: max err {err:.2e}")
    if ok:
        print(f"  [PASS] model(g.x) == g.model(x) for all 8 D4 ops (tol {TOL:.0e})")
    return ok


def check_model_translation_equivariance(model, x: torch.Tensor) -> bool:
    """Only small shifts are meaningful here: pose_bits' bounding box is NOT
    toroidal-aware, so a shift large enough to wrap a pattern around the torus
    edge breaks canonicalization (a data/placement-margin concern, not a
    model bug -- see data.py's MARGIN). x comes from make_batch with
    MARGIN=4, so shifts up to ~3 stay safe regardless of original placement."""
    ok = True
    y0 = model(x)
    offsets = [(0, 0), (1, 0), (0, 1), (-1, 2), (2, -3), (-3, -3)]
    for dr, dc in offsets:
        xt = torch_translate(x, dr, dc)
        yt = model(xt)
        expected = torch_translate(y0, dr, dc)
        err = (yt - expected).abs().max().item()
        if err > TOL:
            ok = False
            print(f"  [FAIL] model translation-equivariance at ({dr},{dc}): max err {err:.2e}")
    if ok:
        print(f"  [PASS] model(t.x) == t.model(x) for {len(offsets)} shifts (tol {TOL:.0e})")
    return ok


def check_canonical_feature_invariance(model, x: torch.Tensor, label: str):
    """Diagnostic only -- see module docstring (point E)."""
    _, aux0 = model(x, return_features=True)
    feat_can0 = aux0["feat_can"]
    errs = []
    for g in range(N_D4):
        xg = torch_d4(x, g)
        _, auxg = model(xg, return_features=True)
        errs.append((auxg["feat_can"] - feat_can0).abs().max().item())
    print(f"  [{label}] feat_can max abs diff across 8 D4 ops: {max(errs):.3f} "
          f"(diagnostic only, see docstring point E)")


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cpu"   # exactness check: keep it on CPU for deterministic float ops
    x, _ = make_batch(64, rng, device)
    x_asym = torch.cat([_place("glider", 15, 15), _place("lwss", 10, 20)], dim=0)

    model = ToyCNNModel1(m=4).to(device).eval()

    print("HARD checks (must pass)")
    with torch.no_grad():
        print("A) full-model D4 equivariance (random init)")
        okA = check_model_d4_equivariance(model, x)

        print("B) full-model translation equivariance (random init)")
        okB = check_model_translation_equivariance(model, x)

        print("C) pose labeling on ASYMMETRIC patterns only (glider, lwss)")
        bad, total = check_pose_labeling(x_asym, "asymmetric")
        okC = bad == 0
        print(f"  [{'PASS' if okC else 'FAIL'}] expected exactly 0 mismatches for asymmetric shapes")

    print("\nDIAGNOSTIC only (informational, see module docstring)")
    check_pose_labeling(x, "full pattern set, mostly symmetric")
    with torch.no_grad():
        check_canonical_feature_invariance(model, x_asym, "asymmetric")

    all_ok = okA and okB and okC
    print(f"\n{'ALL HARD CHECKS PASSED' if all_ok else 'SOME HARD CHECKS FAILED'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
