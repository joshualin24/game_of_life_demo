"""Verify Model 4's guarantees BEFORE training -- these are architectural
properties (must hold at random init), not learned ones.

HARD checks (must pass):
  A. Full-model exact D4 equivariance:         model(g.x) == g.model(x)
  B. Full-model exact translation equivariance: model(t.x) == t.model(x)
     (both A and B hold regardless of canonical_transform's correctness --
     same "1x1 head commutes with any spatial relabeling" argument as
     Model 1; see ../model1/sanity_check.py's docstring point E.)
  C. canonical_shift's equivariant labeling: shift(g.x) == g.shift(x)
     EXACTLY, for every D4 element g -- same property Model 3 established,
     reused verbatim here (`canonical_shift` is copied unchanged from
     ../model3/net.py). Zero tolerance: round() pivoting on the grid's own
     rotation center commutes with D4 exactly (each D4 op only negates/
     permutes the centroid-offset's two components, and a sign-symmetric
     round() commutes with negation and permutation individually), not
     approximately.
  D. canonical_transform's combined consistency on ASYMMETRIC patterns
     (glider, lwss): canonicalizing x under ANY combination of a D4
     transform and translation must land on the exact same grid, bit-for-
     bit. Tested directly (construct the actual canonicalized grid and
     compare) rather than via a Cayley-table relation, since the group
     involved (translations x D4) isn't a small finite table the way D4
     alone is.

THE ACTUAL POINT of Model 4 (not a hard gate): the controlled ablation.
Model 4 = Model 3's translation mechanism (shift once, before rotation)
+ Model 1/2's extreme-based rotation scoring -- isolating whether Model 3's
mixed perturbation results (pulsar improved, random_d0.1 regressed; see
../model3/notes.md) come from the SCORING CRITERION (moments vs extremes)
or from the shift-ordering change, since Model 2 and Model 3 differed in
both dimensions at once. See analysis_perturbation.py for the direct
3-way comparison (Model 2 / Model 3 / Model 4). This script only
re-confirms exact D4 invariance still holds for the previously-broken
symmetric patterns, same as Model 2 and Model 3.

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
from net import (ToyCNNModel4, canonical_transform, canonical_shift,  # noqa: E402
                 apply_d4_batched, apply_shift_batched)
from data import make_batch, _BITMAPS, GRID  # noqa: E402
from adapter import D4_NAMES, N_D4, torch_translate, torch_d4  # noqa: E402

TOL = 1e-4


def _place(name: str, top: int, left: int) -> torch.Tensor:
    bmp = _BITMAPS[name]
    g = np.zeros((GRID, GRID), dtype=np.float32)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return torch.from_numpy(g).unsqueeze(0).unsqueeze(0)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def canonicalize_grid(x: torch.Tensor) -> torch.Tensor:
    """The actual canonicalized grid (shift THEN rotate, matching
    ToyCNNModel4.forward's order), for direct bit-for-bit comparison --
    not just feat_can."""
    gidx, shift = canonical_transform(x)
    return apply_d4_batched(apply_shift_batched(x, shift), gidx)


def check_model_d4_equivariance(model, x: torch.Tensor) -> bool:
    ok = True
    y0 = model(x)
    for g in range(N_D4):
        yg = model(torch_d4(x, g))
        err = (yg - torch_d4(y0, g)).abs().max().item()
        if err > TOL:
            ok = False
            print(f"  [FAIL] D4-equivariance under {D4_NAMES[g]}: max err {err:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] model(g.x) == g.model(x) for all 8 D4 ops (tol {TOL:.0e})")
    return ok


def check_model_translation_equivariance(model, x: torch.Tensor) -> bool:
    ok = True
    y0 = model(x)
    offsets = [(0, 0), (1, 0), (0, 1), (-1, 2), (2, -3), (-3, -3), (10, -15), (20, 20)]
    for dr, dc in offsets:
        yt = model(torch_translate(x, dr, dc))
        err = (yt - torch_translate(y0, dr, dc)).abs().max().item()
        if err > TOL:
            ok = False
            print(f"  [FAIL] translation-equivariance at ({dr},{dc}): max err {err:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] model(t.x) == t.model(x) for {len(offsets)} shifts "
          f"(tol {TOL:.0e}) -- any integer shift is safe here, no margin needed")
    return ok


def check_shift_commutes_with_d4(x: torch.Tensor) -> bool:
    """Defining property: recentering x then rotating by g must equal
    rotating x by g then recentering g.x directly -- i.e. shift(g.x) acts
    as g applied to shift(x) would. Tested on the actual grids (not by
    re-deriving how g acts on a bare 2-vector), so this also implicitly
    confirms shift and rotation share the same pivot."""
    ok = True
    s0 = canonical_shift(x)
    B = x.shape[0]
    for g in range(N_D4):
        xg = torch_d4(x, g)
        sg = canonical_shift(xg)
        g_idx = torch.full((B,), g, dtype=torch.long)
        ref = apply_d4_batched(apply_shift_batched(x, s0), g_idx)   # shift(x) then rotate
        out = apply_shift_batched(xg, sg)                            # rotate(x) then shift(g.x)
        if not torch.equal(ref, out):
            ok = False
            bad = int((ref != out).any(dim=(1, 2, 3)).sum())
            print(f"  [FAIL] shift commutation under {D4_NAMES[g]}: {bad}/{B} mismatched")
    print(f"  [{'PASS' if ok else 'FAIL'}] shift(g.x) and g.shift(x) recenter to the identical "
          f"grid for all 8 D4 ops (zero tolerance)")
    return ok


def check_combined_canonicalization_asymmetric() -> bool:
    """For glider/lwss, canonicalize_grid(x) must be bit-identical across
    every combination of a D4 transform and a margin-safe translation."""
    ok = True
    bad = 0
    total = 0
    for name, top, left in [("glider", 15, 15), ("lwss", 15, 10)]:
        x0 = _place(name, top, left)
        ref = canonicalize_grid(x0)
        for g in range(N_D4):
            for dr, dc in [(0, 0), (1, 0), (0, 1), (2, -3), (-3, 2), (3, 3)]:
                xt = torch_translate(torch_d4(x0, g), dr, dc)
                out = canonicalize_grid(xt)
                total += 1
                if not torch.equal(out, ref):
                    bad += 1
    ok = bad == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] combined canonicalization on asymmetric patterns: "
          f"{bad}/{total} mismatches (expect 0)")
    return ok


def report_the_actual_fix(model):
    """Not a hard gate -- confirms Model 4 still has exact D4 invariance
    for the patterns Model 1 got wrong (pulsar r90 = 0.835) and Model 2
    fixed (1.0000 exactly). Perturbation robustness -- the actual thing
    Model 4 changes -- is in analysis_perturbation.py, not here."""
    print("\nfeat_can cos similarity under D4, symmetric patterns off-center")
    print("(Model 1's worst case: pulsar r90 = 0.835; Model 2 fixed to 1.0000 exactly):")
    with torch.no_grad():
        for name, top, left in [("block", 18, 18), ("pulsar", 13, 13), ("beehive", 18, 18)]:
            x = _place(name, top, left)
            _, aux0 = model(x, return_features=True)
            fc0 = aux0["feat_can"]
            sims = []
            for g in range(N_D4):
                _, auxg = model(torch_d4(x, g), return_features=True)
                sims.append(cos(fc0, auxg["feat_can"]))
            print(f"  {name:<10} min={min(sims):.4f}  all={[round(s, 4) for s in sims]}")


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "cpu"
    x, _ = make_batch(64, rng, device)

    model = ToyCNNModel4(m=4).to(device).eval()

    print("HARD checks (must pass)")
    with torch.no_grad():
        print("A) full-model D4 equivariance (random init)")
        okA = check_model_d4_equivariance(model, x)
        print("B) full-model translation equivariance (random init)")
        okB = check_model_translation_equivariance(model, x)
    print("C) canonical_shift equivariant labeling (zero tolerance)")
    okC = check_shift_commutes_with_d4(x)
    print("D) combined D4+translation canonicalization, asymmetric patterns")
    okD = check_combined_canonicalization_asymmetric()

    all_ok = okA and okB and okC and okD
    print(f"\n{'ALL HARD CHECKS PASSED' if all_ok else 'SOME HARD CHECKS FAILED'}")

    report_the_actual_fix(model)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
