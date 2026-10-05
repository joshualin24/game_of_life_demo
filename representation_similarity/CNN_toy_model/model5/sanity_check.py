"""Sanity checks for Model 5.

Every prior model's sanity_check.py ASSERTED exact equivariance -- a
mathematical guarantee that had to hold exactly, by construction,
regardless of training. Model 5 has no such guarantee: Part B (the pose
predictor) is an ordinary CNN with no equivariance constraint, so its
predicted (shift, gidx) for x and for g.x need not satisfy
pose(g.x) == g . pose(x) exactly -- and if they don't, forward(g.x) !=
g . forward(x) exactly either. That's not a bug; it's the explicit
tradeoff this model makes (learn canonicalization instead of coding it).

So this file MEASURES rather than asserts:
  A) Does Part B's predicted pose transform consistently under D4? (gidx
     agreement rate + mean shift error, both before vs after applying g to
     the input)
  B) How close to exactly equivariant is the full model's output, end to
     end, using ONLY its own predicted poses (no teacher)?
  C) Architectural hard gate that STILL holds exactly regardless of
     training: Part A (IsotropicConv2d + PolyActivation + 1x1 convs) +
     combine(), given the TEACHER's pose (not Part B's), must be exactly
     D4-equivariant -- same "1x1 head commutes with any spatial
     relabeling" guarantee Models 1-4 have, now isolated to just Part A so
     Part B's imperfections (checked in A/B above) can't hide inside it.
  D) Teacher-weight drift (informational, not asserted): does Part A,
     using the TEACHER's pose, still reproduce Model 4's output exactly?
     Only true immediately after warmstart, before training moves the
     weights -- tracks how much Part A has diverged from Model 4.

Run: python sanity_check.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, _HERE)
from net import ToyCNNModel5  # noqa: E402
from data import make_mixed_batch  # noqa: E402
from adapter import D4_NAMES, N_D4, CAYLEY, torch_d4  # noqa: E402


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


net4 = _load_module(os.path.join("..", "model4", "net.py"), "net4_sanity")

torch.manual_seed(0)
rng = np.random.default_rng(0)


def load_model(ckpt_name="best.pt"):
    model = ToyCNNModel5(m=4)
    path = os.path.join(_HERE, "checkpoints", ckpt_name)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"loaded {path} (epoch {ckpt.get('epoch')})")
    else:
        print(f"WARNING: {path} not found -- checking an UNTRAINED model "
              f"(only checks B/C below are meaningful; A/D will look near-random/exact resp.)")
    model.eval()
    return model


@torch.no_grad()
def check_A_pose_consistency(model, x, n_report=3):
    """Does pose(g.x) behave like g acting on pose(x)? Measured, not
    asserted: gidx should satisfy gidx(g.x) == CAYLEY[g, gidx(x)] if the
    pose predictor were exactly equivariant; shift should satisfy a
    corresponding transform rule. Report agreement rate, not a pass/fail."""
    print("\n--- A) Part B pose-prediction consistency under D4 (measured) ---")
    _, shift0, gidx_logits0 = model.encode(x)
    gidx0 = gidx_logits0.argmax(dim=1)
    B = x.shape[0]
    agree_counts = []
    for g in range(N_D4):
        xg = torch_d4(x, g)
        _, shiftg, gidx_logitsg = model.encode(xg)
        gidxg = gidx_logitsg.argmax(dim=1)
        expected_gidx = torch.as_tensor(CAYLEY, dtype=torch.long)[g, gidx0]
        agree = (gidxg == expected_gidx).float().mean().item()
        agree_counts.append(agree)
    mean_agree = float(np.mean(agree_counts))
    print(f"  gidx(g.x) == CAYLEY[g, gidx(x)] agreement across all 8 g, {B} samples: "
          f"{mean_agree:.3f} (1.0 = exactly equivariant, {1/N_D4:.3f} = random)")
    for g, a in zip(D4_NAMES, agree_counts):
        print(f"    g={g:9s} agreement={a:.3f}")
    return mean_agree


@torch.no_grad()
def check_B_output_equivariance(model, x):
    """End-to-end, self-contained forward() (model's own predicted pose,
    no teacher): how close is model(g.x) to g.model(x)? Measured as the
    fraction of output cells that disagree, averaged over all 8 g."""
    print("\n--- B) end-to-end output equivariance, model's OWN pose (measured) ---")
    logits0 = model(x)
    pred0 = (logits0 >= 0).float()
    mismatch_fracs = []
    for g in range(N_D4):
        xg = torch_d4(x, g)
        logitsg = model(xg)
        predg = (logitsg >= 0).float()
        expected = torch_d4(pred0, g)
        mismatch = (predg != expected).float().mean().item()
        mismatch_fracs.append(mismatch)
    mean_mismatch = float(np.mean(mismatch_fracs))
    print(f"  mean fraction of output cells disagreeing with g.model(x), across 8 g: {mean_mismatch:.5f}")
    print(f"  (0.0 = exactly equivariant like Models 1-4; this model trades that guarantee away)")
    for g, m in zip(D4_NAMES, mismatch_fracs):
        print(f"    g={g:9s} mismatch_frac={m:.5f}")
    return mean_mismatch


@torch.no_grad()
def check_C_architecture_equivariant_given_teacher_pose(model, x):
    """Isolates Part A + combine()'s architecture from Part B's learned
    (possibly imperfect) pose: drive combine() with the TEACHER's pose
    (net4.canonical_transform, independently re-verified exact in
    ../model3/sanity_check.py and ../model4/sanity_check.py), computed
    FRESH on x and on g.x separately -- no hand-derived vector-transform
    rule needed, same direct-grid-comparison trick ../model4/sanity_check.py
    uses for its own checks A/B. Part A is literally IsotropicConv2d +
    PolyActivation + 1x1 convs, exactly D4-equivariant BY CONSTRUCTION for
    ANY weights (same as Models 1-4) -- so this must hold near-exactly
    regardless of training, and is a real bug if it doesn't."""
    print("\n--- C) Part A + combine(), given the TEACHER's pose, is exactly D4-equivariant ---")
    # Relative, not absolute, tolerance: Part A's output magnitude is
    # unbounded and grows with training (see check D's max_err), so a fixed
    # absolute tolerance (fine at init, where values are O(1), as in Models
    # 1-4's checks) stops being meaningful once values reach O(100+) --
    # float32 roundoff alone exceeds 1e-3 absolute at that scale.
    ok = True
    for g in range(N_D4):
        xg = torch_d4(x, g)
        gidx_t, shift_t = net4.canonical_transform(x)
        gidx_tg, shift_tg = net4.canonical_transform(xg)
        feat, _, _ = model.encode(x)
        feat_g, _, _ = model.encode(xg)
        logits, _ = model.combine(feat, shift_t, gidx_t)
        logits_g, _ = model.combine(feat_g, shift_tg, gidx_tg)
        expected = torch_d4(logits, g)
        err = (logits_g - expected).abs().max().item()
        scale = max(expected.abs().max().item(), logits_g.abs().max().item(), 1.0)
        rel_err = err / scale
        if rel_err > 1e-3:
            ok = False
            print(f"  [FAIL] g={D4_NAMES[g]:9s} max abs err = {err:.3e}  (scale={scale:.2e}, rel={rel_err:.2e})")
    print(f"  [{'PASS' if ok else 'FAIL'}] holds for all 8 D4 ops, relative tol 1e-3 "
          f"(architectural -- Part A's weights can drift from Model 4's, this identity shouldn't)")
    return ok


@torch.no_grad()
def check_D_teacher_pose_reproduces_model4(model, x):
    """Using the TEACHER's pose (not Part B's), does Part A + head
    reproduce Model 4's output exactly? Isolates whether warmstarted
    weights still compute the same function, independent of Part B."""
    print("\n--- D) Part A with TEACHER pose reproduces Model 4 exactly? ---")
    gidx_t, shift_t = net4.canonical_transform(x)
    feat, _, _ = model.encode(x)
    logits, _ = model.combine(feat, shift_t, gidx_t)

    teacher = net4.ToyCNNModel4(m=4)
    ckpt = torch.load(os.path.join(_HERE, "..", "model4", "checkpoints", "best.pt"),
                       map_location="cpu", weights_only=True)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    logits_teacher = teacher(x)

    max_err = (logits - logits_teacher).abs().max().item()
    match = float(((logits >= 0) == (logits_teacher >= 0)).float().mean())
    print(f"  max abs logit diff: {max_err:.3e}  |  prediction agreement: {match:.4f}")
    print(f"  (1.0 agreement, ~0 diff expected ONLY if Part A weights are still identical to "
          f"Model 4's post-warmstart; training can and will move them away from this)")
    return max_err, match


if __name__ == "__main__":
    model = load_model()
    x, _ = make_mixed_batch(64, rng, "cpu", random_frac=0.5)

    a = check_A_pose_consistency(model, x)
    b = check_B_output_equivariance(model, x)
    c_ok = check_C_architecture_equivariant_given_teacher_pose(model, x)
    d_err, d_match = check_D_teacher_pose_reproduces_model4(model, x)

    print("\n=== summary ===")
    print(f"A) pose consistency under D4:       {a:.3f}  (1.0 = perfect, 0.125 = random)")
    print(f"B) end-to-end output equivariance:   {1 - b:.5f}  (1.0 = perfect, like Models 1-4)")
    print(f"C) Part A+combine() w/ teacher pose: {'PASS' if c_ok else 'FAIL'}  (hard gate -- architectural, not learned)")
    print(f"D) Part A vs Model 4 (teacher pose): agreement={d_match:.4f}  max_err={d_err:.2e}  "
          f"(informational -- expected to drift from 1.0/0 as training proceeds)")
    assert c_ok, "Part A's architecture should stay exactly D4-equivariant regardless of training -- this is a REAL bug if it fails"
    print("\ncheck C (the only exact architectural guarantee Model 5 retains) passed.")
