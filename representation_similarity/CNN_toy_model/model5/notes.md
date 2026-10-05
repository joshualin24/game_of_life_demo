# Model 5 — learned canonicalization instead of hand-coded

## Motivation

Models 1-4 all hand-coded the canonicalization (a deterministic argmax
over 8 D4 candidates, with translation handled by a closed-form centroid
shift). Model 5 asks: can a network LEARN this instead? Two parallel
parts:

- **Part A**: the same backbone as Model 4 (`IsotropicConv2d` + two
  `PolyActivation` layers + a 1x1 head), but with NO hand-coded
  canonicalization — it just produces raw, uncanonicalized `feat`.
- **Part B**: a new `PosePredictor` that sees the FULL input grid and
  predicts a pose: a 2D shift (regression) and a D4 element (8-way
  classification).

Training distills from Model 4 as a frozen teacher with three losses —
`L_task`, `L_feat`, `L_pose` (direct supervision against Model 4's actual
`(shift, gidx)`) — mixed with scheduled sampling between teacher-forced
and self-predicted pose. Part A+head is warmstarted from Model 4;
Part B starts from scratch. Full design in `net.py`/`train.py`.

Architecture diagram (`architecture_diagram.py` -> `results/architecture_diagram.png`):
Part A and Part B run in parallel from the same input, each producing
`feat` / `(shift_pred, gidx_logits)`; the latter is discretized
(`round()`/`argmax()`, `no_grad()`) into `shift_used`/`gidx_used`, which
`combine()` then applies to `feat` (shift → rotate → `head` → undo rotate
→ undo shift) to produce the final `logits`.

## v1: task accuracy is structurally uninformative, and Part B collapsed

Before the fix described below, two things were found (full detail
preserved in git history / `results_v1_collapsed/`):

1. **Task accuracy can't measure whether Part B learned anything.**
   `combine()`'s shift/rotate-then-undo is, for *any* self-consistent
   pose — correct, learned-but-wrong, or a deliberately wrong constant —
   mathematically identical to just computing `head(feat)` directly, a
   direct consequence of the same "1x1 head commutes with relabeling"
   identity every prior model's equivariance guarantee already relies on.
   Verified directly: feeding the same `feat` through Part B's own
   predicted pose, no pose at all, and a deliberately wrong constant pose
   all produced logits identical to float32 roundoff (~2.4e-4). **This
   remains true in v2 as well** — it's a structural property of
   `combine()`, not something the pooling fix below changes. Task F1 and
   rollout accuracy are reported in this file for completeness, but they
   are not evidence about Part B either way.
2. **Part B collapsed.** `gidx_acc`/`shift_mae` against the teacher were
   frozen at their epoch-1 values for 80 straight epochs; the 8-way
   rotation classifier only ever output 2 of 8 classes; direct
   D4-consistency check read exactly 0.125 (chance).

**Root cause, confirmed by direct measurement**: `PosePredictor`
concatenated coordinate channels to the input and pooled with a plain
global SUM over all 1600 grid positions. Coordinate channels are
injected at *every* position, dead or alive — so for a sparse grid,
summing over 1600 mostly-dead positions is dominated by that
position-only background, not by the actual alive pattern. Measured
directly: for grids with 3-78 alive cells, **94-99.8% of the pooled
vector's norm came from dead cells**; even a 640-alive-cell (40%-density)
grid still had 56% of its pooled magnitude coming from dead cells. The
classifier and regressor had no real per-sample signal to learn from.

## v2: the fix

Changed `PosePredictor`'s pooling from a plain global sum to a **masked
mean over alive cells only**, with alive-count reintroduced explicitly:

```python
cnt = x.sum(dim=(2, 3)).clamp(min=1.0)          # (B,1) alive-cell count
pooled = (h * x).sum(dim=(2, 3)) / cnt           # masked MEAN pool, alive cells only
log_cnt = torch.log1p(cnt)                        # explicit, bounded count feature
pooled = torch.cat([pooled, log_cnt], dim=1)
```

This directly mirrors the hand-coded centroid/extent formulas already in
`../model2/net.py`/`../model3/net.py`/`../model4/net.py` (e.g.
`r_mean = (mask*rows).sum()/cnt`) — a masked mean over the alive subset,
**not** the whole-grid mean-pooling this project has otherwise warned
against (that trap is averaging over *all* positions on a mostly-empty
grid, which dilutes position information; this masks out the dead-cell
background instead, which was never informative to begin with). Nothing
else changed — same hyperparameters, same warmstart, same training
procedure — to isolate the pooling fix as the only variable.

## v2 results: large, genuine improvement; incomplete convergence

Retrained for the identical 80 epochs/hyperparameters as v1. Final
(best, epoch 70) checkpoint, compared directly against the v1 numbers
above:

| metric | v1 (collapsed) | v2 (fixed pooling) |
|---|---|---|
| `shift_mae` (teacher-relative, fixed val set) | frozen at 7.108 | **0.163** |
| `gidx_acc`, pattern-grid val set | frozen at 0.489 | **0.726** |
| `gidx_acc`, random-density val set | frozen at 0.117 | **0.289** |
| pose-consistency under D4 (`sanity_check.py` check A) | exactly 0.125 (chance) | **0.217** |
| `gidx_used` class usage (256-512 fresh samples) | only 2 of 8 classes ever used | **all 8 classes used**, roughly tracking the teacher's marginal distribution |

**Shift regression is essentially solved**: MAE dropped by ~97%, and a
fresh 512-sample random-density batch gives `shift_mae=0.105` — the
model is now computing something very close to the true recentering
shift, not a density proxy.

**Rotation/reflection classification improved substantially but did not
converge**: `gidx_acc` on fresh random-density data (0.340 on an
independent 512-sample batch) is well above chance (0.125) and the
class-usage collapse is gone, but it is far from the teacher's exact
8-way convention. `sanity_check.py` check A's per-element breakdown is
the most informative single result here:

```
g=e         agreement=1.000   (trivial: identity always matches itself)
g=r90       agreement=0.297
g=r180      agreement=0.109
g=r270      agreement=0.172
g=flip_v    agreement=0.047
g=flip_h    agreement=0.031
g=transp    agreement=0.031
g=atransp   agreement=0.047
```

This is not uniform noise around chance — the 4 pure rotations
(`e,r90,r180,r270`) do meaningfully better than the 4
reflection-involving elements (`flip_v,flip_h,transp,atransp`), which sit
*below* chance. The network learned something that distinguishes
rotations reasonably but actively confuses reflections, a genuinely new
(and somewhat expected, in hindsight) finding: a plain conv+pool stack
has no architectural bias toward detecting reflections specifically
(unlike `IsotropicConv2d`'s D4-equivariance, nothing here privileges
mirror symmetry), so a flip and its paired rotation are evidently easy
for this architecture to confuse.

**Fresh-seed verification**: task F1 = 1.0000 on 6 independent fresh
seeds (3 pattern-grid, 3 random-density) and zero 20-step rollout drift —
unchanged from v1, and per the structural finding above, this says
nothing new about Part B.

**Perturbation analysis** (`analysis_perturbation.py`, own predicted
pose) now shows real, non-trivial decision-boundary behavior again,
unlike v1's uniformly-flat 0-branch-changes-everywhere result:

| stimulus | mean cos_sim | branch changes / 1600 |
|---|---|---|
| glider | 0.9581 | 3 |
| pulsar | 0.8315 | 0 |
| block | 0.9732 | 6 |
| random d0.1 | 0.8780 | 48 (cos drops to 0.634 when crossed) |
| random d0.3 | 0.9980 | 2 |
| random d0.5 | 0.9994 | 0 |

Pulsar's 0.8315 is in the same ballpark as Model 4's own 0.8504 (not
identical — this pose predictor doesn't reproduce Model 4's exact
convention — but a broadly comparable degree of sensitivity, not the
qualitatively different "trivially smooth everywhere" signature v1
showed). `random_d0.1` now shows a real, sharp decision boundary (48/1600
cells crossing it, cosine dropping to 0.62-0.73 there) — the same kind of
discrete-argmax fragility Models 2-4 found, now reappearing because the
classifier is actually discriminating again instead of being collapsed.

## Verification

`sanity_check.py` (measures rather than asserts; see v1 discussion of
why Model 5 has no exact-equivariance guarantee):
- **A** (pose consistency under D4): 0.217, up from exactly 0.125 —
  genuine improvement, driven mostly by the 4 pure-rotation elements.
- **B** (end-to-end output equivariance, own pose): 1.00000 — per the
  structural finding above, this is guaranteed by `combine()` regardless
  of pose quality and is not evidence of anything Part B did.
- **C** (Part A + `combine()`, teacher pose, exact D4-equivariance — the
  one real architectural hard gate): **PASS**, relative tolerance 1e-3,
  all 8 D4 elements. Unaffected by the Part B change, as expected.
- **D** (Part A vs. Model 4 output, teacher pose — informational):
  prediction agreement 1.0000, max abs logit diff 1.69e3 (up from 8.17e2
  in v1's best checkpoint) — Part A's weights have continued to drift
  from Model 4's with the extra 30 epochs of training, as expected.

## Summary

The pooling fix resolves the actual diagnosed bug (background-dominated
pooling) and produces a large, genuine improvement: shift regression is
essentially solved, rotation classification moves well past chance and
stops being collapsed, and perturbation behavior returns to the kind of
real, discrete decision-boundary sensitivity seen in Models 2-4. But
rotation/reflection classification does not converge to the teacher's
exact convention within this training budget (gidx_acc ~0.29-0.73
depending on data regime, not ~1.0), and the breakdown by D4 element
shows a specific, interpretable remaining weakness: reflections are
learned worse than rotations, plausibly because nothing in this
architecture (unlike `IsotropicConv2d`) privileges mirror symmetry. A
further iteration (longer training now that gradient signal is real,
more capacity for the pose head, or an architectural bias toward
detecting reflections specifically) was not attempted here, since it
would be a new, explicitly-scoped follow-up rather than part of this fix.
As before, the headline caveat stands regardless of how far rotation
classification eventually converges: task accuracy structurally cannot
measure Part B's success, by construction, for any pose it ever learns.
