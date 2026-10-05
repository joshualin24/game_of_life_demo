# Model 5v3 — shared trunk: letting the two parts talk

## Motivation

Model 5 (v1/v2, see `../model5/notes.md`) built Part A (dynamics) and
Part B (pose) as two **completely independent** networks: both read raw
`x`, but shared no layer, and were gradient-isolated on top of that
(`L_task`/`L_feat` reached only Part A; `L_pose` reached only Part B).
That isolation was a deliberate choice and it worked cleanly, but the
user asked for something different: a design "similar to Model 4," where
"some neurons" handle canonicalization and "some others" handle dynamics,
**allowed to talk to each other** while staying **functionally
distinguishable** — and specifically wanted to know whether a model built
this way can learn Model 4's representation *while keeping t+1 accuracy*.

## Design: a shared trunk

```
trunk = PolyActivation(2m)(IsotropicConv2d(1, 2m)(x))      # SHARED, exactly Model 4's first layer
feat  = PolyActivation(m)(Conv2d(2m, m, 1)(trunk))          # dynamics head (Part A's role)
(shift_pred, gidx_logits) = PoseHead(trunk, x)               # pose head (Part B's role), reads the SAME trunk
```

The trunk is consumed by **both** heads, with autograd intact (not
detached) on the pose head's side — this is the entire mechanism of
"talking": `iso1`/`act1`'s weights now receive gradient contributions
from `L_task`/`L_feat` (via the dynamics head) **and** from `L_pose` (via
the pose head), simultaneously. Verified directly before training: a loss
computed purely from the pose head's output produces a nonzero gradient
on the trunk's weights (`iso1.w_c.grad` sum ≈ 0.27 from a single
pose-only backward pass on an untrained model) — confirming real
cross-talk, not just a shared *call*.

Coordinate channels (needed for the pose head's position-sensitivity, as
in `../model5/net.py`) are appended **after** the trunk, on the pose
head's own input only — never on the trunk's input. This preserves the
one thing that must stay exactly true regardless of any of this: the
trunk is `IsotropicConv2d` + `PolyActivation`, exactly D4-equivariant
BY CONSTRUCTION for any weights, and appending non-equivariant fixed
coordinate buffers to its own input would have broken that guarantee.
See `net.py`'s docstring for the full reasoning.

Warmstart is a direct, exact match: `iso1`/`act1`/`conv2`/`act2`/`head`
have literally the same names and shapes as Model 4's own state dict, so
the entire trunk + dynamics head loads from Model 4's checkpoint with no
translation needed; only the pose head starts from scratch. Same
training recipe otherwise as `../model5/train.py` (three losses,
scheduled sampling, 80 epochs, identical hyperparameters) — the shared
trunk is the only variable changed, to isolate its effect.

## Headline result: a genuine, multi-axis tradeoff — not a clean win

Compared directly against `../model5/notes.md`'s v1 (collapsed) and v2
(isolated, partial success) numbers:

| metric | v1 (collapsed) | v2 (isolated) | **v3 (shared trunk)** |
|---|---|---|---|
| pose-consistency under D4 (check A) | 0.125 (chance) | 0.217 | **0.268** |
| `gidx_acc`, pattern data | frozen 0.489 | 0.726 | **0.978** |
| `gidx_acc`, random data (fresh) | frozen 0.117 | 0.340 | **0.168** (worse) |
| `shift_mae` (fresh) | frozen 7.108 | 0.105 | **0.124** (~same) |
| task F1, fresh seeds | 1.0000 | 1.0000 | **0.959–0.988** (regression) |
| rollout diff @ 20 steps, fresh seed | 0 | 0 | **326 / 1600 cells** (major regression) |

**Pose learning genuinely improved on several axes** — pose-consistency
under D4 (the most direct equivariance measure) rose from 0.217 to 0.268,
and `gidx_acc` on structured pattern data jumped from 0.726 to **0.978**,
closing most of the remaining gap to the teacher's convention. The
representation-similarity results below show this most clearly: pulsar's
full D4-orbit now reproduces **feat_can cosine similarity 1.0000** (v2:
0.8426), and named patterns (glider, lwss, block) land at feat_can flat
cosine ≈0.99–1.0000 across all 8 D4 transforms and the tested
translations — the model found a canonicalization that, even where it
doesn't match Model 4's specific labeling (see `gidx_used` columns in
`results/representation_report.txt`), is internally self-consistent
enough to make same-content/different-orientation inputs land on nearly
the same representation.

**But task accuracy, which v1/v2 got "for free," genuinely broke.**
Every prior version of this model (and Models 1-4) had task F1 = 1.0000
on every fresh seed, always, regardless of how good or bad the pose
prediction was — a direct, verified consequence of `combine()`'s
structure (see `../model5/notes.md`'s "headline finding": for *any*
self-consistent pose, `combine()` reduces exactly to `head(feat)`,
independent of the pose used). That identity was re-verified here too
(feeding the same `feat` through the model's own pose, no pose at all,
and a deliberately wrong constant pose all gave logits identical to
within ~1.2e-4) — so the F1 regression is not a bug in `combine()`. It
means `head(feat)` **itself** got worse at the actual GoL rule: tested
directly (bypassing `combine()`/pose entirely), `head(feat)` alone scores
F1 = **0.925** on a sparse random grid, something Model 4's identical
architecture never did. Autoregressive rollout compounds this quickly:
0 cells wrong at step 1 in v1/v2, always; here, 19 cells wrong by step 1,
growing to 326/1600 (≈20%) by step 20 on a fresh seed.

**Root cause, directly implied by the design**: the shared trunk has only
56 parameters (`IsotropicConv2d(1,8)` + `PolyActivation(8)`) and is now
asked to serve two objectives at once — `L_task`/`L_feat` pulling it
toward good GoL-rule features, `L_pose` pulling it toward good
global-pose-classification features. At this tiny capacity, that is a
real, not merely theoretical, source of competition, and the net effect
measured here is that pose learning won a larger share of the trunk's
capacity than task accuracy could afford to give up. This is exactly the
cost the user's request implicitly asked to investigate by removing
v1/v2's isolation.

## Perturbation sensitivity: also a tradeoff, not uniformly better or worse

`analysis_perturbation.py` (own predicted pose, single-cell flips):

| stimulus | v2 mean cos_sim (branch chg/1600) | v3 mean cos_sim (branch chg/1600) |
|---|---|---|
| glider | 0.9581 (3) | 0.9543 (**1570**) |
| pulsar | 0.8315 (0) | 0.8479 (0) |
| block | 0.9732 (6) | 0.9685 (**1575**) |
| random d0.1 | 0.8780 (48) | **0.9969** (0) |
| random d0.3 | 0.9980 (2) | 0.8390 (**500**) |
| random d0.5 | 0.9994 (0) | 0.8584 (**558**) |

Not a clean win either way. Sparse random data (d0.1) is dramatically
more stable in v3. But glider and block — both previously near-rock-solid
in v2 — now flip their predicted `gidx` on essentially every single-cell
perturbation (1570/1600, 1575/1600), and mid-density random data
(d0.3/d0.5), previously among v2's most stable cases, now shows frequent,
large representation jumps (cosine similarity crashing to 0.34–0.60 when
the branch flips). This is **not** a contradiction with the D4-orbit
result above — the D4-orbit test applies *exact* group transforms to
already-near-symmetric inputs (block placed centered is pixel-identical
under all 8 ops, so of course `gidx_used` doesn't change); the
perturbation test breaks that exact symmetry with one flipped cell,
exposing how close the classifier's decision boundaries sit to these
configurations. The more expressive, better-trained v3 pose head
evidently carved out *more* (and in some regimes, more fragile) decision
boundaries than v2's cruder one had.

## Representation-similarity analysis (new script, `analysis_representation.py`)

Unlike v1/v2, this model family now has a dedicated D4-orbit +
translation representation-similarity report (`results/representation_report.txt`),
directly analogous to `../model1/` and `../model2/`'s script, adapted to
use the model's own predicted pose end to end. Headline numbers already
covered above (pulsar D4-orbit cos = 1.0000, named patterns ≈0.99–1.0000);
full per-pattern, per-translation breakdown is in the report file.
Discriminability check (5 unrelated grids, `feat_can`): flattened cosine
mean 0.7015 (reasonably separated — well below the ~0.99-1.0 same-content
numbers above), pooled cosine mean 0.9790 (near-useless for
discrimination, consistent with this project's standing "pooling is a
misleading invariance metric" caution).

## Verification

`sanity_check.py`:
- **A** (pose consistency under D4): 0.268, up from v2's 0.217 — but with
  a different shape: `r90` 0.250, `r180` 0.188, `r270` 0.156, `flip_v`
  0.203, `flip_h` 0.078, `transp` 0.109, `atransp` 0.156. Less cleanly
  "rotations better than reflections" than v2 was (v2: reflections
  uniformly near 0.03–0.05, rotations 0.17–0.30); here `flip_v` (0.203)
  beats two of the three pure rotations.
- **B** (end-to-end output equivariance, own pose): 1.00000 — per the
  standing structural finding, this is guaranteed by `combine()`
  regardless of pose quality (re-confirmed above) and is not evidence of
  anything the pose head did.
- **C** (shared trunk + dynamics head + `combine()`, teacher pose, exact
  D4-equivariance — the one real architectural hard gate): **PASS**,
  relative tolerance 1e-3, all 8 D4 elements. The proof never depended on
  how the trunk's weights got to be what they are, so sharing it with the
  pose head doesn't touch this guarantee.
- **D** (trunk+dynamics vs. Model 4, teacher pose — informational):
  prediction agreement 0.9941 (down from v2's 1.0000) — a smaller, more
  direct signal of the same task-accuracy drift documented above, since
  this comparison uses the teacher's pose and isolates the dynamics path
  specifically.

## Summary

The shared-trunk design does what it was asked to do architecturally
(genuine, verified gradient cross-talk between functionally distinct
dynamics and pose heads) and produces real gains in pose learning — most
notably, pose-consistency-under-D4 improved, pattern-data `gidx_acc`
nearly matched the teacher, and representation similarity under exact D4
transforms reached near-perfect for several stimuli where v2 was
mediocre. But it does **not** keep t+1 accuracy intact the way v1/v2 did:
task F1 and especially multi-step rollout stability both regressed
measurably, traced directly to the shared trunk's limited capacity being
split between two competing objectives — and perturbation robustness
became a genuine mixed bag rather than a uniform improvement. The user's
question — can this model learn Model 4's representation *while keeping*
t+1 accuracy — has an honest answer here: partially, and not for free;
at this trunk capacity, the two goals measurably compete. A natural next
step (not attempted here, a new scoped iteration) would be giving the
trunk more capacity, or a loss-weighting schedule that protects task
accuracy once pose learning has converged enough to stop needing as much
of the trunk.
