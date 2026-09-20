# Model 2 — joint translation + D4 canonicalization

## Why this exists

Model 1 canonicalized D4 (rotation/reflection) only, relying on circular
padding for translation *equivariance*. That's fine for the prediction task
(exact either way — see below), but not for the actual goal of this
`representation_similarity` line of work: a `feat_can` that's genuinely
*comparable* across grids. Equivariant (moves consistently with the input)
and invariant (looks the same regardless of where the content sits) are
different properties, and Model 1 only had the first for translation.

The gap showed up concretely: patterns with real D4 sub-symmetry (block,
pulsar, beehive, ...) placed off the grid's own rotation center showed
`feat_can` cosine similarity as low as **0.8354** (pulsar, under r90) across
the 8 D4 ops, even though asymmetric patterns (glider, lwss) were exactly
1.0000. Root cause (found by direct inspection, see `../model1/notes.md`):
for a symmetric shape, D4 canonicalization can't tell anything needs
correcting, so it always answers "identity" — and since rotating the whole
grid moves an off-center object anyway (rotation pivots on the *grid's*
center, not the shape's), that uncorrected shift is exactly the gap.

## Design

Translation canonicalization has a real advantage over D4's: a shape's
centroid is a single point, not one of 8 discrete candidates a symmetric
shape can make indistinguishable — so there's no discrete-tie mechanism the
way D4 has one.

**First attempt (wrong, caught by testing before it shipped):** compute the
D4 choice and the recentering shift independently from the original `x`,
then compose them. This breaks even for ASYMMETRIC patterns — glider and
lwss, which Model 1 got exactly right, dropped to ~0.91-0.95 cosine
similarity. Cause: `floor(centroid)` does not commute with rotation. A
reflection flips a coordinate's sign; flooring a rotated value isn't the
same as rotating a floored value. Composing two independently-computed
corrections silently assumes a commutation property that doesn't hold.

**Fix — `canonical_transform`:** for each of the 8 D4 elements, compute its
*own* recentering shift fresh (never reuse one computed elsewhere), form
the full rotate-then-recenter candidate, and pick the winner by scoring
those fully-formed candidates (same wide/right/down + exact-relative-
coordinate tiebreak as Model 1's `canonical_gidx`). This sidesteps the
commutation question entirely: the 8 full candidates for `x` and for `g0.x`
are the *same set*, just relabeled, so the argmax lands on the identical
winning candidate either way — the same "argmax over the orbit" principle
that made Model 1's D4-only canonicalization correct, now over the richer
combined orbit. Applied as: rotate first, then recenter; undone in reverse
(unshift, then unrotate).

## Verification

`sanity_check.py`:
- **A, B (hard):** full-model D4 and translation equivariance — pass
  exactly, and (same as Model 1) provably independent of whether
  `canonical_transform` gets the "right" answer, since the head is 1x1 and
  commutes with any spatial relabeling.
- **C (hard):** combined canonicalization on glider/lwss under every
  combination of a D4 op and a margin-safe translation (8 x 6 = 48 cases
  per pattern) — must be bit-identical. 0/96 mismatches.

**The actual fix, verified directly** (feat_can cosine under all 8 D4 ops):

| pattern | Model 1 (D4-only) | Model 2 (D4 + translation) |
|---|---|---|
| block | 0.9952-1.0000 | **1.0000, all 8** |
| pulsar | 0.8354-1.0000 | **1.0000, all 8** |
| beehive | 0.9068-1.0000 | **1.0000, all 8** |

All 10 named patterns (including the previously-broken ones) and 4 random-
density grids (0.1/0.3/0.5/0.8) now show exactly 1.0000 across all 8 D4
ops. The symmetric-shape gap is closed.

## Architecture

Identical backbone/head to Model 1 (`IsotropicConv2d(1,2m)` -> poly ->
`Conv2d(2m,m,k=1)` -> poly -> `Conv2d(m,1,k=1)` head), so weights transfer
directly: `train.py` warmstarts from `../model1/checkpoints/best_v2.pt`.
Only the canonicalization step changed. 109 params, same as Model 1 (the
canonicalization machinery carries no learnable parameters).

## Training

Applies both of Model 1's lessons from the start (no need to rediscover
them): `make_mixed_batch` (50% random-density / 50% single-pattern) and
scheduled sampling (K=3, p ramped 0->1 over 40 of 60 epochs). Converged
immediately thanks to the warmstart (F1=1.0000 from epoch 1; selected
checkpoint is epoch 1, saved as `checkpoints/best.pt`). See
`results/train_log.txt` / `results/run1.out` for the full run.

**Re-verified on the trained checkpoint** with fresh seeds not used
anywhere in training or per-epoch eval: F1=1.0000 at every density tested
(0.05-0.90); D4/translation equivariance error 4.9e-04 (float32 noise, same
order as Model 1's post-training check); the symmetric-pattern fix holds
after training too (block/pulsar/beehive all exactly 1.0000 `feat_can`
cosine across all 8 D4 ops, not just at init); zero cell drift over a
40-step autoregressive rollout at densities 0.3/0.5/0.8.

## Representation analysis: translation invariance has a real scope limit

`analysis_representation.py` extends Model 1's report with a translation
test (Model 1 never canonicalized position at all). Result, `feat_can`
flattened cosine similarity:

| stimulus | D4 (all 8 ops) | translation, small shifts | translation, large shifts |
|---|---|---|---|
| glider / lwss / block (localized) | 1.0000 | 1.0000 | 1.0000 (even (10,15), (-12,-18)) |
| pulsar (localized, but 13x13 -- large) | 1.0000 | 1.0000 | drops to ~0.77-0.78 at (10,15)/(-12,-18) |
| random-density grids | 1.0000 | **drops immediately**, even at shift (1,0) | 0.12-0.86 depending on density |

Two different phenomena, not one:

1. **Pulsar's large-shift drop is the familiar margin caveat** (same as
   Model 1's D4 tiebreak, `canonical_gidx`'s equivariant-labeling proof, and
   `apply_shift_batched` generally): the shift is large enough to wrap a
   16x16-ish footprint around the 40-cell torus, and the centroid math
   isn't toroidal-aware. Expected, and consistent with everything already
   documented about needing margin.

2. **Random-density grids break down at ANY shift, including a single
   pixel** -- a qualitatively different, more fundamental limit. Translation
   canonicalization is built entirely around "a single object has a
   well-defined centroid you can recenter." That assumption doesn't apply
   to a space-filling random grid: alive cells are scattered across the
   *entire* domain, so they're already effectively touching the boundary at
   any reasonable density -- there is no margin to have, and wraparound
   isn't an edge case there, it's the default. D4 canonicalization doesn't
   have this problem (confirmed: exact 1.0000 for dense grids too) because
   rotating/reflecting the whole finite grid never moves content off one
   edge onto another the way a translation roll does -- it only relabels
   existing positions.

**Reading:** "canonicalize translation" is inherently an *object-centric*
operation -- it answers "where does the same shape sit" and that question
is only well-posed for something localized enough to have a position. A
dense/statistical distribution doesn't have one in that sense; asking for
its "translation-canonical form" is close to a category error, not a case
this implementation happens to get wrong. This is a real, principled scope
boundary of Model 2's translation canonicalization, not a further bug to
chase -- worth remembering if a later model tries to extend invariance
claims to dense inputs.

## Files

Self-contained, same convention as `../model1/`: `IsotropicConv2d`,
`PolyActivation`, `_extents`, `canonical_gidx`, `apply_d4_batched` are
copied verbatim from `../model1/net.py`; D4 group machinery still comes
from `../../v1/adapter.py`. `data.py` is also a verbatim copy of Model 1's.
