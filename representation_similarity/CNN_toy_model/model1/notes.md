# Model 1 — isotropic-conv toy CNN with D4 pose canonicalization

## Design goal

A small CNN for GoL t->t+1 prediction that is **exactly** equivariant to the
full symmetry group of the rule (toroidal translation x D4 rotation/
reflection), built architecturally rather than learned -- as a clean,
inspectable baseline for studying what these symmetries do to internal
representations (companion to the V8/V10 study one level up, which found the
transformer's *learned* representations respect these symmetries poorly and
degrade with depth).

## Design history (from the design conversation)

The original idea: encode "the rotation/reflection needed to standardize a
configuration" from simple bounding-box criteria (`x_max >= |x_min|`,
`y_max >= |y_min|`), inject that as a few deterministic nodes at an internal
layer, and have the output layer use it to restore the correct orientation
("manually enforced" for v1, vs. "learned" as a later variant).

Two decisions made before building: (1) cover the **full 8-element D4**
group, not just the 4-element reflection subgroup the two criteria
literally disambiguate -- added a third criterion (compare horizontal vs.
vertical extent) to also resolve 90/270 degree rotations. (2) the pose
signal acts as a **side-channel**: the raw grid flows through the conv
backbone untouched; canonicalization happens on the backbone's *feature
map*, not the input pixels.

**The isotropic-backbone requirement.** Canonicalizing a feature map only
restores real equivariance if the backbone producing that feature map is
*itself* already exactly equivariant -- a generic `nn.Conv2d(3x3)` is not
(9 free weights per channel pair means it doesn't treat the 8 Moore
neighbors symmetrically). Fixed by constraining the first conv layer to only
3 free weights per (in,out) pair -- center / shared-orthogonal-neighbor /
shared-diagonal-neighbor (`IsotropicConv2d` in `net.py`) -- which commutes
with D4 by construction and can still represent GoL exactly (GoL's true
rule is *more* symmetric than this: it doesn't even distinguish orthogonal
from diagonal neighbors, i.e. it's the special case where the two shared
weights end up equal).

## Architecture

```
feat      = poly(IsotropicConv2d(1, 2m))(x)     3x3, circular padding
feat      = poly(Conv2d(2m, m, k=1))(feat)      1x1, pointwise
g*(x)     = canonical_gidx(x)                    deterministic, no grad
feat_can  = g*(x) . feat                         canonicalize
logits_can = Conv2d(m, 1, k=1)(feat_can)         shared 1x1 head
logits    = g*(x)^-1 . logits_can                restore original frame
```

`PolyActivation` (learnable per-channel `w0 + w1*x + w2*x^2`) is reused
verbatim from `../../poly_activation_verify/model.py`, which verified it
reliably reaches 100% train accuracy on minimal GoL CNNs where ReLU at the
same param count often doesn't.

## Two real bugs found via `sanity_check.py` (both would have shipped
silently without it -- the whole point of writing that script first)

1. **`canonical_gidx` must be an argmax over the group orbit, not a
   fixed-order composition of independently-computed bits.** The first
   version computed `flip_h`/`flip_v`/`swap` all from the *original*
   untransformed extents and composed them in a fixed order via the Cayley
   table. This does not correctly label ~30-90% of samples (verified via the
   equivariant-labeling property `CAYLEY[g*(g.x), g] == g*(x)`). Fixed by
   computing, for each of the 8 candidate transforms of x, the same 3
   boolean criteria evaluated *on that candidate*, and taking the argmax --
   this form is provably equivariant (a standard fact: `argmax_g Score(g.x)`
   reparametrizes correctly under the group action, for any Score).

2. **The tiebreak must be translation-invariant, not absolute-pixel-based.**
   The 3 boolean criteria tie far more often than expected -- any pattern
   with a square bounding box (most small GoL patterns, including the
   glider) has equal horizontal/vertical extent under every D4 element, and
   two genuinely different elements (one rotation, one reflection) can
   satisfy all 3 booleans identically. An initial fix (compare exact
   flattened pixel content as a final tiebreak) resolved D4 ties correctly
   but broke translation-equivariance, because absolute pixel content shifts
   with position even when the tie should resolve the same way. Fixed by
   comparing alive-cell coordinates relative to their own bounding-box
   corner instead (translation-invariant, still exact/collision-free).

3. **A head must not be fed the canonicalization label itself.** An early
   version also concatenated `onehot(g*(x))` into the head's input (to
   literally satisfy "these info added to a few nodes"). This breaks
   equivariance: `g*(x)` is *not* invariant across the orbit (only
   `feat_can` is -- that's the whole point of canonicalizing), so feeding it
   into the head makes the head's output depend on which orientation x
   started in. Caught immediately by `sanity_check.py` (check A regressed
   the moment this was added). The pose label is now used *only* to drive
   the external canonicalize/restore transforms, never as head input.

## A non-obvious finding while debugging: canonicalization is provably
redundant for correctness in *this specific* architecture

Because the head is a 1x1 conv (pointwise), it commutes with *any* spatial
relabeling of pixels, not just D4 ops. Working through
`model(g0.x) = Ginv(g*(g0.x)) . head(G(g*(g0.x)) . G(g0) . feat(x))` shows
the canonicalize/restore sandwich algebraically cancels to `g0 . head(feat(x))`
regardless of what `g*(x)` even is -- so full-model D4 + translation
equivariance holds for Model 1 *even with an inconsistently-labeled or
constant `g*`*. Confirmed empirically: `sanity_check.py` checks A/B (full
model equivariance) pass exactly on batches where check D (raw pose-labeling
consistency) shows >50% "mismatches."

This isn't a wasted step, though: (1) the isotropic backbone is *already*
exactly equivariant on its own, so at this scale you don't strictly need
canonicalization for the prediction task either -- but canonicalization is
what turns `feat_can` into a genuinely comparable, pose-normalized
representation for cross-orientation analysis (the actual point of this
representation-similarity line of work), which the raw `feat` is not; and
(2) canonicalization will start to matter for *functional* correctness too
the moment the head gets real spatial extent (a k>1 conv), which later
models in this series may explore.

## Remaining, accepted limitation: symmetric shapes have no consistent label

Patterns with an actual D4 sub-symmetry (block, beehive, blinker, toad,
beacon, pulsar, pentadecathlon -- most of the classic named patterns) have
*multiple* equally-valid canonical orientations, so no deterministic
tiebreak can label them "consistently" across the orbit (same obstruction
as "there's no continuous choice of square root" -- provably impossible in
general, not a bug to fix). Verified: `canonical_gidx` labels the two
*asymmetric* patterns in the stimulus set (glider, lwss) with zero
mismatches across all 8 D4 ops, and for those, `feat_can` is exactly
orientation-*and*-position invariant (0.000 max abs diff, confirming that
what looked like a "position conflation" problem during debugging was
purely a symptom of symmetric-pattern tie-breaking, not a fundamental
limit of orientation-only canonicalization).

## Data

One named pattern (from `../../stimuli.py`'s bitmap set) per grid, random D4
orientation, random placement with a 4-cell margin from the torus edge (so
a single GoL step never wraps -- see `data.py`). 40x40 grid, matching the
rest of the repo.

## Training

Plain BCE-with-logits; no symmetry losses (equivariance is architectural,
not learned). `m=4` hidden channels -> 109 total params. CPU, not MPS/GPU:
`canonical_gidx` does a per-sample CPU/numpy round-trip internally
regardless of device, so MPS dispatch overhead makes it ~10x *slower* than
CPU for this model (271ms/step MPS vs 25ms/step CPU, measured).

Run: `python train.py --epochs 150` -> `results/train_log.txt`,
`results/metrics.json`, `checkpoints/best.pt` / `checkpoints/final.pt`.

**Result:** F1 = 1.0000 (precision = recall = 1.0000, cell accuracy =
1.00000) by epoch 3, held through all 150 epochs. See `results/run1.out`
for the full log. (109-parameter model reaching exact GoL prediction in 3
epochs is consistent with the poly-activation finding this backbone reuses
-- GoL's exact rule is well within reach of a tiny isotropic-conv + poly-
activation network; the pose-canonicalization scaffolding around it adds no
optimization difficulty since it carries no learnable parameters.)

Re-verified equivariance on the *trained* checkpoint (`checkpoints/best.pt`)
against a fresh held-out batch, not just at random init: cell accuracy
1.000000, max D4-equivariance error 7.6e-05, max translation-equivariance
error 3.8e-05 -- both at float32 rounding noise (the squaring term in
`PolyActivation` amplifies small floating-point differences slightly), not a
real violation. The guarantee holds after training exactly as it did at
init, as expected for an architectural (not learned) property.

## Limitation found: does NOT generalize to random dense grids

Training data (`data.py`) is exclusively single-pattern grids -- one named
pattern per grid, so cells with many alive neighbors (5-8) are essentially
absent from training (sparse patterns rarely put a cell next to that many
live neighbors). Random-density grids expose this directly (`visualize_random.py`):

| density | 1-step F1 | precision | recall | cell acc |
|---|---|---|---|---|
| 0.05 | 0.9989 | 1.0000 | 0.9978 | 0.99998 |
| 0.10 | 0.9976 | 1.0000 | 0.9951 | 0.99977 |
| 0.20 | 0.9950 | 1.0000 | 0.9901 | 0.99795 |
| 0.30 | 0.9918 | 1.0000 | 0.9838 | 0.99444 |
| 0.40 | 0.9890 | 1.0000 | 0.9783 | 0.99212 |
| 0.50 | 0.9855 | 1.0000 | 0.9715 | 0.99223 |
| 0.65 | 0.9796 | 1.0000 | 0.9599 | 0.99616 |
| 0.80 | 0.9705 | 1.0000 | 0.9427 | 0.99943 |

**Precision is 1.0000 at every density (zero false positives); recall
degrades monotonically with density (zero false negatives on sparse grids,
climbing to ~5.7% missed positives at density 0.80).** A clean, one-sided
bias -- the model never over-predicts "alive," it under-predicts it more
often as local neighbor counts climb into combinations training never
covered. Under autoregressive rollout on random grids this compounds fast
(GoL is chaotic; `visualize_random.py`'s diff-per-step traces go from
single digits to several hundred cells by step 20 at density >= 0.30),
unlike the single-pattern demos above where diff stays at exactly 0.

This is the same *kind* of finding as the V8 density bias in
[[project-gol-nn-tasks]] Task 13 (over-predicts born on sparse grids, too
many deaths on dense) -- different direction, same root cause: training
distribution didn't cover the neighbor-count combinations the eval
distribution has.

## v2: fixed via mixed-density data + scheduled sampling

Two changes, both requested directly rather than inferred: (1) `data.py`
gained `make_random_batch`/`make_mixed_batch` -- 50% of training grids are
now random-density (`DENSITIES = (0.05,...,0.80)`) instead of exclusively
single-pattern; (2) `train_v2.py` adds scheduled sampling (Bengio et al.
2015) instead of plain 1-step BCE: K=3 unrolled steps, each step's input is
the model's own hard prediction with probability `p` or the true next state
with probability `1-p`, `p` ramped linearly 0->1 over 40 of 60 epochs.
Gradient flows only through the current step's logits, never through the
input-selection decision -- the same design that fixed exposure bias for
the V7 CNN-transformer in [[feedback-gol-nn-training-stability]], chosen
over a naive STE/backprop-through-time unroll for the same reason it won
there (no systematic "predict alive" bias from cascading missed births).

Warmstarted from `checkpoints/best.pt` (v1), LR=1e-4 (much smaller than
v1's 3e-3, since this is a fine-tune, not from-scratch), 60 epochs, ~14s/
epoch on CPU. Saved to `checkpoints/best_v2.pt`.

**Result, verified on fresh seeds not used anywhere in training or
per-epoch eval:**

| density | 1-step F1 (v1 -> v2) |
|---|---|
| 0.05 | 0.9989 -> 1.0000 |
| 0.30 | 0.9918 -> 1.0000 |
| 0.50 | 0.9855 -> 1.0000 |
| 0.80 | 0.9705 -> 1.0000 |
| 0.90, 0.95 | (untested in v1) -> 1.0000 |

40-step autoregressive rollout at densities 0.30/0.50/0.80 (fresh seed):
**0 cell diff at every checkpoint step (1, 5, 10, 20, 30, 40)** -- no
drift at all, versus v1's hundreds-of-cells divergence by step 20 at the
same densities. Re-checked the equivariance guarantee on the new weights
too: max D4 error 2.1e-04, max translation error 2.4e-04 (float32 noise,
same order as v1's post-training check) -- unaffected, as expected for an
architectural property.

Practical note: `best_v2.pt`'s selected checkpoint is from **epoch 1** --
recall hit 1.0000 on the mixed val set almost immediately once random-
density data entered the mix, and the checkpoint-selection score (F1 floor
+ random recall - rollout penalty) saturates at its max from there, so the
first epoch to reach it wins ties. Confirmed this isn't a fluke of the
training-time eval batches by re-testing on entirely fresh seeds above.
Reasonable reading: the *architecture* could already represent the exact
rule (v1 already got 100% on its narrower training distribution in 3
epochs); v2's fixes were purely about which regions of input space it saw
epoch-to-epoch, not about needing to learn a harder function.

## Next steps (not yet done)

- Use `feat` / `feat_can` from this model as the actual object of study:
  compare representations across orientations/patterns the way
  `../analysis_grid_similarity.py` and `../analysis_layerwise.py` do for
  V8/V10 (CKA, RSMs, etc.), now with an architecturally-clean
  invariant-for-asymmetric-shapes baseline to compare a *learned* model
  against.
- model2+: candidates raised during design discussion -- a "learned"
  restore step instead of the exact geometric inverse; extending
  canonicalization to also recenter (translation) so feat_can is invariant
  for symmetric shapes too (not currently possible with a single
  deterministic label, by the argument above -- would need e.g. an
  orbit-averaged or crop-based representation instead of picking one label).
