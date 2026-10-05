# Model 3 — quadrupole-moment canonicalization

## Motivation (user's idea)

Model 2's `canonical_transform` picks orientation from bounding-box
EXTREMES (max/min offset from centroid) -- exactly what made its argmax
flip discontinuously under single-cell perturbation (see
`../model2/notes.md`'s perturbation-sensitivity section: pulsar's mean
cosine similarity was only 0.787 under single-cell flips, versus 0.997+
for the model's prediction accuracy). The proposal: replace the extremes
with the quadrupole moment -- a SUM over every alive cell -- on the theory
that a single flipped cell changes a sum by one bounded increment rather
than potentially replacing which single cell the decision even looks at.
Design, criteria, and the math are below (see "The rotation/reflection
criteria"); this file also documents two real problems found via testing
and discussion, not just the final design.

## v1 (first attempt): fresh shift per candidate, extremes -> moments

Reused Model 2's `canonical_transform` structure verbatim (argmax over 8
rotate-then-recenter candidates, a *fresh* `floor()`-based recentering
shift computed per candidate), changing only the per-candidate score from
bounding-box extremes to quadrupole + cubic moment. All hard checks passed
(exact D4/translation equivariance; exact combined canonicalization on
asymmetric patterns; block/pulsar/beehive all exactly 1.0000 under exact
D4 transforms, same as Model 2).

**Perturbation comparison (v1 design) came back mixed-to-worse**, not
better: pulsar's mean cosine similarity under single-cell flips dropped
from Model 2's 0.787 to 0.709 (worse), and a mid-density random grid
(d0.3) dropped from 0.997 to 0.869 with branch-change count jumping from
0/1600 to 246/1600. Root cause investigation (prompted by the user asking
to double check the shift/rotation order) found **two compounding
problems**, not just "moments weight the periphery":

1. **A genuine measurement artifact.** `_centroid_floor_shift` recenters
   toward the ORIGIN (row~0, col~0) -- which, on a toroidal grid, *is* the
   wraparound seam. Checked directly: after recentering, 2 of glider's 5
   cells landed at row/col 39 (split across the seam), and for a random
   d0.3 grid, 19% of all alive cells ended up within 2px of the seam.
   `_moments` measures distance naively (non-toroidal), so a cell at row
   39 with centroid near row 0.3 is scored as ~39 away when its true
   toroidal distance is ~1.3 -- corrupting the moment sums for a large
   fraction of cells, not just "far" ones in any meaningful geometric
   sense.
2. **Moments really do weight distant cells heavily** (confirmed
   separately): a moment weights each cell's contribution by the square or
   cube of its offset, so a single cell ~20px out contributes 400x-8000x
   more than one near the center; for a roughly-isotropic dense fill,
   `Q_rr`/`Q_cc` are both large and nearly equal, so their difference (the
   "wide" decision) is a fragile residual that one high-leverage cell can
   flip.

Problem (1) likely explains most of the damage; problem (2) is real but
secondary. Neither invalidates the hard correctness checks (argmax-over-
orbit consistency doesn't need the score to be geometrically meaningful,
only to be a well-defined function of each candidate -- which it was), but
both meant the perturbation comparison wasn't testing what it was supposed
to.

## v2 (current): shift ONCE, before rotation is decided -- fixes both

Per explicit request (shift before rotation, not per-candidate), with a
specific additional fix for problem (1) above:

**`canonical_shift`**: computed once, directly from the original `x`,
BEFORE any rotation candidate is considered. Two changes from v1's
`_centroid_floor_shift`, together making a single upfront shift valid:

1. **`round()`, not `floor()`.** Every D4 operation acts on the centroid-
   offset vector by pure component-wise negation and/or permutation (e.g.
   r90 swaps the two components and negates one), never interpolation.
   `round()` with a sign-symmetric tie rule (round-half-to-even, torch's
   default) commutes EXACTLY with negation and permutation applied
   component-wise -- verified directly: `round(flip_c(v)) == flip_c(round(v))`
   for `v=(2.3,-1.7)`, whereas `floor(flip_c(v)) != flip_c(floor(v))`
   (floor is not symmetric under negation -- exactly why v1 needed a fresh
   shift per candidate).
2. **Target the grid's rotation pivot `C=((H-1)/2,(W-1)/2)`, not the
   origin.** This is the actual fixed point `torch_d4`'s rot90/flip/
   transpose all pivot around (confirmed earlier: a shape placed exactly
   at `C` is unaffected by any D4 op). Point 1 only works because the
   recentering target and the rotation pivot are now the SAME point.
   Targeting `C` instead of the origin also moves the toroidal wrap seam
   to the far side of the grid (`C + H/2`) instead of exactly where
   content gets recentered to, fixing problem (1) above as a side effect.

**`canonical_transform`**: shift computed once from `x`; the 8 rotation
candidates are then evaluated ON the already-shifted grid (not 8 fresh
rotate-then-recenter candidates). `ToyCNNModel3.forward` applies shift
first, then rotation; undoes rotation first, then shift (reverse order).

### The rotation/reflection criteria, precisely

For each of the 8 D4 candidates `h` (applied to the shifted grid),
compute, over alive-cell offsets from centroid `(r_off, c_off)`:

```
Q_cc  = sum(c_off^2),  Q_rr  = sum(r_off^2)      # quadrupole (2nd moment)
M3_c  = sum(c_off^3),  M3_r  = sum(r_off^3)      # cubic (3rd moment)

wide  = Q_cc >= Q_rr
right = M3_c >= 0
down  = M3_r >= 0
     + exact relative-coordinate tiebreak (same as Model 1/2, final fallback)
```

argmax over `h` of the lexicographic tuple `(wide, right, down, tiebreak)`.
Quadrupole alone only resolves orientation **mod 180 degrees** -- as a
symmetric rank-2 tensor it's invariant under negating every coordinate (the
same blind spot a PCA/principal-axis angle always has), so it can't tell a
shape from its own point-inversion or reflection. The cubic moment (odd
order, so not identically zero about the centroid the way the 1st moment
always is) breaks that remaining ambiguity. `Q_rc` (the off-diagonal/shear
component) is computed but not used in the decision -- not an oversight:
since canonicalization picks among 8 *already-rotated* discrete candidates
rather than solving for one continuous angle, comparing `Q_cc` vs `Q_rr`
on each rotated copy separately already samples the tensor's orientation
at 8 fixed angles; `Q_rc` would only be needed for a single continuous-
angle computation via `atan2(2Q_rc, Q_cc-Q_rr)/2`, which this design
doesn't do.

### Verification

`sanity_check.py`, all hard checks pass:
- A, B: full-model D4 and translation equivariance (as always, provably
  independent of canonicalization correctness -- the 1x1-head argument).
- **C (new): `shift(g.x) == g.shift(x)` exactly, zero tolerance, all 8 D4
  ops.** This is the property that makes a single upfront shift valid; it
  did not exist as a checkable property in v1 (shift was never meant to be
  used that way there).
- D: combined canonicalization bit-identical across every D4+translation
  combination for asymmetric patterns (glider, lwss) -- 0/96 mismatches.

### A genuinely new, structural finding: pivot-parity mismatch

Exact D4 invariance for the previously-broken symmetric patterns did
**not** fully return to Model 2's 1.0000 across the board:

| pattern | bbox | row/col parity vs pivot (19.5, 19.5) | feat_can cos under D4 |
|---|---|---|---|
| block | 2x2 | even/even -- matches | 1.0000 (all 8, same as Model 2) |
| beehive | 3x4 | odd/even -- one axis mismatches | 0.9526 (down from Model 2's 1.0000) |
| pulsar | 13x13 | odd/odd -- both axes mismatch | 0.8258 (down from Model 2's 1.0000) |

Root cause, confirmed directly: the grid's pivot is at the half-integer
point `(19.5, 19.5)` (even-sized 40-grid). A symmetric pattern's own
natural center lands at an INTEGER coordinate whenever its bounding box has
an ODD dimension. That's an unavoidable 0.5-pixel gap between where the
shape's symmetry is centered and where the grid pivots. `round()` sends
this residual to the identical integer every time regardless of starting
orientation (confirmed: `canonical_shift` still passes check C exactly for
pulsar, shift=(0,0) consistently) -- so it's not an inconsistency bug, it's
a genuine, irreducible rounding residual. Rotation can't compensate for it
either (it's a pure translation gap, not an orientation one), and for
pulsar specifically, its relative shape AND its quadrupole/cubic moments
are *exactly* tied across all 8 D4 elements (verified directly -- `Q_rr`,
`Q_cc`, `M3_r`, `M3_c` identical to several decimal places for every `h`),
so the exact tiebreak falls back to "first in enumeration order" the same
way Model 1's original bug did, just via a different structural cause
(parity mismatch rather than absent translation canonicalization).

This does not affect functional correctness (checks A/B, independent of
this by construction) and is a cleaner, better-understood limitation than
v1's wraparound artifact -- it only affects patterns with real D4 (or
near-D4) sub-symmetry *and* an odd bounding-box dimension, on this specific
even-sized grid. Not fixed here; a finer-than-integer shift (interpolation)
or an odd-sized grid would each trade which patterns are affected, not
eliminate the phenomenon.

## Training and final verification

Warmstarted from Model 2 (identical backbone/head layer shapes), re-trained
after the v2 fix (`results/train_log.txt` / `results/run2.out`;
`results/run1.out` is the superseded v1-design run). Re-verified on fresh,
unseen seeds: F1 = 1.0000 across densities 0.05-0.90; D4/translation
equivariance error 1.46e-3 (float32 noise, doesn't affect any prediction).

## Perturbation sensitivity: final comparison (v2 design)

| stimulus | Model 2 (extremes) | v1 (buggy order) | **v2 (fixed, current)** |
|---|---|---|---|
| glider | 0.9922 | 0.9896 | 0.9892 |
| block | 0.9951 | 0.9955 | 0.9954 |
| pulsar | 0.7871 | 0.7085 (worse) | **0.8504 -- better than Model 2** |
| random d0.1 | 0.9992 (1/1600 branch chg) | 0.9988 (5/1600) | **0.9448 (612/1600) -- new regression** |
| random d0.3 | 0.9967 (0/1600) | 0.8687 (246/1600, seam artifact) | **0.9969 (0/1600) -- artifact gone** |
| random d0.5 | 0.9977 (0/1600) | 0.9987 (0/1600) | 0.9987 (0/1600) |

Two separable findings, not one muddled result:

1. **The wrap-seam fix worked.** `random_d0.3`'s branch-change count
   returned from 246/1600 to 0/1600, matching Model 2 -- that regression in
   v1 really was the measurement artifact (section above), not something
   inherent to moments.
2. **Pulsar genuinely improved over Model 2** once both implementation
   bugs were fixed: 0.8504 vs Model 2's 0.7871 (and v1's buggy 0.7085).
   With the bugs out of the way, quadrupole/cubic moments *do* help on the
   case that originally motivated this model.
3. **`random_d0.1` newly regressed** (612/1600 branch changes, worse than
   both Model 2 and v1). Plausible mechanism, consistent with everything
   else found in this file: with only 153 alive cells, the quadrupole sum
   has few terms, so `Q_rr` vs `Q_cc` is more likely to land near an exact
   tie by chance alone -- the same mechanism that makes pulsar (48 cells,
   exactly tied) fragile. Denser grids (d0.3, d0.5) have enough terms that
   the sums average out and rarely tie; very sparse ones don't. Not
   independently re-verified beyond this plausibility argument -- a natural
   next check if this model is revisited.

**Overall reading:** quadrupole-moment canonicalization is not a strict
improvement over bounding-box extremes -- it trades *which* configurations
are fragile (symmetric and sparse) for others (none obviously better
across the board), rather than reducing fragility in general. The
pulsar win is real and matches the original motivation, but it's a
genuinely mixed result, not a validated win.

## Files

Same self-containment convention as Model 1/2: `IsotropicConv2d`,
`PolyActivation`, `apply_d4_batched`, `apply_shift_batched` copied verbatim
from `../model2/net.py`; `canonical_shift`, `_moments`, and
`canonical_transform` are new/changed. `data.py` is also a verbatim copy.
