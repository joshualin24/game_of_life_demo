# Model 4 — controlled ablation: extremes + shift-first

## Motivation

Model 2 (extremes, fresh shift per rotation candidate) and Model 3
(quadrupole/cubic moments, single upfront shift) differ in TWO dimensions
at once, so the perturbation comparison between them (`../model3/notes.md`)
was never a clean test of "extremes vs moments" alone — Model 3's mixed
result (pulsar improved, `random_d0.1` regressed) could have come from
either the scoring criterion or the shift-ordering change. Model 4 isolates
the variable: Model 3's translation mechanism (`canonical_shift`, copied
verbatim -- shift once, before rotation, `round()`-based, pivoting on the
grid's rotation center) + Model 1/2's extreme-based rotation scoring.

## Design

Identical to Model 3 except `_moments`-based scoring is reverted to
`_extents`-based scoring (copied verbatim from `../model1/net.py`). See
`net.py` for the full pipeline; it's byte-for-byte Model 3's structure
with only `_moments(...)` -> `_extents(...)` and the three criteria
changed back to `(x_max-x_min >= y_max-y_min, x_max >= |x_min|,
y_max >= |y_min|)`.

## Verification

`sanity_check.py`: all hard checks pass, including the shift-commutes-
with-D4 check (reused verbatim from Model 3, since `canonical_shift` is
unchanged). Symmetric-pattern D4 invariance: block 1.0000, pulsar 0.8258,
beehive 0.9526 -- **identical to Model 3's numbers**, confirming the
pivot-parity-mismatch finding (`../model3/notes.md`) is a property of the
shift mechanism itself, independent of which rotation-scoring criterion is
used on top of it.

Trained (warmstarted from Model 2, identical layer shapes) and re-verified
on fresh seeds: F1 = 1.0000 across densities 0.05-0.90, equivariance error
at float32 noise floor (~1.1-1.3e-3).

## The controlled comparison (this is the actual point)

Mean `feat_can` cosine similarity under all 1600 single-cell flips:

| stimulus | Model 2 (extremes, old shift) | Model 3 (quadrupole, new shift) | **Model 4 (extremes, new shift)** |
|---|---|---|---|
| glider | 0.9922 | 0.9892 | 0.9892 |
| block | 0.9951 | 0.9954 | 0.9954 |
| pulsar | 0.7871 | 0.8504 | **0.8504 -- identical to Model 3** |
| random d0.1 | 0.9992 (1/1600 branch chg) | 0.9448 (612/1600) | **0.9991 (2/1600) -- matches Model 2** |
| random d0.3 | 0.9967 (0/1600) | 0.9969 (0/1600) | 0.9948 (4/1600) |
| random d0.5 | 0.9977 (0/1600) | 0.9987 (0/1600) | 0.9978 (2/1600) |

**Decisively separates two previously-confounded effects:**

1. **Pulsar's improvement (0.787 -> 0.850) came from the shift-mechanism
   fix, not from quadrupole moments.** Model 4 reproduces Model 3's exact
   pulsar number using extremes, not moments. A single, correctly-pivoted
   upfront shift improves robustness here regardless of what scores
   rotation on top of it.
2. **The `random_d0.1` regression (Model 3's 0.945) is specific to
   quadrupole/cubic moments**, not the shift change. Model 4 stays at
   0.9991, essentially matching Model 2. Confirms the "few alive cells
   means the moment sum has few terms and ties more easily by chance"
   mechanism proposed in `../model3/notes.md` -- extremes don't have this
   problem at the same density.
3. **A smaller, new finding**: Model 4 shows a handful of branch changes
   at d0.3/d0.5 (4/1600, 2/1600) that neither Model 2 nor Model 3 show at
   those densities, and when they occur, cosine similarity craters (0.164,
   0.249 -- worse than anything in Model 2). So the shift-mechanism change
   isn't entirely free even paired with the original scoring criterion;
   it introduces a small number of new, different decision-boundary
   crossings (plausibly because the actual candidates being compared
   differ: Model 2 scores "rotate then freshly recenter," Model 4 scores
   "rotate the already-centered grid" -- the same logic applied to a
   different set of configurations can tie differently at the margins).

**Overall reading:** the three models now cleanly decompose the design
space. The translation-canonicalization *mechanism* (per-candidate vs
single upfront shift) and the rotation *scoring criterion* (extremes vs
moments) each have their own, mostly-independent effects on perturbation
robustness -- the apparent "moments are mixed" result from Model 3 alone
would have been easy to over- or under-credit to the wrong design choice
without this ablation.

## Files

Same self-containment convention as Model 1/2/3: `IsotropicConv2d`,
`PolyActivation`, `apply_d4_batched`, `apply_shift_batched`,
`canonical_shift` copied verbatim from `../model3/net.py`; `_extents`
copied verbatim from `../model1/net.py`. `data.py` is also a verbatim copy.
