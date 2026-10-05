# CNN toy models — symmetry in Game of Life representations

A small, from-scratch companion to the V8/V10 CNN-Transformer study one
level up (`../notes.md`): instead of probing pretrained models, these are
tiny CNNs built specifically so that rotation, reflection, and translation
are handled *explicitly and inspectably*, to study what those symmetries do
to internal layer representations.

Six models so far, each in its own self-contained folder (own `net.py`,
`data.py`, `train.py`, `notes.md`, `results/`, `checkpoints/` — the last is
gitignored, regenerate via `train.py`).

## Model 1 — `model1/`: isotropic-conv CNN, D4-only canonicalization

**Architecture:** a ~100-parameter CNN (`IsotropicConv2d` — a 3x3 conv
sharing one weight across the 4 orthogonal neighbors and one across the 4
diagonal neighbors, instead of 9 free weights — plus a learnable
polynomial activation) that is **exactly D4-equivariant by construction**,
combined with circular padding for exact translation equivariance. On top
of that, a deterministic "pose code" — computed from the bounding box of
alive cells — picks a canonical D4 orientation, canonicalizes the internal
feature map before a shared 1x1 head, then restores the original
orientation on the way out.

**What it's for:** `feat_can` (the canonicalized feature map, right before
the head) is meant to be a genuinely *comparable* representation across
differently-oriented views of the same content — not just a correctly-
processed one.

**Key results:**
- 109 parameters, F1 = 1.0000 on Conway's Game of Life next-state
  prediction (exact rule, not an approximation).
- D4 and translation equivariance of the *prediction* are exact,
  architecturally guaranteed — proven true regardless of whether the pose
  canonicalization even gets the "right" answer, because the head is a 1x1
  conv (commutes with any spatial relabeling).
- `feat_can` is exactly invariant across all 8 D4 transforms for
  **asymmetric** patterns (glider, lwss) and for random-density grids —
  but genuinely **not** for patterns with real D4 sub-symmetry (block,
  pulsar, beehive) placed off the grid's own rotation center: cosine
  similarity as low as **0.835** there. Root cause, found by direct
  inspection: a symmetric shape gives the canonicalization nothing to
  detect, so it always answers "no correction needed" — but rotating the
  whole grid still moves an off-center object, and nothing undoes that.
- Found and fixed, along the way: a v1 training run (single-pattern data
  only) had a real density-generalization gap (recall dropped from 0.998 at
  low density to 0.943 at density 0.80, precision always 1.0). Fixed in v2
  by mixing random-density grids into training and adding scheduled
  sampling for multi-step rollout stability — verified afterward: F1 =
  1.0000 from density 0.05 to 0.95, zero rollout drift over 40
  autoregressive steps.

Full design history, both bugs, and all numbers: `model1/notes.md`.
Demo: [Isotropic GoL Net](https://claude.ai/artifact/A7fDPiqkXcKm2r1Tvy44Gm).

## Model 2 — `model2/`: adds translation canonicalization

**Why:** Model 1 relied on circular padding for translation *equivariance*,
but never canonicalized *position* — equivariant (moves consistently with
the input) and invariant (looks the same regardless of where content sits)
are different properties, and the symmetric-pattern gap above was exactly
a translation-invariance gap in disguise.

**What changed:** for each of the 8 D4 candidates, compute its own
recentering shift fresh, form the full rotate-then-recenter candidate, and
pick the winner by the same scoring rule Model 1 used for D4 alone — the
same "argmax over the orbit" principle, extended to the richer combined
group. An earlier, more obvious-looking approach (compute the D4 choice and
the recentering shift independently, then compose them) was **wrong** —
caught by testing before it shipped, since `floor(centroid)` doesn't
commute with rotation, and it broke even the asymmetric patterns Model 1
got exactly right.

**Key results:**
- Architecture and parameter count (109) unchanged; weights transfer
  directly from Model 1 v2 via warmstart.
- Fixed exactly: block, pulsar, and beehive all now show 1.0000 `feat_can`
  cosine similarity across all 8 D4 ops (pulsar was Model 1's worst case,
  0.835). Verified for all 10 named patterns and 4 random-density grids.
- Re-verified after training on fresh, unseen seeds: F1 = 1.0000 across
  densities 0.05–0.90, equivariance error at float32 noise floor, zero
  rollout drift over 40 steps.
- **A real, principled scope limit found by testing further:** translation
  canonicalization breaks down for dense random-density grids at *any*
  shift, even one pixel — not the familiar large-shift-wraps-around-the-
  edge caveat, but something more fundamental. A space-filling distribution
  has no localized "position" for a centroid-based scheme to canonicalize;
  D4 canonicalization doesn't have this problem (confirmed exact even for
  dense grids) because rotating a finite grid never causes wraparound the
  way a translation roll does.
- **Perturbation-sensitivity analysis** (`model2/analysis_perturbation.py`,
  adapting `../../perturbation.py`'s classic single-cell-flip study to the
  representation instead of the output): flipping any one of 1600 cells
  reveals that the discrete argmax the canonicalization uses has real
  decision boundaries. For random-density grids, the representation is
  smooth almost everywhere with rare (0-2 of 1600 cells), sharp, *exactly*
  localized discontinuities precisely where the canonicalization's choice
  flips. Pulsar is the standout bad case: mean cosine similarity only 0.787
  under single-cell perturbation (worst: 0.726), because sitting near an
  exact symmetry means ~87% of all possible single-cell flips cross a
  decision boundary. Takeaway: exact invariance under *exact* group
  transforms (what Model 2 targeted) and robustness under *generic*
  perturbation are different properties, and proximity to a symmetric
  configuration predicts fragility in the latter.

Full design history, the bug and its fix, and all numbers: `model2/notes.md`.
Demo: [Translation-Invariant GoL Net](https://claude.ai/artifact/4McBndijqNVjA22PfbRPUn).

## Model 3 — `model3/`: quadrupole-moment canonicalization

**Why:** Model 2's canonicalization picks orientation from bounding-box
*extremes* (max/min offset from centroid) — exactly what made it flip
discontinuously under single-cell perturbation (pulsar: mean cosine
similarity only 0.787 under single-cell flips). Proposal (the user's idea):
score orientation with the quadrupole moment — a *sum* over every alive
cell — on the theory that a single flipped cell changes a sum by one
bounded increment rather than potentially replacing which single cell the
decision even looks at.

**What changed:** orientation is now chosen from `Q_cc = Σc_off²` vs
`Q_rr = Σr_off²` (quadrupole/2nd moment — "wide vs tall" by total spread,
not by the two most extreme cells), with a cubic moment (`Σc_off³`,
`Σr_off³`) resolving the remaining mod-180° ambiguity quadrupole alone
can't see (a symmetric 2nd-moment tensor is blind to point inversion,
the same blind spot a PCA/principal-axis angle always has). Translation
canonicalization was also rebuilt to shift once, *before* rotation is even
decided (per explicit request), using `round()` instead of `floor()` and
pivoting on the grid's actual rotation center instead of the origin — the
origin turned out to *be* the toroidal wrap seam, which was corrupting the
very moments being computed on recentered content.

**Two real bugs found via testing, both now fixed and documented in
`model3/notes.md`:**
1. An initial "shift once, then try rotations" design broke even the
   asymmetric patterns Model 2 got exactly right, because `floor()`
   doesn't commute with rotation (same class of bug Model 2 found, in a
   new spot) — fixed by using `round()`, which *does* commute with the
   negation/permutation every D4 operation reduces to.
2. A **measurement artifact**: recentering toward the origin placed
   content right at the wraparound seam, so naive (non-toroidal) distance
   math fed wildly wrong values into the moments for cells landing near
   row/col 0 — not a geometric property of "far" cells, an artifact of
   where the recentering target happened to sit. Fixed by pivoting on the
   grid's rotation center instead (confirmed: the artifact's symptom,
   `random_d0.3` showing 246/1600 spurious decision-boundary crossings,
   disappeared back to 0/1600 once fixed).

**Key results, final (bugs fixed):**
- 109 parameters, same architecture, weights transfer directly from
  Model 2 via warmstart. F1 = 1.0000 across densities 0.05–0.90.
- A new hard guarantee verified exactly (zero tolerance): `shift(g.x) ==
  g.shift(x)` for every D4 element — proof that a single upfront shift is
  valid here, unlike Model 2's extremes/floor-based version.
- **A genuinely new structural finding**, not a bug: patterns with an odd
  bounding-box dimension (pulsar 13×13, beehive 3×4) can't align their own
  symmetric center with the grid's half-integer rotation pivot — an
  unavoidable 0.5-pixel rounding residual that neither more careful
  shifting nor smarter rotation can close. Block (2×2, even/even) stays
  exactly 1.0000; pulsar and beehive don't fully return to Model 2's
  1.0000 (0.826 / 0.953) for this reason.
- **Perturbation comparison, mixed, not a clean win:** pulsar genuinely
  improved over Model 2 (0.850 vs 0.787 mean cosine similarity under
  single-cell flips) — the original motivation held up once the bugs were
  fixed. But a sparse random grid (density 0.1) newly regressed (0.945,
  worse than Model 2's 0.999), plausibly because fewer alive cells means
  the quadrupole sum has fewer terms and ties more easily by chance — the
  same mechanism that makes pulsar fragile.
- **Important caveat, resolved by Model 4 below:** this comparison
  confounds two things that changed at once (the shift mechanism *and* the
  scoring criterion). Model 4 isolates them and shows the pulsar
  improvement actually came from the shift fix, not from quadrupole
  moments — see Model 4's entry.

Full design history (both bugs, the exact math, and the final numbers):
`model3/notes.md`.

## Model 4 — `model4/`: controlled ablation (extremes + shift-first)

**Why:** Model 2 and Model 3 differ in two dimensions at once — the
translation mechanism (fresh shift per rotation candidate vs a single
upfront shift) and the rotation-scoring criterion (extremes vs moments) —
so Model 3's perturbation comparison was never a clean test of either
alone. Model 4 = Model 3's shift mechanism (`canonical_shift`, copied
verbatim) + Model 1/2's extreme-based rotation scoring, isolating the
scoring criterion as the only remaining variable.

**The 3-way comparison, mean `feat_can` cosine similarity under all 1600
single-cell flips:**

| stimulus | Model 2 (extremes, old shift) | Model 3 (quadrupole, new shift) | Model 4 (extremes, new shift) |
|---|---|---|---|
| pulsar | 0.7871 | 0.8504 | **0.8504 — identical to Model 3** |
| random d0.1 | 0.9992 (1/1600 branch chg) | 0.9448 (612/1600) | **0.9991 (2/1600) — matches Model 2** |
| random d0.3 | 0.9967 (0/1600) | 0.9969 (0/1600) | 0.9948 (4/1600) |

**Decisive result:** pulsar's improvement came entirely from the
shift-mechanism fix — Model 4 reproduces Model 3's exact 0.8504 using
extremes, not moments. The `random_d0.1` regression is specific to
quadrupole/cubic moments — Model 4 doesn't show it at all, confirming the
"few alive cells → few terms in the moment sum → easier accidental ties"
mechanism. A smaller new finding: Model 4 shows a handful of *new*
decision-boundary crossings at d0.3/d0.5 that neither Model 2 nor Model 3
show, with catastrophic cosine drops when they occur (as low as 0.164) —
the shift-mechanism change isn't entirely free even paired with the
original scoring criterion.

**Takeaway:** the quadrupole swap, isolated from the shift fix, is a net
negative on this evidence — it doesn't fix anything the shift fix alone
didn't already fix, and it adds a new sparse-grid regression. The
translation-canonicalization *mechanism* and the rotation *scoring
criterion* are separable design choices with their own, largely
independent effects — worth remembering before crediting any future result
to the wrong one.

Full design history and all numbers: `model4/notes.md`.

## Model 5 — `model5/`: learned canonicalization (fixed a real collapse, partial success)

**Why:** Models 1-4 all hand-coded the canonicalization. Model 5 asks
whether a network can learn it instead: Part A is Model 4's backbone with
the hand-coded canonicalization stripped out; Part B is a new
`PosePredictor` (sees the full grid, predicts a shift + D4 element) that
should learn to replace it. Trained by distilling from Model 4 as a frozen
teacher (`L_task` + `L_feat` + direct pose supervision `L_pose`), with
scheduled sampling between the teacher's pose and Part B's own prediction.

**Headline finding, true in both versions below: task accuracy cannot
measure whether Part B learned anything.** `combine()`'s shift/rotate-
then-undo is, for *any* self-consistent pose (correct, wrong, even a
constant), mathematically identical to just computing `head(feat)`
directly — a direct consequence of the same "1x1 head commutes with
relabeling" identity every prior model's exact-equivariance guarantee
already relies on. Verified directly: feeding the same `feat` through
Part B's own predicted pose, no pose at all, and a deliberately wrong
constant pose all produced logits identical to float32 roundoff
(~2.4e-4). So "the model should predict t+1 correctly" (the user's stated
combined-success criterion) is satisfied by Part A alone, by construction,
regardless of Part B — task F1 = 1.0000 and zero rollout drift hold in
both versions and are not evidence either way.

**v1 collapsed.** `gidx_acc`/`shift_mae` against the teacher were frozen
at their epoch-1 values for all 80 epochs; the 8-way rotation classifier
only ever output 2 of 8 classes; pose-consistency-under-D4 read exactly
0.125 (chance). Root cause, confirmed by direct measurement: coordinate
channels are injected at every grid position, dead or alive, so a plain
global-SUM pool over 1600 mostly-dead positions was dominated by that
position-only background — for sparse grids, 94-99.8% of the pooled
vector's norm came from dead cells, leaving no real per-sample signal for
the heads to learn from.

**v2 fixed the pooling** — masked mean over alive cells only
(`(h*x).sum(spatial)/alive_count`, with `log1p(alive_count)` added back
explicitly), mirroring the hand-coded centroid/extent formulas already in
Models 2-4. Same hyperparameters otherwise, to isolate the fix as the only
variable. Result: a large, genuine improvement that does not fully
converge —

| metric | v1 (collapsed) | v2 (fixed pooling) |
|---|---|---|
| `shift_mae` | frozen at 7.108 | **0.163** (fresh batch: 0.105) |
| `gidx_acc`, pattern data | frozen at 0.489 | **0.726** |
| `gidx_acc`, random data | frozen at 0.117 | **0.289** (fresh batch: 0.340) |
| pose-consistency under D4 | exactly 0.125 (chance) | **0.217** |
| `gidx_used` class usage | only 2 of 8 classes | **all 8 classes**, tracking the teacher's distribution |

Shift regression is essentially solved. Rotation/reflection
classification moved well past chance and the collapse is gone, but it
falls well short of the teacher's exact convention, and the shortfall is
interpretable, not uniform: the per-D4-element breakdown shows the 4 pure
rotations learned reasonably (`r90` agreement 0.297) while the 4
reflection-involving elements score *below* chance (`flip_v`/`flip_h`
~0.03-0.05) — this architecture, unlike `IsotropicConv2d`, has no
structural bias toward detecting mirror symmetry, and evidently confuses
flips with their paired rotation. Perturbation analysis
(`analysis_perturbation.py`) correspondingly shows real decision-boundary
behavior again (e.g. `random_d0.1`: 48/1600 cells cross a boundary, cosine
dropping to ~0.62-0.73 there) instead of v1's uniformly-flat 0-branch-
changes-everywhere signature — pulsar's 0.8315 mean cosine similarity is
in the same ballpark as Model 4's own 0.8504, though not an exact match.

Full design history, both versions' verification details, and the
complete (honest) results: `model5/notes.md`.

## Model 5v3 — `model5v3/`: shared trunk -- letting the two parts talk

**Why:** v1/v2 above kept Part A (dynamics) and Part B (pose) as two
fully independent networks, gradient-isolated by construction. The user
asked for something different: a design where the two functionally
distinct parts are **allowed to talk to each other** (like Model 4's
backbone, with "some neurons" for canonicalization and "some others" for
dynamics) while staying distinguishable, and wanted to know whether such
a model can learn Model 4's representation *while keeping* t+1 accuracy.

**Design:** a SHARED TRUNK — `IsotropicConv2d(1,2m)` + `PolyActivation`,
literally Model 4's first layer — consumed by BOTH a dynamics head
(exactly Model 4's tail) and a pose head (the same masked-mean-pool
design as v2, now reading the trunk's own features instead of raw `x`).
The pose head's gradient is **not** detached from the trunk, so
`L_task`/`L_feat` (via the dynamics head) and `L_pose` (via the pose
head) both shape the trunk's weights — verified directly: a loss computed
purely from the pose head's output produces nonzero gradient on the
trunk at random init. Coordinate channels are appended only on the pose
head's own input, never the trunk's, preserving the trunk's exact
D4-equivariance-by-construction. Same training recipe as v2 otherwise
(warmstart, three losses, scheduled sampling, 80 epochs), to isolate the
shared trunk as the only variable.

**Headline finding: a genuine multi-axis tradeoff, not a clean win.**

| metric | v2 (isolated) | v3 (shared trunk) |
|---|---|---|
| pose-consistency under D4 | 0.217 | **0.268** |
| `gidx_acc`, pattern data | 0.726 | **0.978** |
| `gidx_acc`, random data (fresh) | 0.340 | 0.168 (**worse**) |
| `shift_mae` (fresh) | 0.105 | 0.124 (~same) |
| task F1, fresh seeds | 1.0000 | **0.959–0.988** (regression) |
| rollout diff @ 20 steps | 0 | **326 / 1600 cells** (major regression) |

Pose learning genuinely improved on several axes — most strikingly,
pulsar's full D4-orbit now reproduces `feat_can` cosine similarity
**1.0000** (v2: 0.8426), and named patterns (glider, lwss, block) land at
≈0.99-1.0000 across all 8 D4 ops, even where the model's specific
`gidx_used` choice doesn't match Model 4's convention — a new
`analysis_representation.py` (this model family's first) documents this
directly. But task accuracy, which v1/v2 got "for free" via `combine()`'s
verified pose-independence, genuinely broke: tested directly by
bypassing `combine()` entirely, `head(feat)` alone scores F1 = 0.925 on a
sparse random grid, something Model 4's identical architecture never did.
Root cause: the shared trunk has only 56 parameters, now serving two
competing objectives at once, and pose learning evidently won a larger
share of that tiny capacity than task accuracy could afford to lose.
Perturbation sensitivity is similarly mixed — sparse random data (d0.1)
got dramatically more stable (cosine 0.9969 vs v2's 0.8780), but glider
and block, previously rock-solid in v2, now flip their predicted pose on
~98% of single-cell perturbations (1570-1575/1600, vs v2's 3-6/1600).

Full design rationale, the complete side-by-side v1/v2/v3 comparison, and
all verification details: `model5v3/notes.md`. GIFs (rollout vs. Model 4
and Model 5v2, D4-orbit representation comparison, perturbation
difference example): `results_model5v3_vs_others/`, generated by
`visualize_model5v3_vs_others.py`.

## Reproducing

Each model folder is self-contained (`pytorch-env` conda environment):

```bash
cd model1  # or model2 / model3 / model4 / model5 / model5v3
python sanity_check.py              # architectural guarantees, must pass before training
python train.py --epochs 150        # model1: v1 (single-pattern data only)
python train_v2.py --epochs 60      # model1: v2 (fixes the density gap) -- model1 only
python train.py --epochs 60         # model2/3/4: mixed data + scheduled sampling from the start
python train.py --epochs 80         # model5/model5v3: distillation from model4 + scheduled sampling over pose
python visualize.py                 # rollout / equivariance GIFs, feat_can comparison -- model1/2 only
python analysis_representation.py   # D4 (+ translation) invariance report -- model1/2/5v3
python analysis_perturbation.py     # single-cell-flip sensitivity maps -- model2/3/4/5/5v3
```

`visualize_model5v3_vs_others.py` (at the `CNN_toy_model/` level, needs
`model4`, `model5`, and `model5v3` all trained) generates the 3-way
rollout comparison, D4-orbit representation-similarity, and
perturbation-difference GIFs discussed in Model 5v3's section above.

## What's next (not yet done)

- Resolve Model 5v3's shared-trunk tradeoff: task accuracy and rollout
  stability regressed (F1 0.96-0.99, rollout diff 326/1600 @ 20 steps,
  vs. v1/v2's always-exact 1.0000/0) in exchange for better pose learning
  on several axes. Try giving the trunk more capacity, or a loss-weighting
  schedule that protects task accuracy once pose learning has converged
  enough — not attempted here, a new scoped iteration.
- Close Model 5(v2)'s remaining gap: rotation/reflection classification
  (`gidx_acc` ~0.29-0.73 depending on data regime) still falls well short
  of the teacher's exact convention after the pooling fix, and
  specifically underperforms on reflections vs. pure rotations — try
  longer training now that gradient signal is real, more capacity for the
  pose head, or an architectural bias toward detecting mirror symmetry
  specifically (the thing `IsotropicConv2d` has that `PosePredictor`
  doesn't) — not attempted here.
- A more direct way to evaluate a *learned* canonicalization's success,
  given Model 5's finding that task accuracy structurally can't do it:
  the pose-supervision metrics (`gidx_acc`, `shift_mae`) and the
  pose-consistency-under-D4 check are the only signals that worked here —
  worth building into any future learned-canonicalization model from the
  start, not added after the fact.
- Extend the perturbation study to multi-cell / "move" perturbations,
  matching `../../perturbation_move.py`'s convention.
- A model whose canonicalization is robust to generic perturbation, not
  just exact under exact group transforms. The Model 3/4 ablation shows
  that swapping the scoring statistic alone (extremes -> moments) doesn't
  get there — it moves the fragility around (trading pulsar for sparse
  grids) rather than removing it, and the shift-mechanism choice matters
  at least as much as the scoring criterion did. The argmax's decision-
  boundary nature itself (picking a winner from finitely many discrete
  candidates) seems like the actual thing that would need to change, not
  just which statistic breaks the tie or how translation is handled.
- Investigate Model 4's new d0.3/d0.5 decision-boundary crossings (absent
  in both Model 2 and Model 3) more directly -- the "different candidates
  being compared" explanation in `model4/notes.md` is plausible but not
  independently verified.
- Compare these toy models' representations directly against V8/V10's,
  using the shared tooling in `../common.py` / `../extract.py`.
