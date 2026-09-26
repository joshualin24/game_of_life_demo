# CNN toy models — symmetry in Game of Life representations

A small, from-scratch companion to the V8/V10 CNN-Transformer study one
level up (`../notes.md`): instead of probing pretrained models, these are
tiny CNNs built specifically so that rotation, reflection, and translation
are handled *explicitly and inspectably*, to study what those symmetries do
to internal layer representations.

Two models so far, each in its own self-contained folder (own `net.py`,
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

## Reproducing

Each model folder is self-contained (`pytorch-env` conda environment):

```bash
cd model1  # or model2
python sanity_check.py              # architectural guarantees, must pass before training
python train.py --epochs 150        # model1: v1 (single-pattern data only)
python train_v2.py --epochs 60      # model1: v2 (fixes the density gap) -- model1 only
python train.py --epochs 60         # model2: mixed data + scheduled sampling from the start
python visualize.py                 # rollout / equivariance GIFs, feat_can comparison (model2)
python analysis_representation.py   # D4 (+ translation, model2) invariance report
python analysis_perturbation.py     # model2 only: single-cell-flip sensitivity maps
```

## What's next (not yet done)

- Extend the perturbation study to multi-cell / "move" perturbations,
  matching `../../perturbation_move.py`'s convention.
- A model whose canonicalization is robust to generic perturbation, not
  just exact under exact group transforms (the argmax's decision-boundary
  fragility found above would need a fundamentally different mechanism,
  not just a smarter tiebreak).
- Compare these toy models' representations directly against V8/V10's,
  using the shared tooling in `../common.py` / `../extract.py`.
