# CNN toy models — symmetry-in-representations study

A separate, smaller line of work from the V8/V10 study one level up (see
`../notes.md`). Instead of probing the pretrained CNN-Transformer models,
these are small CNNs trained from scratch, specifically to study what
rotation, reflection, and translation do to *internal layer* representations
-- with an architecture that makes each symmetry's handling explicit and
inspectable, rather than whatever a generic model happens to learn.

There will be more than one model here as the design evolves; each gets its
own numbered subfolder (`model1/`, `model2/`, ...), self-contained (own
`net.py`, `data.py`, `train.py`, `notes.md`, `results/`, `checkpoints/`),
following the `v1/`-style convention used one level up.

## model1/ — isotropic-conv backbone + explicit D4 pose canonicalization

First attempt. Handles translation "for free" via circular (toroidal)
padding (standard). Handles rotation/reflection (D4) via an explicit,
deterministic canonicalization step derived from the bounding box of alive
cells, built on top of an *isotropic* (D4-symmetric-kernel) conv backbone so
the guarantee is architectural, not learned. See `model1/notes.md` for the
full design discussion, the bugs found and fixed along the way, and results.
