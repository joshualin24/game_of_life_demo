# Representation-vector similarity

Study of how similar the internal representation vectors of the trained
GoL CNN-Transformer models are — across grids, patterns, transformations,
and across the two model scales (V8, 291K params; V10, 1.5M params).

## Models under study

| Model | arch | d_model | heads | layers | params | checkpoint |
|-------|------|---------|-------|--------|--------|------------|
| V8  | CNNTransformerV4 | 64  | 4 | 4 | 291,888   | `nn/checkpoints/task16_cnn_transformer_v8_best.pt` |
| V10 | CNNTransformerV4 | 128 | 8 | 6 | 1,504,240 | `nn/checkpoints/task18_cnn_transformer_v10_best.pt` |

Representation vectors available per grid (see `nn/models.py::CNNTransformerV4.forward`):
- per-patch tokens after the CNN encoder + patch projection  — `(n_patches=100, d_model)`
- per-patch tokens after the transformer encoder             — `(100, d_model)`
- (no CLS token in V4; the reconstruction head is per-patch)

## Questions

- (fill in)

## Layout

- `data/`    — cached grids / extracted representation tensors
- `results/` — figures and numeric summaries
