# Embedding Study: Game of Life Inputs

## Goal

Study how different encoders record the embedding of Game of Life input.

Given a Game of Life board state (or a short sequence of states) as input, we want
to understand what different encoder architectures capture in their embedding
representations:

- Do embeddings distinguish structurally different patterns (still lifes,
  oscillators, spaceships, chaotic growth)?
- How do embeddings of the same pattern evolve across time steps?
- Are dynamically equivalent states (e.g. phases of the same oscillator, or
  translated copies of a glider) mapped close together in embedding space?

## Encoders to Compare

Candidate encoder families to evaluate on the same set of board states:

- CNN-based encoders (small ConvNets, ResNet-style)
- Vision transformers (ViT with patch embeddings)
- Autoencoder / VAE latent spaces trained directly on board states
- Pretrained image encoders (e.g. CLIP image encoder) applied to rendered boards

## Method Sketch

1. Generate a dataset of board states from known pattern categories using the
   existing simulation code in this repo (`simulate.py`, `pattern_taxonomy.py`).
2. Encode each state with every encoder under study.
3. Analyze the embedding spaces: nearest-neighbor structure, clustering by
   pattern category, trajectory smoothness over time steps, and invariance to
   translation/rotation.
4. Visualize with dimensionality reduction (PCA / UMAP) and compare across
   encoders.

## Open Questions

- Which encoder best separates pattern taxonomy classes without fine-tuning?
- Do embeddings encode the dynamics (future evolution) or only static texture?
