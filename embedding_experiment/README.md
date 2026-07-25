# Embedding Experiment: Trajectory-Invariant Representations of Game of Life

## Goal

Study whether it is possible to learn an embedding space where all states
along the same GoL trajectory — i.e. states that share the same initial
condition — collapse to the **same** (or very similar) point in embedding space.

In other words: can an encoder learn a representation that is **invariant to
time evolution**, such that

```
embed(S_0) ≈ embed(S_1) ≈ embed(S_2) ≈ ... ≈ embed(S_T)
```

where S_0 → S_1 → ... → S_T is a single GoL trajectory?

## Why This Is Interesting

- Standard VAEs embed individual frames independently — nearby frames in a
  trajectory may end up far apart in embedding space.
- A trajectory-invariant embedding would capture what is **conserved** along
  evolution: the "identity" of a pattern lineage, not its instantaneous state.
- This is related to learning attractors, conserved quantities, and symmetries
  in dynamical systems.
- If successful, the embedding space could be used to:
  - Cluster trajectories by their long-run behavior (still life, oscillator,
    extinction, chaos)
  - Detect when two different initial conditions converge to the same attractor
  - Predict final fate from early states

## Approach

We treat this as a **metric learning / contrastive learning** problem:

- **Positive pairs**: two frames from the *same* trajectory → pull embeddings together
- **Negative pairs**: frames from *different* trajectories → push embeddings apart

The encoder is trained with a **trajectory contrastive loss** (NT-Xent / SimCLR-style
or triplet loss) where the "augmentation" is simply time evolution.

Optionally, a reconstruction head can be added (making it a contrastive VAE)
to ensure the embeddings remain interpretable.

## Directory Structure

```
embedding_experiment/
├── README.md               # this file
├── setup.py                # environment check + data generation entrypoint
├── generate_trajectories.py  # GoL trajectory dataset generator
├── model.py                  # encoder + projection head
├── train.py                  # contrastive training loop
├── evaluate.py               # evaluate trajectory clustering quality
└── data/                     # generated trajectory datasets (gitignored)
└── results/                  # checkpoints, plots (gitignored)
```

## Quick Start

```bash
# 1. Check environment and generate trajectory data
python setup.py

# 2. Train the trajectory-invariant encoder
python train.py

# 3. Evaluate embedding quality
python evaluate.py
```

## Key Questions

1. Does the contrastive loss actually collapse trajectory states together?
2. Does the embedding still separate *different* trajectories?
3. Is there a trade-off between intra-trajectory invariance and inter-trajectory
   discriminability?
4. Do oscillators (period-2, period-3) form tight clusters or do their phases
   separate them?
5. Can the embedding predict the eventual fate of a random soup trajectory?
