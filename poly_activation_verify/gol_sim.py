"""
Self-contained toroidal (periodic boundary) Conway's Game of Life simulator.

Deliberately independent of the main nn/ project's data_gen module, so this
verification experiment has no dependency on our own codebase.
"""

import numpy as np


def step(grid: np.ndarray) -> np.ndarray:
    """
    grid: (H, W) binary array. Returns next state under standard B3/S23 rules
    with periodic (toroidal) boundary conditions.
    """
    neighbor_count = np.zeros_like(grid, dtype=np.int8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor_count += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    born    = (grid == 0) & (neighbor_count == 3)
    survive = (grid == 1) & ((neighbor_count == 2) | (neighbor_count == 3))
    return (born | survive).astype(np.uint8)


def random_grid(rng: np.random.Generator, size: int = 40, density: float = 0.5) -> np.ndarray:
    return (rng.random((size, size)) < density).astype(np.uint8)
