"""Training data for Model 1: one named GoL pattern per grid, randomly D4-
transformed and placed with a margin from the torus edge (so a single step
never wraps and the pattern's bounding box is unambiguous -- no toroidal
wraparound to reason about for pose_bits()).

Reuses the pattern bitmaps from ../../stimuli.py and the D4 grid ops /
toroidal step from ../../symmetry.py and ../../v1/adapter.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))        # representation_similarity/
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))   # v1/
from stimuli import _PATTERNS, _bitmap        # noqa: E402
from symmetry import d4_ops                   # noqa: E402
from adapter import D4_NAMES, N_D4, gol_step_torch  # noqa: E402

GRID = 40
MARGIN = 4
_OPS = d4_ops()
_BITMAPS = {name: _bitmap(rows) for name, (_cat, rows) in _PATTERNS.items()}

# sanity: every pattern (any D4 orientation) must fit inside GRID-2*MARGIN
_MAX_DIM = max(max(b.shape) for b in _BITMAPS.values())
assert _MAX_DIM <= GRID - 2 * MARGIN, (
    f"pattern too large ({_MAX_DIM}) for margin {MARGIN} on grid {GRID}")


def _random_placed_grid(rng: np.random.Generator) -> np.ndarray:
    name = rng.choice(list(_BITMAPS.keys()))
    bmp = _BITMAPS[name]
    r_idx = int(rng.integers(0, N_D4))
    bmp = _OPS[D4_NAMES[r_idx]](bmp[None])[0]          # apply D4 op, (1,h,w) -> (h,w)
    h, w = bmp.shape
    top = int(rng.integers(MARGIN, GRID - MARGIN - h + 1))
    left = int(rng.integers(MARGIN, GRID - MARGIN - w + 1))
    g = np.zeros((GRID, GRID), dtype=np.float32)
    g[top:top + h, left:left + w] = bmp
    return g


def make_batch(bs: int, rng: np.random.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (x, y): x is (bs,1,GRID,GRID) float {0,1} at t, y is the same
    shape at t+1 (toroidal GoL step)."""
    grids = np.stack([_random_placed_grid(rng) for _ in range(bs)])
    x = torch.from_numpy(grids).unsqueeze(1).to(device)
    y = gol_step_torch(x)
    return x, y


def make_fixed_set(n: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    return make_batch(n, rng, device)


# ── v2 data: mix in random-density grids (fixes the density generalization
# gap found via visualize_random.py / the notes.md density sweep -- single-
# pattern grids essentially never present a cell with 5-8 alive neighbors) ──
DENSITIES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)


def _random_density_grid(rng: np.random.Generator, densities=DENSITIES) -> np.ndarray:
    d = rng.choice(densities)
    return (rng.random((GRID, GRID)) < d).astype(np.float32)


def make_random_batch(bs: int, rng: np.random.Generator, device,
                       densities=DENSITIES) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure random-density grids, no placed patterns."""
    grids = np.stack([_random_density_grid(rng, densities) for _ in range(bs)])
    x = torch.from_numpy(grids).unsqueeze(1).to(device)
    y = gol_step_torch(x)
    return x, y


def make_mixed_batch(bs: int, rng: np.random.Generator, device,
                      random_frac: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample coin flip between a single-pattern grid (data.py's original
    distribution) and a random-density grid, so training covers both the
    sparse structured regime and the high-neighbor-count regime density
    grids expose."""
    grids = np.stack([
        _random_density_grid(rng) if rng.random() < random_frac else _random_placed_grid(rng)
        for _ in range(bs)
    ])
    x = torch.from_numpy(grids).unsqueeze(1).to(device)
    y = gol_step_torch(x)
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = make_batch(8, rng, "cpu")
    print("x", x.shape, x.dtype, "alive/grid:", x.sum(dim=(1, 2, 3)).tolist())
    print("y", y.shape, "alive/grid:", y.sum(dim=(1, 2, 3)).tolist())
