"""
Trajectory dataset for transformer embedding experiments.

Grid: 40×40 toroidal, T=60 steps → sequence of 61 frames per sample.

Data composition (configurable, default):
  40% — random grids  (density uniformly drawn from [0.2, 0.5])
  60% — named-pattern grids:
          ~40% single pattern  (random augmentation + random placement)
          ~40% two patterns    (non-overlapping)
          ~20% three patterns  (non-overlapping)

Augmentation (dihedral D4):  4 rotations × 2 reflections = 8 orientations.

Each __getitem__ call runs the GoL simulation on-the-fly from the stored
initial state, so memory usage is just n_samples × H × W bytes (~8 MB for
5000 samples on a 40×40 grid).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ── Pattern library (cells arrays only) ───────────────────────────────────────

def _u8(*rows):
    return np.array(rows, dtype=np.uint8)

PATTERN_CELLS = {
    "block":       _u8([1,1],[1,1]),
    "beehive":     _u8([0,1,1,0],[1,0,0,1],[0,1,1,0]),
    "blinker":     _u8([1,1,1]),
    "toad":        _u8([0,1,1,1],[1,1,1,0]),
    "beacon":      _u8([1,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,1]),
    "pulsar":      np.array([
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
    ], dtype=np.uint8),
    "glider":      _u8([0,1,0],[0,0,1],[1,1,1]),
    "lwss":        _u8([0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]),
    "r_pentomino": _u8([0,1,1],[1,1,0],[0,1,0]),
    "acorn":       _u8([0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]),
}

PATTERN_CATEGORY = {
    "block": "still_life", "beehive": "still_life",
    "blinker": "oscillator", "toad": "oscillator",
    "beacon": "oscillator", "pulsar": "oscillator",
    "glider": "spaceship", "lwss": "spaceship",
    "r_pentomino": "methuselah", "acorn": "methuselah",
}

PATTERN_NAMES = list(PATTERN_CELLS.keys())


# ── GoL engine ─────────────────────────────────────────────────────────────────

def _gol_step(grid: np.ndarray) -> np.ndarray:
    n = sum(
        np.roll(np.roll(grid, i, 0), j, 1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((n == 3) | (grid & (n == 2))).astype(np.uint8)


def run_trajectory(init: np.ndarray, T: int) -> np.ndarray:
    """Return (T+1, H, W) uint8 trajectory."""
    traj = np.empty((T + 1, *init.shape), dtype=np.uint8)
    traj[0] = init
    for t in range(1, T + 1):
        traj[t] = _gol_step(traj[t - 1])
    return traj


# ── Augmentation ───────────────────────────────────────────────────────────────

def _augment(cells: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random element of the dihedral group D4 (4 rotations × 2 reflections)."""
    k = rng.integers(0, 4)
    cells = np.ascontiguousarray(np.rot90(cells, k=int(k))).astype(np.uint8)
    if rng.random() < 0.5:
        cells = np.ascontiguousarray(np.fliplr(cells)).astype(np.uint8)
    return cells


# ── Pattern placement ──────────────────────────────────────────────────────────

def _try_place(grid: np.ndarray, cells: np.ndarray, rng: np.random.Generator,
               max_tries: int = 30) -> bool:
    """Try to place `cells` at a random non-overlapping position. Returns True if placed."""
    H, W = grid.shape
    ph, pw = cells.shape
    if ph > H or pw > W:
        return False
    for _ in range(max_tries):
        r = int(rng.integers(0, H - ph + 1))
        c = int(rng.integers(0, W - pw + 1))
        region = grid[r:r+ph, c:c+pw]
        if not np.any(region & cells):
            grid[r:r+ph, c:c+pw] |= cells
            return True
    return False


def _make_pattern_grid(size: int, n_patterns: int, rng: np.random.Generator,
                       ) -> tuple[np.ndarray, list[str]]:
    """Build a grid with 1–3 randomly chosen, augmented, non-overlapping patterns."""
    grid = np.zeros((size, size), dtype=np.uint8)
    names_used: list[str] = []

    chosen = rng.choice(PATTERN_NAMES, size=n_patterns, replace=True)
    for name in chosen:
        cells = _augment(PATTERN_CELLS[name].copy(), rng)
        placed = _try_place(grid, cells, rng)
        if placed:
            names_used.append(name)

    # Fallback: if nothing was placed (extremely rare), use a small single pattern
    if not names_used:
        cells = PATTERN_CELLS["glider"].copy()
        r, c = int(rng.integers(0, size - 3)), int(rng.integers(0, size - 3))
        grid[r:r+3, c:c+3] |= cells
        names_used = ["glider"]

    return grid, names_used


def _make_random_grid(size: int, rng: np.random.Generator) -> np.ndarray:
    density = float(rng.uniform(0.2, 0.5))
    return (rng.random((size, size)) < density).astype(np.uint8)


# ── Dataset ────────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Returns (T+1, H, W) float32 trajectory tensors in [0, 1].

    Metadata per sample (for analysis, not used during training):
      self.meta[i] = {"type": "random"|"single"|"multi",
                      "patterns": [list of pattern names],
                      "category": "random"|"still_life"|...|"mixed"}
    """

    def __init__(
        self,
        n_samples: int = 5000,
        grid_size: int = 40,
        T: int = 60,
        random_frac: float = 0.4,
        seed: int = 42,
        precompute: bool = True,
    ):
        self.T = T
        self.H = self.W = grid_size

        rng = np.random.default_rng(seed)
        n_random  = round(n_samples * random_frac)
        n_pattern = n_samples - n_random

        # pattern-split: 40% single, 40% dual, 20% triple (of the pattern portion)
        n_single = round(n_pattern * 0.40)
        n_dual   = round(n_pattern * 0.40)
        n_triple = n_pattern - n_single - n_dual

        self.inits: list[np.ndarray] = []
        self.meta:  list[dict]       = []

        for _ in range(n_random):
            self.inits.append(_make_random_grid(grid_size, rng))
            self.meta.append({"type": "random", "patterns": [], "category": "random"})

        for n_pat, tag in [(n_single, "single"), (n_dual, "dual"), (n_triple, "triple")]:
            np_ = 1 if tag == "single" else (2 if tag == "dual" else 3)
            for _ in range(n_pat):
                g, names = _make_pattern_grid(grid_size, np_, rng)
                cats = {PATTERN_CATEGORY[nm] for nm in names}
                category = next(iter(cats)) if len(cats) == 1 else "mixed"
                self.inits.append(g)
                self.meta.append({"type": tag, "patterns": names, "category": category})

        mem_mb = n_samples * (T + 1) * grid_size * grid_size / 1e6
        print(f"[TrajectoryDataset] {len(self.inits)} samples  "
              f"({n_random} random, {n_pattern} pattern-based)  "
              f"grid={grid_size}×{grid_size}  T={T}")

        # Pre-compute all trajectories once to avoid re-running GoL every epoch.
        # Uses ~{mem_mb:.0f} MB RAM but makes each training batch instant.
        self._trajs: np.ndarray | None = None
        if precompute:
            print(f"  Pre-computing trajectories (~{mem_mb:.0f} MB) …", flush=True)
            self._trajs = np.empty(
                (len(self.inits), T + 1, grid_size, grid_size), dtype=np.uint8)
            for i, init in enumerate(self.inits):
                self._trajs[i] = run_trajectory(init, T)
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1}/{len(self.inits)}", flush=True)
            print("  Done.", flush=True)

    def __len__(self) -> int:
        return len(self.inits)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._trajs is not None:
            return torch.from_numpy(self._trajs[idx]).float()
        traj = run_trajectory(self.inits[idx], self.T)   # (T+1, H, W) uint8
        return torch.from_numpy(traj).float()


def make_trajectory_loaders(
    n_samples: int = 5000,
    grid_size: int = 40,
    T: int = 60,
    random_frac: float = 0.4,
    val_frac: float = 0.15,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 2,
    precompute: bool = True,
) -> tuple[DataLoader, DataLoader, TrajectoryDataset]:
    ds = TrajectoryDataset(n_samples, grid_size, T, random_frac, seed, precompute)
    n_val   = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl, ds
