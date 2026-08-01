"""
dataset.py — GoL period classification dataset.

Decodes Catagolue apgcodes from census JSON files into fixed-size binary
grids, assigns period labels, and optionally stacks GoL simulation frames
for temporal models.

Augmentations (all period-preserving):
  - Random 90°/180°/270° rotation
  - Random horizontal / vertical flip
  - Random toroidal translation (±8 cells)
"""

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Reuse decode_apgcode and gol_step from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oscillator_gifs import decode_apgcode, gol_step

# ── Constants ────────────────────────────────────────────────────────────────

PERIODS   = [2, 3, 4, 5, 6, 8, 14, 15, 16, 24, 30]
N_CLASSES = len(PERIODS)
P2C       = {p: i for i, p in enumerate(PERIODS)}   # period → class idx
C2P       = {i: p for i, p in enumerate(PERIODS)}   # class idx → period

GRID_SIZE = 64   # fixed canvas for all patterns
DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _place_pattern(pattern: np.ndarray, size: int = GRID_SIZE, pad: int = 4) -> np.ndarray:
    """Centre a pattern in a size×size toroidal canvas."""
    g = np.zeros((size, size), np.uint8)
    h, w = pattern.shape
    needed = max(h, w) + 2 * pad
    if needed > size:
        size = needed + 2
        g = np.zeros((size, size), np.uint8)
    r = (size - h) // 2
    c = (size - w) // 2
    g[r:r + h, c:c + w] = pattern
    return g


def load_all_records(min_samples_per_class: int = 1):
    """
    Load every oscillator from census JSON files.

    Returns
    -------
    records : list of (grid np.uint8 (H,W), period int)
    class_counts : dict {period: count}
    """
    records = []
    class_counts = {p: 0 for p in PERIODS}

    for period in PERIODS:
        path = os.path.join(DATA_DIR, f"census_xp{period}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            objects = json.load(f)

        for obj in objects:
            code = obj["apgcode"].split("_", 1)[-1]
            try:
                pat = decode_apgcode(code)
                grid = _place_pattern(pat)
            except Exception:
                continue
            records.append((grid, period))
            class_counts[period] += 1

    return records, class_counts


def make_split(records, val_frac=0.10, test_frac=0.10, seed=42):
    """
    Stratified train / val / test split.
    Classes with < 3 samples all go to train to avoid empty val/test splits.
    """
    rng = np.random.default_rng(seed)
    by_class = {p: [] for p in PERIODS}
    for rec in records:
        by_class[rec[1]].append(rec)

    train, val, test = [], [], []
    for period, recs in by_class.items():
        recs = list(recs)
        rng.shuffle(recs)
        n = len(recs)
        if n < 3:                    # too few — all in train
            train.extend(recs)
            continue
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test.extend(recs[:n_test])
        val.extend(recs[n_test:n_test + n_val])
        train.extend(recs[n_test + n_val:])

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


# ── Dataset ──────────────────────────────────────────────────────────────────

class GoLPeriodDataset(Dataset):
    """
    Parameters
    ----------
    records   : list of (grid np.uint8 (H,W), period int)
    augment   : bool — apply random rot/flip/translate
    n_frames  : int  — if > 0, simulate n_frames GoL steps and stack as
                       channels: output shape (n_frames+1, H, W)
                       if 0, output shape (1, H, W) — initial state only
    """

    def __init__(self, records, augment=False, n_frames=0):
        self.records  = records
        self.augment  = augment
        self.n_frames = n_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        grid, period = self.records[idx]
        grid = grid.copy().astype(np.float32)

        if self.augment:
            # Toroidal rotation / flip / translate — all period-preserving
            k = np.random.randint(4)
            grid = np.rot90(grid, k).copy()
            if np.random.rand() < 0.5:
                grid = np.fliplr(grid).copy()
            if np.random.rand() < 0.5:
                grid = np.flipud(grid).copy()
            dr = np.random.randint(-8, 9)
            dc = np.random.randint(-8, 9)
            grid = np.roll(np.roll(grid, dr, 0), dc, 1)

        if self.n_frames > 0:
            frames = [grid.copy()]
            g = grid.astype(np.uint8)
            for _ in range(self.n_frames):
                g = gol_step(g)
                frames.append(g.astype(np.float32))
            x = np.stack(frames, axis=0)          # (T+1, H, W)
        else:
            x = grid[np.newaxis]                   # (1, H, W)

        return torch.from_numpy(x), P2C[period]


# ── Sampler ───────────────────────────────────────────────────────────────────

def make_weighted_sampler(records):
    """Inverse-frequency weighted sampler to counteract class imbalance."""
    counts = np.array([0] * N_CLASSES, dtype=float)
    for _, p in records:
        counts[P2C[p]] += 1
    counts = np.where(counts == 0, 1, counts)
    weights_per_class = 1.0 / counts
    sample_weights = np.array([weights_per_class[P2C[p]] for _, p in records])
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
