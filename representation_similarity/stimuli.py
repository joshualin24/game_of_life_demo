"""Stimulus grid set for the representation-similarity study.

Structured patterns (still lifes, oscillators, spaceships) at several
placements and phases, plus random grids binned by density. Every grid
carries a metadata dict: category / name / period / phase / density / placement.

    grids, meta = build_stimuli()
    # grids: (N, 40, 40) uint8   meta: list[dict] of length N
"""
from __future__ import annotations

import numpy as np

from common import gol_step, detect_period

GRID = 40

# ── pattern bitmaps (O = alive) ─────────────────────────────────────────────
_PATTERNS: dict[str, tuple[str, list[str]]] = {
    # name: (category, rows)
    "block":    ("still_life", ["OO",
                                "OO"]),
    "beehive":  ("still_life", [".OO.",
                                "O..O",
                                ".OO."]),
    "loaf":     ("still_life", [".OO.",
                                "O..O",
                                ".O.O",
                                "..O."]),
    "blinker":  ("oscillator", ["OOO"]),
    "toad":     ("oscillator", [".OOO",
                                "OOO."]),
    "beacon":   ("oscillator", ["OO..",
                                "OO..",
                                "..OO",
                                "..OO"]),
    "pulsar":   ("oscillator", ["..OOO...OOO..",
                                ".............",
                                "O....O.O....O",
                                "O....O.O....O",
                                "O....O.O....O",
                                "..OOO...OOO..",
                                ".............",
                                "..OOO...OOO..",
                                "O....O.O....O",
                                "O....O.O....O",
                                "O....O.O....O",
                                ".............",
                                "..OOO...OOO.."]),
    # pentadecathlon: Conway's original seed is a row of 10 cells
    "pentadecathlon": ("oscillator", ["OOOOOOOOOO"]),
    "glider":   ("spaceship",  [".O.",
                                "..O",
                                "OOO"]),
    "lwss":     ("spaceship",  ["O..O.",
                                "....O",
                                "O...O",
                                ".OOOO"]),
}


def _bitmap(rows: list[str]) -> np.ndarray:
    return np.array([[1 if c == "O" else 0 for c in r] for r in rows], dtype=np.uint8)


def _place(bmp: np.ndarray, top: int, left: int) -> np.ndarray:
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    h, w = bmp.shape
    g[top:top + h, left:left + w] = bmp
    return g


# placement offsets per category (top-left corner)
_PLACEMENTS = {
    "still_life": [(18, 18), (5, 30)],
    "oscillator": [(15, 15), (2, 25)],
    "spaceship":  [(4, 4), (20, 8), (10, 28), (30, 20)],
}
_N_PHASES = {"still_life": 1, "oscillator": None, "spaceship": 4}   # None -> full period
_BURN_IN = 40   # steps to settle onto the limit cycle before sampling phases


def build_stimuli(seed: int = 0, n_random_per_density: int = 20,
                  densities=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    grids: list[np.ndarray] = []
    meta: list[dict] = []

    for name, (cat, rows) in _PATTERNS.items():
        bmp = _bitmap(rows)
        for pi, (top, left) in enumerate(_PLACEMENTS[cat]):
            base = _place(bmp, top, left)
            settled = base
            for _ in range(_BURN_IN):
                settled = gol_step(settled)
            period, _ = detect_period(settled, max_p=40)   # detect on the limit cycle
            if period is None:
                period = 1
            n_ph = _N_PHASES[cat] or period
            n_ph = min(n_ph, period)
            frame = settled
            for ph in range(n_ph):
                grids.append(frame.copy())
                meta.append(dict(category=cat, name=name, period=int(period),
                                 phase=ph, density=None, placement=pi))
                frame = gol_step(frame)

    rng = np.random.default_rng(seed)
    for d in densities:
        for _ in range(n_random_per_density):
            g = (rng.random((GRID, GRID)) < d).astype(np.uint8)
            grids.append(g)
            meta.append(dict(category="random", name=f"rand_d{d:.1f}", period=None,
                             phase=None, density=float(g.mean()), placement=None))

    return np.stack(grids), meta


def summary(meta: list[dict]) -> dict:
    out: dict[str, int] = {}
    for m in meta:
        out[m["category"]] = out.get(m["category"], 0) + 1
    return out


if __name__ == "__main__":
    g, m = build_stimuli()
    print(f"total stimuli: {len(g)}   grids shape: {g.shape}")
    print("by category:", summary(m))
    # per-pattern phase counts
    per = {}
    for r in m:
        if r["category"] != "random":
            per.setdefault(r["name"], 0)
            per[r["name"]] += 1
    print("structured per name:", per)
