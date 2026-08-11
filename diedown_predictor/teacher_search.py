"""
teacher_search.py
------------------
Greedy local search that generates imitation-learning labels for the
die-down predictor: given an initial GoL configuration, greedily choose up
to K cell flips (from a local candidate neighborhood) that drive the
trajectory to extinction as fast as possible, or leave the fewest
survivors after a long horizon if extinction isn't reachable.

Unified score (lower is better):
    score = t_die               if population reaches 0 at step t_die <= T
    score = T + survivors_at_T  otherwise

Any die-down beats any non-die-down outcome; ties are broken by speed
(die-down case) or survivor count (non-die-down case).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nn.data_gen import gol_step

DEFAULT_T = 200


def unified_score_batch(batch: np.ndarray, T: int = DEFAULT_T) -> np.ndarray:
    """
    batch: (B, H, W) initial states.
    Returns (B,) float32 unified die-down score for each.
    """
    cur = batch.astype(np.uint8)
    B = cur.shape[0]
    died_at = np.full(B, -1, dtype=np.int64)
    pop = cur.reshape(B, -1).sum(axis=1)
    died_at[pop == 0] = 0

    for t in range(1, T + 1):
        cur = gol_step(cur)
        pop = cur.reshape(B, -1).sum(axis=1)
        died_at[(pop == 0) & (died_at < 0)] = t
        if np.all(died_at >= 0):
            break

    return np.where(died_at >= 0, died_at, T + pop).astype(np.float32)


def candidate_neighborhood(init: np.ndarray, margin: int = 3) -> list[tuple[int, int]]:
    """
    Bounding box of live cells in `init`, padded by `margin` and clipped to
    grid bounds. Returns the list of (row, col) eligible flip locations.
    """
    H, W = init.shape
    rows, cols = np.nonzero(init)
    if len(rows) == 0:
        return []
    r0, r1 = max(0, int(rows.min()) - margin), min(H - 1, int(rows.max()) + margin)
    c0, c1 = max(0, int(cols.min()) - margin), min(W - 1, int(cols.max()) + margin)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def greedy_search(
    init: np.ndarray,
    candidates: list[tuple[int, int]],
    K: int = 3,
    T: int = DEFAULT_T,
    beam_width: int = 4,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Beam search (width `beam_width`) over up to K cell flips from
    `candidates`, minimizing the unified die-down score. Plain greedy
    (beam_width=1) gets trapped by symmetric configurations where no single
    flip helps individually but a *pair* of flips is synergistic (e.g. a 2x2
    block: flipping either of its top cells alone does nothing, but flipping
    both leaves a domino that dies immediately) — a small beam catches these
    without the combinatorial cost of full brute force. The best score seen
    across *all* explored depths (0..K) is kept, so the search naturally
    "stops early" whenever going deeper doesn't help.

    Returns:
        chosen: list of (row, col), length 0..K
        scores: list of float, length len(chosen)+1 — scores[0] is the
                baseline (no-flip) score, scores[i] is the score after the
                first i flips in `chosen`.
    """
    init = init.astype(np.uint8)
    baseline_score = float(unified_score_batch(init[None], T)[0])
    beam = [(baseline_score, (), init)]
    best_score, best_chosen = baseline_score, ()

    for _ in range(K):
        expanded = []
        for score, chosen, grid in beam:
            remaining = [cell for cell in candidates if cell not in chosen]
            if not remaining:
                continue
            batch = np.tile(grid, (len(remaining), 1, 1))
            rs = np.array([r for r, _ in remaining])
            cs = np.array([c for _, c in remaining])
            batch[np.arange(len(remaining)), rs, cs] ^= 1
            cand_scores = unified_score_batch(batch, T)
            for i, cell in enumerate(remaining):
                expanded.append((float(cand_scores[i]), chosen + (cell,), batch[i]))

        if not expanded:
            break
        expanded.sort(key=lambda e: e[0])
        beam = expanded[:beam_width]
        if beam[0][0] < best_score:
            best_score, best_chosen = beam[0][0], beam[0][1]

    # Recompute the score at each prefix length of the winning sequence, for
    # a clean imitation-learning trace.
    chosen = list(best_chosen)
    scores = [baseline_score]
    grid = init.copy()
    for r, c in chosen:
        grid = grid.copy()
        grid[r, c] ^= 1
        scores.append(float(unified_score_batch(grid[None], T)[0]))

    return chosen, scores
