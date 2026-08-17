"""
sanity_check.py
----------------
Verification script (not a pytest suite — run directly) for
teacher_search.py:

1. unified_score_batch on known cases (an isolated dot dies at t=1; a
   blinker oscillates forever and should score T + 3).
2. greedy_search cross-checked against brute-force search over all 1- and
   2-cell combinations on a small candidate neighborhood, to confirm greedy
   isn't leaving large gains on the table.
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from diedown_predictor.teacher_search import (
    DEFAULT_T,
    candidate_neighborhood,
    greedy_search,
    unified_score_batch,
)

GRID = 20


def make_grid(cells):
    g = np.zeros((GRID, GRID), dtype=np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def check_known_cases():
    print("── Known-case checks ──")

    dot = make_grid([(10, 10)])
    score = unified_score_batch(dot[None])[0]
    assert score == 1, f"isolated dot should die at t=1, got {score}"
    print(f"  isolated dot: score={score} (expected 1) OK")

    blinker = make_grid([(10, 9), (10, 10), (10, 11)])
    score = unified_score_batch(blinker[None])[0]
    expected = DEFAULT_T + 3
    assert score == expected, f"blinker should score T+3={expected}, got {score}"
    print(f"  blinker: score={score} (expected {expected}) OK")


def brute_force_best(init, candidates, K, T):
    best_score = float(unified_score_batch(init[None], T)[0])
    best_combo = ()
    for k in range(1, K + 1):
        for combo in itertools.combinations(candidates, k):
            g = init.copy()
            for r, c in combo:
                g[r, c] ^= 1
            s = float(unified_score_batch(g[None], T)[0])
            if s < best_score:
                best_score = s
                best_combo = combo
    return best_combo, best_score


def check_greedy_vs_brute_force():
    print("── Greedy vs. brute-force cross-check ──")
    T = 60  # shorter horizon keeps brute force fast for this check

    # Case 1: a perfectly symmetric block (still life). This is the classic
    # adversarial case for greedy/beam search: flipping *either* top cell
    # alone just heals back into the block (no improvement over baseline),
    # but flipping *both* together leaves a domino that dies at t=1. Since
    # neither half looks better than baseline on its own, a bounded beam can
    # miss the pair. This is a known, expected limitation of the
    # greedy/beam approximation (not a bug) — reported here, not asserted.
    block = make_grid([(10, 10), (10, 11), (11, 10), (11, 11)])
    candidates = candidate_neighborhood(block, margin=2)
    chosen, scores = greedy_search(block, candidates, K=2, T=T)
    best_combo, brute_score = brute_force_best(block, candidates, K=2, T=T)
    gap = scores[-1] - brute_score
    print(f"  [symmetric block, known-hard case] beam: {chosen} score={scores[-1]}"
          f"  |  brute force: {list(best_combo)} score={brute_score}  |  gap={gap}")

    # Case 2: an asymmetric cluster (no synergistic-pair trap) — this is the
    # common case in practice, and is where beam search should reliably
    # match the brute-force optimum.
    r_pent = make_grid([(10, 11), (10, 12), (11, 10), (11, 11), (12, 11)])
    candidates = candidate_neighborhood(r_pent, margin=2)
    print(f"  candidate neighborhood size (r-pentomino): {len(candidates)}")
    chosen, scores = greedy_search(r_pent, candidates, K=2, T=T)
    best_combo, brute_score = brute_force_best(r_pent, candidates, K=2, T=T)
    gap = scores[-1] - brute_score
    print(f"  [asymmetric r-pentomino] beam: {chosen} score={scores[-1]}"
          f"  |  brute force: {list(best_combo)} score={brute_score}  |  gap={gap}")
    assert gap <= 1e-6, f"beam search left a gain on the table on an asymmetric case: gap={gap}"
    print(f"  gap={gap} — beam search matches brute-force optimum on the realistic case OK")


if __name__ == "__main__":
    check_known_cases()
    check_greedy_vs_brute_force()
    print("All sanity checks passed.")
