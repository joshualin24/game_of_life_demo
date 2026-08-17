"""
generate_dataset.py
--------------------
Builds the training set for the die-down predictor: samples base GoL
configurations (named patterns, multi-pattern combos, random-density grids —
mirroring nn/train_cnn_transformer_v10.py's data mix), stratified by
baseline fate (nn.data_gen.classify_fate) so most examples are configs a
perturbation could plausibly help (still life / oscillator / active) with a
smaller share of already-dying configs (so the model also learns to emit
STOP immediately when no help is needed). Runs the beam-search teacher
(teacher_search.greedy_search) on each to get an up-to-K flip
imitation-learning label, and caches everything to diedown_predictor/data/.

build_step_tensors() expands the cached per-grid sequences into
per-decode-step (input, valid-class mask, target) training tensors consumed
by train.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from nn.data_gen import classify_fate, gol_step
from diedown_predictor.teacher_search import DEFAULT_T, candidate_neighborhood, greedy_search

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

GRID_SIZE = 40
MARGIN = 3
K = 3
T = DEFAULT_T
BEAM_WIDTH = 4
# Dense/random-density base grids can have a candidate bounding box close to
# the full board (~1600 cells); beam search there would dominate runtime.
# Cap and uniformly subsample the candidate set so per-grid search cost stays
# bounded regardless of pattern size/density.
MAX_CANDIDATES = 150


# ── Named patterns (same library as nn/train_cnn_transformer_v10.py) ──────────

def _p(*rows):
    return np.array(rows, dtype=np.uint8)


NAMED_PATTERNS = {
    "block":   _p([1, 1], [1, 1]),
    "beehive": _p([0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]),
    "loaf":    _p([0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]),
    "boat":    _p([1, 1, 0], [1, 0, 1], [0, 1, 0]),
    "blinker": _p([1, 1, 1]),
    "toad":    _p([0, 1, 1, 1], [1, 1, 1, 0]),
    "beacon":  _p([1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]),
    "glider":  _p([0, 1, 0], [0, 0, 1], [1, 1, 1]),
    "lwss":    _p([0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0]),
    "r_pent":  _p([0, 1, 1], [1, 1, 0], [0, 1, 0]),
    "acorn":   _p([0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 1, 1]),
}
PATTERN_LIST = list(NAMED_PATTERNS.values())


def rand_orient(rng, pat):
    k = int(rng.integers(0, 4))
    if k:
        pat = np.rot90(pat, k)
    if rng.random() > 0.5:
        pat = np.fliplr(pat)
    return np.ascontiguousarray(pat)


def place_pattern(grid, pat, r0, c0):
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat


def gen_combo_init(rng, grid_size, n_patterns, bg_density=0.0):
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if bg_density > 0:
        grid = (rng.random((grid_size, grid_size)) < bg_density).astype(np.uint8)
    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    for _ in range(n_patterns):
        pat = rand_orient(rng, PATTERN_LIST[int(rng.integers(0, len(PATTERN_LIST)))])
        ph, pw = pat.shape
        placed = False
        for _ in range(30):
            r0 = int(rng.integers(0, grid_size))
            c0 = int(rng.integers(0, grid_size))
            rows = (r0 + np.arange(ph)) % grid_size
            cols = (c0 + np.arange(pw)) % grid_size
            if not np.any(occupied[np.ix_(rows, cols)][pat.astype(bool)]):
                place_pattern(grid, pat, r0, c0)
                buf = 3
                br = np.arange(r0 - buf, r0 + ph + buf) % grid_size
                bc = np.arange(c0 - buf, c0 + pw + buf) % grid_size
                occupied[np.ix_(br, bc)] = True
                placed = True
                break
        if not placed:
            place_pattern(grid, pat, int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
    return grid


def sample_base_grid(rng, grid_size=GRID_SIZE):
    """One base config: named pattern (40%), 2-3 pattern combo (30%), or a
    random-density grid evolved a few steps to look like a mid-trajectory
    state (30%)."""
    choice = rng.random()
    if choice < 0.4:
        pat = rand_orient(rng, PATTERN_LIST[int(rng.integers(0, len(PATTERN_LIST)))])
        g = np.zeros((grid_size, grid_size), dtype=np.uint8)
        place_pattern(g, pat, int(rng.integers(0, grid_size)), int(rng.integers(0, grid_size)))
    elif choice < 0.7:
        g = gen_combo_init(rng, grid_size, n_patterns=int(rng.integers(2, 4)))
    else:
        density = rng.choice([0.10, 0.20, 0.35])
        g = (rng.random((grid_size, grid_size)) < density).astype(np.uint8)
        for _ in range(int(rng.integers(0, 5))):
            g = gol_step(g)
    return g


def stratified_sample(n_target, rng, grid_size=GRID_SIZE, dies_frac=0.15, fate_steps=T):
    """Oversample non-trivially-dying configs (where a perturbation can
    plausibly help); keep a smaller share of already-dying configs so the
    model also learns to emit STOP."""
    n_dies_target = int(n_target * dies_frac)
    n_other_target = n_target - n_dies_target
    dies_grids, other_grids = [], []
    attempts = 0
    max_attempts = n_target * 30
    while (len(dies_grids) < n_dies_target or len(other_grids) < n_other_target) and attempts < max_attempts:
        attempts += 1
        g = sample_base_grid(rng, grid_size)
        if g.sum() == 0:
            continue
        fate = classify_fate(g, steps=fate_steps)
        if fate == 0 and len(dies_grids) < n_dies_target:
            dies_grids.append(g)
        elif fate != 0 and len(other_grids) < n_other_target:
            other_grids.append(g)
    return dies_grids + other_grids


def build_dataset(n_grids=400, seed=42, out_name="pilot", print_every=20, beam_width=BEAM_WIDTH):
    rng = np.random.default_rng(seed)
    grids = stratified_sample(n_grids, rng)
    N = len(grids)

    grids_arr = np.zeros((N, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    chosen_arr = np.full((N, K, 2), -1, dtype=np.int16)
    n_chosen_arr = np.zeros(N, dtype=np.int8)
    scores_arr = np.full((N, K + 1), np.nan, dtype=np.float32)
    candidates_arr = np.full((N, MAX_CANDIDATES, 2), -1, dtype=np.int16)
    n_candidates_arr = np.zeros(N, dtype=np.int16)

    t0 = time.time()
    for i, g in enumerate(grids):
        candidates = candidate_neighborhood(g, margin=MARGIN)
        if len(candidates) > MAX_CANDIDATES:
            idx = rng.choice(len(candidates), MAX_CANDIDATES, replace=False)
            candidates = [candidates[j] for j in idx]
        chosen, scores = greedy_search(g, candidates, K=K, T=T, beam_width=beam_width)

        grids_arr[i] = g
        n_chosen_arr[i] = len(chosen)
        for j, (r, c) in enumerate(chosen):
            chosen_arr[i, j] = (r, c)
        scores_arr[i, :len(scores)] = scores
        n_candidates_arr[i] = len(candidates)
        for j, (r, c) in enumerate(candidates):
            candidates_arr[i, j] = (r, c)

        if (i + 1) % print_every == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i + 1}/{N}] elapsed={elapsed:.0f}s  ({rate:.2f} grids/s)")

    path = os.path.join(DATA_DIR, f"{out_name}.npz")
    np.savez_compressed(
        path,
        grids=grids_arr, chosen=chosen_arr, n_chosen=n_chosen_arr, scores=scores_arr,
        candidates=candidates_arr, n_candidates=n_candidates_arr,
        grid_size=GRID_SIZE, margin=MARGIN, K=K, T=T, beam_width=beam_width,
    )
    print(f"Saved {N} base grids -> {path}  (beam_width={beam_width})")
    return path


def merge_datasets(paths: list[str], out_name: str) -> str:
    """Concatenate multiple build_dataset() outputs (e.g. generated with
    different seeds) into one .npz, reusing already-computed teacher-search
    results instead of regenerating everything from scratch."""
    loaded = [np.load(p) for p in paths]
    for key in ("grid_size", "margin", "K", "T"):
        vals = {int(d[key]) for d in loaded}
        assert len(vals) == 1, f"mismatched {key} across datasets: {vals}"

    merged = {
        key: np.concatenate([d[key] for d in loaded], axis=0)
        for key in ("grids", "chosen", "n_chosen", "scores", "candidates", "n_candidates")
    }
    for key in ("grid_size", "margin", "K", "T"):
        merged[key] = loaded[0][key]

    path = os.path.join(DATA_DIR, f"{out_name}.npz")
    np.savez_compressed(path, **merged)
    n = merged["grids"].shape[0]
    print(f"Merged {len(paths)} datasets -> {n} base grids -> {path}")
    return path


# ── Expand cached sequences into per-decode-step training tensors ────────────

def build_step_tensors(npz_path):
    """
    Returns:
      X:        (M, 3, H, W) float32 — [grid_with_prior_flips, chosen_mask, k_remaining]
      mask:     (M, H*W + 1) bool   — valid-class mask (neighborhood minus already-chosen,
                                       STOP always valid)
      y:        (M,) int64          — target class index in [0, H*W] (H*W == STOP class)
      grid_idx: (M,) int64          — which base grid (row in the .npz) each example came
                                       from. Steps from the same base grid are highly
                                       correlated (they share most of the board) and must
                                       stay on the same side of any train/val split.
    """
    data = np.load(npz_path)
    grids, chosen, n_chosen = data["grids"], data["chosen"], data["n_chosen"]
    candidates, n_candidates = data["candidates"], data["n_candidates"]
    grid_size, K = int(data["grid_size"]), int(data["K"])
    H = W = grid_size
    STOP = H * W

    X_list, mask_list, y_list, grid_idx_list = [], [], [], []
    for i in range(len(grids)):
        init = grids[i]
        L = int(n_chosen[i])
        n_cand = int(n_candidates[i])
        cand_flat = {int(r) * W + int(c) for r, c in candidates[i, :n_cand]}

        state = init.astype(np.float32).copy()
        chosen_mask = np.zeros((H, W), dtype=np.float32)
        used: set[int] = set()

        n_steps = L + (1 if L < K else 0)
        for s in range(n_steps):
            k_remaining = (K - s) / K
            x = np.stack([state, chosen_mask, np.full((H, W), k_remaining, dtype=np.float32)])
            valid = np.zeros(H * W + 1, dtype=bool)
            valid[STOP] = True
            for idx in cand_flat - used:
                valid[idx] = True

            if s < L:
                r, c = (int(v) for v in chosen[i, s])
                target = r * W + c
            else:
                target = STOP

            X_list.append(x)
            mask_list.append(valid)
            y_list.append(target)
            grid_idx_list.append(i)

            if s < L:
                state = state.copy()
                state[r, c] = 1.0 - state[r, c]
                chosen_mask = chosen_mask.copy()
                chosen_mask[r, c] = 1.0
                used.add(r * W + c)

    X = np.asarray(X_list, dtype=np.float32)
    mask = np.asarray(mask_list, dtype=bool)
    y = np.asarray(y_list, dtype=np.int64)
    grid_idx = np.asarray(grid_idx_list, dtype=np.int64)
    return X, mask, y, grid_idx


# ── D4 (rotation/reflection) augmentation ─────────────────────────────────────
#
# GoL's toroidal dynamics are exactly equivariant under grid rotation and
# reflection, so rotating/reflecting a (grid, chosen-flips) example and its
# target cell in lockstep yields another perfectly valid training example —
# free 8x supervision with no extra teacher-search compute. Apply this only
# *within* a train or val split (never across), since a rotated copy of a
# training example is still the same underlying configuration.

def _d4_funcs():
    """8 functions, each transforming a (..., H, W) array over its last two axes."""
    return [
        lambda a: a,
        lambda a: np.rot90(a, 1, axes=(-2, -1)),
        lambda a: np.rot90(a, 2, axes=(-2, -1)),
        lambda a: np.rot90(a, 3, axes=(-2, -1)),
        lambda a: np.flip(a, axis=-1),
        lambda a: np.flip(np.rot90(a, 1, axes=(-2, -1)), axis=-1),
        lambda a: np.flip(np.rot90(a, 2, axes=(-2, -1)), axis=-1),
        lambda a: np.flip(np.rot90(a, 3, axes=(-2, -1)), axis=-1),
    ]


def _d4_forward_index_maps(H, W):
    """
    For each of the 8 D4 transforms, a flat-index map `fwd` such that a cell
    at old flat index `f` ends up at flat index `fwd[f]` after the transform.
    """
    idx = np.arange(H * W).reshape(H, W)
    maps = []
    for f in _d4_funcs():
        pos_map = f(idx)  # pos_map[r2,c2] = old flat index now sitting at (r2,c2)
        forward = np.empty(H * W, dtype=np.int64)
        forward[pos_map.reshape(-1)] = np.arange(H * W)
        maps.append(forward)
    return maps


def augment_d4(X, mask, y, grid_size):
    """Returns (X, mask, y) each 8x longer, one copy per D4 symmetry."""
    H = W = grid_size
    STOP = H * W
    funcs = _d4_funcs()
    fwd_maps = _d4_forward_index_maps(H, W)

    X_out, mask_out, y_out = [], [], []
    for f, fwd in zip(funcs, fwd_maps):
        Xa = X.copy()
        Xa[:, 0] = f(X[:, 0])   # grid channel (spatial)
        Xa[:, 1] = f(X[:, 1])   # chosen-mask channel (spatial)
        # channel 2 (k_remaining) is a constant broadcast — invariant.
        X_out.append(Xa)

        spatial_mask = mask[:, :H * W].reshape(-1, H, W)
        spatial_mask_t = f(spatial_mask).reshape(-1, H * W)
        mask_out.append(np.concatenate([spatial_mask_t, mask[:, H * W:]], axis=1))

        y_safe = np.where(y == STOP, 0, y)
        y_out.append(np.where(y == STOP, STOP, fwd[y_safe]))

    return (np.concatenate(X_out, axis=0),
            np.concatenate(mask_out, axis=0),
            np.concatenate(y_out, axis=0))


if __name__ == "__main__":
    build_dataset(n_grids=400, seed=42, out_name="pilot")
