"""
Data generation for all 6 GoL neural network tasks.

All GoL simulation uses periodic (toroidal) boundary conditions via np.roll.
Datasets are saved as .npz files in nn/data/.

Tasks
-----
1  next_state        (grid_t)            → (grid_t+1)
2  sensitivity       (grid_t0)           → (sensitivity_map)
3  chaos             (grid_t0, flip_mask)→ (cumulative_divergence)
4  neural_ca         same as task 1 — shared dataset
5  rollout           (grid_t0)           → (grid_t+k)   for k in {1..K}
6  attractor         (grid_t0)           → fate label {0,1,2,3}
"""

import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Core GoL engine ────────────────────────────────────────────────────────────

def _neighbors(cells: np.ndarray) -> np.ndarray:
    """Sum of 8 neighbours; cells shape (..., H, W)."""
    return sum(
        np.roll(np.roll(cells, i, axis=-2), j, axis=-1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )


def gol_step(cells: np.ndarray) -> np.ndarray:
    """One GoL step. Works on single (H,W) or batch (B,H,W)."""
    n = _neighbors(cells)
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


def run_trajectory(init: np.ndarray, steps: int) -> np.ndarray:
    """Return (steps+1, H, W) trajectory from a single (H,W) initial state."""
    traj = np.empty((steps + 1, *init.shape), dtype=np.uint8)
    traj[0] = init
    for t in range(1, steps + 1):
        traj[t] = gol_step(traj[t - 1])
    return traj


def compute_sensitivity_map(init: np.ndarray, steps: int) -> np.ndarray:
    """
    Vectorised perturbation sweep.
    Flips every cell once, runs `steps` steps, sums divergence from baseline.
    Returns (H, W) float32 cumulative-divergence map.
    """
    H, W  = init.shape
    n     = H * W

    # baseline trajectory
    baseline = run_trajectory(init, steps)   # (steps+1, H, W)

    # all perturbed initial states in one batch
    batch = np.tile(init, (n, 1, 1)).copy()  # (n, H, W)
    rows, cols = np.unravel_index(np.arange(n), (H, W))
    batch[np.arange(n), rows, cols] ^= 1

    cumulative = np.zeros(n, dtype=np.float32)
    for t in range(steps):
        batch = gol_step(batch)                                   # (n,H,W)
        diff  = (batch != baseline[t + 1]).sum(axis=(1, 2))      # (n,)
        cumulative += diff

    return cumulative.reshape(H, W)


def classify_fate(init: np.ndarray, steps: int = 200) -> int:
    """
    0 = dies (population → 0)
    1 = still life (period 1)
    2 = oscillator (period 2–15)
    3 = active / complex (still changing after `steps` steps)
    """
    traj = run_trajectory(init, steps)
    if traj[-1].sum() == 0:
        return 0
    for period in range(1, 16):
        if np.array_equal(traj[-1], traj[-1 - period]):
            return 1 if period == 1 else 2
    return 3


# ── Dataset generators ─────────────────────────────────────────────────────────

def _random_grids(n: int, grid_size: int, density: float, rng) -> np.ndarray:
    return (rng.random((n, grid_size, grid_size)) < density).astype(np.uint8)


def generate_trajectory_dataset(
    n_inits: int = 500,
    grid_size: int = 32,
    traj_steps: int = 100,
    density: float = 0.35,
    seed: int = 0,
    save: bool = True,
) -> dict:
    """
    Tasks 1 & 4 & 5.
    Returns:
      states  (N, H, W)  — all states from all trajectories
      nexts   (N, H, W)  — corresponding next states  (task 1 / 4)
      inits   (n_inits, H, W) — initial states
      trajs   (n_inits, traj_steps+1, H, W) — full trajectories (task 5)
    """
    print(f"[data] Generating trajectory dataset  "
          f"n={n_inits}  grid={grid_size}  steps={traj_steps} …")
    rng   = np.random.default_rng(seed)
    inits = _random_grids(n_inits, grid_size, density, rng)

    trajs  = np.empty((n_inits, traj_steps + 1, grid_size, grid_size), dtype=np.uint8)
    for i, init in enumerate(inits):
        trajs[i] = run_trajectory(init, traj_steps)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_inits}")

    # flatten to (state, next) pairs
    states = trajs[:, :-1].reshape(-1, grid_size, grid_size)
    nexts  = trajs[:, 1: ].reshape(-1, grid_size, grid_size)
    print(f"  → {len(states):,} (state, next) pairs")

    out = dict(states=states, nexts=nexts, inits=inits, trajs=trajs)
    if save:
        path = os.path.join(DATA_DIR, "trajectories.npz")
        np.savez_compressed(path, **out)
        print(f"  Saved → {path}")
    return out


def generate_sensitivity_dataset(
    n_samples: int = 1000,
    grid_size: int = 32,
    steps: int = 50,
    density: float = 0.35,
    seed: int = 1,
    save: bool = True,
) -> dict:
    """Task 2. Returns grids (N,H,W) and sensitivity maps (N,H,W)."""
    print(f"[data] Generating sensitivity dataset  "
          f"n={n_samples}  grid={grid_size}  steps={steps} …")
    rng    = np.random.default_rng(seed)
    grids  = _random_grids(n_samples, grid_size, density, rng)
    maps   = np.empty((n_samples, grid_size, grid_size), dtype=np.float32)

    for i, g in enumerate(grids):
        maps[i] = compute_sensitivity_map(g, steps)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_samples}  "
                  f"(max={maps[i].max():.0f} mean={maps[i].mean():.1f})")

    out = dict(grids=grids, maps=maps)
    if save:
        path = os.path.join(DATA_DIR, "sensitivity.npz")
        np.savez_compressed(path, **out)
        print(f"  Saved → {path}")
    return out


def generate_chaos_dataset(
    n_samples: int = 5000,
    grid_size: int = 32,
    steps: int = 50,
    density: float = 0.35,
    seed: int = 2,
    save: bool = True,
) -> dict:
    """
    Task 3.
    For each sample: random grid + random perturbed cell → scalar divergence.
    Returns grids (N,H,W), masks (N,H,W) one-hot flip location, scores (N,).
    """
    print(f"[data] Generating chaos dataset  "
          f"n={n_samples}  grid={grid_size}  steps={steps} …")
    rng    = np.random.default_rng(seed)
    grids  = _random_grids(n_samples, grid_size, density, rng)
    masks  = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8)
    scores = np.zeros(n_samples, dtype=np.float32)

    for i, g in enumerate(grids):
        # pick a random cell to perturb
        r = rng.integers(0, grid_size)
        c = rng.integers(0, grid_size)
        masks[i, r, c] = 1

        baseline_traj = run_trajectory(g, steps)
        p = g.copy(); p[r, c] ^= 1
        p_traj = run_trajectory(p, steps)
        scores[i] = (p_traj[1:] != baseline_traj[1:]).sum(axis=(1, 2)).sum()

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_samples}")

    out = dict(grids=grids, masks=masks, scores=scores)
    if save:
        path = os.path.join(DATA_DIR, "chaos.npz")
        np.savez_compressed(path, **out)
        print(f"  Saved → {path}")
    return out


def generate_attractor_dataset(
    n_samples: int = 2000,
    grid_size: int = 32,
    steps: int = 200,
    density: float = 0.35,
    seed: int = 3,
    save: bool = True,
) -> dict:
    """
    Task 6. Returns grids (N,H,W) and fate labels (N,) in {0,1,2,3}.
    Labels: 0=dies, 1=still life, 2=oscillator(p2-15), 3=active/complex.
    """
    print(f"[data] Generating attractor dataset  "
          f"n={n_samples}  grid={grid_size}  steps={steps} …")
    rng    = np.random.default_rng(seed)
    grids  = _random_grids(n_samples, grid_size, density, rng)
    labels = np.empty(n_samples, dtype=np.int64)

    for i, g in enumerate(grids):
        labels[i] = classify_fate(g, steps)
        if (i + 1) % 500 == 0:
            from collections import Counter
            print(f"  {i+1}/{n_samples}  dist={dict(Counter(labels[:i+1].tolist()))}")

    out = dict(grids=grids, labels=labels)
    if save:
        path = os.path.join(DATA_DIR, "attractor.npz")
        np.savez_compressed(path, **out)
        print(f"  Saved → {path}")
    return out


def load_dataset(name: str) -> dict:
    path = os.path.join(DATA_DIR, f"{name}.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    generate_trajectory_dataset()
    generate_sensitivity_dataset()
    generate_chaos_dataset()
    generate_attractor_dataset()
    print("\nAll datasets generated.")
