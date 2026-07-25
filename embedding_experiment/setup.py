"""
setup.py — Environment check and trajectory dataset generation.

Run this first to verify dependencies and generate the trajectory dataset
needed for training the trajectory-invariant encoder.

Usage:
    python setup.py [--n-trajectories 10000] [--traj-len 30] [--seed 0]
"""

import argparse
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

REQUIRED = ["torch", "numpy", "matplotlib", "sklearn", "PIL"]

GRID_SIZE = 64


# ── Environment check ─────────────────────────────────────────────────────────

def check_env():
    print("=== Environment check ===")
    ok = True
    for pkg in REQUIRED:
        try:
            mod = importlib.import_module(pkg if pkg != "PIL" else "PIL.Image")
            version = getattr(mod, "__version__", "?")
            print(f"  [ok] {pkg} {version}")
        except ImportError:
            print(f"  [MISSING] {pkg}  — install with: pip install {pkg}")
            ok = False

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [ok] torch device: {device}")
    if device == "cpu":
        print("       (GPU not found — training will be slow)")

    print()
    return ok, device


# ── GoL engine ────────────────────────────────────────────────────────────────

def gol_step(cells: np.ndarray) -> np.ndarray:
    n = sum(
        np.roll(np.roll(cells, i, axis=-2), j, axis=-1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


def classify_fate(trajectory: list[np.ndarray]) -> str:
    """Coarse fate classification based on the last few frames."""
    last = trajectory[-1]
    prev = trajectory[-2]
    if last.sum() == 0:
        return "extinct"
    if np.array_equal(last, prev):
        return "still_life"
    if np.array_equal(last, trajectory[-3]):
        return "oscillator_p2"
    if len(trajectory) >= 4 and np.array_equal(last, trajectory[-4]):
        return "oscillator_p3"
    return "other"


# ── Trajectory dataset generation ─────────────────────────────────────────────

def generate_trajectories(n: int, traj_len: int, seed: int, out_path: str):
    """
    Generate N GoL trajectories, each of length traj_len.

    Each trajectory starts from a random soup initial condition (density 0.2–0.5)
    evolved a few warm-up steps so states are non-trivial.

    Output npz:
        frames       (N, T, 64, 64) uint8  — T = traj_len frames per trajectory
        traj_id      (N, T)         int64  — trajectory index repeated T times
        fate         (N,)           str    — coarse fate label
        init_density (N,)           float32
    """
    rng = np.random.default_rng(seed)
    FATES = ["extinct", "still_life", "oscillator_p2", "oscillator_p3", "other"]
    fate_map = {f: i for i, f in enumerate(FATES)}

    frames = np.zeros((n, traj_len, GRID_SIZE, GRID_SIZE), np.uint8)
    fate_ids = np.zeros(n, np.int64)
    init_densities = np.zeros(n, np.float32)

    print(f"Generating {n:,} trajectories × {traj_len} steps …")
    for i in range(n):
        density = rng.uniform(0.2, 0.5)
        grid = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        # warm-up: evolve a few steps so the initial frame isn't pure noise
        warmup = int(rng.integers(3, 10))
        for _ in range(warmup):
            grid = gol_step(grid)

        traj = [grid.copy()]
        for _ in range(traj_len - 1):
            grid = gol_step(grid)
            traj.append(grid.copy())

        frames[i] = np.stack(traj)
        fate_ids[i] = fate_map[classify_fate(traj)]
        init_densities[i] = traj[0].mean()

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{n:,}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        frames=frames,
        fate=fate_ids,
        fate_names=np.array(FATES),
        init_density=init_densities,
        traj_len=np.int64(traj_len),
        grid_size=np.int64(GRID_SIZE),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved {n:,} trajectories → {out_path} ({size_mb:.1f} MB)")

    # fate distribution
    from collections import Counter
    dist = Counter(FATES[f] for f in fate_ids)
    print("Fate distribution:", dict(dist))


def save_preview(npz_path: str, out_png: str, n_traj: int = 6, steps_shown: int = 8):
    """Save a grid showing the first few frames of several trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    frames = d["frames"]            # (N, T, H, W)
    fates = d["fate"]
    fate_names = d["fate_names"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(frames), n_traj, replace=False)
    steps = min(steps_shown, frames.shape[1])

    fig, axes = plt.subplots(n_traj, steps, figsize=(steps * 1.4, n_traj * 1.4))
    for row, i in enumerate(idx):
        for col in range(steps):
            ax = axes[row, col]
            ax.imshow(frames[i, col], cmap="binary", interpolation="nearest")
            ax.axis("off")
            if col == 0:
                ax.set_title(f"traj {i}\n{fate_names[fates[i]]}", fontsize=6)
            elif col == 1:
                ax.set_title(f"t={col}", fontsize=6)
            else:
                ax.set_title(f"t={col}", fontsize=6)
    fig.suptitle("Sample trajectories (each row = one initial condition)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Preview → {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trajectories", type=int, default=10_000,
                    help="number of trajectories to generate")
    ap.add_argument("--traj-len", type=int, default=30,
                    help="number of GoL steps per trajectory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-data", action="store_true",
                    help="skip data generation (only check environment)")
    args = ap.parse_args()

    ok, device = check_env()
    if not ok:
        print("Fix missing dependencies before continuing.")
        sys.exit(1)

    if args.skip_data:
        print("Skipping data generation (--skip-data).")
        return

    out_path = os.path.join(DATA_DIR, "trajectories.npz")
    if os.path.exists(out_path):
        print(f"Dataset already exists at {out_path}")
        ans = input("Regenerate? [y/N] ").strip().lower()
        if ans != "y":
            print("Skipping generation.")
            return

    print(f"\n=== Generating trajectory dataset ===")
    generate_trajectories(args.n_trajectories, args.traj_len, args.seed, out_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_preview(out_path, os.path.join(RESULTS_DIR, "trajectory_preview.png"))

    print("\n=== Setup complete ===")
    print("Next steps:")
    print("  python train.py       # train the trajectory-invariant encoder")
    print("  python evaluate.py    # evaluate embedding quality")


if __name__ == "__main__":
    main()
