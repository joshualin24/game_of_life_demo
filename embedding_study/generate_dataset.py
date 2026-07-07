"""
Generate a Game of Life image dataset emphasizing periodic behavior,
inspired by "Conway's Game of Life is Omniperiodic" (arXiv:2312.02799).

Each image is a 64x64 binary grid containing one of:
  - still lifes         (period 1)
  - oscillators         (periods 2, 3, 15) placed at random phases
  - spaceships          (period 4 up to translation)
  - mixed scenes        (any of the above)
  - random soup         (random init evolved a random number of steps)

Patterns are placed so their full periodic envelopes never interact, so a
composed scene is itself periodic (period = lcm of its parts).

Every pattern's claimed period is verified programmatically at startup.

Output: embedding_study/data/gol_images.npz with
  images          (N, 64, 64) uint8
  category        (N,)        int64   index into CATEGORIES
  pattern_counts  (N, P)      uint8   per-image count of each pattern
  pattern_names   (P,)        str
  pattern_periods (P,)        int64
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

GRID_SIZE = 64
CATEGORIES = ["still_life", "oscillator", "spaceship", "mixed", "soup"]
CATEGORY_PROBS = [0.15, 0.40, 0.10, 0.15, 0.20]

# ── GoL engine (toroidal, same convention as nn/data_gen.py) ─────────────────

def gol_step(cells: np.ndarray) -> np.ndarray:
    n = sum(
        np.roll(np.roll(cells, i, axis=-2), j, axis=-1)
        for i in (-1, 0, 1) for j in (-1, 0, 1)
        if (i, j) != (0, 0)
    )
    return ((n == 3) | (cells & (n == 2))).astype(np.uint8)


# ── Pattern library ───────────────────────────────────────────────────────────

def _parse(ascii_rows):
    return np.array([[1 if ch == "X" else 0 for ch in row] for row in ascii_rows],
                    dtype=np.uint8)


RAW_PATTERNS = {
    # name: (kind, period, ascii art)
    "block":   ("still_life", 1, ["XX",
                                  "XX"]),
    "beehive": ("still_life", 1, [".XX.",
                                  "X..X",
                                  ".XX."]),
    "loaf":    ("still_life", 1, [".XX.",
                                  "X..X",
                                  ".X.X",
                                  "..X."]),
    "boat":    ("still_life", 1, ["XX.",
                                  "X.X",
                                  ".X."]),
    "tub":     ("still_life", 1, [".X.",
                                  "X.X",
                                  ".X."]),
    "blinker": ("oscillator", 2, ["XXX"]),
    "toad":    ("oscillator", 2, [".XXX",
                                  "XXX."]),
    "beacon":  ("oscillator", 2, ["XX..",
                                  "XX..",
                                  "..XX",
                                  "..XX"]),
    "pulsar":  ("oscillator", 3, ["..XXX...XXX..",
                                  ".............",
                                  "X....X.X....X",
                                  "X....X.X....X",
                                  "X....X.X....X",
                                  "..XXX...XXX..",
                                  ".............",
                                  "..XXX...XXX..",
                                  "X....X.X....X",
                                  "X....X.X....X",
                                  "X....X.X....X",
                                  ".............",
                                  "..XXX...XXX.."]),
    "pentadecathlon": ("oscillator", 15, ["..X....X..",
                                          "XX.XXXX.XX",
                                          "..X....X.."]),
    "glider":  ("spaceship", 4, [".X.",
                                 "..X",
                                 "XXX"]),
    "lwss":    ("spaceship", 4, [".X..X",
                                 "X....",
                                 "X...X",
                                 "XXXX."]),
}


@dataclass
class Pattern:
    name: str
    kind: str
    period: int
    phases: list          # list of (h, w) uint8 arrays, all same shape
    envelope: np.ndarray  # (h, w) bool — union of live cells over one period


def build_pattern(name, kind, period, ascii_rows) -> Pattern:
    """Simulate the pattern on a large empty canvas, verify its period, and
    return all phases cropped to the common envelope bounding box."""
    cells = _parse(ascii_rows)
    pad = 4 * period + 4
    canvas = np.zeros((cells.shape[0] + 2 * pad, cells.shape[1] + 2 * pad), np.uint8)
    canvas[pad:pad + cells.shape[0], pad:pad + cells.shape[1]] = cells

    states = [canvas]
    for _ in range(period):
        states.append(gol_step(states[-1]))

    first, last = states[0], states[-1]
    if kind == "spaceship":
        # periodic up to translation
        dy, dx = np.argwhere(last).min(0) - np.argwhere(first).min(0)
        assert np.array_equal(np.roll(last, (-dy, -dx), (0, 1)), first), \
            f"{name}: not periodic (up to translation) with period {period}"
    else:
        assert np.array_equal(last, first), \
            f"{name}: not periodic with period {period}"
        # verify no smaller period
        for p in range(1, period):
            assert not np.array_equal(states[p], first), \
                f"{name}: actual period is {p}, not {period}"

    envelope = np.zeros_like(canvas, dtype=bool)
    for s in states[:period]:
        envelope |= s.astype(bool)
    rows = np.any(envelope, axis=1)
    cols = np.any(envelope, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    phases = [s[r0:r1 + 1, c0:c1 + 1].copy() for s in states[:period]]
    return Pattern(name, kind, period, phases, envelope[r0:r1 + 1, c0:c1 + 1])


def build_library():
    lib = {}
    print("Verifying pattern periods:")
    for name, (kind, period, art) in RAW_PATTERNS.items():
        pat = build_pattern(name, kind, period, art)
        lib[name] = pat
        h, w = pat.envelope.shape
        print(f"  {name:16s} kind={kind:10s} period={period:3d} envelope={h}x{w}  OK")
    return lib


# ── Scene composition ─────────────────────────────────────────────────────────

def compose_scene(rng, library, pool_names, n_patterns, size=GRID_SIZE):
    """Place up to n_patterns non-interacting patterns at random phases,
    positions, rotations and reflections. Returns (grid, placed_names)."""
    grid = np.zeros((size, size), np.uint8)
    occupied = np.zeros((size, size), bool)
    placed = []

    for _ in range(n_patterns):
        pat = library[pool_names[rng.integers(len(pool_names))]]
        phase = pat.phases[rng.integers(pat.period)]
        env = pat.envelope
        k = rng.integers(4)
        phase, env = np.rot90(phase, k), np.rot90(env, k)
        if rng.random() < 0.5:
            phase, env = np.fliplr(phase), np.fliplr(env)

        h, w = phase.shape
        if h > size - 2 or w > size - 2:
            continue
        for _attempt in range(30):
            r = rng.integers(1, size - h)
            c = rng.integers(1, size - w)
            # require a 1-cell halo around the envelope to be free
            if not occupied[r - 1:r + h + 1, c - 1:c + w + 1].any():
                grid[r:r + h, c:c + w] |= phase
                occupied[r:r + h, c:c + w] |= env
                placed.append(pat.name)
                break
    return grid, placed


def make_soup(rng, size=GRID_SIZE):
    density = rng.uniform(0.15, 0.5)
    grid = (rng.random((size, size)) < density).astype(np.uint8)
    for _ in range(rng.integers(5, 61)):
        grid = gol_step(grid)
    return grid


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(n_images: int, seed: int, out_path: str):
    rng = np.random.default_rng(seed)
    library = build_library()
    names = list(RAW_PATTERNS.keys())
    name_idx = {n: i for i, n in enumerate(names)}
    pools = {
        "still_life": [n for n, p in library.items() if p.kind == "still_life"],
        "oscillator": [n for n, p in library.items() if p.kind == "oscillator"],
        "spaceship":  [n for n, p in library.items() if p.kind == "spaceship"],
        "mixed":      names,
    }

    images = np.zeros((n_images, GRID_SIZE, GRID_SIZE), np.uint8)
    category = np.zeros(n_images, np.int64)
    pattern_counts = np.zeros((n_images, len(names)), np.uint8)

    cats = rng.choice(len(CATEGORIES), size=n_images, p=CATEGORY_PROBS)
    print(f"\nGenerating {n_images:,} images ({GRID_SIZE}x{GRID_SIZE}) …")
    for i in range(n_images):
        cat = CATEGORIES[cats[i]]
        category[i] = cats[i]
        if cat == "soup":
            images[i] = make_soup(rng)
        else:
            n_pat = rng.integers(1, 6)
            grid, placed = compose_scene(rng, library, pools[cat], n_pat)
            images[i] = grid
            for pname in placed:
                pattern_counts[i, name_idx[pname]] += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1:,}/{n_images:,}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        category=category,
        pattern_counts=pattern_counts,
        pattern_names=np.array(names),
        pattern_periods=np.array([RAW_PATTERNS[n][1] for n in names], np.int64),
        categories=np.array(CATEGORIES),
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved {n_images:,} images → {out_path} ({size_mb:.1f} MB)")

    from collections import Counter
    dist = Counter(CATEGORIES[c] for c in category)
    print("Category distribution:", dict(dist))
    return out_path


def save_preview(npz_path, out_png, n=64, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path, allow_pickle=False)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(d["images"]), n, replace=False)
    side = int(np.sqrt(n))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.4, side * 1.4))
    for ax, j in zip(axes.flat, idx):
        ax.imshow(d["images"][j], cmap="binary", interpolation="nearest")
        ax.set_title(str(d["categories"][d["category"][j]]), fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Preview grid → {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "gol_images.npz"))
    args = ap.parse_args()

    path = generate(args.n, args.seed, args.out)
    save_preview(path, os.path.join(os.path.dirname(path), "preview.png"))
