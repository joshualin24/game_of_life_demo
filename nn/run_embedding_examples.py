"""
Embedding comparison examples for V8 (and optionally V10).

For each of four transformation families — rotation, translation, flip, perturbation —
produces one figure per example configuration showing:
  - original grid
  - transformed grid
  - grid diff
  - pre-transformer cosine-similarity heatmap (10×10 patches)
  - post-transformer cosine-similarity heatmap (10×10 patches)

Usage:
    python nn/run_embedding_examples.py           # V8 only
    python nn/run_embedding_examples.py --v10     # also run V10
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from nn.embedding_analysis import (
    EmbeddingExtractor, Transforms, compare,
    plot_comparison, plot_perturbation_spread,
    load_v8, load_v10, GRID_SIZE, PATCH_SIZE,
)
from nn.utils import RESULTS_DIR

# ── Configurations ─────────────────────────────────────────────────────────────

def _p(*rows): return np.array(rows, dtype=np.uint8)

PATTERNS = {
    "glider": _p([0,1,0],[0,0,1],[1,1,1]),
    "blinker": _p([1,1,1]),
    "pulsar_seed": _p([0,0,1,1,1,0,0],[1,0,0,0,0,0,1]),
}

def make_glider(r0=5, c0=5):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    pat = PATTERNS["glider"]
    g[r0:r0+pat.shape[0], c0:c0+pat.shape[1]] = pat
    return g

def make_blinker(r0=10, c0=10):
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    pat = PATTERNS["blinker"]
    g[r0:r0+1, c0:c0+pat.shape[1]] = pat
    return g

def make_random(density=0.35, seed=42):
    rng = np.random.default_rng(seed)
    return (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)


# ── Analysis runner ────────────────────────────────────────────────────────────

def run_all(tag: str, extractor: EmbeddingExtractor):
    """Run all example comparisons and save figures to RESULTS_DIR."""
    prefix = os.path.join(RESULTS_DIR, f"embed_{tag}")

    grids = {
        "glider":      make_glider(),
        "blinker":     make_blinker(),
        "random_d35":  make_random(density=0.35),
        "random_d65":  make_random(density=0.65),
    }

    # ── 1. ROTATION ────────────────────────────────────────────────────────────
    print("  [rotation]")
    for k, deg in [(1, 90), (2, 180), (3, 270)]:
        for name, grid in grids.items():
            grid_tf = Transforms.rotate(grid, k=k)
            pre1,  post1  = extractor(grid)
            pre2,  post2  = extractor(grid_tf)
            pre_s  = compare(pre1, pre2)
            post_s = compare(post1, post2)
            title = (f"{tag.upper()} — rotation {deg}° | {name}\n"
                     f"pre cos={pre_s['mean_cos']:.3f}   post cos={post_s['mean_cos']:.3f}")
            out = f"{prefix}_rot{deg}_{name}.png"
            plot_comparison(grid, grid_tf, pre_s, post_s, title,
                            out_path=out,
                            label_tf=f"Rotated {deg}°")
            plt.close("all")

    # ── 2. TRANSLATION ─────────────────────────────────────────────────────────
    print("  [translation]")
    shifts = [(4, 0, "right4"), (0, 4, "down4"), (4, 4, "diag4"),
              (1, 0, "right1"), (0, 1, "down1")]
    for dc, dr, shift_label in shifts:
        for name, grid in grids.items():
            grid_tf = Transforms.translate(grid, dr=dr, dc=dc)
            pre1,  post1  = extractor(grid)
            pre2,  post2  = extractor(grid_tf)
            pre_s  = compare(pre1, pre2)
            post_s = compare(post1, post2)
            title = (f"{tag.upper()} — translation ({shift_label}) | {name}\n"
                     f"pre cos={pre_s['mean_cos']:.3f}   post cos={post_s['mean_cos']:.3f}")
            out = f"{prefix}_trans_{shift_label}_{name}.png"
            plot_comparison(grid, grid_tf, pre_s, post_s, title,
                            out_path=out,
                            label_tf=f"Shifted {shift_label}")
            plt.close("all")

    # ── 3. FLIP ────────────────────────────────────────────────────────────────
    print("  [flip]")
    for flip_fn, flip_label in [(Transforms.flip_h, "horiz"), (Transforms.flip_v, "vert")]:
        for name, grid in grids.items():
            grid_tf = flip_fn(grid)
            pre1,  post1  = extractor(grid)
            pre2,  post2  = extractor(grid_tf)
            pre_s  = compare(pre1, pre2)
            post_s = compare(post1, post2)
            title = (f"{tag.upper()} — flip {flip_label} | {name}\n"
                     f"pre cos={pre_s['mean_cos']:.3f}   post cos={post_s['mean_cos']:.3f}")
            out = f"{prefix}_flip_{flip_label}_{name}.png"
            plot_comparison(grid, grid_tf, pre_s, post_s, title,
                            out_path=out,
                            label_tf=f"Flip {flip_label}")
            plt.close("all")

    # ── 4. PERTURBATION — cell flip ────────────────────────────────────────────
    print("  [perturbation / cell flip]")
    # Flip a live cell (alive→dead) and a dead cell (dead→alive) in the glider
    glider = make_glider(r0=5, c0=5)
    live_cells  = list(zip(*np.where(glider == 1)))
    dead_cells  = list(zip(*np.where(glider == 0)))

    for (r, c), label in [(live_cells[0], "alive2dead"), (dead_cells[0], "dead2alive")]:
        grid_tf = Transforms.cell_flip(glider, r, c)
        pre1,  post1 = extractor(glider)
        pre2,  post2 = extractor(grid_tf)
        pre_s  = compare(pre1, pre2)
        post_s = compare(post1, post2)
        title = (f"{tag.upper()} — cell flip ({label}) at ({r},{c}) | glider\n"
                 f"pre cos={pre_s['mean_cos']:.4f}   post cos={post_s['mean_cos']:.4f}")
        out = f"{prefix}_cellflip_{label}_glider.png"
        plot_perturbation_spread(glider, [(r, c)], pre_s, post_s, title, out_path=out)
        plt.close("all")

    # Also on random grids
    for name, grid in [("random_d35", make_random(0.35)), ("random_d65", make_random(0.65))]:
        lives = list(zip(*np.where(grid == 1)))
        r, c = lives[0]
        grid_tf = Transforms.cell_flip(grid, r, c)
        pre1,  post1 = extractor(grid)
        pre2,  post2 = extractor(grid_tf)
        pre_s  = compare(pre1, pre2)
        post_s = compare(post1, post2)
        title = (f"{tag.upper()} — cell flip (alive→dead) at ({r},{c}) | {name}\n"
                 f"pre cos={pre_s['mean_cos']:.4f}   post cos={post_s['mean_cos']:.4f}")
        out = f"{prefix}_cellflip_alive2dead_{name}.png"
        plot_perturbation_spread(grid, [(r, c)], pre_s, post_s, title, out_path=out)
        plt.close("all")

    # ── 5. PERTURBATION — cell move ────────────────────────────────────────────
    print("  [perturbation / cell move]")
    glider = make_glider(r0=10, c0=10)
    live_cells = list(zip(*np.where(glider == 1)))

    for (r, c) in live_cells[:3]:   # move first 3 live cells
        for (dr, dc), move_label in [((1, 0), "down1"), ((0, 1), "right1"), ((2, 2), "diag2")]:
            nr, nc = (r + dr) % GRID_SIZE, (c + dc) % GRID_SIZE
            if glider[nr, nc] == 1:
                continue   # destination occupied — skip
            grid_tf = Transforms.cell_move(glider, r, c, dr, dc)
            pre1,  post1 = extractor(glider)
            pre2,  post2 = extractor(grid_tf)
            pre_s  = compare(pre1, pre2)
            post_s = compare(post1, post2)
            title = (f"{tag.upper()} — cell move ({r},{c})→({nr},{nc}) [{move_label}] | glider\n"
                     f"pre cos={pre_s['mean_cos']:.4f}   post cos={post_s['mean_cos']:.4f}")
            out = f"{prefix}_cellmove_{move_label}_r{r}c{c}_glider.png"
            plot_perturbation_spread(glider, [(r, c), (nr, nc)], pre_s, post_s, title, out_path=out)
            plt.close("all")


# ── Print summary table ────────────────────────────────────────────────────────

def print_summary(tag: str, extractor: EmbeddingExtractor):
    """Print a compact table of mean cosine similarities for each transform."""
    print(f"\n{'='*70}")
    print(f"  {tag.upper()} — mean cosine similarity summary")
    print(f"{'='*70}")
    print(f"  {'Transform':<35} {'Config':<14} {'Pre':>7} {'Post':>7}")
    print(f"  {'-'*65}")

    grids = {
        "glider":     make_glider(),
        "random_d35": make_random(density=0.35),
        "random_d65": make_random(density=0.65),
    }

    transforms = [
        ("rotate 90°",    lambda g: Transforms.rotate(g, 1)),
        ("rotate 180°",   lambda g: Transforms.rotate(g, 2)),
        ("flip horiz",    Transforms.flip_h),
        ("flip vert",     Transforms.flip_v),
        ("translate +4cols",  lambda g: Transforms.translate(g, 0, 4)),
        ("translate +1col",   lambda g: Transforms.translate(g, 0, 1)),
    ]

    for tf_name, tf_fn in transforms:
        for name, grid in grids.items():
            grid_tf = tf_fn(grid)
            pre1,  post1  = extractor(grid)
            pre2,  post2  = extractor(grid_tf)
            pre_s  = compare(pre1, pre2)
            post_s = compare(post1, post2)
            print(f"  {tf_name:<35} {name:<14} {pre_s['mean_cos']:>7.3f} {post_s['mean_cos']:>7.3f}")

    # perturbation summary
    print(f"  {'-'*65}")
    glider = make_glider()
    lives  = list(zip(*np.where(glider == 1)))
    for (r, c), label in [(lives[0], "alive→dead"), (list(zip(*np.where(glider == 0)))[0], "dead→alive")]:
        grid_tf = Transforms.cell_flip(glider, r, c)
        pre1,  post1 = extractor(glider)
        pre2,  post2 = extractor(grid_tf)
        pre_s  = compare(pre1, pre2)
        post_s = compare(post1, post2)
        print(f"  {'cell flip '+label:<35} {'glider':<14} {pre_s['mean_cos']:>7.4f} {post_s['mean_cos']:>7.4f}")

    glider2 = make_glider(r0=10, c0=10)
    r, c = list(zip(*np.where(glider2 == 1)))[0]
    grid_tf = Transforms.cell_move(glider2, r, c, 1, 0)
    pre1,  post1 = extractor(glider2)
    pre2,  post2 = extractor(grid_tf)
    pre_s  = compare(pre1, pre2)
    post_s = compare(post1, post2)
    print(f"  {'cell move down1':<35} {'glider':<14} {pre_s['mean_cos']:>7.4f} {post_s['mean_cos']:>7.4f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10", action="store_true", help="Also analyse V10")
    parser.add_argument("--no-figs", action="store_true", help="Skip saving figures (summary only)")
    args = parser.parse_args()

    print("Loading V8 …")
    v8  = load_v8()
    ext_v8  = EmbeddingExtractor(v8)

    print_summary("v8", ext_v8)

    if not args.no_figs:
        print("\nGenerating V8 figures …")
        run_all("v8", ext_v8)

    if args.v10:
        print("\nLoading V10 …")
        v10 = load_v10()
        ext_v10 = EmbeddingExtractor(v10)
        print_summary("v10", ext_v10)
        if not args.no_figs:
            print("\nGenerating V10 figures …")
            run_all("v10", ext_v10)

    print("\nDone.")


if __name__ == "__main__":
    main()
