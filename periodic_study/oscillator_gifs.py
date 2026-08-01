"""
oscillator_gifs.py — For each period with data, pick the top 3 oscillators,
decode their apgcode to a grid, simulate one full period, and save a GIF
showing the oscillation. All periods are combined into one summary GIF too.

Apgcode decoding follows the Catagolue/apgsearch encoding scheme:
  characters '0'-'9' -> 0-9, 'a'-'v' -> 10-31 (5-bit column of 5 cells)
  'w' = 2 zeros, 'x' = 3 zeros, 'y'+c = (val(c)+4) zeros, 'z' = row separator

Usage:
    python oscillator_gifs.py [--fps 4] [--pad 4]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

BG     = "#0f1117"
TEXT_C = "white"
PALETTE = ["#3a7bd5", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22"]


# ── GoL ───────────────────────────────────────────────────────────────────────

def gol_step(g):
    n = sum(np.roll(np.roll(g, i, 0), j, 1)
            for i in (-1,0,1) for j in (-1,0,1) if (i,j) != (0,0))
    return ((n == 3) | (g & (n == 2))).astype(np.uint8)


# ── Apgcode decoder ───────────────────────────────────────────────────────────

def _char_val(c):
    if '0' <= c <= '9': return ord(c) - ord('0')
    if 'a' <= c <= 'v': return ord(c) - ord('a') + 10
    return None

def decode_apgcode(code):
    """
    Decode a Catagolue apgcode (the part after the underscore, e.g. '7', '7e')
    to a numpy bool array (H, W).
    """
    # Expand run-length shorthands
    expanded = []
    i = 0
    while i < len(code):
        c = code[i]
        if c == 'w':
            expanded += [0, 0]
        elif c == 'x':
            expanded += [0, 0, 0]
        elif c == 'y':
            i += 1
            n = _char_val(code[i]) + 4
            expanded += [0] * n
        elif c == 'z':
            expanded.append('z')
        else:
            v = _char_val(c)
            if v is not None:
                expanded.append(v)
        i += 1

    # Split into rows of 5-cell-tall blocks separated by 'z'
    rows = []
    current = []
    for token in expanded:
        if token == 'z':
            rows.append(current)
            current = []
        else:
            current.append(token)
    rows.append(current)

    # Each row is a list of 5-bit values, each encoding a column of 5 cells
    # bit 0 = top cell, bit 4 = bottom cell
    # Build a 2D grid of cells
    if not rows:
        return np.zeros((1, 1), bool)

    max_cols = max(len(r) for r in rows)
    h = len(rows) * 5
    w = max_cols
    grid = np.zeros((h, w), bool)

    for row_idx, row in enumerate(rows):
        for col_idx, val in enumerate(row):
            for bit in range(5):
                if val & (1 << bit):
                    grid[row_idx * 5 + bit, col_idx] = True

    # Crop to bounding box
    rows_any = np.any(grid, axis=1)
    cols_any = np.any(grid, axis=0)
    if not rows_any.any():
        return np.zeros((1, 1), bool)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return grid[r0:r1+1, c0:c1+1].astype(np.uint8)


def pattern_to_grid(pattern, size=80, pad=6):
    """Place pattern centred in a size×size grid with padding."""
    g = np.zeros((size, size), np.uint8)
    h, w = pattern.shape
    if h + 2*pad > size or w + 2*pad > size:
        # scale size to fit
        size = max(h, w) + 2*pad + 2
        g = np.zeros((size, size), np.uint8)
    r = (size - h) // 2
    c = (size - w) // 2
    g[r:r+h, c:c+w] = pattern
    return g


def simulate(init_grid, steps):
    frames = [init_grid.copy()]
    g = init_grid.copy()
    for _ in range(steps):
        g = gol_step(g)
        frames.append(g.copy())
    return frames


# ── GIF per period: top N oscillators side by side ────────────────────────────

def make_period_gif(period, objects, fps, n_examples=3, n_cycles=3):
    """objects: list of {apgcode, occurrences} sorted by desc occurrences."""
    chosen = objects[:n_examples]
    steps  = period * n_cycles

    # Decode & simulate each
    all_frames = []
    valid = []
    for obj in chosen:
        code = obj["apgcode"].split("_", 1)[-1]  # strip 'xp2_' prefix
        try:
            pat = decode_apgcode(code)
        except Exception as e:
            print(f"    [skip] {obj['apgcode']}: decode failed ({e})")
            continue
        grid = pattern_to_grid(pat)
        frames = simulate(grid, steps)
        # Crop frames to bounding box with padding for display
        cells = np.any(np.stack(frames), axis=0)
        rr = np.where(np.any(cells, axis=1))[0]
        cc = np.where(np.any(cells, axis=0))[0]
        if len(rr) == 0:
            continue
        pad = 4
        r0 = max(0, rr[0]-pad); r1 = min(cells.shape[0], rr[-1]+pad+1)
        c0 = max(0, cc[0]-pad); c1 = min(cells.shape[1], cc[-1]+pad+1)
        frames_crop = [f[r0:r1, c0:c1] for f in frames]
        all_frames.append(frames_crop)
        valid.append(obj)

    if not valid:
        print(f"  [skip] period {period}: no valid patterns decoded")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4.0))
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Period {period} oscillators  (Catagolue B3/S23)",
                 color=TEXT_C, fontsize=11, fontweight="bold", y=1.01)

    ims, titles = [], []
    for i, (obj, frames) in enumerate(zip(valid, all_frames)):
        axes[i].set_facecolor(BG)
        axes[i].axis("off")
        im = axes[i].imshow(frames[0], cmap="binary", vmin=0, vmax=1,
                             interpolation="nearest", aspect="equal")
        occ = obj["occurrences"]
        occ_str = f"{occ:,.0f}" if occ < 1e6 else f"{occ:.2e}"
        t = axes[i].set_title(f"{obj['apgcode']}\n{occ_str} occurrences",
                               color=TEXT_C, fontsize=7, pad=3)
        ims.append(im); titles.append(t)

    step_label = fig.text(0.5, -0.02, "t=0", ha="center",
                           color=TEXT_C, fontsize=9)
    fig.tight_layout(pad=0.5)

    total_frames = steps + 1

    def update(t):
        for i in range(n):
            f_idx = t % len(all_frames[i])
            ims[i].set_data(all_frames[i][f_idx])
        step_label.set_text(f"t={t}  (period {period})")
        return ims + [step_label]

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=1000//fps, blit=True)
    path = os.path.join(OUT, f"osc_xp{period}.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── Combined summary GIF: one example per period, all in a grid ──────────────

def make_summary_gif(period_data, fps, n_cycles=3):
    """period_data: list of (period, top_object, frames_crop)"""
    n = len(period_data)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.8, nrows * 3.2))
    axes = axes.flat
    fig.patch.set_facecolor(BG)
    fig.suptitle("Naturally-occurring GoL oscillators by period (Catagolue B3/S23)",
                 color=TEXT_C, fontsize=11, fontweight="bold")

    ims = []
    max_frames = 0
    active_axes = []

    for ax, (period, obj, frames) in zip(axes, period_data):
        ax.set_facecolor(BG); ax.axis("off")
        im = ax.imshow(frames[0], cmap="binary", vmin=0, vmax=1,
                        interpolation="nearest", aspect="equal")
        occ = obj["occurrences"]
        occ_str = f"{occ:.1e}" if occ >= 1e6 else f"{occ:,}"
        ax.set_title(f"xp{period}\n{occ_str}", color=TEXT_C, fontsize=7, pad=2)
        ims.append((im, frames))
        max_frames = max(max_frames, len(frames))
        active_axes.append(ax)

    # Hide unused axes
    for ax in list(axes)[len(period_data):]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)

    def update(t):
        artists = []
        for im, frames in ims:
            im.set_data(frames[t % len(frames)])
            artists.append(im)
        return artists

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  interval=1000//fps, blit=True)
    path = os.path.join(OUT, "osc_summary.gif")
    ani.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved → {os.path.basename(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps",       type=int, default=4)
    ap.add_argument("--n-cycles",  type=int, default=4,
                    help="how many full periods to animate")
    ap.add_argument("--n-examples", type=int, default=3,
                    help="top N objects per period")
    args = ap.parse_args()

    with open(os.path.join(DATA, "summary.json")) as f:
        summary = json.load(f)

    summary_data = []  # for combined GIF

    print("Generating per-period oscillator GIFs …")
    for row in summary:
        period = row["period"]
        if row["n_objects"] == 0:
            continue
        path = os.path.join(DATA, f"census_xp{period}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            objects = json.load(f)
        objects = sorted(objects, key=lambda x: -x["occurrences"])

        print(f"  Period {period} ({len(objects)} objects) …")
        make_period_gif(period, objects, args.fps, args.n_examples, args.n_cycles)

        # For summary: pick top object
        for obj in objects[:1]:
            code = obj["apgcode"].split("_", 1)[-1]
            try:
                pat = decode_apgcode(code)
                grid = pattern_to_grid(pat, size=40, pad=3)
                frames = simulate(grid, period * args.n_cycles)
                # Crop
                cells = np.any(np.stack(frames), axis=0)
                rr = np.where(np.any(cells, axis=1))[0]
                cc = np.where(np.any(cells, axis=0))[0]
                if len(rr) == 0: continue
                pad = 3
                r0 = max(0, rr[0]-pad); r1 = min(cells.shape[0], rr[-1]+pad+1)
                c0 = max(0, cc[0]-pad); c1 = min(cells.shape[1], cc[-1]+pad+1)
                frames_crop = [f[r0:r1, c0:c1] for f in frames]
                summary_data.append((period, obj, frames_crop))
            except Exception:
                pass

    if summary_data:
        print("\nGenerating summary GIF …")
        make_summary_gif(summary_data, args.fps, args.n_cycles)

    print("Done.")


if __name__ == "__main__":
    main()
