"""
Generates two architecture diagrams for the trajectory-invariant embedding experiment:

1. architecture_overview.png  — full pipeline: GoL trajectory → Encoder →
   Projection Head → Contrastive Loss, with positive/negative pair annotation
2. architecture_encoder.png   — detailed encoder block diagram (conv layers,
   BN, ReLU, dimensions at each stage)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#0f1117"
GRID_C   = "#e8e8e8"
ENC_C    = "#3a7bd5"
PROJ_C   = "#8e44ad"
LOSS_C   = "#e74c3c"
POS_C    = "#2ecc71"
NEG_C    = "#e67e22"
ARROW_C  = "#aaaaaa"
TEXT_C   = "#ffffff"
DIM_C    = "#aaaaaa"
CONV_C   = "#2471a3"
BN_C     = "#1a8a5e"
RELU_C   = "#b7950b"
FC_C     = "#884ea0"


def box(ax, xy, w, h, color, label, sublabel=None, fontsize=9, alpha=0.92, radius=0.03):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           fc=color, ec="white", lw=0.8, alpha=alpha,
                           zorder=3)
    ax.add_patch(patch)
    cy = y + h / 2
    ax.text(x + w / 2, cy + (0.012 if sublabel else 0), label,
            ha="center", va="center", color=TEXT_C, fontsize=fontsize,
            fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x + w / 2, cy - 0.022, sublabel,
                ha="center", va="center", color=DIM_C, fontsize=6.5, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=ARROW_C, lw=1.4, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)


def gol_mini(ax, x, y, w, h, seed=0):
    """Draw a tiny GoL grid thumbnail."""
    rng = np.random.default_rng(seed)
    grid = (rng.random((12, 12)) < 0.35).astype(float)
    ax.imshow(grid, cmap="binary", extent=[x, x+w, y, y+h],
              aspect="auto", interpolation="nearest", zorder=3, vmin=0, vmax=1)
    ax.add_patch(plt.Rectangle((x, y), w, h, fc="none",
                               ec="white", lw=0.7, zorder=4))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Full pipeline overview
# ═══════════════════════════════════════════════════════════════════════════════

def fig_overview():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Trajectory-Invariant Embedding: Training Pipeline",
                 color=TEXT_C, fontsize=13, fontweight="bold", pad=10)

    # ── Trajectory (positive pair) ──
    ax.text(0.01, 0.93, "GoL Trajectory  (same initial condition = positive pair)",
            color=POS_C, fontsize=8.5, fontweight="bold", va="top")

    grids_x = [0.01, 0.08, 0.15, 0.22]
    for k, gx in enumerate(grids_x):
        gol_mini(ax, gx, 0.60, 0.055, 0.22)
        ax.text(gx + 0.027, 0.58, f"S_{k}", color=DIM_C, fontsize=7, ha="center")
        if k < len(grids_x) - 1:
            arrow(ax, gx + 0.057, 0.71, gx + 0.077, 0.71, color=POS_C, lw=1.2)

    ax.text(0.30, 0.71, "…", color=DIM_C, fontsize=14, ha="center", va="center")

    # anchor = S_i, positive = S_j from same traj
    ax.text(0.01, 0.53, "Sample anchor  Sᵢ", color=TEXT_C, fontsize=7.5)
    ax.text(0.14, 0.53, "Sample positive  Sⱼ  (j ≠ i, same traj)",
            color=POS_C, fontsize=7.5)
    ax.text(0.14, 0.47, "Sample negatives  Sₖ  (different trajectories)",
            color=NEG_C, fontsize=7.5)

    # ── Encoder ──
    EX, EY, EW, EH = 0.37, 0.55, 0.13, 0.30
    box(ax, (EX, EY), EW, EH, ENC_C, "Encoder  f(·)",
        "Conv×4 + BN + ReLU\n→ flatten → FC\nout: latent dim d", fontsize=9)

    # arrows into encoder
    arrow(ax, 0.29, 0.72, EX, 0.76, color=TEXT_C)   # anchor
    arrow(ax, 0.29, 0.66, EX, 0.70, color=POS_C)    # positive
    arrow(ax, 0.29, 0.60, EX, 0.63, color=NEG_C)    # negatives

    ax.text(0.33, 0.75, "anchor", color=TEXT_C, fontsize=7, ha="center")
    ax.text(0.33, 0.69, "pos", color=POS_C, fontsize=7, ha="center")
    ax.text(0.33, 0.61, "neg×B", color=NEG_C, fontsize=7, ha="center")

    # shared weights label
    ax.text(EX + EW/2, EY - 0.04, "shared weights", color=DIM_C,
            fontsize=7, ha="center", style="italic")

    # ── Projection head ──
    PX, PY, PW, PH = 0.56, 0.58, 0.12, 0.22
    box(ax, (PX, PY), PW, PH, PROJ_C, "Projection\nHead  g(·)",
        "FC → ReLU → FC\nout: dim p (e.g. 128)", fontsize=8)
    arrow(ax, EX + EW, 0.70, PX, 0.70, color=ARROW_C)
    ax.text(0.52, 0.73, "h = f(S)", color=DIM_C, fontsize=7, ha="center")

    # ── NT-Xent loss ──
    LX, LY, LW, LH = 0.74, 0.58, 0.14, 0.22
    box(ax, (LX, LY), LW, LH, LOSS_C, "NT-Xent Loss",
        "pull: sim(zᵢ, zⱼ↑)\npush: sim(zᵢ, zₖ↓)\nτ = temperature", fontsize=8)
    arrow(ax, PX + PW, 0.70, LX, 0.70, color=ARROW_C)
    ax.text(0.70, 0.73, "z = g(h)", color=DIM_C, fontsize=7, ha="center")

    # ── Back-prop ──
    arrow(ax, LX + LW/2, LY, LX + LW/2, 0.45, color=LOSS_C, lw=1.2)
    arrow(ax, LX + LW/2, 0.45, EX + EW/2, 0.45, color=LOSS_C, lw=1.2)
    arrow(ax, EX + EW/2, 0.45, EX + EW/2, EY, color=LOSS_C, lw=1.2, style="-|>")
    ax.text(0.60, 0.41, "backprop  ∇L", color=LOSS_C, fontsize=7.5,
            ha="center", style="italic")

    # ── Downstream use (inference) ──
    ax.text(0.37, 0.32, "Inference (encoder only — projection head discarded):",
            color=DIM_C, fontsize=8, style="italic")

    box(ax, (0.37, 0.10), 0.13, 0.18, ENC_C, "Encoder  f(·)", fontsize=9, alpha=0.6)
    arrow(ax, 0.29, 0.19, 0.37, 0.19, color=DIM_C, lw=1.0)
    ax.text(0.27, 0.19, "Sₜ", color=DIM_C, fontsize=9, ha="center", va="center")

    box(ax, (0.56, 0.10), 0.22, 0.18, "#2c3e50", "Embedding space",
        "cluster / classify / visualize\ntrajectories by initial condition", fontsize=8)
    arrow(ax, 0.50, 0.19, 0.56, 0.19, color=DIM_C, lw=1.0)
    ax.text(0.53, 0.22, "h = f(Sₜ)", color=DIM_C, fontsize=7, ha="center")

    # ── Legend ──
    handles = [
        mpatches.Patch(color=POS_C, label="Positive pair (same trajectory)"),
        mpatches.Patch(color=NEG_C, label="Negative pair (different trajectory)"),
        mpatches.Patch(color=ENC_C, label="Encoder (shared weights)"),
        mpatches.Patch(color=PROJ_C, label="Projection head (training only)"),
        mpatches.Patch(color=LOSS_C, label="NT-Xent contrastive loss"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C,
              framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT, "architecture_overview.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Encoder detail
# ═══════════════════════════════════════════════════════════════════════════════

def fig_encoder():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Encoder Architecture Detail", color=TEXT_C,
                 fontsize=13, fontweight="bold", pad=10)

    stages = [
        # (label, sublabel, color, x_center)
        ("Input\n1×64×64",      "",                     GRID_C,  0.04),
        ("Conv2d\n1→32, k4s2",  "64→32",                CONV_C,  0.17),
        ("BN+ReLU",             "32×32×32",              BN_C,    0.26),
        ("Conv2d\n32→64, k4s2", "32→16",                CONV_C,  0.36),
        ("BN+ReLU",             "64×16×16",              BN_C,    0.45),
        ("Conv2d\n64→128,k4s2", "16→8",                 CONV_C,  0.54),
        ("BN+ReLU",             "128×8×8",               BN_C,    0.63),
        ("Conv2d\n128→256,k4s2","8→4",                  CONV_C,  0.72),
        ("BN+ReLU\n+Flatten",   "256×4×4=4096",          BN_C,    0.81),
        ("FC → μ\nFC → logσ²", "4096→d\n(d=64)",       FC_C,    0.91),
    ]

    BW, BH = 0.075, 0.42
    BY = 0.28

    prev_xr = None
    for i, (label, sub, color, xc) in enumerate(stages):
        bx = xc - BW / 2
        fc = color if color != GRID_C else "#2c2c2c"
        ec = "white" if color != GRID_C else "#888"
        patch = FancyBboxPatch((bx, BY), BW, BH,
                               boxstyle="round,pad=0,rounding_size=0.02",
                               fc=fc, ec=ec, lw=0.8, alpha=0.93, zorder=3)
        ax.add_patch(patch)
        ax.text(xc, BY + BH/2 + 0.04, label, ha="center", va="center",
                color=TEXT_C, fontsize=7.2, fontweight="bold", zorder=4,
                multialignment="center")
        if sub:
            ax.text(xc, BY - 0.06, sub, ha="center", va="top",
                    color=DIM_C, fontsize=6.2, zorder=4, multialignment="center")

        if prev_xr is not None:
            arrow(ax, prev_xr, BY + BH/2, bx, BY + BH/2, color=ARROW_C, lw=1.3)
        prev_xr = bx + BW

    # input grid thumbnail
    gol_mini(ax, 0.005, BY, 0.065, BH, seed=7)

    # reparameterize annotation
    ax.text(0.91, BY + BH + 0.12,
            "z = μ + σ·ε,  ε~N(0,I)\n(reparameterization trick)",
            color=DIM_C, fontsize=7.5, ha="center", style="italic")

    # colour legend
    handles = [
        mpatches.Patch(color=CONV_C, label="Conv2d (stride 2)"),
        mpatches.Patch(color=BN_C,   label="BatchNorm + ReLU"),
        mpatches.Patch(color=FC_C,   label="Fully connected"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              facecolor="#1c1f26", edgecolor="#444", labelcolor=TEXT_C,
              framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(OUT, "architecture_encoder.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    fig_overview()
    fig_encoder()
    print("Done.")
