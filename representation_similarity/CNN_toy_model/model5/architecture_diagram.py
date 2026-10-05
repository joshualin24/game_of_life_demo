"""Static architecture diagram for Model 5 -- Part A (backbone), Part B
(pose predictor), and how combine() ties them together. Matches the level
of detail in notes.md; purely illustrative, no model loaded.

Run: python architecture_diagram.py
Outputs -> results/architecture_diagram.png
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

COL_A = "#3b6ea5"    # Part A -- blue
COL_B = "#b5651d"    # Part B -- orange/brown
COL_COMBINE = "#4f7942"  # combine() -- green
COL_IO = "#555555"   # input/output -- grey
FONT = 9.5


def box(ax, xy, w, h, text, color, fontsize=FONT, textcolor="white"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                            linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.88)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, linespacing=1.35)
    return (x + w / 2, y), (x + w / 2, y + h)  # (bottom-center, top-center)


def arrow(ax, p0, p1, color="#333333", style="-", lw=1.4, connectionstyle=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                         linewidth=lw, color=color, linestyle=style,
                         connectionstyle=connectionstyle, shrinkA=2, shrinkB=2, zorder=1)
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    # ---- input ----
    _, in_top = box(ax, (5.3, 10.0), 2.4, 0.6, "input grid  x\n(1, 40, 40)  {0,1}", COL_IO, fontsize=10)
    in_bot = (6.5, 10.0)

    # ======================= PART A (left column) =======================
    ax.text(2.5, 9.55, "PART A -- local backbone (Model 4's, unchanged)", fontsize=11,
            color=COL_A, weight="bold", ha="center")

    a1_top_c, a1_top_t = box(ax, (1.1, 8.65), 2.8, 0.7,
                              "IsotropicConv2d(1→8)\n3x3 circular, 3 free wts/pair", COL_A)
    a2_top_c, a2_top_t = box(ax, (1.1, 7.7), 2.8, 0.55, "PolyActivation(8)\nw0+w1x+w2x²", COL_A)
    a3_top_c, a3_top_t = box(ax, (1.1, 6.75), 2.8, 0.55, "Conv2d 1x1 (8→4)", COL_A)
    a4_top_c, a4_top_t = box(ax, (1.1, 5.8), 2.8, 0.55, "PolyActivation(4)", COL_A)
    feat_c, feat_t = box(ax, (1.1, 4.75), 2.8, 0.7,
                          "feat\n(4, 40, 40) -- RAW,\nNOT canonicalized", COL_A, fontsize=9)

    arrow(ax, in_bot, (2.5, 9.35))
    arrow(ax, (2.5, a1_top_c[1]), a1_top_t)
    arrow(ax, a1_top_c, a2_top_t)
    arrow(ax, a2_top_c, a3_top_t)
    arrow(ax, a3_top_c, a4_top_t)
    arrow(ax, a4_top_c, feat_t)

    ax.text(2.5, 4.55, "109 params total (Part A + head)", fontsize=8, color=COL_A,
            ha="center", style="italic")

    # ======================= PART B (right column) =======================
    ax.text(10.1, 9.55, "PART B -- global pose predictor (new, v2 pooling)", fontsize=11,
            color=COL_B, weight="bold", ha="center")

    coord_c, coord_t = box(ax, (8.7, 8.65), 2.8, 0.7,
                            "concat[x, coord_r, coord_c]\n(3, 40, 40) -- CoordConv", COL_B)
    b1_c, b1_t = box(ax, (8.7, 7.7), 2.8, 0.55, "Conv2d 3x3 (3→16) + GELU", COL_B)
    b2_c, b2_t = box(ax, (8.7, 6.75), 2.8, 0.55, "Conv2d 3x3 (16→16) + GELU", COL_B)
    pool_c, pool_t = box(ax, (8.4, 5.55), 3.4, 0.85,
                          "masked MEAN pool, alive cells only:\n(h·x).sum(spatial)/count\n+ log1p(count)  →  (17,)", COL_B, fontsize=8.7)
    fc_c, fc_t = box(ax, (8.7, 4.75), 2.8, 0.55, "fc1 (17→32) + GELU", COL_B)

    arrow(ax, (6.5, 10.0), (10.1, 9.35), connectionstyle="arc3,rad=0.15")
    arrow(ax, (10.1, coord_c[1]), coord_t)
    arrow(ax, coord_c, b1_t)
    arrow(ax, b1_c, b2_t)
    arrow(ax, b2_c, pool_t)
    arrow(ax, pool_c, fc_t)

    shift_c, shift_t = box(ax, (7.2, 3.35), 2.1, 0.75,
                            "shift_head\n(32→2)\nshift_pred  (B,2)", COL_B, fontsize=8.5)
    gidx_c, gidx_t = box(ax, (9.9, 3.35), 2.1, 0.75,
                          "gidx_head\n(32→8)\ngidx_logits  (B,8)", COL_B, fontsize=8.5)
    arrow(ax, (8.9, fc_c[1] - 0.55), shift_t, connectionstyle="arc3,rad=-0.1")
    arrow(ax, (9.5, fc_c[1] - 0.55), gidx_t, connectionstyle="arc3,rad=0.1")

    ax.text(10.1, 4.42, "x itself masks the pool -- fixes v1's\ndead-cell-background collapse (see notes.md)",
            fontsize=7.2, color=COL_B, ha="center", va="center", style="italic")

    round_c, round_t = box(ax, (7.2, 2.3), 2.1, 0.55, "round()\n→ shift_used (B,2)", "#8a8a8a", fontsize=8.3)
    argmax_c, argmax_t = box(ax, (9.9, 2.3), 2.1, 0.55, "argmax()\n→ gidx_used (B,)", "#8a8a8a", fontsize=8.3)
    arrow(ax, shift_c, round_t, style=":", color="#8a8a8a")
    arrow(ax, gidx_c, argmax_t, style=":", color="#8a8a8a")
    ax.text(10.6, 2.05, "no_grad() -- discretization only;\nonly L_pose trains Part B here",
            fontsize=6.8, color="#666666", ha="center", va="center", style="italic")

    # ======================= combine() (bottom, spans both, single clean row) ===
    cw, gap = 1.95, 0.17
    xs = [0.25 + i * (cw + gap) for i in range(6)]
    cy = 0.85
    cb1_c, cb1_t = box(ax, (xs[0], cy), cw, 0.6, "shift feat\n(apply_shift_batched)", COL_COMBINE, fontsize=7.8)
    cb2_c, cb2_t = box(ax, (xs[1], cy), cw, 0.6, "rotate/reflect\n(apply_d4_batched)", COL_COMBINE, fontsize=7.8)
    cb3_c, cb3_t = box(ax, (xs[2], cy), cw, 0.6, "head\nConv2d 1x1 (4→1)", COL_COMBINE, fontsize=7.8)
    cb4_c, cb4_t = box(ax, (xs[3], cy), cw, 0.6, "undo rotate\n(gidx⁻¹, reused)", COL_COMBINE, fontsize=7.8)
    cb5_c, cb5_t = box(ax, (xs[4], cy), cw, 0.6, "undo shift\n(−shift_used, reused)", COL_COMBINE, fontsize=7.8)
    out_c, out_t = box(ax, (xs[5], cy), cw, 0.6, "logits (1,40,40)\n→ t+1 prediction", COL_IO, fontsize=7.8)

    arrow(ax, feat_c, cb1_t)
    arrow(ax, cb1_c, cb2_t)
    arrow(ax, cb2_c, cb3_t)
    arrow(ax, cb3_c, cb4_t)
    arrow(ax, cb4_c, cb5_t)
    arrow(ax, cb5_c, out_t)
    arrow(ax, round_c, cb1_t, color="#8a8a8a")
    arrow(ax, argmax_c, cb2_t, color="#8a8a8a")

    ax.text(6.5, 0.65, "combine(feat, shift_used, gidx_used)  --  shared pipeline, no learned weights except head",
            fontsize=9.5, color=COL_COMBINE, weight="bold", ha="center")

    # legend
    legend_elems = [
        Line2D([0], [0], color="#333333", lw=1.6, label="forward data flow"),
        Line2D([0], [0], color="#8a8a8a", lw=1.6, linestyle=":", label="no_grad() (discretize, used by combine())"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.02),
              ncol=2, fontsize=8.5, frameon=False)

    fig.suptitle("Model 5 architecture: Part A (local, equivariant-by-construction) +\n"
                 "Part B (global, learned pose) combined via combine()",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(RESULTS, "architecture_diagram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
