"""Direction 3 — layer-wise evolution of the representation (V8 vs V10).

For the full stimulus set, mean-pool each stage to (N, d_model) and track,
per stage:

  spread        mean off-diagonal cosine similarity across stimuli
                (high -> representation collapsed / anisotropic)
  align_to_cnn  mean per-stimulus cosine( stage , cnn stage )
                (how far the transformer has moved each grid from its CNN code)
  step_cka      linear CKA( stage_{j-1} , stage_j )
                (1 -> this layer barely changed the representation)
  part_ratio    participation ratio = effective dimensionality

Plotted against normalised depth so V8's 4 layers and V10's 6 line up.

Outputs: results/layerwise_metrics.png  +  results/layerwise_metrics.json
         findings appended to notes.md
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import cosine_rsm, mean_pool, mean_offdiag, linear_cka, participation_ratio
from extract import embeddings_for_grids, stage_names
from load_models import load_v8, load_v10
from stimuli import build_stimuli

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def per_model_metrics(model, grids):
    names = stage_names(model)
    pooled = {k: mean_pool(v) for k, v in embeddings_for_grids(model, grids).items()}
    cnn = pooled["cnn"]
    cnn_n = cnn / (np.linalg.norm(cnn, axis=1, keepdims=True) + 1e-12)

    m = {"stages": names, "depth": list(np.linspace(0, 1, len(names))),
         "spread": [], "align_to_cnn": [], "step_cka": [], "part_ratio": []}
    prev = None
    for n in names:
        X = pooled[n]
        m["spread"].append(mean_offdiag(cosine_rsm(X)))
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        m["align_to_cnn"].append(float((Xn * cnn_n).sum(1).mean()))
        m["part_ratio"].append(participation_ratio(X))
        m["step_cka"].append(1.0 if prev is None else linear_cka(prev, X))
        prev = X
    return m


def main():
    grids, meta = build_stimuli()
    print(f"stimuli: {len(grids)}")
    out = {}
    for tag, load in (("V8", load_v8), ("V10", load_v10)):
        out[tag] = per_model_metrics(load(), grids)

    panels = [("spread", "mean off-diag cosine (collapse)"),
              ("align_to_cnn", "cosine( stage , CNN stage )"),
              ("step_cka", "linear CKA to previous stage"),
              ("part_ratio", "participation ratio (eff. dim)")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title) in zip(axes.ravel(), panels):
        for tag, style in (("V8", "o-"), ("V10", "s--")):
            d = out[tag]
            ax.plot(d["depth"], d[key], style, label=tag)
            for xi, yi, nm in zip(d["depth"], d[key], d["stages"]):
                ax.annotate(nm, (xi, yi), fontsize=6, alpha=.6,
                            textcoords="offset points", xytext=(0, 4))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("normalised depth  (cnn=0 … last tf layer=1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "layerwise_metrics.png"), dpi=130)
    plt.close(fig)

    with open(os.path.join(RESULTS, "layerwise_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    for tag in ("V8", "V10"):
        d = out[tag]
        lines.append(f"  {tag}")
        for i, n in enumerate(d["stages"]):
            lines.append(f"    {n:<10} spread={d['spread'][i]:+.3f}  "
                         f"align_cnn={d['align_to_cnn'][i]:+.3f}  "
                         f"step_cka={d['step_cka'][i]:.3f}  "
                         f"part_ratio={d['part_ratio'][i]:6.2f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS, "layerwise_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nfigure + json + report -> {RESULTS}")


if __name__ == "__main__":
    main()
