"""Direction 2 — similarity across DIFFERENT grids (inference-only, V8 & V10).

Builds a per-stage representational-similarity matrix (RSM) over the stimulus
set and runs three probes:

  A  same-future    : does GoL(A)==GoL(B)  ->  emb(A) ~ emb(B) ?
                      (B = A + one doomed isolated cell)
  B  phase clustering: do frames of one oscillator stay closer to each other
                      than to other oscillators?
  C  density         : for random grids, does |Δdensity| predict dissimilarity?

Outputs: results/grid_rsm_{v8,v10}.npz  +  results/grid_similarity.png
         findings appended to notes.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import gol_step, cosine_rsm, mean_pool, mean_offdiag
from extract import embeddings_for_grids, stage_names
from load_models import load_v8, load_v10
from stimuli import build_stimuli

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def pooled_by_stage(model, grids):
    emb = embeddings_for_grids(model, grids)
    return {k: mean_pool(v) for k, v in emb.items()}


# ── Probe A: same-future pairs ───────────────────────────────────────────────
def make_same_future_pairs(n=40, density=0.15, seed=7):
    rng = np.random.default_rng(seed)
    A, B = [], []
    while len(A) < n:
        a = (rng.random((40, 40)) < density).astype(np.uint8)
        # find a cell whose 5x5 neighbourhood is empty -> flipping it on is
        # a doomed lone cell; successor is unchanged
        cand = []
        for r in range(2, 38):
            for c in range(2, 38):
                if a[r - 2:r + 3, c - 2:c + 3].sum() == 0:
                    cand.append((r, c))
        if not cand:
            continue
        r, c = cand[rng.integers(len(cand))]
        b = a.copy()
        b[r, c] = 1
        if np.array_equal(gol_step(a), gol_step(b)):
            A.append(a)
            B.append(b)
    return np.stack(A), np.stack(B)


def probe_same_future(models):
    A, B = make_same_future_pairs()
    rng = np.random.default_rng(99)
    C = A[rng.permutation(len(A))]                    # matched-density unrelated grids
    rows = []
    for tag, m in models:
        names = stage_names(m)
        pa = pooled_by_stage(m, A)
        pb = pooled_by_stage(m, B)
        pc = pooled_by_stage(m, C)
        for n in names:
            def cos(x, y):
                x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
                y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
                return (x * y).sum(1)
            rows.append((tag, n, float(cos(pa[n], pb[n]).mean()),
                         float(cos(pa[n], pc[n]).mean())))
    return rows


# ── Probe B: oscillator/spaceship phase clustering ──────────────────────────
def probe_phase_clustering(models, grids, meta):
    idx = [i for i, r in enumerate(meta) if r["category"] in ("oscillator", "spaceship")]
    names_by = {}
    for i in idx:
        names_by.setdefault(meta[i]["name"], []).append(i)
    out = {}
    for tag, m in models:
        stg = pooled_by_stage(m, grids[idx])
        remap = {old: new for new, old in enumerate(idx)}
        per_stage = {}
        for sname, mat in stg.items():
            rsm = cosine_rsm(mat)
            within, between = [], []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    same = meta[idx[i]]["name"] == meta[idx[j]]["name"]
                    (within if same else between).append(rsm[i, j])
            per_stage[sname] = (float(np.mean(within)), float(np.mean(between)))
        out[tag] = per_stage
    return out


# ── Probe C: density gap vs dissimilarity (random grids) ────────────────────
def probe_density(models, grids, meta):
    idx = [i for i, r in enumerate(meta) if r["category"] == "random"]
    dens = np.array([meta[i]["density"] for i in idx])
    ddens = np.abs(dens[:, None] - dens[None, :])
    iu = np.triu_indices(len(idx), k=1)
    out = {}
    for tag, m in models:
        stg = pooled_by_stage(m, grids[idx])
        per_stage = {}
        for sname, mat in stg.items():
            rsm = cosine_rsm(mat)
            x, y = ddens[iu], rsm[iu]
            r = float(np.corrcoef(x, y)[0, 1])
            per_stage[sname] = r
        out[tag] = per_stage
    return out


# ── RSM heatmaps ───────────────────────────────────────────────────────────
def save_rsms(models, grids, meta):
    order = sorted(range(len(meta)), key=lambda i: (
        {"still_life": 0, "oscillator": 1, "spaceship": 2, "random": 3}[meta[i]["category"]],
        meta[i]["name"], meta[i]["phase"] or 0))
    cats = [meta[i]["category"] for i in order]
    bounds = [k for k in range(1, len(cats)) if cats[k] != cats[k - 1]]

    for tag, m in models:
        stg = pooled_by_stage(m, grids)
        names = stage_names(m)
        packed = {}
        fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.4))
        for ax, n in zip(axes, names):
            rsm = cosine_rsm(stg[n][order])
            packed[n] = rsm
            im = ax.imshow(rsm, vmin=-1, vmax=1, cmap="RdBu_r")
            for b in bounds:
                ax.axhline(b - .5, color="k", lw=.4)
                ax.axvline(b - .5, color="k", lw=.4)
            ax.set_title(f"{tag}  {n}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=.6, label="cosine")
        fig.savefig(os.path.join(RESULTS, f"grid_rsm_{tag.lower()}.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(os.path.join(RESULTS, f"grid_rsm_{tag.lower()}.npz"),
                            order=np.array(order), cats=np.array(cats), **packed)


def main():
    v8, v10 = load_v8(), load_v10()
    models = [("V8", v8), ("V10", v10)]
    grids, meta = build_stimuli()
    print(f"stimuli: {len(grids)}")

    save_rsms(models, grids, meta)
    A = probe_same_future(models)
    B = probe_phase_clustering(models, grids, meta)
    C = probe_density(models, grids, meta)

    lines = []
    lines.append("### Probe A — same-future pairs  (cos: A~B doomed-cell pair | A~unrelated)")
    for tag, n, sf, rnd in A:
        lines.append(f"  {tag:>4} {n:<10} same_future={sf:.4f}   unrelated={rnd:.4f}   gap={sf-rnd:+.4f}")
    lines.append("")
    lines.append("### Probe B — phase clustering  (within-pattern cos | between-pattern cos)")
    for tag in ("V8", "V10"):
        for n, (w, b) in B[tag].items():
            lines.append(f"  {tag:>4} {n:<10} within={w:.4f}   between={b:.4f}   sep={w-b:+.4f}")
    lines.append("")
    lines.append("### Probe C — corr(|Δdensity|, cosine sim) over random grids  (want negative)")
    for tag in ("V8", "V10"):
        for n, r in C[tag].items():
            lines.append(f"  {tag:>4} {n:<10} r={r:+.4f}")
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(RESULTS, "grid_similarity_report.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\nfigures + npz + report -> {RESULTS}")


if __name__ == "__main__":
    main()
