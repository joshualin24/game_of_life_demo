"""Phase 0 — symmetry baseline for V8's pooled final representation.

Views a symmetry transformation g as a displacement in embedding space,
    Delta_g(x) = psi(g.x) - psi(x)
and asks how content-independent that displacement is:

    dir_consistency  mean cosine( Delta_g(x_i) , mean_i Delta_g )   -> want 1
    mag_residual     mean ||Delta_g(x_i) - mean_i Delta_g|| / ||mean_i Delta_g||  -> want 0
    rel_move         mean ||Delta_g(x_i)|| / mean ||psi(x_i)||      (context)

psi(x) = mean over the 100 patch tokens of a chosen stage (tf_L4 by default).

Group v1 = toroidal translations only, split into two strata:
    transl_4cell     shifts that are multiples of the 4-cell patch stride
    transl_subpatch  the 16 sub-patch phases (dr, dc in 0..3)

Also times V8 feature extraction and prints a training-cost estimate.

Run:  python symmetry.py
Out:  results/symmetry_baseline.txt
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import embeddings_for_grids, stage_names
from load_models import load_v8

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


# ── toroidal translation on a batch of grids ────────────────────────────────
def translate(grids: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """grids: (N, H, W) -> shifted by (dr, dc) with wraparound."""
    return np.roll(np.roll(grids, dr, axis=1), dc, axis=2)


def subpatch_offsets() -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(4) for dc in range(4) if (dr, dc) != (0, 0)]


def fourcell_offsets(rng, k: int = 24) -> list[tuple[int, int]]:
    pool = [(dr, dc) for dr in range(0, 40, 4) for dc in range(0, 40, 4)
            if (dr, dc) != (0, 0)]
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in idx]


# ── pooled embedding function for a given stage ─────────────────────────────
def make_pooled_emb_fn(model, stage: str, batch: int = 128):
    def emb_fn(grids: np.ndarray) -> np.ndarray:
        st = embeddings_for_grids(model, grids.astype(np.uint8), batch=batch)
        return st[stage].mean(axis=1)          # (N, D)
    return emb_fn


# ── Delta-consistency stats ────────────────────────────────────────────────
def _unit(v, eps=1e-12):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + eps)

def delta_stats_for_offset(emb_fn, base_emb, grids, dr, dc):
    g_emb = emb_fn(translate(grids, dr, dc))       # (N, D)
    d = g_emb - base_emb                            # (N, D)
    dbar = d.mean(0, keepdims=True)                 # (1, D)
    dir_cons = float((_unit(d) * _unit(dbar)).sum(1).mean())
    mag_res = float(np.linalg.norm(d - dbar, axis=1).mean()
                    / (np.linalg.norm(dbar) + 1e-12))
    rel_move = float(np.linalg.norm(d, axis=1).mean()
                     / (np.linalg.norm(base_emb, axis=1).mean() + 1e-12))
    return dir_cons, mag_res, rel_move

def stratum_stats(emb_fn, grids, offsets):
    base = emb_fn(grids)
    dc_, mr_, rm_ = [], [], []
    for (dr, dc) in offsets:
        a, b, c = delta_stats_for_offset(emb_fn, base, grids, dr, dc)
        dc_.append(a); mr_.append(b); rm_.append(c)
    return dict(dir_consistency=float(np.mean(dc_)),
                mag_residual=float(np.mean(mr_)),
                rel_move=float(np.mean(rm_)),
                n_offsets=len(offsets))


# ── D4 point group on a square torus ───────────────────────────────────────
def d4_ops() -> dict:
    """8 elements of D4 acting on a batch of grids (N, H, W)."""
    return {
        "e":     lambda x: x,
        "r90":   lambda x: np.rot90(x, 1, axes=(1, 2)),
        "r180":  lambda x: np.rot90(x, 2, axes=(1, 2)),
        "r270":  lambda x: np.rot90(x, 3, axes=(1, 2)),
        "flip_v": lambda x: x[:, ::-1, :],
        "flip_h": lambda x: x[:, :, ::-1],
        "transp": lambda x: x.transpose(0, 2, 1),
        "atransp": lambda x: x.transpose(0, 2, 1)[:, ::-1, ::-1],
    }


def d4_linear_baseline(emb_fn, grids, split=0.6):
    """For each non-identity D4 element r, fit the optimal LINEAR map rho(r)
    on a train split (rho.T = lstsq(Psi, Psi_r)) and report the held-out
    residual  ||Psi_r - Psi @ rho.T||_F / ||Psi_r||_F , alongside the
    identity residual ||Psi_r - Psi|| / ||Psi_r|| (how far r moves psi at all)."""
    ops = d4_ops()
    N = len(grids)
    ntr = int(N * split)
    base = emb_fn(np.ascontiguousarray(grids))            # (N, D)
    rows = []
    for name, op in ops.items():
        if name == "e":
            continue
        tgt = emb_fn(np.ascontiguousarray(op(grids)))     # (N, D)
        Xtr, Ytr = base[:ntr], tgt[:ntr]
        Xte, Yte = base[ntr:], tgt[ntr:]
        # solve Xtr @ W ≈ Ytr  (W is rho.T, shape D×D), ridge for stability
        lam = 1e-3 * np.trace(Xtr.T @ Xtr) / Xtr.shape[1]
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                            Xtr.T @ Ytr)
        lin_res = float(np.linalg.norm(Yte - Xte @ W) / (np.linalg.norm(Yte) + 1e-12))
        id_res = float(np.linalg.norm(Yte - Xte) / (np.linalg.norm(Yte) + 1e-12))
        rows.append((name, id_res, lin_res))
    return rows


def random_grids(n, seed=0, densities=(0.15, 0.3, 0.45, 0.6)):
    rng = np.random.default_rng(seed)
    out = []
    per = n // len(densities)
    for d in densities:
        out.append((rng.random((per, 40, 40)) < d).astype(np.uint8))
    return np.concatenate(out, 0)


# ── timing trial ───────────────────────────────────────────────────────────
def timing_trial(model, n=512, batch=128):
    g = random_grids(n, seed=123)
    # warmup
    embeddings_for_grids(model, g[:batch], batch=batch)
    t0 = time.time()
    embeddings_for_grids(model, g, batch=batch)
    dt = time.time() - t0
    per_grid_ms = 1000 * dt / n
    return per_grid_ms


def main():
    torch.manual_seed(0)
    m = load_v8()
    dev = next(m.parameters()).device
    lines = [f"device={dev}", ""]

    grids = random_grids(192, seed=7)          # eval sample for the metric
    rng = np.random.default_rng(1)
    strata = {
        "transl_4cell":    fourcell_offsets(rng, k=24),
        "transl_subpatch": subpatch_offsets(),
    }

    lines.append(f"Delta-consistency of raw V8 pooled embeddings  "
                 f"(N={len(grids)} grids)")
    lines.append(f"{'stage':<10} {'stratum':<16} {'dir_cons':>9} "
                 f"{'mag_resid':>10} {'rel_move':>9}")
    for stage in stage_names(m):
        emb_fn = make_pooled_emb_fn(m, stage)
        for sname, offs in strata.items():
            s = stratum_stats(emb_fn, grids, offs)
            lines.append(f"{stage:<10} {sname:<16} {s['dir_consistency']:>9.4f} "
                         f"{s['mag_residual']:>10.4f} {s['rel_move']:>9.4f}")
    lines.append("")
    lines.append("dir_cons -> 1 = transformation is a content-independent direction")
    lines.append("mag_resid -> 0 = also content-independent in magnitude")
    lines.append("")

    # D4 baseline: best linear operator rho(r), held-out residual
    lines.append(f"D4 equivariance of raw V8 pooled embeddings  "
                 f"(fit optimal linear rho(r) on 60%, residual on 40%)")
    lines.append(f"{'stage':<10} {'r':<9} {'id_resid':>9} {'lin_resid':>10}")
    for stage in stage_names(m):
        emb_fn = make_pooled_emb_fn(m, stage)
        for name, id_res, lin_res in d4_linear_baseline(emb_fn, grids):
            lines.append(f"{stage:<10} {name:<9} {id_res:>9.4f} {lin_res:>10.4f}")
    lines.append("")
    lines.append("id_resid  = ||psi(r.x) - psi(x)|| / ||psi(r.x)||   (how far r moves psi)")
    lines.append("lin_resid = residual after best linear rho(r); -> 0 = psi is linearly D4-equivariant")
    lines.append("")

    # timing
    per_ms = timing_trial(m)
    lines.append(f"Timing:  V8 tf-stage extraction = {per_ms:.3f} ms/grid  (batched)")
    for (N, B, k, E) in [(20000, 64, 8, 20), (20000, 64, 16, 20), (40000, 64, 8, 30)]:
        # per step: (1 + k) transformed views through frozen V8
        fwd_per_epoch = N * (1 + k)
        sec_epoch = fwd_per_epoch * per_ms / 1000
        lines.append(f"  train N={N} batch={B} k={k} epochs={E}:  "
                     f"~{sec_epoch/60:.1f} min/epoch (V8 fwd only)  "
                     f"~{sec_epoch*E/3600:.2f} h total (V8 fwd only; "
                     f"adapter+decoder+backward extra)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS, "symmetry_baseline.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n-> {RESULTS}/symmetry_baseline.txt")


if __name__ == "__main__":
    main()
