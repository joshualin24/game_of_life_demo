"""Shared helpers for the representation-similarity study: toroidal GoL,
period detection, pooling, and representational-similarity math.

Pure numpy (+ torch only for the pooling type hints). No repo imports.
"""
from __future__ import annotations

import numpy as np


# ── Game of Life (toroidal, matches the models' circular padding) ─────────────

def gol_step(g: np.ndarray) -> np.ndarray:
    n = sum(np.roll(np.roll(g, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
    return ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)


def simulate(g: np.ndarray, steps: int) -> list[np.ndarray]:
    out = [g.astype(np.uint8)]
    for _ in range(steps):
        out.append(gol_step(out[-1]))
    return out


def detect_period(g: np.ndarray, max_p: int = 40, max_shift: int = 3):
    """Smallest p such that state at step p equals the start up to a toroidal
    shift (handles spaceships). Returns (period, (dr, dc)) or (None, None)."""
    frames = simulate(g, max_p)
    g0 = frames[0]
    for p in range(1, max_p + 1):
        fp = frames[p]
        if fp.sum() != g0.sum():
            continue
        for dr in range(-max_shift, max_shift + 1):
            for dc in range(-max_shift, max_shift + 1):
                if np.array_equal(np.roll(np.roll(fp, -dr, 0), -dc, 1), g0):
                    return p, (dr, dc)
    return None, None


# ── Pooling per-patch embeddings (B, P, D) -> per-grid vectors ────────────────

def mean_pool(emb: np.ndarray) -> np.ndarray:
    """(B, P, D) -> (B, D)"""
    return emb.mean(axis=1)


def flatten_emb(emb: np.ndarray) -> np.ndarray:
    """(B, P, D) -> (B, P*D)"""
    return emb.reshape(emb.shape[0], -1)


# ── Representational similarity ──────────────────────────────────────────────

def l2norm(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)

def cosine_rsm(X: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, N) cosine-similarity matrix."""
    Xn = l2norm(X)
    return Xn @ Xn.T

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear CKA between two representations of the same N stimuli.
    X: (N, Dx), Y: (N, Dy). Symmetric, in [0, 1], invariant to rotation/scale."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.sum((X.T @ Y) ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0

def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality of (N, D) representation:
    (Σλ)² / Σλ²  over covariance eigenvalues. 1 = rank-1, D = isotropic."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else 0.0

def mean_offdiag(M: np.ndarray) -> float:
    n = M.shape[0]
    return float((M.sum() - np.trace(M)) / (n * (n - 1)))
