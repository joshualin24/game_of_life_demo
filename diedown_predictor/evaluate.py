"""
evaluate.py
-----------
Compares the trained PointerPolicyNet against the greedy/beam teacher
(teacher_search.greedy_search) on held-out base grids: average unified-score
gap, % matching die-down outcome, % exact flip-sequence match — plus a few
side-by-side comparison GIFs (baseline | teacher-perturbed | model-perturbed)
mirroring generate_diedown_gifs.py's styling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from nn.data_gen import run_trajectory
from nn.utils import DEVICE
from diedown_predictor.generate_dataset import GRID_SIZE, K, MARGIN, MAX_CANDIDATES, T, stratified_sample
from diedown_predictor.models import PointerPolicyNet
from diedown_predictor.teacher_search import candidate_neighborhood, greedy_search, unified_score_batch
from diedown_predictor.train import CKPT_DIR, RESULTS_DIR, TASK

STOP = GRID_SIZE * GRID_SIZE


@torch.no_grad()
def predict_perturbation(model, init, K=K, margin=MARGIN, max_candidates=MAX_CANDIDATES,
                          rng=None):
    """Autoregressive feedforward decode: up to K flips, no simulator calls."""
    H = W = GRID_SIZE
    candidates = candidate_neighborhood(init, margin=margin)
    if len(candidates) > max_candidates:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(candidates), max_candidates, replace=False)
        candidates = [candidates[j] for j in idx]
    cand_flat = {r * W + c for r, c in candidates}

    state = init.astype(np.float32).copy()
    chosen_mask = np.zeros((H, W), dtype=np.float32)
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()

    for step in range(K):
        k_remaining = (K - step) / K
        x = np.stack([state, chosen_mask, np.full((H, W), k_remaining, dtype=np.float32)])
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        valid = np.zeros(H * W + 1, dtype=bool)
        valid[STOP] = True
        for c in cand_flat - used:
            valid[c] = True
        mask = torch.tensor(valid, dtype=torch.bool, device=DEVICE).unsqueeze(0)

        logits = model(x, valid_mask=mask)
        pred = int(logits.argmax(dim=1).item())
        if pred == STOP:
            break

        r, c = divmod(pred, W)
        chosen.append((r, c))
        used.add(pred)
        state[r, c] = 1.0 - state[r, c]
        chosen_mask[r, c] = 1.0

    return chosen


def apply_flips(init, cells):
    g = init.copy()
    for r, c in cells:
        g[r, c] ^= 1
    return g


def evaluate(n_grids=50, seed=999, ckpt="best"):
    model = PointerPolicyNet().to(DEVICE)
    ckpt_path = os.path.join(CKPT_DIR, f"{TASK}_{ckpt}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"[eval] loaded {ckpt_path}")

    rng = np.random.default_rng(seed)
    grids = stratified_sample(n_grids, rng)

    gaps, outcome_matches, exact_matches = [], [], []
    rows = []
    for g in grids:
        candidates = candidate_neighborhood(g, margin=MARGIN)
        if len(candidates) > MAX_CANDIDATES:
            idx = rng.choice(len(candidates), MAX_CANDIDATES, replace=False)
            candidates = [candidates[j] for j in idx]
        teacher_chosen, teacher_scores = greedy_search(g, candidates, K=K, T=T)
        teacher_score = teacher_scores[-1]

        model_chosen = predict_perturbation(model, g, rng=rng)
        model_grid = apply_flips(g, model_chosen)
        model_score = float(unified_score_batch(model_grid[None], T)[0])

        gap = model_score - teacher_score
        gaps.append(gap)
        outcome_matches.append((teacher_score < T) == (model_score < T))
        exact_matches.append(set(model_chosen) == set(teacher_chosen))
        rows.append((g, teacher_chosen, teacher_score, model_chosen, model_score))

    gaps = np.array(gaps)
    print(f"[eval] n={n_grids}")
    print(f"  mean score gap (model - teacher): {gaps.mean():.2f}  (median {np.median(gaps):.2f})")
    print(f"  die-down-outcome match: {100*np.mean(outcome_matches):.1f}%")
    print(f"  exact flip-set match:   {100*np.mean(exact_matches):.1f}%")
    print(f"  model score <= teacher score: {100*np.mean(gaps <= 0):.1f}%")
    return rows


# ── Comparison GIFs ─────────────────────────────────────────────────────────

def make_comparison_gif(label, baseline_traj, teacher_traj, model_traj,
                          teacher_cells, model_cells, out_dir):
    steps = baseline_traj.shape[0] - 1
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#111111")
    titles = ["Baseline (no flips)", "Teacher (greedy search)", "Model (PointerPolicyNet)"]
    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_title(title, color="white", fontsize=11)

    im0 = axes[0].imshow(baseline_traj[0], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    im1 = axes[1].imshow(teacher_traj[0], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    im2 = axes[2].imshow(model_traj[0], cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
    ims = [im0, im1, im2]

    if teacher_cells:
        rs, cs = zip(*teacher_cells)
        axes[1].scatter(cs, rs, color="cyan", marker="x", s=100, zorder=5)
    if model_cells:
        rs, cs = zip(*model_cells)
        axes[2].scatter(cs, rs, color="lime", marker="x", s=100, zorder=5)

    suptitle = fig.suptitle(f"{label}   t=0", color="white", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    def _update(t):
        ims[0].set_data(baseline_traj[t])
        ims[1].set_data(teacher_traj[t])
        ims[2].set_data(model_traj[t])
        pops = (int(baseline_traj[t].sum()), int(teacher_traj[t].sum()), int(model_traj[t].sum()))
        suptitle.set_text(f"{label}   t={t}   pop(base,teacher,model)={pops}")
        return ims + [suptitle]

    ani = animation.FuncAnimation(fig, _update, frames=steps + 1, interval=200, blit=False)
    path = os.path.join(out_dir, f"{label}.gif")
    ani.save(path, writer=animation.PillowWriter(fps=5))
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def make_gifs(n=4, seed=7, gif_steps=60, ckpt="best"):
    out_dir = os.path.join(RESULTS_DIR, "comparison_gifs")
    os.makedirs(out_dir, exist_ok=True)

    model = PointerPolicyNet().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"{TASK}_{ckpt}.pt"), map_location=DEVICE))
    model.eval()

    rng = np.random.default_rng(seed)
    grids = stratified_sample(n, rng, dies_frac=0.0)  # only non-trivial cases are interesting to watch

    for i, g in enumerate(grids):
        candidates = candidate_neighborhood(g, margin=MARGIN)
        if len(candidates) > MAX_CANDIDATES:
            idx = rng.choice(len(candidates), MAX_CANDIDATES, replace=False)
            candidates = [candidates[j] for j in idx]
        teacher_chosen, _ = greedy_search(g, candidates, K=K, T=T)
        model_chosen = predict_perturbation(model, g, rng=rng)

        baseline_traj = run_trajectory(g, gif_steps)
        teacher_traj = run_trajectory(apply_flips(g, teacher_chosen), gif_steps)
        model_traj = run_trajectory(apply_flips(g, model_chosen), gif_steps)

        make_comparison_gif(f"case{i}", baseline_traj, teacher_traj, model_traj,
                              teacher_chosen, model_chosen, out_dir)


if __name__ == "__main__":
    evaluate(n_grids=50)
    make_gifs(n=4)
