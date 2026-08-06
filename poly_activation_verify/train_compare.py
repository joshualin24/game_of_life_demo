"""
Verify arXiv:2606.23587's central claim on our own 40x40 toroidal Game of
Life setup: does a minimal CNN with ReLU activation fail to learn the
transition rule while the same architecture with a 2nd-degree polynomial
activation succeeds?

Both models share the exact L(1, m) architecture (model.py) and differ
ONLY in the activation function. Trained single-step, teacher-forcing
(input is always the true previous ground-truth state -- no rollout).

Runs N_SEEDS independent training runs per activation and reports success
rate + convergence speed, mirroring the paper's own multi-seed methodology.
Uses a small, fixed dataset ("small amount of data first" per instructions);
easy to scale up N_TRAIN/N_VAL later if the effect isn't clear enough.
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gol_sim import step as gol_step, random_grid
from model   import make_relu_net, make_poly_net

DEVICE = torch.device("cpu")   # model is tiny; CPU avoids MPS launch overhead

GRID_SIZE   = 40
M_WIDTH     = 1        # paper's minimal L(1,1): 25 params (ReLU) / 34 params (Poly)
N_TRAIN     = 100
N_VAL       = 20
DENSITY_LO, DENSITY_HI = 0.1, 0.9
BATCH_SIZE  = 8
LR          = 1e-3
MAX_EPOCHS  = 500
N_SEEDS     = 10
PATIENCE_PERFECT = 2     # stop after this many consecutive perfect val epochs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_dataset(seed, n_grids):
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for _ in range(n_grids):
        d = rng.uniform(DENSITY_LO, DENSITY_HI)
        g = random_grid(rng, GRID_SIZE, d)
        Xs.append(g)
        Ys.append(gol_step(g))
    X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(1)   # (N,1,H,W)
    Y = torch.tensor(np.array(Ys), dtype=torch.float32).unsqueeze(1)
    return X, Y


def cell_accuracy(logits, y):
    pred = (torch.sigmoid(logits) > 0.5).float()
    return (pred == y).float().mean().item()


def train_one(model_factory, seed, X_tr, Y_tr, X_val, Y_val):
    torch.manual_seed(seed)
    model = model_factory().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_tr.shape[0]
    perfect_streak = 0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_losses = []
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_tr[idx].to(DEVICE), Y_tr[idx].to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(DEVICE))
            val_acc = cell_accuracy(val_logits, Y_val.to(DEVICE))

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_acc"].append(val_acc)

        if val_acc >= 0.9999:
            perfect_streak += 1
            if perfect_streak >= PATIENCE_PERFECT:
                return dict(success=True, epochs=epoch, final_acc=val_acc,
                            n_params=n_params, history=history)
        else:
            perfect_streak = 0

    return dict(success=False, epochs=MAX_EPOCHS, final_acc=history["val_acc"][-1],
                n_params=n_params, history=history)


def main():
    print(f"Generating dataset: {N_TRAIN} train + {N_VAL} val grids "
          f"({GRID_SIZE}x{GRID_SIZE}, density {DENSITY_LO}-{DENSITY_HI}) ...")
    X_tr, Y_tr   = make_dataset(seed=1000, n_grids=N_TRAIN)
    X_val, Y_val = make_dataset(seed=2000, n_grids=N_VAL)

    variants = {
        "ReLU": make_relu_net,
        "Poly": make_poly_net,
    }

    results = {name: [] for name in variants}

    for name, factory in variants.items():
        print(f"\n=== {name} (m={M_WIDTH}) ===")
        for seed in range(N_SEEDS):
            t0 = time.time()
            r = train_one(lambda: factory(M_WIDTH), seed, X_tr, Y_tr, X_val, Y_val)
            dt = time.time() - t0
            results[name].append(r)
            status = "OK  " if r["success"] else "FAIL"
            print(f"  seed {seed:2d}  [{status}]  epochs={r['epochs']:>4}  "
                  f"val_acc={r['final_acc']*100:6.2f}%  params={r['n_params']}  t={dt:.1f}s",
                  flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({N_SEEDS} seeds each, m={M_WIDTH})")
    print(f"{'='*70}")
    print(f"  {'Activation':<10} {'Success rate':<14} {'Params':<8} "
          f"{'Mean epochs (success)':<22} {'Mean final acc':<15}")
    for name, runs in results.items():
        n_success = sum(r["success"] for r in runs)
        n_params  = runs[0]["n_params"]
        succ_epochs = [r["epochs"] for r in runs if r["success"]]
        mean_epochs = np.mean(succ_epochs) if succ_epochs else float("nan")
        mean_acc = np.mean([r["final_acc"] for r in runs])
        print(f"  {name:<10} {n_success}/{N_SEEDS:<12} {n_params:<8} "
              f"{mean_epochs:<22.1f} {mean_acc*100:<14.2f}%")

    # ── Save raw results ─────────────────────────────────────────────────────
    serializable = {
        name: [{k: v for k, v in r.items() if k != "history"} for r in runs]
        for name, runs in results.items()
    }
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
        json.dump({name: [r["history"] for r in runs] for name, runs in results.items()},
                   f)

    # ── Plot: val accuracy curves for every seed ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"ReLU": "#C44E52", "Poly": "#4C72B0"}
    for ax, (name, runs) in zip(axes, results.items()):
        for r in runs:
            ax.plot([a * 100 for a in r["history"]["val_acc"]],
                    color=colors[name], alpha=0.5, linewidth=1)
        ax.set_title(f"{name}  ({sum(r['success'] for r in runs)}/{N_SEEDS} converged)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val cell accuracy (%)")
        ax.grid(alpha=0.3)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    fig.suptitle(f"ReLU vs Polynomial activation -- minimal L(1,{M_WIDTH}) CNN "
                 f"on {GRID_SIZE}x{GRID_SIZE} toroidal GoL, single-step teacher forcing",
                 fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "relu_vs_poly_convergence.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
