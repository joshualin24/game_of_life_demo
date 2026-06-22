"""
Task 7: Trajectory Transformer Embedding
-----------------------------------------
Trains two TrajectoryTransformer models (d_model=64 and d_model=128) as
reconstruction autoencoders on GoL trajectories.

Dataset  : 5000 trajectories on 40×40 toroidal grids, T=60 steps.
           40% random initial states, 60% named-pattern grids (with rotation,
           reflection, and 1–3 pattern combinations).
Objective: BCE reconstruction loss on all T+1 frames (per-frame outputs of
           the transformer, not the CLS token).  The CLS token learns to
           aggregate trajectory information through self-attention.
Outputs  :
  nn/checkpoints/traj_emb_d64_best.pt
  nn/checkpoints/traj_emb_d128_best.pt
  nn/results/traj_emb_d64_loss.png
  nn/results/traj_emb_d128_loss.png
"""

import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nn.trajectory_data import make_trajectory_loaders
from nn.models           import TrajectoryTransformer
from nn.utils            import set_seed, CKPT_DIR, RESULTS_DIR, DEVICE

# ── Hyperparameters ────────────────────────────────────────────────────────────

GRID_SIZE  = 40
T          = 60
N_SAMPLES  = 5000
EPOCHS     = 80
BATCH_SIZE = 16
LR         = 3e-4
SEED       = 42
D_MODELS   = [64, 128]
# Alive cells are ~8% of all (frame, cell) pairs — weight live-cell errors 12× higher
# so the model cannot cheat by predicting everything dead.
POS_WEIGHT = 12.0


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one(d_model: int, train_dl, val_dl) -> dict:
    task = f"traj_emb_d{d_model}"
    model = TrajectoryTransformer(
        d_model=d_model, nhead=4, num_layers=4,
        grid_size=GRID_SIZE, T=T, dropout=0.1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  TrajectoryTransformer  d_model={d_model}  params={n_params:,}")
    print(f"  device={DEVICE}  epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    pw        = torch.tensor([POS_WEIGHT], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    ckpt_path = os.path.join(CKPT_DIR, f"{task}_best.pt")
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": []}
    t0        = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_losses = []
        for batch in train_dl:
            x = batch.to(DEVICE)                        # (B, T+1, H, W)
            _, recon = model(x)
            loss = criterion(recon, x)
            train_losses.append(loss.item())   # before backward to avoid MPS timing issues
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── val ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                x = batch.to(DEVICE)
                _, recon = model(x)
                val_losses.append(criterion(recon, x).item())

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step()

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:>4}/{EPOCHS}  "
                  f"train={tl:.5f}  val={vl:.5f}  "
                  f"best={best_val:.5f}  t={elapsed:.0f}s")

    print(f"  Best val loss: {best_val:.5f}  →  {ckpt_path}")

    # save history
    hist_path = os.path.join(RESULTS_DIR, f"{task}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train", color="#4C72B0")
    ax.plot(history["val_loss"],   label="val",   color="#C44E52")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(f"TrajectoryTransformer d_model={d_model} — training curves",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, f"{task}_loss.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve  →  {curve_path}")

    return history


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    print(f"[data] Building dataset  n={N_SAMPLES}  grid={GRID_SIZE}  T={T} …")
    train_dl, val_dl, _ = make_trajectory_loaders(
        n_samples=N_SAMPLES,
        grid_size=GRID_SIZE,
        T=T,
        random_frac=0.4,
        val_frac=0.15,
        batch_size=BATCH_SIZE,
        seed=SEED,
        num_workers=0,        # set >0 if your system supports multiprocess dataloading
    )
    print(f"  train batches: {len(train_dl)}  val batches: {len(val_dl)}")

    for d_model in D_MODELS:
        train_one(d_model, train_dl, val_dl)

    print("\nAll models trained.")


if __name__ == "__main__":
    main()
