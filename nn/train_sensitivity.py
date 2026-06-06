"""
Task 2: Sensitivity map prediction
-----------------------------------
Input : initial grid (1, H, W)
Output: cumulative-divergence heatmap (1, H, W)
Model : SensitivityUNet
Loss  : Huber (robust to large outliers in divergence scores)
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from nn.data_gen  import generate_sensitivity_dataset, load_dataset
from nn.models    import SensitivityUNet
from nn.utils     import (Trainer, make_loaders, set_seed,
                          plot_loss_curves, plot_sensitivity_predictions,
                          DEVICE, RESULTS_DIR)

# ── Config ─────────────────────────────────────────────────────────────────────
TASK        = "task2_sensitivity"
N_SAMPLES   = 2000
GRID_SIZE   = 32
STEPS       = 50
DENSITY     = 0.12   # sparse: gives structured sensitivity maps with clear hot spots
EPOCHS      = 120
BATCH_SIZE  = 32
LR          = 3e-4
SEED        = 42


def main():
    set_seed(SEED)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__), "data", "sensitivity_sparse.npz")
    if os.path.exists(data_path):
        print("[task2] Loading cached sparse sensitivity dataset …")
        data = np.load(data_path)
        data = {k: data[k] for k in data.files}
    else:
        data = generate_sensitivity_dataset(
            n_samples=N_SAMPLES, grid_size=GRID_SIZE, steps=STEPS,
            density=DENSITY, seed=SEED, save=False)
        np.savez_compressed(data_path, **data)
        print(f"  Saved → {data_path}")

    grids = data["grids"].astype(np.float32)  # (N, H, W)
    maps  = data["maps"]                       # (N, H, W) float32

    # per-sample log-normalise to [0,1] — preserves within-map structure
    maps_log  = np.log1p(maps)
    per_max   = maps_log.max(axis=(1, 2), keepdims=True).clip(min=1e-6)
    maps_norm = (maps_log / per_max).astype(np.float32)
    scale     = per_max.mean()
    print(f"  grids: {grids.shape}  maps: {maps.shape}  "
          f"avg_scale: {scale:.2f}  map_std: {maps_norm.std():.3f}")

    # tensors: add channel dim
    X = torch.tensor(grids[:, None])           # (N,1,H,W)
    Y = torch.tensor(maps_norm[:, None])        # (N,1,H,W)

    train_dl, val_dl = make_loaders(X, Y, batch_size=BATCH_SIZE, seed=SEED)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = SensitivityUNet(base_ch=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)
    loss_fn   = nn.HuberLoss(delta=0.1)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: SensitivityUNet  params={n_params:,}  device={DEVICE}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, optimizer, loss_fn, TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=10)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    trainer.load_best()
    plot_loss_curves(history, TASK)
    plot_sensitivity_predictions(model, grids, maps_norm, TASK, n_show=4)

    # pixel-level Pearson correlation on validation set
    model.eval()
    preds_all, trues_all = [], []
    with torch.no_grad():
        for x_b, y_b in val_dl:
            pred = model(x_b.to(DEVICE)).squeeze(1).cpu().numpy()
            preds_all.append(pred)
            trues_all.append(y_b.squeeze(1).numpy())
    preds_all = np.concatenate(preds_all).ravel()
    trues_all = np.concatenate(trues_all).ravel()
    corr = np.corrcoef(preds_all, trues_all)[0, 1]
    print(f"\n  Pixel-level Pearson r = {corr:.4f}")

    print(f"  Done.  Results in nn/results/  checkpoints in nn/checkpoints/")


if __name__ == "__main__":
    main()
