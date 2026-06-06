"""
Task 3: Chaos / perturbation-impact predictor
----------------------------------------------
Input : 2-channel tensor  (grid, flip-location mask)  (2, H, W)
Output: scalar cumulative divergence score
Model : ChaosPredictor  (conv feature extractor + MLP head)
Goal  : Predict how much a single-cell flip will diverge without simulating.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from nn.data_gen import generate_chaos_dataset, load_dataset
from nn.models   import ChaosPredictor
from nn.utils    import (Trainer, make_loaders, set_seed,
                         plot_loss_curves, DEVICE, RESULTS_DIR)

TASK        = "task3_chaos"
N_SAMPLES   = 5000
GRID_SIZE   = 32
STEPS       = 50
EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 3e-4
SEED        = 42


def main():
    set_seed(SEED)

    data_path = os.path.join(os.path.dirname(__file__), "data", "chaos.npz")
    if os.path.exists(data_path):
        data = load_dataset("chaos")
    else:
        data = generate_chaos_dataset(
            n_samples=N_SAMPLES, grid_size=GRID_SIZE, steps=STEPS)

    grids  = data["grids"].astype(np.float32)
    masks  = data["masks"].astype(np.float32)
    scores = data["scores"].astype(np.float32)

    # log-normalise scores
    scores_norm = np.log1p(scores) / np.log1p(scores).max()

    # 2-channel input
    X = torch.tensor(np.stack([grids, masks], axis=1))   # (N,2,H,W)
    Y = torch.tensor(scores_norm[:, None])                # (N,1)

    train_dl, val_dl = make_loaders(X, Y, batch_size=BATCH_SIZE, seed=SEED)

    model     = ChaosPredictor(base_ch=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)
    loss_fn   = nn.HuberLoss(delta=0.1)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ChaosPredictor  params={n_params:,}  device={DEVICE}")

    trainer = Trainer(model, optimizer, loss_fn, TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=10)
    plot_loss_curves(history, TASK)
    print("Done.")


if __name__ == "__main__":
    main()
