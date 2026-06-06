"""
Task 1: Next-state prediction
------------------------------
Input : grid at time t         (1, H, W)
Output: grid at time t+1       (1, H, W)
Model : NextStatePredictor  (residual conv)
Goal  : Can a deeper net perfectly learn GoL?  Compare to NeuralCA.
        Test generalisation to rule variants (random rule, B3/S23 variants).
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from nn.data_gen import generate_trajectory_dataset, load_dataset
from nn.models   import NextStatePredictor
from nn.utils    import (Trainer, make_loaders, set_seed,
                         plot_loss_curves, DEVICE)

TASK        = "task1_next_state"
N_INITS     = 300
TRAJ_STEPS  = 100
GRID_SIZE   = 32
EPOCHS      = 40
BATCH_SIZE  = 128
LR          = 1e-3
SEED        = 42


def main():
    set_seed(SEED)

    data_path = os.path.join(os.path.dirname(__file__), "data", "trajectories.npz")
    if os.path.exists(data_path):
        data = load_dataset("trajectories")
    else:
        data = generate_trajectory_dataset(
            n_inits=N_INITS, grid_size=GRID_SIZE, traj_steps=TRAJ_STEPS)

    states = data["states"].astype(np.float32)
    nexts  = data["nexts"].astype(np.float32)

    X = torch.tensor(states[:, None])
    Y = torch.tensor(nexts[:,  None])
    train_dl, val_dl = make_loaders(X, Y, batch_size=BATCH_SIZE, seed=SEED)

    model     = NextStatePredictor(channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = nn.BCELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: NextStatePredictor  params={n_params:,}  device={DEVICE}")

    trainer = Trainer(model, optimizer, loss_fn, TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=5)
    plot_loss_curves(history, TASK)
    print("Done.")


if __name__ == "__main__":
    main()
