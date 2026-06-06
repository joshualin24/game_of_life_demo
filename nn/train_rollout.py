"""
Task 5: Multi-step rollout predictor
--------------------------------------
Input : initial grid + normalised step k/K     (2, H, W)
Output: grid k steps later                     (1, H, W)
Model : RolloutPredictor  (residual conv tower conditioned on k)
Goal  : Does a direct k-step predictor outperform autoregressive unrolling?
        How fast does error accumulate with k?
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from nn.data_gen import generate_trajectory_dataset, load_dataset
from nn.models   import RolloutPredictor
from nn.utils    import (Trainer, set_seed, plot_loss_curves, DEVICE, CKPT_DIR)

TASK        = "task5_rollout"
N_INITS     = 300
TRAJ_STEPS  = 100
GRID_SIZE   = 32
MAX_K       = 20     # predict up to K steps ahead
EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 3e-4
SEED        = 42


class _RolloutLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class _RolloutTrainer(Trainer):
    """Override _step for the (grid, k_norm, target) batch signature."""
    def _step(self, batch, train):
        grid, k_norm, y = [b.to(DEVICE) for b in batch]
        if train:
            self.model.train(); self.optimizer.zero_grad()
        else:
            self.model.eval()
        with torch.set_grad_enabled(train):
            pred = self.model(grid, k_norm)
            loss = self.loss_fn(pred, y)
        if train:
            loss.backward(); self.optimizer.step()
        return loss.item()


def main():
    set_seed(SEED)

    data_path = os.path.join(os.path.dirname(__file__), "data", "trajectories.npz")
    if os.path.exists(data_path):
        data = load_dataset("trajectories")
    else:
        data = generate_trajectory_dataset(
            n_inits=N_INITS, grid_size=GRID_SIZE, traj_steps=TRAJ_STEPS)

    trajs = data["trajs"]  # (n_inits, steps+1, H, W)
    n_inits, T, H, W = trajs.shape

    # Build (init, k, target) pairs for k in 1..MAX_K
    inits_list, ks_list, targets_list = [], [], []
    for i in range(n_inits):
        for t in range(T - MAX_K):
            for k in range(1, MAX_K + 1):
                inits_list.append(trajs[i, t])
                ks_list.append(k / MAX_K)
                targets_list.append(trajs[i, t + k])

    grids_arr   = np.array(inits_list,   dtype=np.float32)
    ks_arr      = np.array(ks_list,      dtype=np.float32)
    targets_arr = np.array(targets_list, dtype=np.float32)

    X_grid = torch.tensor(grids_arr[:, None])    # (N,1,H,W)
    X_k    = torch.tensor(ks_arr)                # (N,)
    Y      = torch.tensor(targets_arr[:, None])  # (N,1,H,W)

    dataset = TensorDataset(X_grid, X_k, Y)
    n_val   = int(len(dataset) * 0.15)
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          pin_memory=True, num_workers=2)

    print(f"  Pairs: {len(dataset):,}  grid={H}×{W}  max_k={MAX_K}")

    model     = RolloutPredictor(channels=64, n_res=6)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: RolloutPredictor  params={n_params:,}  device={DEVICE}")

    trainer = _RolloutTrainer(model, optimizer, _RolloutLoss(), TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=10)
    plot_loss_curves(history, TASK)
    print("Done.")


if __name__ == "__main__":
    main()
