"""Shared training utilities: Trainer, metrics, result plots."""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
CKPT_DIR     = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Generic Trainer ────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model:      nn.Module,
        optimizer:  torch.optim.Optimizer,
        loss_fn,
        task_name:  str,
        scheduler=None,
    ):
        self.model      = model.to(DEVICE)
        self.optimizer  = optimizer
        self.loss_fn    = loss_fn
        self.scheduler  = scheduler
        self.task_name  = task_name
        self.history    = {"train_loss": [], "val_loss": []}

    def _step(self, batch, train: bool):
        if train:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        # batch can be (x, y) or (x1, x2, y) for multi-input models
        *inputs, y = [b.to(DEVICE) for b in batch]

        with torch.set_grad_enabled(train):
            pred = self.model(*inputs)
            loss = self.loss_fn(pred, y)

        if train:
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        print_every:  int = 5,
    ):
        best_val = float("inf")
        ckpt_path = os.path.join(CKPT_DIR, f"{self.task_name}_best.pt")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            train_losses = [self._step(b, train=True)  for b in train_loader]
            val_losses   = [self._step(b, train=False) for b in val_loader]

            tl = np.mean(train_losses)
            vl = np.mean(val_losses)
            self.history["train_loss"].append(tl)
            self.history["val_loss"].append(vl)

            if self.scheduler is not None:
                self.scheduler.step(vl)

            if vl < best_val:
                best_val = vl
                torch.save(self.model.state_dict(), ckpt_path)

            if epoch % print_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(f"  [{self.task_name}] epoch {epoch:>4}/{epochs}  "
                      f"train={tl:.5f}  val={vl:.5f}  "
                      f"best={best_val:.5f}  t={elapsed:.0f}s")

        print(f"  Best val loss: {best_val:.5f}  →  {ckpt_path}")
        # save history
        hist_path = os.path.join(RESULTS_DIR, f"{self.task_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        return self.history

    def load_best(self):
        path = os.path.join(CKPT_DIR, f"{self.task_name}_best.pt")
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        self.model.eval()


# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_loaders(
    *tensors,
    val_frac: float = 0.15,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    dataset = TensorDataset(*tensors)
    n_val   = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          pin_memory=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          pin_memory=True, num_workers=2)
    return train_dl, val_dl


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, task_name: str, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="train", color="#4C72B0")
    ax.plot(history["val_loss"],   label="val",   color="#C44E52")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{task_name} — training curves", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, f"{task_name}_loss.png")
        fig.savefig(path, dpi=150)
        print(f"  Loss curve → {path}")
    plt.close(fig)


def plot_sensitivity_predictions(
    model: nn.Module,
    grids: np.ndarray,
    maps:  np.ndarray,
    task_name: str,
    n_show: int = 4,
):
    """Side-by-side: input grid | true sensitivity | predicted sensitivity."""
    model.eval()
    indices = np.random.choice(len(grids), n_show, replace=False)
    fig, axes = plt.subplots(n_show, 3, figsize=(10, n_show * 3))

    for row, idx in enumerate(indices):
        g = torch.tensor(grids[idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(g).squeeze().cpu().numpy()
        true = maps[idx]

        axes[row, 0].imshow(grids[idx], cmap="inferno", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[row, 0].set_title("Input grid", fontsize=9)
        vmax = max(true.max(), pred.max(), 1)
        axes[row, 1].imshow(true, cmap="hot", vmin=0, vmax=vmax,
                            interpolation="nearest")
        axes[row, 1].set_title(f"True  (max={true.max():.0f})", fontsize=9)
        axes[row, 2].imshow(pred, cmap="hot", vmin=0, vmax=vmax,
                            interpolation="nearest")
        axes[row, 2].set_title(f"Pred  (max={pred.max():.0f})", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle(f"{task_name} — predictions vs ground truth", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{task_name}_predictions.png")
    fig.savefig(path, dpi=150)
    print(f"  Prediction grid → {path}")
    plt.close(fig)


def plot_neural_ca_rollout(
    model: nn.Module,
    init:  np.ndarray,
    steps: int,
    task_name: str,
    n_show: int = 8,
):
    """Compare true GoL rollout vs neural CA rollout side by side."""
    from nn.data_gen import run_trajectory
    true_traj = run_trajectory(init, steps)

    model.eval()
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    pred_frames = [init.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = (model(curr) > 0.5).float()
            pred_frames.append(curr.squeeze().cpu().numpy().astype(np.uint8))

    show_at = np.linspace(0, steps, n_show, dtype=int)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.2, 5))
    for col, t in enumerate(show_at):
        axes[0, col].imshow(true_traj[t], cmap="inferno", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[0, col].set_title(f"t={t}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].imshow(pred_frames[t], cmap="inferno", vmin=0, vmax=1,
                            interpolation="nearest")
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("True GoL",   fontsize=9)
    axes[1, 0].set_ylabel("Neural CA",  fontsize=9)
    fig.suptitle(f"{task_name} — true vs learned rollout", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{task_name}_rollout.png")
    fig.savefig(path, dpi=150)
    print(f"  Rollout comparison → {path}")
    plt.close(fig)
