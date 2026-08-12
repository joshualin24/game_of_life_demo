"""
Die-down predictor v1: PointerPolicyNet trained to imitate
teacher_search.greedy_search via per-decode-step teacher-forced
cross-entropy (see generate_dataset.build_step_tensors for the
(X, valid_mask, y) tensors this trains on).

Mirrors nn/train_cnn_transformer_v10.py's conventions (hyperparams as
module-level constants, hand-rolled train/val loop, AdamW + cosine LR,
periodic + best/final checkpoints, history JSON + loss-curve PNG) — just
writing outputs under diedown_predictor/ instead of nn/.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from nn.utils import DEVICE, set_seed
from diedown_predictor.generate_dataset import DATA_DIR, GRID_SIZE, augment_d4, build_step_tensors
from diedown_predictor.models import PointerPolicyNet

# ── Hyperparameters ────────────────────────────────────────────────────────────

TASK       = "diedown_v3"
DATASET    = "full_v3.npz"
BASE_CH    = 32
EPOCHS     = 30
BATCH_SIZE = 128
LR         = 1e-3
VAL_FRAC   = 0.15
SEED       = 42
CKPT_EVERY = 5
AUGMENT_D4 = True

CKPT_DIR    = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def _to_loader(X, mask, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(y, dtype=torch.int64),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_loaders(X, mask, y, grid_idx, val_frac=VAL_FRAC, batch_size=BATCH_SIZE, seed=SEED,
                  augment=AUGMENT_D4):
    """
    Splits by *base grid* (not by individual decode-step): steps from the
    same base grid share most of the board and are highly correlated, so
    splitting at the step level would leak train information into val.
    D4 augmentation (see generate_dataset.augment_d4) is applied separately
    to each split afterward, so it never crosses the split boundary either.
    """
    rng = np.random.default_rng(seed)
    unique_grids = np.unique(grid_idx)
    rng.shuffle(unique_grids)
    n_val_grids = max(1, int(len(unique_grids) * val_frac))
    val_grids = set(unique_grids[:n_val_grids].tolist())

    val_mask = np.array([g in val_grids for g in grid_idx])
    train_mask = ~val_mask

    X_tr, mask_tr, y_tr = X[train_mask], mask[train_mask], y[train_mask]
    X_va, mask_va, y_va = X[val_mask], mask[val_mask], y[val_mask]

    if augment:
        X_tr, mask_tr, y_tr = augment_d4(X_tr, mask_tr, y_tr, GRID_SIZE)
        X_va, mask_va, y_va = augment_d4(X_va, mask_va, y_va, GRID_SIZE)

    train_dl = _to_loader(X_tr, mask_tr, y_tr, batch_size, shuffle=True)
    val_dl = _to_loader(X_va, mask_va, y_va, batch_size, shuffle=False)
    return train_dl, val_dl


def run_epoch(model, loader, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            logits = model(x, valid_mask=mask)
            loss = F.cross_entropy(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += x.size(0)

    return total_loss / total_n, total_correct / total_n


def main():
    set_seed(SEED)

    npz_path = os.path.join(DATA_DIR, DATASET)
    print(f"[data] expanding step-tensors from {npz_path} …")
    X, mask, y, grid_idx = build_step_tensors(npz_path)
    print(f"[data] {X.shape[0]} decode-step examples from {len(np.unique(grid_idx))} base grids, "
          f"grid {X.shape[-2]}x{X.shape[-1]}")

    train_loader, val_loader = make_loaders(X, mask, y, grid_idx)
    print(f"[data] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}"
          f"  (augment_d4={AUGMENT_D4})")

    model = PointerPolicyNet(base_ch=BASE_CH).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] PointerPolicyNet base_ch={BASE_CH}  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_path = os.path.join(CKPT_DIR, f"{TASK}_best.pt")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, optimizer)
        vl, va = run_epoch(model, val_loader)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)

        if epoch % CKPT_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{TASK}_ep{epoch:03d}.pt"))

        elapsed = time.time() - t0
        print(f"  epoch {epoch:>3}/{EPOCHS}  train_loss={tl:.4f} acc={ta:.3f}  "
              f"val_loss={vl:.4f} acc={va:.3f}  best_val={best_val:.4f}  t={elapsed:.0f}s")

    final_path = os.path.join(CKPT_DIR, f"{TASK}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"[done] best_val={best_val:.4f} -> {best_path}  |  final -> {final_path}")

    hist_path = os.path.join(RESULTS_DIR, f"{TASK}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(history["val_loss"], label="val", color="#C44E52")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_acc"], label="train", color="#4C72B0")
    axes[1].plot(history["val_acc"], label="val", color="#C44E52")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Step accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{TASK} — training curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)
    print(f"[done] history -> {hist_path}")


if __name__ == "__main__":
    main()
