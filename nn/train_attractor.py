"""
Task 6: Attractor / fate classifier
--------------------------------------
Input : initial grid                           (1, H, W)
Output: fate class logits                      (4,)
Classes: 0=dies  1=still-life  2=oscillator(p2-15)  3=active/complex
Model : FateClassifier  (conv encoder + MLP head)
Goal  : Can we predict the long-term fate from the initial condition alone?
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from nn.data_gen import generate_attractor_dataset, load_dataset
from nn.models   import FateClassifier
from nn.utils    import (Trainer, make_loaders, set_seed,
                         plot_loss_curves, DEVICE, RESULTS_DIR)

TASK        = "task6_attractor"
N_SAMPLES   = 2000
GRID_SIZE   = 32
FATE_STEPS  = 200
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 3e-4
SEED        = 42
LABEL_NAMES = ["dies", "still_life", "oscillator", "active"]


def plot_confusion(model, val_dl, task_name):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x_b, y_b in val_dl:
            logits = model(x_b.to(DEVICE))
            preds  = logits.argmax(1).cpu().numpy()
            all_pred.extend(preds)
            all_true.extend(y_b.numpy())

    cm  = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(LABEL_NAMES)
    plt.colorbar(im, ax=ax)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_title(f"{task_name} — confusion matrix", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{task_name}_confusion.png")
    fig.savefig(path, dpi=150)
    print(f"  Confusion matrix → {path}")
    plt.close(fig)
    print(classification_report(all_true, all_pred, target_names=LABEL_NAMES))


def main():
    set_seed(SEED)

    data_path = os.path.join(os.path.dirname(__file__), "data", "attractor.npz")
    if os.path.exists(data_path):
        data = load_dataset("attractor")
    else:
        data = generate_attractor_dataset(
            n_samples=N_SAMPLES, grid_size=GRID_SIZE, steps=FATE_STEPS)

    grids  = data["grids"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    counts = np.bincount(labels, minlength=4)
    print(f"  Label distribution: { {n:c for n,c in zip(LABEL_NAMES, counts)} }")

    X = torch.tensor(grids[:, None])  # (N,1,H,W)
    Y = torch.tensor(labels)          # (N,)

    train_dl, val_dl = make_loaders(X, Y, batch_size=BATCH_SIZE, seed=SEED)

    # class-weighted loss to handle imbalance
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    weights = weights / weights.sum() * len(LABEL_NAMES)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

    model     = FateClassifier(base_ch=32, n_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: FateClassifier  params={n_params:,}  device={DEVICE}")

    class _FateLoss(nn.Module):
        def forward(self, pred, y):
            return loss_fn(pred, y.squeeze().long())

    trainer = Trainer(model, optimizer, _FateLoss(), TASK, scheduler)
    history = trainer.fit(train_dl, val_dl, epochs=EPOCHS, print_every=10)

    trainer.load_best()
    plot_loss_curves(history, TASK)
    plot_confusion(model, val_dl, TASK)
    print("Done.")


if __name__ == "__main__":
    main()
