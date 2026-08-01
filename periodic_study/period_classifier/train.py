"""
train.py — Train SimpleCNN, ResidualCNN, and EvolutionCNN on GoL period classification.

Usage:
    python train.py [--epochs 60] [--batch 64] [--lr 1e-3] [--n-frames 30]
                    [--device cuda] [--models simple residual evolution]

Outputs (in results/):
    {model}_best.pt          — best checkpoint (val accuracy)
    {model}_history.json     — per-epoch train/val loss & accuracy
    training_curves.png      — combined plot of all models
    confusion_matrix_{model}.png
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, HERE)
from dataset import (GoLPeriodDataset, load_all_records, make_split,
                     make_weighted_sampler, PERIODS, N_CLASSES, C2P, P2C)
from models import get_model

BG      = "#0f1117"
TEXT_C  = "white"
PALETTE = ["#3a7bd5", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22"]


# ── Training utilities ────────────────────────────────────────────────────────

def class_weights(train_records, device, power=0.5, max_ratio=20.0):
    """
    Softer inverse-frequency weights: w_c ∝ (1/count)^power, capped at max_ratio × min_weight.
    power=1.0 → full inverse, power=0.5 → square-root (gentler), power=0.0 → uniform.
    """
    counts = np.zeros(N_CLASSES)
    for _, p in train_records:
        counts[P2C[p]] += 1
    counts = np.where(counts == 0, 1, counts)
    w = (1.0 / counts) ** power
    w = np.clip(w, w.min(), w.min() * max_ratio)
    w = w / w.mean()               # normalise so mean weight = 1
    return torch.tensor(w, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0
    per_class_correct = np.zeros(N_CLASSES)
    per_class_total   = np.zeros(N_CLASSES)
    n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item() * len(y)
            for ci in range(N_CLASSES):
                mask = (y == ci)
                per_class_total[ci]   += mask.sum().item()
                per_class_correct[ci] += (preds[mask] == ci).sum().item()
            n += len(y)
    overall_acc = (per_class_correct.sum() / n) if n > 0 else 0.0
    # Macro accuracy: mean per-class accuracy (only over classes that appear in this split)
    seen = per_class_total > 0
    macro_acc = (per_class_correct[seen] / per_class_total[seen]).mean() if seen.any() else 0.0
    return total_loss / n, overall_acc, macro_acc


@torch.no_grad()
def confusion(model, loader, device):
    model.eval()
    mat = np.zeros((N_CLASSES, N_CLASSES), int)
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu().numpy()
        for gt, pr in zip(y.numpy(), preds):
            mat[gt, pr] += 1
    return mat


def save_confusion_plot(mat, name):
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    labels = [f"xp{C2P[i]}" for i in range(N_CLASSES)]
    ax.set_xticklabels(labels, rotation=45, ha="right", color=TEXT_C, fontsize=8)
    ax.set_yticklabels(labels, color=TEXT_C, fontsize=8)
    ax.set_xlabel("Predicted", color=TEXT_C)
    ax.set_ylabel("True", color=TEXT_C)
    ax.set_title(f"Confusion matrix — {name}", color=TEXT_C, fontweight="bold")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() * 0.5 else "black",
                        fontsize=7)
    fig.colorbar(im, ax=ax).ax.tick_params(colors=TEXT_C)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"confusion_{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved → confusion_{name}.png")


def save_curves_plot(histories):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT_C)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    for i, (name, hist) in enumerate(histories.items()):
        color = PALETTE[i % len(PALETTE)]
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"],  color=color, linestyle="--", alpha=0.5)
        axes[0].plot(epochs, hist["val_loss"],    color=color, label=name)
        axes[1].plot(epochs, hist["train_macro"], color=color, linestyle="--", alpha=0.5)
        axes[1].plot(epochs, hist["val_macro"],   color=color, label=name)

    axes[0].set_title("Loss (— val, -- train)", color=TEXT_C, fontweight="bold")
    axes[1].set_title("Macro Accuracy (— val, -- train)", color=TEXT_C, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Epoch", color=TEXT_C)
        ax.legend(facecolor="#1a1d27", labelcolor=TEXT_C, fontsize=9)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "training_curves.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → training_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int,   default=60)
    ap.add_argument("--batch",    type=int,   default=64)
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--n-frames", type=int,   default=30,
                    help="GoL steps to simulate for EvolutionCNN")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models",   nargs="+",
                    default=["simple", "residual", "evolution"],
                    choices=["simple", "residual", "evolution"])
    args = ap.parse_args()
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load & split data ────────────────────────────────────────────────────
    print("Loading census data …")
    records, counts = load_all_records()
    print(f"  Total: {len(records):,} oscillators across {N_CLASSES} period classes")
    for p in PERIODS:
        if counts[p]:
            print(f"    xp{p:2d}: {counts[p]:5,}")

    train_recs, val_recs, test_recs = make_split(records)
    print(f"  Split → train={len(train_recs):,}  val={len(val_recs):,}  test={len(test_recs):,}")

    # ── Dataset / loader factory ─────────────────────────────────────────────
    def make_loaders(n_frames):
        train_ds = GoLPeriodDataset(train_recs, augment=True,  n_frames=n_frames)
        val_ds   = GoLPeriodDataset(val_recs,   augment=False, n_frames=n_frames)
        test_ds  = GoLPeriodDataset(test_recs,  augment=False, n_frames=n_frames)
        # Natural sampling — class imbalance handled solely via weighted loss
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
        val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)
        test_dl  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)
        return train_dl, val_dl, test_dl

    cw = class_weights(train_recs, device)
    all_histories = {}

    # ── Train each model ──────────────────────────────────────────────────────
    for model_name in args.models:
        print(f"\n{'='*60}")
        n_frames = args.n_frames if model_name == "evolution" else 0
        print(f"Training: {model_name.upper()}  (n_frames={n_frames})")
        print(f"{'='*60}")

        model = get_model(model_name, n_frames=n_frames).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_dl, val_dl, test_dl = make_loaders(n_frames)

        history = {"train_loss": [], "val_loss": [],
                   "train_acc": [], "val_acc": [],
                   "train_macro": [], "val_macro": []}
        best_val_macro = 0.0
        ckpt_path      = os.path.join(OUT_DIR, f"{model_name}_best.pt")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_mac = run_epoch(model, train_dl, criterion, optimizer, device, train=True)
            va_loss, va_acc, va_mac = run_epoch(model, val_dl,   criterion, optimizer, device, train=False)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["train_macro"].append(tr_mac)
            history["val_macro"].append(va_mac)

            elapsed = time.time() - t0
            print(f"  [{epoch:3d}/{args.epochs}] "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.3f} macro={tr_mac:.3f} | "
                  f"val loss={va_loss:.4f} acc={va_acc:.3f} macro={va_mac:.3f} | "
                  f"{elapsed:.1f}s")

            if va_mac > best_val_macro:
                best_val_macro = va_mac
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_macro": float(va_mac), "model": model_name,
                            "n_frames": int(n_frames)}, ckpt_path)

        # ── Test set evaluation ──────────────────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        te_loss, te_acc, te_mac = run_epoch(model, test_dl, criterion, optimizer, device, train=False)
        print(f"\n  Best val macro: {best_val_macro:.3f}  |  Test macro: {te_mac:.3f}  overall acc: {te_acc:.3f}")

        mat = confusion(model, test_dl, device)
        save_confusion_plot(mat, model_name)

        # Per-class accuracy
        print("  Per-class test accuracy:")
        for ci in range(N_CLASSES):
            row_sum = mat[ci].sum()
            if row_sum > 0:
                print(f"    xp{C2P[ci]:2d}: {mat[ci, ci]/row_sum:.2%}  ({row_sum} samples)")

        history["best_val_macro"] = best_val_macro
        history["test_macro"]     = te_mac
        history["test_acc"]       = te_acc
        history["n_params"]       = n_params
        all_histories[model_name] = history

        hist_path = os.path.join(OUT_DIR, f"{model_name}_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary:")
    for name, h in all_histories.items():
        print(f"  {name:12s}  best_val_macro={h['best_val_macro']:.3f}  "
              f"test_macro={h['test_macro']:.3f}  overall_acc={h['test_acc']:.3f}  "
              f"params={h['n_params']:,}")

    save_curves_plot(all_histories)
    print("\nDone.")


if __name__ == "__main__":
    main()
