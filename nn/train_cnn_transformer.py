"""
Task 9: CNN-Transformer Hybrid next-step predictor
----------------------------------------------------
Trains a CNNTransformer to predict the GoL state at t+1 given t.

Architecture:
  Stage 1  CNN local encoder (RF=5×5) → (B, d_model, H, W) feature map
  Stage 2  Spatial avg-pool into patches → (B, 100, d_model) tokens
  Stage 3  4-layer pre-norm transformer encoder → per-patch head → (B, 1, H, W)

Dataset  : 2000 × 100-step trajectories → 200K (state_t, state_t+1) pairs.
           D4 augmentation applied per batch (GoL is D4-equivariant) → 8× effective data.
Objective: BCE loss on next-state prediction.
Outputs  :
  nn/checkpoints/task9_cnn_transformer_best.pt
  nn/results/task9_cnn_transformer_loss.png
  nn/results/task9_cnn_transformer_rollout_random.png
  nn/results/task9_cnn_transformer_rollout_glider.png
"""

import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nn.models   import CNNTransformer
from nn.data_gen import run_trajectory
from nn.utils    import set_seed, CKPT_DIR, RESULTS_DIR, DEVICE

# ── Hyperparameters ────────────────────────────────────────────────────────────

GRID_SIZE  = 40
PATCH_SIZE = 4
D_MODEL    = 64
NHEAD      = 4
NUM_LAYERS = 4
N_TRAJS    = 2000
TRAJ_STEPS = 100
DENSITY    = 0.35
EPOCHS     = 110       # 50 already done + 60 more
BATCH_SIZE = 64
LR         = 3e-4
SEED       = 42
TASK       = "task9_cnn_transformer"


# ── Data generation ────────────────────────────────────────────────────────────

def generate_pairs(n_trajs, grid_size, steps, density, seed):
    rng   = np.random.default_rng(seed)
    inits = (rng.random((n_trajs, grid_size, grid_size)) < density).astype(np.uint8)

    all_states, all_nexts = [], []
    for i, init in enumerate(inits):
        traj = run_trajectory(init, steps)
        all_states.append(traj[:-1])
        all_nexts.append(traj[1:])
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_trajs}", flush=True)

    states = np.concatenate(all_states).astype(np.float32)[:, None]
    nexts  = np.concatenate(all_nexts ).astype(np.float32)[:, None]
    print(f"  → {len(states):,} (state, next) pairs")
    return states, nexts


# ── D4 augmentation ────────────────────────────────────────────────────────────

def d4_augment(x, y):
    """Random element of dihedral group D4; GoL is exactly equivariant under it."""
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
    return x, y


# ── Metrics ────────────────────────────────────────────────────────────────────

def cell_accuracy(logits, targets):
    return ((logits > 0).float() == targets).float().mean().item()


# ── Rollout visualisation ──────────────────────────────────────────────────────

def plot_rollout(model, init, steps=30, label="random"):
    true_traj = run_trajectory(init, steps)

    model.eval()
    curr = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    pred_frames = [init.copy()]
    with torch.no_grad():
        for _ in range(steps):
            curr = model.step(curr)
            pred_frames.append(curr.squeeze().cpu().numpy().astype(np.uint8))

    show_at = np.linspace(0, steps, min(10, steps + 1), dtype=int)
    fig, axes = plt.subplots(2, len(show_at), figsize=(len(show_at) * 2, 5))
    for col, t in enumerate(show_at):
        for row, frames in enumerate([true_traj, pred_frames]):
            axes[row, col].imshow(frames[t], cmap="inferno", vmin=0, vmax=1,
                                  interpolation="nearest")
            axes[row, col].set_title(f"t={t}", fontsize=8)
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("True GoL",       fontsize=9)
    axes[1, 0].set_ylabel("CNN-Transformer", fontsize=9)
    fig.suptitle(f"CNNTransformer — {label} rollout comparison", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{TASK}_rollout_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Rollout → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    print(f"[data] Generating {N_TRAJS} × {TRAJ_STEPS}-step trajectories …")
    states, nexts = generate_pairs(N_TRAJS, GRID_SIZE, TRAJ_STEPS, DENSITY, SEED)

    from torch.utils.data import TensorDataset, DataLoader, random_split
    X, Y    = torch.tensor(states), torch.tensor(nexts)
    n_val   = int(len(X) * 0.15)
    n_train = len(X) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    ds      = TensorDataset(X, Y)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  train={n_train:,}  val={n_val:,}  "
          f"train batches={len(train_dl)}  val batches={len(val_dl)}")

    model = CNNTransformer(
        grid_size=GRID_SIZE, patch_size=PATCH_SIZE,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CNNTransformer  params={n_params:,}  device={DEVICE}")
    print(f"  patches={model.n_patches}  patch_size={PATCH_SIZE}×{PATCH_SIZE}  "
          f"d_model={D_MODEL}  layers={NUM_LAYERS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_path    = os.path.join(CKPT_DIR, f"{TASK}_best.pt")
    history_path = os.path.join(RESULTS_DIR, f"{TASK}_history.json")

    # Resume from existing checkpoint if present
    start_epoch = 1
    best_val    = float("inf")
    history     = {"train_loss": [], "val_loss": [], "val_acc": []}
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        print(f"  Resumed from {ckpt_path}")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        start_epoch = len(history["train_loss"]) + 1
        best_val    = min(history["val_loss"])
        # fast-forward scheduler to match elapsed epochs
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Continuing from epoch {start_epoch}  best_val={best_val:.5f}")

    t0 = time.time()
    print(f"\n{'='*60}")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_dl:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            x, y  = d4_augment(x, y)
            logits = model(x)
            loss   = criterion(logits, y)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y  = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_losses.append(criterion(logits, y).item())
                val_accs.append(cell_accuracy(logits, y))

        tl  = float(np.mean(train_losses))
        vl  = float(np.mean(val_losses))
        acc = float(np.mean(val_accs))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(acc)
        scheduler.step()

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{EPOCHS}  train={tl:.5f}  "
                  f"val={vl:.5f}  val_acc={acc*100:.2f}%  "
                  f"best={best_val:.5f}  t={time.time()-t0:.0f}s", flush=True)

    print(f"\n  Best val loss: {best_val:.5f}  →  {ckpt_path}")

    with open(os.path.join(RESULTS_DIR, f"{TASK}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="val",   color="#C44E52")
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss curves")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot([a * 100 for a in history["val_acc"]], color="#55A868")
    axes[1].set(xlabel="Epoch", ylabel="Cell accuracy (%)",
                title="Validation cell accuracy")
    axes[1].grid(alpha=0.3)
    fig.suptitle("CNNTransformer — training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    rng  = np.random.default_rng(0)
    init = (rng.random((GRID_SIZE, GRID_SIZE)) < DENSITY).astype(np.uint8)
    plot_rollout(model, init, steps=30, label="random")

    glider = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    glider[18:21, 18:21] = np.array([[0,1,0],[0,0,1],[1,1,1]])
    plot_rollout(model, glider, steps=30, label="glider")

    print("\nDone.")


if __name__ == "__main__":
    main()
