"""
Task 8: Next-Step Transformer (ViT-style)
------------------------------------------
Trains a NextStepTransformer to predict the GoL state at t+1 given t.
The model can be unrolled autoregressively for any number of steps.

Dataset  : (state_t, state_t+1) pairs from random 40×40 trajectories.
           Also evaluated on named patterns (glider, blinker, etc.)
Objective: BCE loss on next-state prediction.
           No pos_weight needed — grid density at t+1 is similar to t (~35%).
Outputs  :
  nn/checkpoints/task8_next_step_transformer_best.pt
  nn/results/task8_next_step_transformer_loss.png
  nn/results/task8_next_step_transformer_rollout.png  (qualitative comparison)
"""

import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nn.models  import NextStepTransformer
from nn.data_gen import gol_step, run_trajectory
from nn.utils   import set_seed, CKPT_DIR, RESULTS_DIR, DEVICE

# ── Hyperparameters ────────────────────────────────────────────────────────────

GRID_SIZE  = 40
PATCH_SIZE = 4          # → 100 patches per frame
D_MODEL    = 64
NHEAD      = 4
NUM_LAYERS = 4
N_TRAJS    = 2000       # initial conditions (4× more for diversity)
TRAJ_STEPS = 100        # steps per trajectory → N_TRAJS × TRAJ_STEPS pairs
DENSITY    = 0.35
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 3e-4
SEED       = 42
TASK       = "task8_next_step_transformer"


# ── Data generation ────────────────────────────────────────────────────────────

def generate_pairs(n_trajs: int, grid_size: int, steps: int,
                   density: float, seed: int):
    """Returns (states, nexts) each (N, 1, H, W) float32 tensors."""
    rng   = np.random.default_rng(seed)
    inits = (rng.random((n_trajs, grid_size, grid_size)) < density).astype(np.uint8)

    all_states, all_nexts = [], []
    for i, init in enumerate(inits):
        traj = run_trajectory(init, steps)             # (steps+1, H, W)
        all_states.append(traj[:-1])                   # frames 0..steps-1
        all_nexts.append(traj[1:])                     # frames 1..steps
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_trajs}", flush=True)

    states = np.concatenate(all_states).astype(np.float32)[:, None]  # (N,1,H,W)
    nexts  = np.concatenate(all_nexts ).astype(np.float32)[:, None]
    print(f"  → {len(states):,} (state, next) pairs")
    return states, nexts


# ── D4 augmentation ────────────────────────────────────────────────────────────

def d4_augment(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a random element of the dihedral group D4 to a batch.
    GoL is equivariant under rotation and reflection, so each transformed
    (state_t, state_t+1) pair is a valid training sample.
    x, y: (B, 1, H, W)
    """
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
    return x, y


# ── Accuracy helper ────────────────────────────────────────────────────────────

def cell_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = (logits > 0).float()
    return (pred == targets).float().mean().item()


# ── Rollout visualisation ──────────────────────────────────────────────────────

def plot_rollout(model: nn.Module, init: np.ndarray, steps: int = 20,
                 label: str = "random"):
    """Compare true GoL rollout vs transformer rollout."""
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
        for row, (frames, title) in enumerate([
            (true_traj,   "True GoL"),
            (pred_frames, "Transformer"),
        ]):
            axes[row, col].imshow(frames[t], cmap="inferno", vmin=0, vmax=1,
                                  interpolation="nearest")
            axes[row, col].set_title(f"t={t}", fontsize=8)
            axes[row, col].axis("off")
        axes[0, col].set_ylabel(["True GoL", "Transformer"][0], fontsize=8)

    axes[0, 0].set_ylabel("True GoL",    fontsize=9)
    axes[1, 0].set_ylabel("Transformer", fontsize=9)
    fig.suptitle(f"NextStepTransformer — {label} rollout comparison",
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{TASK}_rollout_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Rollout → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    print(f"[data] Generating {N_TRAJS} × {TRAJ_STEPS}-step trajectories "
          f"on {GRID_SIZE}×{GRID_SIZE} grids …")
    states, nexts = generate_pairs(N_TRAJS, GRID_SIZE, TRAJ_STEPS, DENSITY, SEED)

    X = torch.tensor(states)
    Y = torch.tensor(nexts)
    n_val   = int(len(X) * 0.15)
    n_train = len(X) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    from torch.utils.data import TensorDataset, DataLoader, random_split
    ds = TensorDataset(X, Y)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  train={n_train:,}  val={n_val:,}  "
          f"train batches={len(train_dl)}  val batches={len(val_dl)}")

    model = NextStepTransformer(
        grid_size=GRID_SIZE, patch_size=PATCH_SIZE,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  NextStepTransformer  params={n_params:,}  device={DEVICE}")
    print(f"  patches={model.n_patches}  patch_size={PATCH_SIZE}×{PATCH_SIZE}  "
          f"d_model={D_MODEL}  layers={NUM_LAYERS}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(CKPT_DIR, f"{TASK}_best.pt")
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0        = time.time()

    print(f"\n{'='*60}")
    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_losses = []
        for x, y in train_dl:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            x, y   = d4_augment(x, y)          # random D4 symmetry each batch
            logits  = model(x)
            loss    = criterion(logits, y)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── val ──
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

    # save history
    with open(os.path.join(RESULTS_DIR, f"{TASK}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # loss curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="val",   color="#C44E52")
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss curves")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot([a * 100 for a in history["val_acc"]], color="#55A868")
    axes[1].set(xlabel="Epoch", ylabel="Cell accuracy (%)",
                title="Validation cell accuracy")
    axes[1].grid(alpha=0.3)
    fig.suptitle("NextStepTransformer — training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)

    # ── Rollout comparisons ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    rng  = np.random.default_rng(0)
    init = (rng.random((GRID_SIZE, GRID_SIZE)) < DENSITY).astype(np.uint8)
    plot_rollout(model, init, steps=30, label="random")

    # glider on 40×40
    glider = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    glider[18:21, 18:21] = np.array([[0,1,0],[0,0,1],[1,1,1]])
    plot_rollout(model, glider, steps=30, label="glider")

    print("\nDone.")


if __name__ == "__main__":
    main()
