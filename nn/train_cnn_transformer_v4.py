"""
Task 12: CNN-Transformer V4 — circular padding, fine-tuned from V3
-------------------------------------------------------------------
Architecture: identical to V3 (lossless flatten+linear tokenization) except
the CNN uses circular (toroidal) padding instead of zero padding.  This is
the architecturally correct choice for a grid with periodic boundary conditions.

Training: warm-started from V3's best checkpoint (weights transfer directly).
Fine-tuned at LR=1e-4 with label smoothing (ε=0.1) to prevent memorization.
Label smoothing bounds optimal logits at ±2.94, preventing the gradient
explosion that occurs when circular padding enables perfect training-set fit.

Metrics: GoL event-based (born/survived/died accuracy) + F1 on alive cells.
Raw cell accuracy is omitted — trivially ~90% due to dead-cell majority.

Outputs:
  nn/checkpoints/task12_cnn_transformer_v4_best.pt
  nn/results/task12_cnn_transformer_v4_loss.png
  nn/results/task12_cnn_transformer_v4_rollout_random.png
  nn/results/task12_cnn_transformer_v4_rollout_glider.png
  nn/results/task12_cnn_transformer_v4_rollout_blinker.png
"""

import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nn.models   import CNNTransformerV4
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
EPOCHS       = 100
BATCH_SIZE   = 64
LR           = 1e-4      # fine-tuning LR (V3 was trained at 3e-4)
LABEL_SMOOTH = 0.1       # targets → {0.05, 0.95}; bounds logits at ±2.94
SEED         = 42
TASK         = "task12_cnn_transformer_v4"


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


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment(x, y):
    """D4 rotation/reflection (GoL is equivariant under the dihedral group D4).
    Translation augmentation is omitted: it conflicts with the learnable absolute
    positional embedding, preventing the model from associating spatial structure
    with patch positions and slowing convergence severely."""
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
        y = torch.rot90(y, k, dims=[-2, -1])
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])
        y = torch.flip(y, dims=[-1])
    return x, y


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(logits, x, targets):
    """
    Metrics that measure GoL rule learning without rewarding true-negative prediction.

    The trivial all-dead predictor (~90% raw accuracy) scores:
      born=0%, surv=0%, prec=0%, rec=0%, f1=0%
    so none of these metrics can be gamed by predicting dead everywhere.

    Cells are categorised by GoL transition type — only 'born' and 'surv' are
    used as primary metrics because they measure predicting alive, which is
    entirely immune to the dead-cell majority:

      born : dead→alive  recall — fraction of births the model catches
      surv : alive→alive recall — fraction of survivals the model catches
      prec : precision on alive predictions — when the model predicts alive, is it right?
      rec  : recall on alive cells (= (born_correct+surv_correct)/(born_total+surv_total))
      f1   : harmonic mean of prec and rec — canonical single-number summary

    'died' and 'stayed' are omitted: a model that predicts everything dead gets
    died=100%, stayed=100%, making them indistinguishable from rule-learning.
    """
    pred = (logits > 0).float()

    born_mask = (x == 0) & (targets == 1)   # dead→alive
    surv_mask = (x == 1) & (targets == 1)   # alive→alive
    died_mask = (x == 1) & (targets == 0)   # alive→dead

    def _acc(mask, pred_val):
        n = mask.float().sum().item()
        return (pred[mask] == pred_val).float().sum().item() / (n + 1e-8)

    born_acc = _acc(born_mask, 1)
    surv_acc = _acc(surv_mask, 1)
    died_acc = _acc(died_mask, 0)

    tp   = (pred * targets).sum().item()
    fp   = (pred * (1 - targets)).sum().item()
    fn   = ((1 - pred) * targets).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    return dict(f1=f1, prec=prec, rec=rec, born=born_acc, surv=surv_acc, died=died_acc)


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
    axes[0, 0].set_ylabel("True GoL",    fontsize=9)
    axes[1, 0].set_ylabel("CNN-T V4",    fontsize=9)
    fig.suptitle(f"CNNTransformerV4 — {label} rollout", fontweight="bold")
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

    model = CNNTransformerV4(
        grid_size=GRID_SIZE, patch_size=PATCH_SIZE,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CNNTransformerV4  params={n_params:,}  device={DEVICE}")

    # Warm-start from V3 (architectures identical; only padding_mode differs)
    v3_ckpt = os.path.join(CKPT_DIR, "task11_cnn_transformer_v3_best.pt")
    model.load_state_dict(torch.load(v3_ckpt, map_location=DEVICE, weights_only=True))
    print(f"  Loaded V3 weights from {v3_ckpt}")

    def smooth_bce(logits, targets):
        y_s = targets * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
        return F.binary_cross_entropy_with_logits(logits, y_s)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_path = os.path.join(CKPT_DIR, f"{TASK}_best.pt")
    best_val  = float("inf")
    history   = {"train_loss": [], "val_loss": [],
                 "val_f1": [], "val_prec": [], "val_rec": [],
                 "val_born": [], "val_surv": [], "val_died": []}
    t0        = time.time()

    print(f"\n{'='*60}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y in train_dl:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            x, y   = augment(x, y)
            logits = model(x)
            loss   = smooth_bce(logits, y)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        val_f1s, val_precs, val_recs, val_borns, val_survs, val_dieds = [], [], [], [], [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y   = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_losses.append(smooth_bce(logits, y).item())
                m = compute_metrics(logits, x, y)
                val_f1s.append(m["f1"])
                val_precs.append(m["prec"])
                val_recs.append(m["rec"])
                val_borns.append(m["born"])
                val_survs.append(m["surv"])
                val_dieds.append(m["died"])

        tl   = float(np.mean(train_losses))
        vl   = float(np.mean(val_losses))
        f1   = float(np.mean(val_f1s))
        prec = float(np.mean(val_precs))
        rec  = float(np.mean(val_recs))
        born = float(np.mean(val_borns))
        surv = float(np.mean(val_survs))
        died = float(np.mean(val_dieds))
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_f1"].append(f1)
        history["val_prec"].append(prec)
        history["val_rec"].append(rec)
        history["val_born"].append(born)
        history["val_surv"].append(surv)
        history["val_died"].append(died)
        scheduler.step()

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{EPOCHS}  train={tl:.5f}  val={vl:.5f}  "
                  f"f1={f1:.4f}  prec={prec*100:.1f}%  rec={rec*100:.1f}%  "
                  f"born={born*100:.1f}%  surv={surv*100:.1f}%  died={died*100:.1f}%  "
                  f"best={best_val:.5f}  t={time.time()-t0:.0f}s", flush=True)

    print(f"\n  Best val loss: {best_val:.5f}  →  {ckpt_path}")

    with open(os.path.join(RESULTS_DIR, f"{TASK}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="val",   color="#C44E52")
    axes[0].set(xlabel="Epoch", ylabel="Smoothed BCE Loss", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["val_f1"],   label="F1",        color="#DD8452")
    axes[1].plot(history["val_prec"], label="precision", color="#4C72B0")
    axes[1].plot(history["val_rec"],  label="recall",    color="#C44E52")
    axes[1].set(xlabel="Epoch", ylabel="Score (alive cells)", title="Precision / Recall / F1")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot([b * 100 for b in history["val_born"]], label="born (dead→alive)",  color="#55A868")
    axes[2].plot([s * 100 for s in history["val_surv"]], label="surv (alive→alive)", color="#4C72B0")
    axes[2].plot([d * 100 for d in history["val_died"]], label="died (alive→dead)",  color="#C44E52")
    axes[2].set(xlabel="Epoch", ylabel="Per-event accuracy (%)", title="GoL event accuracies")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[3].plot([b * 100 for b in history["val_born"]], label="born",  color="#55A868")
    axes[3].plot([s * 100 for s in history["val_surv"]], label="surv",  color="#4C72B0")
    axes[3].set(xlabel="Epoch", ylabel="Accuracy (%)",
                title="born vs surv\n(both = 0% for trivial predictor)")
    axes[3].legend(); axes[3].grid(alpha=0.3)
    fig.suptitle("CNNTransformerV4 — fine-tuned from V3", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    rng  = np.random.default_rng(0)
    init = (rng.random((GRID_SIZE, GRID_SIZE)) < DENSITY).astype(np.uint8)
    plot_rollout(model, init, steps=30, label="random")

    # Glider at a random position (not always center)
    glider_pat = np.array([[0,1,0],[0,0,1],[1,1,1]], dtype=np.uint8)
    glider_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    r = int(rng.integers(5, GRID_SIZE - 5))
    c = int(rng.integers(5, GRID_SIZE - 5))
    glider_init[r:r+3, c:c+3] = glider_pat
    plot_rollout(model, glider_init, steps=30, label="glider")

    # Blinker
    blinker_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    blinker_init[20, 19:22] = 1
    plot_rollout(model, blinker_init, steps=30, label="blinker")

    print("\nDone.")


if __name__ == "__main__":
    main()
