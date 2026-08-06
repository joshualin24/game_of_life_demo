"""
Task 19: CNN-Transformer V11 — PolyKAN-inspired activation, V8-sized model
----------------------------------------------------------------------------------
Architecture: CNNTransformerV11 = CNNTransformerV4 with the CNN stage's GELU
replaced by a learnable per-channel 2nd-degree polynomial activation
(PolyActivation, see nn/models.py). Same d_model/nhead/num_layers as V8
(~291K params, only +~400 extra params for the poly coefficients) — NOT the
larger V10 size.

Motivation: Ahmed & Davis, "It's Much Easier for Neural Networks to Learn
Game of Life Dynamics with the Right Activation Function: Polynomial
Kolmogorov-Arnold Networks" (arXiv:2606.23587) found that a monotonic ReLU
CNN fails to learn GoL's transition rule (a non-monotonic bump in neighbor
count) while a 2nd-degree polynomial activation learns it reliably with
similar or fewer parameters. V9 showed our 291K-param GELU model is
capacity-saturated (more data doesn't help); V10 fixed it by 5x-ing the
parameter count. This experiment tests whether the same 291K-param budget
can reach V10-level accuracy if the CNN's inductive bias is fixed instead
of just adding width/depth.

Training recipe: identical to V10 (from scratch, LR=1e-4, scheduled
sampling K=3, V8 dataset config) except architecture size.
"""

import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nn.models   import CNNTransformerV11
from nn.data_gen import run_trajectory
from nn.utils    import set_seed, CKPT_DIR, RESULTS_DIR, DEVICE

# ── Hyperparameters ────────────────────────────────────────────────────────────

GRID_SIZE  = 40
PATCH_SIZE = 4
D_MODEL    = 64    # V8 size (not V10's 128)
NHEAD      = 4
NUM_LAYERS = 4
EPOCHS     = 60    # V10 converged to perfect well before epoch 100
BATCH_SIZE = 64
LR         = 1e-4  # fresh-start LR (same as V10)
LABEL_SMOOTH = 0.1
K_STEPS    = 3
P_MAX      = 1.0
SEED       = 42
TASK       = "task19_cnn_transformer_v11"

# Dataset — V8/V10 config (best performing)
DENSITIES_NORMAL = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
DENSITIES_HIGH   = [0.65, 0.80]
N_NORMAL         = 200
N_HIGH           = 600
RANDOM_STEPS     = 30
N_PATTERN_TRAJS  = 60
PATTERN_STEPS    = 20
N_COMBO_TRAJS    = 1000
COMBO_STEPS      = 30


# ── Named GoL patterns ────────────────────────────────────────────────────────

def _p(*rows): return np.array(rows, dtype=np.uint8)

NAMED_PATTERNS = {
    "block":   _p([1,1],[1,1]),
    "beehive": _p([0,1,1,0],[1,0,0,1],[0,1,1,0]),
    "loaf":    _p([0,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,0]),
    "boat":    _p([1,1,0],[1,0,1],[0,1,0]),
    "tub":     _p([0,1,0],[1,0,1],[0,1,0]),
    "blinker": _p([1,1,1]),
    "toad":    _p([0,1,1,1],[1,1,1,0]),
    "beacon":  _p([1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]),
    "pulsar":  _p([0,0,1,1,1,0,0,0,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [0,0,1,1,1,0,0,0,1,1,1,0,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,0,0,0,1,1,1,0,0],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [1,0,0,0,0,1,0,1,0,0,0,0,1],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,1,1,1,0,0,0,1,1,1,0,0]),
    "glider":  _p([0,1,0],[0,0,1],[1,1,1]),
    "lwss":    _p([0,1,0,0,1],[1,0,0,0,0],[1,0,0,0,1],[1,1,1,1,0]),
    "r_pent":  _p([0,1,1],[1,1,0],[0,1,0]),
    "acorn":   _p([0,1,0,0,0,0,0],[0,0,0,1,0,0,0],[1,1,0,0,1,1,1]),
}
PATTERN_LIST = list(NAMED_PATTERNS.values())


# ── Helpers (identical to V8/V10) ──────────────────────────────────────────────

def rand_orient(rng, pat):
    k = int(rng.integers(0, 4))
    if k: pat = np.rot90(pat, k)
    if rng.random() > 0.5: pat = np.fliplr(pat)
    return np.ascontiguousarray(pat)

def place_pattern(grid, pat, r0, c0):
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat

def gen_combo_init(rng, grid_size, n_patterns, bg_density=0.0):
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    if bg_density > 0:
        grid = (rng.random((grid_size, grid_size)) < bg_density).astype(np.uint8)
    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    for _ in range(n_patterns):
        pat = rand_orient(rng, PATTERN_LIST[int(rng.integers(0, len(PATTERN_LIST)))])
        ph, pw = pat.shape
        placed = False
        for _ in range(30):
            r0 = int(rng.integers(0, grid_size))
            c0 = int(rng.integers(0, grid_size))
            rows = (r0 + np.arange(ph)) % grid_size
            cols = (c0 + np.arange(pw)) % grid_size
            if not np.any(occupied[np.ix_(rows, cols)][pat.astype(bool)]):
                place_pattern(grid, pat, r0, c0)
                buf = 3
                br = np.arange(r0 - buf, r0 + ph + buf) % grid_size
                bc = np.arange(c0 - buf, c0 + pw + buf) % grid_size
                occupied[np.ix_(br, bc)] = True
                placed = True
                break
        if not placed:
            place_pattern(grid, pat, int(rng.integers(0, grid_size)),
                                     int(rng.integers(0, grid_size)))
    return grid

def _collect_kstep(inits, steps, k):
    X_list, Y_list = [], []
    for init in inits:
        traj = run_trajectory(init, steps)
        for i in range(len(traj) - k):
            X_list.append(traj[i])
            Y_list.append(traj[i+1 : i+k+1])
    X  = np.array(X_list, dtype=np.float32)[:, None]
    Yk = np.array(Y_list, dtype=np.float32)[:, :, None]
    return X, Yk

def generate_dataset(seed, k):
    rng = np.random.default_rng(seed)
    Xs, Yks = [], []

    print("[data] Random grids (normal densities) …")
    for d in DENSITIES_NORMAL:
        inits = (rng.random((N_NORMAL, GRID_SIZE, GRID_SIZE)) < d).astype(np.uint8)
        x, yk = _collect_kstep(inits, RANDOM_STEPS, k)
        Xs.append(x); Yks.append(yk)
        print(f"  density={d:.2f}  +{len(x):,} samples", flush=True)

    print(f"[data] High-density grids ({N_HIGH} trajs each) …")
    for d in DENSITIES_HIGH:
        inits = (rng.random((N_HIGH, GRID_SIZE, GRID_SIZE)) < d).astype(np.uint8)
        x, yk = _collect_kstep(inits, RANDOM_STEPS, k)
        Xs.append(x); Yks.append(yk)
        print(f"  density={d:.2f}  +{len(x):,} samples", flush=True)

    print("[data] Named patterns …")
    for name, base_pat in NAMED_PATTERNS.items():
        inits = []
        for _ in range(N_PATTERN_TRAJS):
            pat = rand_orient(rng, base_pat.copy())
            grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            place_pattern(grid, pat, int(rng.integers(0, GRID_SIZE)),
                                     int(rng.integers(0, GRID_SIZE)))
            inits.append(grid)
        x, yk = _collect_kstep(inits, PATTERN_STEPS, k)
        Xs.append(x); Yks.append(yk)
        print(f"  {name:10s}  +{len(x):,} samples", flush=True)

    print("[data] Combo patterns …")
    inits = []
    for i in range(N_COMBO_TRAJS):
        n_pat = int(rng.integers(2, 5))
        bg    = float(rng.choice([0.0, 0.0, 0.0, 0.05, 0.10]))
        inits.append(gen_combo_init(rng, GRID_SIZE, n_pat, bg))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_COMBO_TRAJS}", flush=True)
    x, yk = _collect_kstep(inits, COMBO_STEPS, k)
    Xs.append(x); Yks.append(yk)
    print(f"  combos  +{len(x):,} samples", flush=True)

    X  = np.concatenate(Xs)
    Yk = np.concatenate(Yks)
    print(f"  → total {len(X):,} samples  (each has {k} target frames)")
    return X, Yk


# ── Augmentation / metrics / rollout ──────────────────────────────────────────

def augment(x, y_k):
    rot = int(torch.randint(0, 4, (1,)).item())
    flip = torch.rand(1).item() > 0.5
    if rot:
        x   = torch.rot90(x,   rot, dims=[-2, -1])
        y_k = torch.rot90(y_k, rot, dims=[-2, -1])
    if flip:
        x   = torch.flip(x,   dims=[-1])
        y_k = torch.flip(y_k, dims=[-1])
    return x, y_k

def compute_metrics(logits, x, targets):
    pred = (logits > 0).float()
    born_mask = (x == 0) & (targets == 1)
    surv_mask = (x == 1) & (targets == 1)
    died_mask = (x == 1) & (targets == 0)
    def _acc(mask, pv):
        n = mask.float().sum().item()
        return (pred[mask] == pv).float().sum().item() / (n + 1e-8)
    tp   = (pred * targets).sum().item()
    fp   = (pred * (1 - targets)).sum().item()
    fn   = ((1 - pred) * targets).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return dict(f1=f1, prec=prec, rec=rec,
                born=_acc(born_mask, 1), surv=_acc(surv_mask, 1),
                died=_acc(died_mask, 0))

def plot_rollout(model, init, steps=30, label=""):
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
    axes[0, 0].set_ylabel("True GoL", fontsize=9)
    axes[1, 0].set_ylabel("CNN-T V11", fontsize=9)
    fig.suptitle(f"CNNTransformerV11 (PolyAct) — {label} rollout", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{TASK}_rollout_{label}.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Rollout → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    X, Yk = generate_dataset(SEED, K_STEPS)

    from torch.utils.data import TensorDataset, DataLoader, random_split
    X_t, Yk_t = torch.tensor(X), torch.tensor(Yk)
    n_val   = int(len(X_t) * 0.15)
    n_train = len(X_t) - n_val
    gen     = torch.Generator().manual_seed(SEED)
    ds      = TensorDataset(X_t, Yk_t)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  train={n_train:,}  val={n_val:,}  "
          f"train batches={len(train_dl)}  val batches={len(val_dl)}")

    model = CNNTransformerV11(
        grid_size=GRID_SIZE, patch_size=PATCH_SIZE,
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CNNTransformerV11  params={n_params:,}  device={DEVICE}")
    print(f"  d_model={D_MODEL}  nhead={NHEAD}  num_layers={NUM_LAYERS}  "
          f"dim_ff={D_MODEL*4}  CNN activation=PolyActivation(deg=2)  "
          f"(training from scratch)")

    def smooth_bce(logits, targets):
        y_s = targets * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2
        return F.binary_cross_entropy_with_logits(logits, y_s)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    ckpt_best  = os.path.join(CKPT_DIR, f"{TASK}_best.pt")
    ckpt_final = os.path.join(CKPT_DIR, f"{TASK}_final.pt")
    best_val = float("inf")
    history  = {"train_loss": [], "val_loss": [], "p_sample": [],
                "val_f1": [], "val_prec": [], "val_rec": [],
                "val_born": [], "val_surv": [], "val_died": []}
    t0 = time.time()

    print(f"\n  K_STEPS={K_STEPS}  LR={LR}  P_MAX={P_MAX}  scheduled sampling")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        p_sample = (epoch - 1) / max(EPOCHS - 1, 1) * P_MAX

        model.train()
        train_losses = []
        for x, y_k in train_dl:
            x, y_k = x.to(DEVICE), y_k.to(DEVICE)
            x, y_k = augment(x, y_k)
            curr = x
            total_loss = torch.tensor(0.0, device=DEVICE)
            for step in range(K_STEPS):
                logits = model(curr)
                target = y_k[:, step]
                total_loss = total_loss + smooth_bce(logits, target)
                with torch.no_grad():
                    pred_hard = (torch.sigmoid(logits) > 0.5).float()
                    true_next = y_k[:, step]
                    mask = (torch.rand(x.shape[0], 1, 1, 1, device=DEVICE)
                            < p_sample).float()
                    curr = mask * pred_hard + (1 - mask) * true_next
            loss = total_loss / K_STEPS
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses, val_f1s, val_precs, val_recs = [], [], [], []
        val_borns, val_survs, val_dieds = [], [], []
        with torch.no_grad():
            for x, y_k in val_dl:
                x, y_k = x.to(DEVICE), y_k.to(DEVICE)
                logits  = model(x)
                target1 = y_k[:, 0]
                val_losses.append(smooth_bce(logits, target1).item())
                m = compute_metrics(logits, x, target1)
                val_f1s.append(m["f1"]); val_precs.append(m["prec"])
                val_recs.append(m["rec"]); val_borns.append(m["born"])
                val_survs.append(m["surv"]); val_dieds.append(m["died"])

        tl   = float(np.mean(train_losses))
        vl   = float(np.mean(val_losses))
        f1   = float(np.mean(val_f1s))
        prec = float(np.mean(val_precs))
        rec  = float(np.mean(val_recs))
        born = float(np.mean(val_borns))
        surv = float(np.mean(val_survs))
        died = float(np.mean(val_dieds))
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["p_sample"].append(p_sample)
        history["val_f1"].append(f1); history["val_prec"].append(prec)
        history["val_rec"].append(rec); history["val_born"].append(born)
        history["val_surv"].append(surv); history["val_died"].append(died)
        scheduler.step()

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), ckpt_best)
        torch.save(model.state_dict(), ckpt_final)
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(CKPT_DIR, f"{TASK}_ep{epoch:03d}.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{EPOCHS}  p={p_sample:.2f}  "
                  f"train={tl:.5f}  val={vl:.5f}  "
                  f"f1={f1:.4f}  prec={prec*100:.1f}%  rec={rec*100:.1f}%  "
                  f"born={born*100:.1f}%  surv={surv*100:.1f}%  died={died*100:.1f}%  "
                  f"best={best_val:.5f}  t={time.time()-t0:.0f}s", flush=True)

    print(f"\n  Best val loss: {best_val:.5f}  →  {ckpt_best}")

    with open(os.path.join(RESULTS_DIR, f"{TASK}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history["train_loss"], label="train", color="#4C72B0")
    axes[0].plot(history["val_loss"],   label="val",   color="#C44E52")
    axes[0].set(xlabel="Epoch", ylabel="Smoothed BCE (step 1)", title="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    ax0b = axes[0].twinx()
    ax0b.plot(history["p_sample"], color="gray", linestyle="--", alpha=0.5)
    ax0b.set_ylabel("p_sample", color="gray", fontsize=8)
    axes[1].plot(history["val_f1"],   label="F1",        color="#DD8452")
    axes[1].plot(history["val_prec"], label="precision", color="#4C72B0")
    axes[1].plot(history["val_rec"],  label="recall",    color="#C44E52")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Prec / Rec / F1 (step 1)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot([b*100 for b in history["val_born"]], label="born",  color="#55A868")
    axes[2].plot([s*100 for s in history["val_surv"]], label="surv",  color="#4C72B0")
    axes[2].plot([d*100 for d in history["val_died"]], label="died",  color="#C44E52")
    axes[2].set(xlabel="Epoch", ylabel="Per-event accuracy (%)", title="GoL event accuracies")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[3].plot([b*100 for b in history["val_born"]], label="born", color="#55A868")
    axes[3].plot([s*100 for s in history["val_surv"]], label="surv", color="#4C72B0")
    axes[3].set(xlabel="Epoch", ylabel="Accuracy (%)", title="born vs surv")
    axes[3].legend(); axes[3].grid(alpha=0.3)
    fig.suptitle(
        f"CNNTransformerV11 (PolyAct CNN)  d_model={D_MODEL}  {NUM_LAYERS} layers  "
        f"~{n_params//1000}K params  K={K_STEPS}",
        fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)

    # Rollouts — best checkpoint
    model.load_state_dict(torch.load(ckpt_best, map_location=DEVICE, weights_only=True))
    print("\n[rollouts] best checkpoint")
    rng = np.random.default_rng(0)
    for density, label in [(0.05,"sparse"),(0.35,"medium"),(0.65,"dense65"),(0.80,"dense80")]:
        init = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        plot_rollout(model, init, steps=30, label=f"best_random_{label}")
    for pat_name in ["glider", "blinker", "pulsar", "lwss", "acorn"]:
        init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        place_pattern(init, NAMED_PATTERNS[pat_name], 10, 10)
        plot_rollout(model, init, steps=30, label=f"best_{pat_name}")

    # Rollouts — final checkpoint
    model.load_state_dict(torch.load(ckpt_final, map_location=DEVICE, weights_only=True))
    print(f"\n[rollouts] final checkpoint (epoch {EPOCHS})")
    rng2 = np.random.default_rng(0)
    for density, label in [(0.05,"sparse"),(0.35,"medium"),(0.65,"dense65"),(0.80,"dense80")]:
        init = (rng2.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        plot_rollout(model, init, steps=30, label=f"final_random_{label}")
    for pat_name in ["glider", "blinker", "pulsar", "lwss", "acorn"]:
        init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        place_pattern(init, NAMED_PATTERNS[pat_name], 10, 10)
        plot_rollout(model, init, steps=30, label=f"final_{pat_name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
