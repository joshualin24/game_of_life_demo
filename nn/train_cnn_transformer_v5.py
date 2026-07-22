"""
Task 13: CNN-Transformer V5 — density-diverse + pattern training
-----------------------------------------------------------------
Same architecture as V4 (CNNTransformerV4, circular padding), fine-tuned from
V4's best checkpoint.

V4's failure mode: trained only on density=0.35 random grids, so it developed a
density-regression bias — over-predicting births on sparse grids and over-predicting
deaths on dense grids.  V5 fixes this with three data sources:

  1. Random grids at 8 densities (0.02 – 0.80) for density-uniform coverage.
  2. Named GoL patterns (still lifes, oscillators, spaceships, methuselahs) at
     random positions and D4 orientations.
  3. Combinations of 2-4 patterns at random non-overlapping positions, including
     patterns + sparse random backgrounds.

Data augmentation: D4 only (rotation + reflection).  Translation is NOT used as
a per-batch augmentation because it conflicts with the learnable absolute positional
embedding.  Translation diversity is instead achieved by randomly placing patterns
at different positions during data generation.

Outputs:
  nn/checkpoints/task13_cnn_transformer_v5_best.pt
  nn/results/task13_cnn_transformer_v5_loss.png
  nn/results/task13_cnn_transformer_v5_rollout_*.png
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
EPOCHS     = 100
BATCH_SIZE = 64
LR         = 1e-4
LABEL_SMOOTH = 0.1
SEED       = 42
TASK       = "task13_cnn_transformer_v5"

# Random-grid data
DENSITIES            = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
N_RANDOM_PER_DENSITY = 200    # trajectories per density level
RANDOM_STEPS         = 30

# Named-pattern data
N_PATTERN_TRAJS = 60           # per base pattern (random orient + position each time)
PATTERN_STEPS   = 20

# Combination data
N_COMBO_TRAJS   = 1000         # 2–4 patterns at random non-overlapping positions
COMBO_STEPS     = 30


# ── Named GoL patterns ─────────────────────────────────────────────────────────

def _p(*rows):
    return np.array(rows, dtype=np.uint8)

NAMED_PATTERNS = {
    # Still lifes
    "block":   _p([1,1],
                  [1,1]),
    "beehive": _p([0,1,1,0],
                  [1,0,0,1],
                  [0,1,1,0]),
    "loaf":    _p([0,1,1,0],
                  [1,0,0,1],
                  [0,1,0,1],
                  [0,0,1,0]),
    "boat":    _p([1,1,0],
                  [1,0,1],
                  [0,1,0]),
    "tub":     _p([0,1,0],
                  [1,0,1],
                  [0,1,0]),
    # Oscillators
    "blinker": _p([1,1,1]),
    "toad":    _p([0,1,1,1],
                  [1,1,1,0]),
    "beacon":  _p([1,1,0,0],
                  [1,1,0,0],
                  [0,0,1,1],
                  [0,0,1,1]),
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
    # Spaceships
    "glider":  _p([0,1,0],
                  [0,0,1],
                  [1,1,1]),
    "lwss":    _p([0,1,0,0,1],
                  [1,0,0,0,0],
                  [1,0,0,0,1],
                  [1,1,1,1,0]),
    # Methuselahs
    "r_pent":  _p([0,1,1],
                  [1,1,0],
                  [0,1,0]),
    "acorn":   _p([0,1,0,0,0,0,0],
                  [0,0,0,1,0,0,0],
                  [1,1,0,0,1,1,1]),
}

PATTERN_LIST = list(NAMED_PATTERNS.values())


# ── Pattern placement helpers ──────────────────────────────────────────────────

def rand_orient(rng, pat):
    k = int(rng.integers(0, 4))
    if k:
        pat = np.rot90(pat, k)
    if rng.random() > 0.5:
        pat = np.fliplr(pat)
    return np.ascontiguousarray(pat)


def place_pattern(grid, pat, r0, c0):
    """Place pattern on toroidal grid using modulo arithmetic."""
    H, W = grid.shape
    ph, pw = pat.shape
    rows = (r0 + np.arange(ph)) % H
    cols = (c0 + np.arange(pw)) % W
    grid[np.ix_(rows, cols)] |= pat


def gen_combo_init(rng, grid_size, n_patterns, bg_density=0.0):
    """
    Place n_patterns named patterns at non-overlapping random positions.
    Optionally add a sparse random background.
    """
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
            region = occupied[np.ix_(rows, cols)]
            if not np.any(region[pat.astype(bool)]):
                place_pattern(grid, pat, r0, c0)
                # Mark 3-cell buffer as occupied (prevents immediate interaction)
                buf = 3
                br = np.arange(r0 - buf, r0 + ph + buf) % grid_size
                bc = np.arange(c0 - buf, c0 + pw + buf) % grid_size
                occupied[np.ix_(br, bc)] = True
                placed = True
                break
        if not placed:  # fall back to random placement
            r0 = int(rng.integers(0, grid_size))
            c0 = int(rng.integers(0, grid_size))
            place_pattern(grid, pat, r0, c0)

    return grid


# ── Data generation ────────────────────────────────────────────────────────────

def _collect(inits, steps):
    states, nexts = [], []
    for init in inits:
        traj = run_trajectory(init, steps)
        states.append(traj[:-1])
        nexts.append(traj[1:])
    return (np.concatenate(states).astype(np.float32)[:, None],
            np.concatenate(nexts ).astype(np.float32)[:, None])


def generate_dataset(seed):
    rng = np.random.default_rng(seed)
    all_states, all_nexts = [], []

    # ── 1. Random grids at varied densities ───────────────────────────────
    print("[data] Random grids …")
    for d in DENSITIES:
        inits = (rng.random((N_RANDOM_PER_DENSITY, GRID_SIZE, GRID_SIZE)) < d
                 ).astype(np.uint8)
        s, n = _collect(inits, RANDOM_STEPS)
        all_states.append(s); all_nexts.append(n)
        print(f"  density={d:.2f}  +{len(s):,} pairs", flush=True)

    # ── 2. Named patterns at random positions / orientations ──────────────
    print("[data] Named patterns …")
    for name, base_pat in NAMED_PATTERNS.items():
        inits = []
        for _ in range(N_PATTERN_TRAJS):
            pat = rand_orient(rng, base_pat.copy())
            ph, pw = pat.shape
            r0 = int(rng.integers(0, GRID_SIZE))
            c0 = int(rng.integers(0, GRID_SIZE))
            grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            place_pattern(grid, pat, r0, c0)
            inits.append(grid)
        s, n = _collect(inits, PATTERN_STEPS)
        all_states.append(s); all_nexts.append(n)
        print(f"  {name:10s}  +{len(s):,} pairs", flush=True)

    # ── 3. Combos: 2–4 patterns, optional sparse background ──────────────
    print("[data] Combo patterns …")
    inits = []
    for i in range(N_COMBO_TRAJS):
        n_pat = int(rng.integers(2, 5))           # 2, 3, or 4 patterns
        bg    = float(rng.choice([0.0, 0.0, 0.0, 0.05, 0.10]))  # 60% no bg
        inits.append(gen_combo_init(rng, GRID_SIZE, n_pat, bg))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_COMBO_TRAJS}", flush=True)
    s, n = _collect(inits, COMBO_STEPS)
    all_states.append(s); all_nexts.append(n)
    print(f"  combos  +{len(s):,} pairs", flush=True)

    states = np.concatenate(all_states)
    nexts  = np.concatenate(all_nexts)
    print(f"  → total {len(states):,} (state, next) pairs")
    return states, nexts


# ── Augmentation ───────────────────────────────────────────────────────────────

def augment(x, y):
    """D4 rotation/reflection only — no translation (conflicts with pos_embed)."""
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
    Metrics immune to the dead-cell majority (trivial all-dead predictor = 0% on all).

      born : recall on dead→alive cells  (trivial: 0%)
      surv : recall on alive→alive cells (trivial: 0%)
      died : recall on alive→dead cells  (trivial: 100% — tracked for completeness)
      prec : precision on alive predictions
      rec  : recall on alive cells (= born + surv combined)
      f1   : harmonic mean of prec and rec
    """
    pred = (logits > 0).float()

    born_mask = (x == 0) & (targets == 1)
    surv_mask = (x == 1) & (targets == 1)
    died_mask = (x == 1) & (targets == 0)

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
    axes[1, 0].set_ylabel("CNN-T V5",    fontsize=9)
    fig.suptitle(f"CNNTransformerV5 — {label} rollout", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{TASK}_rollout_{label}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Rollout → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)

    states, nexts = generate_dataset(SEED)

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
    print(f"\n  CNNTransformerV5 (V4 arch)  params={n_params:,}  device={DEVICE}")

    # Warm-start from V4 best checkpoint
    v4_ckpt = os.path.join(CKPT_DIR, "task12_cnn_transformer_v4_best.pt")
    model.load_state_dict(torch.load(v4_ckpt, map_location=DEVICE, weights_only=True))
    print(f"  Loaded V4 weights from {v4_ckpt}")

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

    # Plots
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
    axes[3].plot([b * 100 for b in history["val_born"]], label="born", color="#55A868")
    axes[3].plot([s * 100 for s in history["val_surv"]], label="surv", color="#4C72B0")
    axes[3].set(xlabel="Epoch", ylabel="Accuracy (%)",
                title="born vs surv\n(0% = trivial predictor)")
    axes[3].legend(); axes[3].grid(alpha=0.3)
    fig.suptitle("CNNTransformerV5 — density-diverse + pattern training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{TASK}_loss.png"), dpi=150)
    plt.close(fig)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    rng = np.random.default_rng(0)

    # Random rollout at each density extreme
    for density, label in [(0.05, "sparse"), (0.35, "medium"), (0.70, "dense")]:
        init = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        plot_rollout(model, init, steps=30, label=f"random_{label}")

    # Glider on empty background
    glider_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    place_pattern(glider_init, NAMED_PATTERNS["glider"], 10, 10)
    plot_rollout(model, glider_init, steps=30, label="glider")

    # Two gliders at different positions and orientations
    two_glider_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    g1 = rand_orient(rng, NAMED_PATTERNS["glider"].copy())
    g2 = rand_orient(rng, NAMED_PATTERNS["glider"].copy())
    place_pattern(two_glider_init, g1, 5, 5)
    place_pattern(two_glider_init, g2, 25, 25)
    plot_rollout(model, two_glider_init, steps=30, label="two_gliders")

    # Blinker
    blinker_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    place_pattern(blinker_init, NAMED_PATTERNS["blinker"], 20, 18)
    plot_rollout(model, blinker_init, steps=30, label="blinker")

    # Pulsar
    pulsar_init = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    place_pattern(pulsar_init, NAMED_PATTERNS["pulsar"], 13, 13)
    plot_rollout(model, pulsar_init, steps=30, label="pulsar")

    print("\nDone.")


if __name__ == "__main__":
    main()
