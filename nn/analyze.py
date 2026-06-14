"""
Cross-task analysis and visualization for all 6 GoL neural network experiments.

Generates in nn/results/:
  analysis_01_sensitivity_examples.png  – 6 grid/true/pred triplets
  analysis_02_sensitivity_scatter.png   – pixel-level predicted vs true scatter
  analysis_03_nca_accuracy_by_density.png – NeuralCA acc vs initial density
  analysis_04_nca_rollout_divergence.png  – NeuralCA cumulative error over steps
  analysis_05_task1_vs_task4.png          – NextState vs NeuralCA cell accuracy
  analysis_06_chaos_scatter.png           – ChaosPredictor pred vs true
  analysis_07_rollout_error_vs_k.png      – RolloutPredictor BCE vs k
  analysis_08_attractor_difficulty.png    – fate unpredictability summary
  analysis_09_summary_dashboard.png       – one-page summary of all 6 tasks
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

from nn.data_gen  import (load_dataset, run_trajectory, gol_step,
                           compute_sensitivity_map)
from nn.models    import (SensitivityUNet, NeuralCA, NextStatePredictor,
                           ChaosPredictor, RolloutPredictor, FateClassifier)
from nn.utils     import DEVICE, CKPT_DIR, RESULTS_DIR
from nn.train_attractor import augment, LABEL_NAMES

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(cls, ckpt_name, **kwargs):
    m = cls(**kwargs).to(DEVICE)
    path = os.path.join(CKPT_DIR, ckpt_name)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def savefig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── Load all models & datasets ─────────────────────────────────────────────────

print("Loading models …")
unet    = load_model(SensitivityUNet, "task2_sensitivity_best.pt", base_ch=32)
nca     = load_model(NeuralCA,        "task4_neural_ca_best.pt",   hidden=64)
nsp     = load_model(NextStatePredictor, "task1_next_state_best.pt", channels=64)
chaos   = load_model(ChaosPredictor,  "task3_chaos_best.pt",       base_ch=32)
rollout = load_model(RolloutPredictor,"task5_rollout_best.pt",      channels=64, n_res=6)
fate_m  = load_model(FateClassifier,  "task6_attractor_best.pt",   base_ch=16, n_classes=4)

print("Loading datasets …")
sens_data = load_dataset("sensitivity_sparse") if os.path.exists(
    os.path.join(os.path.dirname(__file__), "data", "sensitivity_sparse.npz")
) else load_dataset("sensitivity")

traj_data = load_dataset("trajectories")
chaos_data= load_dataset("chaos")
attr_data = load_dataset("attractor")

rng = np.random.default_rng(0)

# ── Fig 01: Sensitivity prediction examples ────────────────────────────────────

print("[01] Sensitivity examples …")
grids_s = sens_data["grids"].astype(np.float32)
maps_s  = sens_data["maps"].astype(np.float32)
maps_log = np.log1p(maps_s)
per_max  = maps_log.max(axis=(1,2), keepdims=True).clip(min=1e-6)
maps_norm= (maps_log / per_max).astype(np.float32)

idx6 = rng.choice(len(grids_s), 6, replace=False)
fig, axes = plt.subplots(6, 3, figsize=(9, 14))
for row, idx in enumerate(idx6):
    g = to_tensor(grids_s[idx][None, None])
    with torch.no_grad():
        pred = unet(g).squeeze().cpu().numpy()
    true = maps_norm[idx]
    vmax = max(true.max(), pred.max(), 1e-3)

    axes[row,0].imshow(grids_s[idx], cmap="inferno", vmin=0, vmax=1,
                       interpolation="nearest")
    axes[row,0].set_ylabel(f"sample {idx}", fontsize=8)
    axes[row,1].imshow(true, cmap="hot", vmin=0, vmax=vmax,
                       interpolation="nearest")
    im = axes[row,2].imshow(pred, cmap="hot", vmin=0, vmax=vmax,
                            interpolation="nearest")
    r = np.corrcoef(true.ravel(), pred.ravel())[0,1]
    axes[row,2].set_title(f"r={r:.2f}", fontsize=8)
    for ax in axes[row]: ax.axis("off")

for ax, lbl in zip(axes[0], ["Input grid", "True sensitivity", "Predicted"]):
    ax.set_title(lbl, fontsize=10, fontweight="bold")
fig.suptitle("Task 2 — Sensitivity map prediction examples", fontweight="bold", fontsize=13)
fig.tight_layout()
savefig(fig, "analysis_01_sensitivity_examples.png")

# ── Fig 02: Sensitivity scatter ────────────────────────────────────────────────

print("[02] Sensitivity scatter …")
n_scatter = min(500, len(grids_s))
preds_flat, trues_flat = [], []
for i in rng.choice(len(grids_s), n_scatter, replace=False):
    g = to_tensor(grids_s[i][None, None])
    with torch.no_grad():
        p = unet(g).squeeze().cpu().numpy().ravel()
    preds_flat.append(p)
    trues_flat.append(maps_norm[i].ravel())

P = np.concatenate(preds_flat)
T = np.concatenate(trues_flat)
corr = np.corrcoef(P, T)[0,1]

fig, ax = plt.subplots(figsize=(6,6))
ax.hexbin(T, P, gridsize=60, cmap="hot", mincnt=1)
lim = max(T.max(), P.max())
ax.plot([0, lim],[0, lim], "c--", lw=1.5, label="y=x")
ax.set_xlabel("True sensitivity (normalised)")
ax.set_ylabel("Predicted sensitivity")
ax.set_title(f"Task 2 — pixel-level scatter  (Pearson r = {corr:.3f})",
             fontweight="bold")
ax.legend()
fig.tight_layout()
savefig(fig, "analysis_02_sensitivity_scatter.png")

# ── Fig 03: NeuralCA accuracy vs initial density ───────────────────────────────

print("[03] NCA accuracy vs density …")
densities = np.arange(0.05, 0.96, 0.05)
accs = []
N_TEST = 20
for d in densities:
    hits = tot = 0
    for _ in range(N_TEST):
        init = (rng.random((32,32)) < d).astype(np.float32)
        true_next = gol_step(init.astype(np.uint8)).astype(np.float32)
        x = to_tensor(init[None, None])
        with torch.no_grad():
            pred = (nca(x) > 0.5).float().squeeze().cpu().numpy()
        hits += (pred == true_next).sum()
        tot  += pred.size
    accs.append(hits / tot)

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(densities*100, np.array(accs)*100, "o-", color="#4C72B0", lw=2)
ax.axhline(100, color="gray", ls="--", lw=1, label="Perfect (GoL rule)")
ax.set_xlabel("Initial density (%)")
ax.set_ylabel("Cell accuracy (%)")
ax.set_title("Task 4 — NeuralCA accuracy vs initial density", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "analysis_03_nca_accuracy_by_density.png")

# ── Fig 04: NeuralCA rollout divergence ───────────────────────────────────────

print("[04] NCA rollout divergence …")
MAX_STEPS = 50
N_SEEDS   = 10
divergences = np.zeros((N_SEEDS, MAX_STEPS))

for seed_i in range(N_SEEDS):
    init = (rng.random((32,32)) < 0.35).astype(np.uint8)
    true_traj = run_trajectory(init, MAX_STEPS)
    curr = to_tensor(init.astype(np.float32)[None, None])
    for t in range(MAX_STEPS):
        with torch.no_grad():
            curr = (nca(curr) > 0.5).float()
        pred_np = curr.squeeze().cpu().numpy()
        divergences[seed_i, t] = (pred_np != true_traj[t+1]).mean() * 100

fig, ax = plt.subplots(figsize=(9,4))
for i in range(N_SEEDS):
    ax.plot(range(1, MAX_STEPS+1), divergences[i], alpha=0.4, lw=1,
            color="#C44E52")
ax.plot(range(1, MAX_STEPS+1), divergences.mean(0), lw=2.5,
        color="#C44E52", label="Mean across 10 grids")
ax.set_xlabel("Rollout step")
ax.set_ylabel("% cells wrong")
ax.set_title("Task 4 — NeuralCA rollout error accumulation", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "analysis_04_nca_rollout_divergence.png")

# ── Fig 05: Task 1 vs Task 4 accuracy ─────────────────────────────────────────

print("[05] Task 1 vs 4 comparison …")
states = traj_data["states"].astype(np.float32)
nexts  = traj_data["nexts"].astype(np.float32)
sample_idx = rng.choice(len(states), 2000, replace=False)

nca_correct = nsp_correct = total = 0
for i in sample_idx:
    x = to_tensor(states[i][None, None])
    y = nexts[i]
    with torch.no_grad():
        p_nca = (nca(x) > 0.5).squeeze().cpu().numpy()
        p_nsp = (nsp(x) > 0.5).squeeze().cpu().numpy()
    nca_correct += (p_nca == y).sum()
    nsp_correct += (p_nsp == y).sum()
    total       += y.size

fig, ax = plt.subplots(figsize=(6,5))
labels_bar = ["NeuralCA\n(2.7k params)", "NextState\n(111k params)"]
accs_bar   = [nca_correct/total*100, nsp_correct/total*100]
colors     = ["#4C72B0", "#55A868"]
bars = ax.bar(labels_bar, accs_bar, color=colors, edgecolor="white", width=0.4)
for bar, v in zip(bars, accs_bar):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(95, 100)
ax.set_ylabel("Cell accuracy (%)")
ax.set_title("Task 1 vs 4 — Next-state accuracy comparison", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
savefig(fig, "analysis_05_task1_vs_task4.png")

# ── Fig 06: Chaos predictor scatter ───────────────────────────────────────────

print("[06] Chaos scatter …")
from torch.utils.data import DataLoader, TensorDataset
grids_c  = chaos_data["grids"].astype(np.float32)
masks_c  = chaos_data["masks"].astype(np.float32)
scores_c = chaos_data["scores"].astype(np.float32)
sc_norm  = np.log1p(scores_c) / (np.log1p(scores_c).max() + 1e-6)

X_c = torch.tensor(np.stack([grids_c, masks_c], axis=1))
dl_c = DataLoader(TensorDataset(X_c, torch.tensor(sc_norm[:,None])),
                  batch_size=256, shuffle=False)

pred_c, true_c = [], []
with torch.no_grad():
    for xb, yb in dl_c:
        pred_c.append(chaos(xb.to(DEVICE)).squeeze().cpu().numpy())
        true_c.append(yb.squeeze().numpy())
pred_c = np.concatenate(pred_c)
true_c = np.concatenate(true_c)
r_chaos = np.corrcoef(pred_c, true_c)[0,1]

fig, ax = plt.subplots(figsize=(6,6))
ax.hexbin(true_c, pred_c, gridsize=50, cmap="YlOrRd", mincnt=1)
lim = max(true_c.max(), pred_c.max())
ax.plot([0,lim],[0,lim],"c--",lw=1.5,label="y=x")
ax.set_xlabel("True divergence score (log-normalised)")
ax.set_ylabel("Predicted score")
ax.set_title(f"Task 3 — Chaos predictor  (Pearson r = {r_chaos:.3f})",
             fontweight="bold")
ax.legend()
fig.tight_layout()
savefig(fig, "analysis_06_chaos_scatter.png")

# ── Fig 07: Rollout error vs k ────────────────────────────────────────────────

print("[07] Rollout error vs k …")
MAX_K   = 20
states_r = traj_data["trajs"].astype(np.float32)  # (n_inits, T+1, H, W)
n_inits  = min(50, states_r.shape[0])
bce_by_k = {k: [] for k in range(1, MAX_K+1)}

with torch.no_grad():
    for i in range(n_inits):
        for t in range(0, min(80, states_r.shape[1]-MAX_K), 10):
            init_g = to_tensor(states_r[i, t][None, None])
            for k in range(1, MAX_K+1):
                k_norm = to_tensor(np.array([k/MAX_K]))
                pred = rollout(init_g, k_norm).squeeze().cpu().numpy()
                true = states_r[i, t+k]
                bce  = -(true*np.log(pred+1e-7) +
                          (1-true)*np.log(1-pred+1e-7)).mean()
                bce_by_k[k].append(float(bce))

ks   = list(range(1, MAX_K+1))
mean_bce = [np.mean(bce_by_k[k]) for k in ks]
std_bce  = [np.std(bce_by_k[k])  for k in ks]

fig, ax = plt.subplots(figsize=(9,4))
ax.plot(ks, mean_bce, "o-", color="#8172B2", lw=2, label="Mean BCE")
ax.fill_between(ks,
                np.array(mean_bce)-np.array(std_bce),
                np.array(mean_bce)+np.array(std_bce),
                alpha=0.2, color="#8172B2")
ax.set_xlabel("Steps ahead  k")
ax.set_ylabel("BCE loss")
ax.set_title("Task 5 — Rollout predictor: error vs forecast horizon",
             fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "analysis_07_rollout_error_vs_k.png")

# ── Fig 08: Attractor difficulty ──────────────────────────────────────────────

print("[08] Attractor difficulty …")
from sklearn.metrics import confusion_matrix
grids_a, labels_a = augment(
    attr_data["grids"].astype(np.float32),
    attr_data["labels"].astype(np.int64)
)
from nn.utils import make_loaders
from nn.train_attractor import BATCH_SIZE as BS, SEED as S2
Xa = torch.tensor(np.ascontiguousarray(grids_a[:,None]))
Ya = torch.tensor(labels_a)
_, val_dl_a = make_loaders(Xa, Ya, batch_size=BS, seed=S2)

all_pred_a, all_true_a = [], []
with torch.no_grad():
    for xb, yb in val_dl_a:
        logits = fate_m(xb.to(DEVICE))
        all_pred_a.extend(logits.argmax(1).cpu().numpy())
        all_true_a.extend(yb.numpy())

present = sorted(set(all_true_a)|set(all_pred_a))
cm_a = confusion_matrix(all_true_a, all_pred_a, labels=present)
pnames_a = [LABEL_NAMES[i] for i in present]
acc_a = (np.array(all_pred_a)==np.array(all_true_a)).mean()*100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im = axes[0].imshow(cm_a, cmap="Blues")
axes[0].set_xticks(range(len(present))); axes[0].set_xticklabels(pnames_a, rotation=30, ha="right")
axes[0].set_yticks(range(len(present))); axes[0].set_yticklabels(pnames_a)
plt.colorbar(im, ax=axes[0])
for ii in range(len(present)):
    for jj in range(len(present)):
        axes[0].text(jj, ii, str(cm_a[ii,jj]), ha="center", va="center",
                     color="white" if cm_a[ii,jj]>cm_a.max()/2 else "black", fontsize=9)
axes[0].set_title(f"Confusion matrix  (acc={acc_a:.1f}%)", fontweight="bold")

counts = np.bincount(attr_data["labels"], minlength=4)
axes[1].bar([LABEL_NAMES[i] for i in range(4)], counts, color="#4C72B0", edgecolor="white")
axes[1].set_ylabel("Count")
axes[1].set_title("Class distribution (raw data)", fontweight="bold")
axes[1].set_xlabel("Fate")

fig.suptitle("Task 6 — Fate classifier  (GoL fate is hard to predict!)",
             fontweight="bold", fontsize=13)
fig.tight_layout()
savefig(fig, "analysis_08_attractor_difficulty.png")

# ── Fig 09: Summary dashboard ─────────────────────────────────────────────────

print("[09] Summary dashboard …")
history = {}
for task in ["task1_next_state","task2_sensitivity","task3_chaos",
             "task4_neural_ca","task5_rollout","task6_attractor"]:
    p = os.path.join(RESULTS_DIR, f"{task}_history.json")
    if os.path.exists(p):
        with open(p) as f:
            history[task] = json.load(f)

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

titles = {
    "task1_next_state":   "Task 1 — Next-state predictor",
    "task2_sensitivity":  "Task 2 — Sensitivity U-Net",
    "task3_chaos":        "Task 3 — Chaos predictor",
    "task4_neural_ca":    "Task 4 — NeuralCA",
    "task5_rollout":      "Task 5 — Rollout predictor",
    "task6_attractor":    "Task 6 — Fate classifier",
}
colors_t = {"train_loss": "#4C72B0", "val_loss": "#C44E52"}
findings = {
    "task1_next_state":  f"Val BCE: {min(history.get('task1_next_state',{}).get('val_loss',[0])):.4f}",
    "task2_sensitivity": f"Pearson r = {corr:.3f}",
    "task3_chaos":       f"Pearson r = {r_chaos:.3f}",
    "task4_neural_ca":   f"Cell acc: {nca_correct/total*100:.2f}%",
    "task5_rollout":     f"BCE at k=1: {mean_bce[0]:.4f}  k={MAX_K}: {mean_bce[-1]:.4f}",
    "task6_attractor":   f"Val acc: {acc_a:.1f}%  (hard—fate ≈ unpredictable)",
}

for idx, (task, title) in enumerate(titles.items()):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    if task in history:
        h = history[task]
        ep = range(1, len(h["train_loss"])+1)
        ax.plot(ep, h["train_loss"], color=colors_t["train_loss"], lw=1.5, label="train")
        ax.plot(ep, h["val_loss"],   color=colors_t["val_loss"],   lw=1.5, label="val")
        ax.legend(fontsize=7)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss",  fontsize=8)
    ax.grid(alpha=0.25)
    ax.text(0.98, 0.98, findings.get(task,""), transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

fig.suptitle("Game of Life — Neural Network Experiments Summary",
             fontsize=16, fontweight="bold")
savefig(fig, "analysis_09_summary_dashboard.png")

print("\nAll analysis figures saved to nn/results/")
