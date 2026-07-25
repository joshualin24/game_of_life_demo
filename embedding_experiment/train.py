"""
Train the trajectory-invariant encoder with NT-Xent contrastive loss.

Positive pairs: two randomly sampled frames from the SAME trajectory.
Negative pairs: frames from all other trajectories in the batch.

Usage:
    python train.py [--epochs 50] [--batch 256] [--latent 64] [--tau 0.07]
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import TrajectoryEncoder

HERE = os.path.dirname(os.path.abspath(__file__))


# ── Dataset ───────────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Each __getitem__ returns two randomly sampled frames from the same
    trajectory (anchor + positive). The DataLoader batch forms the negative set.
    """
    def __init__(self, frames: np.ndarray, fate: np.ndarray):
        # frames: (N, T, H, W) uint8
        self.frames = torch.from_numpy(frames).float()  # (N, T, H, W)
        self.fate = torch.from_numpy(fate).long()
        self.N, self.T = frames.shape[:2]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        i, j = np.random.choice(self.T, size=2, replace=False)
        fi = self.frames[idx, i].unsqueeze(0)  # (1, H, W)
        fj = self.frames[idx, j].unsqueeze(0)
        return fi, fj, self.fate[idx]


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(z_i, z_j, tau: float = 0.07):
    """
    NT-Xent (normalized temperature-scaled cross-entropy) loss.
    z_i, z_j: (B, D) unit-norm projections from anchor and positive.
    Negatives are all other pairs within the batch.
    """
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)           # (2B, D)
    sim = torch.mm(z, z.t()) / tau              # (2B, 2B)

    # mask self-similarities
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    # positive indices: (i, i+B) and (i+B, i)
    targets = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, targets)


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "trajectories.npz"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d = np.load(args.data)
    frames = d["frames"]   # (N, T, H, W)
    fate   = d["fate"]
    N = len(frames)
    print(f"Loaded {N:,} trajectories × {frames.shape[1]} steps")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(N)
    n_val = max(1, int(0.05 * N))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TrajectoryDataset(frames[train_idx], fate[train_idx])
    val_ds   = TrajectoryDataset(frames[val_idx],   fate[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=True)

    model = TrajectoryEncoder(args.latent, args.proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_n = 0
        for fi, fj, _ in train_loader:
            fi, fj = fi.to(device), fj.to(device)
            zi = model(fi)
            zj = model(fj)
            loss = nt_xent_loss(zi, zj, args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * fi.size(0)
            tr_n += fi.size(0)
        scheduler.step()

        model.eval()
        va_loss = va_n = 0
        with torch.no_grad():
            for fi, fj, _ in val_loader:
                fi, fj = fi.to(device), fj.to(device)
                loss = nt_xent_loss(model(fi), model(fj), args.tau)
                va_loss += loss.item() * fi.size(0)
                va_n += fi.size(0)

        rec = dict(epoch=epoch,
                   train_loss=tr_loss / tr_n,
                   val_loss=va_loss / va_n,
                   elapsed_s=round(time.time() - t0, 1))
        history.append(rec)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train={rec['train_loss']:.4f}  val={rec['val_loss']:.4f}  "
              f"[{rec['elapsed_s']:.0f}s]", flush=True)

    # Save
    ckpt = os.path.join(args.out_dir, "encoder.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved checkpoint → {ckpt}")

    cfg = dict(latent_dim=args.latent, proj_dim=args.proj_dim,
               tau=args.tau, lr=args.lr, epochs=args.epochs,
               batch=args.batch, seed=args.seed, final=history[-1])
    with open(os.path.join(args.out_dir, "train_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Training complete.")


if __name__ == "__main__":
    main()
