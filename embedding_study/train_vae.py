"""
Train a convolutional VAE on the Game of Life image dataset.

Saves three separate model files under embedding_study/models/:
  encoder.pt   — Encoder state_dict
  decoder.pt   — Decoder state_dict
  vae.pt       — full VAE state_dict
plus config.json (hyperparameters needed to rebuild the modules),
loss_history.json, reconstruction previews, and embeddings.npz
(the mu vector for every image in the dataset).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from vae import VAE

HERE = os.path.dirname(os.path.abspath(__file__))


def loss_fn(logits, x, mu, logvar):
    """Per-sample mean of (reconstruction BCE + KL)."""
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return bce, kl


@torch.no_grad()
def save_recon_grid(model, x, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    logits, _, _ = model(x)
    recon = torch.sigmoid(logits)
    n = x.size(0)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 2.8))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].cpu(), cmap="binary", interpolation="nearest")
        axes[1, i].imshow(recon[i, 0].cpu(), cmap="binary", interpolation="nearest")
        for ax in (axes[0, i], axes[1, i]):
            ax.axis("off")
    axes[0, 0].set_title("input", fontsize=8, loc="left")
    axes[1, 0].set_title("recon", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def compute_embeddings(model, images, device, batch=512):
    model.eval()
    mus = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device)
        mu, _ = model.encoder(x)
        mus.append(mu.cpu())
    return torch.cat(mus).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "gol_images.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--latent", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(HERE, "models"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--smoke", action="store_true",
                    help="quick sanity run: few batches, saves to a temp subdir")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    d = np.load(args.data)
    images = torch.from_numpy(d["images"]).float().unsqueeze(1)  # (N,1,64,64)
    category = d["category"]
    n = len(images)
    print(f"Loaded {n:,} images from {args.data}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(0.05 * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(TensorDataset(images[train_idx]), batch_size=args.batch,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(images[val_idx]), batch_size=args.batch,
                            num_workers=2, pin_memory=True)

    model = VAE(args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.smoke:
        args.epochs = 1
        args.out_dir = os.path.join(args.out_dir, "smoke")
        args.results_dir = os.path.join(args.results_dir, "smoke")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    history = []
    vis_batch = images[val_idx[:8]].to(device)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_bce = tr_kl = tr_n = 0.0
        for step, (x,) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            logits, mu, logvar = model(x)
            bce, kl = loss_fn(logits, x, mu, logvar)
            loss = bce + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_bce += bce.item() * x.size(0)
            tr_kl += kl.item() * x.size(0)
            tr_n += x.size(0)
            if args.smoke and step >= 4:
                break

        model.eval()
        va_bce = va_kl = va_n = 0.0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device, non_blocking=True)
                logits, mu, logvar = model(x)
                bce, kl = loss_fn(logits, x, mu, logvar)
                va_bce += bce.item() * x.size(0)
                va_kl += kl.item() * x.size(0)
                va_n += x.size(0)

        rec = dict(epoch=epoch,
                   train_bce=tr_bce / tr_n, train_kl=tr_kl / tr_n,
                   val_bce=va_bce / va_n, val_kl=va_kl / va_n,
                   elapsed_s=round(time.time() - t0, 1))
        history.append(rec)
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train bce={rec['train_bce']:8.2f} kl={rec['train_kl']:6.2f}  "
              f"val bce={rec['val_bce']:8.2f} kl={rec['val_kl']:6.2f}  "
              f"[{rec['elapsed_s']:.0f}s]", flush=True)

        if epoch % 5 == 0 or epoch == args.epochs:
            save_recon_grid(model, vis_batch,
                            os.path.join(args.results_dir, f"recon_epoch{epoch:03d}.png"))

    # ── Save the three models individually ───────────────────────────────────
    torch.save(model.encoder.state_dict(), os.path.join(args.out_dir, "encoder.pt"))
    torch.save(model.decoder.state_dict(), os.path.join(args.out_dir, "decoder.pt"))
    torch.save(model.state_dict(), os.path.join(args.out_dir, "vae.pt"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(dict(latent_dim=args.latent, image_size=64, epochs=args.epochs,
                       batch=args.batch, lr=args.lr, seed=args.seed,
                       final=history[-1]), f, indent=2)
    print(f"Saved encoder.pt / decoder.pt / vae.pt → {args.out_dir}")

    with open(os.path.join(args.results_dir, "loss_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if not args.smoke:
        emb = compute_embeddings(model, images, device)
        np.savez_compressed(os.path.join(args.results_dir, "embeddings.npz"),
                            mu=emb, category=category,
                            categories=d["categories"],
                            pattern_counts=d["pattern_counts"],
                            pattern_names=d["pattern_names"])
        print(f"Saved embeddings ({emb.shape}) → "
              f"{os.path.join(args.results_dir, 'embeddings.npz')}")

    print("Done.")


if __name__ == "__main__":
    main()
