"""Train Model 1 on GoL t->t+1 prediction.

Data: one named pattern per grid, random D4 orientation + random placement
(see data.py). Loss: plain BCE-with-logits (no symmetry losses needed --
D4 + translation equivariance is architectural, verified in sanity_check.py,
not something the model needs to learn).

Run:
    python train.py --trial          # timing + sanity on a few steps
    python train.py --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "v1"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from net import ToyCNNModel1  # noqa: E402
from data import make_batch, make_fixed_set  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)


@torch.no_grad()
def evaluate(model, x, y):
    logits = model(x)
    pred = (logits >= 0).float()
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y)).sum())
    fn = float(((1 - pred) * y).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    cell_acc = float((pred == y).float().mean())
    return dict(f1=f1, prec=prec, rec=rec, cell_acc=cell_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-per-epoch", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--m", type=int, default=4, help="hidden channel width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-n", type=int, default=1024)
    args = ap.parse_args()

    # CPU, deliberately: the model is ~100 params and canonical_gidx() does a
    # per-sample CPU/numpy round-trip internally regardless of device, so MPS
    # dispatch overhead dominates and makes it ~10x SLOWER than CPU here
    # (measured: 271ms/step on MPS vs 25ms/step on CPU for this model).
    device = "cpu"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ToyCNNModel1(m=args.m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_x, val_y = make_fixed_set(args.val_n, seed=999, device=device)

    print(f"device={device}  params={n_params}  m={args.m}  lr={args.lr}")

    steps_per_epoch = max(1, args.n_per_epoch // args.batch)

    if args.trial:
        n = 30
        t0 = time.time()
        for i in range(n):
            x, y = make_batch(args.batch, rng, device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / n
        print(f"[trial] {dt*1000:.0f} ms/step -> {dt*steps_per_epoch/60:.2f} min/epoch "
              f"-> {dt*steps_per_epoch*args.epochs/3600:.2f} h for {args.epochs} epochs")
        ev = evaluate(model, val_x, val_y)
        print(f"[trial] quick eval (undertrained): {ev}")
        return

    log_path = os.path.join(RESULTS, "train_log.txt")
    hist = []
    best_f1 = -1.0
    with open(log_path, "w") as f:
        f.write(f"device={device} params={n_params} m={args.m} lr={args.lr}\n")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        run_loss = 0.0
        for _ in range(steps_per_epoch):
            x, y = make_batch(args.batch, rng, device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss)
        sched.step()
        run_loss /= steps_per_epoch
        ev = evaluate(model, val_x, val_y)
        row = dict(epoch=ep, train_loss=run_loss, **ev, sec=time.time() - t0)
        hist.append(row)
        line = (f"ep {ep:3d}/{args.epochs}  loss={run_loss:.4f}  "
                f"F1={ev['f1']:.4f} prec={ev['prec']:.4f} rec={ev['rec']:.4f} "
                f"cell_acc={ev['cell_acc']:.5f}  {row['sec']:.1f}s")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if ev["f1"] > best_f1:
            best_f1 = ev["f1"]
            torch.save(dict(model=model.state_dict(), epoch=ep, metrics=ev, args=vars(args)),
                       os.path.join(CKPT, "best.pt"))
        with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
            json.dump(hist, f, indent=2)

    torch.save(dict(model=model.state_dict(), epoch=args.epochs, metrics=hist[-1], args=vars(args)),
               os.path.join(CKPT, "final.pt"))
    print(f"best F1={best_f1:.4f}  -> {CKPT}/best.pt  (final also saved)")


if __name__ == "__main__":
    main()
