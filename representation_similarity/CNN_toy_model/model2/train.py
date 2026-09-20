"""Train Model 2 on GoL t->t+1 prediction.

Applies both lessons from Model 1 from the START (no need to repeat that
two-step discovery): mixed random-density + single-pattern training data,
and scheduled sampling for multi-step rollout stability. Warmstarts from
Model 1 v2's checkpoint -- the learnable layers (iso1, conv2, act1, act2,
head) are identically shaped between the two models; only the
canonicalization mechanism changed, so the weights transfer directly.

Run:
    python train.py --trial
    python train.py --epochs 60
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
from net import ToyCNNModel2  # noqa: E402
from data import make_mixed_batch, make_random_batch, make_fixed_set  # noqa: E402
from adapter import gol_step_torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

MODEL1_CKPT = os.path.join(_HERE, "..", "model1", "checkpoints", "best_v2.pt")


def true_trajectory(x0: torch.Tensor, k: int) -> list[torch.Tensor]:
    ys = []
    cur = x0
    for _ in range(k):
        cur = gol_step_torch(cur)
        ys.append(cur)
    return ys


def scheduled_sampling_loss(model, x0: torch.Tensor, k: int, p: float):
    y_true = true_trajectory(x0, k)
    cur_input = x0
    total = 0.0
    for step in range(k):
        logits = model(cur_input)
        total = total + F.binary_cross_entropy_with_logits(logits, y_true[step])
        with torch.no_grad():
            pred_hard = (logits >= 0).float()
            mask = (torch.rand(x0.shape[0], 1, 1, 1, device=x0.device) < p).float()
            cur_input = mask * pred_hard + (1 - mask) * y_true[step]
    return total / k


@torch.no_grad()
def evaluate_1step(model, x, y):
    logits = model(x)
    pred = (logits >= 0).float()
    tp = float((pred * y).sum())
    fp = float((pred * (1 - y)).sum())
    fn = float(((1 - pred) * y).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(f1=f1, prec=prec, rec=rec)


@torch.no_grad()
def evaluate_rollout(model, x0: torch.Tensor, steps: int) -> float:
    x_true, x_model = x0, x0
    for _ in range(steps):
        x_true = gol_step_torch(x_true)
        x_model = model.step(x_model)
    return float((x_true != x_model).float().sum(dim=(1, 2, 3)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-epoch", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--warmup-epochs", type=int, default=40)
    ap.add_argument("--random-frac", type=float, default=0.5)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmstart", default=MODEL1_CKPT)
    args = ap.parse_args()

    device = "cpu"  # see model1/notes.md: canonicalization's CPU/numpy round-trip makes MPS slower
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ToyCNNModel2(m=args.m).to(device)
    if args.warmstart and os.path.exists(args.warmstart):
        ckpt = torch.load(args.warmstart, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"warmstarted from {args.warmstart} (epoch {ckpt.get('epoch')}); "
              f"missing={missing} unexpected={unexpected}")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_x, val_y = make_fixed_set(1024, seed=999, device=device)
    val_rand_x, val_rand_y = make_random_batch(1024, np.random.default_rng(998), device)
    rollout_x0, _ = make_random_batch(64, np.random.default_rng(997), device)

    print(f"device={device}  params={n_params}  m={args.m}  lr={args.lr}  "
          f"k={args.k}  warmup_epochs={args.warmup_epochs}  random_frac={args.random_frac}")

    steps_per_epoch = max(1, args.n_per_epoch // args.batch)

    if args.trial:
        n = 20
        t0 = time.time()
        for i in range(n):
            x0, _ = make_mixed_batch(args.batch, rng, device, args.random_frac)
            loss = scheduled_sampling_loss(model, x0, args.k, p=0.5)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = (time.time() - t0) / n
        print(f"[trial] {dt*1000:.0f} ms/step -> {dt*steps_per_epoch/60:.2f} min/epoch "
              f"-> {dt*steps_per_epoch*args.epochs/3600:.2f} h for {args.epochs} epochs")
        return

    log_path = os.path.join(RESULTS, "train_log.txt")
    hist = []
    best_score = -1e9
    with open(log_path, "w") as f:
        f.write(f"device={device} params={n_params} lr={args.lr} k={args.k}\n")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        p = min(1.0, ep / max(1, args.warmup_epochs))
        run_loss = 0.0
        for _ in range(steps_per_epoch):
            x0, _ = make_mixed_batch(args.batch, rng, device, args.random_frac)
            loss = scheduled_sampling_loss(model, x0, args.k, p)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss)
        sched.step()
        run_loss /= steps_per_epoch

        ev_pattern = evaluate_1step(model, val_x, val_y)
        ev_random = evaluate_1step(model, val_rand_x, val_rand_y)
        roll10 = evaluate_rollout(model, rollout_x0, steps=10)
        roll20 = evaluate_rollout(model, rollout_x0, steps=20)

        row = dict(epoch=ep, p=p, train_loss=run_loss,
                   pattern_f1=ev_pattern["f1"], pattern_rec=ev_pattern["rec"],
                   random_f1=ev_random["f1"], random_rec=ev_random["rec"],
                   rollout_diff10=roll10, rollout_diff20=roll20,
                   sec=time.time() - t0)
        hist.append(row)
        line = (f"ep {ep:3d}/{args.epochs}  p={p:.2f}  loss={run_loss:.4f}  "
                f"pattern F1={ev_pattern['f1']:.4f}  "
                f"random F1={ev_random['f1']:.4f} rec={ev_random['rec']:.4f}  "
                f"rollout_diff@10={roll10:.1f} @20={roll20:.1f}  {row['sec']:.1f}s")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        f1_ok = ev_pattern["f1"] >= 0.999
        score = (0.0 if f1_ok else -10.0) + ev_random["rec"] - 0.001 * roll20
        if score > best_score:
            best_score = score
            torch.save(dict(model=model.state_dict(), epoch=ep, metrics=row, args=vars(args)),
                       os.path.join(CKPT, "best.pt"))
        with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
            json.dump(hist, f, indent=2)

    torch.save(dict(model=model.state_dict(), epoch=args.epochs, metrics=hist[-1], args=vars(args)),
               os.path.join(CKPT, "final.pt"))
    print(f"best score={best_score:.4f}  -> {CKPT}/best.pt  (final also saved)")


if __name__ == "__main__":
    main()
