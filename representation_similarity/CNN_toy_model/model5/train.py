"""Train Model 5: learn canonicalization instead of hand-coding it,
distilling from Model 4 (frozen teacher).

Three losses per step:
  L_task -- BCE on the actual t+1 prediction, end to end (the real task).
  L_feat -- MSE between Part A's feat_can (canonicalized by the CURRENT
            pose -- teacher's, self-predicted, or a scheduled-sampling mix)
            and Model 4's feat_can for the same input.
  L_pose -- direct supervision for Part B: smooth-L1 on the shift
            regression + cross-entropy on the 8-way rotation classification,
            both against Model 4's actual computed (shift, gidx).

Scheduled sampling (same recipe used throughout this project for exposure
bias): which pose (teacher's or the model's own) is used to compute
L_task/L_feat is teacher-forced early in training (p_self ramps 0->1), so
Part A first learns to use a RELIABLE pose signal before being exposed to
Part B's own, initially-bad predictions.

Warmstarts Part A + head from Model 4's checkpoint (identical layer
shapes) -- Part A's job (predict next-state from an approximately-
canonical input) is close to what Model 4's backbone already learned; only
Part B (new architecture) starts from scratch.

Run:
    python train.py --trial
    python train.py --epochs 60
"""
from __future__ import annotations

import argparse
import importlib.util
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
from net import ToyCNNModel5, _D4_INV_T  # noqa: E402
from data import make_mixed_batch, make_random_batch, make_fixed_set  # noqa: E402
from adapter import gol_step_torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

MODEL4_DIR = os.path.join(_HERE, "..", "model4")


def _load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


net4 = _load_module(os.path.join("..", "model4", "net.py"), "net4_teacher")


def load_teacher():
    model = net4.ToyCNNModel4(m=4)
    ckpt = torch.load(os.path.join(MODEL4_DIR, "checkpoints", "best.pt"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_step(model, teacher, opt, x, y, p_self, lambda_feat, lambda_pose, device):
    with torch.no_grad():
        gidx_t, shift_t = net4.canonical_transform(x)
        _, aux_teacher = teacher(x, return_features=True)
        feat_can_teacher = aux_teacher["feat_can"]

    feat, shift_pred, gidx_logits = model.encode(x)

    with torch.no_grad():
        shift_self = torch.round(shift_pred).long()
        gidx_self = gidx_logits.argmax(dim=1)
        use_self = (torch.rand(x.shape[0], device=device) < p_self)
        shift_used = torch.where(use_self.unsqueeze(1), shift_self, shift_t)
        gidx_used = torch.where(use_self, gidx_self, gidx_t)

    logits, feat_can = model.combine(feat, shift_used, gidx_used)

    L_task = F.binary_cross_entropy_with_logits(logits, y)
    L_feat = F.mse_loss(feat_can, feat_can_teacher)
    L_shift = F.smooth_l1_loss(shift_pred, shift_t.float())
    L_gidx = F.cross_entropy(gidx_logits, gidx_t)
    L_pose = L_shift + L_gidx

    total = L_task + lambda_feat * L_feat + lambda_pose * L_pose
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    return dict(total=float(total), task=float(L_task), feat=float(L_feat),
                shift=float(L_shift), gidx=float(L_gidx))


@torch.no_grad()
def evaluate(model, teacher, x, y, device):
    """Self-contained evaluation: model.forward(x) uses ONLY the model's
    own predicted pose, no teacher, no hand-coded canonicalization."""
    logits, aux = model(x, return_aux=True)
    pred = (logits >= 0).float()
    tp = float((pred * y).sum()); fp = float((pred * (1 - y)).sum()); fn = float(((1 - pred) * y).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    gidx_t, shift_t = net4.canonical_transform(x)
    gidx_acc = float((aux["gidx_used"] == gidx_t).float().mean())
    shift_mae = float((aux["shift_used"].float() - shift_t.float()).abs().mean())

    _, aux_teacher = teacher(x, return_features=True)
    fc, fct = aux["feat_can"].flatten(1), aux_teacher["feat_can"].flatten(1)
    cos = float(((fc * fct).sum(1) / (fc.norm(dim=1) * fct.norm(dim=1) + 1e-12)).mean())

    return dict(f1=f1, prec=prec, rec=rec, gidx_acc=gidx_acc, shift_mae=shift_mae, feat_can_cos=cos)


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
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-per-epoch", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-epochs", type=int, default=50, help="epochs to ramp p_self 0->1")
    ap.add_argument("--random-frac", type=float, default=0.5)
    ap.add_argument("--lambda-feat", type=float, default=0.05)
    ap.add_argument("--lambda-pose", type=float, default=1.0)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmstart", default=os.path.join(MODEL4_DIR, "checkpoints", "best.pt"))
    args = ap.parse_args()

    device = "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = ToyCNNModel5(m=args.m).to(device)
    if args.warmstart and os.path.exists(args.warmstart):
        ckpt = torch.load(args.warmstart, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"warmstarted Part A from {args.warmstart}; missing={missing} unexpected={unexpected}")
    teacher = load_teacher().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_a = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("pose."))
    n_params_b = sum(p.numel() for n, p in model.named_parameters() if n.startswith("pose."))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val_x, val_y = make_fixed_set(1024, seed=999, device=device)
    val_rand_x, val_rand_y = make_random_batch(1024, np.random.default_rng(998), device)
    rollout_x0, _ = make_random_batch(64, np.random.default_rng(997), device)

    print(f"device={device}  params={n_params} (A+head={n_params_a}, B/pose={n_params_b})  "
          f"lr={args.lr}  warmup_epochs={args.warmup_epochs}  "
          f"lambda_feat={args.lambda_feat} lambda_pose={args.lambda_pose}")

    steps_per_epoch = max(1, args.n_per_epoch // args.batch)

    if args.trial:
        n = 20
        t0 = time.time()
        for i in range(n):
            x0, _ = make_mixed_batch(args.batch, rng, device, args.random_frac)
            y0 = gol_step_torch(x0)
            logs = train_step(model, teacher, opt, x0, y0, p_self=0.3,
                               lambda_feat=args.lambda_feat, lambda_pose=args.lambda_pose, device=device)
        dt = (time.time() - t0) / n
        print(f"[trial] {dt*1000:.0f} ms/step -> {dt*steps_per_epoch/60:.2f} min/epoch "
              f"-> {dt*steps_per_epoch*args.epochs/3600:.2f} h for {args.epochs} epochs")
        print(f"[trial] last step losses: {logs}")
        ev = evaluate(model, teacher, val_x[:256], val_y[:256], device)
        print(f"[trial] quick eval (undertrained): {ev}")
        return

    log_path = os.path.join(RESULTS, "train_log.txt")
    hist = []
    best_score = -1e9
    with open(log_path, "w") as f:
        f.write(f"device={device} params={n_params} (A={n_params_a} B={n_params_b}) lr={args.lr}\n")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        p_self = min(1.0, ep / max(1, args.warmup_epochs))
        acc = {}
        for _ in range(steps_per_epoch):
            x0, _ = make_mixed_batch(args.batch, rng, device, args.random_frac)
            y0 = gol_step_torch(x0)
            logs = train_step(model, teacher, opt, x0, y0, p_self,
                               args.lambda_feat, args.lambda_pose, device)
            for kk, vv in logs.items():
                acc[kk] = acc.get(kk, 0.0) + vv
        acc = {kk: vv / steps_per_epoch for kk, vv in acc.items()}
        sched.step()

        ev_pattern = evaluate(model, teacher, val_x, val_y, device)
        ev_random = evaluate(model, teacher, val_rand_x, val_rand_y, device)
        roll20 = evaluate_rollout(model, rollout_x0, steps=20)

        row = dict(epoch=ep, p_self=p_self, **{f"tr_{k}": v for k, v in acc.items()},
                   pattern_f1=ev_pattern["f1"], pattern_gidx_acc=ev_pattern["gidx_acc"],
                   pattern_shift_mae=ev_pattern["shift_mae"], pattern_feat_cos=ev_pattern["feat_can_cos"],
                   random_f1=ev_random["f1"], random_gidx_acc=ev_random["gidx_acc"],
                   random_shift_mae=ev_random["shift_mae"], random_feat_cos=ev_random["feat_can_cos"],
                   rollout_diff20=roll20, sec=time.time() - t0)
        hist.append(row)
        line = (f"ep {ep:3d}/{args.epochs}  p_self={p_self:.2f}  "
                f"loss(task={acc['task']:.4f} feat={acc['feat']:.4f} gidx={acc['gidx']:.4f} shift={acc['shift']:.4f})  "
                f"pattern F1={ev_pattern['f1']:.4f} gidx_acc={ev_pattern['gidx_acc']:.3f} "
                f"shift_mae={ev_pattern['shift_mae']:.3f} feat_cos={ev_pattern['feat_can_cos']:.4f}  "
                f"random F1={ev_random['f1']:.4f} gidx_acc={ev_random['gidx_acc']:.3f} "
                f"feat_cos={ev_random['feat_can_cos']:.4f}  rollout_diff@20={roll20:.1f}  {row['sec']:.1f}s")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        score = ev_pattern["f1"] + ev_random["f1"] + ev_pattern["gidx_acc"] + ev_random["gidx_acc"]
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
