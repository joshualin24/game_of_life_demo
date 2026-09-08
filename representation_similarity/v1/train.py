"""v1 training — hybrid symmetry adapter on frozen V8 tf_L4 ("Design A").

Target (pooled readout psi = mean_tok(adapter(tf_L4))):
  translations  psi(t.x) - psi(x) ~= delta(t)      (content-independent displacement)
  D4            psi(r.x)          ~= rho(r) psi(x)  (learned linear operator)

Design A: adapter is a pure residual with zero-init blocks (z == tf_L4 at init),
decoder defaults to V8's FROZEN patch_head -> F1 starts at ~0.998 and the run
traces the F1-vs-symmetry frontier as the symmetry lambda warms up. A
discriminability regularizer keeps E||psi(x)-psi(x')|| near its raw-V8 value so
"invariance" can't be earned by shrinking psi. Cosine LR decay.

  python train.py --trial
  python train.py --epochs 30

Outputs: results/train_log.txt, results/metrics.json, checkpoints/best.pt
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
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from load_models import load_v8, DEVICE
from adapter import (Adapter, Decoder, SymParams, v8_tfL4, CAYLEY,
                          torch_translate, torch_d4, gol_step_torch, N_D4)
from symmetry import subpatch_offsets, fourcell_offsets, translate, d4_ops, stage_names

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CKPT = os.path.join(HERE, "checkpoints")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

DENSITIES = (0.12, 0.2, 0.3, 0.4, 0.5, 0.62)


def make_batch(bs, rng, dev):
    d = rng.choice(DENSITIES, size=bs)
    x = (rng.random((bs, 40, 40)) < d[:, None, None]).astype(np.float32)
    xt = torch.from_numpy(x).unsqueeze(1).to(dev)          # (bs,1,40,40)
    return xt, gol_step_torch(xt)


# ── one training step ─────────────────────────────────────────────────────
def train_step(v8, adapter, decoder, sym, opt, batch, k, k_pred, lam, dev, rng,
               lam_scale=1.0, unrel_target=1.0):
    x, y = batch
    B = x.shape[0]
    tok = v8_tfL4(v8, x)
    z = adapter(tok)
    psi = z.mean(1)                                        # (B,64)
    scale = psi.pow(2).mean().detach() + 1e-6

    L_pred = F.binary_cross_entropy_with_logits(decoder(z), y)

    # discriminability: unrelated grids must stay ~as far apart as in raw V8
    perm = torch.randperm(B, device=dev)
    unrel = (psi - psi[perm]).norm(dim=1).mean()
    L_disc = F.relu(unrel_target - unrel).pow(2) / (unrel_target ** 2 + 1e-9)

    res_trans, res_d4, res_mixed = [], [], []
    L_pred_aug = torch.zeros((), device=dev)
    n_aug = 0
    for i in range(k):
        kind = ("trans", "d4", "mixed")[i % 3]
        r = int(rng.integers(1, N_D4)) if kind != "trans" else 0
        if kind == "d4":
            dr = dc = 0
        elif rng.random() < 0.65:
            # concentrate on the 16 sub-patch phases — the actual target
            dr = int(rng.integers(0, 4)); dc = int(rng.integers(0, 4))
        else:
            dr = int(rng.integers(0, 40)); dc = int(rng.integers(0, 40))
        xg = x
        if r:
            xg = torch_d4(xg, r)
        if dr or dc:
            xg = torch_translate(xg, dr, dc)
        zg = adapter(v8_tfL4(v8, xg))
        psig = zg.mean(1)

        tgt = psi
        if r:
            tgt = psi @ sym.rho(r, dev).T
        if dr or dc:
            tgt = tgt + sym.delta(torch.tensor(dr), torch.tensor(dc), dev)
        res = ((psig - tgt).pow(2).mean() / scale)
        (res_trans if kind == "trans" else res_d4 if kind == "d4" else res_mixed).append(res)

        if i < k_pred:
            yg = y
            if r:
                yg = torch_d4(yg, r)
            if dr or dc:
                yg = torch_translate(yg, dr, dc)
            L_pred_aug = L_pred_aug + F.binary_cross_entropy_with_logits(decoder(zg), yg)
            n_aug += 1
    if n_aug:
        L_pred_aug = L_pred_aug / n_aug

    def mean0(xs):
        return torch.stack(xs).mean() if xs else torch.zeros((), device=dev)
    L_trans, L_d4, L_mixed = mean0(res_trans), mean0(res_d4), mean0(res_mixed)

    # D4 group-law consistency on the learned operators
    rho_all = sym.rho_all(dev)
    ii = rng.integers(1, N_D4, size=6)
    jj = rng.integers(1, N_D4, size=6)
    gl = 0.0
    for a, b in zip(ii, jj):
        gl = gl + (rho_all[a] @ rho_all[b] - rho_all[CAYLEY[a, b]]).pow(2).mean()
    L_glaw = gl / len(ii)

    total = (L_pred + lam["pred_aug"] * L_pred_aug + lam["disc"] * L_disc
             + lam_scale * (lam["trans"] * L_trans + lam["d4"] * L_d4
                            + lam["mixed"] * L_mixed + lam["glaw"] * L_glaw))

    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], 2.0)
    opt.step()
    return dict(total=float(total), pred=float(L_pred), pred_aug=float(L_pred_aug),
                disc=float(L_disc), unrel=float(unrel),
                trans=float(L_trans), d4=float(L_d4), mixed=float(L_mixed),
                glaw=float(L_glaw))


# ── evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def adapter_psi_np(v8, adapter, grids_np, dev, bs=256):
    out = []
    for s in range(0, len(grids_np), bs):
        x = torch.from_numpy(grids_np[s:s + bs].astype(np.float32)).unsqueeze(1).to(dev)
        z = adapter(v8_tfL4(v8, x))
        out.append(z.mean(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def evaluate(v8, adapter, decoder, sym, val_np, dev):
    # accuracy
    x = torch.from_numpy(val_np.astype(np.float32)).unsqueeze(1).to(dev)
    y = gol_step_torch(x)
    tp = fp = fn = 0
    for s in range(0, x.shape[0], 512):
        z = adapter(v8_tfL4(v8, x[s:s + 512]))
        pred = (decoder(z) >= 0).float()
        yt = y[s:s + 512]
        tp += float((pred * yt).sum()); fp += float((pred * (1 - yt)).sum())
        fn += float(((1 - pred) * yt).sum())
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    # symmetry metrics on pooled psi
    def emb_fn(g):
        return adapter_psi_np(v8, adapter, g, dev)
    base = emb_fn(val_np)

    def _u(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

    base_norm = np.linalg.norm(base, axis=1).mean() + 1e-12

    def transl_relmove_dir(offsets):
        rm, dir_c = [], []
        for (dr, dc) in offsets:
            d = emb_fn(translate(val_np, dr, dc)) - base
            rm.append(float(np.linalg.norm(d, axis=1).mean() / base_norm))
            dbar = d.mean(0, keepdims=True)
            dir_c.append(float((_u(d) * _u(dbar)).sum(1).mean()))
        return float(np.mean(rm)), float(np.mean(dir_c))

    subpatch_relmove, subpatch_dir = transl_relmove_dir(subpatch_offsets())
    fourcell_relmove, _ = transl_relmove_dir(fourcell_offsets(np.random.default_rng(0), 16))

    # discriminability: distance between unrelated grids (matched-ish density)
    perm = np.random.default_rng(0).permutation(len(base))
    unrel_move = float(np.linalg.norm(base - base[perm], axis=1).mean() / base_norm)
    psi_norm = float(base_norm)

    ops = list(d4_ops().items())
    rho_all = sym.rho_all(dev).detach().cpu().numpy()
    d4_learned, d4_id = [], []
    for ridx, (nm, op) in enumerate(ops):
        if nm == "e":
            continue
        tgt = emb_fn(np.ascontiguousarray(op(val_np)))
        pred = base @ rho_all[ridx].T
        d4_learned.append(float(np.linalg.norm(tgt - pred) / (np.linalg.norm(tgt) + 1e-12)))
        d4_id.append(float(np.linalg.norm(tgt - base) / (np.linalg.norm(tgt) + 1e-12)))
    d4_id_resid = float(np.mean(d4_id))
    return dict(f1=f1, prec=prec, rec=rec,
                subpatch_relmove=subpatch_relmove, subpatch_dir=subpatch_dir,
                fourcell_relmove=fourcell_relmove, unrel_move=unrel_move,
                psi_norm=psi_norm,
                sym_ratio_subpatch=subpatch_relmove / (unrel_move + 1e-9),
                sym_ratio_d4=d4_id_resid / (unrel_move + 1e-9),
                d4_learned_resid=float(np.mean(d4_learned)),
                d4_id_resid=d4_id_resid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-per-epoch", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--k-pred", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rho-lr-mult", type=float, default=10.0)
    ap.add_argument("--decoder", choices=["frozen", "fresh"], default="frozen")
    ap.add_argument("--warmup", type=int, default=5, help="epochs to ramp symmetry lambda 0->1")
    ap.add_argument("--f1-floor", type=float, default=0.995)
    ap.add_argument("--lam-trans", type=float, default=20.0)
    ap.add_argument("--lam-d4", type=float, default=20.0)
    ap.add_argument("--lam-mixed", type=float, default=20.0)
    ap.add_argument("--lam-glaw", type=float, default=0.1)
    ap.add_argument("--lam-pred-aug", type=float, default=0.5)
    ap.add_argument("--lam-disc", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = DEVICE
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    v8 = load_v8()
    for p in v8.parameters():
        p.requires_grad_(False)
    adapter = Adapter().to(dev)
    decoder = Decoder().to(dev)
    if args.decoder == "frozen":
        decoder.load_v8_head(v8)
    sym = SymParams().to(dev)
    rho_params = [sym.rho_ne]
    base_params = [p for p in adapter.parameters()] + [sym.delta_lin.weight]
    if args.decoder == "fresh":
        base_params += list(decoder.parameters())
    opt = torch.optim.AdamW(
        [dict(params=base_params, lr=args.lr),
         dict(params=rho_params, lr=args.lr * args.rho_lr_mult, weight_decay=0.0)],
        lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n_params = sum(p.numel() for p in base_params + rho_params)

    lam = dict(trans=args.lam_trans, d4=args.lam_d4, mixed=args.lam_mixed,
               glaw=args.lam_glaw, pred_aug=args.lam_pred_aug, disc=args.lam_disc)

    val_np = (np.random.default_rng(999).random((2000, 40, 40))
              < np.array(DENSITIES)[np.random.default_rng(999).integers(0, len(DENSITIES), 2000)][:, None, None]
              ).astype(np.uint8)

    # unrel_target: distance between unrelated pooled psi with the adapter as
    # identity (its init state) == the raw-V8 value the regularizer must preserve
    with torch.no_grad():
        vp = torch.from_numpy(val_np[:1024].astype(np.float32)).unsqueeze(1).to(dev)
        p0 = adapter(v8_tfL4(v8, vp)).mean(1)
        pm = p0[torch.randperm(p0.shape[0], device=dev)]
        unrel_target = float((p0 - pm).norm(dim=1).mean())
    print(f"device={dev}  decoder={args.decoder}  trainable params={n_params:,}  "
          f"lam={lam}  unrel_target={unrel_target:.4f}")

    steps_per_epoch = args.n_per_epoch // args.batch

    if args.trial:
        n = 50
        t0 = time.time()
        first = last = None
        for i in range(n):
            logs = train_step(v8, adapter, decoder, sym, opt, make_batch(args.batch, rng, dev),
                              args.k, args.k_pred, lam, dev, rng,
                              lam_scale=1.0, unrel_target=unrel_target)
            if i == 0:
                first = logs
            last = logs
        dt = (time.time() - t0) / n
        print(f"\n[trial] {dt*1000:.0f} ms/step  ->  {dt*steps_per_epoch/60:.1f} min/epoch  "
              f"->  {dt*steps_per_epoch*args.epochs/3600:.2f} h for {args.epochs} epochs")
        print(f"[trial] loss step0: {first}")
        print(f"[trial] loss step{n-1}: {last}")
        ev = evaluate(v8, adapter, decoder, sym, val_np[:512], dev)
        print(f"[trial] quick eval (undertrained): {ev}")
        return

    log_path = os.path.join(RESULTS, "train_log.txt")
    hist = []
    best = -1e9
    with open(log_path, "w") as f:
        f.write(f"device={dev} params={n_params} lam={lam}\n")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        lam_scale = min(1.0, ep / max(1, args.warmup))
        acc = {}
        for st in range(steps_per_epoch):
            logs = train_step(v8, adapter, decoder, sym, opt,
                              make_batch(args.batch, rng, dev),
                              args.k, args.k_pred, lam, dev, rng,
                              lam_scale=lam_scale, unrel_target=unrel_target)
            for kk, vv in logs.items():
                acc[kk] = acc.get(kk, 0.0) + vv
        acc = {kk: vv / steps_per_epoch for kk, vv in acc.items()}
        sched.step()
        ev = evaluate(v8, adapter, decoder, sym, val_np, dev)
        row = dict(epoch=ep, lam_scale=lam_scale,
                   **{f"tr_{k}": v for k, v in acc.items()}, **ev,
                   sec=time.time() - t0)
        hist.append(row)
        line = (f"ep {ep:2d}/{args.epochs} λ={lam_scale:.2f}  F1={ev['f1']:.4f} "
                f"r={ev['rec']:.3f}  ‖ψ‖={ev['psi_norm']:.2f} unrel={ev['unrel_move']:.3f}  "
                f"sym/unrel: sp={ev['sym_ratio_subpatch']:.3f} d4={ev['sym_ratio_d4']:.3f}  "
                f"sp_rm={ev['subpatch_relmove']:.4f} d4_id={ev['d4_id_resid']:.4f} "
                f"d4_lin={ev['d4_learned_resid']:.4f}  "
                f"| Lpred={acc['pred']:.3f} disc={acc['disc']:.3f} "
                f"tr={acc['trans']:.4f} d4={acc['d4']:.4f} mix={acc['mixed']:.4f}  {row['sec']:.0f}s")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        # among F1-passing epochs, minimise the symmetry-vs-content ratios
        f1_ok = ev["f1"] >= args.f1_floor
        score = (ev["f1"] if f1_ok else ev["f1"] - 10.0) \
            - ev["sym_ratio_subpatch"] - ev["sym_ratio_d4"]
        if score > best:
            best = score
            torch.save(dict(adapter=adapter.state_dict(), decoder=decoder.state_dict(),
                            sym=sym.state_dict(), epoch=ep, metrics=ev, args=vars(args)),
                       os.path.join(CKPT, "best.pt"))
        with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
            json.dump(hist, f, indent=2)
    print(f"best composite score={best:.4f}  -> {CKPT}/best.pt")


if __name__ == "__main__":
    main()
