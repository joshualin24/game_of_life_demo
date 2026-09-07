"""Phase 1 training — hybrid symmetry adapter on frozen V8 tf_L4.

Target (pooled readout psi = mean_tok(adapter(tf_L4))):
  translations  psi(t.x) - psi(x) ~= delta(t)      (content-independent displacement)
  D4            psi(r.x)          ~= rho(r) psi(x)  (learned linear operator)
while a fresh decoder d(adapter(tf_L4)) still predicts the next state.

  python train_phase1.py --trial                 # ~50 steps, timing + sanity
  python train_phase1.py --epochs 20             # full run (~1 h on MPS)

Outputs: results/phase1_log.txt, results/phase1_metrics.json
         checkpoints/phase1_best.pt
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_models import load_v8, DEVICE
from phase1_model import (Adapter, Decoder, SymParams, v8_tfL4, CAYLEY,
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
def train_step(v8, adapter, decoder, sym, opt, batch, k, k_pred, lam, dev, rng):
    x, y = batch
    B = x.shape[0]
    tok = v8_tfL4(v8, x)
    z = adapter(tok)
    psi = z.mean(1)                                        # (B,64)
    scale = psi.pow(2).mean().detach() + 1e-6

    L_pred = F.binary_cross_entropy_with_logits(decoder(z), y)

    res_trans, res_d4, res_mixed = [], [], []
    L_pred_aug = torch.zeros((), device=dev)
    n_aug = 0
    for i in range(k):
        kind = ("trans", "d4", "mixed")[i % 3]
        r = int(rng.integers(1, N_D4)) if kind != "trans" else 0
        dr = int(rng.integers(0, 40)) if kind != "d4" else 0
        dc = int(rng.integers(0, 40)) if kind != "d4" else 0
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

    total = (L_pred + lam["pred_aug"] * L_pred_aug
             + lam["trans"] * L_trans + lam["d4"] * L_d4 + lam["mixed"] * L_mixed
             + lam["glaw"] * L_glaw)

    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for g in opt.param_groups for p in g["params"]], 2.0)
    opt.step()
    return dict(total=float(total), pred=float(L_pred), pred_aug=float(L_pred_aug),
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

    sp = subpatch_offsets()
    dir_c, mag_r = [], []
    for (dr, dc) in sp:
        d = emb_fn(translate(val_np, dr, dc)) - base
        dbar = d.mean(0, keepdims=True)
        dir_c.append(float((_u(d) * _u(dbar)).sum(1).mean()))
        mag_r.append(float(np.linalg.norm(d - dbar, axis=1).mean()
                           / (np.linalg.norm(dbar) + 1e-12)))
    subpatch_dir = float(np.mean(dir_c)); subpatch_magres = float(np.mean(mag_r))

    ops = list(d4_ops().items())
    rho_all = sym.rho_all(dev).cpu().numpy()
    d4_learned, d4_id = [], []
    for ridx, (nm, op) in enumerate(ops):
        if nm == "e":
            continue
        tgt = emb_fn(np.ascontiguousarray(op(val_np)))
        pred = base @ rho_all[ridx].T
        d4_learned.append(float(np.linalg.norm(tgt - pred) / (np.linalg.norm(tgt) + 1e-12)))
        d4_id.append(float(np.linalg.norm(tgt - base) / (np.linalg.norm(tgt) + 1e-12)))
    return dict(f1=f1, prec=prec, rec=rec,
                subpatch_dir=subpatch_dir, subpatch_magres=subpatch_magres,
                d4_learned_resid=float(np.mean(d4_learned)),
                d4_id_resid=float(np.mean(d4_id)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-per-epoch", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--k-pred", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam-trans", type=float, default=1.0)
    ap.add_argument("--lam-d4", type=float, default=1.0)
    ap.add_argument("--lam-mixed", type=float, default=1.0)
    ap.add_argument("--lam-glaw", type=float, default=0.1)
    ap.add_argument("--lam-pred-aug", type=float, default=0.5)
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
    sym = SymParams().to(dev)
    params = list(adapter.parameters()) + list(decoder.parameters()) + list(sym.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in params)

    lam = dict(trans=args.lam_trans, d4=args.lam_d4, mixed=args.lam_mixed,
               glaw=args.lam_glaw, pred_aug=args.lam_pred_aug)

    val_np = (np.random.default_rng(999).random((2000, 40, 40))
              < np.array(DENSITIES)[np.random.default_rng(999).integers(0, len(DENSITIES), 2000)][:, None, None]
              ).astype(np.uint8)

    print(f"device={dev}  trainable params={n_params:,}  lam={lam}")

    steps_per_epoch = args.n_per_epoch // args.batch

    if args.trial:
        n = 50
        t0 = time.time()
        first = last = None
        for i in range(n):
            logs = train_step(v8, adapter, decoder, sym, opt, make_batch(args.batch, rng, dev),
                              args.k, args.k_pred, lam, dev, rng)
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

    log_path = os.path.join(RESULTS, "phase1_log.txt")
    hist = []
    best = -1e9
    with open(log_path, "w") as f:
        f.write(f"device={dev} params={n_params} lam={lam}\n")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        acc = {}
        for st in range(steps_per_epoch):
            logs = train_step(v8, adapter, decoder, sym, opt,
                              make_batch(args.batch, rng, dev),
                              args.k, args.k_pred, lam, dev, rng)
            for kk, vv in logs.items():
                acc[kk] = acc.get(kk, 0.0) + vv
        acc = {kk: vv / steps_per_epoch for kk, vv in acc.items()}
        ev = evaluate(v8, adapter, decoder, sym, val_np, dev)
        row = dict(epoch=ep, **{f"tr_{k}": v for k, v in acc.items()}, **ev,
                   sec=time.time() - t0)
        hist.append(row)
        line = (f"ep {ep:2d}/{args.epochs}  F1={ev['f1']:.4f} p={ev['prec']:.3f} r={ev['rec']:.3f}  "
                f"subpatch_dir={ev['subpatch_dir']:.3f} magres={ev['subpatch_magres']:.2f}  "
                f"d4_lin={ev['d4_learned_resid']:.3f} (id {ev['d4_id_resid']:.3f})  "
                f"| L_pred={acc['pred']:.3f} trans={acc['trans']:.3f} d4={acc['d4']:.3f} "
                f"mix={acc['mixed']:.3f} glaw={acc['glaw']:.4f}  {row['sec']:.0f}s")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
        score = ev["f1"] + ev["subpatch_dir"] - ev["d4_learned_resid"]
        if score > best:
            best = score
            torch.save(dict(adapter=adapter.state_dict(), decoder=decoder.state_dict(),
                            sym=sym.state_dict(), epoch=ep, metrics=ev, args=vars(args)),
                       os.path.join(CKPT, "phase1_best.pt"))
        with open(os.path.join(RESULTS, "phase1_metrics.json"), "w") as f:
            json.dump(hist, f, indent=2)
    print(f"best composite score={best:.4f}  -> {CKPT}/phase1_best.pt")


if __name__ == "__main__":
    main()
