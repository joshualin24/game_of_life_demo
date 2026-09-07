"""Diagnostic for a Phase-1 checkpoint — the full symmetry picture that the
training-loop eval abbreviates.

Reports, on fresh val grids, for the pooled readout psi = mean_tok(adapter(tf_L4)):

  sub-patch translation
     rel_move    mean ||psi(t.x) - psi(x)|| / mean ||psi(x)||   (0 => INVARIANT)
     dir_cons    consistency of the displacement across content  (1 => clean delta)
     delta_fit   ||(psi(t.x)-psi(x)) - delta(t)|| / ||psi(t.x)-psi(x)||  (learned delta quality)
  4-cell translation
     rel_move    (baseline ~0 already)
  D4
     id_resid    ||psi(r.x) - psi(x)|| / ||psi(r.x)||
     learned     ||psi(r.x) - rho(r) psi(x)|| / ||psi(r.x)||        (trained operator)
     bestlin     ||psi(r.x) - W psi(x)|| / ||psi(r.x)||, W fit on held-out split
  accuracy
     F1 / prec / rec of the fresh decoder vs true next state

Usage:  python diag_phase1.py [checkpoints/phase1_best.pt]
"""
from __future__ import annotations

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_models import load_v8, DEVICE
from phase1_model import Adapter, Decoder, SymParams, v8_tfL4, gol_step_torch
from symmetry import subpatch_offsets, fourcell_offsets, translate, d4_ops

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/phase1_best.pt"


@torch.no_grad()
def psi_np(v8, adapter, g, dev, bs=256):
    out = []
    for s in range(0, len(g), bs):
        x = torch.from_numpy(g[s:s+bs].astype(np.float32)).unsqueeze(1).to(dev)
        out.append(adapter(v8_tfL4(v8, x)).mean(1).cpu().numpy())
    return np.concatenate(out)


def _u(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def main():
    dev = DEVICE
    v8 = load_v8()
    ck = torch.load(CKPT, map_location=dev)
    adapter = Adapter().to(dev); adapter.load_state_dict(ck["adapter"]); adapter.eval()
    decoder = Decoder().to(dev); decoder.load_state_dict(ck["decoder"]); decoder.eval()
    sym = SymParams().to(dev); sym.load_state_dict(ck["sym"]); sym.eval()
    print(f"loaded {CKPT}  (epoch {ck.get('epoch','?')})")

    rng = np.random.default_rng(4321)
    dens = rng.choice([0.12, 0.2, 0.3, 0.4, 0.5, 0.62], size=1500)
    val = (rng.random((1500, 40, 40)) < dens[:, None, None]).astype(np.uint8)

    base = psi_np(v8, adapter, val, dev)
    base_norm = np.linalg.norm(base, axis=1).mean()

    def transl_stats(offsets, want_delta):
        rm, dc_, df_ = [], [], []
        for (dr, dcc) in offsets:
            g = psi_np(v8, adapter, translate(val, dr, dcc), dev)
            d = g - base
            rm.append(np.linalg.norm(d, axis=1).mean() / base_norm)
            dbar = d.mean(0, keepdims=True)
            dc_.append(float((_u(d) * _u(dbar)).sum(1).mean()))
            if want_delta:
                dl = sym.delta(torch.tensor(dr), torch.tensor(dcc), dev).detach().cpu().numpy()
                df_.append(np.linalg.norm(d - dl, axis=1).mean()
                           / (np.linalg.norm(d, axis=1).mean() + 1e-12))
        return (float(np.mean(rm)), float(np.mean(dc_)),
                float(np.mean(df_)) if df_ else None)

    sp_rm, sp_dc, sp_df = transl_stats(subpatch_offsets(), True)
    fc_rm, fc_dc, _ = transl_stats(fourcell_offsets(np.random.default_rng(0), 16), False)

    print("\nsub-patch translation:")
    print(f"  rel_move  = {sp_rm:.4f}   (0 => psi invariant to sub-patch shift)")
    print(f"  dir_cons  = {sp_dc:.4f}   (1 => single content-independent direction)")
    print(f"  delta_fit = {sp_df:.4f}   (residual of learned delta vs actual displacement)")
    print("4-cell translation:")
    print(f"  rel_move  = {fc_rm:.4f}")

    # D4
    ops = list(d4_ops().items())
    rho_all = sym.rho_all(dev).detach().cpu().numpy()
    ntr = 900
    print("\nD4:")
    print(f"  {'r':<9} {'id_resid':>9} {'learned':>9} {'bestlin':>9}")
    ids, lrn, bl = [], [], []
    for ridx, (nm, op) in enumerate(ops):
        if nm == "e":
            continue
        tgt = psi_np(v8, adapter, np.ascontiguousarray(op(val)), dev)
        idr = np.linalg.norm(tgt - base) / (np.linalg.norm(tgt) + 1e-12)
        lr = np.linalg.norm(tgt - base @ rho_all[ridx].T) / (np.linalg.norm(tgt) + 1e-12)
        X, Y = base[:ntr], tgt[:ntr]
        lam = 1e-3 * np.trace(X.T @ X) / X.shape[1]
        W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
        blr = (np.linalg.norm(tgt[ntr:] - base[ntr:] @ W)
               / (np.linalg.norm(tgt[ntr:]) + 1e-12))
        ids.append(idr); lrn.append(lr); bl.append(blr)
        print(f"  {nm:<9} {idr:>9.4f} {lr:>9.4f} {blr:>9.4f}")
    print(f"  {'mean':<9} {np.mean(ids):>9.4f} {np.mean(lrn):>9.4f} {np.mean(bl):>9.4f}")

    # accuracy
    x = torch.from_numpy(val.astype(np.float32)).unsqueeze(1).to(dev)
    y = gol_step_torch(x)
    tp = fp = fn = 0.0
    with torch.no_grad():
        for s in range(0, x.shape[0], 512):
            pred = (decoder(adapter(v8_tfL4(v8, x[s:s+512]))) >= 0).float()
            yt = y[s:s+512]
            tp += float((pred*yt).sum()); fp += float((pred*(1-yt)).sum())
            fn += float(((1-pred)*yt).sum())
    prec, rec = tp/(tp+fp+1e-9), tp/(tp+fn+1e-9)
    print(f"\naccuracy:  F1={2*prec*rec/(prec+rec+1e-9):.4f}  prec={prec:.4f}  rec={rec:.4f}")


if __name__ == "__main__":
    main()
