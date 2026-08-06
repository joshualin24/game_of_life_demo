"""
Compare per-grid error counts between V8 and V10 across densities.
Prints a table of total FP+FN errors across 500 grids per density.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from nn.models   import CNNTransformerV4
from nn.data_gen import run_trajectory
from nn.utils    import CKPT_DIR, DEVICE

GRID_SIZE = 40
SEED      = 7
N_GRIDS   = 500

def load_v8():
    m = CNNTransformerV4(grid_size=GRID_SIZE, patch_size=4,
                         d_model=64, nhead=4, num_layers=4).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "task16_cnn_transformer_v8_best.pt"),
                                 map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def load_v10():
    m = CNNTransformerV4(grid_size=GRID_SIZE, patch_size=4,
                         d_model=128, nhead=8, num_layers=6).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(CKPT_DIR, "task18_cnn_transformer_v10_best.pt"),
                                 map_location=DEVICE, weights_only=True))
    m.eval()
    return m

def predict(model, init):
    x = torch.tensor(init, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return (logits > 0).squeeze().cpu().numpy().astype(np.uint8)

def eval_density(model, density, rng):
    total_fp = total_fn = n_wrong = 0
    for _ in range(N_GRIDS):
        init = (rng.random((GRID_SIZE, GRID_SIZE)) < density).astype(np.uint8)
        true_next = run_trajectory(init, 1)[1]
        pred = predict(model, init)
        fp = int(((pred == 1) & (true_next == 0)).sum())
        fn = int(((pred == 0) & (true_next == 1)).sum())
        total_fp += fp
        total_fn += fn
        if fp + fn > 0:
            n_wrong += 1
    return total_fp, total_fn, n_wrong

def main():
    print("Loading models …")
    v8  = load_v8()
    v10 = load_v10()
    print("Done.\n")

    densities = [0.20, 0.35, 0.50, 0.65, 0.80]

    print(f"{'Density':>8}  {'V8 FP':>8} {'V8 FN':>8} {'V8 err':>8} {'V8 wrong':>9}  "
          f"{'V10 FP':>8} {'V10 FN':>8} {'V10 err':>8} {'V10 wrong':>9}  {'Δerr':>8}")
    print("-" * 100)

    for d in densities:
        rng = np.random.default_rng(SEED)
        fp8, fn8, w8 = eval_density(v8, d, rng)
        rng = np.random.default_rng(SEED)
        fp10, fn10, w10 = eval_density(v10, d, rng)
        delta = (fp10 + fn10) - (fp8 + fn8)
        sign  = "+" if delta >= 0 else ""
        print(f"  d={d:.2f}  {fp8:>8,} {fn8:>8,} {fp8+fn8:>8,} {w8:>9,}  "
              f"{fp10:>8,} {fn10:>8,} {fp10+fn10:>8,} {w10:>9,}  {sign}{delta:,}")

    print("\nDone.")

if __name__ == "__main__":
    main()
