"""
Load the trained V8 / V10 GoL models for the representation-similarity study.

Checkpoints live in ../nn/checkpoints/ (gitignored -> present only in the main
checkout, NOT in git worktrees). CKPT_DIR below points at the main checkout by
absolute path so this works from anywhere. Override with $GOL_CKPT_DIR.

Both models are the same class (CNNTransformerV4); they differ only in the
config dicts below.

    from load_models import load_v8, load_v10, DEVICE
    v8  = load_v8()      # eval(), on DEVICE
    v10 = load_v10()
"""

import os
import torch

from model import CNNTransformerV4

# ── device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# ── checkpoint directory (main checkout; gitignored, not in worktrees) ────────
CKPT_DIR = os.environ.get(
    "GOL_CKPT_DIR",
    "/Users/Hao-Yuan/game_of_life_demo/nn/checkpoints",
)

# ── model configs ────────────────────────────────────────────────────────────
# Source: nn/train_cnn_transformer_v8.py (lines 34-38) and
#         nn/train_cnn_transformer_v10.py (Task 18 header + lines 43-52).
# dim_feedforward is not a constructor arg; it is auto-set to d_model * 4.
V8 = {
    "name":       "V8",
    "task":       "task16_cnn_transformer_v8",
    "ckpt":       "task16_cnn_transformer_v8_best.pt",
    "params":     291_888,
    "config":     dict(grid_size=40, patch_size=4, d_model=64,  nhead=4, num_layers=4),
    "training":   "warm-started from V7 best; +30 epochs, 3x high-density upsampled data; "
                  "scheduled sampling K=3. val_loss=0.20064, F1=1.0000, prec=100%, rec=99.9%.",
}
V10 = {
    "name":       "V10",
    "task":       "task18_cnn_transformer_v10",
    "ckpt":       "task18_cnn_transformer_v10_best.pt",
    "params":     1_504_240,
    "config":     dict(grid_size=40, patch_size=4, d_model=128, nhead=8, num_layers=6),
    "training":   "trained from scratch (d_model change blocks weight transfer); "
                  "scheduled sampling K=3, V8's dataset config, 100 epochs. "
                  "val_loss=0.19852, F1/prec/rec = 100%. Zero errors on 2500/2500 failure-search grids.",
}


def _load(spec: dict, ckpt: str | None = None, eval_mode: bool = True) -> CNNTransformerV4:
    model = CNNTransformerV4(**spec["config"]).to(DEVICE)
    path  = os.path.join(CKPT_DIR, ckpt or spec["ckpt"])
    state = torch.load(path, map_location=DEVICE, weights_only=True)  # bare state_dict
    model.load_state_dict(state)
    if eval_mode:
        model.eval()
    return model


def load_v8(ckpt: str | None = None, eval_mode: bool = True) -> CNNTransformerV4:
    return _load(V8, ckpt, eval_mode)


def load_v10(ckpt: str | None = None, eval_mode: bool = True) -> CNNTransformerV4:
    return _load(V10, ckpt, eval_mode)


if __name__ == "__main__":
    for spec, loader in ((V8, load_v8), (V10, load_v10)):
        m = loader()
        n = sum(p.numel() for p in m.parameters())
        print(f"{spec['name']:>4}  d_model={spec['config']['d_model']:>3}  "
              f"layers={spec['config']['num_layers']}  params={n:,}  "
              f"(expected {spec['params']:,})  device={DEVICE}")
