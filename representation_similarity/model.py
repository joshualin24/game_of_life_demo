"""
CNNTransformerV4 — architecture for the V8 and V10 models.

Copied verbatim from ../nn/models.py (class CNNTransformerV4, Task 12) so this
study folder is a self-contained reference. Both V8 (d_model=64) and V10
(d_model=128) use this exact class — they differ only in constructor args
(see load_models.py). Self-contained: depends only on torch.

If ../nn/models.py::CNNTransformerV4 ever changes, re-sync this file.
"""

import torch
import torch.nn as nn


class CNNTransformerV4(nn.Module):
    """
    CNN-Transformer V4 — circular (toroidal) padding in the CNN encoder.

      Stage 1 — CNN local encoder (RF=5x5, GroupNorm, circular padding)
      Stage 2 — Flatten patches -> Linear(d_model*p^2, d_model) + learnable 2D pos emb
      Stage 3 — pre-norm transformer encoder (num_layers)
      Per-patch linear head -> (B, 1, H, W) next-state logits

    Representation vectors of interest (see forward()):
      - `feat`   : CNN encoder output, (B, d_model, H, W)
      - `tokens` : per-patch tokens after patch_proj + pos_embed, (B, n_patches, d_model)
      - `out`    : per-patch tokens after the transformer, (B, n_patches, d_model)
    There is no CLS token; the head is per-patch.
    """

    def __init__(
        self,
        grid_size:  int   = 40,
        patch_size: int   = 4,
        d_model:    int   = 64,
        nhead:      int   = 4,
        num_layers: int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert grid_size % patch_size == 0
        self.grid_size  = grid_size
        self.patch_size = patch_size
        self.d_model    = d_model
        n_patches_1d    = grid_size // patch_size
        self.n_patches  = n_patches_1d ** 2
        patch_feat_dim  = d_model * patch_size * patch_size

        # Stage 1: CNN with circular padding (respects toroidal grid topology)
        self.cnn = nn.Sequential(
            nn.Conv2d(1,       32,      3, padding=1, padding_mode='circular', bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, d_model,     3, padding=1, padding_mode='circular', bias=False),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        # Stage 2: lossless patch projection + learnable 2D positional embedding
        self.patch_proj = nn.Linear(patch_feat_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 3: transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.patch_head = nn.Linear(d_model, patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B  = x.shape[0]
        p  = self.patch_size
        gs = self.grid_size

        feat = self.cnn(x)
        C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
        h = w = H // p
        feat   = feat.reshape(B, C, h, p, w, p)
        feat   = feat.permute(0, 2, 4, 1, 3, 5)
        feat   = feat.reshape(B, self.n_patches, C * p * p)
        tokens = self.patch_proj(feat) + self.pos_embed

        out    = self.transformer(tokens)
        logits = self.patch_head(out)
        logits = logits.reshape(B, h, w, p, p)
        logits = logits.permute(0, 1, 3, 2, 4)
        return logits.reshape(B, 1, gs, gs)

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (torch.sigmoid(self.forward(x)) >= threshold).float()
