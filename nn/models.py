"""
All neural network architectures for GoL experiments.

Task 1  NextStatePredictor  – predict t+1 from t           (conv)
Task 2  SensitivityUNet     – predict sensitivity map       (U-Net)
Task 3  ChaosPredictor      – predict divergence score      (conv + MLP head)
Task 4  NeuralCA            – learnable GoL update rule     (tiny conv)
Task 5  RolloutPredictor    – predict t+k from t            (residual conv)
Task 6  FateClassifier      – classify attractor type       (conv + classifier)
Task 7  TrajectoryTransformer – trajectory embedding        (CNN frame encoder + transformer)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building blocks ─────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )


# ── Task 4: Neural CA ──────────────────────────────────────────────────────────

class NeuralCA(nn.Module):
    """
    Learnable GoL update rule expressed as a small fully-convolutional net.
    Uses only local 3×3 receptive field, so it generalises to any grid size.

    Architecture:
      Conv(1→64, 3×3) → ReLU → Conv(64→32, 1×1) → ReLU → Conv(32→1, 1×1) → Sigmoid
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) float in [0,1]. Returns (B,1,H,W) next-state probs."""
        return self.net(x)

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Hard-threshold forward pass → binary grid."""
        return (self.forward(x) > threshold).float()


# ── Task 1: Next-state predictor ──────────────────────────────────────────────

class NextStatePredictor(nn.Module):
    """
    Deeper convolutional next-state predictor (same input/output as NeuralCA
    but with more capacity and residual connections for task-1 experiments).
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(1, channels),
            ConvBnRelu(channels, channels),
        )
        self.residual = nn.Sequential(
            ConvBnRelu(channels, channels),
            ConvBnRelu(channels, channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h + self.residual(h)
        return self.decoder(h)


# ── Task 2: Sensitivity U-Net ─────────────────────────────────────────────────

class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd sizes
        x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                      0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([x, skip], dim=1))


class SensitivityUNet(nn.Module):
    """
    U-Net: (B,1,H,W) grid → (B,1,H,W) cumulative sensitivity map.
    Output is passed through ReLU (non-negative divergence scores).
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.enc1 = _EncoderBlock(1,     c)
        self.enc2 = _EncoderBlock(c,   2*c)
        self.enc3 = _EncoderBlock(2*c, 4*c)
        self.bottleneck = DoubleConv(4*c, 8*c)
        self.dec3 = _DecoderBlock(8*c, 4*c, 4*c)
        self.dec2 = _DecoderBlock(4*c, 2*c, 2*c)
        self.dec1 = _DecoderBlock(2*c,   c,   c)
        self.head = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1),
            nn.ReLU(),   # divergence is non-negative
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,H,W) binary grid float."""
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        b       = self.bottleneck(x3)
        d3      = self.dec3(b,  s3)
        d2      = self.dec2(d3, s2)
        d1      = self.dec1(d2, s1)
        return self.head(d1)


# ── Task 3: Chaos predictor ───────────────────────────────────────────────────

class ChaosPredictor(nn.Module):
    """
    Input : 2-channel grid (B, 2, H, W)
              channel 0 = initial grid
              channel 1 = perturbation location mask (single 1, rest 0)
    Output: (B, 1) predicted cumulative divergence score.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.features = nn.Sequential(
            DoubleConv(2,   c),
            nn.MaxPool2d(2),
            DoubleConv(c, 2*c),
            nn.MaxPool2d(2),
            DoubleConv(2*c, 4*c),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*c * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.ReLU(),   # non-negative output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ── Task 5: Rollout predictor ─────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class RolloutPredictor(nn.Module):
    """
    Direct k-step predictor with a residual tower.
    Input : (B, 1+1, H, W) — grid concat step-embedding (normalised k/K)
    Output: (B, 1, H, W) predicted state k steps later.
    """
    def __init__(self, channels: int = 64, n_res: int = 6):
        super().__init__()
        self.stem = ConvBnRelu(2, channels)   # grid + step channel
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(n_res)])
        self.head  = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, grid: torch.Tensor, k_norm: torch.Tensor) -> torch.Tensor:
        """
        grid  : (B, 1, H, W)
        k_norm: (B,) normalised step in [0,1]
        """
        step_map = k_norm.view(-1, 1, 1, 1).expand_as(grid)
        x = torch.cat([grid, step_map], dim=1)
        return self.head(self.tower(self.stem(x)))


# ── Task 6: Fate classifier ───────────────────────────────────────────────────

class FateClassifier(nn.Module):
    """
    Input : (B, 1, H, W) initial grid
    Output: (B, 4) logits for {dies, still_life, oscillator, active}
    """
    def __init__(self, base_ch: int = 32, n_classes: int = 4):
        super().__init__()
        c = base_ch
        self.features = nn.Sequential(
            DoubleConv(1,    c),
            nn.MaxPool2d(2),
            DoubleConv(c,  2*c),
            nn.MaxPool2d(2),
            DoubleConv(2*c, 4*c),
            nn.MaxPool2d(2),
            DoubleConv(4*c, 8*c),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*c * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Task 7: Trajectory Transformer embedding ──────────────────────────────────

class FrameEncoder(nn.Module):
    """
    Encodes a single (B, 1, H, W) binary grid frame to a (B, d_model) vector.
    Three conv layers (fixed at 16→32→64 channels) followed by global avg-pool
    and a linear projection to d_model.  Weights are shared across all frames
    in the trajectory.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,  16, 3, padding=1, bias=False), nn.BatchNorm2d(16),  nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) → (B, d_model)"""
        return self.proj(self.conv(x))


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding added to the time dimension."""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TrajectoryTransformer(nn.Module):
    """
    Encodes a GoL trajectory (T+1 frames) into a single embedding vector via a
    [CLS] token processed by a transformer encoder.

    Architecture
    ────────────
    1. FrameEncoder  : (B·(T+1), 1, H, W) → (B, T+1, d_model)   [shared weights]
    2. Prepend [CLS] : (B, T+2, d_model)
    3. Sinusoidal PE : add time-step positional encoding
    4. Transformer   : 4 pre-norm encoder layers, nhead heads
    5. CLS output    : (B, d_model)  ← trajectory embedding
    6. frame_head    : shared linear (B, T+1, d_model) → (B, T+1, H·W)
                       used as per-frame reconstruction logits (training signal)

    Training objective: BCE on per-frame reconstruction of all T+1 frames.
    The CLS token learns to aggregate trajectory-level information through
    self-attention with all frame tokens.

    Args:
        d_model  : embedding dimension (64 or 128)
        nhead    : attention heads (must divide d_model)
        num_layers: transformer depth
        grid_size: spatial size of each frame (H = W)
        T        : trajectory length (number of GoL steps; frames = T+1)
        dropout  : dropout rate inside the transformer
    """

    def __init__(
        self,
        d_model:    int = 64,
        nhead:      int = 4,
        num_layers: int = 4,
        grid_size:  int = 40,
        T:          int = 60,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_model   = d_model
        self.T         = T
        self.H = self.W = grid_size

        self.frame_encoder = FrameEncoder(d_model)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc       = _SinusoidalPE(d_model, max_len=T + 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable training)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # shared per-frame reconstruction head (training signal for CLS via attention)
        self.frame_head = nn.Linear(d_model, grid_size * grid_size)


# ── Task 8: Next-Step Transformer (ViT-style) ─────────────────────────────────
# ── Task 9: CNN-Transformer Hybrid ────────────────────────────────────────────

class CNNTransformer(nn.Module):
    """
    Hybrid next-step predictor: CNN local encoder → patch tokens → transformer.

    Stage 1 — CNN local encoder (receptive field 5×5):
      Conv(1→32, 3×3) → GroupNorm → GELU
      Conv(32→d_model, 3×3) → GroupNorm → GELU
      Output: (B, d_model, H, W)  — every position already sees its full GoL neighborhood
      GroupNorm (not BatchNorm) avoids train/eval running-statistics divergence.

    Stage 2 — Patch tokenization via spatial avg-pool:
      AvgPool(patch_size×patch_size) → (B, d_model, H//p, W//p)
      Reshape → (B, n_patches, d_model) tokens + learnable 2-D pos embeddings
      Because the CNN already spans patch boundaries, tokens are locally complete.

    Stage 3 — Transformer encoder (global context):
      4 pre-norm layers; self-attention integrates long-range pattern interactions.
      Per-patch head → patch_size² logits → reshape to (B, 1, H, W).

    Args:
        grid_size  : spatial size of each frame (H = W, must be divisible by patch_size)
        patch_size : patch side length (default 4 → 100 patches on 40×40)
        d_model    : channel dimension throughout CNN and transformer
        nhead      : attention heads (must divide d_model)
        num_layers : transformer encoder depth
        dropout    : dropout inside transformer layers
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
        assert grid_size % patch_size == 0, "grid_size must be divisible by patch_size"
        self.grid_size  = grid_size
        self.patch_size = patch_size
        self.d_model    = d_model
        n_patches_1d    = grid_size // patch_size
        self.n_patches  = n_patches_1d ** 2

        # Stage 1: local CNN encoder (two 3×3 conv layers → RF = 5×5)
        # GroupNorm instead of BatchNorm: normalizes per-sample within channel groups,
        # so train and eval modes are identical (no running-statistics divergence).
        self.cnn = nn.Sequential(
            nn.Conv2d(1,       32,      3, padding=1, bias=False),
            nn.GroupNorm(8, 32),        # 8 groups of 4 channels
            nn.GELU(),
            nn.Conv2d(32, d_model,     3, padding=1, bias=False),
            nn.GroupNorm(8, d_model),   # 8 groups of 8 channels (for d_model=64)
            nn.GELU(),
        )

        # Stage 2: learnable 2-D positional embedding (no linear projection needed —
        # avg-pool already maps CNN features to d_model-dim patch tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 3: transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        # Per-patch reconstruction head: d_model → patch_size² cell logits
        self.patch_head = nn.Linear(d_model, patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) binary float → (B, 1, H, W) next-state logits."""
        B  = x.shape[0]
        p  = self.patch_size
        gs = self.grid_size

        # Stage 1: CNN local features
        feat = self.cnn(x)                                       # (B, d_model, H, W)

        # Stage 2: spatial avg-pool within patches → tokens
        tokens = F.avg_pool2d(feat, kernel_size=p, stride=p)    # (B, d_model, H//p, W//p)
        tokens = tokens.flatten(2).transpose(1, 2)              # (B, n_patches, d_model)
        tokens = tokens + self.pos_embed

        # Stage 3: transformer global context
        out    = self.transformer(tokens)                        # (B, n_patches, d_model)

        # Reconstruction
        logits = self.patch_head(out)                            # (B, n_patches, p²)
        h = w  = gs // p
        logits = logits.reshape(B, h, w, p, p)
        logits = logits.permute(0, 1, 3, 2, 4)                 # (B, h, p, w, p)
        return logits.reshape(B, 1, gs, gs)

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Hard next-state prediction. x: (B,1,H,W) → (B,1,H,W) binary."""
        return (torch.sigmoid(self.forward(x)) >= threshold).float()




class NextStepTransformer(nn.Module):
    """
    ViT-style single-step GoL predictor.

    The 40×40 grid is divided into non-overlapping patch_size×patch_size patches.
    Each patch is linearly embedded → transformer encoder → per-patch linear head
    reconstructs the patch cells at t+1.  Can be unrolled autoregressively.

    Args:
        grid_size  : spatial size of the grid (H = W, must be divisible by patch_size)
        patch_size : spatial size of each patch (default 4 → 100 patches on 40×40)
        d_model    : transformer embedding dimension
        nhead      : number of attention heads
        num_layers : transformer encoder depth
        dropout    : dropout inside transformer layers
    """

    def __init__(
        self,
        grid_size:  int = 40,
        patch_size: int = 4,
        d_model:    int = 64,
        nhead:      int = 4,
        num_layers: int = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert grid_size % patch_size == 0, "grid_size must be divisible by patch_size"
        self.grid_size  = grid_size
        self.patch_size = patch_size
        self.d_model    = d_model
        n_patches_1d    = grid_size // patch_size
        self.n_patches  = n_patches_1d ** 2       # total number of patches
        patch_dim       = patch_size ** 2          # cells per patch

        # ── patch embedding ──────────────────────────────────────────────────
        self.patch_embed = nn.Linear(patch_dim, d_model)

        # ── learnable 2-D positional encoding ────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        # ── per-patch reconstruction head ─────────────────────────────────────
        self.patch_head = nn.Linear(d_model, patch_dim)  # → logits per cell

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) → (B, n_patches, patch_dim)"""
        B, _, H, W = x.shape
        p = self.patch_size
        x = x.squeeze(1)
        x = x.reshape(B, H // p, p, W // p, p)
        x = x.permute(0, 1, 3, 2, 4)                           # (B, h, w, p, p)
        x = x.reshape(B, self.n_patches, p * p)
        return x

    def _from_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, n_patches, patch_dim) → (B, 1, H, W)"""
        B = patches.shape[0]
        p = self.patch_size
        H = W = self.grid_size
        h = w = H // p
        x = patches.reshape(B, h, w, p, p)
        x = x.permute(0, 1, 3, 2, 4)                           # (B, h, p, w, p)
        return x.reshape(B, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) binary float grid at time t.
        Returns: (B, 1, H, W) logits for the next state at t+1.
        """
        patches = self._to_patches(x)                          # (B, n_patches, p²)
        tokens  = self.patch_embed(patches) + self.pos_embed   # (B, n_patches, d_model)
        out     = self.transformer(tokens)                     # (B, n_patches, d_model)
        logits  = self.patch_head(out)                         # (B, n_patches, p²)
        return self._from_patches(logits)                      # (B, 1, H, W)

    def step(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Hard next-state prediction. x: (B,1,H,W) → (B,1,H,W) binary."""
        return (torch.sigmoid(self.forward(x)) >= threshold).float()


# ── TrajectoryTransformer methods (separated to avoid nesting bug) ────────────

class TrajectoryTransformer(TrajectoryTransformer):  # type: ignore[no-redef]
    def encode(self, traj: torch.Tensor) -> torch.Tensor:
        """traj: (B, T+1, H, W) → (B, d_model) CLS embedding."""
        B, Tp1, H, W = traj.shape
        frames = traj.reshape(B * Tp1, 1, H, W)
        embs   = self.frame_encoder(frames).reshape(B, Tp1, self.d_model)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, embs], dim=1)
        tokens = self.pos_enc(tokens)
        out    = self.transformer(tokens)
        return out[:, 0]

    def forward(self, traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        traj: (B, T+1, H, W) float in [0, 1]
        Returns: embedding (B, d_model), recon_logits (B, T+1, H, W)
        """
        B, Tp1, H, W = traj.shape
        frames = traj.reshape(B * Tp1, 1, H, W)
        embs   = self.frame_encoder(frames).reshape(B, Tp1, self.d_model)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, embs], dim=1)
        tokens = self.pos_enc(tokens)
        out    = self.transformer(tokens)
        embedding    = out[:, 0]
        frame_outs   = out[:, 1:]
        recon_logits = self.frame_head(frame_outs).reshape(B, Tp1, H, W)
        return embedding, recon_logits
