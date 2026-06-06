"""
All neural network architectures for GoL experiments.

Task 1  NextStatePredictor  – predict t+1 from t           (conv)
Task 2  SensitivityUNet     – predict sensitivity map       (U-Net)
Task 3  ChaosPredictor      – predict divergence score      (conv + MLP head)
Task 4  NeuralCA            – learnable GoL update rule     (tiny conv)
Task 5  RolloutPredictor    – predict t+k from t            (residual conv)
Task 6  FateClassifier      – classify attractor type       (conv + classifier)
"""

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
