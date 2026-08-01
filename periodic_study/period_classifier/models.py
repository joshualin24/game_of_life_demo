"""
models.py — Three NN architectures for GoL period classification.

SimpleCNN      : baseline CNN on initial state only
ResidualCNN    : ResNet-style with residual blocks on initial state
EvolutionCNN   : CNN on a stack of T simulated GoL frames
                 (encodes temporal dynamics explicitly)

All accept input (B, C, H, W) and output logits (B, n_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 11   # periods [2,3,4,5,6,8,14,15,16,24,30]


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    """Pre-activation residual block with optional channel/stride projection."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


# ── Model 1: SimpleCNN ────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Four conv-pool stages → global avg pool → linear classifier.
    Input: (B, 1, H, W) initial state.
    """
    def __init__(self, n_classes=N_CLASSES, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(in_channels,  32), nn.MaxPool2d(2),   # H/2
            ConvBnRelu(32,  64),          nn.MaxPool2d(2),   # H/4
            ConvBnRelu(64,  128),         nn.MaxPool2d(2),   # H/8
            ConvBnRelu(128, 256),         nn.MaxPool2d(2),   # H/16
            ConvBnRelu(256, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ── Model 2: ResidualCNN ──────────────────────────────────────────────────────

class ResidualCNN(nn.Module):
    """
    ResNet-style: stem + 4 stages of residual blocks + classifier.
    Input: (B, 1, H, W) initial state.
    """
    def __init__(self, n_classes=N_CLASSES, in_channels=1):
        super().__init__()
        self.stem = ConvBnRelu(in_channels, 32)
        self.stage1 = nn.Sequential(ResBlock(32,  64,  stride=2),
                                    ResBlock(64,  64))
        self.stage2 = nn.Sequential(ResBlock(64,  128, stride=2),
                                    ResBlock(128, 128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2),
                                    ResBlock(256, 256))
        self.stage4 = nn.Sequential(ResBlock(256, 256, stride=2),
                                    ResBlock(256, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


# ── Model 3: EvolutionCNN ─────────────────────────────────────────────────────

class TemporalFusion(nn.Module):
    """
    Fuse T frames: each frame → spatial features → pool → concat → MLP.
    Shares the spatial encoder across all T frames.
    """
    def __init__(self, n_frames, feat_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(1, 32), nn.MaxPool2d(2),
            ConvBnRelu(32, 64), nn.MaxPool2d(2),
            ConvBnRelu(64, feat_dim), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.n_frames = n_frames
        self.feat_dim = feat_dim

    def forward(self, x):
        # x: (B, T, H, W)  — each channel is one frame
        B, T, H, W = x.shape
        # Process each frame independently
        frames = x.view(B * T, 1, H, W)
        feats  = self.encoder(frames)          # (B*T, feat_dim)
        feats  = feats.view(B, T, self.feat_dim)   # (B, T, feat_dim)
        # Max-pool over time: most salient spatial feature across frames
        pooled = feats.max(dim=1).values       # (B, feat_dim)
        return pooled


class EvolutionCNN(nn.Module):
    """
    Uses T simulated GoL frames (stacked as channels) to expose periodicity.
    Input: (B, T, H, W) where T = n_frames + 1 (initial + T-1 steps).

    Architecture: shared spatial encoder per frame → temporal max-pool → classifier.
    A second branch uses a direct 2D conv on all frames concatenated
    to capture spatial correlations across time.
    """
    def __init__(self, n_classes=N_CLASSES, n_frames=31):
        super().__init__()
        self.n_frames = n_frames

        # Branch A: shared per-frame encoder
        self.temporal = TemporalFusion(n_frames, feat_dim=128)

        # Branch B: 2D conv treating all frames as channels
        self.spatial = nn.Sequential(
            ConvBnRelu(n_frames, 64), nn.MaxPool2d(2),
            ConvBnRelu(64, 128), nn.MaxPool2d(2),
            ConvBnRelu(128, 256), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B, T, H, W)
        a = self.temporal(x)    # (B, 128)
        b = self.spatial(x)     # (B, 256)
        return self.classifier(torch.cat([a, b], dim=1))


# ── Registry ──────────────────────────────────────────────────────────────────

def get_model(name: str, n_frames: int = 30) -> nn.Module:
    """
    name in {'simple', 'residual', 'evolution'}
    n_frames: number of GoL steps simulated for EvolutionCNN input
    """
    if name == "simple":
        return SimpleCNN(in_channels=1)
    elif name == "residual":
        return ResidualCNN(in_channels=1)
    elif name == "evolution":
        return EvolutionCNN(n_frames=n_frames + 1)   # +1 for initial frame
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose simple/residual/evolution")
