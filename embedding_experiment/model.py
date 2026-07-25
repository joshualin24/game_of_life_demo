"""
Encoder + Projection Head for trajectory-invariant GoL embedding.

The encoder maps a 64x64 binary GoL grid to a latent vector h (dim=latent_dim).
The projection head maps h → z (dim=proj_dim) for use during contrastive training
and is discarded at inference time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),    # 64→32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 32→16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 16→8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),# 8→4
        )
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        # x: (B, 1, 64, 64)
        h = self.conv(x).flatten(1)
        return self.fc(h)  # (B, latent_dim)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, h):
        return self.net(h)  # (B, proj_dim)


class TrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.projector = ProjectionHead(latent_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)  # unit-norm for cosine similarity

    def encode(self, x):
        """Inference: return raw embedding (no projection head)."""
        return self.encoder(x)
