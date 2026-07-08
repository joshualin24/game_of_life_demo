"""Convolutional VAE for 64x64 binary Game of Life grids.

Architecture improvements over v1:
- BatchNorm after every conv/deconv (except final output)
- Decoder uses Upsample + Conv2d instead of ConvTranspose2d to avoid
  checkerboard artifacts
- Default latent_dim raised to 64
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),    # 64 → 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),   # 32 → 16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 16 → 8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),# 8 → 4
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.up = nn.Sequential(
            # 4 → 8
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            # 8 → 16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            # 16 → 32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            # 32 → 64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, 1, 1),  # logits; no BN before output
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return self.up(h)  # logits; apply sigmoid for probabilities


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar
