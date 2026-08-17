"""
models.py
---------
PointerPolicyNet: autoregressive set-selection policy for the die-down
predictor. At each decode step it sees the current grid (with previously
chosen flips already applied), a mask of which cells have been chosen so
far, and how much flip budget remains, and outputs a distribution over
"flip this cell next" (one class per grid cell) plus a STOP class —
trained to imitate teacher_search.greedy_search's choices (see
generate_dataset.build_step_tensors).

Architecture is the U-Net skeleton from nn/models.py's SensitivityUNet
(Task 2) — a dense (B,1,H,W) -> (B,1,H,W) map is exactly the shape needed
for a per-cell "flip here" score — adapted with a second global STOP head
sharing the same softmax as an extra class (standard pointer-network-with-
stop-token formulation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(ConvBnRelu(in_ch, out_ch), ConvBnRelu(out_ch, out_ch))


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
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                      0, skip.shape[-2] - x.shape[-2]])
        return self.conv(torch.cat([x, skip], dim=1))


class PointerPolicyNet(nn.Module):
    """
    Input : (B, 3, H, W) — [grid with prior flips applied, chosen-cell mask,
             k_remaining/K broadcast constant]
    Output: (B, H*W + 1) logits — one per grid cell ("flip here next") plus
             a final STOP class. Pass `valid_mask` (B, H*W+1) bool to mask
             out cells outside the candidate neighborhood or already chosen
             (set to -inf before the softmax/cross-entropy).
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.3):
        super().__init__()
        c = base_ch
        self.enc1 = _EncoderBlock(3, c)
        self.enc2 = _EncoderBlock(c, 2 * c)
        self.enc3 = _EncoderBlock(2 * c, 4 * c)
        self.bottleneck = DoubleConv(4 * c, 8 * c)
        # Spatial dropout on the bottleneck (the most compressed, most
        # overfitting-prone representation, shared by both heads below) —
        # same rationale as the Dropout FateClassifier/ChaosPredictor use in
        # nn/models.py before their final classifier layers.
        self.bottleneck_dropout = nn.Dropout2d(p=dropout * 0.67)
        self.dec3 = _DecoderBlock(8 * c, 4 * c, 4 * c)
        self.dec2 = _DecoderBlock(4 * c, 2 * c, 2 * c)
        self.dec1 = _DecoderBlock(2 * c, c, c)
        self.spatial_head = nn.Conv2d(c, 1, kernel_size=1)

        self.stop_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8 * c, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        b = self.bottleneck_dropout(self.bottleneck(x3))
        d3 = self.dec3(b, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        spatial_logits = self.spatial_head(d1).flatten(1)        # (B, H*W)
        stop_logit = self.stop_head(b)                           # (B, 1)
        logits = torch.cat([spatial_logits, stop_logit], dim=1)  # (B, H*W+1)

        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, float("-inf"))
        return logits
