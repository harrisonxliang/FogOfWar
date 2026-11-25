"""Masked-diffusion belief model: predict hidden squares of the true board.

Absorbing-state discrete diffusion over the 64x13 grid (MDLM/MaskGIT style):
visible squares are conditioning (never masked, never predicted); a random
fraction of hidden squares is masked to the MASK token during training and the
network predicts their true contents with cross-entropy.

Evaluated with everything masked, a single pass is exactly "model v1" (the
per-square distribution given the observation) and provides score(board | obs).
Iterative unmasking at inference adds joint structure (model v2).

~1.3M parameters at the default width; CNN over 8x8 with 45 input planes.
"""
from __future__ import annotations

import torch
from torch import nn

from encoding import NUM_CLASSES, PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        h = torch.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return torch.relu(x + h)


class BeliefNet(nn.Module):
    def __init__(self, channels=96, blocks=6):
        super().__init__()
        self.stem = nn.Conv2d(PLANES, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, NUM_CLASSES, 1)

    def forward(self, planes):
        """(N, 45, 8, 8) -> (N, 13, 64) logits over square contents."""
        h = self.blocks(torch.relu(self.stem(planes)))
        return self.head(h).reshape(planes.shape[0], NUM_CLASSES, 64)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
