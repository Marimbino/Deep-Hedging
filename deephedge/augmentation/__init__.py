"""WGAN-GP data augmentation: TCN networks, training, and path generation."""

from __future__ import annotations

from deephedge.augmentation.gan import Critic, Generator, gradient_penalty
from deephedge.augmentation.generator import QuantWGANGenerator
from deephedge.augmentation.tcn import TCN, CausalConv1d, TempBlock
from deephedge.augmentation.trainer import WGANTrainer

__all__ = [
    "CausalConv1d",
    "Critic",
    "Generator",
    "QuantWGANGenerator",
    "TCN",
    "TempBlock",
    "WGANTrainer",
    "gradient_penalty",
]
