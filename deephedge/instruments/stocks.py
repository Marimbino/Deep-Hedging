"""Underlying asset simulators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from deephedge.instruments.base import Primary

if TYPE_CHECKING:
    from deephedge.augmentation.generator import QuantWGANGenerator


class BrownianStock(Primary):
    """Geometric Brownian motion stock: ``dS = sigma * S * dW`` (zero drift).

    Parameters
    ----------
    sigma : float
        Annualised volatility.
    cost : float
        Proportional transaction cost.
    dt : float
        Time step in years.
    dtype : torch.dtype, optional
        Tensor dtype for simulated paths (default float32).
    device : torch.device or str, optional
        Device for simulated paths.
    """

    def __init__(
        self,
        sigma: float = 0.2,
        cost: float = 0.0,
        dt: float = 1 / 252,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(cost=cost, dt=dt)
        self.sigma = sigma
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = torch.device(device) if device is not None else None

    def simulate(self, n_paths: int, n_steps: int, seed: int | None = None) -> Tensor:
        """Simulate GBM paths starting at 1.0.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        n_steps : int
            Number of time steps.
        seed : int, optional
            Random seed.

        Returns
        -------
        Tensor
            Prices of shape ``(n_paths, n_steps + 1)``.
        """
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(n_paths, n_steps, dtype=self.dtype, device=self.device)
        increments = self.sigma * np.sqrt(self.dt) * z - 0.5 * self.sigma**2 * self.dt
        log_paths = torch.cat(
            [
                torch.zeros(n_paths, 1, dtype=self.dtype, device=self.device),
                torch.cumsum(increments, dim=1),
            ],
            dim=1,
        )
        return torch.exp(log_paths)


class WGANStock(Primary):
    """Synthetic stock driven by a pre-trained :class:`QuantWGANGenerator`.

    Parameters
    ----------
    generator : QuantWGANGenerator
        A fitted generator instance.
    cost : float
        Proportional transaction cost.
    dt : float
        Time step in years.
    """

    def __init__(
        self,
        generator: "QuantWGANGenerator",
        cost: float = 0.0,
        dt: float = 1 / 252,
    ) -> None:
        super().__init__(cost=cost, dt=dt)
        self.generator = generator

    def simulate(self, n_paths: int, n_steps: int, seed: int | None = None) -> Tensor:
        """Sample WGAN price paths starting at 1.0.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        n_steps : int
            Number of time steps; must not exceed the generator's ``seq_len``.
        seed : int, optional
            Random seed.

        Returns
        -------
        Tensor
            Prices of shape ``(n_paths, n_steps + 1)``.
        """
        if n_steps > self.generator.seq_len:
            raise ValueError(
                f"n_steps={n_steps} exceeds the generator's seq_len={self.generator.seq_len}"
            )
        prices = self.generator.generate(n_paths, S0=1.0, seed=seed)
        return torch.as_tensor(prices[:, : n_steps + 1], dtype=torch.float32)
