"""Derivative payoff definitions."""

from __future__ import annotations

import torch
from torch import Tensor

from deephedge.instruments.base import Derivative, Primary


class EuropeanOption(Derivative):
    """European call option: ``payoff = max(S_T - K, 0)``.

    Parameters
    ----------
    underlier : Primary
        The underlying asset.
    strike : float
        Strike price ``K``.
    maturity : float
        Time to maturity in years.
    """

    def __init__(self, underlier: Primary, strike: float = 1.0, maturity: float = 30 / 252) -> None:
        super().__init__(underlier, strike=strike, maturity=maturity)

    def payoff(self, prices: Tensor) -> Tensor:
        """Terminal call payoff per path, shape ``(n_paths,)``."""
        return torch.clamp(prices[:, -1] - self.strike, min=0.0)


class LookbackOption(Derivative):
    """Lookback call option: ``payoff = max(max_t S_t - K, 0)``.

    Parameters
    ----------
    underlier : Primary
        The underlying asset.
    strike : float
        Strike price ``K``.
    maturity : float
        Time to maturity in years.
    """

    def __init__(self, underlier: Primary, strike: float = 1.0, maturity: float = 30 / 252) -> None:
        super().__init__(underlier, strike=strike, maturity=maturity)

    def payoff(self, prices: Tensor) -> Tensor:
        """Lookback payoff per path, shape ``(n_paths,)``."""
        running_max = prices.max(dim=1).values
        return torch.clamp(running_max - self.strike, min=0.0)


class BinaryOption(Derivative):
    """Binary (digital) call option: ``payoff = 1 if S_T > K else 0``.

    Parameters
    ----------
    underlier : Primary
        The underlying asset.
    strike : float
        Strike price ``K``.
    maturity : float
        Time to maturity in years.
    """

    def __init__(self, underlier: Primary, strike: float = 1.0, maturity: float = 30 / 252) -> None:
        super().__init__(underlier, strike=strike, maturity=maturity)

    def payoff(self, prices: Tensor) -> Tensor:
        """Digital payoff per path, shape ``(n_paths,)``."""
        return (prices[:, -1] > self.strike).to(prices.dtype)
