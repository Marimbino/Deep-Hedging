"""Abstract base classes for tradeable assets and derivatives."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Primary(ABC, nn.Module):
    """A tradeable underlying asset.

    Parameters
    ----------
    cost : float
        Proportional transaction cost per unit traded.
    dt : float
        Time step in years (default one trading day, ``1/252``).
    """

    def __init__(self, cost: float = 0.0, dt: float = 1 / 252) -> None:
        super().__init__()
        self.cost = cost
        self.dt = dt

    @abstractmethod
    def simulate(self, n_paths: int, n_steps: int, seed: int | None = None) -> Tensor:
        """Simulate price paths.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        n_steps : int
            Number of time steps per path.
        seed : int, optional
            Random seed.

        Returns
        -------
        Tensor
            Price tensor of shape ``(n_paths, n_steps + 1)``.
        """
        ...


class Derivative(ABC, nn.Module):
    """A derivative instrument written on a :class:`Primary`.

    Parameters
    ----------
    underlier : Primary
        The underlying asset.
    strike : float
        Strike price.
    maturity : float
        Time to maturity in years.
    """

    def __init__(self, underlier: Primary, strike: float = 1.0, maturity: float = 30 / 252) -> None:
        super().__init__()
        self.underlier = underlier
        self.strike = strike
        self.maturity = maturity

    @property
    def n_steps(self) -> int:
        """Number of hedging steps implied by maturity and the underlier's dt."""
        return max(1, round(self.maturity / self.underlier.dt))

    def simulate(self, n_paths: int, seed: int | None = None) -> Tensor:
        """Simulate underlier paths over the derivative's life.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        seed : int, optional
            Random seed.

        Returns
        -------
        Tensor
            Price tensor of shape ``(n_paths, n_steps + 1)``.
        """
        return self.underlier.simulate(n_paths, self.n_steps, seed=seed)

    @abstractmethod
    def payoff(self, prices: Tensor) -> Tensor:
        """Terminal payoff for each path.

        Parameters
        ----------
        prices : Tensor
            Price paths of shape ``(n_paths, n_steps + 1)``.

        Returns
        -------
        Tensor
            Payoff of shape ``(n_paths,)``.
        """
        ...
