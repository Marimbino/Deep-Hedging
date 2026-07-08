"""Analytic Black-Scholes delta hedger."""

from __future__ import annotations

import torch
from torch import Tensor, nn

_TAU_MIN = 1e-8


class BlackScholesHedger(nn.Module):
    """Analytic Black-Scholes delta — no trainable parameters.

    ``d1 = (log(S/K) + (r + sigma^2 / 2) tau) / (sigma sqrt(tau))`` and
    ``delta = Phi(d1)``.

    Parameters
    ----------
    r : float
        Risk-free rate.
    sigma : float
        Implied volatility.
    """

    def __init__(self, r: float = 0.045, sigma: float = 0.20) -> None:
        super().__init__()
        self.r = r
        self.sigma = sigma

    def _d1(self, log_moneyness: Tensor, tau: Tensor) -> Tensor:
        tau = torch.clamp(tau, min=_TAU_MIN)
        return (log_moneyness + (self.r + 0.5 * self.sigma**2) * tau) / (
            self.sigma * torch.sqrt(tau)
        )

    def delta(self, S: Tensor | float, K: Tensor | float, tau: Tensor | float) -> Tensor:
        """Call delta ``Phi(d1)``.

        Parameters
        ----------
        S : Tensor or float
            Spot price.
        K : Tensor or float
            Strike price.
        tau : Tensor or float
            Time to maturity in years.

        Returns
        -------
        Tensor
            The Black-Scholes hedge ratio.
        """
        S = torch.as_tensor(S, dtype=torch.get_default_dtype())
        K = torch.as_tensor(K, dtype=S.dtype)
        tau = torch.as_tensor(tau, dtype=S.dtype)
        return torch.special.ndtr(self._d1(torch.log(S / K), tau))

    def price(self, S: Tensor | float, K: Tensor | float, tau: Tensor | float) -> Tensor:
        """Black-Scholes call price.

        Parameters
        ----------
        S : Tensor or float
            Spot price.
        K : Tensor or float
            Strike price.
        tau : Tensor or float
            Time to maturity in years.

        Returns
        -------
        Tensor
            ``S Phi(d1) - K exp(-r tau) Phi(d2)``.
        """
        S = torch.as_tensor(S, dtype=torch.get_default_dtype())
        K = torch.as_tensor(K, dtype=S.dtype)
        tau = torch.clamp(torch.as_tensor(tau, dtype=S.dtype), min=_TAU_MIN)
        d1 = self._d1(torch.log(S / K), tau)
        d2 = d1 - self.sigma * torch.sqrt(tau)
        return S * torch.special.ndtr(d1) - K * torch.exp(-self.r * tau) * torch.special.ndtr(d2)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the analytic delta from step features.

        Parameters
        ----------
        x : Tensor
            Features of shape ``(batch, 3)``: ``(delta_prev, log_moneyness,
            tau)``. The previous delta is ignored.

        Returns
        -------
        Tensor
            Delta of shape ``(batch, 1)``.
        """
        log_m = x[:, 1:2]
        tau = x[:, 2:3]
        return torch.special.ndtr(self._d1(log_m, tau))
