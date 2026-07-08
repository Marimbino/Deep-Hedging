"""Convex risk measures used as training criteria and evaluation metrics."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class EntropicRiskMeasure(nn.Module):
    """Entropic risk measure ``rho(X) = (1/lambda) log E[exp(-lambda X)]``.

    Minimising this risk over hedging strategies is equivalent to maximising
    an exponential utility with risk aversion ``lambda``.

    Parameters
    ----------
    risk_aversion : float
        The risk-aversion parameter ``lambda`` (must be positive).
    """

    def __init__(self, risk_aversion: float = 10.0) -> None:
        super().__init__()
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        self.risk_aversion = risk_aversion

    def forward(self, pnl: Tensor) -> Tensor:
        """Risk of a profit-and-loss sample.

        Parameters
        ----------
        pnl : Tensor
            PnL per path, shape ``(n_paths,)``.

        Returns
        -------
        Tensor
            Scalar risk (computed via log-sum-exp for numerical stability).
        """
        lam = self.risk_aversion
        n = pnl.numel()
        return (torch.logsumexp(-lam * pnl, dim=0) - math.log(n)) / lam


class ExpectedShortfall(nn.Module):
    """Expected shortfall (CVaR): mean loss beyond the ``alpha``-quantile.

    A positive value is a loss. For a sample of ``N`` paths the worst
    ``ceil((1 - alpha) * N)`` losses are averaged, which keeps the measure
    differentiable almost everywhere.

    Parameters
    ----------
    alpha : float
        Confidence level in ``(0, 1)`` (default 0.95).
    """

    def __init__(self, alpha: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self.alpha = alpha

    def forward(self, pnl: Tensor) -> Tensor:
        """Expected shortfall of a profit-and-loss sample.

        Parameters
        ----------
        pnl : Tensor
            PnL per path, shape ``(n_paths,)``.

        Returns
        -------
        Tensor
            Scalar expected shortfall (positive = loss).
        """
        loss = -pnl
        n = loss.numel()
        # small epsilon guards against fp artifacts, e.g. 0.05 * 100 = 5.000000000000001
        k = max(1, math.ceil((1.0 - self.alpha) * n - 1e-9))
        worst = torch.topk(loss, k).values
        return worst.mean()


CVaR = ExpectedShortfall
"""Alias: conditional value-at-risk is the expected shortfall."""
