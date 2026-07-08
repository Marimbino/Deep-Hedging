"""Top-level deep hedging engine: training, pricing, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from deephedge.instruments.base import Derivative
from deephedge.nn.risk_measures import EntropicRiskMeasure, ExpectedShortfall
from deephedge.utils.metrics import pnl_metrics


class Hedger(nn.Module):
    """Deep Hedger: wraps a hedging model, a derivative, and a risk measure.

    Parameters
    ----------
    model : nn.Module
        Network mapping features to hedge ratios. Step models consume
        ``(batch, 3)`` features ``(delta_prev, log_moneyness, tau)`` per time
        step; sequence models (with class attribute ``expects_sequence =
        True``, e.g. :class:`~deephedge.nn.GRUHedger`) consume the full path
        ``(batch, n_steps, 2)`` at once.
    derivative : Derivative
        Instrument defining the underlier and the payoff.
    criterion : nn.Module, optional
        Risk measure to minimise (default :class:`EntropicRiskMeasure`).
    """

    def __init__(
        self,
        model: nn.Module,
        derivative: Derivative,
        criterion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.derivative = derivative
        self.criterion = criterion if criterion is not None else EntropicRiskMeasure()

    @property
    def n_steps(self) -> int:
        """Number of hedging steps of the wrapped derivative."""
        return self.derivative.n_steps

    def _unroll(self, prices: Tensor, fee: float) -> tuple[Tensor, Tensor]:
        """Run the hedging strategy along given price paths.

        Parameters
        ----------
        prices : Tensor
            Price paths of shape ``(n_paths, n_steps + 1)``.
        fee : float
            Proportional transaction cost.

        Returns
        -------
        tuple of Tensor
            ``(deltas, pnl)`` with shapes ``(n_paths, n_steps)`` and
            ``(n_paths,)``.
        """
        n_paths, n_cols = prices.shape
        n_steps = n_cols - 1
        strike = self.derivative.strike
        dt = self.derivative.underlier.dt

        if getattr(self.model, "expects_sequence", False):
            log_m = torch.log(prices[:, :-1] / strike)
            tau_values = torch.arange(
                n_steps, 0, -1, dtype=prices.dtype, device=prices.device
            ) * dt
            tau = tau_values.expand(n_paths, n_steps)
            features = torch.stack([log_m, tau], dim=2)
            deltas = self.model(features).squeeze(-1)
        else:
            delta = torch.zeros(n_paths, 1, dtype=prices.dtype, device=prices.device)
            steps: list[Tensor] = []
            for t in range(n_steps):
                log_m = torch.log(prices[:, t : t + 1] / strike)
                tau = torch.full_like(log_m, (n_steps - t) * dt)
                delta = self.model(torch.cat([delta, log_m, tau], dim=1))
                steps.append(delta)
            deltas = torch.cat(steps, dim=1)

        prev = torch.cat([torch.zeros_like(deltas[:, :1]), deltas[:, :-1]], dim=1)
        gains = (deltas * (prices[:, 1:] - prices[:, :-1])).sum(dim=1)
        costs = (fee * (deltas - prev).abs() * prices[:, :-1]).sum(dim=1)
        pnl = gains - costs - self.derivative.payoff(prices)
        return deltas, pnl

    def compute_pnl(self, prices: Tensor, fee: float = 0.0) -> Tensor:
        """Simulate hedging on given price paths and return the PnL per path.

        ``pnl = sum_t [delta_t (S_{t+1} - S_t) - fee |delta_t - delta_{t-1}|
        S_t] - payoff``.

        Parameters
        ----------
        prices : Tensor
            Price paths of shape ``(n_paths, n_steps + 1)``.
        fee : float
            Proportional transaction cost.

        Returns
        -------
        Tensor
            PnL of shape ``(n_paths,)``. Gradients flow through the model.
        """
        return self._unroll(prices, fee)[1]

    def fit(
        self,
        n_paths: int = 40_000,
        n_epochs: int = 200,
        batch_size: int = 512,
        lr: float = 1e-3,
        fee: float = 0.001,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        seed: int | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Train the hedging model by minimising the risk measure.

        Parameters
        ----------
        n_paths : int
            Number of training paths simulated from the underlier.
        n_epochs : int
            Number of passes over the paths.
        batch_size : int
            Mini-batch size.
        lr : float
            Learning rate.
        fee : float
            Proportional transaction cost used during training.
        optimizer_cls : type
            Optimizer class (default Adam).
        seed : int, optional
            Random seed for path simulation and shuffling.
        verbose : bool
            Print progress every 10% of epochs.

        Returns
        -------
        pd.DataFrame
            Epoch-level history with columns ``epoch``, ``loss``,
            ``mean_pnl``, ``std_pnl``, ``cvar95``.
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError(
                "model has no trainable parameters; analytic hedgers such as "
                "BlackScholesHedger can only be evaluated, not fitted"
            )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        prices = self.derivative.simulate(n_paths)
        device = next(self.model.parameters()).device
        prices = prices.to(device)

        optimizer = optimizer_cls(params, lr=lr)
        es95 = ExpectedShortfall(alpha=0.95)
        rows: list[dict[str, float]] = []

        for epoch in range(n_epochs):
            perm = torch.randperm(n_paths, device=device)
            batch_losses: list[float] = []
            epoch_pnl: list[Tensor] = []

            for start in range(0, n_paths, batch_size):
                batch = prices[perm[start : start + batch_size]]
                pnl = self.compute_pnl(batch, fee=fee)
                loss = self.criterion(pnl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach()))
                epoch_pnl.append(pnl.detach())

            all_pnl = torch.cat(epoch_pnl)
            rows.append(
                {
                    "epoch": epoch,
                    "loss": float(np.mean(batch_losses)),
                    "mean_pnl": float(all_pnl.mean()),
                    "std_pnl": float(all_pnl.std()),
                    "cvar95": float(es95(all_pnl)),
                }
            )
            if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
                print(
                    f"epoch {epoch + 1:4d}/{n_epochs} | "
                    f"loss: {rows[-1]['loss']:+.5f} | "
                    f"mean PnL: {rows[-1]['mean_pnl']:+.5f} | "
                    f"std PnL: {rows[-1]['std_pnl']:.5f} | "
                    f"CVaR95: {rows[-1]['cvar95']:+.5f}"
                )

        return pd.DataFrame(rows)

    def price(self, n_paths: int = 50_000, fee: float | None = None, seed: int | None = None) -> float:
        """Utility-indifference price of the derivative.

        The indifference price ``p*`` satisfies ``rho(PnL_hedged + p*) =
        rho(0) = 0``, hence by cash invariance ``p* = rho(PnL_hedged)``. For
        the entropic risk measure this is
        ``(1/lambda) log E[exp(lambda (payoff - hedge_gain))]``.

        Parameters
        ----------
        n_paths : int
            Number of fresh paths to price on.
        fee : float, optional
            Transaction cost; defaults to the underlier's ``cost``.
        seed : int, optional
            Random seed for path simulation.

        Returns
        -------
        float
            The indifference price (positive for a long option payoff).
        """
        if fee is None:
            fee = self.derivative.underlier.cost
        with torch.no_grad():
            prices = self.derivative.simulate(n_paths, seed=seed)
            pnl = self.compute_pnl(prices, fee=fee)
            return float(self.criterion(pnl))

    def evaluate(
        self,
        n_paths: int = 10_000,
        fee: float = 0.001,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Compute the full metrics dict on fresh paths.

        Parameters
        ----------
        n_paths : int
            Number of evaluation paths.
        fee : float
            Proportional transaction cost.
        seed : int, optional
            Random seed for path simulation.

        Returns
        -------
        dict
            Keys ``mean``, ``std``, ``entropic``, ``cvar95``, ``turnover``.
        """
        risk_aversion = getattr(self.criterion, "risk_aversion", 10.0)
        with torch.no_grad():
            prices = self.derivative.simulate(n_paths, seed=seed)
            deltas, pnl = self._unroll(prices, fee=fee)
            prev = torch.cat([torch.zeros_like(deltas[:, :1]), deltas[:, :-1]], dim=1)
            turnover = float((deltas - prev).abs().sum(dim=1).mean())
        return pnl_metrics(pnl, risk_aversion=risk_aversion, alpha=0.95, turnover=turnover)

    def hedge_path(self, prices: Tensor, fee: float | None = None) -> tuple[Tensor, Tensor]:
        """Hedge a given set of paths without tracking gradients.

        Parameters
        ----------
        prices : Tensor
            Price paths of shape ``(n_paths, n_steps + 1)``.
        fee : float, optional
            Transaction cost; defaults to the underlier's ``cost``.

        Returns
        -------
        tuple of Tensor
            ``(deltas, pnl)`` with shapes ``(n_paths, n_steps)`` and
            ``(n_paths,)``.
        """
        if fee is None:
            fee = self.derivative.underlier.cost
        with torch.no_grad():
            return self._unroll(prices, fee=fee)
