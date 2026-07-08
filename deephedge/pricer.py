"""Utility-indifference pricing and Greeks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from deephedge.hedger import Hedger
from deephedge.instruments.base import Derivative
from deephedge.nn.black_scholes import BlackScholesHedger


class DerivativePricer:
    """Prices derivatives with a fitted :class:`Hedger`.

    The utility-indifference price ``p*`` satisfies
    ``rho(PnL_hedged + p*) = rho(0)``, which by cash invariance gives
    ``p* = rho(PnL_hedged)``. For the entropic risk measure:
    ``p* = (1/lambda) log E[exp(lambda (payoff - hedge_gain))]``.
    """

    def price(
        self,
        hedger: Hedger,
        derivative: Derivative | None = None,
        n_paths: int = 50_000,
        seed: int | None = None,
    ) -> float:
        """Indifference price of a derivative under a hedging strategy.

        Parameters
        ----------
        hedger : Hedger
            A (typically fitted) hedger.
        derivative : Derivative, optional
            Instrument to price; defaults to the hedger's own derivative.
        n_paths : int
            Number of fresh Monte-Carlo paths.
        seed : int, optional
            Random seed.

        Returns
        -------
        float
            The utility-indifference price.
        """
        if derivative is None or derivative is hedger.derivative:
            return hedger.price(n_paths=n_paths, seed=seed)
        temp = Hedger(hedger.model, derivative, hedger.criterion)
        return temp.price(n_paths=n_paths, seed=seed)

    def price_vs_black_scholes(
        self,
        hedger: Hedger,
        derivative: Derivative | None = None,
        S_range: np.ndarray | None = None,
        n_paths: int = 10_000,
        r: float = 0.045,
        sigma: float = 0.20,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Compare NN indifference prices to Black-Scholes over a spot range.

        The same base paths (started at 1.0) are rescaled to each spot value,
        so the comparison is not distorted by Monte-Carlo noise across spots.

        Parameters
        ----------
        hedger : Hedger
            A fitted hedger.
        derivative : Derivative, optional
            Instrument to price; defaults to the hedger's derivative.
        S_range : np.ndarray, optional
            Spot values (default ``linspace(0.8, 1.2, 9)``).
        n_paths : int
            Number of Monte-Carlo paths.
        r : float
            Risk-free rate for the Black-Scholes benchmark.
        sigma : float
            Volatility for the Black-Scholes benchmark.
        seed : int, optional
            Random seed.

        Returns
        -------
        pd.DataFrame
            Columns ``spot``, ``nn_price``, ``bs_price``.
        """
        derivative = derivative if derivative is not None else hedger.derivative
        if S_range is None:
            S_range = np.linspace(0.8, 1.2, 9)
        temp = Hedger(hedger.model, derivative, hedger.criterion)
        bs = BlackScholesHedger(r=r, sigma=sigma)
        fee = derivative.underlier.cost

        with torch.no_grad():
            base = derivative.simulate(n_paths, seed=seed)
            rows = []
            for spot in np.asarray(S_range, dtype=float):
                scaled = base * spot
                pnl = temp.compute_pnl(scaled, fee=fee)
                nn_price = float(temp.criterion(pnl))
                bs_price = float(bs.price(spot, derivative.strike, derivative.maturity))
                rows.append({"spot": spot, "nn_price": nn_price, "bs_price": bs_price})
        return pd.DataFrame(rows)

    def greeks(
        self,
        hedger: Hedger,
        derivative: Derivative | None = None,
        spot: float = 1.0,
        n_paths: int = 10_000,
        method: str = "autograd",
        seed: int | None = None,
    ) -> dict[str, float]:
        """Delta and gamma of the indifference price via automatic differentiation.

        The price is viewed as a function of the initial spot ``S0`` by
        rescaling simulated unit paths, and differentiated with
        ``torch.autograd``.

        Parameters
        ----------
        hedger : Hedger
            A fitted hedger.
        derivative : Derivative, optional
            Instrument; defaults to the hedger's derivative.
        spot : float
            Spot value at which to evaluate the Greeks.
        n_paths : int
            Number of Monte-Carlo paths.
        method : str
            Only ``"autograd"`` is supported.
        seed : int, optional
            Random seed.

        Returns
        -------
        dict
            Keys ``price``, ``delta``, ``gamma``.

        Notes
        -----
        Gamma requires a twice-differentiable model; with ReLU activations the
        second derivative is zero almost everywhere, so prefer ``"tanh"`` or
        ``"selu"`` hedgers when gamma matters.
        """
        if method != "autograd":
            raise ValueError(f"unsupported method {method!r}; only 'autograd' is available")
        derivative = derivative if derivative is not None else hedger.derivative
        temp = Hedger(hedger.model, derivative, hedger.criterion)
        fee = derivative.underlier.cost

        base = derivative.simulate(n_paths, seed=seed)
        spot_t = torch.tensor(float(spot), requires_grad=True, dtype=base.dtype)
        prices = base * spot_t
        pnl = temp.compute_pnl(prices, fee=fee)
        price = temp.criterion(pnl)

        (delta,) = torch.autograd.grad(price, spot_t, create_graph=True)
        (gamma,) = torch.autograd.grad(delta, spot_t, allow_unused=True)
        return {
            "price": float(price.detach()),
            "delta": float(delta.detach()),
            "gamma": float(gamma.detach()) if gamma is not None else 0.0,
        }
