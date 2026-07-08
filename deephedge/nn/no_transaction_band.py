"""No-transaction band network (Imaki et al. 2021, arXiv:2103.01775)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from deephedge.nn.black_scholes import BlackScholesHedger
from deephedge.nn.mlp_hedger import _ACTIVATIONS


class NoTransactionBandNet(nn.Module):
    """Hedger that trades only when outside a learned band around the BS delta.

    An MLP maps ``(log_moneyness, tau)`` to two non-negative half-widths
    ``(w_lo, w_hi)``. The band is ``[delta_BS - w_lo, delta_BS + w_hi]`` and
    the new hedge is the previous hedge clipped into the band (then into
    ``[0, 1]``), which suppresses small trades under transaction costs.

    Parameters
    ----------
    hidden_size : int
        Width of the MLP hidden layers.
    n_layers : int
        Number of hidden layers.
    activation : str
        One of ``"relu"``, ``"selu"``, ``"tanh"``.
    r : float
        Risk-free rate of the underlying Black-Scholes delta.
    sigma : float
        Implied volatility of the underlying Black-Scholes delta.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        n_layers: int = 2,
        activation: str = "relu",
        r: float = 0.045,
        sigma: float = 0.20,
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation!r}")
        act = _ACTIVATIONS[activation]
        self.bs = BlackScholesHedger(r=r, sigma=sigma)
        layers: list[nn.Module] = []
        width = 2
        for _ in range(n_layers):
            layers.append(nn.Linear(width, hidden_size))
            layers.append(act())
            width = hidden_size
        layers.append(nn.Linear(width, 2))
        self.band_net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the banded hedge ratio for one time step.

        Parameters
        ----------
        x : Tensor
            Features of shape ``(batch, 3)``: ``(delta_prev, log_moneyness,
            tau)``.

        Returns
        -------
        Tensor
            Hedge ratio of shape ``(batch, 1)`` in ``[0, 1]``.
        """
        delta_prev = x[:, 0:1]
        log_m = x[:, 1:2]
        tau = x[:, 2:3]
        delta_bs = torch.special.ndtr(self.bs._d1(log_m, tau))
        widths = F.softplus(self.band_net(torch.cat([log_m, tau], dim=1)))
        lower = delta_bs - widths[:, 0:1]
        upper = delta_bs + widths[:, 1:2]
        delta = torch.minimum(torch.maximum(delta_prev, lower), upper)
        return torch.clamp(delta, 0.0, 1.0)
