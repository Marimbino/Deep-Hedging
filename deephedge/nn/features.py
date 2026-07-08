"""Feature extractors for hedging models."""

from __future__ import annotations

import torch
from torch import Tensor


def log_moneyness(S: Tensor, K: float) -> Tensor:
    """Log-moneyness ``log(S / K)``.

    Parameters
    ----------
    S : Tensor
        Spot prices (any shape).
    K : float
        Strike price.

    Returns
    -------
    Tensor
        ``log(S / K)`` with the same shape as ``S``.
    """
    return torch.log(S / K)


def time_to_maturity(t: int, n_steps: int, dt: float = 1 / 252) -> Tensor:
    """Remaining time to maturity in years at step ``t``.

    Parameters
    ----------
    t : int
        Current step index (``0 <= t < n_steps``).
    n_steps : int
        Total number of hedging steps.
    dt : float
        Length of one step in years.

    Returns
    -------
    Tensor
        Scalar tensor ``(n_steps - t) * dt``.
    """
    return torch.tensor((n_steps - t) * dt)


def prev_hedge(delta: Tensor) -> Tensor:
    """Previous hedge ratio, passed through unchanged.

    Parameters
    ----------
    delta : Tensor
        Hedge ratio from the previous step.

    Returns
    -------
    Tensor
        The same tensor.
    """
    return delta
