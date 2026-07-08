"""PnL evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def pnl_metrics(
    pnl: Tensor | np.ndarray,
    risk_aversion: float = 10.0,
    alpha: float = 0.95,
    turnover: float | None = None,
) -> dict[str, float]:
    """Summarise a profit-and-loss sample.

    Parameters
    ----------
    pnl : Tensor or np.ndarray
        PnL per path.
    risk_aversion : float
        Lambda of the entropic risk measure.
    alpha : float
        Confidence level of the CVaR.
    turnover : float, optional
        Total absolute trades (computed separately from the hedge ratios);
        reported as NaN when not provided.

    Returns
    -------
    dict
        Keys ``mean``, ``std``, ``entropic``, ``cvar95``, ``turnover``.
        ``entropic`` and ``cvar95`` are risks: positive values are losses.
    """
    pnl_t = torch.as_tensor(pnl, dtype=torch.float64).ravel()
    n = pnl_t.numel()
    entropic = (torch.logsumexp(-risk_aversion * pnl_t, dim=0) - math.log(n)) / risk_aversion
    # small epsilon guards against fp artifacts, e.g. 0.05 * 100 = 5.000000000000001
    k = max(1, math.ceil((1.0 - alpha) * n - 1e-9))
    cvar = torch.topk(-pnl_t, k).values.mean()
    return {
        "mean": float(pnl_t.mean()),
        "std": float(pnl_t.std()),
        "entropic": float(entropic),
        "cvar95": float(cvar),
        "turnover": float(turnover) if turnover is not None else float("nan"),
    }
