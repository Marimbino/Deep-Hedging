"""Lambert-W transforms for gaussianizing heavy-tailed returns.

The Lambert W x Gaussian framework (Goerg 2011) models a heavy-tailed random
variable ``Y`` as ``Y = U * exp(0.5 * delta * U**2)`` where ``U`` is standard
normal. The *inverse* transform recovers the latent Gaussian variable from
observed data, and the IGMM algorithm estimates the tail parameter ``delta``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import lambertw
from scipy.stats import kurtosis

DELTA_MAX = 0.49
_DELTA_EPS = 1e-12


def inverse_lambert_w(y: np.ndarray, delta: float) -> np.ndarray:
    """Map heavy-tailed data to its latent Gaussian representation.

    Computes ``u = sign(y) * sqrt(W(delta * y**2) / delta)`` where ``W`` is the
    principal branch of the Lambert W function.

    Parameters
    ----------
    y : np.ndarray
        Observed (heavy-tailed) data.
    delta : float
        Tail parameter, ``0 <= delta <= 0.49``.

    Returns
    -------
    np.ndarray
        Gaussianized data with the same shape as ``y``.
    """
    y = np.asarray(y, dtype=float)
    if delta < _DELTA_EPS:
        return y.copy()
    w = np.real(lambertw(delta * y**2))
    return np.sign(y) * np.sqrt(w / delta)


def forward_lambert_w(u: np.ndarray, delta: float) -> np.ndarray:
    """Map latent Gaussian data back to the heavy-tailed domain.

    Computes ``y = u * exp(0.5 * delta * u**2)``, the inverse of
    :func:`inverse_lambert_w`.

    Parameters
    ----------
    u : np.ndarray
        Gaussianized data.
    delta : float
        Tail parameter, ``0 <= delta <= 0.49``.

    Returns
    -------
    np.ndarray
        Heavy-tailed data with the same shape as ``u``.
    """
    u = np.asarray(u, dtype=float)
    return u * np.exp(0.5 * delta * u**2)


def igmm(y: np.ndarray, max_iter: int = 200, tol: float = 1e-9) -> float:
    """Estimate the Lambert-W tail parameter via Iterative Generalized Method of Moments.

    Starting from ``delta = 0``, the update rule

    ``delta <- delta + 0.5 * log(1 + max(kurt(u), 0) / 66)``

    is iterated, where ``u = inverse_lambert_w(y, delta)`` and ``kurt`` is the
    excess kurtosis. The estimate is clipped to ``[0, 0.49]`` and iteration
    stops when the update is smaller than ``tol``.

    Parameters
    ----------
    y : np.ndarray
        Observed data (will be standardised internally).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the change in ``delta``.

    Returns
    -------
    float
        The estimated tail parameter ``delta_hat``.
    """
    y = np.asarray(y, dtype=float)
    std = y.std()
    if std <= 0:
        raise ValueError("igmm requires data with positive standard deviation")
    y = (y - y.mean()) / std

    delta = 0.0
    for _ in range(max_iter):
        u = inverse_lambert_w(y, delta)
        k = float(kurtosis(u, fisher=True, bias=True))
        step = 0.5 * np.log1p(max(k, 0.0) / 66.0)
        new_delta = float(np.clip(delta + step, 0.0, DELTA_MAX))
        if abs(new_delta - delta) < tol:
            delta = new_delta
            break
        delta = new_delta
    return delta
