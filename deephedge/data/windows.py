"""Rolling-window construction for time series."""

from __future__ import annotations

import numpy as np


def rolling_windows(x: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Slide a window along a 1-D series.

    Parameters
    ----------
    x : np.ndarray
        1-D input series of length ``N``.
    window : int
        Window length.
    stride : int
        Step between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, window)`` where
        ``n_windows = (N - window) // stride + 1``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < window:
        raise ValueError(f"series of length {x.size} is shorter than window {window}")
    views = np.lib.stride_tricks.sliding_window_view(x, window)
    return views[::stride].copy()


def normalized_price_windows(prices: np.ndarray, n_steps: int, stride: int = 1) -> np.ndarray:
    """Build rolling price windows normalised so the first price equals 1.0.

    Each window has ``n_steps + 1`` points, matching the convention that a
    hedge over ``n_steps`` periods observes ``n_steps + 1`` prices.

    Parameters
    ----------
    prices : np.ndarray
        1-D price series.
    n_steps : int
        Number of hedging steps per window.
    stride : int
        Step between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, n_steps + 1)`` with column 0 equal to 1.
    """
    windows = rolling_windows(prices, n_steps + 1, stride=stride)
    return windows / windows[:, :1]


def train_test_split_windows(
    windows: np.ndarray,
    train_frac: float = 0.8,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into train and test sets by trajectory index.

    The split shuffles trajectory indices (not dates), so overlapping windows
    from different periods can land in either set without look-ahead inside a
    single trajectory.

    Parameters
    ----------
    windows : np.ndarray
        Array of shape ``(n_windows, ...)``.
    train_frac : float
        Fraction of windows assigned to the training set.
    seed : int, optional
        Random seed for the shuffle.

    Returns
    -------
    tuple of np.ndarray
        ``(train, test)`` arrays.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    n_train = int(round(train_frac * len(windows)))
    return windows[idx[:n_train]], windows[idx[n_train:]]
