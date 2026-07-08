"""Matplotlib visualisations for hedging results and stylized facts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import Tensor

if TYPE_CHECKING:
    from deephedge.augmentation.generator import QuantWGANGenerator


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pooled_acf(windows: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation of windowed series, pooled across windows."""
    x = windows - windows.mean()
    denom = (x**2).mean()
    acf = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = (x[:, :-lag] * x[:, lag:]).mean() / denom
    return acf


def _leverage(windows: np.ndarray, max_lag: int) -> np.ndarray:
    """Leverage effect: corr(r_t, r_{t+lag}^2) pooled across windows."""
    lev = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        a = windows[:, :-lag].ravel()
        b = (windows[:, lag:] ** 2).ravel()
        lev[lag - 1] = np.corrcoef(a, b)[0, 1]
    return lev


def plot_training_history(history: dict | pd.DataFrame) -> tuple:
    """4-panel training history: entropic loss, mean PnL, std PnL, CVaR95.

    Parameters
    ----------
    history : dict or pd.DataFrame
        History returned by :meth:`deephedge.Hedger.fit`.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("loss", "Entropic loss"),
        ("mean_pnl", "Mean PnL"),
        ("std_pnl", "Std PnL"),
        ("cvar95", "CVaR 95%"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        if col in df.columns:
            ax.plot(df[col].to_numpy())
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_hedge_ratios(
    deltas_mlp: Tensor | np.ndarray,
    deltas_gru: Tensor | np.ndarray,
    deltas_bs: Tensor | np.ndarray,
    prices: Tensor | np.ndarray,
    path_index: int = 0,
) -> tuple:
    """2-panel plot: hedge ratios over time (top) and the price path (bottom).

    Parameters
    ----------
    deltas_mlp, deltas_gru, deltas_bs : Tensor or np.ndarray
        Hedge ratios of shape ``(n_paths, n_steps)`` from each model.
    prices : Tensor or np.ndarray
        Price paths of shape ``(n_paths, n_steps + 1)``.
    path_index : int
        Which path to display.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for deltas, label in (
        (deltas_mlp, "MLP"),
        (deltas_gru, "GRU"),
        (deltas_bs, "Black-Scholes"),
    ):
        axes[0].plot(_to_numpy(deltas)[path_index], label=label)
    axes[0].set_ylabel("hedge ratio $\\delta_t$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(_to_numpy(prices)[path_index], color="black")
    axes[1].set_ylabel("price $S_t$")
    axes[1].set_xlabel("time step")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_pnl_distribution(pnl_dict: dict[str, Tensor | np.ndarray], bins: int = 80) -> tuple:
    """Overlapping PnL histograms, one per model.

    Parameters
    ----------
    pnl_dict : dict
        Mapping model name -> PnL sample.
    bins : int
        Number of histogram bins.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, pnl in pnl_dict.items():
        ax.hist(_to_numpy(pnl), bins=bins, alpha=0.5, label=name, density=True)
    ax.set_xlabel("PnL")
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_stylized_facts(
    generator: "QuantWGANGenerator",
    real_windows: np.ndarray,
    n_gen: int = 2000,
    max_lag: int = 15,
) -> tuple:
    """5-panel stylized-fact diagnostics of generated vs real returns.

    Panels: marginal distribution (log-scale), QQ-plot, ACF of returns, ACF of
    squared returns, and the leverage effect ``corr(r_t, r_{t+lag}^2)``.

    Parameters
    ----------
    generator : QuantWGANGenerator
        A fitted generator.
    real_windows : np.ndarray
        Real log-return windows of shape ``(n_windows, seq_len)``.
    n_gen : int
        Number of sequences to generate.
    max_lag : int
        Maximum lag for the ACF and leverage panels.

    Returns
    -------
    tuple
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    real_windows = np.asarray(real_windows, dtype=float)
    gen_windows = generator.generate_returns(n_gen)
    real_flat = real_windows.ravel()
    gen_flat = gen_windows.ravel()
    lags = np.arange(1, max_lag + 1)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    # 1. marginal distribution (log-scale y)
    bins = np.linspace(
        min(real_flat.min(), gen_flat.min()), max(real_flat.max(), gen_flat.max()), 80
    )
    axes[0].hist(real_flat, bins=bins, alpha=0.5, label="real", density=True)
    axes[0].hist(gen_flat, bins=bins, alpha=0.5, label="generated", density=True)
    axes[0].set_yscale("log")
    axes[0].set_title("Marginal distribution")
    axes[0].legend()

    # 2. QQ-plot
    q = np.linspace(0.001, 0.999, 200)
    axes[1].scatter(np.quantile(real_flat, q), np.quantile(gen_flat, q), s=8)
    lims = [
        min(np.quantile(real_flat, 0.001), np.quantile(gen_flat, 0.001)),
        max(np.quantile(real_flat, 0.999), np.quantile(gen_flat, 0.999)),
    ]
    axes[1].plot(lims, lims, "k--", linewidth=1)
    axes[1].set_xlabel("real quantiles")
    axes[1].set_ylabel("generated quantiles")
    axes[1].set_title("QQ-plot")

    # 3-4. ACF of returns and squared returns
    for ax, transform, title in (
        (axes[2], lambda w: w, "ACF returns"),
        (axes[3], lambda w: w**2, "ACF squared returns"),
    ):
        ax.plot(lags, _pooled_acf(transform(real_windows), max_lag), "o-", label="real")
        ax.plot(lags, _pooled_acf(transform(gen_windows), max_lag), "s-", label="generated")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("lag")
        ax.set_title(title)
        ax.legend()

    # 5. leverage effect
    axes[4].plot(lags, _leverage(real_windows, max_lag), "o-", label="real")
    axes[4].plot(lags, _leverage(gen_windows, max_lag), "s-", label="generated")
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].set_xlabel("lag")
    axes[4].set_title("Leverage effect")
    axes[4].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, axes


def plot_price_paths(prices: np.ndarray, n_plot: int = 100) -> tuple:
    """Plot a subset of simulated price paths.

    Parameters
    ----------
    prices : np.ndarray
        Price paths of shape ``(n_paths, n_steps + 1)``.
    n_plot : int
        Number of paths to display.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    prices = _to_numpy(prices)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prices[:n_plot].T, linewidth=0.7, alpha=0.6)
    ax.set_xlabel("time step")
    ax.set_ylabel("price")
    ax.set_title(f"{min(n_plot, len(prices))} simulated price paths")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax
