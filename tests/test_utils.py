from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from deephedge.augmentation import QuantWGANGenerator, WGANTrainer
from deephedge.data import rolling_windows
from deephedge.utils import (
    plot_hedge_ratios,
    plot_pnl_distribution,
    plot_price_paths,
    plot_stylized_facts,
    plot_training_history,
    pnl_metrics,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_pnl_metrics_keys_and_values():
    pnl = torch.zeros(100)
    metrics = pnl_metrics(pnl, turnover=2.5)
    assert set(metrics) == {"mean", "std", "entropic", "cvar95", "turnover"}
    assert metrics["mean"] == pytest.approx(0.0)
    assert metrics["entropic"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["turnover"] == 2.5


def test_pnl_metrics_turnover_default_nan():
    metrics = pnl_metrics(np.random.default_rng(0).normal(size=500))
    assert math.isnan(metrics["turnover"])


def test_plot_training_history():
    history = pd.DataFrame(
        {
            "epoch": range(5),
            "loss": np.linspace(1, 0.5, 5),
            "mean_pnl": np.zeros(5),
            "std_pnl": np.linspace(0.1, 0.05, 5),
            "cvar95": np.linspace(0.2, 0.1, 5),
        }
    )
    fig, axes = plot_training_history(history)
    assert axes.size == 4


def test_plot_hedge_ratios():
    deltas = torch.rand(3, 30)
    prices = torch.rand(3, 31) + 0.5
    fig, axes = plot_hedge_ratios(deltas, deltas, deltas, prices)
    assert len(axes) == 2


def test_plot_pnl_distribution():
    pnl_dict = {"MLP": torch.randn(500), "BS": np.random.default_rng(0).normal(size=500)}
    fig, ax = plot_pnl_distribution(pnl_dict)
    assert len(ax.get_legend().get_texts()) == 2


def test_plot_price_paths():
    prices = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, (50, 31)), axis=1)
    fig, ax = plot_price_paths(prices, n_plot=10)
    assert len(ax.get_lines()) == 10


def test_plot_stylized_facts_and_wgan_history():
    rng = np.random.default_rng(0)
    returns = rng.standard_t(5, 500) * 0.01
    gen = QuantWGANGenerator(seq_len=15, noise_dim=2, hidden=4, n_layers=1, device="cpu")
    history = gen.fit_on_returns(returns, epochs=1, batch=64, seed=0)

    real_windows = rolling_windows(returns, 15)
    fig, axes = plot_stylized_facts(gen, real_windows, n_gen=50, max_lag=5)
    assert len(axes) == 5

    fig2, axes2 = WGANTrainer.plot_history(history)
    assert axes2.size == 4

    fig3, axes3 = gen.evaluate(n_gen=50)
    assert len(axes3) == 5
