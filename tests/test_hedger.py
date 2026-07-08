from __future__ import annotations

import pandas as pd
import pytest
import torch

from deephedge import (
    BlackScholesHedger,
    BrownianStock,
    EuropeanOption,
    GRUHedger,
    Hedger,
    MLPHedger,
)


@pytest.fixture()
def small_setup():
    stock = BrownianStock(sigma=0.2, cost=0.0)
    option = EuropeanOption(stock, strike=1.0, maturity=10 / 252)
    return stock, option


def test_hedger_fit_runs(small_setup):
    _, option = small_setup
    hedger = Hedger(MLPHedger(), option)
    history = hedger.fit(n_paths=256, n_epochs=2, batch_size=128, seed=0, verbose=False)
    assert isinstance(history, pd.DataFrame)
    assert len(history) == 2
    assert {"epoch", "loss", "mean_pnl", "std_pnl", "cvar95"} <= set(history.columns)


def test_hedger_fit_gru_runs(small_setup):
    _, option = small_setup
    hedger = Hedger(GRUHedger(), option)
    history = hedger.fit(n_paths=256, n_epochs=2, batch_size=128, seed=0, verbose=False)
    assert len(history) == 2


def test_hedger_compute_pnl_shape(small_setup):
    stock, option = small_setup
    hedger = Hedger(MLPHedger(), option)
    prices = stock.simulate(64, option.n_steps, seed=0)
    pnl = hedger.compute_pnl(prices, fee=0.001)
    assert pnl.shape == (64,)
    assert torch.isfinite(pnl).all()


def test_hedger_price_positive(small_setup):
    _, option = small_setup
    hedger = Hedger(MLPHedger(), option)
    hedger.fit(n_paths=512, n_epochs=3, batch_size=256, seed=0, verbose=False)
    price = hedger.price(n_paths=2_000, seed=1)
    assert price > 0


def test_hedger_evaluate_keys(small_setup):
    _, option = small_setup
    hedger = Hedger(MLPHedger(), option)
    metrics = hedger.evaluate(n_paths=500, fee=0.001, seed=0)
    assert set(metrics) == {"mean", "std", "entropic", "cvar95", "turnover"}
    assert metrics["turnover"] > 0


def test_hedger_hedge_path(small_setup):
    stock, option = small_setup
    hedger = Hedger(MLPHedger(), option)
    prices = stock.simulate(32, option.n_steps, seed=0)
    deltas, pnl = hedger.hedge_path(prices)
    assert deltas.shape == (32, option.n_steps)
    assert pnl.shape == (32,)
    assert not deltas.requires_grad


def test_hedger_bs_model_evaluates(small_setup):
    _, option = small_setup
    hedger = Hedger(BlackScholesHedger(r=0.0, sigma=0.2), option)
    metrics = hedger.evaluate(n_paths=500, fee=0.0, seed=0)
    assert abs(metrics["mean"]) < 0.05

    with pytest.raises(ValueError):
        hedger.fit(n_paths=64, n_epochs=1)
