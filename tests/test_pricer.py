from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedge import (
    BlackScholesHedger,
    BrownianStock,
    DerivativePricer,
    EuropeanOption,
    Hedger,
    MLPHedger,
)


@pytest.fixture(scope="module")
def fitted_hedger():
    torch.manual_seed(0)
    stock = BrownianStock(sigma=0.2, cost=0.0)
    option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)
    hedger = Hedger(MLPHedger(), option)
    hedger.fit(n_paths=4_000, n_epochs=15, batch_size=512, fee=0.0, seed=0, verbose=False)
    return hedger


def test_pricer_vs_black_scholes_atm(fitted_hedger):
    pricer = DerivativePricer()
    nn_price = pricer.price(fitted_hedger, n_paths=20_000, seed=1)
    # driftless GBM is already risk-neutral with r = 0
    bs = BlackScholesHedger(r=0.0, sigma=0.2)
    bs_price = float(bs.price(1.0, 1.0, 30 / 252))
    assert bs_price > 0
    assert abs(nn_price - bs_price) / bs_price < 0.15


def test_price_vs_black_scholes_dataframe(fitted_hedger):
    pricer = DerivativePricer()
    df = pricer.price_vs_black_scholes(
        fitted_hedger, S_range=np.array([0.9, 1.0, 1.1]), n_paths=2_000, r=0.0, seed=1
    )
    assert list(df.columns) == ["spot", "nn_price", "bs_price"]
    assert len(df) == 3
    # option value increases with spot
    assert df["nn_price"].is_monotonic_increasing
    assert df["bs_price"].is_monotonic_increasing


def test_greeks_autograd(fitted_hedger):
    pricer = DerivativePricer()
    greeks = pricer.greeks(fitted_hedger, spot=1.0, n_paths=2_000, seed=1)
    assert set(greeks) == {"price", "delta", "gamma"}
    # ATM call delta should be around 0.5
    assert 0.2 < greeks["delta"] < 0.8

    with pytest.raises(ValueError):
        pricer.greeks(fitted_hedger, method="finite-difference")
