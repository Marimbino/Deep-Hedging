from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedge.augmentation import QuantWGANGenerator
from deephedge.instruments import (
    BinaryOption,
    BrownianStock,
    EuropeanOption,
    LookbackOption,
    WGANStock,
)


def test_brownian_stock_simulate_shape():
    stock = BrownianStock(sigma=0.2)
    prices = stock.simulate(n_paths=100, n_steps=30, seed=0)
    assert prices.shape == (100, 31)
    torch.testing.assert_close(prices[:, 0], torch.ones(100))
    assert torch.all(prices > 0)


def test_wgan_stock_simulate_shape():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(400) * 0.01
    gen = QuantWGANGenerator(seq_len=15, noise_dim=2, hidden=4, n_layers=1, device="cpu")
    gen.fit_on_returns(returns, epochs=1, batch=64, seed=0)

    stock = WGANStock(gen, cost=1e-4)
    prices = stock.simulate(n_paths=8, n_steps=10, seed=0)
    assert prices.shape == (8, 11)
    torch.testing.assert_close(prices[:, 0], torch.ones(8))

    with pytest.raises(ValueError):
        stock.simulate(n_paths=8, n_steps=99)


def test_european_option_payoff():
    stock = BrownianStock()
    option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)
    prices = torch.tensor([[1.0, 1.1, 1.2], [1.0, 0.9, 0.8]])
    payoff = option.payoff(prices)
    torch.testing.assert_close(payoff, torch.tensor([0.2, 0.0]))


def test_lookback_option_payoff():
    stock = BrownianStock()
    option = LookbackOption(stock, strike=1.0)
    prices = torch.tensor([[1.0, 1.3, 1.1], [1.0, 0.9, 0.95]])
    payoff = option.payoff(prices)
    torch.testing.assert_close(payoff, torch.tensor([0.3, 0.0]))


def test_binary_option_payoff():
    stock = BrownianStock()
    option = BinaryOption(stock, strike=1.0)
    prices = torch.tensor([[1.0, 1.1, 1.2], [1.0, 1.1, 0.8], [1.0, 1.0, 1.0]])
    payoff = option.payoff(prices)
    torch.testing.assert_close(payoff, torch.tensor([1.0, 0.0, 0.0]))


def test_derivative_n_steps():
    stock = BrownianStock(dt=1 / 252)
    option = EuropeanOption(stock, maturity=30 / 252)
    assert option.n_steps == 30
