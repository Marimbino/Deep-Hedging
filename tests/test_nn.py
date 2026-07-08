from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedge.nn import (
    BlackScholesHedger,
    EntropicRiskMeasure,
    ExpectedShortfall,
    GRUHedger,
    MLPHedger,
    NoTransactionBandNet,
    log_moneyness,
    time_to_maturity,
)


def test_mlp_hedger_output_shape():
    model = MLPHedger()
    x = torch.randn(16, 3)
    out = model(x)
    assert out.shape == (16, 1)
    assert torch.all(out > 0) and torch.all(out < 1)


def test_gru_hedger_output_shape():
    model = GRUHedger()
    x = torch.randn(16, 30, 2)
    out = model(x)
    assert out.shape == (16, 30, 1)
    assert torch.all(out > 0) and torch.all(out < 1)


def test_bs_hedger_delta_atm():
    bs = BlackScholesHedger(r=0.0, sigma=0.20)
    delta = bs.delta(1.0, 1.0, 0.5)
    assert abs(float(delta) - 0.5) < 0.05


def test_bs_hedger_price_positive_and_monotone():
    bs = BlackScholesHedger(r=0.045, sigma=0.20)
    p_atm = float(bs.price(1.0, 1.0, 30 / 252))
    p_itm = float(bs.price(1.1, 1.0, 30 / 252))
    assert p_atm > 0
    assert p_itm > p_atm


def test_bs_hedger_forward_ignores_prev_delta():
    bs = BlackScholesHedger()
    x1 = torch.tensor([[0.0, 0.05, 0.1]])
    x2 = torch.tensor([[0.9, 0.05, 0.1]])
    torch.testing.assert_close(bs(x1), bs(x2))


def test_no_transaction_band_output():
    model = NoTransactionBandNet()
    x = torch.cat(
        [torch.rand(16, 1), torch.randn(16, 1) * 0.1, torch.rand(16, 1) * 0.2 + 0.01], dim=1
    )
    out = model(x)
    assert out.shape == (16, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_entropic_risk_measure():
    risk = EntropicRiskMeasure(risk_aversion=10.0)
    pnl = torch.zeros(1_000)
    assert abs(float(risk(pnl))) < 1e-6

    # deterministic loss of c has risk exactly c
    pnl_loss = torch.full((1_000,), -0.3)
    assert float(risk(pnl_loss)) == pytest.approx(0.3, abs=1e-6)

    with pytest.raises(ValueError):
        EntropicRiskMeasure(risk_aversion=-1.0)


def test_expected_shortfall():
    es = ExpectedShortfall(alpha=0.95)
    pnl = torch.cat([torch.full((5,), -1.0), torch.full((95,), 1.0)])
    assert float(es(pnl)) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        ExpectedShortfall(alpha=1.5)


def test_feature_extractors():
    S = torch.tensor([1.0, 1.1])
    lm = log_moneyness(S, 1.0)
    torch.testing.assert_close(lm, torch.log(S))
    tau = time_to_maturity(t=10, n_steps=30, dt=1 / 252)
    assert float(tau) == pytest.approx(20 / 252)
