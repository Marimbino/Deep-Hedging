from __future__ import annotations

import numpy as np
import pytest
import torch

from deephedge.augmentation import (
    TCN,
    Critic,
    Generator,
    QuantWGANGenerator,
    WGANTrainer,
    gradient_penalty,
)


def test_tcn_output_shape():
    tcn = TCN(1, 1, hidden=8, n_layers=3)
    x = torch.randn(4, 1, 30)
    out = tcn(x)
    assert out.shape == (4, 1, 30)


def test_causal_conv_causality():
    tcn = TCN(1, 1, hidden=8, n_layers=3).eval()
    t = 10
    x = torch.randn(2, 1, 30)
    x_perturbed = x.clone()
    x_perturbed[:, :, t + 1 :] += 100.0
    with torch.no_grad():
        out = tcn(x)
        out_perturbed = tcn(x_perturbed)
    torch.testing.assert_close(out[:, :, : t + 1], out_perturbed[:, :, : t + 1])


def test_generator_output_shape():
    gen = Generator(noise_dim=5, hidden=8, n_layers=2)
    z = torch.randn(8, 5, 30)
    out = gen(z)
    assert out.shape == (8, 1, 30)


def test_critic_output_shape():
    critic = Critic(hidden=8, n_layers=2)
    x = torch.randn(8, 1, 30)
    out = critic(x)
    assert out.shape == (8,)


def test_gradient_penalty_nonzero():
    critic = Critic(hidden=8, n_layers=2)
    real = torch.randn(16, 1, 20)
    fake = torch.randn(16, 1, 20)
    gp = gradient_penalty(critic, real, fake)
    assert gp.item() > 0.0
    assert gp.requires_grad


def test_wgan_trainer_runs():
    gen = Generator(noise_dim=3, hidden=8, n_layers=2)
    critic = Critic(hidden=8, n_layers=2)
    trainer = WGANTrainer(gen, critic, noise_dim=3, n_critic=2, device="cpu")
    data = torch.randn(64, 1, 20)
    history = trainer.fit(data, epochs=2, batch=32, seed=0)
    assert set(history) == {"w_dist", "loss_c", "loss_g", "gp"}
    for values in history.values():
        assert len(values) == 2
        assert all(np.isfinite(v) for v in values)


@pytest.fixture(scope="module")
def tiny_fitted_generator():
    rng = np.random.default_rng(0)
    returns = rng.standard_t(5, 600) * 0.01
    gen = QuantWGANGenerator(seq_len=20, noise_dim=3, hidden=8, n_layers=2, device="cpu")
    gen.fit_on_returns(returns, epochs=1, batch=64, seed=0)
    return gen


def test_generate_prices_shape(tiny_fitted_generator):
    prices = tiny_fitted_generator.generate(n_paths=16, S0=1.0, seed=0)
    assert prices.shape == (16, 21)
    np.testing.assert_allclose(prices[:, 0], 1.0)
    assert np.all(prices > 0)


def test_generator_save_load(tiny_fitted_generator, tmp_path):
    tiny_fitted_generator.save(tmp_path / "model")
    loaded = QuantWGANGenerator.load(tmp_path / "model", device="cpu")
    a = tiny_fitted_generator.generate(8, seed=123)
    b = loaded.generate(8, seed=123)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)


def test_generate_unfitted_raises():
    gen = QuantWGANGenerator(seq_len=10, noise_dim=2, hidden=4, n_layers=1, device="cpu")
    with pytest.raises(RuntimeError):
        gen.generate(4)
