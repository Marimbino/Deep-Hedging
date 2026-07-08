from __future__ import annotations

import numpy as np
import pytest

from deephedge.preprocessing import (
    ReturnPreprocessor,
    forward_lambert_w,
    igmm,
    inverse_lambert_w,
)


def test_igmm_converges():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(20_000)
    delta_hat = igmm(y)
    assert 0.0 <= delta_hat < 0.05


def test_igmm_heavy_tail():
    rng = np.random.default_rng(42)
    y = rng.standard_t(3, 20_000)
    delta_hat = igmm(y)
    assert delta_hat > 0.05
    assert delta_hat <= 0.49


def test_inverse_lambert_w_identity():
    rng = np.random.default_rng(0)
    y = rng.standard_t(4, 5_000)
    delta = 0.2
    u = inverse_lambert_w(y, delta)
    roundtrip = forward_lambert_w(u, delta)
    np.testing.assert_allclose(roundtrip, y, rtol=1e-8, atol=1e-10)


def test_inverse_lambert_w_zero_delta_is_identity():
    y = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    np.testing.assert_array_equal(inverse_lambert_w(y, 0.0), y)
    np.testing.assert_array_equal(forward_lambert_w(y, 0.0), y)


def test_pipeline_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.standard_t(5, 5_000) * 0.01
    pp = ReturnPreprocessor()
    u2 = pp.fit_transform(x)
    recovered = pp.inverse_transform(u2)
    np.testing.assert_allclose(recovered, x, rtol=1e-8, atol=1e-12)
    # u2 should be approximately standard normal
    assert abs(u2.mean()) < 0.05
    assert abs(u2.std() - 1.0) < 0.05


def test_pipeline_save_load(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.standard_t(4, 3_000) * 0.02
    pp = ReturnPreprocessor().fit(x)
    path = tmp_path / "preprocessor.json"
    pp.save(path)

    loaded = ReturnPreprocessor.load(path)
    assert loaded.mu1 == pytest.approx(pp.mu1)
    assert loaded.sigma1 == pytest.approx(pp.sigma1)
    assert loaded.delta_hat == pytest.approx(pp.delta_hat)
    assert loaded.mu2 == pytest.approx(pp.mu2)
    assert loaded.sigma2 == pytest.approx(pp.sigma2)

    u2 = pp.transform(x)
    np.testing.assert_allclose(loaded.transform(x), u2)


def test_pipeline_unfitted_raises():
    pp = ReturnPreprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(np.zeros(10))
