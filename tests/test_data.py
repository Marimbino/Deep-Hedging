from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deephedge.data import (
    download_prices,
    log_returns,
    normalized_price_windows,
    rolling_windows,
    train_test_split_windows,
)


def test_rolling_windows():
    x = np.arange(10.0)
    w = rolling_windows(x, window=4)
    assert w.shape == (7, 4)
    np.testing.assert_array_equal(w[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(w[-1], [6, 7, 8, 9])

    strided = rolling_windows(x, window=4, stride=2)
    assert strided.shape == (4, 4)
    np.testing.assert_array_equal(strided[1], [2, 3, 4, 5])

    with pytest.raises(ValueError):
        rolling_windows(np.arange(3.0), window=5)


def test_normalized_price_windows():
    prices = np.array([100.0, 110.0, 121.0, 133.1, 146.41])
    w = normalized_price_windows(prices, n_steps=2)
    assert w.shape == (3, 3)
    np.testing.assert_allclose(w[:, 0], 1.0)
    np.testing.assert_allclose(w[0], [1.0, 1.1, 1.21])


def test_train_test_split_windows():
    windows = np.arange(100.0).reshape(50, 2)
    train, test = train_test_split_windows(windows, train_frac=0.8, seed=0)
    assert train.shape == (40, 2)
    assert test.shape == (10, 2)
    # no overlap and full coverage
    combined = np.vstack([train, test])
    assert {tuple(row) for row in combined} == {tuple(row) for row in windows}

    with pytest.raises(ValueError):
        train_test_split_windows(windows, train_frac=1.5)


def test_log_returns():
    prices = np.array([1.0, np.e, np.e**2])
    np.testing.assert_allclose(log_returns(prices), [1.0, 1.0])
    series = pd.Series([100.0, 105.0, 110.25])
    np.testing.assert_allclose(log_returns(series), np.log(1.05), rtol=1e-10)


def test_download_prices_cache_hit(tmp_path):
    # a pre-existing cache file must be served without touching the network
    cached = pd.DataFrame(
        {"FAKE": [100.0, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    cache_file = tmp_path / "FAKE_2024-01-01_2024-01-04_Close.csv"
    cached.to_csv(cache_file)

    series = download_prices("FAKE", start="2024-01-01", end="2024-01-04", cache_dir=tmp_path)
    np.testing.assert_allclose(series.to_numpy(), [100.0, 101.0, 102.0])
