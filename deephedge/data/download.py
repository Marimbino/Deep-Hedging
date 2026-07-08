"""Market data download utilities (yfinance wrapper with caching)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(
    ticker: str,
    start: str,
    end: str,
    field: str = "Close",
    cache_dir: str | Path | None = None,
) -> pd.Series:
    """Download a single ticker's adjusted price series via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g. ``"SPY"`` or ``"^GSPC"``).
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    field : str
        OHLC field to extract (default ``"Close"``; prices are auto-adjusted).
    cache_dir : str or Path, optional
        If given, cache the download as a CSV in this directory and reuse it
        on subsequent calls.

    Returns
    -------
    pd.Series
        Price series indexed by date, NaNs dropped.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        safe = ticker.replace("^", "_").replace("/", "_")
        cache_path = Path(cache_dir) / f"{safe}_{start}_{end}_{field}.csv"
        if cache_path.exists():
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return cached.iloc[:, 0].dropna()

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"no data returned for ticker {ticker!r} between {start} and {end}")

    series = df[field]
    if isinstance(series, pd.DataFrame):  # yfinance may return MultiIndex columns
        series = series.iloc[:, 0]
    series = series.dropna()
    series.name = ticker

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_csv(cache_path)
    return series


def download_ohlc(
    tickers: list[str],
    start: str,
    end: str,
    field: str = "Close",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download several tickers into one DataFrame (columns = tickers).

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    field : str
        OHLC field to extract.
    cache_dir : str or Path, optional
        Per-ticker CSV cache directory.

    Returns
    -------
    pd.DataFrame
        Prices with one column per ticker, rows with any NaN dropped.
    """
    frames = {
        ticker: download_prices(ticker, start=start, end=end, field=field, cache_dir=cache_dir)
        for ticker in tickers
    }
    return pd.DataFrame(frames).dropna()


def log_returns(prices: np.ndarray | pd.Series) -> np.ndarray:
    """Compute log-returns ``log(S_{t+1} / S_t)`` from a price series.

    Parameters
    ----------
    prices : np.ndarray or pd.Series
        Price levels.

    Returns
    -------
    np.ndarray
        1-D array of log-returns, one element shorter than ``prices``.
    """
    values = np.asarray(prices, dtype=float).ravel()
    return np.diff(np.log(values))
