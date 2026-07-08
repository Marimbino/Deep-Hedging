"""Three-stage preprocessing pipeline for log-returns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deephedge.preprocessing.lambert import forward_lambert_w, igmm, inverse_lambert_w


class ReturnPreprocessor:
    """Fits and applies the three-stage log-return preprocessing.

    The pipeline is::

        raw log-returns -> standardise -> inverse Lambert-W -> re-standardise

    Attributes
    ----------
    mu1, sigma1 : float
        First standardisation parameters (mean and std of the raw returns).
    delta_hat : float
        Lambert-W tail parameter estimated by IGMM.
    mu2, sigma2 : float
        Second standardisation parameters (mean and std of the gaussianized
        returns).
    """

    def __init__(self) -> None:
        self.mu1: float | None = None
        self.sigma1: float | None = None
        self.delta_hat: float | None = None
        self.mu2: float | None = None
        self.sigma2: float | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.mu1 is not None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("ReturnPreprocessor must be fitted before use; call fit() first")

    def fit(self, log_returns: np.ndarray) -> "ReturnPreprocessor":
        """Estimate all five parameters from raw log-returns.

        Parameters
        ----------
        log_returns : np.ndarray
            1-D array of raw log-returns.

        Returns
        -------
        ReturnPreprocessor
            ``self``, to allow chaining.
        """
        x = np.asarray(log_returns, dtype=float).ravel()
        self.mu1 = float(x.mean())
        self.sigma1 = float(x.std())
        if self.sigma1 <= 0:
            raise ValueError("log_returns must have positive standard deviation")
        r1 = (x - self.mu1) / self.sigma1
        self.delta_hat = igmm(r1)
        u1 = inverse_lambert_w(r1, self.delta_hat)
        self.mu2 = float(u1.mean())
        self.sigma2 = float(u1.std())
        if self.sigma2 <= 0:
            raise ValueError("gaussianized returns have zero standard deviation")
        return self

    def transform(self, log_returns: np.ndarray) -> np.ndarray:
        """Apply the fitted pipeline: log-returns -> ``u2``.

        Parameters
        ----------
        log_returns : np.ndarray
            Raw log-returns.

        Returns
        -------
        np.ndarray
            Fully preprocessed returns ``u2`` (approximately standard normal).
        """
        self._check_fitted()
        x = np.asarray(log_returns, dtype=float)
        r1 = (x - self.mu1) / self.sigma1
        u1 = inverse_lambert_w(r1, self.delta_hat)
        return (u1 - self.mu2) / self.sigma2

    def inverse_transform(self, u2: np.ndarray) -> np.ndarray:
        """Invert the pipeline: ``u2`` -> log-returns.

        Parameters
        ----------
        u2 : np.ndarray
            Preprocessed returns (e.g. produced by a generator network).

        Returns
        -------
        np.ndarray
            Raw log-returns.
        """
        self._check_fitted()
        u2 = np.asarray(u2, dtype=float)
        u1 = u2 * self.sigma2 + self.mu2
        r1 = forward_lambert_w(u1, self.delta_hat)
        return r1 * self.sigma1 + self.mu1

    def fit_transform(self, log_returns: np.ndarray) -> np.ndarray:
        """Fit the pipeline and transform the data in one call."""
        return self.fit(log_returns).transform(log_returns)

    def save(self, path: str | Path) -> None:
        """Serialise the fitted parameters to a JSON file."""
        self._check_fitted()
        params = {
            "mu1": self.mu1,
            "sigma1": self.sigma1,
            "delta_hat": self.delta_hat,
            "mu2": self.mu2,
            "sigma2": self.sigma2,
        }
        Path(path).write_text(json.dumps(params, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ReturnPreprocessor":
        """Load a fitted preprocessor from a JSON file."""
        params = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.mu1 = float(params["mu1"])
        obj.sigma1 = float(params["sigma1"])
        obj.delta_hat = float(params["delta_hat"])
        obj.mu2 = float(params["mu2"])
        obj.sigma2 = float(params["sigma2"])
        return obj
