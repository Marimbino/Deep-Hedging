"""High-level synthetic price-path generator: preprocessing + WGAN + inversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from deephedge.augmentation.gan import Critic, Generator
from deephedge.augmentation.trainer import WGANTrainer
from deephedge.data.windows import rolling_windows
from deephedge.preprocessing.pipeline import ReturnPreprocessor


class QuantWGANGenerator:
    """End-to-end synthetic price-path generator.

    Wires together the :class:`ReturnPreprocessor`, the TCN WGAN-GP, and the
    inverse transform back to price space.

    Usage::

        gen = QuantWGANGenerator(seq_len=30, noise_dim=5)
        gen.fit(ticker="^GSPC", start="2006-01-01", end="2025-12-31", epochs=200)
        prices = gen.generate(n_paths=50_000)   # np.ndarray (n_paths, seq_len+1)

    Alternatively, pass raw log-returns::

        gen.fit_on_returns(log_returns, epochs=200)

    Parameters
    ----------
    seq_len : int
        Length of the generated return sequences (price paths have
        ``seq_len + 1`` points).
    noise_dim : int
        Latent noise channels of the generator.
    hidden : int
        Hidden channel width of both TCNs.
    n_layers : int
        Number of temporal blocks in both TCNs.
    kernel_size : int
        Kernel size of every causal convolution.
    device : torch.device or str, optional
        Device for training and generation; defaults to CUDA when available.
    """

    def __init__(
        self,
        seq_len: int = 30,
        noise_dim: int = 5,
        hidden: int = 64,
        n_layers: int = 5,
        kernel_size: int = 2,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.preprocessor = ReturnPreprocessor()
        self.generator = Generator(noise_dim, hidden, n_layers, kernel_size).to(self.device)
        self.critic = Critic(hidden, n_layers, kernel_size).to(self.device)
        self.history: dict[str, list[float]] | None = None
        self._train_returns: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the generator has been trained (or loaded)."""
        return self.preprocessor.is_fitted

    def fit(self, ticker: str, start: str, end: str, **train_kwargs) -> dict[str, list[float]]:
        """Download historical prices and train on their log-returns.

        Parameters
        ----------
        ticker : str
            Ticker symbol understood by yfinance (e.g. ``"^GSPC"``).
        start, end : str
            Date range in ``YYYY-MM-DD`` format.
        **train_kwargs
            Forwarded to :meth:`fit_on_returns` (``epochs``, ``batch``, ...).

        Returns
        -------
        dict of str to list of float
            The training history.
        """
        from deephedge.data.download import download_prices, log_returns

        prices = download_prices(ticker, start=start, end=end)
        returns = log_returns(prices.to_numpy())
        return self.fit_on_returns(returns, **train_kwargs)

    def fit_on_returns(
        self,
        log_returns: np.ndarray,
        epochs: int = 200,
        batch: int = 256,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lr: float = 1e-4,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Fit the preprocessor and train the WGAN-GP on raw log-returns.

        Parameters
        ----------
        log_returns : np.ndarray
            1-D array of raw log-returns.
        epochs : int
            Training epochs.
        batch : int
            Batch size.
        n_critic : int
            Critic steps per generator step.
        lambda_gp : float
            Gradient penalty coefficient.
        lr : float
            Learning rate for both networks.
        seed : int, optional
            Random seed.
        verbose : bool
            Print training progress.

        Returns
        -------
        dict of str to list of float
            The training history.
        """
        returns = np.asarray(log_returns, dtype=float).ravel()
        if returns.size < self.seq_len + 1:
            raise ValueError(
                f"need at least {self.seq_len + 1} returns to build training windows; "
                f"got {returns.size}"
            )
        u2 = self.preprocessor.fit_transform(returns)
        windows = rolling_windows(u2, self.seq_len)
        data = torch.as_tensor(windows, dtype=torch.float32).unsqueeze(1)

        trainer = WGANTrainer(
            self.generator,
            self.critic,
            noise_dim=self.noise_dim,
            n_critic=n_critic,
            lambda_gp=lambda_gp,
            lr=lr,
            device=self.device,
        )
        self.history = trainer.fit(data, epochs=epochs, batch=batch, seed=seed, verbose=verbose)
        self._train_returns = returns
        return self.history

    def generate_returns(
        self,
        n_paths: int,
        seed: int | None = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Sample synthetic log-return sequences.

        Parameters
        ----------
        n_paths : int
            Number of sequences to generate.
        seed : int, optional
            Random seed.
        batch_size : int
            Generation batch size (memory control).

        Returns
        -------
        np.ndarray
            Log-returns of shape ``(n_paths, seq_len)``.
        """
        if not self.is_fitted:
            raise RuntimeError("QuantWGANGenerator must be fitted before generating; call fit()")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.generator.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            remaining = n_paths
            while remaining > 0:
                bs = min(batch_size, remaining)
                z = torch.randn(bs, self.noise_dim, self.seq_len, device=self.device)
                u2 = self.generator(z).squeeze(1).cpu().numpy()
                chunks.append(u2)
                remaining -= bs
        self.generator.train()
        u2_all = np.concatenate(chunks, axis=0)
        return self.preprocessor.inverse_transform(u2_all)

    def generate(self, n_paths: int, S0: float = 1.0, seed: int | None = None) -> np.ndarray:
        """Sample synthetic price paths.

        Parameters
        ----------
        n_paths : int
            Number of paths.
        S0 : float
            Initial price of every path.
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Prices of shape ``(n_paths, seq_len + 1)``; column 0 equals ``S0``.
        """
        log_ret = self.generate_returns(n_paths, seed=seed)
        log_paths = np.cumsum(log_ret, axis=1)
        prices = S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_paths], axis=1))
        return prices

    def evaluate(
        self,
        n_gen: int = 2000,
        real_returns: np.ndarray | None = None,
        max_lag: int = 15,
    ) -> tuple:
        """Plot the five stylized-fact diagnostics against real data.

        Parameters
        ----------
        n_gen : int
            Number of generated sequences to use.
        real_returns : np.ndarray, optional
            Real log-returns; defaults to the returns used during fitting.
        max_lag : int
            Maximum lag for the autocorrelation and leverage panels.

        Returns
        -------
        tuple
            ``(fig, axes)`` from matplotlib.
        """
        from deephedge.utils.plotting import plot_stylized_facts

        if real_returns is None:
            real_returns = self._train_returns
        if real_returns is None:
            raise RuntimeError(
                "no real returns available; pass real_returns= or fit on data first"
            )
        real_windows = rolling_windows(np.asarray(real_returns, dtype=float), self.seq_len)
        return plot_stylized_facts(self, real_windows, n_gen=n_gen, max_lag=max_lag)

    def save(self, dir: str | Path) -> None:
        """Serialise networks, preprocessor, and config to a directory."""
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted QuantWGANGenerator")
        path = Path(dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.generator.state_dict(), path / "generator.pt")
        torch.save(self.critic.state_dict(), path / "critic.pt")
        self.preprocessor.save(path / "preprocessor.json")
        config = {
            "seq_len": self.seq_len,
            "noise_dim": self.noise_dim,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "kernel_size": self.kernel_size,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        dir: str | Path,
        device: torch.device | str | None = None,
    ) -> "QuantWGANGenerator":
        """Load a generator previously stored with :meth:`save`."""
        path = Path(dir)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        obj = cls(device=device, **config)
        obj.generator.load_state_dict(
            torch.load(path / "generator.pt", map_location=obj.device, weights_only=True)
        )
        critic_path = path / "critic.pt"
        if critic_path.exists():
            obj.critic.load_state_dict(
                torch.load(critic_path, map_location=obj.device, weights_only=True)
            )
        obj.preprocessor = ReturnPreprocessor.load(path / "preprocessor.json")
        return obj
