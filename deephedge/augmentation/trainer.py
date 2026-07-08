"""Training loop for the WGAN-GP."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from deephedge.augmentation.gan import Critic, Generator, gradient_penalty


class WGANTrainer:
    """Trains a WGAN-GP.

    Parameters
    ----------
    generator : Generator
        The generator network.
    critic : Critic
        The critic network.
    noise_dim : int
        Latent dimension of the generator input.
    n_critic : int
        Critic steps per generator step.
    lambda_gp : float
        Gradient penalty coefficient.
    lr : float
        Learning rate for both networks.
    betas : tuple of float
        Adam momentum parameters.
    device : torch.device or str, optional
        Device to train on; defaults to CUDA when available.
    """

    def __init__(
        self,
        generator: Generator,
        critic: Critic,
        noise_dim: int = 5,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.0, 0.9),
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)

    def fit(
        self,
        data: Tensor,
        epochs: int = 200,
        batch: int = 256,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the WGAN-GP on a set of real sequences.

        Parameters
        ----------
        data : Tensor
            Real sequences of shape ``(N, T)`` or ``(N, 1, T)``.
        epochs : int
            Number of passes over the data.
        batch : int
            Batch size.
        seed : int, optional
            Random seed for reproducibility.
        verbose : bool
            Print progress every 10% of epochs.

        Returns
        -------
        dict of str to list of float
            History with keys ``w_dist``, ``loss_c``, ``loss_g``, ``gp`` —
            one entry per epoch.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() == 2:
            data = data.unsqueeze(1)
        if data.dim() != 3 or data.size(1) != 1:
            raise ValueError(f"data must have shape (N, T) or (N, 1, T); got {tuple(data.shape)}")

        loader = DataLoader(TensorDataset(data), batch_size=batch, shuffle=True)
        history: dict[str, list[float]] = {"w_dist": [], "loss_c": [], "loss_g": [], "gp": []}
        step = 0
        last_loss_g = float("nan")

        for epoch in range(epochs):
            w_dists: list[float] = []
            losses_c: list[float] = []
            losses_g: list[float] = []
            gps: list[float] = []

            for (real,) in loader:
                real = real.to(self.device)
                bs, seq_len = real.size(0), real.size(2)

                # --- critic step ---
                z = torch.randn(bs, self.noise_dim, seq_len, device=self.device)
                with torch.no_grad():
                    fake = self.generator(z)
                score_real = self.critic(real).mean()
                score_fake = self.critic(fake).mean()
                gp = gradient_penalty(self.critic, real, fake, device=self.device)
                loss_c = -(score_real - score_fake) + self.lambda_gp * gp

                self.opt_c.zero_grad()
                loss_c.backward()
                self.opt_c.step()

                w_dists.append(float((score_real - score_fake).detach()))
                losses_c.append(float(loss_c.detach()))
                gps.append(float(gp.detach()))
                step += 1

                # --- generator step every n_critic critic steps ---
                if step % self.n_critic == 0:
                    z = torch.randn(bs, self.noise_dim, seq_len, device=self.device)
                    loss_g = -self.critic(self.generator(z)).mean()
                    self.opt_g.zero_grad()
                    loss_g.backward()
                    self.opt_g.step()
                    last_loss_g = float(loss_g.detach())
                    losses_g.append(last_loss_g)

            history["w_dist"].append(float(np.mean(w_dists)))
            history["loss_c"].append(float(np.mean(losses_c)))
            history["loss_g"].append(float(np.mean(losses_g)) if losses_g else last_loss_g)
            history["gp"].append(float(np.mean(gps)))

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(
                    f"epoch {epoch + 1:4d}/{epochs} | "
                    f"W: {history['w_dist'][-1]:+.4f} | "
                    f"loss_C: {history['loss_c'][-1]:+.4f} | "
                    f"loss_G: {history['loss_g'][-1]:+.4f} | "
                    f"GP: {history['gp'][-1]:.4f}"
                )

        return history

    @staticmethod
    def plot_history(history: dict[str, list[float]]) -> tuple:
        """Plot the four training curves in a 2x2 grid.

        Parameters
        ----------
        history : dict
            The dictionary returned by :meth:`fit`.

        Returns
        -------
        tuple
            ``(fig, axes)`` from matplotlib.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        titles = {
            "w_dist": "Wasserstein distance",
            "loss_c": "Critic loss",
            "loss_g": "Generator loss",
            "gp": "Gradient penalty",
        }
        for ax, key in zip(axes.ravel(), ("w_dist", "loss_c", "loss_g", "gp")):
            ax.plot(history[key])
            ax.set_title(titles[key])
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig, axes
