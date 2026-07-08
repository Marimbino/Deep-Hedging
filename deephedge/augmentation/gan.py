"""WGAN-GP generator, critic, and gradient penalty."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from deephedge.augmentation.tcn import TCN


class Generator(nn.Module):
    """TCN generator mapping latent noise to a return sequence.

    Parameters
    ----------
    noise_dim : int
        Number of latent noise channels.
    hidden : int
        Hidden channel width of the TCN.
    n_layers : int
        Number of temporal blocks.
    kernel_size : int
        Kernel size of every causal convolution.
    """

    def __init__(
        self,
        noise_dim: int = 5,
        hidden: int = 64,
        n_layers: int = 5,
        kernel_size: int = 2,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.tcn = TCN(noise_dim, 1, hidden=hidden, n_layers=n_layers, kernel_size=kernel_size)

    def forward(self, z: Tensor) -> Tensor:
        """Generate a batch of return sequences.

        Parameters
        ----------
        z : Tensor
            Latent noise of shape ``(batch, noise_dim, T)``.

        Returns
        -------
        Tensor
            Generated sequences of shape ``(batch, 1, T)``.
        """
        return self.tcn(z)


class Critic(nn.Module):
    """TCN critic scoring return sequences; higher means more realistic.

    Parameters
    ----------
    hidden : int
        Hidden channel width of the TCN.
    n_layers : int
        Number of temporal blocks.
    kernel_size : int
        Kernel size of every causal convolution.
    dropout : float
        Dropout probability inside each block.
    """

    def __init__(
        self,
        hidden: int = 64,
        n_layers: int = 5,
        kernel_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tcn = TCN(1, 1, hidden=hidden, n_layers=n_layers, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Score a batch of sequences.

        Parameters
        ----------
        x : Tensor
            Sequences of shape ``(batch, 1, T)``.

        Returns
        -------
        Tensor
            Scores of shape ``(batch,)`` — the per-sequence mean over time.
        """
        return self.tcn(x).mean(dim=(1, 2))


def gradient_penalty(
    critic: nn.Module,
    real: Tensor,
    fake: Tensor,
    device: torch.device | str | None = None,
) -> Tensor:
    """Two-sided gradient penalty of WGAN-GP (Gulrajani et al. 2017).

    ``GP = E[(||grad critic(interp)||_2 - 1)**2]`` where ``interp`` is a random
    convex combination of real and fake samples.

    Parameters
    ----------
    critic : nn.Module
        The critic network.
    real : Tensor
        Real samples of shape ``(batch, 1, T)``.
    fake : Tensor
        Generated samples of the same shape as ``real``.
    device : torch.device or str, optional
        Device for the interpolation coefficients; defaults to ``real.device``.

    Returns
    -------
    Tensor
        Scalar penalty term.
    """
    device = real.device if device is None else torch.device(device)
    eps = torch.rand(real.size(0), 1, 1, device=device, dtype=real.dtype)
    interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    scores = critic(interp)
    grads = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grads.flatten(start_dim=1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()
