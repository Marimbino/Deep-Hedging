"""Recurrent hedger processing the whole path in one pass."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GRUHedger(nn.Module):
    """GRU hedger: processes the full path in one forward pass.

    The GRU hidden state carries the hedging memory, so the previous delta is
    not fed back explicitly.

    Input: ``(batch, n_steps, 2)`` with features ``(log(S_t / K), tau_t)``.
    Output: ``(batch, n_steps, 1)`` — a delta in ``(0, 1)`` at every step.

    Parameters
    ----------
    in_features : int
        Number of input features per step (default 2).
    hidden_size : int
        GRU hidden dimension.
    n_layers : int
        Number of stacked GRU layers.
    """

    expects_sequence = True
    """Marks this model as consuming the full path at once (see Hedger)."""

    def __init__(self, in_features: int = 2, hidden_size: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_size, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute hedge ratios for the whole path.

        Parameters
        ----------
        x : Tensor
            Features of shape ``(batch, n_steps, in_features)``.

        Returns
        -------
        Tensor
            Hedge ratios of shape ``(batch, n_steps, 1)`` in ``(0, 1)``.
        """
        out, _ = self.gru(x)
        return torch.sigmoid(self.head(out))
