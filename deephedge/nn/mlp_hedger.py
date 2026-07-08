"""Feed-forward hedger applied step by step along the path."""

from __future__ import annotations

from torch import Tensor, nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "tanh": nn.Tanh,
}


class MLPHedger(nn.Module):
    """MLP-based hedger called at every time step.

    Input features at step ``t``: ``(delta_{t-1}, log(S_t / K), tau_t)`` of
    shape ``(batch, 3)``. Output: ``delta_t`` of shape ``(batch, 1)``,
    squashed into ``(0, 1)`` by a sigmoid.

    Parameters
    ----------
    in_features : int
        Number of input features (default 3).
    hidden_size : int
        Width of the hidden layers.
    n_layers : int
        Number of hidden layers.
    activation : str
        One of ``"relu"``, ``"selu"``, ``"tanh"``.
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_size: int = 32,
        n_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}; got {activation!r}")
        act = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        width = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(width, hidden_size))
            layers.append(act())
            width = hidden_size
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the hedge ratio for one time step.

        Parameters
        ----------
        x : Tensor
            Features of shape ``(batch, in_features)``.

        Returns
        -------
        Tensor
            Hedge ratio of shape ``(batch, 1)`` in ``(0, 1)``.
        """
        return self.net(x)
