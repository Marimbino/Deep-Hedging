"""Temporal Convolutional Network building blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CausalConv1d(nn.Module):
    """1-D convolution that never looks into the future.

    The input is left-padded by ``(kernel_size - 1) * dilation`` so that the
    output at time ``t`` depends only on inputs at times ``<= t``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    dilation : int
        Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.pad,
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the causal convolution.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, in_channels, T)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, out_channels, T)``.
        """
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, : -self.pad]
        return out


class TempBlock(nn.Module):
    """Residual temporal block: two causal convolutions with PReLU activations.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size.
    dilation : int
        Dilation factor for both convolutions.
    dropout : float
        Dropout probability applied after the second activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.act1 = nn.PReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.act2 = nn.PReLU()
        self.drop = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.skip: nn.Module = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the block with a residual skip connection.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, in_channels, T)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, out_channels, T)``.
        """
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        h = self.drop(h)
        return h + self.skip(x)


class TCN(nn.Module):
    """Stack of :class:`TempBlock` layers with exponentially growing dilation.

    Layer ``i`` uses dilation ``2**i``, so the receptive field grows
    exponentially with depth. A final 1x1 convolution maps the hidden
    channels to ``out_channels``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden : int
        Hidden channel width of every temporal block.
    n_layers : int
        Number of temporal blocks.
    kernel_size : int
        Kernel size of every causal convolution.
    dropout : float
        Dropout probability inside each block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden: int = 64,
        n_layers: int = 5,
        kernel_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(n_layers):
            blocks.append(
                TempBlock(
                    in_channels if i == 0 else hidden,
                    hidden,
                    kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply all temporal blocks followed by the 1x1 head.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch, in_channels, T)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, out_channels, T)``.
        """
        for block in self.blocks:
            x = block(x)
        return self.head(x)
