"""deephedge: Deep Hedging with WGAN-GP data augmentation.

A PyTorch package for pricing and hedging financial derivatives using the
Deep Hedging framework (Buhler et al. 2019), with realistic synthetic market
data from a TCN WGAN-GP (Wiese et al. 2020).
"""

from __future__ import annotations

from deephedge._version import __version__
from deephedge.augmentation.generator import QuantWGANGenerator
from deephedge.hedger import Hedger
from deephedge.instruments.options import BinaryOption, EuropeanOption, LookbackOption
from deephedge.instruments.stocks import BrownianStock, WGANStock
from deephedge.nn.black_scholes import BlackScholesHedger
from deephedge.nn.gru_hedger import GRUHedger
from deephedge.nn.mlp_hedger import MLPHedger
from deephedge.nn.no_transaction_band import NoTransactionBandNet
from deephedge.nn.risk_measures import CVaR, EntropicRiskMeasure, ExpectedShortfall
from deephedge.pricer import DerivativePricer
from deephedge.preprocessing.pipeline import ReturnPreprocessor

__all__ = [
    "BinaryOption",
    "BlackScholesHedger",
    "BrownianStock",
    "CVaR",
    "DerivativePricer",
    "EntropicRiskMeasure",
    "EuropeanOption",
    "ExpectedShortfall",
    "GRUHedger",
    "Hedger",
    "LookbackOption",
    "MLPHedger",
    "NoTransactionBandNet",
    "QuantWGANGenerator",
    "ReturnPreprocessor",
    "WGANStock",
    "__version__",
]
