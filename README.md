# deephedge

[![PyPI version](https://img.shields.io/pypi/v/deephedge.svg)](https://pypi.org/project/deephedge/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/rriccardo320/deephedge/actions/workflows/ci.yml/badge.svg)](https://github.com/rriccardo320/deephedge/actions/workflows/ci.yml)

**Deep Hedging with WGAN-GP data augmentation, in native PyTorch.**

Deep Hedging (Bühler et al. 2019) replaces closed-form hedging rules with a neural network trained
to minimise a convex risk measure of the hedged profit-and-loss — naturally handling transaction
costs, discrete rebalancing, and market frictions that break Black-Scholes. Most implementations
train on geometric Brownian motion, which misses the fat tails, volatility clustering, and leverage
effect of real markets. `deephedge` closes that gap with a **Temporal Convolutional Network
WGAN-GP** (Wiese et al. 2020) that learns realistic return dynamics from historical data and
supplies unlimited synthetic training paths.

## Installation

```bash
pip install deephedge
```

## Quickstart

```python
from deephedge import BrownianStock, EuropeanOption, Hedger, MLPHedger

stock = BrownianStock(sigma=0.2, cost=1e-4)
option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)

hedger = Hedger(MLPHedger(), option)
hedger.fit(n_paths=20_000, n_epochs=100)

print("indifference price:", hedger.price())
print(hedger.evaluate())
```

Swap `BrownianStock` for a `WGANStock` driven by a generator fitted to real S&P 500 returns:

```python
from deephedge import QuantWGANGenerator, WGANStock

gen = QuantWGANGenerator(seq_len=30)
gen.fit(ticker="^GSPC", start="2006-01-01", end="2025-12-31", epochs=200)
stock = WGANStock(gen, cost=1e-4)
```

## Architecture overview

- **`deephedge.data`** — yfinance download wrapper with CSV caching, plus rolling-window
  construction and trajectory-level train/test splits.
- **`deephedge.preprocessing`** — the Lambert-W gaussianization pipeline: standardise, invert the
  heavy tails with an IGMM-estimated tail parameter, re-standardise. Fully invertible and
  JSON-serialisable.
- **`deephedge.augmentation`** — causal TCN building blocks, WGAN-GP generator/critic, a training
  loop with diagnostics, and `QuantWGANGenerator`, the high-level fit/generate/save/load interface.
- **`deephedge.instruments`** — `Primary` underliers (`BrownianStock`, `WGANStock`) and
  `Derivative` payoffs (`EuropeanOption`, `LookbackOption`, `BinaryOption`).
- **`deephedge.nn`** — hedging models (`MLPHedger`, `GRUHedger`, analytic `BlackScholesHedger`,
  `NoTransactionBandNet`) and risk measures (`EntropicRiskMeasure`, `ExpectedShortfall`/`CVaR`).
- **`deephedge.hedger` / `deephedge.pricer`** — the top-level `Hedger` (fit, price, evaluate,
  hedge_path) and `DerivativePricer` (indifference pricing, BS comparison, autograd Greeks).
- **`deephedge.utils`** — PnL metrics and matplotlib plots (training history, hedge ratios, PnL
  distributions, stylized facts, price paths).

## WGAN data augmentation

Raw log-returns are far from Gaussian, which destabilises GAN training. `deephedge` first applies
a three-stage transform: standardise, then remove the heavy tails with the **inverse Lambert-W
transform** `u = sign(y)·√(W(δy²)/δ)` whose tail parameter `δ` is estimated by the IGMM algorithm,
then standardise again. The WGAN-GP is trained on rolling windows of these gaussianized returns.

Both generator and critic are **Temporal Convolutional Networks**: stacks of causal, dilated
convolutions (dilation `2^i` at depth `i`) with PReLU activations and residual skip connections,
so the receptive field covers the whole window without ever looking into the future. Training uses
the Wasserstein loss with gradient penalty, 5 critic steps per generator step, and
`Adam(lr=1e-4, betas=(0.0, 0.9))`. Generated sequences are pushed back through the inverse of the
preprocessing pipeline and cumulated into price paths. Five stylized-fact diagnostics (marginal
distribution, QQ-plot, ACF of returns / squared returns, leverage effect) verify realism.

## API summary

| Object | Purpose |
|---|---|
| `QuantWGANGenerator` | Fit a TCN WGAN-GP on returns; `generate()` price paths; `save()`/`load()` |
| `ReturnPreprocessor` | Invertible Lambert-W gaussianization pipeline |
| `BrownianStock`, `WGANStock` | Underlier simulators (GBM / WGAN-driven) |
| `EuropeanOption`, `LookbackOption`, `BinaryOption` | Derivative payoffs |
| `MLPHedger`, `GRUHedger` | Trainable hedging networks (per-step / full-path) |
| `BlackScholesHedger` | Analytic delta and price benchmark |
| `NoTransactionBandNet` | Learned no-trade band around the BS delta |
| `EntropicRiskMeasure`, `ExpectedShortfall`, `CVaR` | Training criteria / risk metrics |
| `Hedger` | `fit()`, `price()`, `evaluate()`, `hedge_path()` |
| `DerivativePricer` | Indifference prices, BS comparison, autograd Greeks |
| `pnl_metrics`, `plot_*` | Evaluation metrics and plots |

## References

- Bühler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). *Deep Hedging*. Quantitative Finance,
  19(8), 1271–1291. [arXiv:1802.03042](https://arxiv.org/abs/1802.03042)
- Imaki, S., Imajo, K., Ito, K., Minami, K., & Nakagawa, K. (2021). *No-Transaction Band Network:
  A Neural Network Architecture for Efficient Deep Hedging*.
  [arXiv:2103.01775](https://arxiv.org/abs/2103.01775)
- Wiese, M., Knobloch, R., Korn, R., & Kretschmer, P. (2020). *Quant GANs: Deep Generation of
  Financial Time Series*. Quantitative Finance, 20(9), 1419–1440.
  [arXiv:1907.06673](https://arxiv.org/abs/1907.06673)

## Contributing

Contributions are welcome! To get started:

```bash
git clone https://github.com/rriccardo320/deephedge
cd deephedge
pip install -e ".[dev]"
pytest tests/ -v
```

Please run `black .` and `ruff check .` before opening a pull request, and add tests for any new
functionality.

## License

MIT — see [LICENSE](LICENSE).
