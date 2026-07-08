"""Full comparison: MLP vs GRU vs Black-Scholes hedgers on WGAN market data.

Requires a trained WGAN generator (run 02_wgan_training.py first, or this
script falls back to a quickly-trained generator on downloaded data).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deephedge import (
    BlackScholesHedger,
    EuropeanOption,
    GRUHedger,
    Hedger,
    MLPHedger,
    QuantWGANGenerator,
    WGANStock,
)
from deephedge.utils.plotting import plot_hedge_ratios, plot_pnl_distribution

MODEL_DIR = Path("./wgan_model")
FEE = 1e-4


def load_or_train_generator() -> QuantWGANGenerator:
    if MODEL_DIR.exists():
        print(f"loading generator from {MODEL_DIR}")
        return QuantWGANGenerator.load(MODEL_DIR)
    print("no saved generator found; training a small one on ^GSPC (2006-2025)")
    gen = QuantWGANGenerator(seq_len=30, noise_dim=5)
    gen.fit(
        ticker="^GSPC",
        start="2006-01-01",
        end="2025-12-31",
        epochs=50,
        seed=42,
        verbose=True,
    )
    gen.save(MODEL_DIR)
    return gen


def main() -> None:
    gen = load_or_train_generator()
    stock = WGANStock(gen, cost=FEE)
    option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)

    # 1. Train the neural hedgers
    hedgers = {
        "MLP": Hedger(MLPHedger(), option),
        "GRU": Hedger(GRUHedger(), option),
    }
    for name, hedger in hedgers.items():
        print(f"\n=== training {name} hedger ===")
        hedger.fit(n_paths=20_000, n_epochs=100, batch_size=1024, fee=FEE, seed=0)

    # 2. Black-Scholes benchmark (no training)
    hedgers["BS"] = Hedger(BlackScholesHedger(r=0.0, sigma=0.20), option)

    # 3. Metrics table on a common set of evaluation paths
    rows = {}
    for name, hedger in hedgers.items():
        rows[name] = hedger.evaluate(n_paths=10_000, fee=FEE, seed=123)
    table = pd.DataFrame(rows).T
    print("\n=== evaluation metrics ===")
    print(table.round(5).to_string())

    # 4. Hedge ratios and PnL distributions on shared paths
    eval_prices = stock.simulate(1_000, option.n_steps, seed=7)
    deltas = {}
    pnls = {}
    for name, hedger in hedgers.items():
        deltas[name], pnls[name] = hedger.hedge_path(eval_prices, fee=FEE)

    plot_hedge_ratios(deltas["MLP"], deltas["GRU"], deltas["BS"], eval_prices)
    plot_pnl_distribution(pnls)
    plt.show()


if __name__ == "__main__":
    main()
