"""Quickstart: hedge a European call on a GBM stock with an MLP hedger."""

from __future__ import annotations

from deephedge import BrownianStock, EuropeanOption, Hedger, MLPHedger


def main() -> None:
    # 1. Underlier and derivative
    stock = BrownianStock(sigma=0.2, cost=1e-4)
    option = EuropeanOption(stock, strike=1.0, maturity=30 / 252)

    # 2. Hedging model wrapped in the top-level Hedger
    hedger = Hedger(MLPHedger(), option)

    # 3. Train
    hedger.fit(n_paths=20_000, n_epochs=100, batch_size=1024, fee=1e-4, seed=42)

    # 4. Price and evaluate
    price = hedger.price(n_paths=50_000, seed=0)
    metrics = hedger.evaluate(n_paths=10_000, fee=1e-4, seed=1)

    print(f"\nutility-indifference price: {price:.5f}")
    print("evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key:>10s}: {value:+.5f}")


if __name__ == "__main__":
    main()
