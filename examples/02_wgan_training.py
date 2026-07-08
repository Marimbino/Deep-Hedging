"""Train the TCN WGAN-GP on S&P 500 returns and generate synthetic paths."""

from __future__ import annotations

import matplotlib.pyplot as plt

from deephedge import QuantWGANGenerator
from deephedge.utils.plotting import plot_price_paths


def main() -> None:
    # 1. Create the generator
    gen = QuantWGANGenerator(seq_len=30, noise_dim=5)

    # 2. Download ^GSPC and train (this takes a while on CPU)
    history = gen.fit(
        ticker="^GSPC",
        start="2006-01-01",
        end="2025-12-31",
        epochs=200,
        batch=256,
        seed=42,
        verbose=True,
    )
    print(f"final Wasserstein distance: {history['w_dist'][-1]:+.4f}")

    # 3. Stylized-fact diagnostics (marginal, QQ, ACFs, leverage effect)
    gen.evaluate(n_gen=2000)

    # 4. Generate synthetic price paths
    prices = gen.generate(50_000)
    print(f"generated paths: {prices.shape}")
    plot_price_paths(prices, n_plot=100)

    # 5. Persist for reuse (e.g. by WGANStock)
    gen.save("./wgan_model")
    print("model saved to ./wgan_model")

    plt.show()


if __name__ == "__main__":
    main()
