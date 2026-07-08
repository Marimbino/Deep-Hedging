"""Sensitivity analysis: sweep transaction fees, risk aversion, and maturity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deephedge import (
    BrownianStock,
    EntropicRiskMeasure,
    EuropeanOption,
    Hedger,
    MLPHedger,
)

FEES_BPS = [0.0, 0.5, 1.0, 5.0]
RISK_AVERSIONS = [1.0, 5.0, 10.0, 20.0]
MATURITIES_DAYS = [10, 30, 60]

N_PATHS = 10_000
N_EPOCHS = 40
BATCH_SIZE = 1024


def train_and_evaluate(fee: float, risk_aversion: float, maturity_days: int) -> dict:
    stock = BrownianStock(sigma=0.2, cost=fee)
    option = EuropeanOption(stock, strike=1.0, maturity=maturity_days / 252)
    hedger = Hedger(MLPHedger(), option, criterion=EntropicRiskMeasure(risk_aversion))
    hedger.fit(
        n_paths=N_PATHS,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        fee=fee,
        seed=0,
        verbose=False,
    )
    metrics = hedger.evaluate(n_paths=5_000, fee=fee, seed=1)
    metrics.update(fee_bps=fee * 1e4, risk_aversion=risk_aversion, maturity=maturity_days)
    return metrics


def sweep() -> pd.DataFrame:
    rows = []
    base = dict(fee=1e-4, risk_aversion=10.0, maturity_days=30)

    print("sweep 1/3: transaction fees")
    for fee_bps in FEES_BPS:
        cfg = dict(base, fee=fee_bps * 1e-4)
        rows.append({**train_and_evaluate(**cfg), "sweep": "fee"})

    print("sweep 2/3: risk aversion")
    for lam in RISK_AVERSIONS:
        cfg = dict(base, risk_aversion=lam)
        rows.append({**train_and_evaluate(**cfg), "sweep": "risk_aversion"})

    print("sweep 3/3: maturity")
    for days in MATURITIES_DAYS:
        cfg = dict(base, maturity_days=days)
        rows.append({**train_and_evaluate(**cfg), "sweep": "maturity"})

    return pd.DataFrame(rows)


def plot(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sweeps = [
        ("fee", "fee_bps", "transaction fee (bps)"),
        ("risk_aversion", "risk_aversion", "risk aversion $\\lambda$"),
        ("maturity", "maturity", "maturity (days)"),
    ]
    metrics = ["std", "entropic", "cvar95"]
    width = 0.25

    for ax, (sweep_name, x_col, x_label) in zip(axes, sweeps):
        sub = results[results["sweep"] == sweep_name]
        x = np.arange(len(sub))
        for i, metric in enumerate(metrics):
            ax.bar(x + (i - 1) * width, sub[metric].to_numpy(), width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(sub[x_col].to_numpy())
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("risk")
    axes[0].legend()
    fig.suptitle("Deep hedging sensitivity (MLP hedger, European call)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    results = sweep()
    print("\n=== sweep results ===")
    print(
        results[
            ["sweep", "fee_bps", "risk_aversion", "maturity", "mean", "std", "entropic", "cvar95", "turnover"]
        ]
        .round(5)
        .to_string(index=False)
    )
    plot(results)


if __name__ == "__main__":
    main()
