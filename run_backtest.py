from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.broker import SimpleBroker
from src.models.momentum import MomentumModel
from src.decisions.equal_weight import TopKEqualWeightDecision


def generate_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    symbols = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    rows = []
    last = {s: 100.0 for s in symbols}
    for d in dates:
        for s in symbols:
            open_p = last[s] * (1 + rng.normal(0, 0.01))
            close_p = open_p * (1 + rng.normal(0, 0.01))
            last[s] = close_p
            rows.append({"datetime": d, "symbol": s, "open": round(open_p, 2), "close": round(close_p, 2)})
    return pd.DataFrame(rows)


def main():
    panel = generate_panel()
    dates = panel["datetime"].drop_duplicates().sort_values().tolist()

    model = MomentumModel(window=3)
    decision = TopKEqualWeightDecision(k=1)

    signals = {}
    for i in range(len(dates) - 1):
        d = dates[i]
        hist = panel[panel["datetime"] <= d]
        score = model.score(hist)
        weights = decision.allocate(score)
        signals[dates[i + 1]] = weights

    broker = SimpleBroker(initial_cash=1_000_000, commission_bps=5, slippage_bps=5)
    trades, equity = broker.simulate(panel, signals)
    print("Trades sample:\n", trades.head())
    print("Equity curve tail:\n", equity.tail())

    equity.set_index("datetime")["equity"].plot(title="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
