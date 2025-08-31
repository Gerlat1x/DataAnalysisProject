from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.broker import SimpleBroker
from src.models.momentum import MomentumModel
from src.decisions.equal_weight import TopKEqualWeightDecision


def generate_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq=pd.offsets.BDay())
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

    # T 日算信号，安排到 T+1 执行
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

    # 绘图（兼容 equity 已经是 datetime 索引或仍为列的情况）
    series = equity["equity"] if "datetime" not in equity.columns else equity.set_index("datetime")["equity"]
    plt.figure(figsize=(8, 4))
    series.plot(title="Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    # 保存一份，便于自动化跑批查看
    plt.savefig("reports/equity_curve_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
