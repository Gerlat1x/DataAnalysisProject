from __future__ import annotations
import pandas as pd


class MomentumModel:
    """
    简单动量模型：
    - 对每个 symbol 按 datetime 排序计算收盘价的滚动均值 (MA)
    - 用最新一行的 (close - MA_window) 作为该标的的动量分数
    - 返回一个以 symbol 为索引的 Series（列名 'score'）

    注意：
    - 传入的 panel 必须包含多行历史数据（至少 >= window 行），
      并且包含列 {'datetime', 'symbol', 'close'}。
    - 若某些标的历史长度不足 window，则它们的 score 为 NaN。
      你可以在上层决策/风控里决定是丢弃还是填充（如 .fillna(0)）。
    """

    def __init__(self, window: int = 5):
        """
        Args:
            window: 计算动量基准的均线窗口长度（交易日数）
        """
        self.window = window

    def score(self, panel: pd.DataFrame) -> pd.Series:
        """
        对传入的“面板数据”计算每个 symbol 的最新动量分数。

        Args:
            panel: 包含多只标的、多个交易日的 DataFrame，
                   至少需要列 {'datetime', 'symbol', 'close'}。

        Returns:
            pd.Series:
                index = symbol
                name  = 'score'
                值为该 symbol 最新一行的 (close - MA_window)
        """
        required = {"datetime", "symbol", "close"}
        if not required.issubset(panel.columns):
            missing = required.difference(panel.columns)
            raise ValueError(f"Missing columns: {missing}")

        # 按 (symbol, datetime) 排序，确保 rolling 计算顺序正确
        ordered = panel.sort_values(["symbol", "datetime"]).copy()

        # 分组滚动均值：对每个 symbol 的 close 计算 MA(window)
        # 注：reset_index(level=0, drop=True) 把 groupby 扩展出的层级索引去掉，方便赋回列
        ordered["ma"] = (
            ordered.groupby("symbol")["close"]
            .rolling(self.window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 取每个 symbol 的“最新一行”
        latest = ordered.groupby("symbol").tail(1).copy()

        # 动量分数：最新收盘 - 最新MA
        latest["score"] = latest["close"] - latest["ma"]

        # 返回以 symbol 为索引的 Series
        return latest.set_index("symbol")["score"].rename("score")
