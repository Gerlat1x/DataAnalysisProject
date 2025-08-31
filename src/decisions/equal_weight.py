from __future__ import annotations
from typing import Optional
import pandas as pd


class TopKEqualWeightDecision:
    """
    简单的 Top-K 等权决策：
    - 输入一个股票得分序列 (pd.Series, index=symbol, value=score)
    - 选出分数最高的前 K 个标的
    - 给它们分配相等的权重（1/K）
    - 返回一个新的 pd.Series，index=symbol, value=weight
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): 要选择的股票数量 (Top-K)。必须为正数。
        """
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def allocate(self, scores: Optional[pd.Series]) -> pd.Series:
        """
        根据输入的得分序列分配权重。
        Args:
            scores (pd.Series | None): 股票得分，
            index = symbol, values = score。
            如果 None 或空，则返回空权重。

        Returns:
            pd.Series: 股票权重，index = symbol, values = weight。
            如果没有合格的股票，则返回空 Series。
        """
        if scores is None:
            # 没有分数 → 返回空权重
            return pd.Series(dtype=float, name="weight")

        # 丢掉 NaN 分数
        cleaned = scores.dropna()
        if cleaned.empty:
            return pd.Series(dtype=float, name="weight")

        # 取 Top-K 得分最高的股票
        top = cleaned.sort_values(ascending=False).head(self.k)
        n = len(top)
        if n == 0:
            return pd.Series(dtype=float, name="weight")

        # 等权分配，每只股票权重 = 1/n
        weight = 1.0 / n
        return pd.Series(weight, index=top.index, name="weight")
